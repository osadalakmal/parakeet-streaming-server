from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from parakeet_mlx import from_pretrained

from app.session import StreamConfig, StreamingSession

LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger("parakeet_streaming_service")

DEFAULT_MODEL = "mlx-community/parakeet-tdt-1.1b-v2"


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading parakeet model: %s", DEFAULT_MODEL)
    app.state.model = from_pretrained(DEFAULT_MODEL)
    logger.info("Model loaded and warm.")
    yield
    logger.info("Shutting down streaming service.")


app = FastAPI(title="Parakeet Local Streaming Service", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/debug/config")
def debug_config() -> dict[str, Any]:
    return {
        "bind": "127.0.0.1",
        "model": DEFAULT_MODEL,
        "audio_format": "PCM16LE mono, 16kHz",
        "websocket": "/ws/transcribe",
        "default_streaming": {
            "context_size": 256,
            "depth": 8,
            "keep_original_attention": True,
        },
    }


@app.websocket("/ws/transcribe")
async def ws_transcribe(websocket: WebSocket):
    await websocket.accept()
    await websocket.send_json({"type": "ready"})

    session: StreamingSession | None = None

    try:
        while True:
            message = await websocket.receive()

            if "text" in message and message["text"] is not None:
                control = _parse_json(message["text"])
                if not control:
                    await websocket.send_json({"type": "error", "message": "Malformed JSON message."})
                    continue

                msg_type = control.get("type")
                if msg_type == "start":
                    if session is not None:
                        session.close()

                    config = StreamConfig(
                        sample_rate=int(control.get("sample_rate", 16_000)),
                        context_size=int(control.get("context_size", 256)),
                        depth=int(control.get("depth", 8)),
                        keep_original_attention=bool(control.get("keep_original_attention", True)),
                    )
                    if config.sample_rate != 16_000:
                        await websocket.send_json(
                            {
                                "type": "error",
                                "message": "This service currently supports sample_rate=16000 only.",
                            }
                        )
                        continue

                    session = StreamingSession(websocket.app.state.model, config)
                    session.start()
                    await websocket.send_json({"type": "started", "sample_rate": config.sample_rate})

                elif msg_type == "flush":
                    if session is None:
                        await websocket.send_json({"type": "error", "message": "Send start before flush."})
                        continue
                    await websocket.send_json(session.flush())

                elif msg_type == "stop":
                    if session is None:
                        await websocket.send_json(
                            {"type": "final", "text": "", "is_final": True}
                        )
                    else:
                        await websocket.send_json(session.stop())
                        session = None

                else:
                    await websocket.send_json(
                        {
                            "type": "error",
                            "message": "Unsupported control message type. Use start/flush/stop.",
                        }
                    )

            elif "bytes" in message and message["bytes"] is not None:
                if session is None:
                    await websocket.send_json({"type": "error", "message": "Send start before audio."})
                    continue

                try:
                    payload = session.add_pcm16_chunk(message["bytes"])
                except ValueError as err:
                    await websocket.send_json({"type": "error", "message": str(err)})
                    continue
                await websocket.send_json(payload)

            elif message.get("type") == "websocket.disconnect":
                break

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected.")
    except Exception:
        logger.exception("Unexpected websocket error.")
        if websocket.client_state.name == "CONNECTED":
            await websocket.send_json({"type": "error", "message": "Server error."})
    finally:
        if session is not None:
            session.close()


def _parse_json(text_message: str) -> dict[str, Any] | None:
    try:
        parsed = json.loads(text_message)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None
