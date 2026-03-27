from __future__ import annotations

import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager
from typing import Any

import mlx_whisper
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from app.session import StreamConfig, StreamingSession

LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger("parakeet_streaming_service")

DEFAULT_MODEL = os.getenv("WHISPER_MODEL", "mlx-community/whisper-large-v3-turbo")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Warming up Whisper model: %s", DEFAULT_MODEL)
    # Run a silent dummy transcription so the model weights are loaded and
    # compiled before the first real request arrives.
    await asyncio.to_thread(
        mlx_whisper.transcribe,
        np.zeros(16_000, dtype=np.float32),
        path_or_hf_repo=DEFAULT_MODEL,
    )
    app.state.model_path = DEFAULT_MODEL
    logger.info("Model ready.")
    yield
    logger.info("Shutting down streaming service.")


app = FastAPI(title="Whisper Local Streaming Service", lifespan=lifespan)
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
            "language": None,
            "beam_size": 5,
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
                        language=control.get("language") or None,
                        beam_size=int(control.get("beam_size", 5)),
                    )
                    if config.sample_rate != 16_000:
                        await websocket.send_json(
                            {
                                "type": "error",
                                "message": "This service currently supports sample_rate=16000 only.",
                            }
                        )
                        continue

                    session = StreamingSession(websocket.app.state.model_path, config)
                    session.start()
                    await websocket.send_json({"type": "started", "sample_rate": config.sample_rate})

                elif msg_type == "flush":
                    if session is None:
                        await websocket.send_json({"type": "error", "message": "Send start before flush."})
                        continue
                    payload = await asyncio.to_thread(session.flush)
                    await websocket.send_json(payload)

                elif msg_type == "stop":
                    if session is None:
                        await websocket.send_json(
                            {"type": "final", "text": "", "is_final": True}
                        )
                    else:
                        payload = await asyncio.to_thread(session.stop)
                        await websocket.send_json(payload)
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
