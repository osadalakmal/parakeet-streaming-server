#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import wave

import websockets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stream a WAV file to the local websocket service.")
    parser.add_argument("wav_path", help="Path to a mono 16-bit PCM WAV file")
    parser.add_argument("--url", default="ws://127.0.0.1:8765/ws/transcribe")
    parser.add_argument(
        "--chunk-ms",
        type=int,
        default=250,
        help="Chunk duration in milliseconds (recommended 200-500)",
    )
    parser.add_argument("--language", default=None, help="Language code (e.g. 'en'), or omit for auto-detect")
    parser.add_argument("--beam-size", type=int, default=None, help="Beam size for decoding (omit for greedy)")
    return parser.parse_args()


async def recv_updates(ws: websockets.WebSocketClientProtocol) -> None:
    try:
        async for message in ws:
            if isinstance(message, bytes):
                continue
            payload = json.loads(message)
            msg_type = payload.get("type")
            if msg_type == "partial":
                print(
                    f"[partial] finalized='{payload.get('finalized_text', '')}' "
                    f"live='{payload.get('text', '')}'"
                )
            elif msg_type == "final":
                print(f"[final] {payload.get('text', '')}")
            else:
                print(f"[{msg_type}] {payload}")
    except websockets.ConnectionClosed:
        return


async def stream_wav(args: argparse.Namespace) -> None:
    with wave.open(args.wav_path, "rb") as wav_file:
        sample_rate = wav_file.getframerate()
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()

        if channels != 1:
            raise ValueError("Expected mono WAV input.")
        if sample_width != 2:
            raise ValueError("Expected 16-bit PCM WAV input.")
        if sample_rate != 16_000:
            raise ValueError("Expected 16 kHz WAV input.")

        frames_per_chunk = int(sample_rate * (args.chunk_ms / 1000.0))

        async with websockets.connect(args.url, max_size=4_000_000) as ws:
            receiver = asyncio.create_task(recv_updates(ws))

            await ws.send(
                json.dumps(
                    {
                        "type": "start",
                        "sample_rate": sample_rate,
                        "language": args.language,
                        "beam_size": args.beam_size,
                    }
                )
            )

            while True:
                chunk = wav_file.readframes(frames_per_chunk)
                if not chunk:
                    break
                await ws.send(chunk)
                await asyncio.sleep(args.chunk_ms / 1000.0)

            await ws.send(json.dumps({"type": "flush"}))
            await ws.send(json.dumps({"type": "stop"}))
            await asyncio.sleep(0.5)
            await ws.close()
            await receiver


def main() -> None:
    args = parse_args()
    asyncio.run(stream_wav(args))


if __name__ == "__main__":
    main()
