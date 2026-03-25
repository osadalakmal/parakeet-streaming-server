# Parakeet Local Streaming Service (macOS Apple Silicon)

Tiny streaming-first local transcription service built on `parakeet-mlx` using its `transcribe_stream(...)` API (no existing wrapper used).

## What this is

- Local FastAPI + WebSocket server for low-latency dictation workflows.
- Loads the Parakeet model once at startup and keeps it warm.
- Per-WebSocket session state uses `transcribe_stream`, `add_audio`, and incremental reads from `transcriber.result`.
- Simple test client that streams a WAV file in small chunks.

## Requirements

- macOS Apple Silicon
- Python 3.12
- Local audio input format for v1: **raw PCM16LE mono at 16 kHz** over websocket binary messages

## Setup

Using `venv`:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

Using `uv`:

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e .
```

## Run the server

```bash
uvicorn app.main:app --host 127.0.0.1 --port 8765
```

Endpoints:

- `GET /healthz`
- `GET /debug/config`
- `WS /ws/transcribe`

## WebSocket protocol

Client -> server control (JSON text):

```json
{"type":"start","sample_rate":16000,"context_size":256,"depth":8,"keep_original_attention":false}
```

```json
{"type":"flush"}
```

```json
{"type":"stop"}
```

Client -> server audio (binary):

- raw PCM16LE mono bytes at 16kHz
- suggested chunk size: 200ms to 500ms per frame

Server -> client messages (JSON):

```json
{"type":"ready"}
```

```json
{"type":"partial","text":"current partial transcript","finalized_text":"stable finalized portion","is_final":false}
```

```json
{"type":"final","text":"full final transcript","is_final":true}
```

```json
{"type":"error","message":"..."}
```

## Test client (WAV streaming)

`scripts/test_stream_client.py` streams a local WAV file in chunks.

Expected WAV format:

- mono
- 16-bit PCM
- 16 kHz

Run:

```bash
python scripts/test_stream_client.py /path/to/audio.wav --chunk-ms 250
```

## Obsidian plugin integration notes

- Connect directly to `ws://127.0.0.1:8765/ws/transcribe` from the plugin process.
- Send `start` first, then PCM16 audio chunks as websocket binary frames.
- Render `partial` messages live in editor UI.
- Replace/commit text on `final`.
- Because this binds to localhost and uses websocket for core transport, this avoids most browser-style CORS friction for local plugin integration.

## Tunables for latency/quality experiments

Pass with `start` control message:

- `context_size`
- `depth`
- `keep_original_attention`

Also experiment with smaller/larger `chunk-ms` on client side (200–500 ms recommended).

## Tradeoffs

### WebSocket vs HTTP

- WebSocket fits incremental low-latency transcription and bi-directional events naturally.
- HTTP batch uploads are simpler for one-shot processing, but worse UX for live dictation.

### PCM vs WAV uploads

- PCM binary frames are simple and low-overhead for streaming.
- WAV uploads include headers and are better for offline file workflows.

### Partial vs final semantics

- `partial` is for live, unstable text display.
- `final` is best-effort stable text at stop/flush boundary.
- Depending on model behavior, finalized and live text may occasionally overlap and should be reconciled on the client.
