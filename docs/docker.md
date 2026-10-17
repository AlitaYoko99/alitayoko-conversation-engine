# Docker

The engine can run in a container, but it still needs access to an Ollama server.

For **macOS/Apple Silicon**, native Ollama is recommended because Ollama's own guidance notes that Docker Desktop does not provide the Mac GPU path used by native Ollama.

## Build

```bash
docker build -t alitayoko-conversation-engine:0.9.8 .
```

## Linux: use host networking with a native Ollama service

With Ollama listening on `127.0.0.1:11434` on the Linux host:

```bash
docker run --rm \
  --network host \
  --env-file .env \
  alitayoko-conversation-engine:0.9.8
```

The container starts Uvicorn on `0.0.0.0:8787`; with host networking this becomes host port `8787`. Protect it with a firewall/reverse proxy.

## Remote Ollama

Set:

```text
AY_OLLAMA_BASE_URL=http://OLLAMA_HOST:11434
```

The Ollama endpoint should be on a trusted/private network. Do not expose Ollama directly to the public Internet.
