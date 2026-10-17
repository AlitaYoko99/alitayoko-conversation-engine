# AlitaYoko Conversation Engine

Self-hosted AI backend for the **AlitaYoko Conversations** WordPress plugin.

The engine turns a structured conversation request into a validated conversation JSON document. It is designed for editorial dialogue generation rather than general-purpose chat, and can optionally use self-hosted web research before generation and during the final factual audit.

> **Current release:** `0.9.8`  
> **Status:** stable engine API; the AI Generator in the WordPress plugin is released as a **BETA feature**.

## What it does

- Generates structured conversations with multiple participants.
- Supports narration, typing indicators, image intents and link intents.
- Supports multilingual output controlled by the request language.
- Uses Ollama as the local model runtime.
- Ships tuned profiles for `gpt-oss:20b` and `ministral-3:14b`.
- Optionally performs self-hosted web research through SearXNG.
- Can fetch bounded public HTML/PDF sources after search discovery.
- Runs deterministic grounding, semantic validation and repair.
- Runs a final factual and language/coherence audit before returning JSON.
- Exposes a small HTTP API designed for the AlitaYoko Conversations WordPress plugin.

## Architecture

```text
WordPress / another client
        │ HTTPS + X-AY-API-Key
        ▼
AlitaYoko Conversation Engine :8787
        │
        ├── Ollama :11434
        │     └── gpt-oss:20b (default)
        │
        └── optional SearXNG
              └── bounded direct-source fetching
```

The recommended public deployment keeps the engine bound to `127.0.0.1:8787` and places a TLS reverse proxy in front of it. Do **not** expose Ollama directly to the Internet.

## Requirements

- Python 3.11+
- Ollama
- A locally available model (default: `gpt-oss:20b`)
- Optional: SearXNG for the Research feature

Ollama runs models locally; performance depends heavily on the available CPU/GPU and memory. On Apple Silicon, native Ollama is recommended rather than running Ollama inside Docker.

## Quick start

### 1. Install and prepare Ollama

Install Ollama for your platform, then pull the default model:

```bash
ollama pull gpt-oss:20b
```

Confirm that Ollama answers locally:

```bash
curl http://127.0.0.1:11434/api/version
```

### 2. Install the engine

```bash
git clone https://github.com/AlitaYoko99/alitayoko-conversation-engine.git
cd alitayoko-conversation-engine

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e .

cp .env.example .env
```

Generate an API key and put it into `.env` as `AY_API_KEY`:

```bash
openssl rand -hex 32
```

### 3. Start the API

```bash
.venv/bin/python -m uvicorn app.main:app \
  --host 127.0.0.1 \
  --port 8787
```

Health check:

```bash
curl http://127.0.0.1:8787/health
```

### 4. Generate a test conversation

```bash
curl -sS -X POST http://127.0.0.1:8787/v1/conversations/generate \
  -H 'Content-Type: application/json' \
  -H 'X-AY-API-Key: YOUR_API_KEY' \
  --data-binary @examples/request.de.json
```

## WordPress integration

In AlitaYoko Conversations, open **Conversations → AI Settings** and select:

```text
AlitaYoko Conversation Engine (self-hosted)
```

Then enter the public HTTPS endpoint and the matching API key.

For managed installations the same values can be configured in `wp-config.php`:

```php
define( 'ALYCON_GENERATOR_BACKEND', 'engine' );
define( 'ALYCON_ENGINE_URL', 'https://engine.example.com' );
define( 'ALYCON_ENGINE_API_KEY', 'replace-with-your-key' );
```

See [docs/wordpress.md](docs/wordpress.md).

## API

| Method | Endpoint | Auth | Purpose |
|---|---|---:|---|
| `GET` | `/health` | No | Health/status information |
| `GET` | `/v1/model-profiles` | Yes | Available model profiles |
| `POST` | `/v1/warmup` | Yes | Preload the configured model |
| `POST` | `/v1/schema/preview` | Yes | Preview the generated response schema |
| `POST` | `/v1/conversations/generate` | Yes | Generate a conversation |

Authentication uses:

```text
X-AY-API-Key: <your key>
```

When `AY_API_KEY` is empty, protected endpoints accept unauthenticated requests. That is useful for local development only and is **not recommended** for an Internet-facing deployment.

## Research

Research is optional and intentionally separate from visible link attachments.

- `features.research=true` allows factual research.
- Research uses a self-hosted SearXNG endpoint configured with `AY_SEARXNG_BASE_URL`.
- Search discovery can be followed by bounded direct fetching of selected public HTML/PDF sources.
- Reverse Research in Final Audit is only allowed when the original request enabled Research.
- Retrieved web content is treated as untrusted evidence, not as instructions.

If Research is enabled in a request but SearXNG is not configured, the engine fails explicitly rather than silently pretending that research occurred.

## Configuration

Copy `.env.example` to `.env`. Important settings include:

```text
AY_OLLAMA_BASE_URL=http://127.0.0.1:11434
AY_OLLAMA_MODEL=gpt-oss:20b
AY_API_KEY=change-me
AY_NUM_CTX=4096
AY_KEEP_ALIVE=30m
AY_PREWARM_ON_START=true
```

Research and final-audit settings are documented directly in `.env.example`.

## Production deployment

Detailed guides:

- [macOS](docs/installation-macos.md)
- [Linux](docs/installation-linux.md)
- [Docker](docs/docker.md)
- [HTTPS reverse proxy](docs/reverse-proxy.md)
- [WordPress](docs/wordpress.md)
- [Security](SECURITY.md)

## Tests

The exact `0.9.8` release source passes **175 automated tests**.

```bash
python -m pip install -e '.[dev]'
pytest -q
```

## v0.9.8

`0.9.8` tightens deterministic handling of unsupported recommendations. When Research verifies compatibility/support but does not verify the requested recommendation, phrasing such as “for stability this would be safer” or “the only confirmed option” is neutralized instead of being presented as sourced advice.

See [CHANGELOG.md](CHANGELOG.md).

## License

GPL-3.0-or-later. See [LICENSE](LICENSE).
