# Security

## Supported version

Security fixes are currently targeted at the latest published engine release.

## Deployment rules

For an Internet-facing deployment:

1. Set a strong `AY_API_KEY`.
2. Keep Uvicorn bound to `127.0.0.1:8787` or another private interface.
3. Put a TLS reverse proxy in front of the engine.
4. Do not expose Ollama (`11434`) directly to the Internet.
5. Do not commit `.env` or other credential files.
6. Keep SearXNG private or independently authenticated if it is reachable outside the host.

`/health` is intentionally unauthenticated and returns engine/model/research status metadata. If that information should not be public, restrict `/health` at the reverse proxy.

## API key

Protected endpoints use the `X-AY-API-Key` header. If `AY_API_KEY` is empty, authentication is disabled; use that only for isolated local development.

Generate a strong key with:

```bash
openssl rand -hex 32
```

## Research / direct fetching

The Research subsystem may retrieve public HTTP(S) sources. It includes safeguards against private, loopback, link-local and reserved destinations, revalidates redirects, restricts ports, and caps response size and PDF processing.

Fetched pages are untrusted evidence and must never be treated as instructions.

## Reporting a vulnerability

Please report suspected vulnerabilities privately to the project maintainer rather than publishing exploit details in a public issue before a fix is available.
