# HTTPS reverse proxy

The recommended deployment is:

```text
Internet
  → HTTPS reverse proxy :443
  → 127.0.0.1:8787
  → AlitaYoko Conversation Engine
  → 127.0.0.1:11434 (Ollama)
```

Only the reverse proxy should be exposed publicly.

## Caddy example

```caddy
engine.example.com {
    reverse_proxy 127.0.0.1:8787
}
```

When a public hostname is used and ports 80/443 are reachable, Caddy can manage HTTPS certificates automatically.

## Firewall

Do not expose these backend ports to the Internet:

```text
8787  Conversation Engine
11434 Ollama
```

If SearXNG is local, keep its port private too.

## Health endpoint

`/health` is intentionally unauthenticated. Restrict it in the proxy if you do not want engine/model metadata visible publicly.
