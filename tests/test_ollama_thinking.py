import asyncio

from app.ollama_client import OllamaClient


class FakeResponse:
    def raise_for_status(self):
        pass

    def json(self):
        return {
            "message": {
                "content": '{"e":[]}',
                "thinking": "short reasoning",
            },
            "done_reason": "stop",
            "prompt_eval_count": 10,
            "eval_count": 20,
            "load_duration": 1,
            "prompt_eval_duration": 2,
            "eval_duration": 3,
            "total_duration": 6,
        }


class FakeAsyncClient:
    last_payload = None

    def __init__(self, *args, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    async def post(self, url, json):
        type(self).last_payload = json
        return FakeResponse()


def test_chat_sends_native_think_parameter(monkeypatch):
    import app.ollama_client as module

    monkeypatch.setattr(module.httpx, "AsyncClient", FakeAsyncClient)

    client = OllamaClient("http://127.0.0.1:11434")
    result = asyncio.run(
        client.chat(
            model="gpt-oss:20b",
            messages=[{"role": "user", "content": "JSON please"}],
            schema={"type": "object"},
            temperature=0.2,
            num_ctx=4096,
            num_predict=1200,
            keep_alive="30m",
            think="low",
        )
    )

    assert FakeAsyncClient.last_payload["think"] == "low"
    assert result.thinking == "short reasoning"
    assert result.done_reason == "stop"
