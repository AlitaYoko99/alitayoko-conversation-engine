import asyncio

from app.engine import ConversationEngine
from app.model_profiles import get_profile
from app.models import GenerationRequest
from app.ollama_client import OllamaResult


def make_request():
    return GenerationRequest.model_validate(
        {
            "language": "de",
            "topic": "Zwei Personen besprechen einen Termin.",
            "entry_count": 2,
            "tone": "natural",
            "chaos": 10,
            "participants": [
                {"id": "p1", "name": "Lukas"},
                {"id": "p2", "name": "Markus"},
            ],
            "features": {
                "narration": False,
                "typing_indicator": False,
                "images": False,
                "links": False,
            },
            "limits": {
                "max_narrations": 0,
                "max_words_per_entry": 30,
                "max_images": 0,
                "max_links": 0,
            },
        }
    )


class FakeClient:
    def __init__(self):
        self.last_kwargs = None

    async def chat(self, **kwargs):
        self.last_kwargs = kwargs
        return OllamaResult(
            content='{"e":[{"s":"p1","t":"Passt dir zehn Uhr?"},{"s":"p2","t":"Ja, das passt."}]}',
            prompt_tokens=10,
            generated_tokens=20,
            load_duration_ns=1,
            prompt_eval_duration_ns=1,
            eval_duration_ns=1,
            total_duration_ns=3,
        )


def test_gpt_oss_profile_uses_native_thinking_and_larger_budget():
    engine = ConversationEngine()
    fake = FakeClient()
    engine.client = fake

    response = asyncio.run(
        engine.generate(
            make_request(),
            profile=get_profile("gpt-oss-20b"),
        )
    )

    assert fake.last_kwargs["model"] == "gpt-oss:20b"
    assert fake.last_kwargs["think"] == "low"
    assert "Exact JSON schema:" in fake.last_kwargs["messages"][0]["content"]
    assert fake.last_kwargs["num_predict"] > 520
    assert response.meta.model_profile == "gpt-oss-20b"


def test_ministral_profile_has_no_native_thinking():
    engine = ConversationEngine()
    fake = FakeClient()
    engine.client = fake

    asyncio.run(
        engine.generate(
            make_request(),
            profile=get_profile("ministral-14b"),
        )
    )

    assert fake.last_kwargs["model"] == "ministral-3:14b"
    assert fake.last_kwargs["think"] is None
    assert "Exact JSON schema:" not in fake.last_kwargs["messages"][0]["content"]
