import asyncio

from app.engine import ConversationEngine
from app.model_profiles import get_profile
from app.models import GenerationRequest
from app.ollama_client import OllamaResult


def make_request():
    return GenerationRequest.model_validate(
        {
            "language": "de",
            "topic": (
                "Lukas und Markus richten einen gebrauchten 3D-Drucker ein "
                "und möchten die Bedienungsanleitung nachschlagen."
            ),
            "entry_count": 4,
            "tone": "natural",
            "chaos": 10,
            "participants": [
                {"id": "p1", "name": "Lukas"},
                {"id": "p2", "name": "Markus"},
            ],
            "features": {
                "narration": True,
                "typing_indicator": True,
                "images": False,
                "links": True,
            },
            "limits": {
                "max_narrations": 1,
                "max_words_per_entry": 30,
                "max_images": 0,
                "max_links": 1,
            },
        }
    )


class FakeClient:
    def __init__(self):
        self.calls = 0

    async def chat(self, **kwargs):
        self.calls += 1
        return OllamaResult(
            content=(
                '{"e":['
                '{"s":"p1","t":"Wir sollten zuerst die Bedienungsanleitung prüfen.",'
                '"a":{"k":"l","q":"https://example.com/manual","x":"example.com"}},'
                '{"n":"Markus stellt den Drucker auf den Tisch."},'
                '{"s":"p2","t":"Gut, dann ändern wir vorher nichts."},'
                '{"s":"p1","t":"Danach vergleichen wir die Einstellungen."}'
                ']}'
            ),
            prompt_tokens=10,
            generated_tokens=20,
            load_duration_ns=1,
            prompt_eval_duration_ns=1,
            eval_duration_ns=1,
            total_duration_ns=3,
        )


def test_unsafe_model_link_is_normalized_without_repair_call():
    engine = ConversationEngine()
    fake = FakeClient()
    engine.client = fake

    response = asyncio.run(
        engine.generate(
            make_request(),
            profile=get_profile("gpt-oss-20b"),
        )
    )

    assert fake.calls == 1
    assert response.meta.repair_attempts == 0
    assert response.meta.feature_adjustments.links_sanitized == 1
    assert response.meta.feature_usage.links == 1

    link = next(
        entry.attachment
        for entry in response.conversation.entries
        if entry.attachment.type == "link"
    )
    assert "http" not in link.query.casefold()
    assert "." not in link.label
