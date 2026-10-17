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
                "Lukas und Markus richten einen gebrauchten 3D-Drucker ein. "
                "Sie möchten zuerst die Bedienungsanleitung prüfen."
            ),
            "entry_count": 4,
            "tone": "natural",
            "chaos": 10,
            "participants": [
                {"id": "p1", "name": "Lukas"},
                {"id": "p2", "name": "Markus"},
            ],
            "features": {
                "narration": False,
                "typing_indicator": True,
                "images": False,
                "links": True,
            },
            "limits": {
                "max_narrations": 0,
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
                '{"s":"p1","t":"Das ist bestimmt ein Ender 3 V2."},'
                '{"s":"p2","t":"Der Extruder schafft 260 °C."},'
                '{"s":"p1","t":"Wir sollten die Anleitung prüfen.",'
                '"a":{"k":"l","q":"Ender 3 V2 official manual","x":"Ender 3 V2 Manual"}},'
                '{"s":"p2","t":"Danach ändern wir nichts auf Verdacht."}'
                ']}'
            ),
            prompt_tokens=10,
            generated_tokens=20,
            load_duration_ns=1,
            prompt_eval_duration_ns=1,
            eval_duration_ns=1,
            total_duration_ns=3,
        )


def test_factual_sanitizer_avoids_llm_repair():
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
    assert response.meta.feature_adjustments.technical_specifics_neutralized == 2
    assert response.meta.feature_adjustments.links_sanitized == 1
