import asyncio

from app.engine import ConversationEngine
from app.model_profiles import get_profile
from app.models import GenerationRequest
from app.ollama_client import OllamaResult


def make_request():
    return GenerationRequest.model_validate(
        {
            "language":"de",
            "topic":"Lukas und Markus richten einen gebrauchten 3D-Drucker ein.",
            "entry_count":6,
            "tone":"natural",
            "chaos":10,
            "participants":[{"id":"p1","name":"Lukas"},{"id":"p2","name":"Markus"}],
            "features":{"narration":True,"typing_indicator":True,"images":True,"links":True},
            "limits":{"max_narrations":1,"max_words_per_entry":35,"max_images":1,"max_links":1},
        }
    )


class FakeClient:
    def __init__(self): self.calls = 0
    async def chat(self, **kwargs):
        self.calls += 1
        return OllamaResult(
            content=(
                '{"e":['
                '{"s":"p1","t":"Der Halter sitzt schief."},'
                '{"s":"p2","t":"Dann suchen wir die Anleitung.","a":{"k":"l","q":"3D-Drucker Bedienungsanleitung","x":"Bedienungsanleitung"}},'
                '{"s":"p1","t":"Okay."},'
                '{"s":"p2","t":"Ich prüfe die Halterung.","a":{"k":"i","d":"3D-Drucker mit schiefem Filamenthalter","x":"Schiefer Filamenthalter"}},'
                '{"s":"p1","t":"Dann richten wir sie neu aus."},'
                '{"s":"p2","t":"Danach testen wir den Drucker."}'
                ']}'
            ),
            prompt_tokens=10, generated_tokens=20,
            load_duration_ns=1, prompt_eval_duration_ns=1,
            eval_duration_ns=1, total_duration_ns=3,
        )


def test_missing_narration_does_not_trigger_llm_repair():
    engine = ConversationEngine()
    fake = FakeClient()
    engine.client = fake

    response = asyncio.run(engine.generate(make_request(), profile=get_profile("gpt-oss-20b")))

    assert fake.calls == 1
    assert response.meta.repair_attempts == 0
    assert response.meta.feature_adjustments.narrations_added == 1
    assert response.meta.feature_usage.narrations == 1
    assert response.meta.feature_usage.images == 1
    assert response.meta.feature_usage.links == 1
    assert response.meta.feature_usage.typing_indicators == 1
