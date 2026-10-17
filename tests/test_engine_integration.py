import asyncio

from app.engine import ConversationEngine
from app.models import GenerationRequest
from app.ollama_client import OllamaResult


def make_request():
    return GenerationRequest.model_validate(
        {
            "language": "de",
            "topic": "Lukas und Markus prüfen ein technisches Problem.",
            "entry_count": 2,
            "tone": "natural",
            "chaos": 20,
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
                "max_words_per_entry": 35,
                "max_images": 0,
                "max_links": 0,
            },
        }
    )


def response(content: str) -> OllamaResult:
    return OllamaResult(
        content=content,
        prompt_tokens=10,
        generated_tokens=10,
        load_duration_ns=1_000_000,
        prompt_eval_duration_ns=2_000_000,
        eval_duration_ns=3_000_000,
        total_duration_ns=6_000_000,
    )


class FakeClient:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = 0

    async def chat(self, **kwargs):
        self.calls += 1
        if not self.outputs:
            raise AssertionError("FakeClient received more calls than expected")
        return response(self.outputs.pop(0))


def run_engine(outputs):
    engine = ConversationEngine()
    fake = FakeClient(outputs)
    engine.client = fake
    result = asyncio.run(engine.generate(make_request()))
    return result, fake.calls


def test_initial_parse_error_regenerates_without_attribute_error():
    result, calls = run_engine(
        [
            "not-json",
            '{"e":[{"s":"p1","t":"Ich prüfe das."},{"s":"p2","t":"Gut, ich warte."}]}',
        ]
    )

    assert calls == 2
    assert result.meta.generation_attempts == 2
    assert result.meta.repair_attempts == 0
    assert result.conversation.entries[0].speaker_id == "p1"


def test_semantic_repair_path_does_not_use_removed_adjustments_attribute():
    result, calls = run_engine(
        [
            '{"e":[{"s":"p1","t":"Siehe https://example.com"},{"s":"p2","t":"Das prüfe ich."}]}',
            '{"e":[{"s":"p1","t":"Ich habe den Hinweis gefunden."},{"s":"p2","t":"Gut, dann prüfen wir ihn."}]}',
        ]
    )

    assert calls == 2
    assert result.meta.generation_attempts == 1
    assert result.meta.repair_attempts == 1
    assert all("http" not in entry.text for entry in result.conversation.entries)


def test_grounded_repair_path_does_not_use_removed_adjustments_attribute():
    result, calls = run_engine(
        [
            '{"e":[{"s":"p1","t":"Siehe https://example.com"},{"s":"p2","t":"Das prüfe ich."}]}',
            "broken-repair",
            '{"e":[{"s":"p1","t":"Ich habe den Hinweis gefunden."},{"s":"p2","t":"Gut, dann prüfen wir ihn."}]}',
        ]
    )

    assert calls == 3
    assert result.meta.repair_attempts == 2
    assert result.conversation.entries[1].speaker_id == "p2"


def test_final_language_audit_runs_before_response(monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "final_audit_enabled", True)
    monkeypatch.setattr(settings, "final_audit_language_enabled", True)
    monkeypatch.setattr(settings, "final_audit_fact_enabled", True)

    engine = ConversationEngine()
    fake = FakeClient(
        [
            '{"e":[{"s":"p1","t":"Die Tests laufen stabil."},{"s":"p2","t":"Dann passt es."}]}',
            '{"issues":[{"entry_index":1,"kind":"grammar","replacement":"Dann passt es gut.","reason":"Natürlicher vollständiger Satz."}]}',
        ]
    )
    engine.client = fake
    result = asyncio.run(engine.generate(make_request()))

    assert fake.calls == 2
    assert result.meta.calls == 2
    assert result.meta.final_audit is not None
    assert result.meta.final_audit.enabled is True
    assert result.meta.final_audit.fact_audit_run is False
    assert result.meta.final_audit.language_audit_run is True
    assert result.meta.final_audit.language_rewrites == 1
    assert result.conversation.entries[1].text == "Dann passt es gut."


def test_parse_retry_gets_more_output_budget():
    class CapturingClient:
        def __init__(self):
            self.kwargs = []
            self.outputs = [
                "{\"e\":[{\"s\":\"p1\",\"t\":\"abgeschnitten\"}",
                '{"e":[{"s":"p1","t":"Ich prüfe das."},{"s":"p2","t":"Gut, ich warte."}]}',
            ]

        async def chat(self, **kwargs):
            self.kwargs.append(kwargs)
            return response(self.outputs.pop(0))

    engine = ConversationEngine()
    fake = CapturingClient()
    engine.client = fake
    result = asyncio.run(engine.generate(make_request()))

    assert result.meta.generation_attempts == 2
    assert len(fake.kwargs) == 2
    assert fake.kwargs[1]["num_predict"] > fake.kwargs[0]["num_predict"]
    assert fake.kwargs[1]["temperature"] <= fake.kwargs[0]["temperature"]
