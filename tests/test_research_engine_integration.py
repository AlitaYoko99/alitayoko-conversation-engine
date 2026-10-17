import asyncio

from app.engine import ConversationEngine
from app.model_profiles import get_profile
from app.models import GenerationRequest, ResearchFact, ResearchSource
from app.ollama_client import OllamaResult
from app.research import ResearchBundle


def make_request():
    return GenerationRequest.model_validate(
        {
            "language": "de",
            "topic": "Lukas und Markus besprechen einen 3D-Drucker und seine Firmware.",
            "entry_count": 4,
            "tone": "natural",
            "chaos": 10,
            "seriousness": 80,
            "participants": [
                {"id": "p1", "name": "Lukas"},
                {"id": "p2", "name": "Markus"},
            ],
            "features": {
                "narration": False,
                "typing_indicator": False,
                "images": False,
                "links": False,
                "research": True,
            },
            "limits": {
                "max_narrations": 0,
                "max_words_per_entry": 30,
                "max_images": 0,
                "max_links": 0,
            },
        }
    )


class FakeResearchService:
    async def research(self, request):
        return ResearchBundle(
            provider="searxng",
            queries=["3D-Drucker Firmware"],
            sources=[
                ResearchSource(
                    id="S1",
                    title="Hersteller-Dokumentation",
                    url="https://example.test/docs",
                    snippet="Für dieses Gerät ist Firmware-Version 1.4.2 dokumentiert.",
                    engine="mock",
                )
            ],
            elapsed_seconds=0.01,
            evidence_sufficient=True,
            verified_facts=[
                ResearchFact(
                    aspect="Firmware-Version",
                    value="1.4.2",
                    confidence="secondary",
                    source_ids=["S1"],
                )
            ],
        )


class FakeClient:
    def __init__(self):
        self.calls = 0
        self.last_system = ""

    async def chat(self, **kwargs):
        self.calls += 1
        self.last_system = kwargs["messages"][0]["content"]
        return OllamaResult(
            content=(
                '{"e":['
                '{"s":"p1","t":"Die dokumentierte Firmware-Version ist 1.4.2."},'
                '{"s":"p2","t":"Dann können wir diesen Stand als Ausgangspunkt verwenden."},'
                '{"s":"p1","t":"Weitere technische Werte sollten wir nur übernehmen, wenn sie ebenfalls belegt sind."},'
                '{"s":"p2","t":"Einverstanden, wir bleiben bei den recherchierten Angaben."}'
                ']}'
            ),
            prompt_tokens=10,
            generated_tokens=20,
            load_duration_ns=1,
            prompt_eval_duration_ns=1,
            eval_duration_ns=1,
            total_duration_ns=3,
        )


def test_research_evidence_flows_into_generation_and_meta():
    engine = ConversationEngine(research_service=FakeResearchService())
    fake = FakeClient()
    engine.client = fake

    response = asyncio.run(
        engine.generate(make_request(), profile=get_profile("gpt-oss-20b"))
    )

    assert fake.calls == 1
    assert "Firmware-Version: 1.4.2" in fake.last_system
    assert response.meta.research is not None
    assert response.meta.research.enabled is True
    assert response.meta.research.source_count == 1
    assert response.meta.feature_usage.links == 0
    assert response.meta.feature_adjustments.technical_specifics_neutralized == 0
    assert "1.4.2" in response.conversation.entries[0].text
