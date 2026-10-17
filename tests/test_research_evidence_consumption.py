import asyncio

from app.engine import ConversationEngine
from app.model_profiles import get_profile
from app.models import ConversationResult, GenerationRequest, ResearchFact, ResearchSource
from app.ollama_client import OllamaResult
from app.research import ResearchBundle, extract_verified_facts
from app.validator import validate_semantics


def request():
    return GenerationRequest.model_validate(
        {
            "language": "de",
            "topic": (
                "Lukas und Markus diskutieren, welche offiziellen technischen Daten "
                "für den Ender 3 V2 zur maximalen Düsentemperatur und zum Heizbett "
                "dokumentiert sind. Sie sollen nur belegte Angaben verwenden."
            ),
            "entry_count": 4,
            "tone": "natural",
            "chaos": 10,
            "seriousness": 75,
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


def official_source():
    return ResearchSource(
        id="S1",
        title="Ender-3 V2 vs Ender-3 Neo vs Ender-3 V2 Neo - Creality Store",
        url="https://store.creality.com/blogs/all/ender-3-v2-3-neo-3-v2-neo-comparison-guide",
        snippet=(
            "Ender-3 V2 Max. Heatbed Temperature 100℃. "
            "Max. Nozzle Temperature 260℃."
        ),
        engine="mock",
        identity_match="exact",
        identity_basis="title_url",
    )


def test_extracts_primary_verified_facts_from_official_exact_model_source():
    facts = extract_verified_facts(request(), [official_source()], brand_hint="Creality")
    by_aspect = {fact.aspect: fact for fact in facts}

    assert by_aspect["maximale Düsentemperatur"].value == "260 °C"
    assert by_aspect["maximale Düsentemperatur"].confidence == "primary"
    assert by_aspect["maximale Heizbetttemperatur"].value == "100 °C"
    assert by_aspect["maximale Heizbetttemperatur"].confidence == "primary"


def test_validator_rejects_sufficient_research_conversation_that_omits_verified_values():
    result = ConversationResult.model_validate(
        {
            "schema_version": "1.0",
            "language": "de",
            "entries": [
                {"type": "message", "speaker_id": "p1", "text": "Die Werte müssen wir noch prüfen.", "typing_indicator": False, "attachment": {"type": "none"}},
                {"type": "message", "speaker_id": "p2", "text": "Dann schauen wir in die Herstellerangaben.", "typing_indicator": False, "attachment": {"type": "none"}},
                {"type": "message", "speaker_id": "p1", "text": "Wir bleiben bei belegten Daten.", "typing_indicator": False, "attachment": {"type": "none"}},
                {"type": "message", "speaker_id": "p2", "text": "Einverstanden.", "typing_indicator": False, "attachment": {"type": "none"}},
            ],
        }
    )
    facts = [
        ResearchFact(aspect="maximale Düsentemperatur", value="260 °C", confidence="primary", source_ids=["S1"]),
        ResearchFact(aspect="maximale Heizbetttemperatur", value="100 °C", confidence="primary", source_ids=["S1"]),
    ]

    violations = validate_semantics(
        request(),
        result,
        factual_context="260 °C 100 °C Ender 3 V2",
        required_research_facts=facts,
    )

    missing = [v for v in violations if v.code == "MISSING_VERIFIED_RESEARCH_FACT"]
    assert len(missing) == 2
    assert any("260 °C" in (v.expected or "") for v in missing)
    assert any("100 °C" in (v.expected or "") for v in missing)


class SufficientResearchService:
    async def research(self, req):
        source = official_source()
        facts = extract_verified_facts(req, [source], brand_hint="Creality")
        return ResearchBundle(
            provider="searxng",
            queries=["official"],
            sources=[source],
            elapsed_seconds=0.01,
            evidence_sufficient=True,
            verified_facts=facts,
            target_identity="Ender 3 V2",
        )


class RepairingClient:
    def __init__(self):
        self.calls = 0
        self.system = ""

    async def chat(self, **kwargs):
        self.calls += 1
        self.system = kwargs["messages"][0]["content"]
        if self.calls == 1:
            content = (
                '{"e":['
                '{"s":"p1","t":"Die genauen Maximalwerte müssen wir noch prüfen."},'
                '{"s":"p2","t":"Dann bleiben wir vorerst bei der Dokumentation."},'
                '{"s":"p1","t":"Wir sollten nichts Unbelegtes übernehmen."},'
                '{"s":"p2","t":"Einverstanden, wir prüfen die Herstellerangaben."}'
                ']}'
            )
        else:
            content = (
                '{"e":['
                '{"s":"p1","t":"Creality nennt für die maximale Düsentemperatur 260 °C."},'
                '{"s":"p2","t":"Für das Heizbett sind maximal 100 °C angegeben."},'
                '{"s":"p1","t":"Damit haben wir für beide Punkte offizielle Werte."},'
                '{"s":"p2","t":"Dann bleiben wir genau bei diesen belegten Angaben."}'
                ']}'
            )
        return OllamaResult(
            content=content,
            prompt_tokens=10,
            generated_tokens=20,
            load_duration_ns=1,
            prompt_eval_duration_ns=1,
            eval_duration_ns=1,
            total_duration_ns=3,
        )


def test_engine_deterministically_grounds_dialogue_before_llm_repair():
    engine = ConversationEngine(research_service=SufficientResearchService())
    client = RepairingClient()
    engine.client = client

    response = asyncio.run(engine.generate(request(), profile=get_profile("gpt-oss-20b")))

    text = " ".join(entry.text for entry in response.conversation.entries)
    assert client.calls == 1
    assert response.meta.repair_attempts == 0
    assert "260 °C" in text
    assert "100 °C" in text
    assert response.meta.research is not None
    assert len(response.meta.research.verified_facts) == 2
    assert "Required verified facts" in client.system


class StubbornRepairClient:
    """Simulates a model that ignores verified facts even during repairs."""

    def __init__(self):
        self.calls = 0

    async def chat(self, **kwargs):
        self.calls += 1
        content = (
            '{"e":['
            '{"s":"p1","t":"Die genauen Maximalwerte müssen wir noch prüfen."},'
            '{"s":"p2","t":"Dann suchen wir zuerst eine belastbare Quelle."},'
            '{"s":"p1","t":"Ohne die Dokumentation wäre das nur eine Vermutung."},'
            '{"s":"p2","t":"Einverstanden, wir schlagen die technischen Angaben nach."}'
            ']}'
        )
        return OllamaResult(
            content=content,
            prompt_tokens=10,
            generated_tokens=20,
            load_duration_ns=1,
            prompt_eval_duration_ns=1,
            eval_duration_ns=1,
            total_duration_ns=3,
        )


def test_engine_deterministically_injects_verified_facts_without_wasting_repair_calls():
    engine = ConversationEngine(research_service=SufficientResearchService())
    client = StubbornRepairClient()
    engine.client = client

    response = asyncio.run(engine.generate(request(), profile=get_profile("gpt-oss-20b")))

    text = " ".join(entry.text for entry in response.conversation.entries)
    assert client.calls == 1
    assert response.meta.repair_attempts == 0
    assert "260 °C" in text
    assert "100 °C" in text
    assert "müssen wir noch prüfen" not in text.casefold()
    assert response.meta.feature_adjustments.verified_research_facts_injected == 2
    assert response.meta.feature_adjustments.stale_research_uncertainty_rewritten >= 1
