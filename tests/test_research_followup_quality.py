import asyncio

from app.models import ConversationResult, GenerationRequest, ResearchSource
from app.content_normalizer import normalize_content_features
from app.research import ResearchService, assess_evidence_coverage, plan_followup_queries


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


INITIAL = [
    ResearchSource(
        id="S1",
        title="CREALITY ENDER-3 V2 BEDIENUNGSANLEITUNG Pdf-Herunterladen | ManualsLib",
        url="https://www.manualslib.de/manual/728910/Creality-Ender-3-V2.html",
        snippet=(
            "Die minimale Düsentemperatur, damit der Extruder arbeiten kann, ist 180°C. "
            "Heizbett Maximaltemperatur: 100°C."
        ),
        engine="brave",
    ),
    ResearchSource(
        id="S2",
        title="CREALITY ENDER-3 V2 USER MANUAL Pdf Download | ManualsLib",
        url="https://www.manualslib.com/manual/1869756/Creality-Ender-3-V2.html",
        snippet="User guide for assembly and use of the Ender-3 V2.",
        engine="duckduckgo",
    ),
]

OFFICIAL = ResearchSource(
    id="Sx",
    title="Ender Series 3D Printers Parameter Comparison | Creality Official",
    url="https://store.creality.com/blogs/all/ender-series-3d-printers-parameter-comparison",
    snippet=(
        "Ender-3 V2 Max. Heatbed Temperature 100℃. "
        "Max. Nozzle Temperature 260℃."
    ),
    engine="mock",
)


def test_initial_manual_snippets_are_insufficient_for_official_maximum_values():
    coverage = assess_evidence_coverage(request(), INITIAL, brand_hint="Creality")

    assert coverage.sufficient is False
    assert "maximale Düsentemperatur" in coverage.missing_aspects


def test_followup_query_uses_inferred_manufacturer_and_missing_fact():
    queries = plan_followup_queries(
        request(),
        INITIAL,
        ["maximale Düsentemperatur"],
        1,
    )

    assert len(queries) == 1
    assert "Creality" in queries[0]
    assert "Ender 3 V2" in queries[0]
    assert '"Max. Nozzle Temperature"' in queries[0]
    assert "official specifications" in queries[0]
    assert "site:creality.com" in queries[0]


class FollowupProvider:
    def __init__(self):
        self.calls = []

    async def search(self, query: str, *, language: str, limit: int):
        self.calls.append(query)
        if '"Max. Nozzle Temperature"' in query:
            return [OFFICIAL]
        return INITIAL


def test_research_service_runs_followup_and_promotes_official_source(monkeypatch):
    from app import research as research_module

    monkeypatch.setattr(research_module.settings, "research_max_queries", 2)
    monkeypatch.setattr(research_module.settings, "research_max_followup_queries", 1)
    monkeypatch.setattr(research_module.settings, "research_max_sources", 5)
    monkeypatch.setattr(research_module.settings, "research_provider", "searxng")

    provider = FollowupProvider()
    bundle = asyncio.run(ResearchService(provider).research(request()))

    assert len(provider.calls) == 2
    assert len(bundle.followup_queries) == 1
    assert bundle.evidence_sufficient is True
    assert bundle.missing_aspects == []
    assert "creality.com" in bundle.sources[0].url
    assert "260" in bundle.sources[0].snippet
    assert "100" in bundle.sources[0].snippet


def test_unsupported_negative_manual_claim_is_neutralized():
    req = request()
    result = ConversationResult.model_validate(
        {
            "schema_version": "1.0",
            "language": "de",
            "entries": [
                {
                    "type": "message",
                    "speaker_id": "p1",
                    "text": "Also keine explizite Maximaltemperatur im Handbuch.",
                    "typing_indicator": False,
                    "attachment": {"type": "none"},
                },
                {
                    "type": "message",
                    "speaker_id": "p2",
                    "text": "Dann suchen wir gezielter nach der Herstellerangabe.",
                    "typing_indicator": False,
                    "attachment": {"type": "none"},
                },
                {
                    "type": "message",
                    "speaker_id": "p1",
                    "text": "Die vorhandenen Snippets nennen nur eine Mindesttemperatur.",
                    "typing_indicator": False,
                    "attachment": {"type": "none"},
                },
                {
                    "type": "message",
                    "speaker_id": "p2",
                    "text": "Wir bleiben bei den belegten Angaben.",
                    "typing_indicator": False,
                    "attachment": {"type": "none"},
                },
            ],
        }
    )
    evidence = "[S1] Manual\nSnippet: Die minimale Düsentemperatur beträgt 180°C."

    adjustments = normalize_content_features(req, result, evidence)

    assert adjustments.unsupported_absence_claims_neutralized == 1
    assert "keine explizite Maximaltemperatur im Handbuch" not in result.entries[0].text
    assert "Evidenz" in result.entries[0].text or "Suchergebnisse" in result.entries[0].text


def test_secondary_exact_model_evidence_is_exposed_but_does_not_satisfy_official_request():
    source = ResearchSource(
        id="S1",
        title="Ender 3 V2 – Alles, was Sie wissen müssen - Obico",
        url="https://www.obico.io/de-DE/blog/ender-3-v2/",
        snippet="Ender 3 V2 technical overview.",
        engine="mock",
        content_excerpt=(
            "Temperaturbereich: Das Heizbett erreicht Temperaturen bis zu 100 °C. "
            "Das Hotend erreicht Temperaturen bis zu 255 °C."
        ),
        fetched=True,
        fetch_status="fetched",
    )

    coverage = assess_evidence_coverage(request(), [source], brand_hint="Creality")

    assert coverage.sufficient is False
    assert "maximale Düsentemperatur" in coverage.missing_aspects
    assert "maximale Heizbetttemperatur" in coverage.missing_aspects
    assert "maximale Düsentemperatur" in coverage.secondary_supported_aspects
    assert "maximale Heizbetttemperatur" in coverage.secondary_supported_aspects
