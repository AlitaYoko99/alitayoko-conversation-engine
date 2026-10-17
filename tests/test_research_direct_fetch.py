import asyncio

import pytest

from app.models import GenerationRequest, ResearchSource
from app.research import (
    FetchedDocument,
    ResearchService,
    SafeSourceFetcher,
    SourceFetchBlocked,
    _VisibleHTMLParser,
    _relevant_excerpt,
    assess_evidence_coverage,
    plan_followup_queries,
)


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


def test_html_extractor_ignores_active_content():
    parser = _VisibleHTMLParser()
    parser.feed(
        "<html><body><h1>Ender-3 V2</h1><script>ignore me 999°C</script>"
        "<p>Max. Nozzle Temperature 260℃</p></body></html>"
    )
    text = parser.text()
    assert "Ender-3 V2" in text
    assert "260℃" in text
    assert "999°C" not in text


def test_fetcher_blocks_local_hosts_before_request():
    fetcher = SafeSourceFetcher()
    with pytest.raises(SourceFetchBlocked):
        asyncio.run(fetcher._validate_public_url("http://localhost/private"))


def test_relevant_excerpt_keeps_requested_values():
    text = (
        "Navigation and unrelated introduction. " * 50
        + "Ender-3 V2 specifications. Max. Heatbed Temperature 100℃. "
        + "Max. Nozzle Temperature 260℃. "
        + "Unrelated footer. " * 50
    )
    excerpt = _relevant_excerpt(request(), text, 1200)
    assert "100℃" in excerpt
    assert "260℃" in excerpt
    assert len(excerpt) <= 1200


def test_fetched_excerpt_can_satisfy_official_coverage():
    source = ResearchSource(
        id="S1",
        title="Creality Ender-3 V2 Specifications",
        url="https://store.creality.com/ender-3-v2-specifications",
        snippet="Official Ender-3 V2 technical information.",
        engine="mock",
        content_excerpt=(
            "Ender-3 V2. Max. Heatbed Temperature 100℃. "
            "Max. Nozzle Temperature 260℃."
        ),
        fetched=True,
        fetch_status="fetched",
    )
    coverage = assess_evidence_coverage(request(), [source], brand_hint="Creality")
    assert coverage.sufficient is True
    assert coverage.missing_aspects == []


def test_followup_uses_probable_manufacturer_domain_when_not_yet_found():
    sources = [
        ResearchSource(
            id="S1",
            title="Creality Ender-3 V2 User Manual",
            url="https://example-manuals.test/ender-3-v2",
            snippet="Creality Ender-3 V2 manual.",
            engine="mock",
        ),
        ResearchSource(
            id="S2",
            title="Creality Ender-3 V2 review",
            url="https://example-review.test/ender-3-v2",
            snippet="Creality Ender-3 V2 review.",
            engine="mock",
        ),
    ]
    query = plan_followup_queries(
        request(), sources, ["maximale Düsentemperatur", "maximale Heizbetttemperatur"], 1
    )[0]
    assert "site:creality.com" in query


class FetchProvider:
    async def search(self, query: str, *, language: str, limit: int):
        if "site:creality.com" in query:
            return [
                ResearchSource(
                    id="Sx",
                    title="Creality Ender-3 V2 Specifications",
                    url="https://store.creality.com/ender-3-v2-specifications",
                    snippet="Official specifications for Ender-3 V2.",
                    engine="mock",
                )
            ]
        return [
            ResearchSource(
                id="S1",
                title="Creality Ender-3 V2 User Manual",
                url="https://manual.example.test/ender-3-v2",
                snippet="Ender-3 V2 user manual without temperatures in the snippet.",
                engine="mock",
            ),
            ResearchSource(
                id="S2",
                title="Creality Ender-3 V2 Overview",
                url="https://review.example.test/ender-3-v2",
                snippet="Creality Ender-3 V2 overview.",
                engine="mock",
            ),
        ]


class Fetcher:
    async def fetch(self, url: str):
        if "store.creality.com" in url:
            return FetchedDocument(
                url=url,
                content_type="text/html",
                text=(
                    "Ender-3 V2 technical specifications. "
                    "Max. Heatbed Temperature 100℃. Max. Nozzle Temperature 260℃."
                ),
            )
        return FetchedDocument(
            url=url,
            content_type="text/html",
            text="Ender-3 V2 manual assembly instructions.",
        )


def test_research_service_fetches_source_content_before_final_coverage(monkeypatch):
    from app import research as research_module

    monkeypatch.setattr(research_module.settings, "research_max_queries", 2)
    monkeypatch.setattr(research_module.settings, "research_max_followup_queries", 1)
    monkeypatch.setattr(research_module.settings, "research_max_sources", 5)
    monkeypatch.setattr(research_module.settings, "research_fetch_enabled", True)
    monkeypatch.setattr(research_module.settings, "research_fetch_max_sources", 3)
    monkeypatch.setattr(research_module.settings, "research_provider", "searxng")

    bundle = asyncio.run(ResearchService(FetchProvider(), Fetcher()).research(request()))

    assert any("site:creality.com" in query for query in bundle.followup_queries)
    assert bundle.fetch_count >= 1
    assert bundle.evidence_sufficient is True
    official = next(source for source in bundle.sources if "store.creality.com" in source.url)
    assert official.fetch_status == "fetched"
    assert "260℃" in official.content_excerpt
    assert "100℃" in official.content_excerpt
