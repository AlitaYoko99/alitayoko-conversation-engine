from app.models import GenerationRequest, ResearchSource
from app.research import (
    _rank_sources,
    _source_identity_match,
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


def real_world_like_sources():
    return [
        ResearchSource(
            id="S1",
            title="Ender-3 V3 Support: Firmware, Videos, Spezifikationen und Hilfe - Creality",
            url="https://www.creality.com/de/support/creality-ender-3-v3",
            snippet="Parameterspezifikationen für den Ender-3 V3.",
            engine="duckduckgo",
        ),
        ResearchSource(
            id="S2",
            title="Ender-5 Max Support: Firmware, Videos, Spezifikationen und Hilfe",
            url="https://www.creality.com/de/support/creality-ender-5-max",
            snippet="Heizbett-Temperatur ≤ 100 ℃.",
            engine="google cse",
        ),
        ResearchSource(
            id="S3",
            title="Ender-3 V2 Neo 3D-Drucker - Creality",
            url="https://www.creality.com/de/products/ender-3-v2-neo-3d-drucker",
            snippet="Technische Daten des Ender-3 V2 Neo.",
            engine="duckduckgo",
        ),
        ResearchSource(
            id="S4",
            title="Ender-3 Neo | FFF 3D-Drucker - Creality Cloud",
            url="https://www.crealitycloud.com/de/product/fff-3d-printer/Ender-3-Neo-example",
            snippet="Ender-3 Neo Maximale Düsentemperatur 260℃.",
            engine="google cse",
        ),
        ResearchSource(
            id="S5",
            title="Bedienungsanleitung Ender Creality Ender-3 V2 3D-Drucker",
            url="https://manuall.de/ender-creality-ender-3-v2-3d-drucker/",
            snippet="Bedienungsanleitung für Ender Creality Ender-3 V2 3D-Drucker.",
            engine="duckduckgo",
        ),
    ]


def test_related_models_do_not_count_as_exact_identity():
    req = request()
    sources = real_world_like_sources()

    assert _source_identity_match(req, sources[0]) == "mismatch"
    assert _source_identity_match(req, sources[1]) == "mismatch"
    assert _source_identity_match(req, sources[2]) == "related"
    assert _source_identity_match(req, sources[3]) == "mismatch"
    assert _source_identity_match(req, sources[4]) == "exact"

    coverage = assess_evidence_coverage(req, sources, brand_hint="Creality")
    assert coverage.sufficient is False
    assert "maximale Düsentemperatur" in coverage.missing_aspects
    assert "maximale Heizbetttemperatur" in coverage.missing_aspects


def test_exact_identity_is_ranked_above_wrong_official_models():
    req = request()
    sources = real_world_like_sources()
    ranked = _rank_sources(req, sources, brand_hint="Creality")

    assert "ender-3-v2" in ranked[0].url.casefold()


def test_followup_uses_exact_identity_and_manufacturer_domain():
    req = request()
    queries = plan_followup_queries(
        req,
        real_world_like_sources(),
        ["maximale Düsentemperatur"],
        1,
    )

    assert len(queries) == 1
    assert '"Ender 3 V2"' in queries[0]
    assert "Creality" in queries[0]
    assert "site:creality.com" in queries[0]


def test_conflicting_exact_primary_sources_are_reported():
    req = request()
    sources = [
        ResearchSource(
            id="S1",
            title="Comprehensive Guide to Ender-3 V2 3D Printer Operation | Creality",
            url="https://store.creality.com/blogs/all/ender-3-v2-user-guide",
            snippet=(
                "Ender-3 V2 highest nozzle temperature 250°C. "
                "Maximum hot bed temperature 110°C."
            ),
            engine="mock",
        ),
        ResearchSource(
            id="S2",
            title="Ender-3 V2 vs Ender-3 Neo comparison | Creality",
            url="https://store.creality.com/blogs/all/ender-3-v2-3-neo-comparison-guide",
            snippet=(
                "Ender-3 V2 Max. Heatbed Temperature 100℃. "
                "Max. Nozzle Temperature 260℃."
            ),
            engine="mock",
        ),
    ]

    coverage = assess_evidence_coverage(req, sources, brand_hint="Creality")

    assert coverage.sufficient is False
    assert "maximale Düsentemperatur" in coverage.conflicting_aspects
    assert "maximale Heizbetttemperatur" in coverage.conflicting_aspects
