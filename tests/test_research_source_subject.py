from app.models import GenerationRequest, ResearchSource
from app.research import (
    _rank_sources,
    _source_identity_basis,
    _source_identity_match,
    assess_evidence_coverage,
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


def sources_from_real_run():
    return [
        ResearchSource(
            id="S1",
            title="Creality Ender-3 V2 3D-Drucker - Bedienungsanleitung - Amazon S3",
            url="http://s3.amazonaws.com/example/Ender-3_V2_Manual_DE_V1.0.pdf",
            snippet="Eingestellte Heizbett-Temperatur. Aktuelle Heizbett-Temperatur.",
            engine="google cse",
        ),
        ResearchSource(
            id="S2",
            title="Vorheizen - Creality Ender-3 V2 Bedienungsanleitung [Seite 27]",
            url="https://www.manualslib.de/manual/728910/Creality-Ender-3-V2.html?page=27",
            snippet="Düsentemperatur: Damit der Extruder arbeiten kann ...",
            engine="google cse",
        ),
        ResearchSource(
            id="S3",
            title="Der Ender 3 V3 – ein Wolf im Schafspelz oder doch nur ein Lamm",
            url="https://drucktipps3d.de/der-ender-3-v3-test/",
            snippet=(
                "Düsentemperatur ≤300°C. Heizbetttemperatur ≤110°C. "
                "Als Beispiel, der Ender 3 V2 kommt mit 900gr Gewicht ..."
            ),
            engine="google cse",
        ),
        ResearchSource(
            id="S4",
            title="Ender 3 V2 – Alles, was Sie wissen müssen - Obico",
            url="https://www.obico.io/de-DE/blog/ender-3-v2/",
            snippet="Carborundum-Glasplattform und weitere Informationen zum Ender 3 V2.",
            engine="google cse",
        ),
        ResearchSource(
            id="S5",
            title="Ender 3 kaufen zum TOP-Preis – 3D-Drucker von Creality 3D im Shop",
            url="https://www.3d-grenzenlos.de/shop/ender-3-kaufen-creality-3d/",
            snippet=(
                "Die maximale Düsentemperatur liegt bei 255 Grad Celsius, das Heizbett "
                "erreicht maximal 110 Grad Celsius. Creality Ender 3 V2. Details."
            ),
            engine="google cse",
        ),
    ]


def test_snippet_mentions_do_not_make_source_exact():
    req = request()
    sources = sources_from_real_run()

    assert _source_identity_match(req, sources[0]) == "exact"
    assert _source_identity_basis(req, sources[0]) == "title_url"

    assert _source_identity_match(req, sources[2]) == "mismatch"
    assert _source_identity_basis(req, sources[2]) == "family"

    assert _source_identity_match(req, sources[3]) == "exact"
    assert _source_identity_basis(req, sources[3]) == "title_url"

    assert _source_identity_match(req, sources[4]) == "related"
    assert _source_identity_basis(req, sources[4]) == "title_url"


def test_wrong_model_and_generic_shop_values_do_not_satisfy_coverage():
    req = request()
    sources = sources_from_real_run()
    coverage = assess_evidence_coverage(req, sources, brand_hint="Creality")

    assert coverage.sufficient is False
    assert "maximale Düsentemperatur" in coverage.missing_aspects
    assert "maximale Heizbetttemperatur" in coverage.missing_aspects


def test_dedicated_exact_sources_rank_above_incidental_mentions():
    req = request()
    ranked = _rank_sources(req, sources_from_real_run(), brand_hint="Creality")

    exact_positions = [
        index for index, source in enumerate(ranked)
        if _source_identity_match(req, source) == "exact"
    ]
    mismatch_positions = [
        index for index, source in enumerate(ranked)
        if _source_identity_match(req, source) == "mismatch"
    ]

    assert exact_positions
    assert mismatch_positions
    assert min(exact_positions) < min(mismatch_positions)


def test_official_comparison_snippet_can_be_exact_contextual_evidence():
    req = request()
    source = ResearchSource(
        id="Sx",
        title="Ender Series 3D Printers Parameter Comparison | Creality Official",
        url="https://store.creality.com/blogs/all/ender-series-3d-printers-parameter-comparison",
        snippet=(
            "Ender-3 V2 Max. Heatbed Temperature 100℃. "
            "Max. Nozzle Temperature 260℃."
        ),
        engine="mock",
    )

    assert _source_identity_match(req, source, brand_hint="Creality") == "exact"
    assert _source_identity_basis(req, source, brand_hint="Creality") == "snippet_context"
