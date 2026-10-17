import json
from pathlib import Path

from app.content_normalizer import normalize_content_features
from app.models import ConversationResult, GenerationRequest, ResearchFact, ResearchSource
from app.research import (
    ResearchBundle,
    _is_primary_like,
    _target_identity,
    assess_evidence_coverage,
    extract_verified_facts,
)
from app.research_fact_enforcer import enforce_verified_research_facts


ROOT = Path(__file__).resolve().parents[1]


def load_example(name: str) -> GenerationRequest:
    return GenerationRequest.model_validate(
        json.loads((ROOT / "examples" / "research" / name).read_text(encoding="utf-8"))
    )


def test_software_primary_evidence_beats_outdated_secondary_status():
    req = load_example("02-software.de.json")
    official = ResearchSource(
        id="S1",
        title="PHP support clarification, spring 2026 edition - Make WordPress",
        url="https://make.wordpress.org/core/2026/05/22/php-support-clarification-2026/",
        snippet=(
            "WordPress 6.8 and later are now documented as fully supporting PHP 8.4. "
            "WordPress recommends PHP 8.3 or greater."
        ),
        engine="mock",
    )
    outdated = ResearchSource(
        id="S2",
        title="Older third-party guide",
        url="https://example.com/old-wordpress-guide",
        snippet="WordPress 6.8 supports PHP 8.4 with beta support. Recommended PHP 8.3.",
        engine="mock",
    )

    assert _target_identity(req) == "WordPress 6.8"
    assert _is_primary_like(official, "", req) is True
    assert _is_primary_like(outdated, "", req) is False

    coverage = assess_evidence_coverage(req, [official, outdated])
    facts = {fact.aspect: fact for fact in extract_verified_facts(req, [official, outdated])}

    assert coverage.sufficient is True
    assert facts["Unterstützungsstatus für PHP 8.4"].value == "vollständig unterstützt"
    assert facts["Unterstützungsstatus für PHP 8.4"].confidence == "primary"
    assert facts["empfohlene PHP-Version"].value == "PHP 8.3"
    assert facts["empfohlene PHP-Version"].confidence == "primary"


def test_newer_dated_primary_software_source_supersedes_older_primary_status():
    req = load_example("02-software.de.json")
    old_official = ResearchSource(
        id="S1",
        title="PHP support clarification - Make WordPress",
        url="https://make.wordpress.org/core/2025/04/09/php-8-support-clarification/",
        snippet="WordPress 6.8 supports PHP 8.4 with beta support. WordPress recommends PHP 8.3 or greater.",
        engine="mock",
    )
    new_official = ResearchSource(
        id="S2",
        title="PHP support clarification, spring 2026 edition - Make WordPress",
        url="https://make.wordpress.org/core/2026/05/22/php-support-clarification-2026/",
        snippet="WordPress 6.8 and later are now documented as fully supporting PHP 8.4. WordPress recommends PHP 8.3 or greater.",
        engine="mock",
    )

    coverage = assess_evidence_coverage(req, [old_official, new_official])
    facts = {fact.aspect: fact for fact in extract_verified_facts(req, [old_official, new_official])}

    assert coverage.sufficient is True
    assert coverage.conflicting_aspects == []
    assert facts["Unterstützungsstatus für PHP 8.4"].value == "vollständig unterstützt"
    assert facts["Unterstützungsstatus für PHP 8.4"].source_ids == ["S2"]


def test_consumer_hardware_extracts_generic_display_and_storage_facts():
    req = load_example("03-consumer-hardware.de.json")
    source = ResearchSource(
        id="S1",
        title="Nintendo Switch – OLED Model",
        url="https://www.nintendo.com/us/gaming-systems/switch/oled-model/",
        snippet="The system features a 7-inch OLED screen and 64 GB of internal storage.",
        engine="mock",
    )

    coverage = assess_evidence_coverage(req, [source])
    facts = {fact.aspect: fact for fact in extract_verified_facts(req, [source])}

    assert coverage.sufficient is True
    assert facts["Displaygröße"].value == "7 inch"
    assert facts["interner Speicher"].value == "64 GB"
    assert facts["Displaygröße"].confidence == "primary"
    assert facts["interner Speicher"].confidence == "primary"


def test_science_evidence_requires_both_numeric_fact_and_relation():
    req = load_example("04-science.de.json")
    source = ResearchSource(
        id="S1",
        title="Water boiling point and pressure",
        url="https://example.edu/water-boiling-point",
        snippet=(
            "At normal atmospheric pressure water boils at 100 °C. "
            "At lower atmospheric pressure the boiling point decreases."
        ),
        engine="mock",
    )

    coverage = assess_evidence_coverage(req, [source])
    facts = {fact.aspect: fact for fact in extract_verified_facts(req, [source])}

    assert coverage.sufficient is True
    assert facts["Siedepunkt von Wasser"].value == "100 °C"
    assert facts["Einfluss geringeren Luftdrucks auf den Siedepunkt"].value == "sinkt bei geringerem Luftdruck"
    assert all(fact.confidence == "secondary" for fact in facts.values())


def test_matching_topic_without_concrete_values_is_not_sufficient():
    req = load_example("03-consumer-hardware.de.json")
    source = ResearchSource(
        id="S1",
        title="Nintendo Switch OLED overview",
        url="https://example.com/switch-oled",
        snippet="An overview of the OLED display and internal storage of the console.",
        engine="mock",
    )

    coverage = assess_evidence_coverage(req, [source])

    assert coverage.sufficient is False
    assert "Displaygröße" in coverage.missing_aspects
    assert "interner Speicher" in coverage.missing_aspects


def test_secondary_verified_fact_is_grounded_without_claiming_official_authority():
    req = load_example("04-science.de.json")
    source = ResearchSource(
        id="S1",
        title="Water boiling point and pressure",
        url="https://example.edu/water-boiling-point",
        snippet="At normal atmospheric pressure water boils at 100 °C.",
        engine="mock",
    )
    facts = extract_verified_facts(req, [source])
    result = ConversationResult.model_validate(
        {
            "schema_version": "1.0",
            "language": "de",
            "entries": [
                {"type": "message", "speaker_id": "p1", "text": "Das müssen wir noch prüfen.", "typing_indicator": False, "attachment": {"type": "none"}},
                {"type": "message", "speaker_id": "p2", "text": "Dann suchen wir eine Quelle.", "typing_indicator": False, "attachment": {"type": "none"}},
                {"type": "message", "speaker_id": "p1", "text": "Wir bleiben vorsichtig.", "typing_indicator": False, "attachment": {"type": "none"}},
                {"type": "message", "speaker_id": "p2", "text": "Einverstanden.", "typing_indicator": False, "attachment": {"type": "none"}},
                {"type": "message", "speaker_id": "p1", "text": "Noch ist der Wert offen.", "typing_indicator": False, "attachment": {"type": "none"}},
                {"type": "message", "speaker_id": "p2", "text": "Wir prüfen weiter.", "typing_indicator": False, "attachment": {"type": "none"}},
                {"type": "message", "speaker_id": "p1", "text": "Dann vergleichen wir die Angaben.", "typing_indicator": False, "attachment": {"type": "none"}},
                {"type": "message", "speaker_id": "p2", "text": "Gut.", "typing_indicator": False, "attachment": {"type": "none"}},
            ],
        }
    )

    injected, _, _ = enforce_verified_research_facts(req, result, facts)
    text = " ".join(entry.text for entry in result.entries)

    assert injected >= 1
    assert "100 °C" in text
    assert "geprüften Quellen" in text


def test_official_attribution_is_neutralized_when_no_primary_source_exists():
    req = load_example("03-consumer-hardware.de.json")
    result = ConversationResult.model_validate(
        {
            "schema_version": "1.0",
            "language": "de",
            "entries": [
                {"type": "message", "speaker_id": "p1", "text": "Das Display misst 7 Zoll, wie Nintendo selbst angibt.", "typing_indicator": False, "attachment": {"type": "none"}},
                {"type": "message", "speaker_id": "p2", "text": "Die 64 GB stehen laut offizieller Spezifikation fest.", "typing_indicator": False, "attachment": {"type": "none"}},
                {"type": "message", "speaker_id": "p1", "text": "Damit sind beide Werte klar.", "typing_indicator": False, "attachment": {"type": "none"}},
                {"type": "message", "speaker_id": "p2", "text": "Gut.", "typing_indicator": False, "attachment": {"type": "none"}},
                {"type": "message", "speaker_id": "p1", "text": "Dann vergleichen wir weiter.", "typing_indicator": False, "attachment": {"type": "none"}},
                {"type": "message", "speaker_id": "p2", "text": "Einverstanden.", "typing_indicator": False, "attachment": {"type": "none"}},
                {"type": "message", "speaker_id": "p1", "text": "Mehr brauchen wir hier nicht.", "typing_indicator": False, "attachment": {"type": "none"}},
                {"type": "message", "speaker_id": "p2", "text": "Passt.", "typing_indicator": False, "attachment": {"type": "none"}},
            ],
        }
    )

    adjustments = normalize_content_features(
        req,
        result,
        "[S1] Third-party review\nAuthority: secondary\nSearch snippet: 7 Zoll, 64 GB",
    )
    text = " ".join(entry.text for entry in result.entries)

    assert adjustments.source_claims_neutralized >= 2
    assert "Nintendo selbst" not in text
    assert "offizieller Spezifikation" not in text
    assert "geprüften Quellen" in text


def test_science_standard_pressure_value_ignores_other_pressure_conditions():
    req = load_example("04-science.de.json")
    sources = [
        ResearchSource(
            id="S1",
            title="Boiling point calculator",
            url="https://example.org/sea-level",
            snippet="At sea level, at 1 atmosphere or 101.3 kPa, water boils at 100 °C.",
            engine="mock",
        ),
        ResearchSource(
            id="S2",
            title="Pressure cooker",
            url="https://example.org/pressure-cooker",
            snippet="In a pressure cooker water can boil at 120 °C because pressure is higher.",
            engine="mock",
        ),
        ResearchSource(
            id="S3",
            title="Mountain boiling",
            url="https://example.org/mountain",
            snippet="At lower atmospheric pressure the boiling point decreases and can be below 100 °C.",
            engine="mock",
        ),
    ]
    coverage = assess_evidence_coverage(req, sources)
    facts = {fact.aspect: fact.value for fact in extract_verified_facts(req, sources)}
    assert coverage.sufficient is True
    assert coverage.conflicting_aspects == []
    assert facts["Siedepunkt von Wasser"] == "100 °C"
    assert facts["Einfluss geringeren Luftdrucks auf den Siedepunkt"] == "sinkt bei geringerem Luftdruck"


def test_german_display_value_is_normalized_without_hyphen():
    req = load_example("03-consumer-hardware.de.json")
    source = ResearchSource(
        id="S1",
        title="Nintendo Switch OLED",
        url="https://www.nintendo.com/de-de/hardware/switch-oled/",
        snippet="Nintendo Switch OLED verfügt über einen 7-Zoll-OLED-Bildschirm und 64 GB internen Speicher.",
        engine="mock",
    )
    facts = {fact.aspect: fact.value for fact in extract_verified_facts(req, [source])}
    assert facts["Displaygröße"] == "7 Zoll"


def test_generation_evidence_is_compact_and_excludes_raw_source_corpus():
    source = ResearchSource(
        id="S1",
        title="Official product page",
        url="https://example.com/product",
        snippet="RAW_SNIPPET_SHOULD_NOT_REACH_GENERATION",
        engine="mock",
        authority="primary",
        identity_match="exact",
        identity_basis="title_url",
        fetched=True,
        fetch_status="fetched",
        content_excerpt="RAW_FETCHED_EXCERPT_SHOULD_NOT_REACH_GENERATION",
    )
    bundle = ResearchBundle(
        provider="mock",
        queries=["example"],
        sources=[source],
        elapsed_seconds=0.1,
        evidence_sufficient=True,
        verified_facts=[
            ResearchFact(
                aspect="Displaygröße",
                value="7 Zoll",
                confidence="primary",
                source_ids=["S1"],
            )
        ],
        target_identity="Example Product",
    )

    compact = bundle.generation_evidence_text
    full = bundle.evidence_text

    assert "Displaygröße: 7 Zoll" in compact
    assert "authority=primary" in compact
    assert "RAW_SNIPPET_SHOULD_NOT_REACH_GENERATION" not in compact
    assert "RAW_FETCHED_EXCERPT_SHOULD_NOT_REACH_GENERATION" not in compact
    assert "RAW_FETCHED_EXCERPT_SHOULD_NOT_REACH_GENERATION" in full


def test_generation_evidence_preserves_missing_and_conflicting_aspects():
    bundle = ResearchBundle(
        provider="mock",
        queries=["example"],
        sources=[],
        elapsed_seconds=0.1,
        evidence_sufficient=False,
        missing_aspects=["maximale Temperatur"],
        conflicting_aspects=["Firmware-Version"],
        secondary_supported_aspects=["Speicherkapazität"],
    )

    compact = bundle.generation_evidence_text

    assert "VERIFIED FACTS\n- none" in compact
    assert "maximale Temperatur" in compact
    assert "Firmware-Version" in compact
    assert "Speicherkapazität" in compact


def test_pressure_boiling_relation_handles_natural_causal_word_order():
    req = load_example("04-science.de.json").model_copy(update={"topic": "Wie wirkt sich der Luftdruck auf den Siedepunkt von Wasser aus? Bei geringerem Luftdruck soll der Siedepunkt sinken."})
    source = ResearchSource(
        id="S1",
        title="Boiling point and pressure",
        url="https://example.org/boiling",
        snippet=(
            "In größeren Höhen ist der atmosphärische Druck niedriger, sodass Wasser "
            "bei niedrigeren Temperaturen siedet."
        ),
        engine="duckduckgo",
        authority="secondary",
    )

    facts = {fact.aspect: fact.value for fact in extract_verified_facts(req, [source])}
    assert facts["Einfluss geringeren Luftdrucks auf den Siedepunkt"] == "sinkt bei geringerem Luftdruck"


def test_pressure_boiling_relation_handles_pressure_then_lower_temperature():
    req = load_example("04-science.de.json").model_copy(update={"topic": "Wie wirkt sich geringerer Luftdruck auf den Siedepunkt von Wasser aus?"})
    source = ResearchSource(
        id="S1",
        title="Water boiling point",
        url="https://example.org/water",
        snippet=(
            "Wenn die Höhe zunimmt, sinkt der atmosphärische Druck, wodurch Wasser "
            "bei Temperaturen unter den standardmäßigen 100 °C siedet."
        ),
        engine="duckduckgo",
        authority="secondary",
    )

    facts = {fact.aspect: fact.value for fact in extract_verified_facts(req, [source])}
    assert facts["Einfluss geringeren Luftdrucks auf den Siedepunkt"] == "sinkt bei geringerem Luftdruck"
