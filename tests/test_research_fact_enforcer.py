from app.models import ConversationEntry, ConversationResult, ResearchFact
from app.research_fact_enforcer import enforce_verified_research_facts
from tests.test_research_evidence_consumption import request


def test_enforcer_replaces_stale_uncertainty_with_verified_values_without_changing_structure():
    req = request()
    result = ConversationResult.model_validate(
        {
            "schema_version": "1.0",
            "language": "de",
            "entries": [
                {"type": "message", "speaker_id": "p1", "text": "Die Werte müssen wir noch prüfen.", "typing_indicator": False, "attachment": {"type": "none"}},
                {"type": "message", "speaker_id": "p2", "text": "Dann suchen wir eine passende Quelle.", "typing_indicator": False, "attachment": {"type": "none"}},
                {"type": "message", "speaker_id": "p1", "text": "Wir bleiben bei belegten Daten.", "typing_indicator": False, "attachment": {"type": "none"}},
                {"type": "message", "speaker_id": "p2", "text": "Einverstanden.", "typing_indicator": False, "attachment": {"type": "none"}},
            ],
        }
    )
    facts = [
        ResearchFact(aspect="maximale Düsentemperatur", value="260 °C", confidence="primary", source_ids=["S1"]),
        ResearchFact(aspect="maximale Heizbetttemperatur", value="100 °C", confidence="primary", source_ids=["S1"]),
    ]

    speakers_before = [entry.speaker_id for entry in result.entries]
    injected, rewritten, recommendation_rewrites = enforce_verified_research_facts(req, result, facts)
    text = " ".join(entry.text for entry in result.entries)

    assert injected == 2
    assert rewritten == 0
    assert "260 °C" in text
    assert "100 °C" in text
    assert [entry.speaker_id for entry in result.entries] == speakers_before
    assert len(result.entries) == 4


def test_enforcer_prefers_stale_unknown_and_known_model_filler_from_real_research_output():
    req = request()
    result = ConversationResult.model_validate(
        {
            "schema_version": "1.0",
            "language": "de",
            "entries": [
                {"type": "message", "speaker_id": "p1", "text": "Das genaue technische Detail ist noch nicht geklärt. Prüfen wir zuerst die passende Dokumentation.", "typing_indicator": False, "attachment": {"type": "none"}},
                {"type": "message", "speaker_id": "p2", "text": "Darauf sollten wir uns ohne die genaue Modellangabe nicht festlegen. Schauen wir zuerst in die passende Dokumentation.", "typing_indicator": False, "attachment": {"type": "none"}},
                {"type": "message", "speaker_id": "p1", "text": "Genau, beide Werte sind im Hersteller-Blog vermerkt, keine Schätzungen.", "typing_indicator": False, "attachment": {"type": "none"}},
                {"type": "message", "speaker_id": "p2", "text": "Ich halte mich an die belegten Zahlen.", "typing_indicator": False, "attachment": {"type": "none"}},
            ],
        }
    )
    facts = [
        ResearchFact(aspect="maximale Düsentemperatur", value="260 °C", confidence="primary", source_ids=["S1"]),
        ResearchFact(aspect="maximale Heizbetttemperatur", value="100 °C", confidence="primary", source_ids=["S1"]),
    ]

    injected, rewritten, recommendation_rewrites = enforce_verified_research_facts(
        req,
        result,
        facts,
        target_identity="Ender 3 V2",
    )

    assert injected == 2
    assert rewritten == 0
    assert result.entries[0].text == "Die maximale Düsentemperatur ist mit 260 °C dokumentiert."
    assert result.entries[1].text == "Die maximale Heizbetttemperatur ist mit 100 °C dokumentiert."
    text = " ".join(entry.text for entry in result.entries).casefold()
    assert "noch nicht geklärt" not in text
    assert "ohne die genaue modellangabe" not in text


def test_missing_recommendation_neutralizes_support_to_recommendation_inference():
    req = request().model_copy(update={"topic": "WordPress 6.8 unterstützt PHP 8.4. Welche PHP-Version wird empfohlen?"})
    result = ConversationResult(
        schema_version="1.0",
        language="de",
        entries=[
            ConversationEntry(type="message", speaker_id="p1", text="PHP 8.4 ist vollständig unterstützt.", typing_indicator=False, attachment={"type":"none"}),
            ConversationEntry(type="message", speaker_id="p2", text="Dann sollten wir bei 8.4 bleiben.", typing_indicator=False, attachment={"type":"none"}),
            ConversationEntry(type="message", speaker_id="p1", text="Welche PHP-Version wird offiziell empfohlen?", typing_indicator=False, attachment={"type":"none"}),
        ],
    )
    facts = [
        ResearchFact(
            aspect="Unterstützungsstatus für PHP 8.4",
            value="vollständig unterstützt",
            confidence="primary",
            source_ids=["S1"],
        )
    ]

    injected, rewritten, recommendation_rewrites = enforce_verified_research_facts(
        req,
        result,
        facts,
        missing_aspects=["empfohlene PHP-Version"],
    )

    assert recommendation_rewrites == 1
    assert result.entries[0].text == "PHP 8.4 ist vollständig unterstützt."
    assert result.entries[1].text == "Welche PHP-Version empfohlen wird, ist mit den vorhandenen Quellen noch nicht belegt."
    assert result.entries[2].text == "Welche PHP-Version wird offiziell empfohlen?"


def test_recommendation_guard_does_not_fire_when_recommendation_is_verified():
    req = request().model_copy(update={"topic": "Welche PHP-Version wird für WordPress empfohlen?"})
    result = ConversationResult(
        schema_version="1.0",
        language="de",
        entries=[
            ConversationEntry(type="message", speaker_id="p1", text="Wir sollten PHP 8.3 verwenden.", typing_indicator=False, attachment={"type":"none"}),
        ],
    )
    facts = [
        ResearchFact(
            aspect="empfohlene PHP-Version",
            value="PHP 8.3",
            confidence="primary",
            source_ids=["S1"],
        )
    ]

    _, _, recommendation_rewrites = enforce_verified_research_facts(
        req, result, facts, missing_aspects=[]
    )
    assert recommendation_rewrites == 0
    assert result.entries[0].text == "Wir sollten PHP 8.3 verwenden."


def test_qualitative_relation_injection_uses_validator_visible_verified_value():
    req = request().model_copy(update={"topic": "Wie beeinflusst geringerer Luftdruck den Siedepunkt von Wasser?"})
    result = ConversationResult(
        schema_version="1.0",
        language="de",
        entries=[
            ConversationEntry(type="message", speaker_id="p1", text="Das müssen wir noch prüfen.", typing_indicator=False, attachment={"type":"none"}),
            ConversationEntry(type="message", speaker_id="p2", text="Ich würde dazu erst eine Quelle suchen.", typing_indicator=False, attachment={"type":"none"}),
            ConversationEntry(type="message", speaker_id="p1", text="Der Zusammenhang ist noch offen.", typing_indicator=False, attachment={"type":"none"}),
            ConversationEntry(type="message", speaker_id="p2", text="Dann bleiben wir vorsichtig.", typing_indicator=False, attachment={"type":"none"}),
        ],
    )
    facts = [
        ResearchFact(
            aspect="Einfluss geringeren Luftdrucks auf den Siedepunkt",
            value="sinkt bei geringerem Luftdruck",
            confidence="secondary",
            source_ids=["S1"],
        )
    ]

    injected, _, _ = enforce_verified_research_facts(req, result, facts)
    text = " ".join(entry.text for entry in result.entries)

    assert injected == 1
    assert "sinkt bei geringerem luftdruck" in text.casefold()


def test_missing_recommendation_neutralizes_conservative_version_choice():
    req = request().model_copy(update={"topic": "WordPress 6.8 unterstützt PHP 8.4. Welche PHP-Version wird empfohlen?"})
    result = ConversationResult(
        schema_version="1.0",
        language="de",
        entries=[
            ConversationEntry(type="message", speaker_id="p1", text="PHP 8.4 ist vollständig unterstützt.", typing_indicator=False, attachment={"type":"none"}),
            ConversationEntry(type="message", speaker_id="p2", text="Das wäre konservativ, solange 8.3 noch unterstützt wird.", typing_indicator=False, attachment={"type":"none"}),
            ConversationEntry(type="message", speaker_id="p1", text="Welche PHP-Version wird empfohlen?", typing_indicator=False, attachment={"type":"none"}),
        ],
    )
    facts = [
        ResearchFact(
            aspect="Unterstützungsstatus für PHP 8.4",
            value="vollständig unterstützt",
            confidence="primary",
            source_ids=["S1"],
        )
    ]

    _, _, recommendation_rewrites = enforce_verified_research_facts(
        req,
        result,
        facts,
        missing_aspects=["empfohlene PHP-Version"],
    )

    assert recommendation_rewrites == 1
    assert result.entries[1].text == "Welche PHP-Version empfohlen wird, ist mit den vorhandenen Quellen noch nicht belegt."


def test_complete_research_final_pass_removes_generic_insufficient_evidence_claims():
    from app.research_fact_enforcer import neutralize_complete_research_uncertainty

    req = request().model_copy(update={"topic": "Wie beeinflusst geringerer Luftdruck den Siedepunkt von Wasser?"})
    result = ConversationResult(
        schema_version="1.0",
        language="de",
        entries=[
            ConversationEntry(type="message", speaker_id="p1", text="Der Siedepunkt sinkt bei geringerem Luftdruck.", typing_indicator=False, attachment={"type":"none"}),
            ConversationEntry(type="message", speaker_id="p2", text="Für eine belastbare Antwort fehlt uns noch ausreichende Evidenz.", typing_indicator=False, attachment={"type":"none"}),
            ConversationEntry(type="message", speaker_id="p1", text="Diese Aussage ist mit den vorhandenen Quellen noch nicht ausreichend belegt.", typing_indicator=False, attachment={"type":"none"}),
        ],
    )

    rewritten = neutralize_complete_research_uncertainty(
        req,
        result,
        evidence_sufficient=True,
        missing_aspects=[],
        conflicting_aspects=[],
    )

    text = " ".join(entry.text for entry in result.entries).casefold()
    assert rewritten == 2
    assert "fehlt uns noch ausreichende evidenz" not in text
    assert "noch nicht ausreichend belegt" not in text
    assert "sinkt bei geringerem luftdruck" in text


def test_complete_research_final_pass_preserves_uncertainty_when_aspect_is_missing():
    from app.research_fact_enforcer import neutralize_complete_research_uncertainty

    req = request().model_copy(update={"topic": "Welche PHP-Version wird für WordPress 6.8 empfohlen?"})
    result = ConversationResult(
        schema_version="1.0",
        language="de",
        entries=[
            ConversationEntry(type="message", speaker_id="p1", text="Welche PHP-Version empfohlen wird, ist mit den vorhandenen Quellen noch nicht belegt.", typing_indicator=False, attachment={"type":"none"}),
        ],
    )

    rewritten = neutralize_complete_research_uncertainty(
        req,
        result,
        evidence_sufficient=False,
        missing_aspects=["empfohlene PHP-Version"],
        conflicting_aspects=[],
    )

    assert rewritten == 0
    assert "noch nicht belegt" in result.entries[0].text.casefold()


def test_missing_recommendation_neutralizes_stability_safe_version_choice():
    req = request().model_copy(update={"topic": "WordPress 6.8 unterstützt PHP 8.4. Welche PHP-Version wird empfohlen?"})
    result = ConversationResult(
        schema_version="1.0",
        language="de",
        entries=[
            ConversationEntry(type="message", speaker_id="p1", text="PHP 8.4 ist vollständig unterstützt.", typing_indicator=False, attachment={"type":"none"}),
            ConversationEntry(type="message", speaker_id="p2", text="Genau. Für Stabilität wäre PHP 8.3 immer noch sicher, wenn es unterstützt wird.", typing_indicator=False, attachment={"type":"none"}),
        ],
    )
    facts = [ResearchFact(aspect="Unterstützungsstatus für PHP 8.4", value="vollständig unterstützt", confidence="primary", source_ids=["S1"])]

    _, _, recommendation_rewrites = enforce_verified_research_facts(
        req, result, facts, missing_aspects=["empfohlene PHP-Version"]
    )

    assert recommendation_rewrites == 1
    assert result.entries[1].text == "Welche PHP-Version empfohlen wird, ist mit den vorhandenen Quellen noch nicht belegt."


def test_missing_recommendation_neutralizes_exclusive_confirmed_option_claim():
    req = request().model_copy(update={"topic": "WordPress 6.8 unterstützt PHP 8.4. Welche PHP-Version wird empfohlen?"})
    result = ConversationResult(
        schema_version="1.0",
        language="de",
        entries=[
            ConversationEntry(type="message", speaker_id="p1", text="PHP 8.4 ist vollständig unterstützt.", typing_indicator=False, attachment={"type":"none"}),
            ConversationEntry(type="message", speaker_id="p2", text="Falls 8.3 nicht explizit erwähnt ist, bleibt 8.4 die einzige bestätigte Option.", typing_indicator=False, attachment={"type":"none"}),
        ],
    )
    facts = [ResearchFact(aspect="Unterstützungsstatus für PHP 8.4", value="vollständig unterstützt", confidence="primary", source_ids=["S1"])]

    _, _, recommendation_rewrites = enforce_verified_research_facts(
        req, result, facts, missing_aspects=["empfohlene PHP-Version"]
    )

    assert recommendation_rewrites == 1
    assert result.entries[1].text == "Welche PHP-Version empfohlen wird, ist mit den vorhandenen Quellen noch nicht belegt."
