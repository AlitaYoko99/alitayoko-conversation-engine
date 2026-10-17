from app.content_normalizer import normalize_content_features
from app.models import ConversationResult, GenerationRequest


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


def result_with(*texts: str):
    return ConversationResult.model_validate(
        {
            "schema_version": "1.0",
            "language": "de",
            "entries": [
                {
                    "type": "message",
                    "speaker_id": "p1" if i % 2 == 0 else "p2",
                    "text": text,
                    "typing_indicator": False,
                    "attachment": {"type": "none"},
                }
                for i, text in enumerate(texts)
            ],
        }
    )


def test_invented_support_api_is_neutralized_when_not_in_evidence():
    result = result_with(
        "Wir rufen die Creality-Support-API auf.",
        "Dann sehen wir weiter.",
        "Wir bleiben bei belegten Daten.",
        "Gut.",
    )
    evidence = "[S1] Creality documentation\nFetched excerpt: Ender-3 V2 manual."

    adjustments = normalize_content_features(request(), result, evidence)

    assert adjustments.unsupported_source_mechanisms_neutralized == 1
    assert "API" not in result.entries[0].text
    assert "offizielle" in result.entries[0].text.casefold()


def test_api_is_kept_when_evidence_actually_mentions_one():
    result = result_with(
        "Wir rufen die Creality-Support-API auf.",
        "Dann sehen wir weiter.",
        "Wir bleiben bei belegten Daten.",
        "Gut.",
    )
    evidence = "[S1] Creality Support API documentation endpoint"

    adjustments = normalize_content_features(request(), result, evidence)

    assert adjustments.unsupported_source_mechanisms_neutralized == 0
    assert "API" in result.entries[0].text


def test_dangling_value_reference_is_rewritten_after_values_are_absent():
    result = result_with(
        "Die offizielle Angabe fehlt noch.",
        "Wir prüfen weiter.",
        "Dann können wir mit diesen Werten starten.",
        "Gut.",
    )

    adjustments = normalize_content_features(request(), result, "[S1] no numeric value")

    assert adjustments.dangling_value_references_rewritten == 1
    assert "diesen Werten" not in result.entries[2].text
    assert "belastbare Werte" in result.entries[2].text


def test_value_reference_is_kept_when_prior_numeric_value_exists():
    result = result_with(
        "Eine Sekundärquelle nennt 255 °C.",
        "Das ist noch nicht offiziell bestätigt.",
        "Dann können wir mit diesen Werten weiter vergleichen.",
        "Gut.",
    )

    adjustments = normalize_content_features(
        request(), result, "[S1] exact secondary source: Hotend bis zu 255 °C"
    )

    assert adjustments.dangling_value_references_rewritten == 0
    assert "diesen Werten" in result.entries[2].text
