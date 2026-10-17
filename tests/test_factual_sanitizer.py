from app.content_normalizer import normalize_content_features
from app.models import ConversationResult, GenerationRequest
from app.validator import validate_semantics


def make_request():
    return GenerationRequest.model_validate(
        {
            "language": "de",
            "topic": (
                "Lukas und Markus richten einen gebrauchten 3D-Drucker ein. "
                "Sie möchten zuerst die Bedienungsanleitung und technischen Daten prüfen."
            ),
            "entry_count": 5,
            "tone": "natural",
            "chaos": 10,
            "participants": [
                {"id": "p1", "name": "Lukas"},
                {"id": "p2", "name": "Markus"},
            ],
            "features": {
                "narration": False,
                "typing_indicator": True,
                "images": False,
                "links": True,
            },
            "limits": {
                "max_narrations": 0,
                "max_words_per_entry": 30,
                "max_images": 0,
                "max_links": 1,
            },
        }
    )


def make_result():
    return ConversationResult.model_validate(
        {
            "schema_version": "1.0",
            "language": "de",
            "entries": [
                {
                    "type": "message",
                    "speaker_id": "p1",
                    "text": "Das ist bestimmt ein Ender 3 V2.",
                    "typing_indicator": False,
                    "attachment": {"type": "none"},
                },
                {
                    "type": "message",
                    "speaker_id": "p2",
                    "text": "Dann schafft der Extruder 260 °C.",
                    "typing_indicator": False,
                    "attachment": {"type": "none"},
                },
                {
                    "type": "message",
                    "speaker_id": "p1",
                    "text": "Firmware-Version 1.1.0 sollte aktuell sein.",
                    "typing_indicator": True,
                    "attachment": {"type": "none"},
                },
                {
                    "type": "message",
                    "speaker_id": "p2",
                    "text": "Wir sollten lieber die Bedienungsanleitung prüfen.",
                    "typing_indicator": False,
                    "attachment": {
                        "type": "link",
                        "query": "Ender 3 V2 official manual",
                        "label": "Ender 3 V2 Manual",
                    },
                },
                {
                    "type": "message",
                    "speaker_id": "p1",
                    "text": "Danach ändern wir die Einstellungen.",
                    "typing_indicator": False,
                    "attachment": {"type": "none"},
                },
            ],
        }
    )


def test_unsupported_specifics_are_neutralized_before_validation():
    request = make_request()
    result = make_result()

    adjustments = normalize_content_features(request, result)
    violations = validate_semantics(request, result)

    assert adjustments.technical_specifics_neutralized == 3
    assert adjustments.links_sanitized == 1
    assert "UNSUPPORTED_TECHNICAL_SPECIFICITY" not in {
        violation.code for violation in violations
    }

    combined = " ".join(entry.text for entry in result.entries)
    assert "V2" not in combined
    assert "260" not in combined
    assert "1.1.0" not in combined

    link = result.entries[3].attachment
    assert link.type == "link"
    assert "V2" not in link.query


def test_supported_specificity_from_request_is_preserved():
    request = GenerationRequest.model_validate(
        {
            "language": "de",
            "topic": "Lukas und Markus richten einen Ender 3 V2 ein.",
            "entry_count": 2,
            "tone": "natural",
            "chaos": 10,
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
                "max_words_per_entry": 30,
                "max_images": 0,
                "max_links": 0,
            },
        }
    )

    result = ConversationResult.model_validate(
        {
            "schema_version": "1.0",
            "language": "de",
            "entries": [
                {
                    "type": "message",
                    "speaker_id": "p1",
                    "text": "Dann ist es also der Ender 3 V2.",
                    "typing_indicator": False,
                    "attachment": {"type": "none"},
                },
                {
                    "type": "message",
                    "speaker_id": "p2",
                    "text": "Gut, dann suchen wir die passende Anleitung.",
                    "typing_indicator": False,
                    "attachment": {"type": "none"},
                },
            ],
        }
    )

    adjustments = normalize_content_features(request, result)

    assert adjustments.technical_specifics_neutralized == 0
    assert "Ender 3 V2" in result.entries[0].text


def test_remaining_violation_reports_exact_detected_value():
    request = make_request()
    result = ConversationResult.model_validate(
        {
            "schema_version": "1.0",
            "language": "de",
            "entries": [
                {
                    "type": "message",
                    "speaker_id": "p1",
                    "text": "Der Extruder schafft 260 °C.",
                    "typing_indicator": False,
                    "attachment": {"type": "none"},
                },
                {
                    "type": "message",
                    "speaker_id": "p2",
                    "text": "Das sollten wir prüfen.",
                    "typing_indicator": True,
                    "attachment": {"type": "none"},
                },
                {
                    "type": "message",
                    "speaker_id": "p1",
                    "text": "Wir kennen das Modell nicht.",
                    "typing_indicator": False,
                    "attachment": {
                        "type": "link",
                        "query": "3D-Drucker Bedienungsanleitung",
                        "label": "Bedienungsanleitung",
                    },
                },
                {
                    "type": "message",
                    "speaker_id": "p2",
                    "text": "Also keine Vermutungen.",
                    "typing_indicator": False,
                    "attachment": {"type": "none"},
                },
                {
                    "type": "message",
                    "speaker_id": "p1",
                    "text": "Einverstanden.",
                    "typing_indicator": False,
                    "attachment": {"type": "none"},
                },
            ],
        }
    )

    violation = next(
        v for v in validate_semantics(request, result)
        if v.code == "UNSUPPORTED_TECHNICAL_SPECIFICITY"
    )

    assert "260 °C" in (violation.actual or "")


def test_multiple_neutralized_entries_do_not_become_duplicates():
    request = make_request()
    result = make_result()

    normalize_content_features(request, result)
    texts = [entry.text for entry in result.entries]

    assert len(texts) == len(set(texts))
    assert "DUPLICATE_ENTRY_TEXT" not in {
        violation.code for violation in validate_semantics(request, result)
    }


def test_unsupported_measurements_are_neutralized_in_research_dialogue():
    request = GenerationRequest.model_validate(
        {
            "language": "de",
            "topic": "Wie verändert sich der Siedepunkt von Wasser mit dem Luftdruck?",
            "entry_count": 2,
            "participants": [{"id": "p1", "name": "A"}, {"id": "p2", "name": "B"}],
            "features": {"narration": False, "typing_indicator": False, "images": False, "links": False, "research": True},
            "limits": {"max_narrations": 0, "max_words_per_entry": 35, "max_images": 0, "max_links": 0},
        }
    )
    result = ConversationResult.model_validate(
        {
            "schema_version": "1.0",
            "language": "de",
            "entries": [
                {"type": "message", "speaker_id": "p1", "text": "Und bei 800 m Höhe?", "typing_indicator": False, "attachment": {"type": "none"}},
                {"type": "message", "speaker_id": "p2", "text": "Bei 0 kPa bleibt Wasser flüssig.", "typing_indicator": False, "attachment": {"type": "none"}},
            ],
        }
    )
    adjustments = normalize_content_features(request, result, "")
    text = " ".join(entry.text for entry in result.entries)
    assert adjustments.technical_specifics_neutralized == 2
    assert "800 m" not in text
    assert "0 kPa" not in text
    assert "Modellangabe" not in text
