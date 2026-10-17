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
            "entry_count": 4,
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


def result_with(entries):
    return ConversationResult.model_validate(
        {
            "schema_version": "1.0",
            "language": "de",
            "entries": entries,
        }
    )


def base_entries():
    return [
        {
            "type": "message",
            "speaker_id": "p1",
            "text": "Wir kennen das genaue Modell noch nicht.",
            "typing_indicator": False,
            "attachment": {"type": "none"},
        },
        {
            "type": "message",
            "speaker_id": "p2",
            "text": "Dann sollten wir zuerst das Typenschild prüfen.",
            "typing_indicator": True,
            "attachment": {"type": "none"},
        },
        {
            "type": "message",
            "speaker_id": "p1",
            "text": "Danach suchen wir die passende Bedienungsanleitung.",
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
            "text": "So ändern wir nichts auf Verdacht.",
            "typing_indicator": False,
            "attachment": {"type": "none"},
        },
    ]


def test_generic_unknown_facts_pass():
    request = make_request()
    result = result_with(base_entries())

    codes = {v.code for v in validate_semantics(request, result)}
    assert "UNSUPPORTED_TECHNICAL_SPECIFICITY" not in codes


def test_invented_temperature_is_rejected():
    request = make_request()
    entries = base_entries()
    entries[0]["text"] = "Der Extruder schafft 260 °C."
    result = result_with(entries)

    codes = {v.code for v in validate_semantics(request, result)}
    assert "UNSUPPORTED_TECHNICAL_SPECIFICITY" in codes


def test_invented_firmware_version_is_rejected():
    request = make_request()
    entries = base_entries()
    entries[1]["text"] = "Firmware-Version 1.1.0 ist die neueste."
    result = result_with(entries)

    codes = {v.code for v in validate_semantics(request, result)}
    assert "UNSUPPORTED_TECHNICAL_SPECIFICITY" in codes


def test_invented_model_token_in_link_is_canonicalized():
    request = make_request()
    entries = base_entries()
    entries[2]["attachment"] = {
        "type": "link",
        "query": "Ender 3 V2 official manual",
        "label": "Ender 3 V2 Manual",
    }
    result = result_with(entries)

    adjustments = normalize_content_features(request, result)
    link = result.entries[2].attachment

    assert adjustments.links_sanitized == 1
    assert link.type == "link"
    assert "Ender" not in link.query
    assert "V2" not in link.query
