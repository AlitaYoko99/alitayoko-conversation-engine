from app.content_normalizer import normalize_content_features
from app.factuality import unsupported_specifics
from app.models import ConversationResult, GenerationRequest
from app.validator import validate_semantics


def request():
    return GenerationRequest.model_validate(
        {
            "language": "de",
            "topic": (
                "Lukas und Markus richten einen gebrauchten 3D-Drucker ein. "
                "Sie möchten zuerst die Bedienungsanleitung prüfen."
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


def test_nonbreaking_hyphen_model_is_detected():
    hits = unsupported_specifics(
        request(),
        "Auf dem Gerät steht Model X‑200.",
    )
    assert any("X-200" == hit for hit in hits)


def test_firmware_version_with_hyphen_and_verb_is_detected():
    hits = unsupported_specifics(
        request(),
        "Die Firmware-Version ist 1.4.2.",
    )
    assert any("1.4.2" in hit for hit in hits)


def test_real_world_sample_specifics_are_neutralized():
    req = request()
    result = ConversationResult.model_validate(
        {
            "schema_version": "1.0",
            "language": "de",
            "entries": [
                {
                    "type": "message",
                    "speaker_id": "p1",
                    "text": "Hast du die Modellnummer gesehen?",
                    "typing_indicator": False,
                    "attachment": {"type": "none"},
                },
                {
                    "type": "message",
                    "speaker_id": "p2",
                    "text": "Ja, da steht Model X‑200.",
                    "typing_indicator": False,
                    "attachment": {"type": "none"},
                },
                {
                    "type": "message",
                    "speaker_id": "p1",
                    "text": "Dann sollten wir die Bedienungsanleitung suchen.",
                    "typing_indicator": True,
                    "attachment": {
                        "type": "link",
                        "query": "Model X‑200 Bedienungsanleitung",
                        "label": "Bedienungsanleitung",
                    },
                },
                {
                    "type": "message",
                    "speaker_id": "p2",
                    "text": "Die Firmware-Version ist 1.4.2.",
                    "typing_indicator": False,
                    "attachment": {"type": "none"},
                },
            ],
        }
    )

    adjustments = normalize_content_features(req, result)
    codes = {v.code for v in validate_semantics(req, result)}

    assert adjustments.technical_specifics_neutralized == 2
    assert adjustments.links_sanitized == 1
    assert "UNSUPPORTED_TECHNICAL_SPECIFICITY" not in codes
