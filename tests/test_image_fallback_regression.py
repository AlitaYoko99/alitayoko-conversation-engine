from app.content_normalizer import normalize_content_features
from app.feature_normalizer import normalize_presentation_features
from app.models import ConversationResult, GenerationRequest
from app.validator import validate_semantics


def test_image_fallback_does_not_create_invalid_attachment_violation():
    request = GenerationRequest.model_validate(
        {
            "language": "de",
            "topic": "Ein Saugroboter transportiert immer wieder dieselbe rote Socke.",
            "entry_count": 4,
            "tone": "natural",
            "chaos": 30,
            "participants": [
                {"id": "p1", "name": "Lukas"},
                {"id": "p2", "name": "Markus"},
            ],
            "features": {
                "narration": True,
                "typing_indicator": True,
                "images": True,
                "links": False,
            },
            "limits": {
                "max_narrations": 1,
                "max_words_per_entry": 35,
                "max_images": 1,
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
                    "text": "Da ist die rote Socke schon wieder.",
                    "typing_indicator": False,
                    "attachment": {"type": "none"},
                },
                {
                    "type": "narration",
                    "speaker_id": "narrator",
                    "text": "Der Saugroboter zieht die rote Socke quer durch den Flur.",
                    "typing_indicator": False,
                    "attachment": {"type": "none"},
                },
                {
                    "type": "message",
                    "speaker_id": "p2",
                    "text": "Vielleicht hält er sie für eine Trophäe.",
                    "typing_indicator": False,
                    "attachment": {"type": "none"},
                },
                {
                    "type": "message",
                    "speaker_id": "p1",
                    "text": "Dann bekommt er besser keinen Wäschekorb.",
                    "typing_indicator": False,
                    "attachment": {"type": "none"},
                },
            ],
        }
    )

    normalize_presentation_features(request, result)
    normalize_content_features(request, result)

    violations = validate_semantics(request, result)
    codes = {violation.code for violation in violations}

    assert "INVALID_ATTACHMENT" not in codes
    assert "MISSING_IMAGE" not in codes
    assert any(
        entry.type == "message" and entry.attachment.type == "image"
        for entry in result.entries
    )
    assert all(
        entry.attachment.type == "none"
        for entry in result.entries
        if entry.type == "narration"
    )
