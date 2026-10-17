from app.content_normalizer import normalize_content_features
from app.models import ConversationResult, GenerationRequest


def make_request():
    return GenerationRequest.model_validate(
        {
            "language": "de",
            "topic": (
                "Lukas und Markus richten einen gebrauchten 3D-Drucker ein. "
                "Bevor sie Einstellungen verändern, möchten sie die offizielle "
                "Bedienungsanleitung und die technischen Daten des genauen Modells nachschlagen."
            ),
            "entry_count": 3,
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


def test_fallback_query_is_short_and_resource_focused():
    req = make_request()
    result = ConversationResult.model_validate(
        {
            "schema_version": "1.0",
            "language": "de",
            "entries": [
                {
                    "type": "message",
                    "speaker_id": "p1",
                    "text": "Wir sollten zuerst die Bedienungsanleitung nachsehen.",
                    "typing_indicator": False,
                    "attachment": {"type": "none"},
                },
                {
                    "type": "message",
                    "speaker_id": "p2",
                    "text": "Gute Idee.",
                    "typing_indicator": False,
                    "attachment": {"type": "none"},
                },
                {
                    "type": "message",
                    "speaker_id": "p1",
                    "text": "Danach können wir weiterarbeiten.",
                    "typing_indicator": False,
                    "attachment": {"type": "none"},
                },
            ],
        }
    )

    adjustments = normalize_content_features(req, result)
    link = next(
        entry.attachment
        for entry in result.entries
        if entry.attachment.type == "link"
    )

    assert adjustments.links_added == 1
    assert link.query == "3D-Drucker Bedienungsanleitung"
    assert link.label == "Bedienungsanleitung"
    assert "." not in link.query
    assert len(link.query.split()) <= 4
