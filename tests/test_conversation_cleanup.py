from app.content_normalizer import normalize_content_features
from app.models import ConversationResult, GenerationRequest
from app.validator import validate_semantics


def make_request(*, images=False, links=True):
    return GenerationRequest.model_validate(
        {
            "language": "de",
            "topic": (
                "Lukas und Markus richten einen gebrauchten 3D-Drucker ein und "
                "möchten vor Änderungen eine passende Anleitung suchen."
            ),
            "entry_count": 5,
            "tone": "natural",
            "chaos": 10,
            "participants": [
                {"id": "p1", "name": "Lukas"},
                {"id": "p2", "name": "Markus"},
            ],
            "features": {
                "narration": True,
                "typing_indicator": True,
                "images": images,
                "links": links,
            },
            "limits": {
                "max_narrations": 1,
                "max_words_per_entry": 30,
                "max_images": 1 if images else 0,
                "max_links": 1 if links else 0,
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
                    "text": "Lukas: Wir sollten zuerst die Anleitung suchen.",
                    "typing_indicator": False,
                    "attachment": {"type": "none"},
                },
                {
                    "type": "message",
                    "speaker_id": "p2",
                    "text": "Markus: Gute Idee. Ich habe die offizielle PDF bereits geöffnet.",
                    "typing_indicator": True,
                    "attachment": {
                        "type": "link",
                        "query": "3D-Drucker Bedienungsanleitung",
                        "label": "Bedienungsanleitung",
                    },
                },
                {
                    "type": "message",
                    "speaker_id": "p1",
                    "text": "Dann prüfen wir das genaue Modell.",
                    "typing_indicator": False,
                    "attachment": {"type": "none"},
                },
                {
                    "type": "narration",
                    "speaker_id": "narrator",
                    "text": "Die beiden sind zufrieden mit den gefundenen Daten.",
                    "typing_indicator": False,
                    "attachment": {"type": "none"},
                },
                {
                    "type": "message",
                    "speaker_id": "p2",
                    "text": "Danach können wir weitermachen.",
                    "typing_indicator": False,
                    "attachment": {"type": "none"},
                },
            ],
        }
    )


def test_cleanup_removes_speaker_prefixes_and_source_claims():
    request = make_request()
    result = make_result()

    adjustments = normalize_content_features(request, result)
    violations = validate_semantics(request, result)

    assert adjustments.speaker_prefixes_removed == 2
    assert adjustments.source_claims_neutralized == 2
    assert not result.entries[0].text.startswith("Lukas:")
    assert not result.entries[1].text.startswith("Markus:")

    codes = {violation.code for violation in violations}
    assert "REDUNDANT_SPEAKER_PREFIX" not in codes
    assert "UNRESOLVED_SOURCE_CLAIM" not in codes


def test_combined_image_and_link_remain_on_separate_messages():
    request = make_request(images=True, links=True)
    result = make_result()

    normalize_content_features(request, result)

    image_indices = [
        index
        for index, entry in enumerate(result.entries)
        if entry.attachment.type == "image"
    ]
    link_indices = [
        index
        for index, entry in enumerate(result.entries)
        if entry.attachment.type == "link"
    ]

    assert len(image_indices) == 1
    assert len(link_indices) == 1
    assert image_indices[0] != link_indices[0]
    assert all(
        entry.attachment.type == "none"
        for entry in result.entries
        if entry.type == "narration"
    )
