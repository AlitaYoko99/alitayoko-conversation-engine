from app.content_normalizer import normalize_content_features
from app.models import ConversationResult, GenerationRequest


def make_request(*, images=True, links=False, max_words=10):
    return GenerationRequest.model_validate(
        {
            "language": "de",
            "topic": "Ein Saugroboter transportiert ständig eine rote Socke",
            "entry_count": 4,
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
                "max_words_per_entry": max_words,
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
                    "type": "narration",
                    "speaker_id": "narrator",
                    "text": "Der Saugroboter zieht die rote Socke langsam durch den Flur.",
                    "typing_indicator": False,
                    "attachment": {"type": "none"},
                },
                {
                    "type": "message",
                    "speaker_id": "p1",
                    "text": "Das ist schon wieder dieselbe Socke.",
                    "typing_indicator": False,
                    "attachment": {"type": "none"},
                },
                {
                    "type": "message",
                    "speaker_id": "p2",
                    "text": "Ich habe dir doch gesagt, dass er sie wie eine Trophäe behandelt und nicht mehr loslässt.",
                    "typing_indicator": False,
                    "attachment": {"type": "none"},
                },
                {
                    "type": "message",
                    "speaker_id": "p1",
                    "text": "Vielleicht sammelt er Beweise gegen deinen Kleiderstil.",
                    "typing_indicator": False,
                    "attachment": {"type": "none"},
                },
            ],
        }
    )


def test_missing_image_is_added():
    request = make_request(images=True, links=False, max_words=35)
    result = make_result()

    adjustments = normalize_content_features(request, result)

    assert adjustments.images_added == 1
    assert any(entry.attachment.type == "image" for entry in result.entries)


def test_missing_link_is_added():
    request = make_request(images=False, links=True, max_words=35)
    result = make_result()

    adjustments = normalize_content_features(request, result)

    assert adjustments.links_added == 1
    assert any(entry.attachment.type == "link" for entry in result.entries)


def test_overlong_entry_is_shortened():
    request = make_request(images=False, links=False, max_words=8)
    result = make_result()

    adjustments = normalize_content_features(request, result)

    assert adjustments.entries_shortened >= 1
    assert all(len(entry.text.split()) <= 8 for entry in result.entries)


def test_fallback_image_is_never_attached_to_narration():
    request = make_request(images=True, links=False, max_words=35)
    result = make_result()

    adjustments = normalize_content_features(request, result)

    assert adjustments.images_added == 1

    narration_entries = [
        entry for entry in result.entries if entry.type == "narration"
    ]
    message_entries = [
        entry for entry in result.entries if entry.type == "message"
    ]

    assert all(entry.attachment.type == "none" for entry in narration_entries)
    assert sum(entry.attachment.type == "image" for entry in message_entries) == 1


def test_fallback_image_can_use_narration_as_description_source_but_message_as_anchor():
    request = make_request(images=True, links=False, max_words=35)
    result = make_result()

    normalize_content_features(request, result)

    image_entries = [
        entry for entry in result.entries
        if entry.type == "message" and entry.attachment.type == "image"
    ]

    assert len(image_entries) == 1
    image = image_entries[0].attachment
    assert image.type == "image"
    assert "Saugroboter" in image.description
