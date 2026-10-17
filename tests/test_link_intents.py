from app.content_normalizer import normalize_content_features
from app.models import ConversationResult, GenerationRequest
from app.validator import validate_semantics


def make_request(*, images=False):
    return GenerationRequest.model_validate(
        {
            "language": "de",
            "topic": (
                "Lukas und Markus richten einen gebrauchten 3D-Drucker ein "
                "und wollen die offizielle Bedienungsanleitung nachschlagen."
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
                "links": True,
            },
            "limits": {
                "max_narrations": 1,
                "max_words_per_entry": 30,
                "max_images": 1 if images else 0,
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
                    "text": "Bevor wir etwas ändern, sollten wir die Bedienungsanleitung nachsehen.",
                    "typing_indicator": False,
                    "attachment": {"type": "none"},
                },
                {
                    "type": "narration",
                    "speaker_id": "narrator",
                    "text": "Markus dreht den Drucker vorsichtig zur Seite.",
                    "typing_indicator": False,
                    "attachment": {"type": "none"},
                },
                {
                    "type": "message",
                    "speaker_id": "p2",
                    "text": "Gut, dann suchen wir zuerst die technischen Daten.",
                    "typing_indicator": True,
                    "attachment": {"type": "none"},
                },
                {
                    "type": "message",
                    "speaker_id": "p1",
                    "text": "Danach können wir die Einstellungen vergleichen.",
                    "typing_indicator": False,
                    "attachment": {"type": "none"},
                },
                {
                    "type": "message",
                    "speaker_id": "p2",
                    "text": "Einverstanden.",
                    "typing_indicator": False,
                    "attachment": {"type": "none"},
                },
            ],
        }
    )


def test_link_fallback_prefers_resource_related_message():
    request = make_request()
    result = make_result()

    adjustments = normalize_content_features(request, result)

    assert adjustments.links_added == 1
    link_entries = [
        entry for entry in result.entries if entry.attachment.type == "link"
    ]
    assert len(link_entries) == 1
    assert "Bedienungsanleitung" in link_entries[0].attachment.label
    assert "3D-Drucker" in link_entries[0].attachment.query


def test_link_fallback_never_attaches_to_narration():
    request = make_request()
    result = make_result()

    normalize_content_features(request, result)

    assert all(
        entry.attachment.type == "none"
        for entry in result.entries
        if entry.type == "narration"
    )


def test_link_fallback_avoids_existing_image_attachment():
    request = make_request(images=True)
    result = make_result()
    raw = result.model_dump()
    raw["entries"][0]["attachment"] = {
        "type": "image",
        "description": "Ein gebrauchter 3D-Drucker auf einem Tisch.",
        "alt_text": "3D-Drucker auf einem Tisch",
    }
    result = ConversationResult.model_validate(raw)

    normalize_content_features(request, result)

    image_indices = [
        i for i, entry in enumerate(result.entries)
        if entry.attachment.type == "image"
    ]
    link_indices = [
        i for i, entry in enumerate(result.entries)
        if entry.attachment.type == "link"
    ]

    assert len(image_indices) == 1
    assert len(link_indices) == 1
    assert image_indices[0] != link_indices[0]


def test_validator_rejects_url_inside_link_intent():
    request = make_request()
    result = make_result()
    raw = result.model_dump()
    raw["entries"][0]["attachment"] = {
        "type": "link",
        "query": "https://example.com/manual",
        "label": "Bedienungsanleitung",
    }
    result = ConversationResult.model_validate(raw)

    codes = {item.code for item in validate_semantics(request, result)}

    assert "LINK_INTENT_CONTAINS_URL" in codes


def test_normalized_fallback_link_passes_link_validation():
    request = make_request()
    result = make_result()

    normalize_content_features(request, result)
    violations = validate_semantics(request, result)

    codes = {item.code for item in violations}
    assert "MISSING_LINK" not in codes
    assert "INVALID_ATTACHMENT" not in codes
    assert "LINK_INTENT_CONTAINS_URL" not in codes


def test_model_url_link_is_sanitized_without_semantic_repair():
    request = make_request()
    result = make_result()

    raw = result.model_dump()
    raw["entries"][0]["attachment"] = {
        "type": "link",
        "query": "https://manufacturer.example.com/manual.pdf",
        "label": "https://manufacturer.example.com/manual.pdf",
    }
    result = ConversationResult.model_validate(raw)

    adjustments = normalize_content_features(request, result)
    violations = validate_semantics(request, result)

    assert adjustments.links_sanitized == 1
    assert adjustments.links_added == 0
    assert result.entries[0].attachment.type == "link"
    assert "http" not in result.entries[0].attachment.query.casefold()
    assert "http" not in result.entries[0].attachment.label.casefold()
    assert "LINK_INTENT_CONTAINS_URL" not in {v.code for v in violations}


def test_too_many_model_links_are_capped_deterministically():
    request = make_request()
    result = make_result()

    raw = result.model_dump()
    raw["entries"][0]["attachment"] = {
        "type": "link",
        "query": "3D-Drucker Bedienungsanleitung",
        "label": "Bedienungsanleitung",
    }
    raw["entries"][2]["attachment"] = {
        "type": "link",
        "query": "3D-Drucker technische Daten",
        "label": "Technische Daten",
    }
    result = ConversationResult.model_validate(raw)

    adjustments = normalize_content_features(request, result)
    violations = validate_semantics(request, result)

    assert adjustments.links_removed == 1
    assert sum(entry.attachment.type == "link" for entry in result.entries) == 1
    assert "TOO_MANY_LINKS" not in {v.code for v in violations}
