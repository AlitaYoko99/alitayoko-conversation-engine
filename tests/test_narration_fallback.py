from app.content_normalizer import normalize_content_features
from app.feature_normalizer import normalize_presentation_features
from app.models import ConversationResult, GenerationRequest
from app.validator import validate_semantics


def make_request():
    return GenerationRequest.model_validate(
        {
            "language": "de",
            "topic": "Lukas und Markus richten gemeinsam einen gebrauchten 3D-Drucker ein.",
            "entry_count": 6,
            "tone": "natural",
            "chaos": 10,
            "participants": [
                {"id": "p1", "name": "Lukas"},
                {"id": "p2", "name": "Markus"},
            ],
            "features": {
                "narration": True,
                "typing_indicator": True,
                "images": True,
                "links": True,
            },
            "limits": {
                "max_narrations": 1,
                "max_words_per_entry": 35,
                "max_images": 1,
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
                {"type":"message","speaker_id":"p1","text":"Der Halter sitzt schief.","typing_indicator":False,"attachment":{"type":"none"}},
                {"type":"message","speaker_id":"p2","text":"Dann suchen wir die Anleitung.","typing_indicator":False,"attachment":{"type":"link","query":"3D-Drucker Bedienungsanleitung","label":"Bedienungsanleitung"}},
                {"type":"message","speaker_id":"p1","text":"Okay.","typing_indicator":False,"attachment":{"type":"none"}},
                {"type":"message","speaker_id":"p2","text":"Ich prüfe die Halterung genauer.","typing_indicator":False,"attachment":{"type":"image","description":"3D-Drucker mit schiefem Filamenthalter","alt_text":"Schiefer Filamenthalter"}},
                {"type":"message","speaker_id":"p1","text":"Dann richten wir sie neu aus.","typing_indicator":False,"attachment":{"type":"none"}},
                {"type":"message","speaker_id":"p2","text":"Danach testen wir den Drucker.","typing_indicator":False,"attachment":{"type":"none"}},
            ],
        }
    )


def test_missing_narration_is_added_without_changing_entry_count():
    request = make_request()
    result = make_result()

    adjustments = normalize_content_features(request, result)
    normalize_presentation_features(request, result)

    assert adjustments.narrations_added == 1
    assert len(result.entries) == request.entry_count
    assert sum(entry.type == "narration" for entry in result.entries) == 1
    assert sum(entry.attachment.type == "image" for entry in result.entries) == 1
    assert sum(entry.attachment.type == "link" for entry in result.entries) == 1


def test_narration_fallback_keeps_every_participant_speaking():
    request = make_request()
    result = make_result()

    normalize_content_features(request, result)

    speakers = {
        entry.speaker_id
        for entry in result.entries
        if entry.type == "message"
    }
    assert speakers == {"p1", "p2"}


def test_typing_is_assigned_after_narration_fallback():
    request = make_request()
    result = make_result()

    normalize_content_features(request, result)
    typing = normalize_presentation_features(request, result)

    assert typing.typing_indicators_added == 1
    assert sum(entry.typing_indicator for entry in result.entries) == 1
    assert all(
        not entry.typing_indicator
        for entry in result.entries
        if entry.type == "narration"
    )


def test_combined_result_has_no_missing_narration_violation():
    request = make_request()
    result = make_result()

    normalize_content_features(request, result)
    normalize_presentation_features(request, result)
    codes = {violation.code for violation in validate_semantics(request, result)}

    assert "MISSING_NARRATION" not in codes
    assert "MISSING_PARTICIPANTS" not in codes
    assert "MISSING_IMAGE" not in codes
    assert "MISSING_LINK" not in codes
