from app.models import ConversationResult, GenerationRequest
from app.validator import validate_semantics


def make_request():
    return GenerationRequest.model_validate(
        {
            "language": "de",
            "topic": "Test topic",
            "entry_count": 4,
            "participants": [
                {"id": "p1", "name": "Lukas"},
                {"id": "p2", "name": "Markus"},
            ],
            "features": {
                "narration": True,
                "typing_indicator": True,
                "images": False,
                "links": False,
            },
            "limits": {
                "max_narrations": 1,
                "max_words_per_entry": 8,
                "max_images": 0,
                "max_links": 0,
            },
        }
    )


def test_valid_conversation_passes():
    request = make_request()
    result = ConversationResult.model_validate(
        {
            "schema_version": "1.0",
            "language": "de",
            "entries": [
                {
                    "type": "narration",
                    "speaker_id": "narrator",
                    "text": "Die Küche wird plötzlich still.",
                    "typing_indicator": False,
                    "attachment": {"type": "none"},
                },
                {
                    "type": "message",
                    "speaker_id": "p1",
                    "text": "Hast du das gehört?",
                    "typing_indicator": True,
                    "attachment": {"type": "none"},
                },
                {
                    "type": "message",
                    "speaker_id": "p2",
                    "text": "Leider ja.",
                    "typing_indicator": False,
                    "attachment": {"type": "none"},
                },
                {
                    "type": "message",
                    "speaker_id": "p1",
                    "text": "Dann war das keine Einbildung.",
                    "typing_indicator": False,
                    "attachment": {"type": "none"},
                },
            ],
        }
    )

    assert validate_semantics(request, result) == []


def test_too_many_narrations_is_detected():
    request = make_request()
    result = ConversationResult.model_validate(
        {
            "schema_version": "1.0",
            "language": "de",
            "entries": [
                {
                    "type": "narration",
                    "speaker_id": "narrator",
                    "text": "Erste Szene.",
                    "typing_indicator": False,
                    "attachment": {"type": "none"},
                },
                {
                    "type": "message",
                    "speaker_id": "p1",
                    "text": "Hallo.",
                    "typing_indicator": False,
                    "attachment": {"type": "none"},
                },
                {
                    "type": "message",
                    "speaker_id": "p2",
                    "text": "Hallo.",
                    "typing_indicator": False,
                    "attachment": {"type": "none"},
                },
                {
                    "type": "narration",
                    "speaker_id": "narrator",
                    "text": "Zweite Szene.",
                    "typing_indicator": False,
                    "attachment": {"type": "none"},
                },
            ],
        }
    )

    codes = {item.code for item in validate_semantics(request, result)}
    assert "TOO_MANY_NARRATIONS" in codes


def test_overlong_entry_is_detected():
    request = make_request()
    result = ConversationResult.model_validate(
        {
            "schema_version": "1.0",
            "language": "de",
            "entries": [
                {
                    "type": "message",
                    "speaker_id": "p1",
                    "text": "Eins zwei drei vier fünf sechs sieben acht neun.",
                    "typing_indicator": False,
                    "attachment": {"type": "none"},
                },
                {
                    "type": "message",
                    "speaker_id": "p2",
                    "text": "Kurz.",
                    "typing_indicator": False,
                    "attachment": {"type": "none"},
                },
                {
                    "type": "message",
                    "speaker_id": "p1",
                    "text": "Kurz.",
                    "typing_indicator": False,
                    "attachment": {"type": "none"},
                },
                {
                    "type": "message",
                    "speaker_id": "p2",
                    "text": "Kurz.",
                    "typing_indicator": False,
                    "attachment": {"type": "none"},
                },
            ],
        }
    )

    codes = {item.code for item in validate_semantics(request, result)}
    assert "ENTRY_TOO_LONG" in codes


def test_foreign_language_intrusion_is_detected():
    request = make_request()
    result = ConversationResult.model_validate(
        {
            "schema_version": "1.0",
            "language": "de",
            "entries": [
                {
                    "type": "message",
                    "speaker_id": "p1",
                    "text": "Das wird probably nicht gut enden.",
                    "typing_indicator": False,
                    "attachment": {"type": "none"},
                },
                {
                    "type": "message",
                    "speaker_id": "p2",
                    "text": "Das sehe ich genauso.",
                    "typing_indicator": False,
                    "attachment": {"type": "none"},
                },
                {
                    "type": "message",
                    "speaker_id": "p1",
                    "text": "Dann schalten wir es ab.",
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

    codes = {item.code for item in validate_semantics(request, result)}
    assert "FOREIGN_LANGUAGE_INTRUSION" in codes


def test_duplicate_entry_is_detected():
    request = make_request()
    result = ConversationResult.model_validate(
        {
            "schema_version": "1.0",
            "language": "de",
            "entries": [
                {
                    "type": "message",
                    "speaker_id": "p1",
                    "text": "Das ist seltsam.",
                    "typing_indicator": False,
                    "attachment": {"type": "none"},
                },
                {
                    "type": "message",
                    "speaker_id": "p2",
                    "text": "Das ist seltsam.",
                    "typing_indicator": False,
                    "attachment": {"type": "none"},
                },
                {
                    "type": "message",
                    "speaker_id": "p1",
                    "text": "Wir prüfen die Logs.",
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
            ],
        }
    )

    codes = {item.code for item in validate_semantics(request, result)}
    assert "DUPLICATE_ENTRY_TEXT" in codes


def test_hostage_code_switch_is_detected_in_german():
    request = make_request()
    result = ConversationResult.model_validate(
        {
            "schema_version": "1.0",
            "language": "de",
            "entries": [
                {
                    "type": "message",
                    "speaker_id": "p1",
                    "text": "Das ist eine Hostage-Situation.",
                    "typing_indicator": False,
                    "attachment": {"type": "none"},
                },
                {
                    "type": "message",
                    "speaker_id": "p2",
                    "text": "Dann ziehen wir den Stecker.",
                    "typing_indicator": False,
                    "attachment": {"type": "none"},
                },
                {
                    "type": "message",
                    "speaker_id": "p1",
                    "text": "Mach das.",
                    "typing_indicator": False,
                    "attachment": {"type": "none"},
                },
                {
                    "type": "message",
                    "speaker_id": "p2",
                    "text": "Erledigt.",
                    "typing_indicator": False,
                    "attachment": {"type": "none"},
                },
            ],
        }
    )

    codes = {item.code for item in validate_semantics(request, result)}
    assert "FOREIGN_LANGUAGE_INTRUSION" in codes


def test_enabled_typing_indicator_is_required():
    request = make_request()
    request.features.typing_indicator = True

    result = ConversationResult.model_validate(
        {
            "schema_version": "1.0",
            "language": "de",
            "entries": [
                {
                    "type": "narration",
                    "speaker_id": "narrator",
                    "text": "Kurze Szene.",
                    "typing_indicator": False,
                    "attachment": {"type": "none"},
                },
                {
                    "type": "message",
                    "speaker_id": "p1",
                    "text": "Hallo.",
                    "typing_indicator": False,
                    "attachment": {"type": "none"},
                },
                {
                    "type": "message",
                    "speaker_id": "p2",
                    "text": "Hallo.",
                    "typing_indicator": False,
                    "attachment": {"type": "none"},
                },
                {
                    "type": "message",
                    "speaker_id": "p1",
                    "text": "Alles gut.",
                    "typing_indicator": False,
                    "attachment": {"type": "none"},
                },
            ],
        }
    )

    codes = {item.code for item in validate_semantics(request, result)}
    assert "MISSING_TYPING_INDICATOR" in codes


def test_typing_metadata_as_dialogue_text_is_rejected():
    request = make_request()
    result = ConversationResult.model_validate(
        {
            "schema_version": "1.0",
            "language": "de",
            "entries": [
                {
                    "type": "narration",
                    "speaker_id": "narrator",
                    "text": "Kurze Szene.",
                    "typing_indicator": False,
                    "attachment": {"type": "none"},
                },
                {
                    "type": "message",
                    "speaker_id": "p1",
                    "text": "*i*:true",
                    "typing_indicator": True,
                    "attachment": {"type": "none"},
                },
                {
                    "type": "message",
                    "speaker_id": "p2",
                    "text": "Hallo.",
                    "typing_indicator": False,
                    "attachment": {"type": "none"},
                },
                {
                    "type": "message",
                    "speaker_id": "p1",
                    "text": "Alles gut.",
                    "typing_indicator": False,
                    "attachment": {"type": "none"},
                },
            ],
        }
    )

    codes = {item.code for item in validate_semantics(request, result)}
    assert "TYPING_METADATA_IN_TEXT" in codes
