from app.models import GenerationRequest
from app.schema_builder import build_conversation_schema


def make_request(**overrides):
    data = {
        "language": "de",
        "topic": "Test topic",
        "entry_count": 8,
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
            "max_words_per_entry": 20,
            "max_images": 0,
            "max_links": 0,
        },
    }
    data.update(overrides)
    return GenerationRequest.model_validate(data)


def test_schema_has_exact_entry_count():
    request = make_request()
    schema = build_conversation_schema(request)
    entries = schema["properties"]["e"]

    assert entries["minItems"] == 8
    assert entries["maxItems"] == 8


def test_message_schema_restricts_speakers():
    request = make_request()
    schema = build_conversation_schema(request)
    message_schema = schema["properties"]["e"]["items"]["oneOf"][0]

    assert message_schema["properties"]["s"]["enum"] == ["p1", "p2"]


def test_disabled_attachments_have_no_wire_field():
    request = make_request()
    schema = build_conversation_schema(request)
    message_schema = schema["properties"]["e"]["items"]["oneOf"][0]

    assert "a" not in message_schema["properties"]


def test_disabled_typing_has_no_wire_field():
    request = make_request(
        features={
            "narration": True,
            "typing_indicator": False,
            "images": False,
            "links": False,
        }
    )
    schema = build_conversation_schema(request)
    message_schema = schema["properties"]["e"]["items"]["oneOf"][0]

    assert "i" not in message_schema["properties"]


def test_enabled_typing_is_still_absent_from_model_schema():
    request = make_request(
        features={
            "narration": True,
            "typing_indicator": True,
            "images": False,
            "links": False,
        }
    )
    schema = build_conversation_schema(request)
    message_schema = schema["properties"]["e"]["items"]["oneOf"][0]

    assert "i" not in message_schema["properties"]
