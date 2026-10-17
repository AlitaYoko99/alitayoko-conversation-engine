from app.models import GenerationRequest
from app.wire import parse_wire_result, result_to_wire_dict


def make_request():
    return GenerationRequest.model_validate(
        {
            "language": "de",
            "topic": "Test",
            "entry_count": 3,
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
    )


def test_compact_wire_expands_to_public_contract():
    request = make_request()
    result = parse_wire_result(
        request,
        '{"e":[{"s":"p1","t":"Hallo."},{"n":"Kurze Pause."},{"s":"p2","t":"Hi."}]}',
    )

    assert result.schema_version == "1.0"
    assert result.language == "de"
    assert result.entries[0].type == "message"
    assert result.entries[0].speaker_id == "p1"
    assert result.entries[0].typing_indicator is False
    assert result.entries[1].type == "narration"
    assert result.entries[1].speaker_id == "narrator"
    assert result.entries[2].typing_indicator is False


def test_public_result_compacts_again():
    request = make_request()
    result = parse_wire_result(
        request,
        '{"e":[{"s":"p1","t":"Hallo."},{"n":"Pause."},{"s":"p2","t":"Hi."}]}',
    )
    compact = result_to_wire_dict(result)

    assert compact == {
        "e": [
            {"s": "p1", "t": "Hallo."},
            {"n": "Pause."},
            {"s": "p2", "t": "Hi."},
        ]
    }


def test_model_wire_rejects_typing_indicator_field():
    import pytest
    from app.wire import WireFormatError

    request = make_request()

    with pytest.raises(WireFormatError):
        parse_wire_result(
            request,
            '{"e":[{"s":"p1","t":"Hallo.","i":true},{"n":"Pause."},{"s":"p2","t":"Hi."}]}',
        )


def test_wire_accepts_json_wrapped_in_markdown_fence():
    request = make_request()
    result = parse_wire_result(
        request,
        '```json\n{"e":[{"s":"p1","t":"Hallo."},{"n":"Pause."},{"s":"p2","t":"Hi."}]}\n```',
    )
    assert result.entries[0].text == "Hallo."


def test_wire_accepts_short_text_wrapper_around_valid_json():
    request = make_request()
    result = parse_wire_result(
        request,
        'Hier ist das JSON:\n{"e":[{"s":"p1","t":"Hallo."},{"n":"Pause."},{"s":"p2","t":"Hi."}]}\nEnde.',
    )
    assert result.entries[2].speaker_id == "p2"
