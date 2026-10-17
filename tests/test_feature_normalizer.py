from app.feature_normalizer import normalize_presentation_features
from app.models import ConversationResult, GenerationRequest


def make_request(typing=True):
    return GenerationRequest.model_validate(
        {
            "language": "de",
            "topic": "Test",
            "entry_count": 5,
            "participants": [
                {"id": "p1", "name": "Lukas"},
                {"id": "p2", "name": "Markus"},
            ],
            "features": {
                "narration": True,
                "typing_indicator": typing,
                "images": False,
                "links": False,
            },
            "limits": {
                "max_narrations": 1,
                "max_words_per_entry": 35,
                "max_images": 0,
                "max_links": 0,
            },
        }
    )


def make_result(indicators=None):
    indicators = indicators or set()
    return ConversationResult.model_validate(
        {
            "schema_version": "1.0",
            "language": "de",
            "entries": [
                {
                    "type": "message",
                    "speaker_id": "p1",
                    "text": "Kurzer Einstieg.",
                    "typing_indicator": 0 in indicators,
                    "attachment": {"type": "none"},
                },
                {
                    "type": "message",
                    "speaker_id": "p2",
                    "text": "Diese Nachricht ist etwas länger und eignet sich für eine Pause.",
                    "typing_indicator": 1 in indicators,
                    "attachment": {"type": "none"},
                },
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
                    "text": "Noch eine recht ausführliche Antwort für den Dialog.",
                    "typing_indicator": 3 in indicators,
                    "attachment": {"type": "none"},
                },
                {
                    "type": "message",
                    "speaker_id": "p2",
                    "text": "Schluss.",
                    "typing_indicator": 4 in indicators,
                    "attachment": {"type": "none"},
                },
            ],
        }
    )


def test_missing_typing_indicator_is_added_without_llm_repair():
    request = make_request(typing=True)
    result = make_result()

    adjustments = normalize_presentation_features(request, result)

    assert adjustments.typing_indicators_added == 1
    assert sum(entry.typing_indicator for entry in result.entries) == 1


def test_disabled_typing_indicators_are_removed():
    request = make_request(typing=False)
    result = make_result({1, 3})

    adjustments = normalize_presentation_features(request, result)

    assert adjustments.typing_indicators_removed == 2
    assert sum(entry.typing_indicator for entry in result.entries) == 0


def test_too_many_typing_indicators_are_capped():
    request = make_request(typing=True)
    result = make_result({0, 1, 3, 4})

    adjustments = normalize_presentation_features(request, result)

    assert adjustments.typing_indicators_removed == 1
    assert sum(entry.typing_indicator for entry in result.entries) == 3
