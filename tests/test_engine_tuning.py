import pytest

from app.engine import num_predict_for, temperature_from_chaos
from app.models import GenerationRequest
from app.prompts import preferred_message_words, preferred_narration_words


def make_request():
    return GenerationRequest.model_validate(
        {
            "language": "de",
            "topic": "Test",
            "entry_count": 12,
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
                "max_narrations": 2,
                "max_words_per_entry": 35,
                "max_images": 0,
                "max_links": 0,
            },
        }
    )


def test_temperature_curve_is_conservative_at_chaos_35():
    assert temperature_from_chaos(35) == pytest.approx(0.349, abs=0.001)


def test_temperature_curve_is_monotonic():
    values = [temperature_from_chaos(value) for value in range(0, 101, 10)]
    assert values == sorted(values)


def test_preferred_bubble_lengths_do_not_replace_hard_limit():
    request = make_request()
    assert preferred_message_words(request) == (6, 15)
    assert preferred_narration_words(request) == (7, 18)
    assert request.limits.max_words_per_entry == 35


def test_prediction_budget_is_smaller_than_v012_for_example():
    request = make_request()
    # v0.1.2 would budget roughly 854 tokens for this request.
    assert num_predict_for(request) < 854
    assert num_predict_for(request) >= 520
