import pytest
from pydantic import ValidationError

from app.engine import temperature_from_chaos
from app.models import GenerationRequest
from app.prompts import build_system_prompt, seriousness_guidance


def request_data(**overrides):
    data = {
        "schema_version": "1.0",
        "language": "de",
        "topic": "Lukas und Markus planen gemeinsam den nächsten Arbeitsschritt.",
        "entry_count": 6,
        "tone": "natural",
        "chaos": 35,
        "participants": [
            {"id": "p1", "name": "Lukas", "personality": "pragmatisch"},
            {"id": "p2", "name": "Markus", "personality": "ruhig"},
        ],
        "features": {
            "narration": False,
            "typing_indicator": False,
            "images": False,
            "links": False,
        },
        "limits": {
            "max_narrations": 0,
            "max_words_per_entry": 35,
            "max_images": 0,
            "max_links": 0,
        },
    }
    data.update(overrides)
    return data


def test_seriousness_defaults_to_neutral_50():
    request = GenerationRequest.model_validate(request_data())
    assert request.seriousness == 50


@pytest.mark.parametrize("value", [0, 1, 15, 50, 84, 100])
def test_seriousness_accepts_full_public_range(value):
    request = GenerationRequest.model_validate(request_data(seriousness=value))
    assert request.seriousness == value


@pytest.mark.parametrize("value", [-1, 101])
def test_seriousness_rejects_values_outside_public_range(value):
    with pytest.raises(ValidationError):
        GenerationRequest.model_validate(request_data(seriousness=value))


def test_seriousness_does_not_change_sampling_temperature():
    playful = GenerationRequest.model_validate(request_data(chaos=35, seriousness=0))
    sober = GenerationRequest.model_validate(request_data(chaos=35, seriousness=100))

    assert temperature_from_chaos(playful.chaos) == temperature_from_chaos(sober.chaos)
    assert temperature_from_chaos(playful.chaos) == pytest.approx(0.349, abs=0.001)


def test_prompt_explicitly_separates_chaos_and_seriousness():
    request = GenerationRequest.model_validate(request_data(chaos=80, seriousness=90))
    prompt = build_system_prompt(request)

    assert "Chaos 80/100" in prompt
    assert "Seriousness 90/100" in prompt
    assert "independent controls" in prompt
    assert "Chaos changes unpredictability and absurdity" in prompt
    assert "seriousness changes register, humor and emotional weight" in prompt
    assert "very sober and professional" in prompt


def test_low_seriousness_guidance_is_playful_not_chaotic():
    guidance = seriousness_guidance(5)
    assert "playful" in guidance
    assert "humor" in guidance
    assert "absurd" not in guidance


def test_neutral_seriousness_guidance_is_balanced():
    assert "balanced and natural" in seriousness_guidance(50)
