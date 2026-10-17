import pytest

from app.model_profiles import get_profile, profile_for_model


def test_ministral_profile():
    profile = get_profile("ministral-14b")
    assert profile.model == "ministral-3:14b"
    assert profile.reasoning_effort is None
    assert profile.ollama_think is None
    assert profile.num_predict_multiplier == 1.0


def test_gpt_oss_profile_uses_low_reasoning():
    profile = get_profile("gpt-oss-20b")
    assert profile.model == "gpt-oss:20b"
    assert profile.reasoning_effort == "low"
    assert profile.ollama_think == "low"
    assert profile.num_predict_multiplier > 1.0
    assert profile.force_schema_grounding is True


def test_profile_for_known_model():
    profile = profile_for_model("gpt-oss:20b")
    assert profile.id == "gpt-oss-20b"


def test_profile_for_custom_model():
    profile = profile_for_model("example:latest")
    assert profile.id == "custom"
    assert profile.model == "example:latest"


def test_unknown_profile_fails_clearly():
    with pytest.raises(ValueError):
        get_profile("does-not-exist")
