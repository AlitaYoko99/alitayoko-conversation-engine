from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ModelProfile:
    id: str
    model: str
    label: str
    reasoning_effort: str | None = None
    ollama_think: bool | str | None = None
    num_predict_multiplier: float = 1.0
    force_schema_grounding: bool = False


PROFILES: dict[str, ModelProfile] = {
    "ministral-14b": ModelProfile(
        id="ministral-14b",
        model="ministral-3:14b",
        label="Ministral 3 14B",
    ),
    "gpt-oss-20b": ModelProfile(
        id="gpt-oss-20b",
        model="gpt-oss:20b",
        label="gpt-oss 20B",
        reasoning_effort="low",
        ollama_think="low",
        # Reasoning and final content share the generation budget in Ollama.
        # 2.25x gives enough room for low reasoning + compact JSON without
        # changing the public conversation length.
        num_predict_multiplier=2.25,
        # gpt-oss/Ollama structured-output combinations have historically been
        # more reliable when the schema is also explicit in the prompt.
        force_schema_grounding=True,
    ),
}


def get_profile(profile_id: str) -> ModelProfile:
    try:
        return PROFILES[profile_id]
    except KeyError as exc:
        raise ValueError(
            f"Unknown model profile '{profile_id}'. "
            f"Available: {', '.join(sorted(PROFILES))}"
        ) from exc


def profile_for_model(model: str) -> ModelProfile:
    for profile in PROFILES.values():
        if profile.model == model:
            return profile

    return ModelProfile(
        id="custom",
        model=model,
        label=model,
    )


def list_profiles() -> list[ModelProfile]:
    return list(PROFILES.values())
