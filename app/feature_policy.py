from __future__ import annotations

from dataclasses import dataclass

from app.models import ConversationResult, GenerationRequest, ImageAttachment, LinkAttachment


@dataclass(frozen=True, slots=True)
class FeaturePolicy:
    min_narrations: int
    max_narrations: int
    min_typing_indicators: int
    max_typing_indicators: int
    min_images: int
    max_images: int
    min_links: int
    max_links: int


@dataclass(frozen=True, slots=True)
class FeatureUsageCounts:
    narrations: int
    typing_indicators: int
    images: int
    links: int


def build_feature_policy(request: GenerationRequest) -> FeaturePolicy:
    # "Enabled" means the user asked for the feature to occur, not merely that
    # the model is permitted to use it.
    min_narrations = 1 if request.features.narration else 0
    max_narrations = request.limits.max_narrations if request.features.narration else 0

    if request.features.typing_indicator:
        # Enough to make the feature visible without turning every message into
        # an artificial pause.
        message_capacity = request.entry_count - min_narrations
        max_typing = max(1, min(3, message_capacity))
        min_typing = 1
    else:
        min_typing = 0
        max_typing = 0

    return FeaturePolicy(
        min_narrations=min_narrations,
        max_narrations=max_narrations,
        min_typing_indicators=min_typing,
        max_typing_indicators=max_typing,
        min_images=1 if request.features.images else 0,
        max_images=request.limits.max_images if request.features.images else 0,
        min_links=1 if request.features.links else 0,
        max_links=request.limits.max_links if request.features.links else 0,
    )


def count_feature_usage(result: ConversationResult) -> FeatureUsageCounts:
    narrations = 0
    typing_indicators = 0
    images = 0
    links = 0

    for entry in result.entries:
        if entry.type == "narration":
            narrations += 1
        if entry.typing_indicator:
            typing_indicators += 1
        if isinstance(entry.attachment, ImageAttachment):
            images += 1
        elif isinstance(entry.attachment, LinkAttachment):
            links += 1

    return FeatureUsageCounts(
        narrations=narrations,
        typing_indicators=typing_indicators,
        images=images,
        links=links,
    )
