from __future__ import annotations

from dataclasses import dataclass

from app.feature_policy import build_feature_policy
from app.models import ConversationResult, GenerationRequest


@dataclass(slots=True)
class FeatureAdjustmentCounts:
    typing_indicators_added: int = 0
    typing_indicators_removed: int = 0


def _typing_candidate_score(index: int, total: int, text: str) -> tuple[int, int]:
    """
    Prefer a substantial participant message around the middle of the conversation.
    The typing indicator is shown *before* that message, so avoiding the first and
    final entry generally produces more natural pacing.
    """
    edge_penalty = 0 if 0 < index < total - 1 else 1
    middle_distance = abs(index - (total // 2))
    # Sort ascending: non-edge first, then closer to middle, then longer text.
    return (edge_penalty * 1000 + middle_distance, -len(text))


def normalize_presentation_features(
    request: GenerationRequest,
    result: ConversationResult,
) -> FeatureAdjustmentCounts:
    """
    Normalize features that are presentation metadata rather than semantic content.

    Typing indicators are intentionally handled here rather than through an LLM
    repair pass. This is deterministic, cheap and avoids regenerating otherwise
    valid dialogue.
    """
    adjustments = FeatureAdjustmentCounts()
    policy = build_feature_policy(request)

    message_indices = [
        index
        for index, entry in enumerate(result.entries)
        if entry.type == "message"
    ]

    active = [
        index
        for index in message_indices
        if result.entries[index].typing_indicator
    ]

    if not request.features.typing_indicator:
        for index in active:
            result.entries[index].typing_indicator = False
            adjustments.typing_indicators_removed += 1
        return adjustments

    # If the model overuses the indicator, retain the most useful placements.
    if len(active) > policy.max_typing_indicators:
        ranked_active = sorted(
            active,
            key=lambda index: _typing_candidate_score(
                index,
                len(result.entries),
                result.entries[index].text,
            ),
        )
        keep = set(ranked_active[: policy.max_typing_indicators])

        for index in active:
            if index not in keep:
                result.entries[index].typing_indicator = False
                adjustments.typing_indicators_removed += 1

        active = sorted(keep)

    # If the model ignores the feature, add exactly the minimum required number.
    while len(active) < policy.min_typing_indicators:
        candidates = [
            index
            for index in message_indices
            if index not in active
        ]
        if not candidates:
            break

        best = min(
            candidates,
            key=lambda index: _typing_candidate_score(
                index,
                len(result.entries),
                result.entries[index].text,
            ),
        )
        result.entries[best].typing_indicator = True
        active.append(best)
        adjustments.typing_indicators_added += 1

    return adjustments
