from __future__ import annotations

import re
from collections import Counter

from app.factuality import unsupported_specifics
from app.feature_policy import build_feature_policy, count_feature_usage
from app.source_state import contains_unresolved_source_claim
from app.models import (
    ConversationResult,
    GenerationRequest,
    ImageAttachment,
    LinkAttachment,
    NoAttachment,
    ResearchFact,
    ValidationViolation,
)


WORD_RE = re.compile(r"\S+")
TOKEN_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ]+(?:['’][A-Za-zÀ-ÖØ-öø-ÿ]+)?")


# Deliberately conservative lint lists. These are not intended to be full
# language detectors. They catch common accidental code-switching without
# rejecting proper names or legitimate quotations.
FOREIGN_INTRUSIONS: dict[str, set[str]] = {
    "de": {
        "probably",
        "anyway",
        "actually",
        "basically",
        "literally",
        "whatever",
        "somehow",
        "randomly",
        "obviously",
        "hostage",
    },
    "en": {
        "wahrscheinlich",
        "eigentlich",
        "irgendwie",
        "jedenfalls",
        "offensichtlich",
    },
}


def word_count(text: str) -> int:
    return len(WORD_RE.findall(text))


def normalized_text(text: str) -> str:
    return " ".join(TOKEN_RE.findall(text.casefold()))


def find_foreign_intrusions(language: str, text: str) -> list[str]:
    forbidden = FOREIGN_INTRUSIONS.get(language.casefold(), set())
    if not forbidden:
        return []

    tokens = set(TOKEN_RE.findall(text.casefold()))
    return sorted(tokens.intersection(forbidden))


def _normalized_fact_value(value: str) -> str:
    value = value.casefold().replace("℃", "°c").replace("℉", "°f")
    value = value.replace(",", ".")
    return re.sub(r"\s+", "", value)


def _conversation_contains_fact_value(result: ConversationResult, value: str) -> bool:
    haystack = _normalized_fact_value(" ".join(entry.text for entry in result.entries))
    return _normalized_fact_value(value) in haystack


def validate_semantics(
    request: GenerationRequest,
    result: ConversationResult,
    factual_context: str = "",
    required_research_facts: list[ResearchFact] | None = None,
) -> list[ValidationViolation]:
    violations: list[ValidationViolation] = []
    participant_ids = {participant.id for participant in request.participants}
    participant_names = {
        participant.id: participant.name
        for participant in request.participants
    }
    feature_policy = build_feature_policy(request)

    if result.language != request.language:
        violations.append(
            ValidationViolation(
                code="LANGUAGE_CODE_MISMATCH",
                message=f"language must be '{request.language}'",
                expected=request.language,
                actual=result.language,
            )
        )

    if len(result.entries) != request.entry_count:
        violations.append(
            ValidationViolation(
                code="ENTRY_COUNT_MISMATCH",
                message=f"exactly {request.entry_count} entries are required",
                expected=request.entry_count,
                actual=len(result.entries),
            )
        )

    narration_indices: list[int] = []
    image_indices: list[int] = []
    link_indices: list[int] = []
    overlong_indices: list[int] = []
    invalid_speaker_indices: list[int] = []
    invalid_narrator_indices: list[int] = []
    invalid_typing_indices: list[int] = []
    invalid_attachment_indices: list[int] = []
    foreign_language_indices: list[int] = []
    foreign_language_hits: dict[int, list[str]] = {}
    excessive_caps_indices: list[int] = []
    leaked_typing_metadata_indices: list[int] = []
    invalid_link_intent_indices: list[int] = []
    unsupported_specificity_hits: dict[int, list[str]] = {}
    unresolved_source_claim_indices: list[int] = []
    redundant_speaker_prefix_indices: list[int] = []
    speakers = Counter()

    for index, entry in enumerate(result.entries):
        if word_count(entry.text) > request.limits.max_words_per_entry:
            overlong_indices.append(index)

        if entry.type == "narration":
            narration_indices.append(index)

            if entry.speaker_id != "narrator":
                invalid_narrator_indices.append(index)

            if entry.typing_indicator:
                invalid_typing_indices.append(index)

            if not isinstance(entry.attachment, NoAttachment):
                invalid_attachment_indices.append(index)

        elif entry.type == "message":
            if entry.speaker_id not in participant_ids:
                invalid_speaker_indices.append(index)
            else:
                speakers[entry.speaker_id] += 1

            if not request.features.typing_indicator and entry.typing_indicator:
                invalid_typing_indices.append(index)

        intrusions = find_foreign_intrusions(request.language, entry.text)
        if intrusions:
            foreign_language_indices.append(index)
            foreign_language_hits[index] = intrusions

        caps_words = [
            token
            for token in TOKEN_RE.findall(entry.text)
            if len(token) >= 5 and token.isupper()
        ]
        if len(caps_words) >= 2:
            excessive_caps_indices.append(index)

        if re.fullmatch(
            r"\s*[*_`]*\s*i\s*[*_`]*\s*:\s*(?:true|false)\s*[*_`]*\s*",
            entry.text,
            flags=re.IGNORECASE,
        ):
            leaked_typing_metadata_indices.append(index)

        if contains_unresolved_source_claim(request, entry.text):
            unresolved_source_claim_indices.append(index)

        if entry.type == "message":
            participant_name = participant_names.get(entry.speaker_id)
            if participant_name and re.match(
                rf"^\s*{re.escape(participant_name)}\s*[:：\-–—]\s*",
                entry.text,
                flags=re.IGNORECASE,
            ):
                redundant_speaker_prefix_indices.append(index)

        specificity_hits = unsupported_specifics(request, entry.text, factual_context)

        if isinstance(entry.attachment, LinkAttachment):
            specificity_hits.extend(
                unsupported_specifics(request, entry.attachment.query, factual_context)
            )
            specificity_hits.extend(
                unsupported_specifics(request, entry.attachment.label, factual_context)
            )

        if specificity_hits:
            unsupported_specificity_hits[index] = list(
                dict.fromkeys(specificity_hits)
            )

        if isinstance(entry.attachment, ImageAttachment):
            image_indices.append(index)
            if not request.features.images:
                invalid_attachment_indices.append(index)

        if isinstance(entry.attachment, LinkAttachment):
            link_indices.append(index)
            if not request.features.links:
                invalid_attachment_indices.append(index)

            if (
                re.search(
                    r"(?:https?://|www\.|\b[a-z0-9.-]+\.(?:com|org|net|ch|de|io|ai)\b)",
                    entry.attachment.query,
                    flags=re.IGNORECASE,
                )
                or re.search(
                    r"(?:https?://|www\.)",
                    entry.attachment.label,
                    flags=re.IGNORECASE,
                )
            ):
                invalid_link_intent_indices.append(index)

    usage = count_feature_usage(result)

    feature_checks = [
        (
            "NARRATION",
            usage.narrations,
            feature_policy.min_narrations,
            feature_policy.max_narrations,
            narration_indices,
        ),
        (
            "TYPING_INDICATOR",
            usage.typing_indicators,
            feature_policy.min_typing_indicators,
            feature_policy.max_typing_indicators,
            [
                index
                for index, entry in enumerate(result.entries)
                if entry.typing_indicator
            ],
        ),
        (
            "IMAGE",
            usage.images,
            feature_policy.min_images,
            feature_policy.max_images,
            image_indices,
        ),
        (
            "LINK",
            usage.links,
            feature_policy.min_links,
            feature_policy.max_links,
            link_indices,
        ),
    ]

    for label, actual, minimum, maximum, indices in feature_checks:
        if actual < minimum:
            violations.append(
                ValidationViolation(
                    code=f"MISSING_{label}",
                    message=(
                        f"{label.lower().replace('_', ' ')} is enabled and must occur "
                        f"at least {minimum} time(s); found {actual}"
                    ),
                    expected=minimum,
                    actual=actual,
                )
            )
        elif actual > maximum:
            violations.append(
                ValidationViolation(
                    code=f"TOO_MANY_{label}S",
                    message=(
                        f"maximum {maximum} {label.lower().replace('_', ' ')} "
                        f"occurrence(s) allowed; found {actual}"
                    ),
                    entry_indices=indices,
                    expected=maximum,
                    actual=actual,
                )
            )

    if overlong_indices:
        violations.append(
            ValidationViolation(
                code="ENTRY_TOO_LONG",
                message=(
                    f"every entry must contain at most "
                    f"{request.limits.max_words_per_entry} words"
                ),
                entry_indices=overlong_indices,
                expected=request.limits.max_words_per_entry,
            )
        )

    missing_participants = [
        participant.id
        for participant in request.participants
        if speakers[participant.id] == 0
    ]

    if missing_participants:
        violations.append(
            ValidationViolation(
                code="MISSING_PARTICIPANTS",
                message=(
                    "every participant must speak at least once; missing: "
                    + ", ".join(missing_participants)
                ),
            )
        )

    if invalid_speaker_indices:
        violations.append(
            ValidationViolation(
                code="INVALID_MESSAGE_SPEAKER",
                message="message entries must use a configured participant ID",
                entry_indices=invalid_speaker_indices,
            )
        )

    if invalid_narrator_indices:
        violations.append(
            ValidationViolation(
                code="INVALID_NARRATOR_ID",
                message="narration entries must use speaker_id='narrator'",
                entry_indices=invalid_narrator_indices,
            )
        )

    if invalid_typing_indices:
        violations.append(
            ValidationViolation(
                code="INVALID_TYPING_INDICATOR",
                message=(
                    "typing indicators are not permitted for these entries "
                    "or are globally disabled"
                ),
                entry_indices=sorted(set(invalid_typing_indices)),
            )
        )

    if invalid_attachment_indices:
        violations.append(
            ValidationViolation(
                code="INVALID_ATTACHMENT",
                message=(
                    "an attachment is disabled or is not allowed on a narration"
                ),
                entry_indices=sorted(set(invalid_attachment_indices)),
            )
        )

    if unresolved_source_claim_indices:
        violations.append(
            ValidationViolation(
                code="UNRESOLVED_SOURCE_CLAIM",
                message=(
                    "the conversation claims that a source was already opened, "
                    "read or resolved even though only a link intent exists"
                ),
                entry_indices=unresolved_source_claim_indices,
            )
        )

    if redundant_speaker_prefix_indices:
        violations.append(
            ValidationViolation(
                code="REDUNDANT_SPEAKER_PREFIX",
                message=(
                    "message text must not repeat the participant name because "
                    "speaker_id already identifies the speaker"
                ),
                entry_indices=redundant_speaker_prefix_indices,
            )
        )

    if unsupported_specificity_hits:
        details = "; ".join(
            f"{index}: {', '.join(values)}"
            for index, values in sorted(unsupported_specificity_hits.items())
        )
        violations.append(
            ValidationViolation(
                code="UNSUPPORTED_TECHNICAL_SPECIFICITY",
                message=(
                    "a technical model/version/code/value was introduced without "
                    "support in the request; keep unknown details generic or say "
                    "they need to be checked"
                ),
                entry_indices=sorted(unsupported_specificity_hits),
                actual=details,
            )
        )

    if invalid_link_intent_indices:
        violations.append(
            ValidationViolation(
                code="LINK_INTENT_CONTAINS_URL",
                message=(
                    "link intents must contain a resolver search query and label, "
                    "never a URL or domain"
                ),
                entry_indices=sorted(set(invalid_link_intent_indices)),
            )
        )

    if foreign_language_indices:
        details = "; ".join(
            f"{index}: {', '.join(foreign_language_hits[index])}"
            for index in foreign_language_indices
        )
        violations.append(
            ValidationViolation(
                code="FOREIGN_LANGUAGE_INTRUSION",
                message=(
                    f"unexpected foreign-language words were detected in "
                    f"{request.language}: {details}"
                ),
                entry_indices=foreign_language_indices,
            )
        )

    if excessive_caps_indices:
        violations.append(
            ValidationViolation(
                code="EXCESSIVE_ALL_CAPS",
                message="avoid multiple long ALL-CAPS words in a normal chat entry",
                entry_indices=excessive_caps_indices,
            )
        )

    if leaked_typing_metadata_indices:
        violations.append(
            ValidationViolation(
                code="TYPING_METADATA_IN_TEXT",
                message=(
                    "typing indicators are backend metadata and must never appear "
                    "as dialogue text"
                ),
                entry_indices=leaked_typing_metadata_indices,
            )
        )

    normalized_entries = [normalized_text(entry.text) for entry in result.entries]
    duplicate_indices: list[int] = []
    seen: dict[str, int] = {}

    for index, text in enumerate(normalized_entries):
        if not text:
            continue
        if text in seen:
            duplicate_indices.extend([seen[text], index])
        else:
            seen[text] = index

    if duplicate_indices:
        violations.append(
            ValidationViolation(
                code="DUPLICATE_ENTRY_TEXT",
                message="identical conversation entry text must not be repeated",
                entry_indices=sorted(set(duplicate_indices)),
            )
        )

    # Raw URLs must go through the later link-intent resolver.
    url_indices = [
        index
        for index, entry in enumerate(result.entries)
        if re.search(r"https?://|www\.", entry.text, flags=re.IGNORECASE)
    ]
    if url_indices:
        violations.append(
            ValidationViolation(
                code="RAW_URL_IN_TEXT",
                message="raw URLs are not allowed in generated entry text",
                entry_indices=url_indices,
            )
        )

    required_research_facts = required_research_facts or []
    missing_verified_facts = [
        fact for fact in required_research_facts
        if not _conversation_contains_fact_value(result, fact.value)
    ]
    for fact in missing_verified_facts:
        violations.append(
            ValidationViolation(
                code="MISSING_VERIFIED_RESEARCH_FACT",
                message=(
                    f"retrieval verified '{fact.aspect} = {fact.value}', but the conversation "
                    "does not use that requested fact; state the verified value naturally instead "
                    "of saying it is still unknown or needs another lookup"
                ),
                expected=f"{fact.aspect}: {fact.value}",
                actual="not mentioned",
            )
        )

    return violations
