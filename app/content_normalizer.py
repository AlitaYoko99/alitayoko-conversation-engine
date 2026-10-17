from __future__ import annotations

import re
from dataclasses import dataclass
from collections import Counter

from app.factuality import (
    has_unsupported_specificity,
    neutral_factual_text,
    unsupported_specifics,
)
from app.feature_policy import build_feature_policy, count_feature_usage
from app.source_state import (
    contains_unresolved_source_claim,
    contains_unsupported_absence_claim,
    contains_unsupported_source_mechanism,
    neutral_absence_text,
    neutral_source_mechanism_text,
    neutral_source_text,
)
from app.models import (
    ConversationEntry,
    ConversationResult,
    GenerationRequest,
    ImageAttachment,
    LinkAttachment,
    NoAttachment,
)


WORD_RE = re.compile(r"\S+")
MARKUP_RE = re.compile(r"[*_`]+")


@dataclass(slots=True)
class ContentAdjustmentCounts:
    images_added: int = 0
    narrations_added: int = 0
    links_added: int = 0
    links_sanitized: int = 0
    links_removed: int = 0
    technical_specifics_neutralized: int = 0
    source_claims_neutralized: int = 0
    unsupported_absence_claims_neutralized: int = 0
    unsupported_source_mechanisms_neutralized: int = 0
    dangling_value_references_rewritten: int = 0
    speaker_prefixes_removed: int = 0
    wire_artifacts_removed: int = 0
    attachment_queries_refined: int = 0
    normalization_duplicates_avoided: int = 0
    entries_shortened: int = 0


def _word_count(text: str) -> int:
    return len(WORD_RE.findall(text))


def _strip_markup(text: str) -> str:
    text = MARKUP_RE.sub("", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _trim_to_words(text: str, max_words: int) -> str:
    cleaned = _strip_markup(text)
    words = WORD_RE.findall(cleaned)
    if len(words) <= max_words:
        return cleaned

    trimmed = words[:max_words]
    if trimmed:
        # Avoid dangling punctuation-only final tokens when possible.
        while trimmed and re.fullmatch(r"[.,;:!?…\-–—]+", trimmed[-1]):
            trimmed.pop()
        if not trimmed:
            trimmed = words[:max_words]

    result = " ".join(trimmed).strip()
    return result if result else cleaned



WIRE_ARTIFACT_RE = re.compile(
    r'(?:^|\s)(?:a|i|l|x|q|k)\s*:\s*\{?\s*$',
    flags=re.IGNORECASE,
)

GERMAN_QUERY_BAD_WORDS = {
    "dabei", "entdecken", "vergleichen", "gerät", "schnell", "sicher",
    "weitere", "informationen", "offizielle", "finden", "findenwir",
}

def strip_wire_artifacts(
    request: GenerationRequest,
    result: ConversationResult,
) -> int:
    """
    Remove compact wire-format fragments accidentally leaked into message text,
    for example trailing `a:{` or `i:true`.
    """
    removed = 0

    for entry in result.entries:
        original = entry.text
        cleaned = WIRE_ARTIFACT_RE.sub("", original).strip()

        # Common inline booleans from compact format.
        cleaned = re.sub(
            r'\s*(?:i\s*:\s*(?:true|false)|t\s*:\s*".*?)\s*$',
            "",
            cleaned,
            flags=re.IGNORECASE,
        ).strip()

        if cleaned and cleaned != original:
            entry.text = cleaned
            removed += 1

    return removed


def _best_subject_query_from_context(
    request: GenerationRequest,
    entry_text: str,
) -> str:
    """
    Build a short, stable resolver query from the current request/topic and the
    local message context. Goal: better than random topic fragments, but still
    deterministic and generic.
    """
    language = request.language.casefold().split("-", 1)[0]
    text = _strip_markup(entry_text)
    topic = _strip_markup(request.topic)
    combined = f"{text} {topic}"

    tokens = re.findall(
        r"[A-Za-zÀ-ÖØ-öø-ÿ0-9]+(?:[-‑–—][A-Za-zÀ-ÖØ-öø-ÿ0-9]+)*",
        combined,
        flags=re.UNICODE,
    )

    participant_names = {p.name.casefold() for p in request.participants}
    chosen: list[str] = []
    seen: set[str] = set()

    for token in tokens:
        folded = token.casefold()
        if folded in participant_names:
            continue
        if language == "de" and folded in GERMAN_STOPWORDS:
            continue
        if folded in GERMAN_QUERY_BAD_WORDS:
            continue
        if len(folded) <= 2 and not any(ch.isdigit() for ch in folded):
            continue
        if folded in seen:
            continue
        seen.add(folded)
        chosen.append(token)
        if len(chosen) >= 3:
            break

    query = " ".join(chosen).strip()

    # Prefer a generic but strong fallback.
    if not query:
        if "3d" in topic.casefold() or "druck" in topic.casefold():
            query = "3D-Drucker"
        else:
            query = _topic_search_terms(request)

    return query[:80].rstrip()


def refine_attachment_queries(
    request: GenerationRequest,
    result: ConversationResult,
) -> int:
    """
    Improve link queries that are structurally valid but low quality.
    Example:
      '3D-Drucker vergleichen Gerät entdecken dabei'
    becomes:
      '3D-Drucker Bedienungsanleitung'
    """
    refined = 0

    for entry in result.entries:
        attachment = entry.attachment
        if not isinstance(attachment, LinkAttachment):
            continue

        query = _strip_markup(attachment.query).strip()
        lowered = query.casefold()
        tokens = query.split()

        needs_refinement = (
            len(tokens) > 5
            or any(word in lowered for word in GERMAN_QUERY_BAD_WORDS)
            or lowered.startswith("offizielle ")
            or lowered.endswith(" modell")
        )

        if not needs_refinement:
            continue

        _, label, suffix = _link_hint(entry.text, request.language)
        subject = _best_subject_query_from_context(request, entry.text)
        new_query = f"{subject} {suffix}".strip()

        if new_query != attachment.query:
            attachment.query = new_query[:120].rstrip()
            attachment.label = label
            refined += 1

    return refined


NARRATION_FALLBACKS: dict[str, tuple[str, ...]] = {
    "de": (
        "Beide gehen ihre bisherigen Beobachtungen noch einmal gemeinsam durch, bevor sie weitermachen.",
        "Für einen Moment prüfen beide den bisherigen Stand und überlegen den nächsten Schritt.",
        "Kurz vergleichen beide ihre bisherigen Beobachtungen, bevor sie die nächsten Schritte angehen.",
    ),
    "en": (
        "They briefly review what they have observed so far before continuing.",
        "For a moment, they compare the current situation and consider the next step.",
        "They pause to review their observations before moving on.",
    ),
    "fr": (
        "Ils reprennent brièvement leurs observations avant de poursuivre.",
        "Un instant, ils font le point et réfléchissent à l’étape suivante.",
        "Ils comparent brièvement leurs observations avant de continuer.",
    ),
    "it": (
        "Rivedono brevemente ciò che hanno osservato finora prima di proseguire.",
        "Per un momento fanno il punto e valutano il passo successivo.",
        "Confrontano brevemente le loro osservazioni prima di continuare.",
    ),
    "rm": (
        "Els examineschan curtamain lur observaziuns avant che cuntinuar.",
        "Per in mument fan els il punct e pondereschan il proxim pass.",
        "Els cumpareglian curtamain lur observaziuns avant che cuntinuar.",
    ),
}


def _fallback_narration_text(
    request: GenerationRequest,
    index: int,
) -> str:
    language = request.language.casefold().split("-", 1)[0]
    variants = NARRATION_FALLBACKS.get(language, NARRATION_FALLBACKS["en"])

    text = variants[index % len(variants)]
    if len(request.participants) != 2:
        # Keep the wording correct for multi-participant conversations.
        replacements = {
            "Beide": "Die Beteiligten",
            "beide": "die Beteiligten",
            "They": "The participants",
            "they": "the participants",
            "Ils": "Les participants",
            "ils": "les participants",
            "Rivedono": "I partecipanti rivedono",
            "fanno": "i partecipanti fanno",
            "Confrontano": "I partecipanti confrontano",
            "Els": "Ils participants",
            "els": "ils participants",
        }
        for old, new in replacements.items():
            if text.startswith(old):
                text = new + text[len(old):]
                break
    return text


def add_missing_narration(
    request: GenerationRequest,
    result: ConversationResult,
) -> int:
    """
    Guarantee the requested minimum narration without changing entry_count.

    A short, attachment-free message from a participant who speaks more than
    once is replaced with a neutral scene transition. This preserves every
    participant and avoids losing image/link intents.
    """
    policy = build_feature_policy(request)
    usage = count_feature_usage(result)

    if not request.features.narration or usage.narrations >= policy.min_narrations:
        return 0

    speaker_counts = Counter(
        entry.speaker_id
        for entry in result.entries
        if entry.type == "message"
    )

    candidates: list[int] = []
    for index, entry in enumerate(result.entries):
        if entry.type != "message":
            continue
        if speaker_counts[entry.speaker_id] <= 1:
            continue
        if not isinstance(entry.attachment, NoAttachment):
            continue
        candidates.append(index)

    if not candidates:
        return 0

    # Prefer a short, non-edge message near the middle. This usually replaces
    # acknowledgements/filler rather than a substantive or concluding line.
    best_index = min(
        candidates,
        key=lambda index: (
            1 if index in {0, len(result.entries) - 1} else 0,
            _word_count(result.entries[index].text),
            abs(index - (len(result.entries) // 2)),
        ),
    )

    result.entries[best_index] = ConversationEntry(
        type="narration",
        speaker_id="narrator",
        text=_fallback_narration_text(request, best_index),
        typing_indicator=False,
        attachment=NoAttachment(),
    )
    return 1


def remove_redundant_speaker_prefixes(
    request: GenerationRequest,
    result: ConversationResult,
) -> int:
    """
    speaker_id already identifies the participant. Remove model-written
    prefixes such as "Lukas:" or "Markus:" from message text.
    """
    removed = 0
    names_by_id = {
        participant.id: participant.name
        for participant in request.participants
    }

    for entry in result.entries:
        if entry.type != "message":
            continue

        name = names_by_id.get(entry.speaker_id)
        if not name:
            continue

        pattern = re.compile(
            rf"^\s*{re.escape(name)}\s*[:：\-–—]\s*",
            flags=re.IGNORECASE,
        )
        new_text = pattern.sub("", entry.text, count=1).strip()

        if new_text and new_text != entry.text:
            entry.text = new_text
            removed += 1

    return removed


NORMALIZATION_TOKEN_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ]+(?:['’][A-Za-zÀ-ÖØ-öø-ÿ]+)?")


def _normalization_key(text: str) -> str:
    return " ".join(NORMALIZATION_TOKEN_RE.findall(text.casefold()))


def _alpha_label(index: int) -> str:
    """Return A..Z, AA..AZ, BA.. style labels for rare fallback cases."""
    value = index + 1
    chars: list[str] = []
    while value:
        value, remainder = divmod(value - 1, 26)
        chars.append(chr(ord("A") + remainder))
    return "".join(reversed(chars))


def _unique_neutral_text(
    request: GenerationRequest,
    result: ConversationResult,
    index: int,
    candidate_factory,
    *,
    entry_type: str = "message",
    kind: str = "factual",
) -> tuple[str, bool]:
    """
    Pick a neutralization phrase that is not already present in the
    conversation. The bool indicates that the preferred index-based variant
    collided and an alternate was selected.
    """
    used = {
        _normalization_key(entry.text)
        for other_index, entry in enumerate(result.entries)
        if other_index != index and entry.text.strip()
    }

    preferred = candidate_factory(index)
    if _normalization_key(preferred) not in used:
        return preferred, False

    # Existing neutral-text helpers currently rotate over a small wording pool.
    # Probe a generous range; repeated values are skipped by the key check.
    seen_candidates: set[str] = set()
    for offset in range(1, 24):
        candidate = candidate_factory(index + offset)
        key = _normalization_key(candidate)
        if key in seen_candidates:
            continue
        seen_candidates.add(key)
        if key not in used:
            return candidate, True

    # Extremely defensive fallback: keep the sentence natural while adding a
    # stable alphabetic check-point label. This guarantees that normalization
    # itself can never create DUPLICATE_ENTRY_TEXT, even for very long chats.
    label = _alpha_label(index)
    language = request.language.casefold().split("-", 1)[0]

    if kind == "source":
        if entry_type == "narration":
            fallbacks = {
                "de": f"Die Quelle zu Prüfpunkt {label} bleibt zunächst ungeklärt.",
                "en": f"The source for check point {label} remains unresolved for now.",
                "fr": f"La source du point de contrôle {label} reste à vérifier.",
                "it": f"La fonte per il punto di verifica {label} resta da controllare.",
                "rm": f"La funtauna per il punct da controlla {label} resta anc averta.",
            }
        else:
            fallbacks = {
                "de": f"Diese Quellenangabe bleibt bis zur Prüfung als Prüfpunkt {label} offen.",
                "en": f"This source claim remains open as check point {label} until verified.",
                "fr": f"Cette indication de source reste ouverte comme point de contrôle {label} jusqu’à vérification.",
                "it": f"Questa indicazione della fonte resta aperta come punto di verifica {label} finché non viene controllata.",
                "rm": f"Questa indicaziun da funtauna resta averta sco punct da controlla {label} fin ch’ella è verifitgada.",
            }
    else:
        fallbacks = {
            "de": f"Dieses technische Detail bleibt bis zur Prüfung als Prüfpunkt {label} offen.",
            "en": f"This technical detail remains open as check point {label} until verified.",
            "fr": f"Ce détail technique reste ouvert comme point de contrôle {label} jusqu’à vérification.",
            "it": f"Questo dettaglio tecnico resta aperto come punto di verifica {label} finché non viene controllato.",
            "rm": f"Quest detagl tecnic resta avert sco punct da controlla {label} fin ch’el è verifitgà.",
        }

    return fallbacks.get(language, fallbacks["en"]), True




def deduplicate_exact_entry_texts(
    request: GenerationRequest,
    result: ConversationResult,
) -> int:
    """Replace later exact/normalized duplicate utterances deterministically.

    A duplicate already stated once carries no additional factual payload, so
    the later occurrence can safely become a short acknowledgement. This keeps
    the requested entry count and speaker sequence intact while preventing an
    otherwise unnecessary LLM repair cycle.
    """
    language = request.language.casefold().split("-", 1)[0]
    message_variants = {
        "de": (
            "Genau, das halten wir so fest.",
            "Damit ist dieser Punkt festgehalten.",
            "Einverstanden, dabei bleiben wir.",
            "Das ist damit klar festgehalten.",
        ),
        "en": (
            "Exactly, we can keep that as established.",
            "That point is now clearly established.",
            "Agreed, we can leave it at that.",
            "That is now clearly recorded.",
        ),
        "fr": (
            "Exactement, nous retenons ce point.",
            "Ce point est maintenant clairement établi.",
            "D'accord, nous en restons là.",
            "C'est maintenant clairement établi.",
        ),
        "it": (
            "Esatto, teniamo fermo questo punto.",
            "Questo punto è ora chiaramente stabilito.",
            "D'accordo, restiamo su questo punto.",
            "Ora è chiaramente stabilito.",
        ),
        "rm": (
            "Exact, nus tegnain ferm quest punct.",
            "Quest punct è ussa cleramain fixà.",
            "D'accord, nus restain tar quai.",
            "Quai è ussa cleramain fixà.",
        ),
    }
    narration_variants = {
        "de": (
            "Der Punkt bleibt damit festgehalten.",
            "Damit ist dieser Punkt geklärt.",
            "Die Feststellung bleibt bestehen.",
        ),
        "en": (
            "The point remains established.",
            "That settles this point.",
            "The finding stands.",
        ),
        "fr": (
            "Le point reste ainsi établi.",
            "Ce point est donc réglé.",
            "Le constat reste valable.",
        ),
        "it": (
            "Il punto resta così stabilito.",
            "Questo chiarisce il punto.",
            "La constatazione resta valida.",
        ),
        "rm": (
            "Il punct resta uschia fixà.",
            "Uschia è quest punct sclerì.",
            "La constataziun resta valaivla.",
        ),
    }

    seen: set[str] = set()
    changed = 0
    for index, entry in enumerate(result.entries):
        key = _normalization_key(entry.text)
        if not key:
            continue
        if key not in seen:
            seen.add(key)
            continue

        variants_by_language = (
            narration_variants if entry.type == "narration" else message_variants
        )
        variants = variants_by_language.get(
            language, variants_by_language["en"]
        )
        replacement = None
        for offset in range(len(variants)):
            candidate = variants[(index + offset) % len(variants)]
            candidate_key = _normalization_key(candidate)
            if candidate_key not in seen:
                replacement = candidate
                break

        if replacement is None:
            # Extremely rare case: every stock acknowledgement is already in
            # the conversation. Keep the wording natural and make it unique.
            label = _alpha_label(index)
            if language == "de":
                replacement = f"Damit ist auch Prüfpunkt {label} festgehalten."
            elif language == "fr":
                replacement = f"Le point de contrôle {label} est ainsi établi."
            elif language == "it":
                replacement = f"Il punto di verifica {label} è così stabilito."
            elif language == "rm":
                replacement = f"Il punct da controlla {label} è uschia fixà."
            else:
                replacement = f"Check point {label} is now established."

        entry.text = _trim_to_words(replacement, request.limits.max_words_per_entry)
        seen.add(_normalization_key(entry.text))
        changed += 1

    return changed


def neutralize_unresolved_source_claims(
    request: GenerationRequest,
    result: ConversationResult,
) -> tuple[int, int]:
    """
    Link/image intents are not resolved resources. Until a future Research or
    Resolver layer supplies evidence, the conversation may suggest consulting
    sources but cannot claim that a source was already opened/read/resolved.

    Returns (neutralized_count, duplicate_avoidance_count).
    """
    neutralized = 0
    duplicate_avoidance = 0

    for index, entry in enumerate(result.entries):
        if contains_unresolved_source_claim(request, entry.text):
            replacement, avoided = _unique_neutral_text(
                request,
                result,
                index,
                lambda variant: neutral_source_text(
                    request,
                    entry_type=entry.type,
                    variant=variant,
                ),
                entry_type=entry.type,
                kind="source",
            )
            entry.text = replacement
            neutralized += 1
            duplicate_avoidance += int(avoided)

    return neutralized, duplicate_avoidance


def neutralize_unsupported_authority_claims(
    request: GenerationRequest,
    result: ConversationResult,
    factual_context: str = "",
) -> int:
    """Remove unsupported claims that evidence came from an official source.

    Facts may still be supported by secondary evidence. What must not survive is
    an attribution such as "Nintendo itself says" or "official specification"
    when the research bundle contains no source marked as primary authority.
    """
    if not request.features.research or not factual_context:
        return 0
    if re.search(r"\bAuthority:\s*primary\b", factual_context, flags=re.IGNORECASE):
        return 0

    language = request.language.casefold().split("-", 1)[0]
    changed = 0
    if language == "de":
        substitutions = (
            (
                re.compile(r"\bwie\s+[A-ZÄÖÜ][\wÄÖÜäöüß.-]*\s+selbst\s+angibt\b", re.IGNORECASE),
                "wie die geprüften Quellen angeben",
            ),
            (
                re.compile(
                    r"\blaut\s+(?:der\s+)?(?:offiziellen?|herstellerseitigen?)\s+"
                    r"(?:spezifikation(?:en)?|dokumentation|angabe(?:n)?|datenblatt(?:es)?)\b",
                    re.IGNORECASE,
                ),
                "laut den geprüften Quellen",
            ),
            (
                re.compile(r"\b(?:vom|durch den|durch die)\s+hersteller\w*\s+(?:bestätigt|dokumentiert|angegeben)\b", re.IGNORECASE),
                "durch die geprüften Quellen belegt",
            ),
        )
    else:
        substitutions = (
            (
                re.compile(r"\bas\s+[A-Z][\w.-]*\s+itself\s+(?:states|says)\b", re.IGNORECASE),
                "as the checked sources state",
            ),
            (
                re.compile(r"\baccording to (?:the )?(?:official|manufacturer)\s+(?:specification|documentation|datasheet)\b", re.IGNORECASE),
                "according to the checked sources",
            ),
        )

    for entry in result.entries:
        original = entry.text
        updated = original
        for pattern, replacement in substitutions:
            updated = pattern.sub(replacement, updated)
        # A broad final cleanup prevents bare "official" attribution from
        # surviving when there is no primary source at all.
        updated = re.sub(r"\boffizielle[nrms]?\s+(?:quelle|quellen|angabe|angaben|spezifikation|spezifikationen|dokumentation)\b", "geprüfte Quellen", updated, flags=re.IGNORECASE)
        if updated != original:
            entry.text = re.sub(r"\s+", " ", updated).strip()
            changed += 1
    return changed


def neutralize_unsupported_absence_claims(
    request: GenerationRequest,
    result: ConversationResult,
    factual_context: str = "",
) -> tuple[int, int]:
    """Neutralize unsupported claims that a searched source lacks a fact."""
    neutralized = 0
    duplicate_avoidance = 0

    for index, entry in enumerate(result.entries):
        if contains_unsupported_absence_claim(request, entry.text, factual_context):
            replacement, avoided = _unique_neutral_text(
                request,
                result,
                index,
                lambda variant: neutral_absence_text(
                    request,
                    entry_type=entry.type,
                    variant=variant,
                ),
                entry_type=entry.type,
                kind="source",
            )
            entry.text = replacement
            neutralized += 1
            duplicate_avoidance += int(avoided)

    return neutralized, duplicate_avoidance


def neutralize_unsupported_technical_specifics(
    request: GenerationRequest,
    result: ConversationResult,
    factual_context: str = "",
) -> tuple[int, int]:
    """
    Replace only entries that contain high-confidence unsupported technical
    specifics. The rest of the generated conversation stays untouched.

    Returns (neutralized_count, duplicate_avoidance_count).
    """
    neutralized = 0
    duplicate_avoidance = 0

    for index, entry in enumerate(result.entries):
        if unsupported_specifics(request, entry.text, factual_context):
            replacement, avoided = _unique_neutral_text(
                request,
                result,
                index,
                lambda variant: neutral_factual_text(
                    request,
                    entry.text,
                    variant=variant,
                ),
                entry_type=entry.type,
                kind="factual",
            )
            entry.text = replacement
            neutralized += 1
            duplicate_avoidance += int(avoided)

    return neutralized, duplicate_avoidance


def shorten_overlong_entries(
    request: GenerationRequest,
    result: ConversationResult,
) -> int:
    shortened = 0
    max_words = request.limits.max_words_per_entry

    for entry in result.entries:
        if _word_count(entry.text) > max_words:
            new_text = _trim_to_words(entry.text, max_words)
            if new_text != entry.text:
                entry.text = new_text
                shortened += 1

    return shortened


def _entry_visual_score(index: int, total: int, entry: ConversationEntry) -> tuple[int, int]:
    """
    Rank eligible participant-message anchors. Attachments are never allowed on
    narration entries by the public contract.
    """
    middle_distance = abs(index - (total // 2))
    text_length_rank = -len(_strip_markup(entry.text))
    return (middle_distance, text_length_rank)


def _make_image_attachment(
    anchor: ConversationEntry,
    visual_source: ConversationEntry,
) -> ImageAttachment:
    base = _strip_markup(visual_source.text)

    if visual_source.type == "narration":
        description = f"Illustration der beschriebenen Szene: {base}"
    else:
        description = f"Illustration eines Moments aus der Unterhaltung: {base}"

    if len(description) > 500:
        description = description[:497].rstrip() + "..."

    words = WORD_RE.findall(base)
    alt = " ".join(words[:8]).strip() if words else "Szene aus der Unterhaltung"
    if not alt:
        alt = "Szene aus der Unterhaltung"
    if len(alt) > 250:
        alt = alt[:247].rstrip() + "..."

    return ImageAttachment(
        type="image",
        description=description,
        alt_text=alt,
    )


def add_missing_images(
    request: GenerationRequest,
    result: ConversationResult,
) -> int:
    policy = build_feature_policy(request)
    usage = count_feature_usage(result)

    if not request.features.images or usage.images >= policy.min_images:
        return 0

    # The attachment anchor must always be a participant message. Narrations are
    # deliberately excluded because the public contract forbids attachments on
    # narration entries.
    candidates: list[tuple[tuple[int, int], int]] = []

    for index, entry in enumerate(result.entries):
        if (
            entry.type == "message"
            and isinstance(entry.attachment, NoAttachment)
        ):
            candidates.append(
                (_entry_visual_score(index, len(result.entries), entry), index)
            )

    if not candidates:
        return 0

    candidates.sort()
    _, best_index = candidates[0]
    anchor = result.entries[best_index]

    # For the image description, a nearby narration is often more visual than
    # the message itself. It may inspire the intent without receiving the
    # attachment.
    visual_source = anchor
    narration_candidates = [
        (abs(index - best_index), entry)
        for index, entry in enumerate(result.entries)
        if entry.type == "narration"
    ]
    if narration_candidates:
        narration_candidates.sort(key=lambda item: item[0])
        distance, nearest_narration = narration_candidates[0]
        if distance <= 2:
            visual_source = nearest_narration

    anchor.attachment = _make_image_attachment(anchor, visual_source)
    return 1



GERMAN_STOPWORDS = {
    # Function words
    "aber", "als", "auch", "auf", "aus", "bei", "bevor", "bis", "das", "dass",
    "dem", "den", "der", "des", "die", "ein", "eine", "einen", "einer", "einem",
    "eines", "er", "es", "für", "haben", "hat", "ihr", "ihre", "ihren", "im",
    "in", "ist", "mit", "nach", "nicht", "noch", "oder", "sich", "sie", "sind",
    "und", "von", "vor", "warum", "was", "wenn", "wie", "wir", "zu", "zum",
    "zur", "über",
    # Procedural filler that is rarely useful in a resolver query
    "immer", "wieder", "versuchen", "herauszufinden", "richten", "einrichten",
    "gebraucht", "gebrauchte", "gebrauchten", "möchten", "wollen", "sollen",
    "sollten", "zuerst", "genau", "genaue", "genauen", "modell", "modells",
    "einstellungen", "verändern", "ändern", "nachschlagen", "suchen", "prüfen",
    "offiziell", "offizielle", "offiziellen",
    # Resource words are added intentionally from the selected link hint
    "bedienungsanleitung", "anleitung", "handbuch", "manual",
    "technisch", "technische", "technischen", "daten", "datenblatt",
    "spezifikation", "spezifikationen",
}


LINK_HINTS_DE: tuple[tuple[tuple[str, ...], str, str, int], ...] = (
    (
        ("bedienungsanleitung", "anleitung", "handbuch", "manual"),
        "Bedienungsanleitung",
        "Bedienungsanleitung",
        7,
    ),
    (
        ("datenblatt", "technische daten", "spezifikation", "spezifikationen"),
        "Technische Daten",
        "technische Daten",
        6,
    ),
    (
        ("hersteller", "support", "service", "hilfe"),
        "Herstellerinformationen",
        "Hersteller Support",
        5,
    ),
    (
        ("vergleich", "test", "review"),
        "Test und Vergleich",
        "Test Vergleich",
        4,
    ),
    (
        ("nachsehen", "nachschauen", "nachlesen", "recherchieren", "suchen", "prüfen"),
        "Weitere Informationen",
        "",
        2,
    ),
)


def _topic_search_terms(request: GenerationRequest) -> str:
    """
    Build compact subject terms for the future resolver.

    Resource words such as "Bedienungsanleitung" are intentionally excluded
    here; `_make_link_attachment` adds the resource suffix selected from the
    actual message context.
    """
    cleaned = _strip_markup(request.topic)
    # Keep normal compound/model hyphens but drop sentence punctuation.
    tokens = re.findall(
        r"[A-Za-zÀ-ÖØ-öø-ÿ0-9]+(?:[-‑–—][A-Za-zÀ-ÖØ-öø-ÿ0-9]+)*",
        cleaned,
        flags=re.UNICODE,
    )
    participant_names = {p.name.casefold() for p in request.participants}

    selected: list[str] = []
    seen: set[str] = set()

    for token in tokens:
        folded = token.casefold()
        if not folded or folded in participant_names:
            continue
        if request.language.casefold().startswith("de") and folded in GERMAN_STOPWORDS:
            continue
        if len(folded) <= 2 and not any(ch.isdigit() for ch in folded):
            continue
        if folded in seen:
            continue

        seen.add(folded)
        selected.append(token)
        if len(selected) >= 5:
            break

    if not selected:
        selected = [
            token for token in tokens
            if token.casefold() not in participant_names
        ][:5]

    query = " ".join(selected).strip()
    return query[:120].rstrip()


def _link_hint(text: str, language: str) -> tuple[int, str, str]:
    folded = _strip_markup(text).casefold()

    if language.casefold().startswith("de"):
        for keywords, label, suffix, weight in LINK_HINTS_DE:
            if any(keyword in folded for keyword in keywords):
                return weight, label, suffix

    return 0, "Weitere Informationen", ""


def _link_candidate_score(
    index: int,
    total: int,
    entry: ConversationEntry,
    language: str,
) -> tuple[int, int, int, int]:
    hint_weight, _, _ = _link_hint(entry.text, language)
    edge_penalty = 1 if index in {0, total - 1} else 0
    question_bonus = 1 if "?" in entry.text else 0
    middle_distance = abs(index - (total // 2))

    # Sort ascending: stronger semantic hint first, then questions, non-edge
    # messages, proximity to the middle.
    return (
        -hint_weight,
        -question_bonus,
        edge_penalty,
        middle_distance,
    )


def _make_link_attachment(
    request: GenerationRequest,
    anchor: ConversationEntry,
) -> LinkAttachment:
    _, label, suffix = _link_hint(anchor.text, request.language)
    query = _topic_search_terms(request)

    if suffix:
        query = f"{query} {suffix}".strip()

    if not query:
        query = _strip_markup(request.topic)[:180].rstrip()

    return LinkAttachment(
        type="link",
        query=query[:180].rstrip(),
        label=label[:80].rstrip(),
    )



URL_OR_DOMAIN_RE = re.compile(
    r"(?:https?://|www\.|\b[a-z0-9.-]+\.(?:com|org|net|ch|de|io|ai|dev|app)\b)",
    flags=re.IGNORECASE,
)


def _contains_url_or_domain(text: str) -> bool:
    return bool(URL_OR_DOMAIN_RE.search(text))


def _canonical_link_for_entry(
    request: GenerationRequest,
    entry: ConversationEntry,
) -> LinkAttachment:
    """
    Build a safe resolver intent from the conversation context. This is used
    whenever a model-provided link contains a URL/domain or otherwise needs to
    be canonicalized.
    """
    return _make_link_attachment(request, entry)


def normalize_existing_links(
    request: GenerationRequest,
    result: ConversationResult,
    factual_context: str = "",
) -> tuple[int, int]:
    """
    Canonicalize model-provided link intents before semantic validation.

    Returns:
        (sanitized_count, removed_count)
    """
    if not request.features.links:
        return (0, 0)

    policy = build_feature_policy(request)
    link_indices = [
        index
        for index, entry in enumerate(result.entries)
        if isinstance(entry.attachment, LinkAttachment)
    ]

    sanitized = 0
    removed = 0

    # First sanitize unsafe/resolver-hostile intents in place.
    for index in link_indices:
        entry = result.entries[index]
        attachment = entry.attachment
        assert isinstance(attachment, LinkAttachment)

        unsafe = (
            _contains_url_or_domain(attachment.query)
            or _contains_url_or_domain(attachment.label)
            or has_unsupported_specificity(request, attachment.query, factual_context)
            or has_unsupported_specificity(request, attachment.label, factual_context)
        )

        # Queries should be concise resolver input, not prose copied from a
        # model answer. Long intents are normalized even when they contain no
        # literal URL.
        overly_verbose = (
            len(attachment.query) > 180
            or len(attachment.label) > 80
            or len(attachment.query.split()) > 18
        )

        if unsafe or overly_verbose:
            entry.attachment = _canonical_link_for_entry(request, entry)
            sanitized += 1

    # Then cap overproduction deterministically. Keep the links attached to the
    # most contextually appropriate participant messages.
    link_indices = [
        index
        for index, entry in enumerate(result.entries)
        if isinstance(entry.attachment, LinkAttachment)
    ]

    if len(link_indices) > policy.max_links:
        ranked = sorted(
            link_indices,
            key=lambda index: _link_candidate_score(
                index,
                len(result.entries),
                result.entries[index],
                request.language,
            ),
        )
        keep = set(ranked[: policy.max_links])

        for index in link_indices:
            if index not in keep:
                result.entries[index].attachment = NoAttachment()
                removed += 1

    return (sanitized, removed)


def add_missing_links(
    request: GenerationRequest,
    result: ConversationResult,
) -> int:
    policy = build_feature_policy(request)
    usage = count_feature_usage(result)

    if not request.features.links or usage.links >= policy.min_links:
        return 0

    candidates: list[int] = [
        index
        for index, entry in enumerate(result.entries)
        if (
            entry.type == "message"
            and isinstance(entry.attachment, NoAttachment)
        )
    ]

    if not candidates:
        return 0

    best_index = min(
        candidates,
        key=lambda index: _link_candidate_score(
            index,
            len(result.entries),
            result.entries[index],
            request.language,
        ),
    )

    anchor = result.entries[best_index]
    anchor.attachment = _make_link_attachment(request, anchor)
    return 1




DANGLING_VALUE_REFERENCE_RE = re.compile(
    r"\b(?:mit\s+)?(?:diesen|diese|den)\s+(?:werten|werte)\b|\bthese\s+values\b",
    flags=re.IGNORECASE,
)
PRIOR_VALUE_RE = re.compile(
    r"\b\d+(?:[.,]\d+)?\s*(?:°\s*[CF]|℃|℉|mm(?:\s*/\s*s)?|cm|V|W|A|%)\b",
    flags=re.IGNORECASE,
)


def neutralize_unsupported_source_mechanisms(
    request: GenerationRequest,
    result: ConversationResult,
    factual_context: str = "",
) -> tuple[int, int]:
    neutralized = 0
    duplicate_avoidance = 0
    for index, entry in enumerate(result.entries):
        if not contains_unsupported_source_mechanism(request, entry.text, factual_context):
            continue
        replacement, avoided = _unique_neutral_text(
            request,
            result,
            index,
            lambda variant: neutral_source_mechanism_text(
                request, entry_type=entry.type, variant=variant
            ),
            entry_type=entry.type,
            kind="source",
        )
        entry.text = replacement
        neutralized += 1
        duplicate_avoidance += int(avoided)
    return neutralized, duplicate_avoidance


def rewrite_dangling_value_references(
    request: GenerationRequest,
    result: ConversationResult,
) -> int:
    """Rewrite references to 'these values' when no earlier value survived normalization."""
    changed = 0
    seen_value = False
    language = request.language.casefold().split("-", 1)[0]
    replacements = {
        "de": "Sobald wir belastbare Werte haben, können wir den nächsten Schritt sinnvoll beurteilen.",
        "en": "Once we have reliable values, we can assess the next step properly.",
    }
    replacement = replacements.get(language, replacements["en"])

    for entry in result.entries:
        if DANGLING_VALUE_REFERENCE_RE.search(entry.text) and not seen_value:
            entry.text = replacement
            changed += 1
        if PRIOR_VALUE_RE.search(entry.text):
            seen_value = True
    return changed


def normalize_content_features(
    request: GenerationRequest,
    result: ConversationResult,
    factual_context: str = "",
) -> ContentAdjustmentCounts:
    adjustments = ContentAdjustmentCounts()

    adjustments.narrations_added = add_missing_narration(request, result)
    adjustments.speaker_prefixes_removed = (
        remove_redundant_speaker_prefixes(request, result)
    )
    adjustments.wire_artifacts_removed = (
        strip_wire_artifacts(request, result)
    )
    (
        adjustments.source_claims_neutralized,
        source_duplicate_avoidance,
    ) = neutralize_unresolved_source_claims(request, result)
    adjustments.source_claims_neutralized += neutralize_unsupported_authority_claims(
        request, result, factual_context
    )
    (
        adjustments.unsupported_absence_claims_neutralized,
        absence_duplicate_avoidance,
    ) = neutralize_unsupported_absence_claims(request, result, factual_context)
    (
        adjustments.unsupported_source_mechanisms_neutralized,
        mechanism_duplicate_avoidance,
    ) = neutralize_unsupported_source_mechanisms(request, result, factual_context)
    (
        adjustments.technical_specifics_neutralized,
        factual_duplicate_avoidance,
    ) = neutralize_unsupported_technical_specifics(
        request,
        result,
        factual_context,
    )
    adjustments.normalization_duplicates_avoided = (
        source_duplicate_avoidance
        + absence_duplicate_avoidance
        + mechanism_duplicate_avoidance
        + factual_duplicate_avoidance
    )
    adjustments.dangling_value_references_rewritten = rewrite_dangling_value_references(
        request, result
    )
    adjustments.entries_shortened = shorten_overlong_entries(request, result)

    # v0.9.7: exact duplicate utterances are a structural quality issue, not a
    # creative repair task. Resolve them deterministically before validation so
    # they cannot consume the limited LLM repair budget.
    adjustments.normalization_duplicates_avoided += deduplicate_exact_entry_texts(
        request, result
    )

    links_sanitized, links_removed = normalize_existing_links(request, result, factual_context)
    adjustments.attachment_queries_refined = (
        refine_attachment_queries(request, result)
    )
    adjustments.links_sanitized = links_sanitized
    adjustments.links_removed = links_removed

    adjustments.images_added = add_missing_images(request, result)
    adjustments.links_added = add_missing_links(request, result)

    return adjustments
