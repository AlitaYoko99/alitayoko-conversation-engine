from __future__ import annotations

import re

from app.models import GenerationRequest


# High-confidence claims that a source/resource has already been resolved,
# opened, read or returned data. These are forbidden before the future
# Research/Resolver layer supplies evidence.
SOURCE_CLAIM_PATTERNS_DE = (
    re.compile(
        r"\b(?:pdf|anleitung|handbuch|datenblatt|support[-‑–— ]?seite|webseite|datenbank)"
        r".{0,45}\b(?:geöffnet|geladen|gefunden|gelesen|geprüft|zeigt|enthält|sagt)\b",
        flags=re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:ich|wir)\s+(?:habe|haben)\s+.{0,30}"
        r"\b(?:geöffnet|geladen|gefunden|gelesen|geprüft)\b",
        flags=re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:laut|gemäss|gemäß)\s+(?:der\s+)?"
        r"(?:anleitung|handbuch|pdf|datenblatt|hersteller(?:seite)?|support[-‑–— ]?seite)\b",
        flags=re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:gefundenen|abgerufenen|geladenen|gelesenen)\s+"
        r"(?:daten|informationen|angaben|werte)\b",
        flags=re.IGNORECASE,
    ),
)

SOURCE_CLAIM_PATTERNS_EN = (
    re.compile(
        r"\b(?:manual|pdf|datasheet|support page|website|database)"
        r".{0,45}\b(?:opened|loaded|found|read|checked|shows|contains|says)\b",
        flags=re.IGNORECASE,
    ),
    re.compile(
        r"\baccording to (?:the )?(?:manual|datasheet|manufacturer|support page)\b",
        flags=re.IGNORECASE,
    ),
)

NEUTRAL_SOURCE_TEXT = {
    "de": {
        "message": (
            "Die Quelle ist noch nicht geprüft. Wir sollten sie zuerst auflösen und dann die Angaben vergleichen.",
            "Bevor wir daraus etwas ableiten, brauchen wir erst eine tatsächlich gefundene und geprüfte Quelle.",
            "Das sollten wir erst bestätigen, wenn die passende Quelle wirklich gefunden wurde.",
        ),
        "narration": (
            "Die beiden bereiten den nächsten Prüfschritt vor, ohne die noch nicht aufgelöste Quelle vorwegzunehmen.",
            "Die Recherche ist noch offen; konkrete Angaben wollen sie erst nach der Quellenprüfung verwenden.",
            "Sie halten die technischen Details bewusst offen, bis eine passende Quelle tatsächlich gefunden ist.",
        ),
    },
    "en": {
        "message": (
            "The source has not been verified yet. We should resolve it first and then compare the details.",
            "Before relying on that, we need an actually resolved and verified source.",
            "We should only confirm that once the relevant source has really been found.",
        ),
        "narration": (
            "They prepare the next verification step without assuming that the unresolved source has already been retrieved.",
            "The research is still open, so they leave concrete details unconfirmed for now.",
            "They keep the technical details open until a relevant source is actually resolved.",
        ),
    },
}


def contains_unresolved_source_claim(
    request: GenerationRequest,
    text: str,
) -> bool:
    language = request.language.casefold().split("-", 1)[0]
    patterns = SOURCE_CLAIM_PATTERNS_DE if language == "de" else SOURCE_CLAIM_PATTERNS_EN
    return any(pattern.search(text) for pattern in patterns)


def neutral_source_text(
    request: GenerationRequest,
    *,
    entry_type: str,
    variant: int = 0,
) -> str:
    language = request.language.casefold().split("-", 1)[0]
    choices = NEUTRAL_SOURCE_TEXT.get(language, NEUTRAL_SOURCE_TEXT["en"])
    variants = choices["narration" if entry_type == "narration" else "message"]
    return variants[variant % len(variants)]


UNSUPPORTED_ABSENCE_PATTERNS_DE = (
    re.compile(
        r"\bkein(?:e|en|er|es)?\b.{0,80}\b(?:im|in\s+der|in\s+den)\s+"
        r"(?:handbuch|pdf|dokumentation|quelle|datenblatt|spezifikation(?:en)?)\b",
        flags=re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:handbuch|pdf|dokumentation|quelle|datenblatt|spezifikation(?:en)?)\b"
        r".{0,80}\b(?:enthält|nennt|listet|zeigt|gibt)\b.{0,30}\bkein(?:e|en|er|es)?\b",
        flags=re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:ist|sind)\b.{0,45}\bnicht\s+(?:angegeben|dokumentiert|genannt|aufgeführt)\b",
        flags=re.IGNORECASE,
    ),
)

UNSUPPORTED_ABSENCE_PATTERNS_EN = (
    re.compile(
        r"\bno\b.{0,80}\b(?:in|inside)\s+(?:the\s+)?(?:manual|pdf|documentation|source|datasheet|specifications?)\b",
        flags=re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:manual|pdf|documentation|source|datasheet|specifications?)\b"
        r".{0,80}\b(?:contains|lists|shows|gives)\b.{0,30}\bno\b",
        flags=re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:is|are)\b.{0,45}\bnot\s+(?:specified|documented|listed|given)\b",
        flags=re.IGNORECASE,
    ),
)

NEGATIVE_EVIDENCE_MARKERS = (
    "nicht angegeben", "nicht dokumentiert", "nicht genannt", "keine angabe",
    "kein wert", "keine maximal", "not specified", "not documented", "not listed",
    "no maximum", "no value",
)

CHECK_INTENT_MARKERS = (
    "prüfen, ob", "prüfen ob", "klären, ob", "klären ob", "sicherzugehen, dass",
    "sicherzugehen dass", "check whether", "verify whether", "find out whether",
)


def contains_unsupported_absence_claim(
    request: GenerationRequest,
    text: str,
    evidence_text: str = "",
) -> bool:
    """
    Detect claims that a source/manual does *not* contain a fact when the search
    snippets merely failed to surface it. Missing evidence is not evidence of
    absence.
    """
    if not request.features.research or not evidence_text:
        return False

    folded = text.casefold()
    if any(marker in folded for marker in CHECK_INTENT_MARKERS):
        return False

    evidence_folded = evidence_text.casefold()
    if any(marker in evidence_folded for marker in NEGATIVE_EVIDENCE_MARKERS):
        return False

    language = request.language.casefold().split("-", 1)[0]
    patterns = UNSUPPORTED_ABSENCE_PATTERNS_DE if language == "de" else UNSUPPORTED_ABSENCE_PATTERNS_EN
    return any(pattern.search(text) for pattern in patterns)


def neutral_absence_text(
    request: GenerationRequest,
    *,
    entry_type: str,
    variant: int = 0,
) -> str:
    language = request.language.casefold().split("-", 1)[0]
    variants = {
        "de": (
            "Die verfügbaren Suchergebnisse belegen zu diesem Punkt noch keine eindeutige Aussage.",
            "Aus den vorhandenen Suchhinweisen lässt sich dieser Punkt noch nicht sicher ableiten.",
            "Die bisherige Evidenz reicht für diese Aussage noch nicht aus.",
            "Zu diesem Detail liefern die vorhandenen Suchergebnisse noch keinen eindeutigen Beleg.",
        ),
        "en": (
            "The available search evidence does not establish that point yet.",
            "That conclusion cannot be drawn reliably from the current search evidence.",
            "The current evidence is not sufficient for that claim yet.",
            "The available search results do not clearly support that detail yet.",
        ),
    }
    choices = variants.get(language, variants["en"])
    return choices[variant % len(choices)]

# Resource mechanisms such as APIs or endpoints are factual entities too. A
# model may suggest checking official documentation, but it must not invent a
# product-specific API/endpoint merely because research is incomplete.
UNSUPPORTED_SOURCE_MECHANISM_PATTERNS_DE = (
    re.compile(r"\b(?:[A-Za-zÀ-ÖØ-öø-ÿ0-9-]+[-‑–— ]*)?(?:support[-‑–— ]*)?api\b", flags=re.IGNORECASE),
    re.compile(r"\b(?:api[-‑–— ]*)?endpoint\b", flags=re.IGNORECASE),
    re.compile(r"\b(?:support[-‑–— ]*)?schnittstelle\b", flags=re.IGNORECASE),
)

UNSUPPORTED_SOURCE_MECHANISM_PATTERNS_EN = (
    re.compile(r"\b(?:[A-Za-z0-9-]+[- ]*)?(?:support[- ]*)?api\b", flags=re.IGNORECASE),
    re.compile(r"\b(?:api[- ]*)?endpoint\b", flags=re.IGNORECASE),
)


def contains_unsupported_source_mechanism(
    request: GenerationRequest,
    text: str,
    evidence_text: str = "",
) -> bool:
    language = request.language.casefold().split("-", 1)[0]
    patterns = (
        UNSUPPORTED_SOURCE_MECHANISM_PATTERNS_DE
        if language == "de"
        else UNSUPPORTED_SOURCE_MECHANISM_PATTERNS_EN
    )
    matches = [pattern.search(text) for pattern in patterns]
    matches = [match for match in matches if match]
    if not matches:
        return False

    context = f"{request.topic} {evidence_text}".casefold()
    # If an API/endpoint is actually part of the request/evidence, let the
    # normal factual rules handle it. Otherwise it is an invented mechanism.
    if "api" in context or "endpoint" in context or "schnittstelle" in context:
        return False
    return True


def neutral_source_mechanism_text(
    request: GenerationRequest,
    *,
    entry_type: str,
    variant: int = 0,
) -> str:
    language = request.language.casefold().split("-", 1)[0]
    variants = {
        "de": (
            "Wir sollten stattdessen die offizielle Supportseite oder Dokumentation des Herstellers prüfen.",
            "Als nächsten Schritt sollten wir die offiziellen Herstellerunterlagen prüfen, ohne eine spezielle Schnittstelle vorauszusetzen.",
            "Prüfen wir die offiziellen Support- und Dokumentationsseiten des Herstellers, bevor wir einen weiteren Datenweg annehmen.",
        ),
        "en": (
            "We should check the manufacturer's official support page or documentation instead.",
            "The next step should be the official manufacturer documentation without assuming a special interface exists.",
            "Let's check the manufacturer's official support and documentation pages before assuming another data interface.",
        ),
    }
    choices = variants.get(language, variants["en"])
    return choices[variant % len(choices)]
