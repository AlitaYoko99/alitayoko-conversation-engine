from __future__ import annotations

import re

from app.models import GenerationRequest


DASH_TRANSLATION = str.maketrans(
    {
        "\u2010": "-",  # hyphen
        "\u2011": "-",  # non-breaking hyphen
        "\u2012": "-",  # figure dash
        "\u2013": "-",  # en dash
        "\u2014": "-",  # em dash
        "\u2212": "-",  # minus sign
    }
)


TECH_TEMPERATURE_RE = re.compile(
    r"\b\d{2,3}(?:[.,]\d+)?\s*°\s*[CF]\b",
    flags=re.IGNORECASE,
)
TECH_MEASUREMENT_RE = re.compile(
    r"\b\d+(?:[.,]\d+)?\s*(?:kPa|hPa|Pa|mbar|bar|mm|cm|km|m|mAh|Wh|W|V|A|Hz|kHz|MHz|GHz)\b",
    flags=re.IGNORECASE,
)
TECH_HEX_CODE_RE = re.compile(r"\b0x[0-9a-f]{2,}\b", flags=re.IGNORECASE)
TECH_VERSION_RE = re.compile(
    r"\b(?:"
    r"firmware(?:[-\s]?version)?|"
    r"software(?:[-\s]?version)?|"
    r"version"
    r")\s*(?:(?:ist|is|=|:)\s*)?(?:v\s*)?"
    r"\d+(?:\.\d+){1,3}\b",
    flags=re.IGNORECASE,
)
TECH_MODEL_TOKEN_RE = re.compile(
    r"\b(?=[A-Za-z0-9-]*[A-Za-z])(?=[A-Za-z0-9-]*\d)"
    r"[A-Za-z][A-Za-z0-9-]{1,20}\b"
)


def normalize_factual_text(text: str) -> str:
    return text.translate(DASH_TRANSLATION)


NEUTRAL_FACTUAL_TEXT: dict[str, dict[str, tuple[str, ...]]] = {
    "de": {
        "statement": (
            "Diese Aussage ist mit den vorhandenen Quellen noch nicht ausreichend belegt.",
            "Darauf sollten wir uns mit der bisherigen Evidenz noch nicht festlegen.",
            "Das wäre ohne belastbare Belege nur eine Vermutung.",
        ),
        "question": (
            "Das lässt sich mit den vorhandenen Quellen noch nicht zuverlässig beantworten.",
            "Für eine belastbare Antwort fehlt uns noch ausreichende Evidenz.",
            "Bevor wir das behaupten, brauchen wir dafür eine passende Quelle.",
        ),
    },
    "en": {
        "statement": (
            "That technical detail is not established yet. Let's check the appropriate documentation first.",
            "We should not assume that without the exact model information. Let's verify it in the documentation.",
            "That would only be a guess without reliable technical data. We should look it up first.",
        ),
        "question": (
            "We cannot answer that reliably without the exact model information. Let's check the documentation first.",
            "We can only answer that once we have verified the exact model.",
            "Before claiming that, we should check the technical documentation.",
        ),
    },
    "fr": {
        "statement": (
            "Ce détail technique n’est pas encore établi. Vérifions d’abord la documentation correspondante.",
            "Sans le modèle exact, mieux vaut ne pas l’affirmer. Vérifions d’abord la documentation.",
            "Sans données techniques fiables, ce ne serait qu’une supposition. Vérifions avant de conclure.",
        ),
        "question": (
            "Impossible de répondre avec certitude sans le modèle exact. Vérifions d’abord la documentation correspondante.",
            "Nous pourrons répondre lorsque nous aurons vérifié le modèle exact.",
            "Avant de l’affirmer, consultons les données techniques correspondantes.",
        ),
    },
    "it": {
        "statement": (
            "Quel dettaglio tecnico non è ancora accertato. Controlliamo prima la documentazione corretta.",
            "Senza il modello esatto è meglio non affermarlo. Verifichiamo prima la documentazione.",
            "Senza dati tecnici affidabili sarebbe solo un’ipotesi. Controlliamo prima.",
        ),
        "question": (
            "Senza il modello esatto non possiamo rispondere con certezza. Controlliamo prima la documentazione.",
            "Potremo rispondere solo dopo aver verificato il modello esatto.",
            "Prima di affermarlo, controlliamo i dati tecnici nella documentazione.",
        ),
    },
    "rm": {
        "statement": (
            "Quest detagl tecnic n’è anc betg cler. Controllain l’emprim la documentaziun adattada.",
            "Senza il model exact na duessan nus betg supponer quai. Controllain l’emprim la documentaziun.",
            "Senza datas tecnicas fidablas fiss quai mo ina supposiziun. Controllain quai l’emprim.",
        ),
        "question": (
            "Senza il model exact na pudain nus betg respunder cun segirezza. Controllain l’emprim la documentaziun.",
            "Nus pudain respunder pir suenter avair controllà il model exact.",
            "Avant che pretender quai, controllain las datas tecnicas correspundentas.",
        ),
    },
}


def request_factual_text(request: GenerationRequest, extra_context: str = "") -> str:
    parts = [request.topic]
    for participant in request.participants:
        parts.append(participant.name)
        if participant.personality:
            parts.append(participant.personality)
    if extra_context:
        parts.append(extra_context)
    return normalize_factual_text(" ".join(parts))


def unsupported_specifics(
    request: GenerationRequest,
    text: str,
    extra_context: str = "",
) -> list[str]:
    """
    Return high-confidence technical specifics that are present in generated
    text but absent from the factual request context.
    """
    context = request_factual_text(request, extra_context).casefold()
    normalized_text = normalize_factual_text(text)
    hits: list[str] = []

    for pattern in (TECH_TEMPERATURE_RE, TECH_MEASUREMENT_RE, TECH_HEX_CODE_RE, TECH_VERSION_RE):
        for match in pattern.finditer(normalized_text):
            value = match.group(0)
            support_key = value
            if pattern is TECH_VERSION_RE:
                version_match = re.search(r"\d+(?:\.\d+){1,3}", value)
                if version_match:
                    support_key = version_match.group(0)
            if support_key.casefold() not in context:
                hits.append(value)

    for token in TECH_MODEL_TOKEN_RE.findall(normalized_text):
        if token.casefold() not in context:
            hits.append(token)

    # Stable de-duplication.
    result: list[str] = []
    seen: set[str] = set()
    for hit in hits:
        key = hit.casefold()
        if key not in seen:
            seen.add(key)
            result.append(hit)

    return result


def has_unsupported_specificity(
    request: GenerationRequest,
    text: str,
    extra_context: str = "",
) -> bool:
    return bool(unsupported_specifics(request, text, extra_context))


def neutral_factual_text(
    request: GenerationRequest,
    original_text: str,
    *,
    variant: int = 0,
) -> str:
    language = request.language.casefold().split("-", 1)[0]
    choices = NEUTRAL_FACTUAL_TEXT.get(
        language,
        NEUTRAL_FACTUAL_TEXT["en"],
    )
    category = "question" if "?" in original_text else "statement"
    variants = choices[category]
    return variants[variant % len(variants)]
