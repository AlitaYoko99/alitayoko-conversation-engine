from __future__ import annotations

import re

from app.models import ConversationResult, GenerationRequest, NoAttachment, ResearchFact


UNCERTAINTY_PATTERNS: dict[str, tuple[re.Pattern[str], ...]] = {
    "de": tuple(
        re.compile(pattern, flags=re.IGNORECASE)
        for pattern in (
            r"\b(?:noch|weiterhin)\b.{0,35}\b(?:unklar|offen|ungeklärt|unbestätigt|nicht\s+geklärt)\b",
            r"\b(?:nicht\s+geklärt|nicht\s+bestätigt)\b",
            r"\b(?:müssen|sollten|erst)\b.{0,45}\b(?:prüfen|nachschlagen|nachsehen|recherchieren|finden)\b",
            r"\b(?:prüfen|nachschlagen|nachsehen|recherchieren|suchen|finden)\b.{0,55}\b(?:quelle|dokumentation|handbuch|pdf|angabe|wert|werte)\b",
            r"\b(?:quelle|dokumentation|handbuch|pdf)\b.{0,45}\b(?:prüfen|finden|fehlt|auflösen)\b",
            r"\b(?:suchen|finden)\b.{0,45}\b(?:quelle|dokumentation|handbuch|pdf)\b",
            r"\b(?:nachschlagen|nachsehen)\b",
            r"\b(?:keine|kein|keinen)\b.{0,35}\b(?:beleg|angabe|wert|werte|quelle)\b",
            r"\b(?:vermutung|erfahrungswert)\b",
        )
    ),
    "en": tuple(
        re.compile(pattern, flags=re.IGNORECASE)
        for pattern in (
            r"\b(?:still|remains?)\b.{0,35}\b(?:unknown|unclear|unverified|open)\b",
            r"\b(?:need|should|must)\b.{0,45}\b(?:check|verify|look up|research|find)\b",
            r"\b(?:source|documentation|manual|pdf)\b.{0,45}\b(?:check|find|missing)\b",
            r"\b(?:no|without)\b.{0,35}\b(?:evidence|value|values|source)\b",
            r"\b(?:guess|estimate)\b",
        )
    ),
}

COMPLETE_EVIDENCE_UNCERTAINTY_PATTERNS: dict[str, tuple[re.Pattern[str], ...]] = {
    "de": tuple(
        re.compile(pattern, flags=re.IGNORECASE)
        for pattern in (
            r"\b(?:fehlt|fehlen)\b.{0,60}\b(?:evidenz|beleg|belege|quelle|quellen|nachweis|nachweise)\b",
            r"\b(?:noch\s+)?nicht\s+(?:ausreichend|hinreichend|genügend|genuegend)\s+belegt\b",
            r"\b(?:aussage|antwort|punkt|zusammenhang)\b.{0,55}\b(?:noch\s+)?nicht\s+(?:ausreichend|hinreichend|zuverlässig|zuverlaessig|sicher)\b",
            r"\blässt\s+sich\b.{0,55}\b(?:noch\s+)?nicht\s+(?:zuverlässig|zuverlaessig|sicher)\s+beantworten\b",
            r"\b(?:noch\s+)?nicht\s+auf\s+eine\s+feste\s+aussage\s+stützen\b",
        )
    ),
    "en": tuple(
        re.compile(pattern, flags=re.IGNORECASE)
        for pattern in (
            r"\b(?:lack|lacks|missing|insufficient)\b.{0,45}\b(?:evidence|support|source|sources)\b",
            r"\b(?:not|isn't|is not)\s+(?:sufficiently|adequately|reliably)\s+(?:supported|established|verified)\b",
            r"\b(?:cannot|can't)\b.{0,45}\b(?:answer|conclude|establish)\b.{0,35}\b(?:reliably|yet)\b",
        )
    ),
}



def _language(request: GenerationRequest) -> str:
    return request.language.casefold().split("-", 1)[0]


def _fact_sentence(request: GenerationRequest, fact: ResearchFact) -> str:
    language = _language(request)
    aspect = fact.aspect.strip().rstrip(".:")

    if language == "de":
        support_match = re.match(r"^Unterstützungsstatus für\s+(.+)$", aspect, flags=re.IGNORECASE)
        if support_match:
            subject = support_match.group(1).strip()
            if fact.confidence == "primary":
                return f"{subject} ist {fact.value}."
            return f"Die geprüften Quellen stützen den Status: {subject} ist {fact.value}."
        if aspect.casefold().startswith("empfohlene "):
            if fact.confidence == "primary":
                return f"Die {aspect} ist {fact.value}."
            return f"Die geprüften Quellen nennen als {aspect} {fact.value}."
        if aspect.casefold() == "displaygröße":
            if fact.confidence == "primary":
                return f"Die Displaygröße beträgt {fact.value}."
            return f"Die geprüften Quellen nennen eine Displaygröße von {fact.value}."
        if aspect.casefold() == "interner speicher":
            if fact.confidence == "primary":
                return f"Der interne Speicher beträgt {fact.value}."
            return f"Die geprüften Quellen nennen einen internen Speicher von {fact.value}."
        if aspect.casefold().startswith("siedepunkt von wasser"):
            if fact.confidence == "primary":
                return f"Bei normalem Atmosphärendruck liegt der Siedepunkt von Wasser bei {fact.value}."
            return f"Die geprüften Quellen nennen bei normalem Atmosphärendruck einen Siedepunkt von {fact.value}."
        if aspect.casefold().startswith("einfluss geringeren luftdrucks auf den siedepunkt"):
            if fact.confidence == "primary":
                return "Der Siedepunkt von Wasser sinkt bei geringerem Luftdruck."
            return "Die geprüften Quellen stützen den Zusammenhang: Der Siedepunkt von Wasser sinkt bei geringerem Luftdruck."
        if fact.confidence == "primary":
            if re.match(r"^(?:maximale|minimale|empfohlene|zulässige|offizielle)\b", aspect, flags=re.IGNORECASE):
                return f"Die {aspect} ist mit {fact.value} dokumentiert."
            return f"Für {aspect} ist {fact.value} dokumentiert."
        if re.match(r"^(?:maximale|minimale|empfohlene|zulässige|offizielle)\b", aspect, flags=re.IGNORECASE):
            return f"Die geprüften Quellen nennen für die {aspect} {fact.value}."
        return f"Die geprüften Quellen nennen für {aspect} {fact.value}."
    if language == "fr":
        return f"La valeur documentée pour {aspect} est {fact.value}."
    if language == "it":
        return f"Il valore documentato per {aspect} è {fact.value}."
    if language == "rm":
        return f"La valur documentada per {aspect} è {fact.value}."
    return f"The documented value for {aspect} is {fact.value}."


def _contains_value(text: str, value: str) -> bool:
    def norm(raw: str) -> str:
        raw = raw.casefold().replace("℃", "°c").replace("℉", "°f").replace(",", ".")
        return re.sub(r"\s+", "", raw)

    return norm(value) in norm(text)


def _looks_stale_uncertainty(
    request: GenerationRequest,
    text: str,
    *,
    target_identity: str = "",
) -> bool:
    patterns = UNCERTAINTY_PATTERNS.get(_language(request), ())
    if any(pattern.search(text) for pattern in patterns):
        return True

    # When Research has already resolved an exact target identity, generic
    # fallback language that claims the model is unknown is stale even if no
    # numeric fact appears in the line itself.
    if target_identity and _language(request) == "de":
        if re.search(
            r"\b(?:ohne|wenn|bevor)\b.{0,35}\b(?:genaue[nrms]?\s+)?modell(?:angabe|bezeichnung)?\b",
            text,
            flags=re.IGNORECASE,
        ):
            return True
        if re.search(
            r"\bgenaue[srnm]?\s+modell\b.{0,35}\b(?:prüfen|kennen|feststellen)\b",
            text,
            flags=re.IGNORECASE,
        ):
            return True

    return False


def _grounded_followup(request: GenerationRequest, variant: int) -> str:
    language = _language(request)
    templates = {
        "de": (
            "Damit ist dieser Punkt durch die Recherche belegt.",
            "Für die belegten Angaben können wir uns auf die vorhandenen Quellen stützen.",
            "Die bestätigten Angaben sollten wir von weiterhin offenen Punkten trennen.",
            "Damit haben wir für diesen Teil eine belastbare Grundlage.",
        ),
        "en": (
            "That point is supported by the research.",
            "We can rely on the supported facts while keeping open points separate.",
            "The confirmed facts should stay separate from anything still unresolved.",
            "That gives us a reliable basis for this part of the discussion.",
        ),
        "fr": (
            "Nous disposons maintenant de limites documentées pour les valeurs techniques demandées.",
            "Nous pouvons poursuivre l'évaluation sur la base de ces valeurs documentées.",
        ),
        "it": (
            "Ora disponiamo di limiti documentati per i valori tecnici richiesti.",
            "Possiamo basare la valutazione successiva su questi valori documentati.",
        ),
        "rm": (
            "Uschia avain nus ussa valurs documentadas per ils cunfins tecnics dumandads.",
            "Nus pudain basar la valitaziun sin questas valurs documentadas.",
        ),
    }
    options = templates.get(language, templates["en"])
    return options[variant % len(options)]


def _missing_recommendation_components(missing_aspects: list[str] | None) -> list[str]:
    components: list[str] = []
    for aspect in missing_aspects or []:
        match = re.match(r"^empfohlene\s+(.+?)-Version$", aspect.strip(), flags=re.IGNORECASE)
        if match:
            components.append(match.group(1).strip())
    return components


def _unsupported_recommendation_replacement(request: GenerationRequest, component: str) -> str:
    language = _language(request)
    if language == "de":
        return f"Welche {component}-Version empfohlen wird, ist mit den vorhandenen Quellen noch nicht belegt."
    if language == "fr":
        return f"La version recommandée de {component} n'est pas encore établie par les sources disponibles."
    if language == "it":
        return f"La versione consigliata di {component} non è ancora stabilita dalle fonti disponibili."
    if language == "rm":
        return f"Las funtaunas disponiblas na conferman anc betg tge versiun da {component} è recumandada."
    return f"The available sources do not yet establish which {component} version is recommended."


def neutralize_unsupported_recommendation_inference(
    request: GenerationRequest,
    result: ConversationResult,
    *,
    missing_aspects: list[str] | None = None,
) -> int:
    """Remove recommendation inferences when research explicitly left them unresolved.

    Compatibility/support is not a recommendation. This guard is intentionally
    narrow: it activates only when Research has an explicit missing
    ``recommended <component> version`` aspect, and it rewrites declarative
    recommendation/choice language that names a concrete version. Questions are
    preserved because they do not assert an answer.
    """
    components = _missing_recommendation_components(missing_aspects)
    if not components:
        return 0

    language = _language(request)
    recommendation_patterns = {
        "de": re.compile(
            r"\b(?:soll(?:te|ten)?|empfehl\w*|nutzen|verwenden|wählen|waehlen|bleiben|setzen\s+auf|"
            r"stabile\s+basis|konservativ\w*|produktions?\w*|beste\w*\s+wahl|vorzuziehen|"
            r"für\s+stabilität\s+wäre|fuer\s+stabilitaet\s+waere|sicher(?:er)?|"
            r"beste\w*\s+option|einzige\s+(?:bestätigte|bestaetigte)\s+option)\b",
            re.IGNORECASE,
        ),
        "en": re.compile(
            r"\b(?:should|recommend\w*|use|choose|stay\s+on|stick\s+with|preferred|conservative|best\s+choice|production\s+default)\b",
            re.IGNORECASE,
        ),
        "fr": re.compile(r"\b(?:devr|recommand\w*|utilis\w*|choisir|préfér\w*)\b", re.IGNORECASE),
        "it": re.compile(r"\b(?:dovr|consigli\w*|usare|utilizz\w*|scegli\w*|prefer\w*)\b", re.IGNORECASE),
        "rm": re.compile(r"\b(?:duess|recumand\w*|duvr\w*|tscherner|prefer\w*)\b", re.IGNORECASE),
    }
    pattern = recommendation_patterns.get(language, recommendation_patterns["en"])
    version_re = re.compile(r"\b\d+(?:\.\d+){1,3}\b")

    rewritten = 0
    for entry in result.entries:
        if entry.type != "message":
            continue
        text = entry.text.strip()
        if not text or text.endswith("?"):
            continue
        if not pattern.search(text) or not version_re.search(text):
            continue

        lower = text.casefold()
        for component in components:
            # The component name may be omitted in a short conversational reply
            # such as "Dann sollten wir bei 8.4 bleiben". Since the research
            # bundle has exactly this unresolved recommendation aspect, a
            # recommendation marker plus a concrete version is sufficient.
            if component.casefold() in lower or version_re.search(text):
                entry.text = _unsupported_recommendation_replacement(request, component)
                rewritten += 1
                break
    return rewritten



def neutralize_complete_research_uncertainty(
    request: GenerationRequest,
    result: ConversationResult,
    *,
    evidence_sufficient: bool,
    missing_aspects: list[str] | None = None,
    conflicting_aspects: list[str] | None = None,
) -> int:
    """Remove generic evidence-insufficiency claims only when research is complete.

    This is deliberately stricter than the ordinary stale-uncertainty cleanup.
    It must never erase legitimate uncertainty for unresolved or conflicting
    aspects (for example an unsupported recommendation). It is intended as the
    final deterministic pass after all LLM-based audit/coherence rewrites.
    """
    if not evidence_sufficient or missing_aspects or conflicting_aspects:
        return 0

    patterns = COMPLETE_EVIDENCE_UNCERTAINTY_PATTERNS.get(_language(request), ())
    if not patterns:
        return 0

    existing = {entry.text.casefold().strip() for entry in result.entries}
    rewritten = 0
    variant = 0
    for entry in result.entries:
        if entry.type != "message":
            continue
        text = entry.text.strip()
        if not text or text.endswith("?"):
            continue
        if not any(pattern.search(text) for pattern in patterns):
            continue

        for _ in range(12):
            replacement = _grounded_followup(request, variant)
            variant += 1
            if replacement.casefold().strip() not in existing:
                break
        else:
            continue

        existing.discard(entry.text.casefold().strip())
        entry.text = replacement
        existing.add(replacement.casefold().strip())
        rewritten += 1

    return rewritten

def enforce_verified_research_facts(
    request: GenerationRequest,
    result: ConversationResult,
    facts: list[ResearchFact],
    *,
    target_identity: str = "",
    missing_aspects: list[str] | None = None,
) -> tuple[int, int, int]:
    """Deterministically ground a conversation after LLM repair is exhausted.

    Both primary and qualified secondary facts are enforced. Existing mentions
    are preserved. Missing facts replace stale uncertainty/filler messages first, without changing
    entry count, speaker allocation, attachments, or presentation features.
    Afterwards, stale "still unknown / needs lookup" messages are rewritten
    when they directly contradict the now-complete verified evidence state.
    """
    grounded_facts = list(facts)
    if not grounded_facts:
        recommendation_rewrites = neutralize_unsupported_recommendation_inference(
            request, result, missing_aspects=missing_aspects
        )
        return 0, 0, recommendation_rewrites

    all_text = " ".join(entry.text for entry in result.entries)
    missing = [fact for fact in grounded_facts if not _contains_value(all_text, fact.value)]

    message_indices = [
        index for index, entry in enumerate(result.entries)
        if entry.type == "message"
    ]
    if not message_indices:
        return 0, 0

    used_indices: set[int] = set()
    injected = 0

    def candidate_key(index: int) -> tuple[int, int, int, int]:
        entry = result.entries[index]
        # Prefer stale uncertainty text, then attachment-free messages, then
        # shorter lines near the middle. This preserves substantive dialogue.
        return (
            0 if _looks_stale_uncertainty(
                request, entry.text, target_identity=target_identity
            ) else 1,
            0 if isinstance(entry.attachment, NoAttachment) else 1,
            len(entry.text.split()),
            abs(index - len(result.entries) // 2),
        )

    for fact in missing:
        candidates = [index for index in message_indices if index not in used_indices]
        if not candidates:
            break
        index = min(candidates, key=candidate_key)
        result.entries[index].text = _fact_sentence(request, fact)
        used_indices.add(index)
        injected += 1

    # Once every requested primary fact is present, stale research uncertainty
    # is contradictory. Rewrite only those remaining message lines; leave
    # substantive discussion untouched.
    all_text = " ".join(entry.text for entry in result.entries)
    all_present = all(_contains_value(all_text, fact.value) for fact in grounded_facts)
    rewritten = 0
    if all_present:
        existing = {entry.text.casefold().strip() for entry in result.entries}
        variant = 0
        for index, entry in enumerate(result.entries):
            if index in used_indices or entry.type != "message":
                continue
            if not _looks_stale_uncertainty(
                request, entry.text, target_identity=target_identity
            ):
                continue
            for _ in range(12):
                replacement = _grounded_followup(request, variant)
                variant += 1
                if replacement.casefold().strip() not in existing:
                    break
            else:
                continue
            existing.discard(entry.text.casefold().strip())
            entry.text = replacement
            existing.add(replacement.casefold().strip())
            rewritten += 1

    recommendation_rewrites = neutralize_unsupported_recommendation_inference(
        request, result, missing_aspects=missing_aspects
    )
    return injected, rewritten, recommendation_rewrites
