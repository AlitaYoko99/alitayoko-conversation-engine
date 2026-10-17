from __future__ import annotations

import asyncio
import html
import io
import ipaddress
import re
import socket
import time
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from html.parser import HTMLParser
from typing import Protocol
from urllib.parse import urljoin, urlparse

import httpx
from pypdf import PdfReader

from app.config import settings
from app.models import GenerationRequest, ResearchFact, ResearchSource


class ResearchError(RuntimeError):
    pass


class ResearchUnavailable(ResearchError):
    pass


class SearchProvider(Protocol):
    async def search(self, query: str, *, language: str, limit: int) -> list[ResearchSource]: ...


class SourceFetchError(RuntimeError):
    pass


class SourceFetchBlocked(SourceFetchError):
    pass


@dataclass(slots=True, frozen=True)
class FetchedDocument:
    url: str
    content_type: str
    text: str


class SourceFetcherProtocol(Protocol):
    async def fetch(self, url: str) -> FetchedDocument: ...


class _VisibleHTMLParser(HTMLParser):
    """Extract human-visible text while dropping active/non-content elements."""

    _SKIP_TAGS = {"script", "style", "noscript", "svg", "canvas", "template", "form"}
    _BREAK_TAGS = {
        "p", "div", "section", "article", "header", "footer", "main",
        "h1", "h2", "h3", "h4", "h5", "h6", "li", "tr", "td", "th",
        "br", "dt", "dd", "blockquote",
    }

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._skip_depth = 0
        self._parts: list[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:  # type: ignore[override]
        tag = tag.casefold()
        if tag in self._SKIP_TAGS:
            self._skip_depth += 1
            return
        if not self._skip_depth and tag in self._BREAK_TAGS:
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.casefold()
        if tag in self._SKIP_TAGS:
            if self._skip_depth:
                self._skip_depth -= 1
            return
        if not self._skip_depth and tag in self._BREAK_TAGS:
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._skip_depth and data.strip():
            self._parts.append(data)

    def text(self) -> str:
        joined = html.unescape(" ".join(self._parts))
        joined = re.sub(r"[ \t\f\v]+", " ", joined)
        joined = re.sub(r" *\n *", "\n", joined)
        joined = re.sub(r"\n{3,}", "\n\n", joined)
        return joined.strip()


class SafeSourceFetcher:
    """Fetch public web documents with SSRF/redirect/size guards."""

    REDIRECT_CODES = {301, 302, 303, 307, 308}
    ALLOWED_TEXT_TYPES = (
        "text/html", "application/xhtml+xml", "text/plain", "text/xml",
        "application/xml",
    )

    def __init__(
        self,
        *,
        timeout_seconds: float = 8.0,
        max_bytes: int = 5_000_000,
        max_redirects: int = 3,
        pdf_max_pages: int = 40,
    ) -> None:
        self.timeout_seconds = max(1.0, float(timeout_seconds))
        self.max_bytes = max(64_000, int(max_bytes))
        self.max_redirects = max(0, min(8, int(max_redirects)))
        self.pdf_max_pages = max(1, min(200, int(pdf_max_pages)))

    async def _validate_public_url(self, url: str) -> None:
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise SourceFetchBlocked("Only public HTTP(S) source URLs are allowed.")
        if parsed.username or parsed.password:
            raise SourceFetchBlocked("Credential-bearing source URLs are not allowed.")
        if parsed.port not in {None, 80, 443}:
            raise SourceFetchBlocked("Only standard web ports are allowed for source fetching.")

        hostname = parsed.hostname.casefold().rstrip(".")
        if hostname == "localhost" or hostname.endswith(".localhost") or hostname.endswith(".local"):
            raise SourceFetchBlocked("Local source hostnames are blocked.")

        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        try:
            infos = await asyncio.to_thread(
                socket.getaddrinfo,
                hostname,
                port,
                0,
                socket.SOCK_STREAM,
            )
        except OSError as exc:
            raise SourceFetchError(f"DNS resolution failed for source host: {exc}") from exc

        addresses = {item[4][0] for item in infos if item and item[4]}
        if not addresses:
            raise SourceFetchError("Source hostname did not resolve to an address.")
        for address in addresses:
            try:
                ip = ipaddress.ip_address(address.split("%", 1)[0])
            except ValueError as exc:
                raise SourceFetchBlocked("Source hostname resolved to an invalid address.") from exc
            if not ip.is_global:
                raise SourceFetchBlocked("Private, loopback, link-local and reserved source addresses are blocked.")

    async def fetch(self, url: str) -> FetchedDocument:
        current = url
        headers = {
            "Accept": "text/html,application/xhtml+xml,application/pdf,text/plain;q=0.9,*/*;q=0.1",
            "User-Agent": "AlitaYoko-Research/0.7 (+local research fetcher)",
        }

        async with httpx.AsyncClient(
            timeout=self.timeout_seconds,
            follow_redirects=False,
            headers=headers,
        ) as client:
            for redirect_index in range(self.max_redirects + 1):
                await self._validate_public_url(current)
                try:
                    async with client.stream("GET", current) as response:
                        if response.status_code in self.REDIRECT_CODES:
                            location = response.headers.get("location", "").strip()
                            if not location:
                                raise SourceFetchError("Source returned a redirect without a Location header.")
                            if redirect_index >= self.max_redirects:
                                raise SourceFetchError("Source exceeded the redirect limit.")
                            current = urljoin(current, location)
                            continue

                        response.raise_for_status()
                        content_type = response.headers.get("content-type", "").split(";", 1)[0].casefold().strip()
                        if not (
                            content_type in self.ALLOWED_TEXT_TYPES
                            or content_type == "application/pdf"
                        ):
                            raise SourceFetchBlocked(f"Unsupported source content type: {content_type or 'unknown'}")

                        chunks: list[bytes] = []
                        total = 0
                        async for chunk in response.aiter_bytes():
                            total += len(chunk)
                            if total > self.max_bytes:
                                raise SourceFetchBlocked("Source exceeded the configured response size limit.")
                            chunks.append(chunk)
                        body = b"".join(chunks)

                        if content_type == "application/pdf":
                            text = self._pdf_text(body)
                        else:
                            encoding = response.encoding or "utf-8"
                            raw = body.decode(encoding, errors="replace")
                            text = self._html_text(raw) if "html" in content_type else self._plain_text(raw)

                        if not text.strip():
                            raise SourceFetchError("Source contained no usable textual content.")
                        return FetchedDocument(url=current, content_type=content_type, text=text)
                except SourceFetchError:
                    raise
                except httpx.HTTPError as exc:
                    raise SourceFetchError(f"Source request failed: {exc}") from exc

        raise SourceFetchError("Source fetch failed unexpectedly.")

    @staticmethod
    def _plain_text(value: str) -> str:
        value = html.unescape(value)
        value = re.sub(r"\s+", " ", value)
        return value.strip()

    @staticmethod
    def _html_text(value: str) -> str:
        parser = _VisibleHTMLParser()
        parser.feed(value)
        parser.close()
        return parser.text()

    def _pdf_text(self, value: bytes) -> str:
        try:
            reader = PdfReader(io.BytesIO(value), strict=False)
        except Exception as exc:
            raise SourceFetchError(f"PDF could not be parsed: {exc}") from exc

        pages: list[str] = []
        for page in reader.pages[: self.pdf_max_pages]:
            try:
                text = page.extract_text() or ""
            except Exception:
                continue
            if text.strip():
                pages.append(text)
        return self._plain_text("\n".join(pages))


@dataclass(slots=True, frozen=True)
class ResearchRequirement:
    key: str
    label: str
    subject_terms: tuple[str, ...]
    value_pattern: re.Pattern[str] | None = None
    require_maximum: bool = False
    kind: str = "pattern"
    target_terms: tuple[str, ...] = ()
    context_terms: tuple[str, ...] = ()


@dataclass(slots=True)
class ResearchCoverage:
    sufficient: bool = True
    matched_aspects: list[str] = field(default_factory=list)
    missing_aspects: list[str] = field(default_factory=list)
    conflicting_aspects: list[str] = field(default_factory=list)
    secondary_supported_aspects: list[str] = field(default_factory=list)
    requires_primary_source: bool = False


@dataclass(slots=True)
class ResearchBundle:
    provider: str
    queries: list[str]
    sources: list[ResearchSource]
    elapsed_seconds: float
    evidence_sufficient: bool = True
    missing_aspects: list[str] = field(default_factory=list)
    conflicting_aspects: list[str] = field(default_factory=list)
    secondary_supported_aspects: list[str] = field(default_factory=list)
    verified_facts: list[ResearchFact] = field(default_factory=list)
    target_identity: str = ""
    followup_queries: list[str] = field(default_factory=list)
    fetch_count: int = 0
    fetch_seconds: float = 0.0

    @property
    def generation_evidence_text(self) -> str:
        """Compact, trusted grounding context for dialogue generation.

        Raw search snippets and fetched excerpts remain available to deterministic
        validation and the final audit, but are deliberately kept out of the
        dialogue-generation prompt. By this stage the research layer has already
        classified the requested facts, authority and coverage.
        """
        lines: list[str] = []

        if self.verified_facts:
            lines.append("VERIFIED FACTS")
            for fact in self.verified_facts:
                source_text = ", ".join(fact.source_ids) or "unspecified"
                lines.append(
                    f"- {fact.aspect}: {fact.value} "
                    f"[confidence={fact.confidence}; sources={source_text}]"
                )
        else:
            lines.append("VERIFIED FACTS\n- none")

        if self.missing_aspects:
            lines.append("MISSING OR INSUFFICIENT\n- " + "\n- ".join(self.missing_aspects))
        if self.conflicting_aspects:
            lines.append("CONFLICTING\n- " + "\n- ".join(self.conflicting_aspects))
        if self.secondary_supported_aspects:
            lines.append(
                "SECONDARY-ONLY SUPPORT\n- "
                + "\n- ".join(self.secondary_supported_aspects)
            )

        relevant_ids = {
            source_id
            for fact in self.verified_facts
            for source_id in fact.source_ids
        }
        authority_lines: list[str] = []
        for source in self.sources:
            if source.id not in relevant_ids:
                continue
            authority_lines.append(
                f"- {source.id}: authority={source.authority}; "
                f"identity_match={source.identity_match}; title={source.title}"
            )
        if authority_lines:
            lines.append("SOURCE AUTHORITY FOR VERIFIED FACTS\n" + "\n".join(authority_lines))

        return "\n\n".join(lines)

    @property
    def evidence_text(self) -> str:
        chunks: list[str] = []
        if self.verified_facts:
            fact_lines = [
                f"- {fact.aspect}: {fact.value} ({fact.confidence}; sources: {', '.join(fact.source_ids)})"
                for fact in self.verified_facts
            ]
            chunks.append(
                "VERIFIED FACTS FOR GENERATION\n"
                + "\n".join(fact_lines)
            )
        for source in self.sources:
            evidence_label = "Fetched excerpt" if source.content_excerpt else "Search snippet"
            evidence = source.content_excerpt or source.snippet
            chunks.append(
                f"[{source.id}] {source.title}\n"
                f"Authority: {source.authority}\n"
                f"Identity match: {source.identity_match} ({source.identity_basis})\n"
                f"Fetch status: {source.fetch_status}\n"
                f"URL: {source.url}\n"
                f"{evidence_label}: {evidence}"
            )
        return "\n\n".join(chunks)


STOPWORDS_DE = {
    "aber", "als", "auch", "auf", "aus", "bei", "bevor", "das", "dass", "dem", "den",
    "der", "des", "die", "ein", "eine", "einen", "einer", "einem", "eines", "für", "haben",
    "hat", "ihr", "ihre", "im", "in", "ist", "mit", "nach", "nicht", "noch", "oder", "sich",
    "sie", "sind", "und", "von", "vor", "was", "wenn", "wie", "wir", "zu", "zum", "zur",
    "über", "ob", "beim", "gebraucht", "gebrauchte", "gebrauchten", "diskutieren", "diskutiert",
    "sprechen", "besprechen", "möchten", "wollen", "sollen", "sollten", "versuchen", "welche",
    "welcher", "welchen", "welchem", "welches", "offiziell", "offizielle", "offiziellen",
    "technisch", "technische", "technischen", "daten", "maximal", "maximale", "maximalen",
    "dokumentiert", "dokumentierte", "belegt", "belegte", "belegten", "angaben", "verwenden",
    "verwendet", "aktuell", "aktuellen", "stand", "genau", "genauen", "genaue", "nur",
    "darüber", "darueber", "klären", "klaeren", "geklärt", "geklaert", "auswirkt",
    "reines", "reine", "reiner", "normalem", "normalen", "ungefähr", "ungefaehr",
    "viel", "gross", "groß", "internen", "interner", "interne", "modell", "aussagen",
    "unterstützt", "unterstuetzt", "empfohlen", "wird", "bevorzugt", "verlässliche",
    "verlaessliche", "quellen", "quelle", "durch", "bleibt", "bleiben",
    "besitzt", "werden", "andere", "klar", "zuordenbare", "primärquelle", "primärquellen",
    "primaerquelle", "primaerquellen", "herstellerangaben", "wissenschaftlich",
    "wissenschaftliche", "wissenschaftlichen", "sein", "sowie", "beeinflusst",
}
STOPWORDS_EN = {
    "a", "an", "and", "are", "as", "at", "be", "before", "by", "for", "from", "how", "in",
    "is", "it", "of", "on", "or", "the", "their", "they", "to", "want", "with", "would",
    "discuss", "discussing", "talk", "talking", "which", "official", "technical", "data",
    "documented", "maximum", "max", "current", "use", "using",
}
TOKEN_RE = re.compile(
    r"(?:\d+(?:\.\d+){1,3})|(?:[A-Za-zÀ-ÖØ-öø-ÿ0-9]+(?:[-‑–—][A-Za-zÀ-ÖØ-öø-ÿ0-9]+)*)",
    re.UNICODE,
)
TEMPERATURE_RE = re.compile(
    r"\b\d{1,3}(?:[.,]\d+)?\s*(?:°\s*[CF]|℃|℉)",
    re.IGNORECASE,
)
VERSION_RE = re.compile(r"\b\d+(?:\.\d+){1,3}\b")
STORAGE_RE = re.compile(r"\b\d+(?:[.,]\d+)?\s*(?:kb|mb|gb|tb)\b", re.IGNORECASE)
DISPLAY_SIZE_RE = re.compile(
    r"\b\d+(?:[.,]\d+)?\s*[-‑–—]?\s*(?:\"|zoll|inch(?:es)?)\b",
    re.IGNORECASE,
)
SPEED_RE = re.compile(r"\b\d+(?:[.,]\d+)?\s*mm\s*/\s*s\b", re.IGNORECASE)
DIMENSION_RE = re.compile(
    r"\b\d+(?:[.,]\d+)?\s*[x×]\s*\d+(?:[.,]\d+)?(?:\s*[x×]\s*\d+(?:[.,]\d+)?)?\s*mm\b",
    re.IGNORECASE,
)
MAXIMUM_MARKERS = (
    "max", "maximum", "maximal", "höchst", "highest", "bis zu", "up to", "≤",
)
OFFICIAL_TOPIC_MARKERS = (
    "offiziell", "official", "hersteller", "manufacturer", "primary source", "primärquelle",
)

# Discovery sources can still be useful, but should lose to a true manufacturer
# domain when the request explicitly asks for official/primary information.
LOWER_TRUST_HOST_PARTS = {
    "reddit.com", "youtube.com", "amazon.", "ebay.", "mediamarkt.", "idealo.",
    "testsieger.", "chip.de", "otto.de", "billiger.de", "forum", "manualslib.",
    "manuals.plus", "manualslib.tech",
}
PRIMARY_PATH_HINTS = (
    "/support", "/manual", "/manuals", "/download", "/downloads", "/docs", "/documentation",
    "/product", "/products", "/spec", "/specs", "/resources", "/faq",
)
PRIMARY_TEXT_HINTS = (
    "official", "offiziell", "manufacturer", "hersteller", "support", "manual", "handbuch",
    "bedienungsanleitung", "documentation", "dokumentation", "specification", "specifications",
    "spezifikation", "spezifikationen", "technical data", "technische daten", "datasheet",
    "datenblatt",
)
GENERIC_BRAND_TOKENS = {
    "manual", "manuals", "manualslib", "support", "official", "user", "guide", "download",
    "bedienungsanleitung", "benutzerhandbuch", "specs", "specifications", "printer", "drucker",
    "3d", "pdf", "seite", "page", "neo", "tech", "de", "com",
}

ASPECT_TOKENS = {
    "düsentemperatur", "duesentemperatur", "heizbett", "temperatur", "firmware",
    "druckgeschwindigkeit", "geschwindigkeit", "bauraum", "nozzle", "heatbed", "bed",
    "temperature", "speed", "specifications", "spezifikationen", "dokumentation",
    "bedienungsanleitung", "technical", "data", "drucktemperatur", "firmware-stand",
    "display", "oled-display", "bildschirm", "screen", "speicher", "storage",
    "kapazität", "kapazitaet", "capacity", "php-version", "kompatibilität",
    "kompatibilitaet", "compatibility", "unterstützung", "unterstuetzung", "support",
    "siedepunkt", "siedetemperatur", "boiling", "boiling-point", "luftdruck",
    "atmosphärendruck", "atmosphaerendruck", "pressure", "siedet", "boils",
    "geringerem", "geringerer", "geringer", "niedriger", "niedrigerem", "lower",
    "wasser", "water", "sinkt", "sink", "decreases", "decrease",
}
MODEL_VARIANT_MARKERS = {
    "neo", "pro", "plus", "max", "mini", "se", "ke", "s1", "s1pro", "s1-pro",
    "combo", "ultra", "lite", "go", "xl", "mk2", "mk3", "mk4",
}
IDENTITY_BOUNDARY_WORDS = {
    "3d", "printer", "drucker", "manual", "guide", "support", "specs", "specifications",
    "comparison", "vergleich", "vs", "bedienungsanleitung", "user", "official", "review",
    "test", "faq", "documentation", "dokumentation", "parameter", "technical", "data",
}


def _clean_query_terms(request: GenerationRequest) -> list[str]:
    names = {p.name.casefold() for p in request.participants}
    language = request.language.casefold().split("-", 1)[0]
    stopwords = STOPWORDS_DE if language == "de" else STOPWORDS_EN
    result: list[str] = []
    seen: set[str] = set()

    for token in TOKEN_RE.findall(request.topic):
        folded = token.casefold()
        if folded in names or folded in stopwords:
            continue
        if len(folded) <= 2 and not any(ch.isdigit() for ch in folded):
            continue
        if folded in seen:
            continue
        seen.add(folded)
        result.append(token)
        if len(result) >= 10:
            break
    return result


def _looks_technical(terms: list[str]) -> bool:
    tech_words = {
        "drucker", "3d-drucker", "software", "firmware", "gerät", "geraet", "hardware",
        "technik", "düsentemperatur", "duesentemperatur", "heizbett", "temperatur", "printer",
        "device", "technical", "nozzle", "bed", "hotend", "extruder",
    }
    lowered = {term.casefold() for term in terms}
    return bool(lowered.intersection(tech_words)) or any(any(ch.isdigit() for ch in term) for term in terms)


def _identity_terms_from_query_terms(terms: list[str]) -> list[str]:
    identity: list[str] = []
    for index, term in enumerate(terms):
        if term.casefold() in ASPECT_TOKENS:
            continue
        if term.isdigit():
            next_term = terms[index + 1].casefold() if index + 1 < len(terms) else ""
            if not re.fullmatch(r"v\d+[a-z0-9-]*", next_term):
                continue
        identity.append(term)
        if len(identity) >= 4:
            break
    return identity


def _target_identity(request: GenerationRequest) -> str:
    terms = _identity_terms_from_query_terms(_clean_query_terms(request))
    if not terms:
        return ""

    # For relationship questions involving multiple versioned components
    # (e.g. "WordPress 6.8" + "PHP 8.4"), the first versioned entity is the
    # subject identity. Later versioned entities are usually aspects/targets.
    for index, term in enumerate(terms):
        if VERSION_RE.fullmatch(term) and index >= 1:
            return " ".join(terms[: index + 1]).strip()

    return " ".join(terms[:4]).strip()


def _quoted_target_identity(request: GenerationRequest) -> str:
    identity = _target_identity(request)
    return f'"{identity}"' if len(identity.split()) >= 2 else identity


def _aspect_query_terms(request: GenerationRequest) -> list[str]:
    identity = {term.casefold() for term in _identity_terms_from_query_terms(_clean_query_terms(request))}
    return [
        term for term in _clean_query_terms(request)
        if term.casefold() not in identity
    ]


def plan_research_queries(request: GenerationRequest, max_queries: int = 2) -> list[str]:
    """Create one broad, compact discovery query.

    Search is responsible for finding candidate sources, not for deciding source
    authority or factual sufficiency. Those decisions stay in the later research
    stages. A targeted follow-up query may still be generated after evidence
    assessment when a concrete aspect remains unresolved.
    """
    terms = _clean_query_terms(request)
    if not terms:
        return [request.topic[:180].strip()]

    # Keep discovery deliberately simple: no quotes, no site filters and no
    # "official/specifications" suffixes. The first five meaningful topic terms
    # provide enough subject context while preserving recall across domains.
    query = " ".join(terms[:5]).strip()
    return _dedupe_and_cap_queries([query], 1)


def _dedupe_and_cap_queries(queries: list[str], max_queries: int) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for query in queries:
        query = " ".join(query.split())
        if len(query) > 180:
            compact: list[str] = []
            for word in query.split():
                candidate = " ".join(compact + [word])
                if len(candidate) > 180:
                    break
                compact.append(word)
            query = " ".join(compact)
        folded = query.casefold()
        if query and folded not in seen:
            seen.add(folded)
            result.append(query)
    return result[: max(1, max_queries)]


def _plain_text(value: object, limit: int = 1800) -> str:
    text = html.unescape(str(value or ""))
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:limit].rstrip()


def _hostname(url: str) -> str:
    return (urlparse(url).hostname or "").casefold().removeprefix("www.")


def _topic_relevance_terms(request: GenerationRequest) -> list[str]:
    return [term.casefold() for term in _clean_query_terms(request)[:8]]


def _topic_requests_primary_source(request: GenerationRequest) -> bool:
    return _topic_primary_policy(request) == "require"


def _topic_primary_policy(request: GenerationRequest) -> str:
    """Return ``require``, ``prefer`` or ``none`` for source authority.

    "Official preferred" is deliberately weaker than "only official / primary
    sources". This avoids turning a preference into a hard research failure.
    """
    folded = request.topic.casefold()
    if any(marker in folded for marker in ("bevorzugt", "preferred", "prefer")):
        return "prefer"
    hard_markers = (
        "nur aussagen", "nur belegte angaben", "nur durch offizielle",
        "primärquellen belegt", "primaerquellen belegt", "only official",
        "only primary", "official technical data", "offiziellen technischen daten",
    )
    if any(marker in folded for marker in hard_markers):
        return "require"
    if any(marker in folded for marker in OFFICIAL_TOPIC_MARKERS):
        return "require"
    return "none"


def _versioned_entities(request: GenerationRequest) -> list[tuple[str, str]]:
    """Extract generic ``name + dotted-version`` pairs from the topic."""
    pattern = re.compile(
        r"\b([A-Za-zÀ-ÖØ-öø-ÿ][A-Za-zÀ-ÖØ-öø-ÿ0-9+.#-]{1,24})\s+(\d+(?:\.\d+){1,3})\b"
    )
    seen: set[tuple[str, str]] = set()
    result: list[tuple[str, str]] = []
    for name, version in pattern.findall(request.topic):
        pair = (name, version)
        folded = (name.casefold(), version)
        if folded in seen:
            continue
        seen.add(folded)
        result.append(pair)
    return result


def _research_requirements(request: GenerationRequest) -> list[ResearchRequirement]:
    """Extract a small set of high-confidence factual needs from the topic."""
    topic = request.topic.casefold()
    requirements: list[ResearchRequirement] = []
    asks_maximum = any(marker in topic for marker in MAXIMUM_MARKERS)

    nozzle_terms = ("düse", "duese", "nozzle", "hotend")
    bed_terms = ("heizbett", "druckbett", "heatbed", "hot bed", "heated bed", "bed temperature")

    has_temperature_context = (
        "temperatur" in topic or "temperature" in topic or any(term in topic for term in nozzle_terms)
    )

    if has_temperature_context and any(term in topic for term in nozzle_terms):
        requirements.append(
            ResearchRequirement(
                key="nozzle_temperature",
                label=("maximale Düsentemperatur" if asks_maximum else "Düsentemperatur"),
                subject_terms=nozzle_terms,
                value_pattern=TEMPERATURE_RE,
                require_maximum=asks_maximum,
            )
        )

    if has_temperature_context and any(term in topic for term in bed_terms):
        requirements.append(
            ResearchRequirement(
                key="bed_temperature",
                label=("maximale Heizbetttemperatur" if asks_maximum else "Heizbetttemperatur"),
                subject_terms=bed_terms,
                value_pattern=TEMPERATURE_RE,
                require_maximum=asks_maximum,
            )
        )

    if "firmware" in topic and any(word in topic for word in ("version", "stand", "release")):
        requirements.append(
            ResearchRequirement(
                key="firmware_version",
                label="Firmware-Version",
                subject_terms=("firmware",),
                value_pattern=VERSION_RE,
            )
        )

    if any(word in topic for word in ("druckgeschwindigkeit", "printing speed", "geschwindigkeit")):
        requirements.append(
            ResearchRequirement(
                key="printing_speed",
                label="Druckgeschwindigkeit",
                subject_terms=("druckgeschwindigkeit", "printing speed", "speed"),
                value_pattern=SPEED_RE,
                require_maximum=asks_maximum,
            )
        )

    if any(word in topic for word in ("bauraum", "build volume", "druckvolumen")):
        requirements.append(
            ResearchRequirement(
                key="build_volume",
                label="Bauraum",
                subject_terms=("bauraum", "build volume", "druckvolumen"),
                value_pattern=DIMENSION_RE,
            )
        )

    # Generic display-size and storage-capacity questions. These fact types are
    # intentionally product-agnostic and apply equally to consoles, phones,
    # laptops, tablets and other hardware.
    if any(term in topic for term in ("display", "bildschirm", "screen")) and any(
        marker in topic for marker in ("gross", "groß", "größe", "groesse", "size", "diagonal")
    ):
        requirements.append(
            ResearchRequirement(
                key="display_size",
                label="Displaygröße",
                subject_terms=("display", "bildschirm", "screen", "oled"),
                value_pattern=DISPLAY_SIZE_RE,
            )
        )

    if any(term in topic for term in ("speicher", "storage", "capacity", "kapazität", "kapazitaet")):
        requirements.append(
            ResearchRequirement(
                key="storage_capacity",
                label="interner Speicher",
                subject_terms=("speicher", "storage", "capacity", "kapazität", "kapazitaet"),
                value_pattern=STORAGE_RE,
            )
        )

    # Boiling-point questions are another generic numeric fact class. The
    # separate pressure relation below captures the qualitative direction.
    if any(term in topic for term in ("siedepunkt", "siedetemperatur", "boiling point", "siedet")):
        requirements.append(
            ResearchRequirement(
                key="boiling_temperature",
                label="Siedepunkt von Wasser",
                subject_terms=("siedepunkt", "siedetemperatur", "boiling point", "siedet", "boils"),
                value_pattern=TEMPERATURE_RE,
                kind="boiling_standard_temperature",
            )
        )
        if any(term in topic for term in ("luftdruck", "atmosphärendruck", "atmosphaerendruck", "pressure")):
            requirements.append(
                ResearchRequirement(
                    key="pressure_boiling_relation",
                    label="Einfluss geringeren Luftdrucks auf den Siedepunkt",
                    subject_terms=("luftdruck", "atmosphärendruck", "atmosphaerendruck", "pressure", "siedepunkt", "boiling point"),
                    kind="pressure_boiling_relation",
                )
            )

    # Generic software compatibility/recommendation relationships. We derive
    # component/version pairs from the topic instead of hard-coding products.
    entities = _versioned_entities(request)
    support_markers = ("unterstützt", "unterstuetzt", "support", "compatible", "kompatibel")
    if len(entities) >= 2 and any(marker in topic for marker in support_markers):
        context_name, context_version = entities[0]
        target_name, target_version = entities[-1]
        requirements.append(
            ResearchRequirement(
                key="compatibility_status",
                label=f"Unterstützungsstatus für {target_name} {target_version}",
                subject_terms=(target_name.casefold(), f"{target_name} {target_version}".casefold()),
                kind="support_status",
                target_terms=(target_name.casefold(), target_version.casefold()),
                context_terms=(context_name.casefold(), context_version.casefold()),
            )
        )

    if entities and any(marker in topic for marker in ("empfohlen", "empfehl", "recommended", "recommend")):
        component = entities[-1][0]
        requirements.append(
            ResearchRequirement(
                key="recommended_version",
                label=f"empfohlene {component}-Version",
                subject_terms=(component.casefold(), "empfohlen", "recommended"),
                kind="recommended_version",
                target_terms=(component.casefold(),),
            )
        )

    # Stable de-duplication by semantic key.
    unique: list[ResearchRequirement] = []
    seen_keys: set[str] = set()
    for requirement in requirements:
        if requirement.key in seen_keys:
            continue
        seen_keys.add(requirement.key)
        unique.append(requirement)
    return unique


def _source_evidence_text(source: ResearchSource) -> str:
    return " ".join(part for part in (source.title, source.snippet, source.content_excerpt) if part)


def _source_window_supports_requirement(source: ResearchSource, requirement: ResearchRequirement) -> bool:
    if requirement.kind != "pattern":
        return bool(_source_requirement_values(source, requirement))

    text = _source_evidence_text(source).casefold()
    for subject in requirement.subject_terms:
        start = 0
        while True:
            index = text.find(subject, start)
            if index < 0:
                break
            window = text[max(0, index - 90): min(len(text), index + 170)]
            if requirement.value_pattern and requirement.value_pattern.search(window):
                if not requirement.require_maximum or any(marker in window for marker in MAXIMUM_MARKERS):
                    return True
            start = index + len(subject)
    return False


def _canonical_value(value: str) -> str:
    value = value.casefold().replace("℃", "°c").replace("℉", "°f")
    value = re.sub(r"\s+", "", value)
    value = value.replace(",", ".")
    return value


def _source_requirement_values(
    source: ResearchSource,
    requirement: ResearchRequirement,
) -> set[str]:
    if requirement.kind == "support_status":
        return _support_status_values(source, requirement)
    if requirement.kind == "recommended_version":
        return _recommended_version_values(source, requirement)
    if requirement.kind == "pressure_boiling_relation":
        return _pressure_boiling_relation_values(source, requirement)
    if requirement.kind == "boiling_standard_temperature":
        return _boiling_standard_temperature_values(source, requirement)

    text = _source_evidence_text(source).casefold()
    values: set[str] = set()
    if requirement.value_pattern is None:
        return values

    for subject in requirement.subject_terms:
        start = 0
        while True:
            index = text.find(subject, start)
            if index < 0:
                break
            window_start = max(0, index - 45)
            window_end = min(len(text), index + 120)
            window = text[window_start:window_end]
            if requirement.require_maximum and not any(marker in window for marker in MAXIMUM_MARKERS):
                start = index + len(subject)
                continue

            matches = list(requirement.value_pattern.finditer(window))
            if matches:
                subject_pos = index - window_start
                # Use the value nearest to the subject term. This avoids
                # confusing nozzle and bed values that occur in the same snippet.
                after = [match for match in matches if match.start() >= subject_pos]
                if after:
                    nearest = min(after, key=lambda match: match.start() - subject_pos)
                else:
                    nearest = min(
                        matches,
                        key=lambda match: subject_pos - match.end(),
                    )
                values.add(_canonical_value(nearest.group(0)))
            start = index + len(subject)

    return values


def _boiling_standard_temperature_values(
    source: ResearchSource,
    requirement: ResearchRequirement,
) -> set[str]:
    """Extract boiling temperature only in standard/sea-level pressure context.

    This prevents temperatures from mountains or pressure cookers from being
    treated as conflicts with the normal-atmospheric-pressure value requested
    by the topic.
    """
    text = re.sub(r"\s+", " ", _source_evidence_text(source).casefold())
    context_patterns = (
        r"normal(?:em|er|en)?\s+atmosphär(?:ischem|ischer|ischen)?\s+druck",
        r"standard(?:druck| pressure)",
        r"normal\s+atmospheric\s+pressure",
        r"meereshöhe|sea level",
        r"\b1\s*(?:atm|atmosphäre|atmosphere)\b",
        r"\b101(?:[.,]\d+)?\s*kpa\b",
        r"\b1013(?:[.,]\d+)?\s*hpa\b",
    )
    values: set[str] = set()
    for pattern in context_patterns:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            window = text[max(0, match.start() - 180): min(len(text), match.end() + 220)]
            if not any(term in window for term in requirement.subject_terms):
                continue
            temperatures = list(TEMPERATURE_RE.finditer(window))
            if not temperatures:
                continue
            # Prefer Celsius when both Celsius and Fahrenheit equivalents are
            # printed together (for example 100 °C / 212 °F). Treating those
            # two unit representations as conflicting values would create a
            # false evidence conflict.
            celsius = [
                item for item in temperatures
                if re.search(r"(?:°\s*C|℃)", item.group(0), flags=re.IGNORECASE)
            ]
            candidates = celsius or temperatures
            context_pos = match.start() - max(0, match.start() - 180)
            nearest = min(candidates, key=lambda item: abs(item.start() - context_pos))
            values.add(_canonical_value(nearest.group(0)))
    return values


def _support_status_values(
    source: ResearchSource,
    requirement: ResearchRequirement,
) -> set[str]:
    text = re.sub(r"\s+", " ", _source_evidence_text(source).casefold())
    target_name = requirement.target_terms[0] if requirement.target_terms else ""
    target_version = requirement.target_terms[1] if len(requirement.target_terms) > 1 else ""
    if not target_name or not target_version:
        return set()

    target_re = re.compile(rf"\b{re.escape(target_name)}\s*{re.escape(target_version)}\b", re.IGNORECASE)
    values: set[str] = set()
    for match in target_re.finditer(text):
        window = text[max(0, match.start() - 180): min(len(text), match.end() + 220)]
        if requirement.context_terms:
            context_name = requirement.context_terms[0]
            context_version = requirement.context_terms[1] if len(requirement.context_terms) > 1 else ""
            context_direct = f"{context_name} {context_version}".strip()
            if context_direct and context_direct not in window:
                # Keep this strict: relationship evidence must name the subject
                # version directly rather than relying on unstated inference.
                continue

        if re.search(r"\b(?:not|nicht|kein(?:e|en|er|es)?)\b.{0,35}\b(?:support|unterstütz|kompatib)", window):
            values.add("nicht unterstützt")
            continue
        if re.search(r"\b(?:beta|experimental|experimentell|preview)\b", window):
            values.add("Beta-Support")
            continue
        if re.search(
            r"\b(?:fully|full|vollständig|vollstaendig|complete(?:ly)?)\b.{0,35}\b(?:support|unterstütz|kompatib)",
            window,
        ) or re.search(
            r"\b(?:support|unterstütz|kompatib)\w*\b.{0,35}\b(?:fully|full|vollständig|vollstaendig)\b",
            window,
        ):
            values.add("vollständig unterstützt")
            continue
        if re.search(r"\b(?:support|supports|supported|unterstütz\w*|kompatib\w*)\b", window):
            values.add("unterstützt")
    return values


def _recommended_version_values(
    source: ResearchSource,
    requirement: ResearchRequirement,
) -> set[str]:
    text = re.sub(r"\s+", " ", _source_evidence_text(source).casefold())
    component = requirement.target_terms[0] if requirement.target_terms else ""
    if not component:
        return set()
    recommendation_markers = re.compile(
        r"\b(?:recommend(?:ed|s|ation)?|empfehl\w*|empfohlen|mindestens empfohlen)\b",
        re.IGNORECASE,
    )
    component_version_re = re.compile(
        rf"\b{re.escape(component)}\s*(\d+(?:\.\d+){{1,3}})\b",
        re.IGNORECASE,
    )
    values: set[str] = set()
    for marker in recommendation_markers.finditer(text):
        window = text[max(0, marker.start() - 140): min(len(text), marker.end() + 180)]
        matches = list(component_version_re.finditer(window))
        if not matches:
            continue
        nearest = min(matches, key=lambda match: abs(match.start() - (marker.start() - max(0, marker.start() - 140))))
        values.add(f"{component.upper() if len(component) <= 4 else component} {nearest.group(1)}")
    return values


def _pressure_boiling_relation_values(
    source: ResearchSource,
    requirement: ResearchRequirement,
) -> set[str]:
    text = re.sub(r"\s+", " ", _source_evidence_text(source).casefold())
    has_pressure = any(term in text for term in ("luftdruck", "atmosphär", "atmosphaer", "pressure"))
    has_boiling = any(term in text for term in ("siedepunkt", "siedetemperatur", "boiling point", "boils", "siedet", "boil"))
    if not (has_pressure and has_boiling):
        return set()

    # First handle common direct phrasings. These patterns are deliberately
    # order-tolerant enough for both German and English prose.
    direction_patterns = (
        r"(?:lower|niedriger|geringer)\w*.{0,40}(?:air\s+|atmospheric\s+)?(?:pressure|luftdruck|atmosphär\w*druck).{0,140}(?:boiling point|siedepunkt|siedetemperatur|sied\w*|boil\w*).{0,90}(?:decreas\w*|sinkt|niedriger|geringer|unter|below|lower)",
        r"(?:pressure|luftdruck|atmosphär\w*druck).{0,80}(?:sinkt|niedriger|geringer|lower|decreas\w*).{0,180}(?:sied\w*|boil\w*).{0,100}(?:unter|below|niedriger|geringer|lower|decreas\w*)?",
        r"(?:siedepunkt|siedetemperatur|boiling point).{0,90}(?:sinkt|niedriger|geringer|lower|decreas\w*).{0,180}(?:luftdruck|atmosphär\w*druck|pressure)",
    )
    if any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in direction_patterns):
        return {"sinkt bei geringerem Luftdruck"}

    # Then use sentence-local semantic cues instead of requiring one exact word
    # order. This catches formulations such as "der Luftdruck ist niedriger,
    # sodass Wasser bei niedrigeren Temperaturen siedet" and "mit zunehmender
    # Höhe sinkt der Luftdruck, wodurch Wasser unter 100 °C siedet".
    sentences = [part.strip() for part in re.split(r"(?<=[.!?;])\s+|\n+", text) if part.strip()]
    pressure_terms = ("luftdruck", "atmosphär", "atmosphaer", "pressure")
    pressure_lower = ("niedriger", "geringer", "sinkt", "fällt", "faellt", "lower", "decrease", "decreases", "decreased")
    boiling_terms = ("sied", "boil", "siedepunkt", "siedetemperatur", "boiling point")
    boiling_lower = ("niedriger", "geringer", "unter", "below", "lower", "sinkt", "decrease", "decreases")
    altitude_terms = ("höhe", "hoehe", "altitude", "higher altitude", "größeren höhen", "groesseren hoehen")

    for index, sentence in enumerate(sentences):
        # Relation may span two adjacent clauses/sentences, so inspect a small
        # local window while remaining far stricter than whole-document co-occurrence.
        window = sentence
        if index + 1 < len(sentences):
            window += " " + sentences[index + 1]
        if index > 0:
            window = sentences[index - 1] + " " + window

        has_p = any(term in window for term in pressure_terms)
        has_p_lower = any(term in window for term in pressure_lower)
        has_b = any(term in window for term in boiling_terms)
        has_b_lower = any(term in window for term in boiling_lower)
        altitude_context = any(term in window for term in altitude_terms)

        if has_p and has_p_lower and has_b and has_b_lower:
            return {"sinkt bei geringerem Luftdruck"}
        if altitude_context and has_p and has_p_lower and has_b and has_b_lower:
            return {"sinkt bei geringerem Luftdruck"}

    return set()


def _display_canonical_value(value: str) -> str:
    value = value.replace("°c", " °C").replace("°f", " °F")
    value = re.sub(r"(?<=\d)(KB|MB|GB|TB)\b", r" \1", value, flags=re.IGNORECASE)
    value = re.sub(r"\b(kb|mb|gb|tb)\b", lambda match: match.group(1).upper(), value, flags=re.IGNORECASE)
    value = re.sub(r"\s*[-‑–—]\s*(zoll|inch(?:es)?)\b", r" \1", value, flags=re.IGNORECASE)
    value = re.sub(r"\bzoll\b", "Zoll", value, flags=re.IGNORECASE)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def _source_year(source: ResearchSource) -> int | None:
    """Best-effort publication/update year from URL/title/snippet metadata text."""
    years = [
        int(match)
        for match in re.findall(r"\b(20\d{2})\b", f"{source.url} {source.title} {source.snippet}")
    ]
    return max(years) if years else None


def _prefer_freshest_primary_sources(
    requirement: ResearchRequirement,
    sources: list[ResearchSource],
) -> list[ResearchSource]:
    """For version support/recommendation, newer dated primary docs supersede older ones.

    This is intentionally conservative: if any primary source is undated, keep
    all sources and let conflict detection preserve uncertainty.
    """
    if requirement.kind not in {"support_status", "recommended_version"} or len(sources) < 2:
        return sources
    dated = [(source, _source_year(source)) for source in sources]
    if any(year is None for _, year in dated):
        return sources
    years = {year for _, year in dated if year is not None}
    if len(years) <= 1:
        return sources
    newest = max(years)
    return [source for source, year in dated if year == newest]


def extract_verified_facts(
    request: GenerationRequest,
    sources: list[ResearchSource],
    *,
    brand_hint: str = "",
) -> list[ResearchFact]:
    """Return unambiguous requirement/value pairs from qualified exact-model evidence."""
    requirements = _research_requirements(request)
    primary_policy = _topic_primary_policy(request)
    requires_primary = primary_policy == "require"
    target_identity = _target_identity(request)
    require_exact_identity = bool(target_identity and any(ch.isdigit() for ch in target_identity))
    facts: list[ResearchFact] = []

    for requirement in requirements:
        supporting = [
            source for source in sources
            if _source_window_supports_requirement(source, requirement)
        ]
        if require_exact_identity:
            supporting = [
                source for source in supporting
                if _source_identity_match(request, source, brand_hint=brand_hint) == "exact"
            ]

        all_primary = [
            source for source in supporting
            if _is_primary_like(source, brand_hint, request)
        ]
        primary = _prefer_freshest_primary_sources(requirement, all_primary)
        secondary = [source for source in supporting if source not in all_primary]
        qualified = primary if (requires_primary or (primary_policy == "prefer" and primary)) else supporting
        confidence = "primary" if primary and qualified is primary else "secondary"

        values_to_sources: dict[str, list[str]] = {}
        for source in qualified:
            for value in _source_requirement_values(source, requirement):
                values_to_sources.setdefault(value, []).append(source.id)

        if len(values_to_sources) == 1:
            value, source_ids = next(iter(values_to_sources.items()))
            facts.append(
                ResearchFact(
                    aspect=requirement.label,
                    value=_display_canonical_value(value),
                    confidence=confidence,
                    source_ids=list(dict.fromkeys(source_ids)),
                )
            )
            continue

        # If official evidence is absent but an exact-model secondary source is
        # unambiguous, retain it explicitly as secondary context. It must not
        # satisfy an official-source request, but it is still useful to generation.
        if requires_primary and not primary:
            secondary_values: dict[str, list[str]] = {}
            for source in secondary:
                for value in _source_requirement_values(source, requirement):
                    secondary_values.setdefault(value, []).append(source.id)
            if len(secondary_values) == 1:
                value, source_ids = next(iter(secondary_values.items()))
                facts.append(
                    ResearchFact(
                        aspect=requirement.label,
                        value=_display_canonical_value(value),
                        confidence="secondary",
                        source_ids=list(dict.fromkeys(source_ids)),
                    )
                )

    return facts


def _infer_brand_hint(request: GenerationRequest, sources: list[ResearchSource]) -> str:
    """
    Infer a likely manufacturer token from repeated result titles.

    This is a retrieval hint only, never treated as factual evidence.
    """
    request_terms = {token.casefold() for token in TOKEN_RE.findall(request.topic)}
    counts: Counter[str] = Counter()
    display: dict[str, str] = {}

    for source in sources:
        host = _hostname(source.url)
        host_noise = {
            label for label in host.split(".")
            if label and any(part in host for part in LOWER_TRUST_HOST_PARTS)
        }
        seen_in_source: set[str] = set()
        for token in TOKEN_RE.findall(source.title):
            folded = token.casefold().strip("-")
            if len(folded) < 4 or any(ch.isdigit() for ch in folded):
                continue
            if folded in request_terms or folded in GENERIC_BRAND_TOKENS or folded in host_noise:
                continue
            if folded in seen_in_source:
                continue
            seen_in_source.add(folded)
            counts[folded] += 1
            display.setdefault(
                folded,
                token.title() if token.isupper() and len(token) > 3 else token,
            )

    if not counts:
        return ""
    token, count = counts.most_common(1)[0]
    return display[token] if count >= 2 else ""


def _request_primary_host_tokens(request: GenerationRequest) -> set[str]:
    tokens: set[str] = set()
    for term in _identity_terms_from_query_terms(_clean_query_terms(request)):
        folded = unicodedata.normalize("NFKD", term).encode("ascii", "ignore").decode("ascii").casefold()
        folded = re.sub(r"[^a-z0-9]+", "", folded)
        if len(folded) < 4 or any(ch.isdigit() for ch in folded):
            continue
        if folded in {"switch", "display", "screen", "water", "wasser", "pressure"}:
            continue
        tokens.add(folded)
    return tokens


def _is_primary_like(
    source: ResearchSource,
    brand_hint: str,
    request: GenerationRequest | None = None,
) -> bool:
    host = _hostname(source.url)
    if any(part in host for part in LOWER_TRUST_HOST_PARTS):
        return False
    if brand_hint and brand_hint.casefold() in host:
        return True
    if request is not None:
        labels = {
            re.sub(r"[^a-z0-9]+", "", label.casefold())
            for label in re.split(r"[.\-]", host)
            if label
        }
        if labels.intersection(_request_primary_host_tokens(request)):
            return True
    title = source.title.casefold()
    return "official" in title or "offiziell" in title


def assess_evidence_coverage(
    request: GenerationRequest,
    sources: list[ResearchSource],
    *,
    brand_hint: str = "",
) -> ResearchCoverage:
    requirements = _research_requirements(request)
    if not requirements:
        return ResearchCoverage(sufficient=True)

    primary_policy = _topic_primary_policy(request)
    requires_primary = primary_policy == "require"
    target_identity = _target_identity(request)
    require_exact_identity = bool(target_identity and any(ch.isdigit() for ch in target_identity))
    matched: list[str] = []
    missing: list[str] = []
    conflicting: list[str] = []
    secondary_supported: list[str] = []

    for requirement in requirements:
        all_supporting = [
            source for source in sources
            if _source_window_supports_requirement(source, requirement)
        ]
        if require_exact_identity:
            all_supporting = [
                source for source in all_supporting
                if _source_identity_match(request, source, brand_hint=brand_hint) == "exact"
            ]

        supporting = list(all_supporting)
        preferred_primary = [
            source for source in all_supporting
            if _is_primary_like(source, brand_hint, request)
        ]
        preferred_primary = _prefer_freshest_primary_sources(requirement, preferred_primary)
        if requires_primary:
            supporting = preferred_primary

            # Exact-model secondary evidence is still useful context, but it
            # must never satisfy a request that explicitly asks for official /
            # manufacturer data. Expose that distinction so the generator can
            # say what is known provisionally instead of falling back to empty
            # placeholder dialogue.
            secondary = [
                source for source in all_supporting
                if not _is_primary_like(source, brand_hint, request)
            ]
            secondary_values: set[str] = set()
            for source in secondary:
                secondary_values.update(_source_requirement_values(source, requirement))
            if secondary and len(secondary_values) == 1:
                secondary_supported.append(requirement.label)
        elif primary_policy == "prefer" and preferred_primary:
            supporting = preferred_primary

        values: set[str] = set()
        for source in supporting:
            values.update(_source_requirement_values(source, requirement))

        if not supporting or not values:
            missing.append(requirement.label)
        elif len(values) > 1:
            matched.append(requirement.label)
            conflicting.append(requirement.label)
        else:
            matched.append(requirement.label)

    return ResearchCoverage(
        sufficient=not missing and not conflicting,
        matched_aspects=matched,
        missing_aspects=missing,
        conflicting_aspects=conflicting,
        secondary_supported_aspects=secondary_supported,
        requires_primary_source=requires_primary,
    )


def _product_identity_terms(request: GenerationRequest) -> list[str]:
    identity = _target_identity(request)
    return TOKEN_RE.findall(identity) if identity else []


def _normalized_identity_tokens(text: str) -> list[str]:
    normalized = text.casefold()
    normalized = re.sub(r"[-‑–—_/]+", " ", normalized)
    return [token.casefold() for token in TOKEN_RE.findall(normalized)]


def _sequence_index(tokens: list[str], sequence: list[str]) -> int:
    if not sequence or len(tokens) < len(sequence):
        return -1
    for index in range(len(tokens) - len(sequence) + 1):
        if tokens[index:index + len(sequence)] == sequence:
            return index
    return -1


def _looks_like_model_variant(token: str) -> bool:
    folded = token.casefold()
    return (
        folded in MODEL_VARIANT_MARKERS
        or any(ch.isdigit() for ch in folded)
        or bool(re.fullmatch(r"v\d+[a-z0-9-]*", folded))
    )


def _target_family_tokens(target: list[str]) -> tuple[list[str], str]:
    """
    Split a target such as ``Ender 3 V2`` into family ``Ender 3`` and
    variant ``V2``. This is intentionally conservative: if the final token
    does not look version/model-like, keep the complete target as the family.
    """
    if len(target) >= 2 and _looks_like_model_variant(target[-1]):
        return target[:-1], target[-1]
    return target, ""


def _subject_family_status(tokens: list[str], target: list[str]) -> str:
    """
    Classify the model identity expressed by title/URL tokens only.

    Returns one of: exact, related, mismatch, unknown.
    Snippet text is deliberately excluded here because a page about another
    product can mention the target in passing.
    """
    target_index = _sequence_index(tokens, target)
    if target_index >= 0:
        after_index = target_index + len(target)
        next_token = tokens[after_index] if after_index < len(tokens) else ""
        if next_token in MODEL_VARIANT_MARKERS and next_token not in target:
            following = tokens[after_index + 1] if after_index + 1 < len(tokens) else ""
            if next_token != "max" or following not in ASPECT_TOKENS:
                return "related"
        return "exact"

    family, target_variant = _target_family_tokens(target)
    family_index = _sequence_index(tokens, family)
    if family_index >= 0:
        after_index = family_index + len(family)
        next_token = tokens[after_index] if after_index < len(tokens) else ""

        # A dedicated sibling model page (V3, Neo, Max, etc.) is a mismatch.
        if next_token and _looks_like_model_variant(next_token):
            if not target_variant or next_token != target_variant:
                return "mismatch"

        # A family/category page can be relevant for discovery but is not
        # dedicated enough to prove model-specific technical values.
        return "related"

    # Same named product family followed immediately by a different model
    # number is a mismatch, e.g. Ender-5 Max for target Ender-3 V2. Do not
    # inspect arbitrary later numbers: generic titles such as "Ender Series 3D
    # Printers" contain digits that are not product identities.
    alpha_family = [token for token in family if not any(ch.isdigit() for ch in token)]
    if alpha_family:
        alpha_index = _sequence_index(tokens, alpha_family)
        if alpha_index >= 0:
            after_alpha = alpha_index + len(alpha_family)
            next_token = tokens[after_alpha] if after_alpha < len(tokens) else ""
            target_number = next(
                (token for token in family if any(ch.isdigit() for ch in token)),
                "",
            )
            if (
                next_token
                and target_number
                and _looks_like_model_variant(next_token)
                and next_token != target_number
            ):
                return "mismatch"

    return "unknown"


def _target_regex(target: list[str]) -> re.Pattern[str]:
    joined = r"[-\s_/]*".join(re.escape(token) for token in target)
    return re.compile(rf"\b{joined}\b", re.IGNORECASE)


def _snippet_has_target_technical_context(
    request: GenerationRequest,
    source: ResearchSource,
    target: list[str],
) -> bool:
    """
    Official comparison/specification pages can support a target model when
    the snippet labels the target and then gives technical data for it.
    Values that appear only before a trailing target mention do not qualify.
    """
    snippet = " ".join(part for part in (source.snippet, source.content_excerpt) if part).casefold()
    match = _target_regex(target).search(snippet)
    if not match:
        return False

    after = snippet[match.start(): min(len(snippet), match.end() + 320)]
    has_aspect = any(token in after for token in ASPECT_TOKENS)
    has_value = bool(
        TEMPERATURE_RE.search(after)
        or VERSION_RE.search(after)
        or SPEED_RE.search(after)
        or DIMENSION_RE.search(after)
    )
    return has_aspect and has_value


def _source_identity_classification(
    request: GenerationRequest,
    source: ResearchSource,
    *,
    brand_hint: str = "",
) -> tuple[str, str]:
    target = [token.casefold() for token in _product_identity_terms(request)]
    if not target or not any(any(ch.isdigit() for ch in token) for token in target):
        return "unknown", "none"

    title_tokens = _normalized_identity_tokens(source.title)
    url_tokens = _normalized_identity_tokens(urlparse(source.url).path)
    snippet_tokens = _normalized_identity_tokens(" ".join(part for part in (source.snippet, source.content_excerpt) if part))
    subject_tokens = title_tokens + url_tokens

    title_status = _subject_family_status(title_tokens, target)
    url_status = _subject_family_status(url_tokens, target)

    if "exact" in {title_status, url_status}:
        if "mismatch" in {title_status, url_status}:
            return "related", "title_url"
        return "exact", "title_url"

    # A page dedicated to a sibling model stays a mismatch even when it names
    # the target model in a comparison sentence.
    if "mismatch" in {title_status, url_status}:
        return "mismatch", "family"

    snippet_has_target = _sequence_index(snippet_tokens, target) >= 0

    # Official comparison/specification pages may cover many products. If the
    # snippet explicitly names the target and immediately presents technical
    # values, those values are target-specific evidence.
    if (
        snippet_has_target
        and _is_primary_like(source, brand_hint, request)
        and _snippet_has_target_technical_context(request, source, target)
    ):
        return "exact", "snippet_context"

    if "related" in {title_status, url_status}:
        return "related", "title_url"

    if snippet_has_target:
        return "related", "snippet"

    combined = subject_tokens + snippet_tokens
    if all(token in combined for token in target):
        return "related", "snippet"

    return "unknown", "none"


def _source_identity_match(
    request: GenerationRequest,
    source: ResearchSource,
    *,
    brand_hint: str = "",
) -> str:
    return _source_identity_classification(request, source, brand_hint=brand_hint)[0]


def _source_identity_basis(
    request: GenerationRequest,
    source: ResearchSource,
    *,
    brand_hint: str = "",
) -> str:
    return _source_identity_classification(request, source, brand_hint=brand_hint)[1]


def _manufacturer_domain_hint(sources: list[ResearchSource], brand_hint: str) -> str:
    if not brand_hint:
        return ""
    brand = brand_hint.casefold()
    candidates: list[str] = []
    for source in sources:
        host = _hostname(source.url)
        if not host or brand not in host or any(part in host for part in LOWER_TRUST_HOST_PARTS):
            continue
        labels = host.split(".")
        if len(labels) >= 2:
            candidates.append(".".join(labels[-2:]))
        else:
            candidates.append(host)
    if not candidates:
        return ""
    return sorted(set(candidates), key=lambda value: (len(value), value))[0]


def _probable_brand_domain(brand_hint: str) -> str:
    """Best-effort retrieval hint only; never treated as proof of ownership."""
    if not brand_hint:
        return ""
    normalized = unicodedata.normalize("NFKD", brand_hint).encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^a-z0-9]+", "", normalized.casefold())
    if 3 <= len(slug) <= 40:
        return f"{slug}.com"
    return ""


def _relevant_excerpt(request: GenerationRequest, text: str, max_chars: int) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return ""
    max_chars = max(400, min(6000, int(max_chars)))
    lowered = text.casefold()
    needles: list[str] = []
    for requirement in _research_requirements(request):
        needles.extend(requirement.subject_terms)
    identity = _target_identity(request).casefold()
    if identity:
        needles.append(identity)
    needles.extend(MAXIMUM_MARKERS)

    positions: list[int] = []
    for needle in needles:
        start = 0
        while needle and len(positions) < 20:
            idx = lowered.find(needle.casefold(), start)
            if idx < 0:
                break
            positions.append(idx)
            start = idx + len(needle)
    if not positions:
        return text[:max_chars]

    windows: list[tuple[int, int]] = []
    for pos in sorted(set(positions)):
        windows.append((max(0, pos - 350), min(len(text), pos + 650)))
    merged: list[tuple[int, int]] = []
    for start, end in windows:
        if merged and start <= merged[-1][1] + 80:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    excerpt = " … ".join(text[a:b].strip() for a, b in merged)
    return excerpt[:max_chars].strip()


def plan_followup_queries(
    request: GenerationRequest,
    sources: list[ResearchSource],
    missing_aspects: list[str],
    max_queries: int = 1,
) -> list[str]:
    if max_queries <= 0 or not missing_aspects:
        return []

    language = request.language.casefold().split("-", 1)[0]
    brand = _infer_brand_hint(request, sources)
    identity = _quoted_target_identity(request)
    missing = " ".join(missing_aspects[:3])
    domain_hint = _manufacturer_domain_hint(sources, brand) or _probable_brand_domain(brand)
    site_filter = f"site:{domain_hint}" if domain_hint else ""

    # Manufacturer documentation is very often indexed in English even when
    # the requested conversation language is not English. For explicit primary
    # source requests, use canonical English specification labels so SearXNG is
    # more likely to surface the actual product/specification page instead of
    # German mirrors and reseller summaries.
    canonical_terms: list[str] = []
    for aspect in missing_aspects[:3]:
        folded = aspect.casefold()
        if "düsentemperatur" in folded or "duesentemperatur" in folded:
            canonical_terms.append('"Max. Nozzle Temperature"')
        elif "heizbetttemperatur" in folded or "druckbetttemperatur" in folded:
            canonical_terms.append('"Max. Heatbed Temperature"')
        elif "firmware" in folded:
            canonical_terms.append('"Firmware Version"')
        elif "druckgeschwindigkeit" in folded:
            canonical_terms.append('"Max. Printing Speed"')
        elif "bauraum" in folded:
            canonical_terms.append('"Build Volume"')

    if _topic_requests_primary_source(request) and canonical_terms:
        prefix = " ".join(
            part for part in (
                identity, brand, " ".join(canonical_terms),
                "official specifications", site_filter,
            ) if part
        ).strip()
    else:
        suffix = "Hersteller offizielle Spezifikationen" if language == "de" else "manufacturer official specifications"
        prefix = " ".join(
            part for part in (identity, brand, missing, suffix, site_filter) if part
        ).strip()

    return _dedupe_and_cap_queries([prefix], max_queries)


def _source_score(
    request: GenerationRequest,
    source: ResearchSource,
    *,
    brand_hint: str = "",
) -> float:
    title = source.title.casefold()
    snippet = " ".join(part for part in (source.snippet, source.content_excerpt) if part).casefold()
    url = source.url.casefold()
    host = _hostname(source.url)
    score = 0.0

    for term in _topic_relevance_terms(request):
        if term in title:
            score += 3.0
        elif term in url:
            score += 2.0
        elif term in snippet:
            score += 1.0

    for hint in PRIMARY_TEXT_HINTS:
        if hint in title:
            score += 2.5
        elif hint in snippet:
            score += 1.0
    if any(hint in url for hint in PRIMARY_PATH_HINTS):
        score += 3.0

    # A manufacturer token inferred from multiple titles is only a retrieval
    # hint. If it also appears in the hostname, that is strong evidence that
    # the result is from the manufacturer's own domain.
    if brand_hint and brand_hint.casefold() in host:
        score += 12.0

    primary_policy = _topic_primary_policy(request)
    if _is_primary_like(source, brand_hint, request):
        if primary_policy == "require":
            score += 18.0
        elif primary_policy == "prefer":
            score += 8.0

    identity_match = _source_identity_match(request, source, brand_hint=brand_hint)
    if identity_match == "exact":
        score += 20.0
    elif identity_match == "related":
        score += 1.0
    elif identity_match == "mismatch":
        score -= 30.0

    if any(part in host for part in LOWER_TRUST_HOST_PARTS):
        score -= 5.0

    score += min(len(source.snippet), 800) / 800.0
    if source.fetched and source.content_excerpt:
        score += 2.0
    return score


def _rank_sources(
    request: GenerationRequest,
    sources: list[ResearchSource],
    *,
    brand_hint: str = "",
) -> list[ResearchSource]:
    return sorted(
        sources,
        key=lambda source: (
            _source_score(request, source, brand_hint=brand_hint),
            len(source.snippet),
        ),
        reverse=True,
    )


class SearXNGProvider:
    def __init__(self, base_url: str, timeout_seconds: float = 10.0, safe_search: int = 1) -> None:
        base_url = base_url.strip()
        if not base_url:
            raise ResearchUnavailable("Research is enabled but AY_SEARXNG_BASE_URL is not configured.")
        parsed = urlparse(base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ResearchUnavailable("AY_SEARXNG_BASE_URL must be an http(s) URL.")
        self.base_url = base_url.rstrip("/") + "/"
        self.timeout_seconds = timeout_seconds
        self.safe_search = max(0, min(2, int(safe_search)))

    async def search(self, query: str, *, language: str, limit: int) -> list[ResearchSource]:
        endpoint = urljoin(self.base_url, "search")
        params = {
            "q": query,
            "format": "json",
            "language": language,
            "safesearch": str(self.safe_search),
        }
        try:
            async with httpx.AsyncClient(timeout=self.timeout_seconds, follow_redirects=False) as client:
                response = await client.get(endpoint, params=params, headers={"Accept": "application/json"})
                response.raise_for_status()
                payload = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise ResearchError(f"SearXNG search failed: {exc}") from exc

        raw_results = payload.get("results", []) if isinstance(payload, dict) else []
        sources: list[ResearchSource] = []
        seen_urls: set[str] = set()

        for item in raw_results:
            if not isinstance(item, dict):
                continue
            url = str(item.get("url") or "").strip()
            title = _plain_text(item.get("title"), 300)
            snippet = _plain_text(item.get("content") or item.get("snippet"), 1800)
            if not url or not title or not snippet or url in seen_urls:
                continue
            if urlparse(url).scheme not in {"http", "https"}:
                continue
            seen_urls.add(url)
            engine = item.get("engine") or ""
            if not engine and isinstance(item.get("engines"), list) and item["engines"]:
                engine = str(item["engines"][0])
            sources.append(
                ResearchSource(
                    id=f"S{len(sources) + 1}",
                    title=title,
                    url=url[:2000],
                    snippet=snippet,
                    engine=_plain_text(engine, 80),
                )
            )
            if len(sources) >= limit:
                break
        return sources


class ResearchService:
    def __init__(
        self,
        provider: SearchProvider | None = None,
        fetcher: SourceFetcherProtocol | None = None,
    ) -> None:
        self._provider = provider
        self._fetcher = fetcher

    def _provider_for_settings(self) -> SearchProvider:
        if self._provider is not None:
            return self._provider
        if settings.research_provider.casefold() != "searxng":
            raise ResearchUnavailable(f"Unsupported research provider: {settings.research_provider}")
        return SearXNGProvider(
            settings.searxng_base_url,
            settings.research_timeout_seconds,
            settings.research_safe_search,
        )

    def _fetcher_for_settings(self) -> SourceFetcherProtocol:
        if self._fetcher is not None:
            return self._fetcher
        return SafeSourceFetcher(
            timeout_seconds=settings.research_fetch_timeout_seconds,
            max_bytes=settings.research_fetch_max_bytes,
            max_redirects=settings.research_fetch_max_redirects,
            pdf_max_pages=settings.research_fetch_pdf_max_pages,
        )

    async def _fetch_best_sources(
        self,
        request: GenerationRequest,
        candidates: list[ResearchSource],
        *,
        brand_hint: str,
    ) -> tuple[int, float]:
        if not settings.research_fetch_enabled or settings.research_fetch_max_sources <= 0:
            return 0, 0.0

        ranked = _rank_sources(request, candidates, brand_hint=brand_hint)
        exact = [
            source for source in ranked
            if _source_identity_match(request, source, brand_hint=brand_hint) == "exact"
        ]
        primary = [
            source for source in ranked
            if _is_primary_like(source, brand_hint, request)
        ]

        # Fetching is an evidence step, not an identity-only step. For generic
        # science/software questions there may be no exact product page at all.
        # Prefer primary sources, then exact-identity sources, then the best
        # remaining candidates; keep stable uniqueness by URL.
        selected: list[ResearchSource] = []
        for pool in (primary, exact, ranked):
            for source in pool:
                if any(existing.url == source.url for existing in selected):
                    continue
                selected.append(source)
                if len(selected) >= settings.research_fetch_max_sources:
                    break
            if len(selected) >= settings.research_fetch_max_sources:
                break

        fetcher = self._fetcher_for_settings()
        started = time.perf_counter()
        fetched = 0
        for source in selected:
            try:
                document = await fetcher.fetch(source.url)
            except SourceFetchBlocked:
                source.fetch_status = "skipped"
                continue
            except SourceFetchError:
                source.fetch_status = "failed"
                continue
            except Exception:
                # A single broken page must not take the whole research request down.
                source.fetch_status = "failed"
                continue

            excerpt = _relevant_excerpt(
                request,
                document.text,
                settings.research_fetch_max_chars,
            )
            if not excerpt:
                source.fetch_status = "failed"
                continue
            source.url = document.url[:2000]
            source.content_excerpt = excerpt[:6000]
            source.fetched = True
            source.fetch_status = "fetched"
            fetched += 1

        return fetched, time.perf_counter() - started

    async def reverse_search(
        self,
        query: str,
        *,
        language: str,
        max_sources: int = 3,
    ) -> tuple[list[ResearchSource], float]:
        """Generic claim-verification search used by the final audit.

        This intentionally does not use domain-specific aspect/identity heuristics.
        It discovers a few web results, safely fetches the best textual sources,
        and returns compact evidence for semantic adjudication.
        """
        query = re.sub(r"\s+", " ", query).strip()[:300]
        if not query:
            return [], 0.0

        provider = self._provider_for_settings()
        started = time.perf_counter()
        results = await provider.search(
            query,
            language=language,
            limit=max(8, max_sources * 4),
        )
        if not results:
            return [], time.perf_counter() - started

        selected = results[: max(1, max_sources)]
        if settings.research_fetch_enabled:
            fetcher = self._fetcher_for_settings()
            query_terms = [
                token.casefold() for token in TOKEN_RE.findall(query)
                if len(token) >= 4 and token.casefold() not in STOPWORDS_DE and token.casefold() not in STOPWORDS_EN
            ][:12]

            for source in selected:
                try:
                    document = await fetcher.fetch(source.url)
                except SourceFetchBlocked:
                    source.fetch_status = "skipped"
                    continue
                except Exception:
                    source.fetch_status = "failed"
                    continue

                text = re.sub(r"\s+", " ", document.text).strip()
                if not text:
                    source.fetch_status = "failed"
                    continue

                folded = text.casefold()
                positions = [folded.find(term) for term in query_terms if term in folded]
                positions = [pos for pos in positions if pos >= 0]
                if positions:
                    center = min(positions)
                    start = max(0, center - 700)
                    end = min(len(text), start + settings.research_fetch_max_chars)
                    excerpt = text[start:end]
                else:
                    excerpt = text[: settings.research_fetch_max_chars]

                source.url = document.url[:2000]
                source.content_excerpt = excerpt[:6000]
                source.fetched = True
                source.fetch_status = "fetched"

        for index, source in enumerate(selected, start=1):
            source.id = f"R{index}"
            source.identity_match = "unknown"
            source.identity_basis = "none"

        return selected, time.perf_counter() - started

    async def research(self, request: GenerationRequest) -> ResearchBundle:
        if not request.features.research:
            return ResearchBundle(provider="", queries=[], sources=[], elapsed_seconds=0.0)

        provider = self._provider_for_settings()
        initial_queries = plan_research_queries(request, settings.research_max_queries)
        all_queries = list(initial_queries)
        followup_queries: list[str] = []
        started = time.perf_counter()

        candidate_limit = max(30, settings.research_max_sources * 8)
        candidates: list[ResearchSource] = []
        seen: set[str] = set()

        async def run_query(query: str) -> None:
            results = await provider.search(
                query,
                language=request.language,
                limit=candidate_limit,
            )
            for source in results:
                if source.url in seen:
                    continue
                seen.add(source.url)
                candidates.append(source)

        for query in initial_queries:
            await run_query(query)

        if not candidates:
            raise ResearchError("Research was enabled, but SearXNG returned no usable sources.")

        brand_hint = _infer_brand_hint(request, candidates)
        coverage = assess_evidence_coverage(request, candidates, brand_hint=brand_hint)

        if not coverage.sufficient and settings.research_max_followup_queries > 0:
            unresolved_aspects = list(dict.fromkeys(
                coverage.missing_aspects + coverage.conflicting_aspects
            ))
            followup_queries = plan_followup_queries(
                request,
                candidates,
                unresolved_aspects,
                settings.research_max_followup_queries,
            )
            for query in followup_queries:
                if query.casefold() in {existing.casefold() for existing in all_queries}:
                    continue
                all_queries.append(query)
                await run_query(query)

            brand_hint = _infer_brand_hint(request, candidates) or brand_hint

        # Search snippets are discovery evidence. For the strongest exact-identity
        # results, retrieve the actual page/PDF and reduce it to a small relevant
        # excerpt before the final coverage decision.
        fetch_count, fetch_seconds = await self._fetch_best_sources(
            request,
            candidates,
            brand_hint=brand_hint,
        )

        ranked = _rank_sources(
            request,
            candidates,
            brand_hint=brand_hint,
        )[: settings.research_max_sources]
        for index, source in enumerate(ranked, start=1):
            source.id = f"S{index}"
            source.authority = (
                "primary"
                if _is_primary_like(source, brand_hint, request)
                else "secondary"
            )
            source.identity_match = _source_identity_match(request, source, brand_hint=brand_hint)
            source.identity_basis = _source_identity_basis(request, source, brand_hint=brand_hint)

        final_coverage = assess_evidence_coverage(
            request,
            ranked,
            brand_hint=brand_hint,
        )
        verified_facts = extract_verified_facts(
            request,
            ranked,
            brand_hint=brand_hint,
        )
        elapsed = time.perf_counter() - started

        return ResearchBundle(
            provider=settings.research_provider,
            queries=all_queries,
            sources=ranked,
            elapsed_seconds=elapsed,
            evidence_sufficient=final_coverage.sufficient,
            missing_aspects=final_coverage.missing_aspects,
            conflicting_aspects=final_coverage.conflicting_aspects,
            secondary_supported_aspects=final_coverage.secondary_supported_aspects,
            verified_facts=verified_facts,
            target_identity=_target_identity(request),
            followup_queries=followup_queries,
            fetch_count=fetch_count,
            fetch_seconds=fetch_seconds,
        )

