from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator


NonEmptyText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
ParticipantId = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=32, pattern=r"^[A-Za-z0-9_-]+$"),
]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Participant(StrictModel):
    id: ParticipantId
    name: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=64)]
    personality: Annotated[str, StringConstraints(strip_whitespace=True, max_length=500)] = ""


class Features(StrictModel):
    narration: bool = True
    typing_indicator: bool = True
    images: bool = False
    links: bool = False
    research: bool = False


class Limits(StrictModel):
    max_narrations: int = Field(default=2, ge=0, le=20)
    max_words_per_entry: int = Field(default=35, ge=3, le=250)
    max_images: int = Field(default=0, ge=0, le=20)
    max_links: int = Field(default=0, ge=0, le=20)


class GenerationRequest(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    language: Annotated[
        str,
        StringConstraints(
            strip_whitespace=True,
            min_length=2,
            max_length=16,
            pattern=r"^[a-z]{2,3}(?:-[A-Za-z]{2,8})?$",
        ),
    ] = "de"
    topic: Annotated[str, StringConstraints(strip_whitespace=True, min_length=3, max_length=4000)]
    entry_count: int = Field(default=12, ge=2, le=100)
    tone: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=80)] = "natural"
    chaos: int = Field(default=35, ge=0, le=100)
    seriousness: int = Field(default=50, ge=0, le=100)
    participants: list[Participant] = Field(min_length=2, max_length=8)
    features: Features = Field(default_factory=Features)
    limits: Limits = Field(default_factory=Limits)

    @model_validator(mode="after")
    def validate_contract(self) -> "GenerationRequest":
        ids = [participant.id for participant in self.participants]
        names = [participant.name.casefold() for participant in self.participants]

        if len(ids) != len(set(ids)):
            raise ValueError("participant IDs must be unique")

        if "narrator" in {value.casefold() for value in ids}:
            raise ValueError("'narrator' is a reserved participant ID")

        if len(names) != len(set(names)):
            raise ValueError("participant names must be unique")

        if self.entry_count < len(self.participants):
            raise ValueError(
                "entry_count must be at least the number of participants "
                "because every participant must speak at least once"
            )

        if not self.features.narration and self.limits.max_narrations != 0:
            raise ValueError("max_narrations must be 0 when narration is disabled")

        if self.features.narration and self.limits.max_narrations < 1:
            raise ValueError("max_narrations must be greater than 0 when narration is enabled")

        minimum_entries = len(self.participants) + (1 if self.features.narration else 0)
        if self.entry_count < minimum_entries:
            raise ValueError(
                "entry_count is too small for the enabled features: every participant "
                "must speak and enabled narration requires at least one narration entry"
            )

        if not self.features.images and self.limits.max_images != 0:
            raise ValueError("max_images must be 0 when images are disabled")

        if not self.features.links and self.limits.max_links != 0:
            raise ValueError("max_links must be 0 when links are disabled")

        if self.features.images and self.limits.max_images == 0:
            raise ValueError("max_images must be greater than 0 when images are enabled")

        if self.features.links and self.limits.max_links == 0:
            raise ValueError("max_links must be greater than 0 when links are enabled")

        return self


class NoAttachment(StrictModel):
    type: Literal["none"] = "none"


class ImageAttachment(StrictModel):
    type: Literal["image"]
    description: Annotated[str, StringConstraints(strip_whitespace=True, min_length=3, max_length=500)]
    alt_text: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=250)]


class LinkAttachment(StrictModel):
    type: Literal["link"]
    query: Annotated[str, StringConstraints(strip_whitespace=True, min_length=2, max_length=300)]
    label: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=120)]


Attachment = Annotated[
    NoAttachment | ImageAttachment | LinkAttachment,
    Field(discriminator="type"),
]


class ConversationEntry(StrictModel):
    type: Literal["message", "narration"]
    speaker_id: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=32)]
    text: NonEmptyText
    typing_indicator: bool
    attachment: Attachment


class ConversationResult(StrictModel):
    schema_version: Literal["1.0"]
    language: str
    entries: list[ConversationEntry]


class ValidationViolation(StrictModel):
    code: str
    message: str
    entry_indices: list[int] = Field(default_factory=list)
    expected: str | int | None = None
    actual: str | int | None = None


class FeatureUsage(StrictModel):
    narrations: int = 0
    typing_indicators: int = 0
    images: int = 0
    links: int = 0


class FeatureAdjustments(StrictModel):
    typing_indicators_added: int = 0
    typing_indicators_removed: int = 0
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
    verified_research_facts_injected: int = 0
    stale_research_uncertainty_rewritten: int = 0
    unsupported_recommendations_neutralized: int = 0


class ResearchSource(StrictModel):
    id: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=16)]
    title: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=300)]
    url: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=2000)]
    snippet: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=2000)]
    engine: Annotated[str, StringConstraints(strip_whitespace=True, max_length=80)] = ""
    authority: Literal["primary", "secondary", "unknown"] = "unknown"
    identity_match: Literal["exact", "related", "unknown", "mismatch"] = "unknown"
    identity_basis: Literal["title_url", "snippet_context", "snippet", "family", "none"] = "none"
    fetched: bool = False
    fetch_status: Literal["not_attempted", "fetched", "skipped", "failed"] = "not_attempted"
    content_excerpt: Annotated[str, StringConstraints(strip_whitespace=True, max_length=6000)] = ""


class ResearchFact(StrictModel):
    aspect: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=120)]
    value: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=80)]
    confidence: Literal["primary", "secondary"]
    source_ids: list[str] = Field(default_factory=list)


class ResearchMeta(StrictModel):
    enabled: bool = False
    provider: str = ""
    queries: list[str] = Field(default_factory=list)
    followup_queries: list[str] = Field(default_factory=list)
    evidence_sufficient: bool | None = None
    missing_aspects: list[str] = Field(default_factory=list)
    conflicting_aspects: list[str] = Field(default_factory=list)
    secondary_supported_aspects: list[str] = Field(default_factory=list)
    verified_facts: list[ResearchFact] = Field(default_factory=list)
    target_identity: str = ""
    source_count: int = 0
    sources: list[ResearchSource] = Field(default_factory=list)
    search_seconds: float | None = None
    fetch_count: int = 0
    fetch_seconds: float | None = None


class FinalAuditMeta(StrictModel):
    enabled: bool = False
    fact_audit_run: bool = False
    language_audit_run: bool = False
    factual_claims_checked: int = 0
    existing_evidence_supported: int = 0
    reverse_searches: int = 0
    reverse_sources: int = 0
    claims_corrected: int = 0
    unresolved_claims: int = 0
    language_issues_found: int = 0
    language_rewrites: int = 0
    coherence_issues_found: int = 0
    coherence_rewrites: int = 0
    coherence_candidates: int = 0
    language_rewrites_rejected: int = 0
    fact_rewrites_rejected: int = 0
    audit_errors: int = 0
    audit_error_details: list[str] = Field(default_factory=list)
    audit_calls: int = 0
    audit_seconds: float | None = None


class GenerationMeta(StrictModel):
    model: str
    model_profile: str = "custom"
    wire_format: Literal["compact-v1"] = "compact-v1"
    calls: int
    generation_attempts: int
    repair_attempts: int
    feature_usage: FeatureUsage
    feature_adjustments: FeatureAdjustments
    research: ResearchMeta | None = None
    final_audit: FinalAuditMeta | None = None
    prompt_tokens: int | None = None
    generated_tokens: int | None = None
    tokens_per_second: float | None = None
    load_duration_seconds: float | None = None
    prompt_eval_seconds: float | None = None
    generation_seconds: float | None = None
    total_duration_seconds: float | None = None


class GenerationResponse(StrictModel):
    conversation: ConversationResult
    meta: GenerationMeta
