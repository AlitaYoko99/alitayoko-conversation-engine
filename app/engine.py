from __future__ import annotations

from dataclasses import dataclass
import logging

from app.config import settings
from app.content_normalizer import (
    ContentAdjustmentCounts,
    normalize_content_features,
)
from app.feature_normalizer import (
    FeatureAdjustmentCounts,
    normalize_presentation_features,
)
from app.feature_policy import count_feature_usage
from app.final_audit import FinalAuditor, FinalAuditStats
from app.model_profiles import ModelProfile, profile_for_model
from app.models import (
    ConversationResult,
    FeatureAdjustments,
    FeatureUsage,
    GenerationMeta,
    FinalAuditMeta,
    GenerationRequest,
    GenerationResponse,
    ResearchFact,
    ResearchMeta,
    ValidationViolation,
)
from app.ollama_client import OllamaClient, OllamaResult
from app.research import ResearchBundle, ResearchService
from app.research_fact_enforcer import (
    enforce_verified_research_facts,
    neutralize_complete_research_uncertainty,
)
from app.prompts import (
    build_repair_prompt,
    build_system_prompt,
    build_user_prompt,
    schema_grounding_text,
)
from app.schema_builder import build_conversation_schema
from app.validator import validate_semantics
from app.wire import WireFormatError, parse_wire_result, result_to_wire_json


logger = logging.getLogger(__name__)


class GenerationFailed(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        violations: list[ValidationViolation] | None = None,
        raw_output: str | None = None,
    ) -> None:
        super().__init__(message)
        self.violations = violations or []
        self.raw_output = raw_output


@dataclass(slots=True)
class ParsedAttempt:
    result: ConversationResult | None
    violations: list[ValidationViolation]
    presentation_adjustments: FeatureAdjustmentCounts
    content_adjustments: ContentAdjustmentCounts
    parse_error: str | None = None


@dataclass(slots=True)
class AdjustmentTotals:
    typing_added: int = 0
    typing_removed: int = 0
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

    def add(self, parsed: ParsedAttempt) -> None:
        self.typing_added += (
            parsed.presentation_adjustments.typing_indicators_added
        )
        self.typing_removed += (
            parsed.presentation_adjustments.typing_indicators_removed
        )
        self.images_added += parsed.content_adjustments.images_added
        self.narrations_added += parsed.content_adjustments.narrations_added
        self.links_added += parsed.content_adjustments.links_added
        self.links_sanitized += parsed.content_adjustments.links_sanitized
        self.links_removed += parsed.content_adjustments.links_removed
        self.technical_specifics_neutralized += (
            parsed.content_adjustments.technical_specifics_neutralized
        )
        self.source_claims_neutralized += (
            parsed.content_adjustments.source_claims_neutralized
        )
        self.unsupported_absence_claims_neutralized += (
            parsed.content_adjustments.unsupported_absence_claims_neutralized
        )
        self.unsupported_source_mechanisms_neutralized += (
            parsed.content_adjustments.unsupported_source_mechanisms_neutralized
        )
        self.dangling_value_references_rewritten += (
            parsed.content_adjustments.dangling_value_references_rewritten
        )
        self.speaker_prefixes_removed += (
            parsed.content_adjustments.speaker_prefixes_removed
        )
        self.wire_artifacts_removed += (
            parsed.content_adjustments.wire_artifacts_removed
        )
        self.attachment_queries_refined += (
            parsed.content_adjustments.attachment_queries_refined
        )
        self.normalization_duplicates_avoided += (
            parsed.content_adjustments.normalization_duplicates_avoided
        )
        self.entries_shortened += parsed.content_adjustments.entries_shortened


def temperature_from_chaos(chaos: int) -> float:
    # Non-linear mapping keeps low/medium chaos materially more conservative
    # while preserving creative headroom at the top of the 0..100 scale.
    normalized = chaos / 100
    return round(0.20 + 0.65 * (normalized ** 1.4), 3)


def num_predict_for(request: GenerationRequest) -> int:
    # Budget for the *preferred* bubble length rather than the hard maximum.
    # There is still ample headroom for German tokenisation and JSON syntax.
    preferred_words = min(request.limits.max_words_per_entry, 18)

    attachment_allowance = 0
    if request.features.images:
        attachment_allowance += request.limits.max_images * 60
    if request.features.links:
        attachment_allowance += request.limits.max_links * 45

    estimate = (
        140
        + int(request.entry_count * preferred_words * 1.75)
        + attachment_allowance
    )
    return max(520, min(2400, estimate))


def parse_and_validate(
    request: GenerationRequest,
    content: str,
    factual_context: str = "",
    required_research_facts: list[ResearchFact] | None = None,
) -> ParsedAttempt:
    try:
        result = parse_wire_result(request, content)
    except (WireFormatError, ValueError) as exc:
        return ParsedAttempt(
            result=None,
            violations=[],
            presentation_adjustments=FeatureAdjustmentCounts(),
            content_adjustments=ContentAdjustmentCounts(),
            parse_error=str(exc),
        )

    content_adjustments = normalize_content_features(request, result, factual_context)
    presentation_adjustments = normalize_presentation_features(request, result)
    violations = validate_semantics(
        request,
        result,
        factual_context,
        required_research_facts=required_research_facts,
    )

    return ParsedAttempt(
        result=result,
        violations=violations,
        presentation_adjustments=presentation_adjustments,
        content_adjustments=content_adjustments,
        parse_error=None,
    )


def _seconds(value: int | None) -> float | None:
    return value / 1_000_000_000 if value is not None else None


def _sum_optional(values: list[int | None]) -> int | None:
    actual = [value for value in values if value is not None]
    return sum(actual) if actual else None


class ConversationEngine:
    def __init__(self, research_service: ResearchService | None = None) -> None:
        self.client = OllamaClient(
            settings.ollama_base_url,
            settings.request_timeout_seconds,
        )
        self.research_service = research_service or ResearchService()

    def _system_content(
        self,
        system_prompt: str,
        schema: dict,
        *,
        force_grounding: bool = False,
    ) -> str:
        if settings.ground_schema_in_prompt or force_grounding:
            return system_prompt + "\n\n" + schema_grounding_text(schema)
        return system_prompt

    async def generate(
        self,
        request: GenerationRequest,
        *,
        profile: ModelProfile | None = None,
    ) -> GenerationResponse:
        profile = profile or profile_for_model(settings.ollama_model)
        model_name = profile.model

        research_bundle = ResearchBundle(
            provider="", queries=[], sources=[], elapsed_seconds=0.0
        )
        if request.features.research:
            research_bundle = await self.research_service.research(request)
        research_evidence = research_bundle.evidence_text
        generation_evidence = research_bundle.generation_evidence_text

        schema = build_conversation_schema(request)
        temperature = temperature_from_chaos(request.chaos)
        base_num_predict = num_predict_for(request)
        num_predict = min(
            4096,
            max(
                base_num_predict,
                int(base_num_predict * profile.num_predict_multiplier),
            ),
        )
        system_prompt = build_system_prompt(
            request,
            generation_evidence,
            research_evidence_sufficient=(
                research_bundle.evidence_sufficient
                if request.features.research
                else None
            ),
            research_missing_aspects=research_bundle.missing_aspects,
            research_conflicting_aspects=research_bundle.conflicting_aspects,
            research_secondary_supported_aspects=research_bundle.secondary_supported_aspects,
            research_verified_facts=research_bundle.verified_facts,
            research_target_identity=research_bundle.target_identity,
        )

        # Verified facts are aspect-local. A globally incomplete research bundle
        # may still contain one or more facts that are independently supported.
        # Ground those facts instead of discarding them just because another
        # requested aspect remains open.
        required_research_facts = (
            list(research_bundle.verified_facts)
            if request.features.research
            else []
        )

        generation_attempts = 1
        repair_attempts = 0
        calls: list[OllamaResult] = []
        adjustments = AdjustmentTotals()

        ollama_result = await self.client.chat(
            model=model_name,
            messages=[
                {
                    "role": "system",
                    "content": self._system_content(
                        system_prompt,
                        schema,
                        force_grounding=profile.force_schema_grounding,
                    ),
                },
                {
                    "role": "user",
                    "content": build_user_prompt(request),
                },
            ],
            schema=schema,
            temperature=temperature,
            num_ctx=settings.num_ctx,
            num_predict=num_predict,
            keep_alive=settings.keep_alive,
            think=profile.ollama_think,
        )
        calls.append(ollama_result)

        parsed = parse_and_validate(request, ollama_result.content, research_evidence, required_research_facts)
        adjustments.add(parsed)

        if parsed.parse_error:
            logger.warning(
                "Conversation generation parse failure: error=%s done_reason=%s generated_tokens=%s content_chars=%s",
                parsed.parse_error,
                ollama_result.done_reason,
                ollama_result.generated_tokens,
                len(ollama_result.content or ""),
            )
            # Regenerate once with explicit schema grounding and the full safe
            # output budget. The retry uses the compact post-evidence prompt,
            # never the raw search/fetch corpus.
            generation_attempts += 1
            retry_num_predict = 4096
            ollama_result = await self.client.chat(
                model=model_name,
                messages=[
                    {
                        "role": "system",
                        "content": self._system_content(
                            system_prompt,
                            schema,
                            force_grounding=True,
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            "Generate a fresh complete compact conversation. "
                            "The previous output could not be parsed. Return only the JSON object."
                        ),
                    },
                ],
                schema=schema,
                temperature=max(0.10, temperature - 0.15),
                num_ctx=settings.num_ctx,
                num_predict=retry_num_predict,
                keep_alive=settings.keep_alive,
                think=profile.ollama_think,
            )
            calls.append(ollama_result)
            parsed = parse_and_validate(request, ollama_result.content, research_evidence, required_research_facts)
            adjustments.add(parsed)

        if parsed.parse_error:
            logger.warning(
                "Conversation regeneration parse failure: error=%s done_reason=%s generated_tokens=%s content_chars=%s",
                parsed.parse_error,
                ollama_result.done_reason,
                ollama_result.generated_tokens,
                len(ollama_result.content or ""),
            )
            raise GenerationFailed(
                "The model did not return a parseable conversation.",
                raw_output=ollama_result.content,
            )

        assert parsed.result is not None

        def apply_verified_research_grounding(attempt: ParsedAttempt) -> ParsedAttempt:
            if not required_research_facts or attempt.result is None:
                return attempt

            injected, rewritten, recommendation_rewrites = enforce_verified_research_facts(
                request,
                attempt.result,
                required_research_facts,
                target_identity=research_bundle.target_identity,
                missing_aspects=research_bundle.missing_aspects,
            )
            adjustments.verified_research_facts_injected += injected
            adjustments.stale_research_uncertainty_rewritten += rewritten
            adjustments.unsupported_recommendations_neutralized += recommendation_rewrites

            if not (injected or rewritten or recommendation_rewrites):
                return attempt

            grounded_json = result_to_wire_json(attempt.result)
            grounded = parse_and_validate(
                request,
                grounded_json,
                research_evidence,
                required_research_facts,
            )
            adjustments.add(grounded)
            return grounded

        # Verified Research is deterministic data, not a creative repair task.
        # Ground missing facts and remove stale "still unknown" filler
        # before spending an additional LLM call on semantic repair.
        parsed = apply_verified_research_grounding(parsed)

        repair_trigger_violations: list[ValidationViolation] = []

        while parsed.violations and repair_attempts < settings.max_repair_attempts:
            repair_trigger_violations = list(parsed.violations)
            repair_attempts += 1

            repair_result = await self.client.chat(
                model=model_name,
                messages=[
                    {
                        "role": "system",
                        "content": self._system_content(system_prompt, schema),
                    },
                    {
                        "role": "user",
                        "content": build_repair_prompt(
                            request,
                            result_to_wire_json(parsed.result),
                            parsed.violations,
                        ),
                    },
                ],
                schema=schema,
                temperature=min(temperature, 0.35),
                num_ctx=settings.num_ctx,
                num_predict=num_predict,
                keep_alive=settings.keep_alive,
                think=profile.ollama_think,
            )
            calls.append(repair_result)

            ollama_result = repair_result
            parsed = parse_and_validate(request, repair_result.content, research_evidence, required_research_facts)
            adjustments.add(parsed)

            if parsed.parse_error:
                # A malformed repair may be retried by the loop only after a
                # fresh grounded regeneration; semantic repair cannot operate
                # without a parsed result.
                if repair_attempts >= settings.max_repair_attempts:
                    break

                repair_attempts += 1
                grounded_repair = await self.client.chat(
                    model=model_name,
                    messages=[
                        {
                            "role": "system",
                            "content": self._system_content(
                                system_prompt,
                                schema,
                                force_grounding=True,
                            ),
                        },
                        {
                            "role": "user",
                            "content": (
                                "Return a complete repaired compact conversation "
                                "matching the schema exactly."
                            ),
                        },
                    ],
                    schema=schema,
                    temperature=0.25,
                    num_ctx=settings.num_ctx,
                    num_predict=num_predict,
                    keep_alive=settings.keep_alive,
                    think=profile.ollama_think,
                )
                calls.append(grounded_repair)
                ollama_result = grounded_repair
                parsed = parse_and_validate(
                    request,
                    grounded_repair.content,
                    research_evidence,
                    required_research_facts,
                )
                adjustments.add(parsed)

        if parsed.parse_error or parsed.result is None:
            raise GenerationFailed(
                "The repair response could not be parsed.",
                violations=repair_trigger_violations,
                raw_output=ollama_result.content,
            )

        # A semantic repair may reintroduce stale uncertainty or drop a required
        # verified value, so enforce the same deterministic grounding once more
        # after the repair loop as a final coherence pass.
        parsed = apply_verified_research_grounding(parsed)

        if parsed.violations:
            raise GenerationFailed(
                "The conversation still violates semantic rules after repair.",
                violations=parsed.violations,
                raw_output=ollama_result.content,
            )

        final_audit_stats = FinalAuditStats()
        if settings.final_audit_enabled:
            auditor = FinalAuditor(client=self.client, research_service=self.research_service)
            audited = await auditor.run(
                request=request,
                conversation=parsed.result,
                research_bundle=research_bundle,
                model=model_name,
                num_ctx=settings.num_ctx,
                keep_alive=settings.keep_alive,
                think=profile.ollama_think,
            )
            final_audit_stats = audited.stats
            calls.extend(audited.calls)

            # Re-run the same deterministic normalizers and semantic validator
            # after the audit. Audit text changes are never trusted implicitly.
            audited_json = result_to_wire_json(audited.conversation)
            parsed = parse_and_validate(
                request,
                audited_json,
                research_evidence,
                required_research_facts,
            )
            adjustments.add(parsed)
            parsed = apply_verified_research_grounding(parsed)

            # v0.9.6: the language/coherence audit can reintroduce generic
            # "evidence is still insufficient" language after deterministic
            # grounding. Only when Research itself reports complete coverage do
            # we remove those stale claims in a final deterministic pass.
            if parsed.result is not None:
                complete_rewrites = neutralize_complete_research_uncertainty(
                    request,
                    parsed.result,
                    evidence_sufficient=research_bundle.evidence_sufficient,
                    missing_aspects=research_bundle.missing_aspects,
                    conflicting_aspects=research_bundle.conflicting_aspects,
                )
                if complete_rewrites:
                    adjustments.stale_research_uncertainty_rewritten += complete_rewrites
                    final_grounded_json = result_to_wire_json(parsed.result)
                    parsed = parse_and_validate(
                        request,
                        final_grounded_json,
                        research_evidence,
                        required_research_facts,
                    )
                    adjustments.add(parsed)

            if parsed.parse_error or parsed.result is None:
                raise GenerationFailed(
                    "The final audit produced an unparsable conversation.",
                    raw_output=audited_json,
                )
            if parsed.violations:
                raise GenerationFailed(
                    "The final audit conversation violates semantic rules.",
                    violations=parsed.violations,
                    raw_output=audited_json,
                )

        prompt_tokens = _sum_optional([call.prompt_tokens for call in calls])
        generated_tokens = _sum_optional([call.generated_tokens for call in calls])
        load_ns = _sum_optional([call.load_duration_ns for call in calls])
        prompt_eval_ns = _sum_optional(
            [call.prompt_eval_duration_ns for call in calls]
        )
        eval_ns = _sum_optional([call.eval_duration_ns for call in calls])
        total_ns = _sum_optional([call.total_duration_ns for call in calls])

        aggregate_tps = None
        if generated_tokens and eval_ns:
            eval_seconds = eval_ns / 1_000_000_000
            if eval_seconds > 0:
                aggregate_tps = generated_tokens / eval_seconds

        usage = count_feature_usage(parsed.result)

        return GenerationResponse(
            conversation=parsed.result,
            meta=GenerationMeta(
                model=model_name,
                model_profile=profile.id,
                wire_format="compact-v1",
                calls=len(calls),
                generation_attempts=generation_attempts,
                repair_attempts=repair_attempts,
                feature_usage=FeatureUsage(
                    narrations=usage.narrations,
                    typing_indicators=usage.typing_indicators,
                    images=usage.images,
                    links=usage.links,
                ),
                feature_adjustments=FeatureAdjustments(
                    typing_indicators_added=adjustments.typing_added,
                    typing_indicators_removed=adjustments.typing_removed,
                    images_added=adjustments.images_added,
                    narrations_added=adjustments.narrations_added,
                    links_added=adjustments.links_added,
                    links_sanitized=adjustments.links_sanitized,
                    links_removed=adjustments.links_removed,
                    technical_specifics_neutralized=(
                        adjustments.technical_specifics_neutralized
                    ),
                    source_claims_neutralized=adjustments.source_claims_neutralized,
                    unsupported_absence_claims_neutralized=(
                        adjustments.unsupported_absence_claims_neutralized
                    ),
                    unsupported_source_mechanisms_neutralized=(
                        adjustments.unsupported_source_mechanisms_neutralized
                    ),
                    dangling_value_references_rewritten=(
                        adjustments.dangling_value_references_rewritten
                    ),
                    speaker_prefixes_removed=adjustments.speaker_prefixes_removed,
                    wire_artifacts_removed=adjustments.wire_artifacts_removed,
                    attachment_queries_refined=adjustments.attachment_queries_refined,
                    normalization_duplicates_avoided=(
                        adjustments.normalization_duplicates_avoided
                    ),
                    entries_shortened=adjustments.entries_shortened,
                    verified_research_facts_injected=(
                        adjustments.verified_research_facts_injected
                    ),
                    stale_research_uncertainty_rewritten=(
                        adjustments.stale_research_uncertainty_rewritten
                    ),
                    unsupported_recommendations_neutralized=(
                        adjustments.unsupported_recommendations_neutralized
                    ),
                ),
                research=ResearchMeta(
                    enabled=request.features.research,
                    provider=research_bundle.provider,
                    queries=research_bundle.queries,
                    followup_queries=research_bundle.followup_queries,
                    evidence_sufficient=(
                        research_bundle.evidence_sufficient
                        if request.features.research
                        else None
                    ),
                    missing_aspects=research_bundle.missing_aspects,
                    conflicting_aspects=research_bundle.conflicting_aspects,
                    secondary_supported_aspects=research_bundle.secondary_supported_aspects,
                    verified_facts=research_bundle.verified_facts,
                    target_identity=research_bundle.target_identity,
                    source_count=len(research_bundle.sources),
                    sources=research_bundle.sources,
                    search_seconds=(
                        research_bundle.elapsed_seconds
                        if request.features.research
                        else None
                    ),
                    fetch_count=research_bundle.fetch_count,
                    fetch_seconds=(
                        research_bundle.fetch_seconds
                        if request.features.research
                        else None
                    ),
                ),
                final_audit=FinalAuditMeta(
                    enabled=settings.final_audit_enabled,
                    fact_audit_run=final_audit_stats.fact_audit_run,
                    language_audit_run=final_audit_stats.language_audit_run,
                    factual_claims_checked=final_audit_stats.factual_claims_checked,
                    existing_evidence_supported=final_audit_stats.existing_evidence_supported,
                    reverse_searches=final_audit_stats.reverse_searches,
                    reverse_sources=final_audit_stats.reverse_sources,
                    claims_corrected=final_audit_stats.claims_corrected,
                    unresolved_claims=final_audit_stats.unresolved_claims,
                    language_issues_found=final_audit_stats.language_issues_found,
                    language_rewrites=final_audit_stats.language_rewrites,
                    coherence_issues_found=final_audit_stats.coherence_issues_found,
                    coherence_rewrites=final_audit_stats.coherence_rewrites,
                    coherence_candidates=final_audit_stats.coherence_candidates,
                    language_rewrites_rejected=final_audit_stats.language_rewrites_rejected,
                    fact_rewrites_rejected=final_audit_stats.fact_rewrites_rejected,
                    audit_errors=final_audit_stats.audit_errors,
                    audit_error_details=final_audit_stats.audit_error_details,
                    audit_calls=final_audit_stats.audit_calls,
                    audit_seconds=(
                        final_audit_stats.audit_seconds
                        if settings.final_audit_enabled
                        else None
                    ),
                ),
                prompt_tokens=prompt_tokens,
                generated_tokens=generated_tokens,
                tokens_per_second=aggregate_tps,
                load_duration_seconds=_seconds(load_ns),
                prompt_eval_seconds=_seconds(prompt_eval_ns),
                generation_seconds=_seconds(eval_ns),
                total_duration_seconds=_seconds(total_ns),
            ),
        )
