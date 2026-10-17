from __future__ import annotations

import json

import pytest

from app.config import settings
from app.final_audit import FinalAuditor
from app.models import (
    ConversationEntry,
    ConversationResult,
    Features,
    GenerationRequest,
    Limits,
    NoAttachment,
    Participant,
    ResearchSource,
)
from app.ollama_client import OllamaResult
from app.research import ResearchBundle


def _request(*, research: bool = True) -> GenerationRequest:
    return GenerationRequest(
        schema_version="1.0",
        language="de",
        topic="Mira und Tom prüfen, ob Acme CMS PHP 8.4 unterstützt.",
        entry_count=2,
        tone="natural",
        chaos=10,
        seriousness=70,
        participants=[
            Participant(id="p1", name="Mira", personality="präzise"),
            Participant(id="p2", name="Tom", personality="skeptisch"),
        ],
        features=Features(
            narration=False,
            typing_indicator=False,
            images=False,
            links=False,
            research=research,
        ),
        limits=Limits(max_narrations=0, max_words_per_entry=40, max_images=0, max_links=0),
    )


def _conversation(*texts: str) -> ConversationResult:
    return ConversationResult(
        schema_version="1.0",
        language="de",
        entries=[
            ConversationEntry(
                type="message",
                speaker_id=f"p{(idx % 2) + 1}",
                text=text,
                typing_indicator=False,
                attachment=NoAttachment(),
            )
            for idx, text in enumerate(texts)
        ],
    )


def _ollama(content: dict) -> OllamaResult:
    return OllamaResult(
        content=json.dumps(content, ensure_ascii=False),
        prompt_tokens=20,
        generated_tokens=10,
        load_duration_ns=1,
        prompt_eval_duration_ns=1,
        eval_duration_ns=1,
        total_duration_ns=1,
    )


class FakeClient:
    def __init__(self, payloads: list[dict]) -> None:
        self.payloads = list(payloads)
        self.calls = 0

    async def chat(self, **kwargs):
        self.calls += 1
        if not self.payloads:
            raise AssertionError("Unexpected audit call")
        return _ollama(self.payloads.pop(0))


class FakeReverseResearch:
    def __init__(self, snippet: str) -> None:
        self.snippet = snippet
        self.queries: list[str] = []

    async def reverse_search(self, query: str, *, language: str, max_sources: int = 3):
        self.queries.append(query)
        return [
            ResearchSource(
                id="R1",
                title="Acme CMS compatibility documentation",
                url="https://docs.example.test/acme/php",
                snippet=self.snippet,
                engine="test",
            )
        ], 0.01


@pytest.mark.asyncio
async def test_reverse_research_is_generic_and_verifies_software_claim(monkeypatch):
    monkeypatch.setattr(settings, "final_audit_fact_enabled", True)
    monkeypatch.setattr(settings, "final_audit_language_enabled", True)
    monkeypatch.setattr(settings, "final_audit_reverse_research_enabled", True)

    client = FakeClient([
        {
            "claims": [
                {
                    "entry_index": 0,
                    "claim": "Acme CMS unterstützt PHP 8.4.",
                    "status": "needs_reverse_research",
                    "query": "Acme CMS PHP 8.4 support",
                }
            ]
        },
        {
            "decisions": [
                {
                    "entry_index": 0,
                    "verdict": "supported",
                    "replacement": "",
                    "reason": "Die Dokumentation bestätigt PHP 8.4.",
                }
            ]
        },
        {"issues": []},
    ])
    research = FakeReverseResearch("Acme CMS 4.2 supports PHP 8.4 and PHP 8.3.")
    auditor = FinalAuditor(client=client, research_service=research)  # type: ignore[arg-type]

    outcome = await auditor.run(
        request=_request(research=True),
        conversation=_conversation(
            "Acme CMS unterstützt PHP 8.4.",
            "Dann können wir das Upgrade einplanen.",
        ),
        research_bundle=ResearchBundle(provider="searxng", queries=[], sources=[], elapsed_seconds=0.0),
        model="gpt-oss:20b",
        num_ctx=4096,
        keep_alive="30m",
        think=None,
    )

    assert research.queries == ["Acme CMS PHP 8.4 support"]
    assert outcome.stats.factual_claims_checked == 1
    assert outcome.stats.reverse_searches == 1
    assert outcome.stats.reverse_sources == 1
    assert outcome.stats.claims_corrected == 0
    assert outcome.conversation.entries[0].text == "Acme CMS unterstützt PHP 8.4."


@pytest.mark.asyncio
async def test_reverse_research_can_correct_contradicted_claim(monkeypatch):
    monkeypatch.setattr(settings, "final_audit_fact_enabled", True)
    monkeypatch.setattr(settings, "final_audit_language_enabled", True)
    monkeypatch.setattr(settings, "final_audit_reverse_research_enabled", True)

    client = FakeClient([
        {
            "claims": [
                {
                    "entry_index": 0,
                    "claim": "Acme CMS unterstützt nur PHP 8.3.",
                    "status": "needs_reverse_research",
                    "query": "Acme CMS supported PHP versions",
                }
            ]
        },
        {
            "decisions": [
                {
                    "entry_index": 0,
                    "verdict": "contradicted",
                    "replacement": "Laut Dokumentation unterstützt Acme CMS auch PHP 8.4.",
                    "reason": "Die gefundene Dokumentation nennt PHP 8.4 ausdrücklich.",
                }
            ]
        },
        {"issues": []},
    ])
    research = FakeReverseResearch("Supported runtimes: PHP 8.3 and PHP 8.4.")
    auditor = FinalAuditor(client=client, research_service=research)  # type: ignore[arg-type]

    outcome = await auditor.run(
        request=_request(research=True),
        conversation=_conversation(
            "Acme CMS unterstützt nur PHP 8.3.",
            "Dann wäre 8.4 ausgeschlossen.",
        ),
        research_bundle=ResearchBundle(provider="searxng", queries=[], sources=[], elapsed_seconds=0.0),
        model="gpt-oss:20b",
        num_ctx=4096,
        keep_alive="30m",
        think=None,
    )

    assert outcome.stats.claims_corrected == 1
    assert "PHP 8.4" in outcome.conversation.entries[0].text


@pytest.mark.asyncio
async def test_language_audit_uses_no_web_and_cannot_change_factual_numbers(monkeypatch):
    monkeypatch.setattr(settings, "final_audit_fact_enabled", True)
    monkeypatch.setattr(settings, "final_audit_language_enabled", True)

    client = FakeClient([
        {
            "issues": [
                {
                    "entry_index": 0,
                    "kind": "grammar",
                    "replacement": "Die Version 8.4 läuft stabil.",
                    "reason": "Subjekt und Verb stimmen nicht überein.",
                },
                {
                    "entry_index": 1,
                    "kind": "coherence",
                    "replacement": "Dann bleiben wir bei Version 8.3.",
                    "reason": "Straffer formuliert.",
                },
            ]
        }
    ])
    research = FakeReverseResearch("should never be used")
    auditor = FinalAuditor(client=client, research_service=research)  # type: ignore[arg-type]

    outcome = await auditor.run(
        request=_request(research=False),
        conversation=_conversation(
            "Die Version 8.4 laufen stabil.",
            "Dann bleiben wir bei Version 8.4.",
        ),
        research_bundle=ResearchBundle(provider="", queries=[], sources=[], elapsed_seconds=0.0),
        model="gpt-oss:20b",
        num_ctx=4096,
        keep_alive="30m",
        think=None,
    )

    assert research.queries == []
    assert outcome.stats.fact_audit_run is False
    assert outcome.stats.language_rewrites == 1
    assert outcome.stats.language_rewrites_rejected == 1
    assert outcome.conversation.entries[0].text == "Die Version 8.4 läuft stabil."
    assert outcome.conversation.entries[1].text == "Dann bleiben wir bei Version 8.4."


@pytest.mark.asyncio
async def test_fact_audit_rejects_new_numeric_value_not_present_in_evidence(monkeypatch):
    monkeypatch.setattr(settings, "final_audit_fact_enabled", True)
    monkeypatch.setattr(settings, "final_audit_language_enabled", False)
    monkeypatch.setattr(settings, "final_audit_reverse_research_enabled", True)

    client = FakeClient([
        {
            "claims": [
                {
                    "entry_index": 0,
                    "claim": "Acme CMS unterstützt nur PHP 8.3.",
                    "status": "needs_reverse_research",
                    "query": "Acme CMS supported PHP versions",
                }
            ]
        },
        {
            "decisions": [
                {
                    "entry_index": 0,
                    "verdict": "contradicted",
                    "replacement": "Acme CMS unterstützt PHP 9.1.",
                    "reason": "Soll angeblich korrigiert werden.",
                }
            ]
        },
    ])
    research = FakeReverseResearch("Supported runtimes: PHP 8.3 and PHP 8.4.")
    auditor = FinalAuditor(client=client, research_service=research)  # type: ignore[arg-type]

    outcome = await auditor.run(
        request=_request(research=True),
        conversation=_conversation(
            "Acme CMS unterstützt nur PHP 8.3.",
            "Dann prüfen wir die Dokumentation.",
        ),
        research_bundle=ResearchBundle(provider="searxng", queries=[], sources=[], elapsed_seconds=0.0),
        model="gpt-oss:20b",
        num_ctx=4096,
        keep_alive="30m",
        think=None,
    )

    assert outcome.stats.fact_rewrites_rejected == 1
    assert outcome.stats.claims_corrected == 1
    assert outcome.conversation.entries[0].text == "Die geprüften Quellen widersprechen dieser Aussage."


def test_coherence_candidates_detect_near_duplicate_and_stale_recheck():
    from app.final_audit import _coherence_candidate_hints
    from app.models import ResearchFact

    conversation = _conversation(
        "Die maximale Heizbetttemperatur ist mit 100 °C dokumentiert.",
        "Beide Angaben kommen aus dem offiziellen Vergleichs-Guide.",
        "Genau, die Angaben stammen aus dem offiziellen Vergleichs-Guide.",
        "Ich prüfe noch die Bedienungsanleitung, um sicherzugehen.",
    )
    bundle = ResearchBundle(
        provider="searxng",
        queries=[],
        sources=[],
        elapsed_seconds=0.0,
        evidence_sufficient=True,
        verified_facts=[
            ResearchFact(
                aspect="maximale Heizbetttemperatur",
                value="100 °C",
                confidence="primary",
                source_ids=["S1"],
            )
        ],
    )

    hints = _coherence_candidate_hints(conversation, bundle, "de")

    assert any("near-duplicate entries 1 and 2" in hint for hint in hints)
    assert any("entry 3" in hint and "stale uncertainty" in hint for hint in hints)


@pytest.mark.asyncio
async def test_fact_audit_ignores_action_plan_false_positive(monkeypatch):
    monkeypatch.setattr(settings, "final_audit_fact_enabled", True)
    monkeypatch.setattr(settings, "final_audit_language_enabled", False)
    monkeypatch.setattr(settings, "final_audit_reverse_research_enabled", True)
    monkeypatch.setattr(settings, "final_audit_think", False)

    client = FakeClient([
        {
            "claims": [
                {
                    "entry_index": 0,
                    "claim": "Du testest mit PLA, um die Grenzen zu prüfen.",
                    "status": "needs_reverse_research",
                    "query": "PLA test printer limits",
                }
            ]
        }
    ])
    research = FakeReverseResearch("should never be used")
    auditor = FinalAuditor(client=client, research_service=research)  # type: ignore[arg-type]

    outcome = await auditor.run(
        request=_request(research=True),
        conversation=_conversation(
            "Und du testest mit PLA, um die Grenzen zu prüfen.",
            "Gut, dann machen wir weiter.",
        ),
        research_bundle=ResearchBundle(provider="searxng", queries=[], sources=[], elapsed_seconds=0.0),
        model="gpt-oss:20b",
        num_ctx=4096,
        keep_alive="30m",
        think="medium",
    )

    assert research.queries == []
    assert outcome.stats.factual_claims_checked == 0
    assert outcome.stats.reverse_searches == 0
    assert outcome.stats.audit_calls == 1


@pytest.mark.asyncio
async def test_gpt_oss_legacy_false_think_is_coerced_to_low(monkeypatch):
    monkeypatch.setattr(settings, "final_audit_fact_enabled", False)
    monkeypatch.setattr(settings, "final_audit_language_enabled", True)
    monkeypatch.setattr(settings, "final_audit_think", False)

    class CapturingClient(FakeClient):
        def __init__(self):
            super().__init__([{"issues": []}])
            self.think_values = []

        async def chat(self, **kwargs):
            self.think_values.append(kwargs.get("think"))
            return await super().chat(**kwargs)

    client = CapturingClient()
    auditor = FinalAuditor(client=client, research_service=FakeReverseResearch("unused"))  # type: ignore[arg-type]

    outcome = await auditor.run(
        request=_request(research=False),
        conversation=_conversation("Hallo Tom.", "Hallo Mira."),
        research_bundle=ResearchBundle(provider="", queries=[], sources=[], elapsed_seconds=0.0),
        model="gpt-oss:20b",
        num_ctx=4096,
        keep_alive="30m",
        think="medium",
    )

    assert client.think_values == ["low"]
    assert outcome.stats.audit_errors == 0
    assert outcome.stats.audit_calls == 1


@pytest.mark.asyncio
async def test_audit_error_details_include_phase(monkeypatch):
    monkeypatch.setattr(settings, "final_audit_fact_enabled", True)
    monkeypatch.setattr(settings, "final_audit_language_enabled", True)
    monkeypatch.setattr(settings, "final_audit_think", "low")

    class BrokenClient:
        async def chat(self, **kwargs):
            raise RuntimeError("simulated Ollama rejection")

    auditor = FinalAuditor(
        client=BrokenClient(),  # type: ignore[arg-type]
        research_service=FakeReverseResearch("unused"),  # type: ignore[arg-type]
    )

    outcome = await auditor.run(
        request=_request(research=True),
        conversation=_conversation("Acme CMS unterstützt PHP 8.4.", "Das klingt gut."),
        research_bundle=ResearchBundle(provider="searxng", queries=[], sources=[], elapsed_seconds=0.0),
        model="gpt-oss:20b",
        num_ctx=4096,
        keep_alive="30m",
        think="medium",
    )

    assert outcome.stats.audit_errors == 2
    assert outcome.stats.audit_calls == 0
    assert any(detail.startswith("fact_plan: RuntimeError:") for detail in outcome.stats.audit_error_details)
    assert any(detail.startswith("language_plan: RuntimeError:") for detail in outcome.stats.audit_error_details)


def test_plan_filter_does_not_hide_factual_clause_followed_by_action():
    from app.final_audit import _looks_like_nonfactual_plan
    assert _looks_like_nonfactual_plan(
        "Die Firmware zeigt einen Maximalwert an – wir können dort nachsehen.",
        "de",
    ) is False
