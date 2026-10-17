from app.models import GenerationRequest
from app.prompts import build_system_prompt


def request():
    return GenerationRequest.model_validate(
        {
            "language": "de",
            "topic": "Lukas und Markus prüfen offizielle technische Daten eines Ender 3 V2.",
            "entry_count": 2,
            "participants": [
                {"id": "p1", "name": "Lukas"},
                {"id": "p2", "name": "Markus"},
            ],
            "features": {
                "narration": False,
                "typing_indicator": False,
                "images": False,
                "links": False,
                "research": True,
            },
            "limits": {
                "max_narrations": 0,
                "max_words_per_entry": 30,
                "max_images": 0,
                "max_links": 0,
            },
        }
    )


def test_prompt_tells_model_to_use_sufficient_evidence_directly():
    prompt = build_system_prompt(
        request(),
        "[S1] Official specs\nSnippet: Max. Nozzle Temperature 260°C.",
        research_evidence_sufficient=True,
    )
    assert "Retrieval coverage: sufficient" in prompt
    assert "state it naturally" in prompt


def test_prompt_does_not_equate_missing_snippet_with_absence():
    prompt = build_system_prompt(
        request(),
        "[S1] Manual\nSnippet: General information.",
        research_evidence_sufficient=False,
        research_missing_aspects=["maximale Düsentemperatur"],
    )
    assert "Retrieval coverage: incomplete" in prompt
    assert "maximale Düsentemperatur" in prompt
    assert "Absence of a fact from the snippets is not evidence" in prompt


def test_prompt_preserves_secondary_exact_model_evidence_without_calling_it_official():
    prompt = build_system_prompt(
        request(),
        "[S1] Obico Ender 3 V2\nFetched excerpt: Hotend reaches up to 255°C.",
        research_evidence_sufficient=False,
        research_missing_aspects=["maximale Düsentemperatur"],
        research_secondary_supported_aspects=["maximale Düsentemperatur"],
    )
    assert "exact-model secondary evidence only" in prompt
    assert "secondary/non-official evidence" in prompt
    assert "do not replace useful secondary evidence" in prompt
    assert "Do not invent APIs" in prompt
