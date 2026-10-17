from app.content_normalizer import normalize_content_features
from app.models import ConversationResult, GenerationRequest
from app.validator import validate_semantics


def make_request() -> GenerationRequest:
    return GenerationRequest.model_validate(
        {
            "language": "de",
            "topic": "Zwei Personen prüfen technische Daten eines unbekannten Geräts.",
            "entry_count": 5,
            "tone": "natural",
            "chaos": 10,
            "seriousness": 70,
            "participants": [
                {"id": "p1", "name": "Lukas"},
                {"id": "p2", "name": "Markus"},
            ],
            "features": {
                "narration": False,
                "typing_indicator": False,
                "images": False,
                "links": False,
                "research": False,
            },
            "limits": {
                "max_narrations": 0,
                "max_words_per_entry": 35,
                "max_images": 0,
                "max_links": 0,
            },
        }
    )


def test_factual_neutralization_at_indices_zero_and_three_stays_unique():
    request = make_request()
    result = ConversationResult.model_validate(
        {
            "schema_version": "1.0",
            "language": "de",
            "entries": [
                {
                    "type": "message",
                    "speaker_id": "p1",
                    "text": "Die Düse erreicht 255 °C.",
                    "typing_indicator": False,
                    "attachment": {"type": "none"},
                },
                {
                    "type": "message",
                    "speaker_id": "p2",
                    "text": "Wir sollten die technischen Daten prüfen.",
                    "typing_indicator": False,
                    "attachment": {"type": "none"},
                },
                {
                    "type": "message",
                    "speaker_id": "p1",
                    "text": "Einverstanden, wir bleiben bei belegten Angaben.",
                    "typing_indicator": False,
                    "attachment": {"type": "none"},
                },
                {
                    "type": "message",
                    "speaker_id": "p2",
                    "text": "Das Heizbett erreicht 100 °C.",
                    "typing_indicator": False,
                    "attachment": {"type": "none"},
                },
                {
                    "type": "message",
                    "speaker_id": "p1",
                    "text": "Danach entscheiden wir über die Einstellungen.",
                    "typing_indicator": False,
                    "attachment": {"type": "none"},
                },
            ],
        }
    )

    adjustments = normalize_content_features(request, result)
    violations = validate_semantics(request, result)

    assert adjustments.technical_specifics_neutralized == 2
    assert adjustments.normalization_duplicates_avoided == 1
    assert result.entries[0].text != result.entries[3].text
    assert "DUPLICATE_ENTRY_TEXT" not in {v.code for v in violations}


def test_source_claim_neutralization_at_indices_zero_and_three_stays_unique():
    request = make_request()
    result = ConversationResult.model_validate(
        {
            "schema_version": "1.0",
            "language": "de",
            "entries": [
                {
                    "type": "message",
                    "speaker_id": "p1",
                    "text": "Die PDF ist geladen und enthält die Werte.",
                    "typing_indicator": False,
                    "attachment": {"type": "none"},
                },
                {
                    "type": "message",
                    "speaker_id": "p2",
                    "text": "Wir vergleichen danach die Angaben.",
                    "typing_indicator": False,
                    "attachment": {"type": "none"},
                },
                {
                    "type": "message",
                    "speaker_id": "p1",
                    "text": "Gut, wir bleiben vorsichtig.",
                    "typing_indicator": False,
                    "attachment": {"type": "none"},
                },
                {
                    "type": "message",
                    "speaker_id": "p2",
                    "text": "Die Anleitung wurde geöffnet und geprüft.",
                    "typing_indicator": False,
                    "attachment": {"type": "none"},
                },
                {
                    "type": "message",
                    "speaker_id": "p1",
                    "text": "Dann machen wir weiter.",
                    "typing_indicator": False,
                    "attachment": {"type": "none"},
                },
            ],
        }
    )

    adjustments = normalize_content_features(request, result)
    violations = validate_semantics(request, result)

    assert adjustments.source_claims_neutralized == 2
    assert adjustments.normalization_duplicates_avoided == 1
    assert result.entries[0].text != result.entries[3].text
    assert "DUPLICATE_ENTRY_TEXT" not in {v.code for v in violations}


def test_raw_duplicate_entries_are_deduplicated_before_validation():
    request = make_request()
    duplicate = "Welche PHP-Version empfohlen wird, ist mit den vorhandenen Quellen noch nicht belegt."
    result = ConversationResult.model_validate(
        {
            "schema_version": "1.0",
            "language": "de",
            "entries": [
                {"type": "message", "speaker_id": "p1", "text": "PHP 8.4 ist vollständig unterstützt.", "typing_indicator": False, "attachment": {"type": "none"}},
                {"type": "message", "speaker_id": "p2", "text": "Welche Version wird empfohlen?", "typing_indicator": False, "attachment": {"type": "none"}},
                {"type": "message", "speaker_id": "p1", "text": "Dafür fehlt noch eine belastbare Angabe.", "typing_indicator": False, "attachment": {"type": "none"}},
                {"type": "message", "speaker_id": "p2", "text": duplicate, "typing_indicator": False, "attachment": {"type": "none"}},
                {"type": "message", "speaker_id": "p1", "text": duplicate, "typing_indicator": False, "attachment": {"type": "none"}},
            ],
        }
    )

    adjustments = normalize_content_features(request, result)
    violations = validate_semantics(request, result)

    assert adjustments.normalization_duplicates_avoided >= 1
    assert result.entries[3].text != result.entries[4].text
    assert "DUPLICATE_ENTRY_TEXT" not in {v.code for v in violations}
    assert len(result.entries) == request.entry_count


def test_duplicate_deduplication_keeps_first_factual_statement_unchanged():
    request = make_request()
    fact = "PHP 8.4 ist vollständig unterstützt."
    result = ConversationResult.model_validate(
        {
            "schema_version": "1.0",
            "language": "de",
            "entries": [
                {"type": "message", "speaker_id": "p1", "text": fact, "typing_indicator": False, "attachment": {"type": "none"}},
                {"type": "message", "speaker_id": "p2", "text": "Gut, das ist belegt.", "typing_indicator": False, "attachment": {"type": "none"}},
                {"type": "message", "speaker_id": "p1", "text": "Welche Version wird empfohlen?", "typing_indicator": False, "attachment": {"type": "none"}},
                {"type": "message", "speaker_id": "p2", "text": "Das ist noch offen.", "typing_indicator": False, "attachment": {"type": "none"}},
                {"type": "message", "speaker_id": "p1", "text": fact, "typing_indicator": False, "attachment": {"type": "none"}},
            ],
        }
    )

    normalize_content_features(request, result)

    assert result.entries[0].text == fact
    assert result.entries[4].text != fact
