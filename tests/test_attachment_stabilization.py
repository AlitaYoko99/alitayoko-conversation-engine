from app.content_normalizer import normalize_content_features
from app.models import ConversationResult, GenerationRequest
from app.validator import validate_semantics


def make_request():
    return GenerationRequest.model_validate(
        {
            "language": "de",
            "topic": (
                "Lukas und Markus vergleichen einen gebrauchten 3D-Drucker und "
                "möchten vor Änderungen die Anleitung suchen und den Aufbau prüfen."
            ),
            "entry_count": 10,
            "tone": "natural",
            "chaos": 10,
            "participants": [
                {"id": "p1", "name": "Lukas"},
                {"id": "p2", "name": "Markus"},
            ],
            "features": {
                "narration": True,
                "typing_indicator": True,
                "images": True,
                "links": True,
            },
            "limits": {
                "max_narrations": 1,
                "max_words_per_entry": 35,
                "max_images": 1,
                "max_links": 1,
            },
        }
    )


def test_stray_wire_fragment_is_removed():
    request = make_request()
    result = ConversationResult.model_validate(
        {
            "schema_version": "1.0",
            "language": "de",
            "entries": [
                {
                    "type": "message",
                    "speaker_id": "p1",
                    "text": "Schau, hier ein Bild vom korrekt montierten Halter. a:{",
                    "typing_indicator": False,
                    "attachment": {"type": "none"},
                },
                {
                    "type": "message",
                    "speaker_id": "p2",
                    "text": "Gute Idee.",
                    "typing_indicator": True,
                    "attachment": {
                        "type": "image",
                        "description": "Bild des montierten Halters",
                        "alt_text": "Montierter Halter",
                    },
                },
                {
                    "type": "message",
                    "speaker_id": "p1",
                    "text": "Wir sollten auch die Anleitung suchen.",
                    "typing_indicator": False,
                    "attachment": {
                        "type": "link",
                        "query": "3D-Drucker vergleichen Gerät entdecken dabei",
                        "label": "Weitere Informationen",
                    },
                },
                {
                    "type": "narration",
                    "speaker_id": "narrator",
                    "text": "Die beiden arbeiten weiter.",
                    "typing_indicator": False,
                    "attachment": {"type": "none"},
                },
            ],
        }
    )

    adjustments = normalize_content_features(request, result)
    assert adjustments.wire_artifacts_removed == 1
    assert result.entries[0].text == "Schau, hier ein Bild vom korrekt montierten Halter."


def test_low_quality_link_query_is_refined():
    request = make_request()
    result = ConversationResult.model_validate(
        {
            "schema_version": "1.0",
            "language": "de",
            "entries": [
                {
                    "type": "message",
                    "speaker_id": "p1",
                    "text": "Vielleicht fehlt ein Schraubenkopf. Lass uns das prüfen.",
                    "typing_indicator": False,
                    "attachment": {"type": "none"},
                },
                {
                    "type": "message",
                    "speaker_id": "p2",
                    "text": "Vielleicht brauchen wir die Anleitung.",
                    "typing_indicator": False,
                    "attachment": {
                        "type": "link",
                        "query": "3D-Drucker vergleichen Gerät entdecken dabei",
                        "label": "Weitere Informationen",
                    },
                },
                {
                    "type": "message",
                    "speaker_id": "p1",
                    "text": "Einverstanden.",
                    "typing_indicator": True,
                    "attachment": {"type": "none"},
                },
                {
                    "type": "narration",
                    "speaker_id": "narrator",
                    "text": "Sie prüfen den Aufbau Schritt für Schritt.",
                    "typing_indicator": False,
                    "attachment": {"type": "none"},
                },
            ],
        }
    )

    adjustments = normalize_content_features(request, result)
    link = result.entries[1].attachment

    assert adjustments.attachment_queries_refined == 1
    assert link.type == "link"
    assert "dabei" not in link.query.casefold()
    assert "gerät" not in link.query.casefold()
    assert "anleitung" in link.query.casefold()


def test_result_stays_semantically_valid_after_cleanup():
    request = make_request()
    result = ConversationResult.model_validate(
        {
            "schema_version": "1.0",
            "language": "de",
            "entries": [
                {
                    "type": "message",
                    "speaker_id": "p1",
                    "text": "Ich sehe, der Filamenthalter sitzt nicht richtig.",
                    "typing_indicator": False,
                    "attachment": {"type": "none"},
                },
                {
                    "type": "message",
                    "speaker_id": "p2",
                    "text": "Vielleicht fehlt ein Schraubenkopf. Lass uns das prüfen.",
                    "typing_indicator": False,
                    "attachment": {
                        "type": "link",
                        "query": "3D-Drucker vergleichen Gerät entdecken dabei",
                        "label": "Weitere Informationen",
                    },
                },
                {
                    "type": "message",
                    "speaker_id": "p1",
                    "text": "Das sollten wir erst bestätigen, wenn die passende Quelle wirklich gefunden wurde.",
                    "typing_indicator": False,
                    "attachment": {"type": "none"},
                },
                {
                    "type": "message",
                    "speaker_id": "p2",
                    "text": "Ja, aber die Mutter ist locker. Ich brauche einen Schraubenschlüssel.",
                    "typing_indicator": True,
                    "attachment": {
                        "type": "image",
                        "description": "Illustration des lockeren Halters",
                        "alt_text": "Lockerer Halter",
                    },
                },
                {
                    "type": "message",
                    "speaker_id": "p1",
                    "text": "Schau, hier ein Bild vom korrekt montierten Halter. a:{",
                    "typing_indicator": False,
                    "attachment": {"type": "none"},
                },
                {
                    "type": "narration",
                    "speaker_id": "narrator",
                    "text": "Der Drucker startet das erste Testmodell.",
                    "typing_indicator": False,
                    "attachment": {"type": "none"},
                },
            ],
        }
    )

    adjustments = normalize_content_features(request, result)
    codes = {v.code for v in validate_semantics(request, result)}

    assert adjustments.wire_artifacts_removed == 1
    assert adjustments.attachment_queries_refined == 1
    assert "UNRESOLVED_SOURCE_CLAIM" not in codes
    assert "REDUNDANT_SPEAKER_PREFIX" not in codes
