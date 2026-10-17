from app.feature_policy import build_feature_policy, count_feature_usage
from app.models import ConversationResult, GenerationRequest


def make_request(
    *,
    narration=True,
    typing=True,
    images=True,
    links=False,
):
    return GenerationRequest.model_validate(
        {
            "language": "de",
            "topic": "Test",
            "entry_count": 12,
            "participants": [
                {"id": "p1", "name": "Lukas"},
                {"id": "p2", "name": "Markus"},
            ],
            "features": {
                "narration": narration,
                "typing_indicator": typing,
                "images": images,
                "links": links,
            },
            "limits": {
                "max_narrations": 2 if narration else 0,
                "max_words_per_entry": 35,
                "max_images": 1 if images else 0,
                "max_links": 1 if links else 0,
            },
        }
    )


def test_enabled_features_are_required():
    request = make_request()
    policy = build_feature_policy(request)

    assert policy.min_narrations == 1
    assert policy.min_typing_indicators == 1
    assert policy.max_typing_indicators == 3
    assert policy.min_images == 1
    assert policy.min_links == 0


def test_disabled_features_have_zero_budget():
    request = make_request(
        narration=False,
        typing=False,
        images=False,
        links=False,
    )
    policy = build_feature_policy(request)

    assert policy.min_narrations == 0
    assert policy.max_narrations == 0
    assert policy.min_typing_indicators == 0
    assert policy.max_typing_indicators == 0
    assert policy.min_images == 0
    assert policy.max_images == 0
    assert policy.min_links == 0
    assert policy.max_links == 0


def test_feature_usage_counts_public_result():
    result = ConversationResult.model_validate(
        {
            "schema_version": "1.0",
            "language": "de",
            "entries": [
                {
                    "type": "narration",
                    "speaker_id": "narrator",
                    "text": "Der Roboter fährt vorbei.",
                    "typing_indicator": False,
                    "attachment": {"type": "none"},
                },
                {
                    "type": "message",
                    "speaker_id": "p1",
                    "text": "Da ist die Socke.",
                    "typing_indicator": True,
                    "attachment": {
                        "type": "image",
                        "description": "Ein Saugroboter zieht eine rote Socke über Parkett.",
                        "alt_text": "Saugroboter mit roter Socke"
                    },
                },
            ],
        }
    )

    usage = count_feature_usage(result)
    assert usage.narrations == 1
    assert usage.typing_indicators == 1
    assert usage.images == 1
    assert usage.links == 0
