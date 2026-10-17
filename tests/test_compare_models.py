from app.compare_models import _installed_names, _technical_meta
from app.models import (
    ConversationResult,
    FeatureAdjustments,
    FeatureUsage,
    GenerationMeta,
    GenerationResponse,
)


def test_installed_names_accepts_name_and_model_fields():
    result = _installed_names(
        {
            "models": [
                {"name": "gpt-oss:20b", "model": "gpt-oss:20b"},
                {"name": "ministral-3:14b"},
            ]
        }
    )
    assert "gpt-oss:20b" in result
    assert "ministral-3:14b" in result


def test_blind_metrics_remove_model_identity():
    response = GenerationResponse(
        conversation=ConversationResult(
            schema_version="1.0",
            language="de",
            entries=[],
        ),
        meta=GenerationMeta(
            model="gpt-oss:20b",
            model_profile="gpt-oss-20b",
            calls=1,
            generation_attempts=1,
            repair_attempts=0,
            feature_usage=FeatureUsage(),
            feature_adjustments=FeatureAdjustments(),
        ),
    )

    meta = _technical_meta(response)
    assert "model" not in meta
    assert "model_profile" not in meta
