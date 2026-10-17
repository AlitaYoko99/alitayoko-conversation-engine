from pathlib import Path


def test_removed_parsed_adjustments_attribute_is_not_referenced():
    engine_source = (
        Path(__file__).resolve().parents[1] / "app" / "engine.py"
    ).read_text(encoding="utf-8")

    assert "parsed.adjustments" not in engine_source
