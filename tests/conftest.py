import pytest

from app.config import settings


@pytest.fixture(autouse=True)
def _disable_final_audit_by_default(monkeypatch):
    """Keep legacy engine tests focused on their original stage.

    Dedicated final-audit tests explicitly enable the new pipeline.
    Production default remains enabled.
    """
    monkeypatch.setattr(settings, "final_audit_enabled", False)
