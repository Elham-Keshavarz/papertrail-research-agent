from pydantic import ValidationError
import pytest

from papertrail.config import Settings
from papertrail.health import doctor_summary


def test_defaults(monkeypatch):
    monkeypatch.delenv("PAPERTRAIL_MAX_PAPERS", raising=False)
    settings = Settings(_env_file=None)
    assert settings.max_papers == 5
    assert settings.max_retries == 2


def test_invalid_limits_rejected():
    with pytest.raises(ValidationError):
        Settings(_env_file=None, max_papers=0)
    with pytest.raises(ValidationError):
        Settings(_env_file=None, max_retries=100)


def test_env_overrides(monkeypatch):
    monkeypatch.setenv("PAPERTRAIL_MAX_PAPERS", "8")
    assert Settings(_env_file=None).max_papers == 8


def test_doctor_never_leaks_api_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "super_private_test_token_123")
    summary = doctor_summary(Settings(_env_file=None))
    assert summary["api_key_configured"] is True
    assert "super_private_test_token_123" not in str(summary)
