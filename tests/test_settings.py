import pytest
from pydantic import ValidationError

from complaint_triage.exceptions import ConfigurationError
from complaint_triage.settings import Settings


def test_defaults_are_sane():
    settings = Settings(_env_file=None)
    assert settings.temperature == 0.0
    assert settings.max_tokens >= 1


def test_temperature_is_zero_by_default():
    """A routing decision that varies run to run is not a routing decision."""
    assert Settings(_env_file=None).temperature == 0.0


@pytest.mark.parametrize("value", [-0.1, 2.1])
def test_temperature_out_of_range_is_rejected(value):
    with pytest.raises(ValidationError):
        Settings(_env_file=None, temperature=value)


def test_zero_max_tokens_is_rejected():
    with pytest.raises(ValidationError):
        Settings(_env_file=None, max_tokens=0)


def test_missing_model_fails_with_an_actionable_message(tmp_path):
    settings = Settings(_env_file=None, model_path=tmp_path / "absent.gguf")
    with pytest.raises(ConfigurationError, match="COMPLAINT_MODEL_PATH"):
        settings.require_model_path()


def test_existing_model_path_is_returned(tmp_path):
    weights = tmp_path / "m.gguf"
    weights.touch()
    settings = Settings(_env_file=None, model_path=weights)
    assert settings.require_model_path() == weights


def test_token_is_not_exposed_by_repr():
    settings = Settings(_env_file=None, hf_token="hf_not_a_real_token")
    assert "hf_not_a_real_token" not in repr(settings)


def test_settings_are_immutable():
    with pytest.raises(ValidationError):
        Settings(_env_file=None).max_tokens = 99
