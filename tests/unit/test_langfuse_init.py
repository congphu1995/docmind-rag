"""Tests for Langfuse initialization and graceful disable."""

from unittest.mock import patch


def test_langfuse_disabled_when_keys_empty():
    with patch("backend.app.core.langfuse.settings") as mock_settings:
        mock_settings.langfuse_public_key = ""
        mock_settings.langfuse_secret_key = ""
        mock_settings.langfuse_host = "http://localhost:3001"

        from backend.app.core.langfuse import is_langfuse_enabled

        assert is_langfuse_enabled() is False


def test_langfuse_enabled_when_keys_set():
    with patch("backend.app.core.langfuse.settings") as mock_settings:
        mock_settings.langfuse_public_key = "pk-lf-test"
        mock_settings.langfuse_secret_key = "sk-lf-test"
        mock_settings.langfuse_host = "http://localhost:3001"

        from backend.app.core.langfuse import is_langfuse_enabled

        assert is_langfuse_enabled() is True


def test_callback_returns_none_when_disabled():
    with patch("backend.app.core.langfuse.settings") as mock_settings:
        mock_settings.langfuse_public_key = ""
        mock_settings.langfuse_secret_key = ""
        mock_settings.langfuse_host = "http://localhost:3001"

        from backend.app.core.langfuse import get_langfuse_callback

        assert get_langfuse_callback() is None
