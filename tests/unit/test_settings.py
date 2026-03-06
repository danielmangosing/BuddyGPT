"""Unit tests for typed settings coercion."""

from __future__ import annotations

from src.settings import AppSettings


def test_app_settings_from_dict_falls_back_on_invalid_values():
    settings = AppSettings.from_dict(
        {
            "enable_monitor": "false",
            "show_token_cost": "true",
            "max_tokens": "not-a-number",
            "screenshot_interval": "oops",
            "context_max_chars": None,
            "ocr_preferred_apps": 123,
            "daily_chat": {
                "enabled": "false",
                "push_times": 456,
                "max_topic_retry": "bad",
            },
        }
    )

    assert settings.enable_monitor is False
    assert settings.show_token_cost is True
    assert settings.max_tokens == 400
    assert settings.screenshot_interval == 3.0
    assert settings.context_max_chars == 9000
    assert settings.ocr_preferred_apps == ["terminal", "vscode", "gmail", "outlook", "word", "pdf_reader"]
    assert settings.daily_chat.enabled is False
    assert settings.daily_chat.push_times == ["15:00", "20:00"]
    assert settings.daily_chat.max_topic_retry == 3


def test_app_settings_from_dict_accepts_comma_separated_lists():
    settings = AppSettings.from_dict(
        {
            "ocr_preferred_apps": "terminal, vscode , word",
            "daily_chat": {
                "push_times": "09:00, 17:00",
            },
        }
    )

    assert settings.ocr_preferred_apps == ["terminal", "vscode", "word"]
    assert settings.daily_chat.push_times == ["09:00", "17:00"]
