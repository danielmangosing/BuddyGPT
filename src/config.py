"""Load config from config.json with .env fallback for API key."""

import json
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
logger = logging.getLogger(__name__)

DEFAULT_CONFIG = {
    "api_key": "",
    "openai_api_key": "",
    "onboarding_done": False,
    "force_onboarding": False,
    "backend": "anthropic",
    "model": "claude-sonnet-4-20250514",
    "openai_base_url": "https://api.openai.com/v1",
    "ollama_base_url": "http://127.0.0.1:11434",
    "backend_timeout_sec": 45,
    "personality": "buddy",
    "hotkey_activate": "ctrl+shift+space",
    "hotkey_clipboard": "ctrl+shift+v",
    "hotkey_quit": "ctrl+shift+q",
    "screenshot_interval": 3.0,
    "hash_threshold": 12,
    "max_tokens": 400,
    "history_window_turns": 6,
    "history_summary_every_turns": 6,
    "history_summary_max_chars": 1800,
    "enable_monitor": False,
    "allow_private_url_browse": True,
    "context_max_chars": 9000,
    "context_reference_refresh_turns": 3,
    "url_cache_ttl_sec": 300,
    "ocr_cache_ttl_sec": 300,
    "context_telemetry": True,
    "tray_mode": False,
    "show_token_cost": False,
    "enable_ocr_fallback": False,
    "ocr_max_chars": 3000,
    "ocr_timeout_sec": 5,
    "ocr_preferred_apps": [
        "terminal",
        "vscode",
        "gmail",
        "outlook",
        "word",
        "pdf_reader",
    ],
    "tesseract_cmd": "",
    "proactive_hints": False,
    "proactive_sensitivity": "medium",
    "proactive_cooldown_sec": 90,
    "proactive_max_per_hour": 8,
    "proactive_quiet_hours_enabled": False,
    "proactive_quiet_start": "22:00",
    "proactive_quiet_end": "08:00",
    "daily_chat": {
        "enabled": True,
        "push_times": ["15:00", "20:00"],
        "max_topic_retry": 3,
    },
}


def user_data_dir() -> Path:
    appdata = os.environ.get("APPDATA")
    if appdata:
        return Path(appdata) / "BuddyGPT"
    return Path.home() / ".buddygpt"


def _config_candidates() -> list[Path]:
    user_config = user_data_dir() / "config.json"
    local_config = ROOT / "config.json"
    if getattr(sys, "frozen", False):
        return [user_config, local_config]
    return [local_config, user_config]


def _load_env_files() -> None:
    # For installed app, prefer user-scoped env file; for dev, keep local .env support.
    load_dotenv(user_data_dir() / ".env", override=False)
    load_dotenv(ROOT / ".env", override=False)


def _user_data_dir() -> Path:
    """Backward-compatible alias for older imports."""
    return user_data_dir()


def load_config() -> dict:
    _load_env_files()
    config = dict(DEFAULT_CONFIG)
    for config_path in _config_candidates():
        if not config_path.exists():
            continue
        try:
            with open(config_path, "r", encoding="utf-8-sig") as f:
                user = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Ignoring unreadable config file %s: %s", config_path, exc)
            continue
        config.update({k: v for k, v in user.items() if v != ""})
        break

    # Merge nested daily_chat settings.
    daily_chat_cfg = config.get("daily_chat")
    if not isinstance(daily_chat_cfg, dict):
        daily_chat_cfg = {}
    merged_daily_chat = dict(DEFAULT_CONFIG["daily_chat"])
    merged_daily_chat.update(daily_chat_cfg)
    config["daily_chat"] = merged_daily_chat

    # .env overrides empty keys
    if not config["api_key"]:
        config["api_key"] = os.environ.get("ANTHROPIC_API_KEY", "")
    if not config["openai_api_key"]:
        config["openai_api_key"] = os.environ.get("OPENAI_API_KEY", "")

    return config


def save_user_config(updates: dict) -> None:
    """Persist selected config fields to config.json."""
    config_path = _config_candidates()[0]
    config_path.parent.mkdir(parents=True, exist_ok=True)
    current = {}
    if config_path.exists():
        try:
            with open(config_path, "r", encoding="utf-8-sig") as f:
                current = json.load(f)
        except (json.JSONDecodeError, OSError):
            current = {}

    current.update(updates)
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(current, f, ensure_ascii=False, indent=4)
