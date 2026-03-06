"""Typed runtime settings derived from config.json and .env."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


def _coerce_str(value, default: str) -> str:
    if value is None:
        return default
    return str(value)


def _coerce_bool(value, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    if isinstance(value, (int, float)):
        return bool(value)
    return default


def _coerce_int(value, default: int) -> int:
    try:
        if value is None or value == "":
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _coerce_float(value, default: float) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _coerce_str_list(value, default: list[str]) -> list[str]:
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",")]
        cleaned = [part for part in parts if part]
        return cleaned or list(default)
    if isinstance(value, (list, tuple, set)):
        cleaned = [str(item).strip() for item in value if str(item).strip()]
        return cleaned or list(default)
    return list(default)


@dataclass(slots=True)
class DailyChatSettings:
    enabled: bool = True
    push_times: list[str] = field(default_factory=lambda: ["15:00", "20:00"])
    max_topic_retry: int = 3


@dataclass(slots=True)
class AppSettings:
    api_key: str = ""
    openai_api_key: str = ""
    onboarding_done: bool = False
    force_onboarding: bool = False
    backend: str = "anthropic"
    model: str = "claude-sonnet-4-20250514"
    openai_base_url: str = "https://api.openai.com/v1"
    ollama_base_url: str = "http://127.0.0.1:11434"
    backend_timeout_sec: int = 45
    personality: str = "buddy"
    hotkey_activate: str = "ctrl+shift+space"
    hotkey_clipboard: str = "ctrl+shift+v"
    hotkey_quit: str = "ctrl+shift+q"
    screenshot_interval: float = 3.0
    hash_threshold: int = 12
    max_tokens: int = 400
    history_window_turns: int = 6
    history_summary_every_turns: int = 6
    history_summary_max_chars: int = 1800
    enable_monitor: bool = False
    allow_private_url_browse: bool = True
    context_max_chars: int = 9000
    context_reference_refresh_turns: int = 3
    url_cache_ttl_sec: int = 300
    ocr_cache_ttl_sec: int = 300
    search_cache_ttl_sec: int = 90
    context_telemetry: bool = True
    tray_mode: bool = False
    show_token_cost: bool = False
    enable_ocr_fallback: bool = False
    ocr_max_chars: int = 3000
    ocr_timeout_sec: int = 5
    ocr_preferred_apps: list[str] = field(
        default_factory=lambda: ["terminal", "vscode", "gmail", "outlook", "word", "pdf_reader"]
    )
    tesseract_cmd: str = ""
    proactive_hints: bool = False
    proactive_sensitivity: str = "medium"
    proactive_cooldown_sec: int = 90
    proactive_max_per_hour: int = 8
    proactive_quiet_hours_enabled: bool = False
    proactive_quiet_start: str = "22:00"
    proactive_quiet_end: str = "08:00"
    adaptive_output_caps: bool = True
    session_recovery_ttl_sec: int = 120
    daily_chat: DailyChatSettings = field(default_factory=DailyChatSettings)

    @classmethod
    def from_dict(cls, raw: dict | None) -> "AppSettings":
        raw = raw or {}
        daily_chat_raw = raw.get("daily_chat", {})
        if not isinstance(daily_chat_raw, dict):
            daily_chat_raw = {}
        daily_chat = DailyChatSettings(
            enabled=_coerce_bool(daily_chat_raw.get("enabled", True), True),
            push_times=_coerce_str_list(daily_chat_raw.get("push_times", ["15:00", "20:00"]), ["15:00", "20:00"]),
            max_topic_retry=_coerce_int(daily_chat_raw.get("max_topic_retry", 3), 3),
        )
        return cls(
            api_key=_coerce_str(raw.get("api_key", ""), ""),
            openai_api_key=_coerce_str(raw.get("openai_api_key", ""), ""),
            onboarding_done=_coerce_bool(raw.get("onboarding_done", False), False),
            force_onboarding=_coerce_bool(raw.get("force_onboarding", False), False),
            backend=_coerce_str(raw.get("backend", "anthropic"), "anthropic"),
            model=_coerce_str(raw.get("model", "claude-sonnet-4-20250514"), "claude-sonnet-4-20250514"),
            openai_base_url=_coerce_str(raw.get("openai_base_url", "https://api.openai.com/v1"), "https://api.openai.com/v1"),
            ollama_base_url=_coerce_str(raw.get("ollama_base_url", "http://127.0.0.1:11434"), "http://127.0.0.1:11434"),
            backend_timeout_sec=_coerce_int(raw.get("backend_timeout_sec", 45), 45),
            personality=_coerce_str(raw.get("personality", "buddy"), "buddy"),
            hotkey_activate=_coerce_str(raw.get("hotkey_activate", "ctrl+shift+space"), "ctrl+shift+space"),
            hotkey_clipboard=_coerce_str(raw.get("hotkey_clipboard", "ctrl+shift+v"), "ctrl+shift+v"),
            hotkey_quit=_coerce_str(raw.get("hotkey_quit", "ctrl+shift+q"), "ctrl+shift+q"),
            screenshot_interval=_coerce_float(raw.get("screenshot_interval", 3.0), 3.0),
            hash_threshold=_coerce_int(raw.get("hash_threshold", 12), 12),
            max_tokens=_coerce_int(raw.get("max_tokens", 400), 400),
            history_window_turns=_coerce_int(raw.get("history_window_turns", 6), 6),
            history_summary_every_turns=_coerce_int(raw.get("history_summary_every_turns", 6), 6),
            history_summary_max_chars=_coerce_int(raw.get("history_summary_max_chars", 1800), 1800),
            enable_monitor=_coerce_bool(raw.get("enable_monitor", False), False),
            allow_private_url_browse=_coerce_bool(raw.get("allow_private_url_browse", True), True),
            context_max_chars=_coerce_int(raw.get("context_max_chars", 9000), 9000),
            context_reference_refresh_turns=_coerce_int(raw.get("context_reference_refresh_turns", 3), 3),
            url_cache_ttl_sec=_coerce_int(raw.get("url_cache_ttl_sec", 300), 300),
            ocr_cache_ttl_sec=_coerce_int(raw.get("ocr_cache_ttl_sec", 300), 300),
            search_cache_ttl_sec=_coerce_int(raw.get("search_cache_ttl_sec", 90), 90),
            context_telemetry=_coerce_bool(raw.get("context_telemetry", True), True),
            tray_mode=_coerce_bool(raw.get("tray_mode", False), False),
            show_token_cost=_coerce_bool(raw.get("show_token_cost", False), False),
            enable_ocr_fallback=_coerce_bool(raw.get("enable_ocr_fallback", False), False),
            ocr_max_chars=_coerce_int(raw.get("ocr_max_chars", 3000), 3000),
            ocr_timeout_sec=_coerce_int(raw.get("ocr_timeout_sec", 5), 5),
            ocr_preferred_apps=_coerce_str_list(
                raw.get("ocr_preferred_apps", ["terminal", "vscode", "gmail", "outlook", "word", "pdf_reader"]),
                ["terminal", "vscode", "gmail", "outlook", "word", "pdf_reader"],
            ),
            tesseract_cmd=_coerce_str(raw.get("tesseract_cmd", ""), ""),
            proactive_hints=_coerce_bool(raw.get("proactive_hints", False), False),
            proactive_sensitivity=_coerce_str(raw.get("proactive_sensitivity", "medium"), "medium"),
            proactive_cooldown_sec=_coerce_int(raw.get("proactive_cooldown_sec", 90), 90),
            proactive_max_per_hour=_coerce_int(raw.get("proactive_max_per_hour", 8), 8),
            proactive_quiet_hours_enabled=_coerce_bool(raw.get("proactive_quiet_hours_enabled", False), False),
            proactive_quiet_start=_coerce_str(raw.get("proactive_quiet_start", "22:00"), "22:00"),
            proactive_quiet_end=_coerce_str(raw.get("proactive_quiet_end", "08:00"), "08:00"),
            adaptive_output_caps=_coerce_bool(raw.get("adaptive_output_caps", True), True),
            session_recovery_ttl_sec=_coerce_int(raw.get("session_recovery_ttl_sec", 120), 120),
            daily_chat=daily_chat,
        )

    def to_dict(self) -> dict:
        data = asdict(self)
        data["daily_chat"] = asdict(self.daily_chat)
        return data
