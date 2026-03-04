"""BuddyGPT - Screen AI Assistant main program."""

from __future__ import annotations

import logging
import hashlib
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from PIL import Image

from src.ai_assistant import AIAssistant
from src.app_detector import AppInfo, detect_app
from src.clipboard_utils import get_clipboard_text
from src.config import load_config, save_user_config
from src.content_filter import build_context_prompt, filter_content
from src.hotkey import HotkeyManager, parse_hotkey
from src.intent_router import classify_response_mode
from src.interaction_mode import AssistantTurnResult, ResponseMode
from src.monitor import MonitorConfig, ScreenMonitor
from src.notifications import DailyChatSource, NotificationManager
from src.ocr import DEFAULT_PREFERRED_APPS, extract_ocr_text, should_use_ocr
from src.overlay import OverlayWindow
from src.pet import PetState
from src.proactive import ProactiveHintController
from src.prompts import PERSONALITIES
from src.screenshot import capture_window, get_active_hwnd
from src.url_browse import (
    DEFAULT_GLOBAL_TIMEOUT,
    DEFAULT_MAX_BYTES,
    DEFAULT_MAX_CHARS_PER_URL,
    DEFAULT_MAX_TOTAL_CHARS,
    DEFAULT_PER_URL_TIMEOUT,
    FetchedPage,
    build_browse_context,
    extract_urls,
    fetch_public_page,
)

WAKE_FIRST_SLOT = "wake_first"
SCHEDULE_POLL_SECONDS = 20
URL_PER_TIMEOUT = DEFAULT_PER_URL_TIMEOUT
URL_GLOBAL_TIMEOUT = DEFAULT_GLOBAL_TIMEOUT
URL_MAX_BYTES = DEFAULT_MAX_BYTES
URL_MAX_CHARS_PER_URL = DEFAULT_MAX_CHARS_PER_URL
URL_MAX_TOTAL_CHARS = DEFAULT_MAX_TOTAL_CHARS


def _safe_set_utf8(stream):
    if stream and hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8")


_safe_set_utf8(sys.stdout)
_safe_set_utf8(sys.stderr)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


def _backend_name(config: dict) -> str:
    return str(config.get("backend", "anthropic")).strip().lower()


def _backend_requires_key(backend_name: str) -> bool:
    return backend_name in {"anthropic", "openai"}


def _has_backend_auth(config: dict, backend_name: str) -> bool:
    if backend_name == "openai":
        return bool(str(config.get("openai_api_key", "")).strip())
    if backend_name == "anthropic":
        return bool(str(config.get("api_key", "")).strip())
    return True


def _onboarding_prompt(config: dict) -> str:
    backend = _backend_name(config)
    if backend == "openai":
        return (
            "Hi! Before we start, please paste your OpenAI API key here.\n\n"
            "You can also configure it manually in:\n"
            "- config.json -> openai_api_key\n"
            "- .env -> OPENAI_API_KEY"
        )
    if backend == "ollama":
        return (
            "Hi! BuddyGPT is set to Ollama backend.\n\n"
            "I will validate local Ollama connectivity when you press Enter.\n"
            "If needed, set ollama_base_url in config.json."
        )
    return (
        "Hi! Before we start, please paste your Anthropic API key here.\n\n"
        "You can also configure it manually in:\n"
        "- config.json -> api_key\n"
        "- .env -> ANTHROPIC_API_KEY"
    )


def _onboarding_hint(config: dict) -> str:
    backend = _backend_name(config)
    if backend == "ollama":
        return "Press Enter to validate - type 'skip' to dismiss"
    return "Paste key + Enter - type 'skip' for manual setup"


def _compute_onboarding_needed(config: dict) -> bool:
    backend = _backend_name(config)
    force = bool(config.get("force_onboarding"))
    if not _backend_requires_key(backend):
        return force
    config_done = bool(config.get("onboarding_done"))
    return force or not (config_done and _has_backend_auth(config, backend))


# `cfg` is kept as a stable module-level dict so tests can monkeypatch it.
cfg = load_config()


def _build_ai_instance(
    api_key: str | None = None,
    *,
    cfg_override: dict | None = None,
    openai_api_key_override: str | None = None,
) -> AIAssistant:
    config = cfg_override or cfg
    backend_name = _backend_name(config)
    personality = str(config.get("personality", "buddy")).lower()
    max_tokens_raw = config.get("max_tokens", 400)
    max_tokens = int(max_tokens_raw)
    buddy_default = int(PERSONALITIES["buddy"]["max_tokens"])

    max_tokens_override = max_tokens
    if personality != "buddy" and max_tokens == buddy_default:
        max_tokens_override = None

    return AIAssistant(
        api_key=api_key if api_key is not None else config["api_key"],
        model=config["model"],
        personality=personality,
        max_tokens=max_tokens_override,
        history_window_turns=int(config.get("history_window_turns", 6)),
        history_summary_every_turns=int(config.get("history_summary_every_turns", 6)),
        history_summary_max_chars=int(config.get("history_summary_max_chars", 1800)),
        backend=backend_name,
        openai_api_key=(
            openai_api_key_override
            if openai_api_key_override is not None
            else config.get("openai_api_key", "")
        ),
        ollama_base_url=config.get("ollama_base_url", "http://127.0.0.1:11434"),
        openai_base_url=config.get("openai_base_url", "https://api.openai.com/v1"),
        backend_timeout_sec=int(config.get("backend_timeout_sec", 45)),
    )


@dataclass
class AppRuntime:
    cfg: dict
    ai: AIAssistant
    target_hwnd: int = 0
    current_app: AppInfo | None = None
    clipboard_context_text: str = ""
    clipboard_context_pending: bool = False
    onboarding_needed: bool = False
    daily_chat_source: DailyChatSource | None = None
    notification_manager: NotificationManager | None = None
    active_overlay: OverlayWindow | None = None
    proactive_controller: ProactiveHintController | None = None
    news_lock: threading.Lock = field(default_factory=threading.Lock)
    activation_lock: threading.Lock = field(default_factory=threading.Lock)
    state_lock: threading.Lock = field(default_factory=threading.Lock)
    activation_in_progress: bool = False
    turn_counter: int = 0
    static_context_hash: str = ""
    static_context_id: str = ""
    static_context_last_full_turn: int = 0
    url_cache: dict[str, tuple[float, FetchedPage]] = field(default_factory=dict)
    ocr_cache: dict[str, tuple[float, str]] = field(default_factory=dict)


def _build_notification_manager(rt: AppRuntime) -> NotificationManager:
    manager = NotificationManager()
    rt.daily_chat_source = DailyChatSource(ai_assistant=rt.ai, config=rt.cfg)
    manager.register(rt.daily_chat_source)
    return manager


def _create_runtime(config: dict | None = None) -> AppRuntime:
    config = config or load_config()
    rt = AppRuntime(
        cfg=config,
        ai=_build_ai_instance(cfg_override=config),
        onboarding_needed=_compute_onboarding_needed(config),
    )
    rt.notification_manager = _build_notification_manager(rt)
    return rt


runtime = _create_runtime(cfg)


def _refresh_runtime_after_ai_change(rt: AppRuntime) -> None:
    rt.notification_manager = _build_notification_manager(rt)


def _looks_like_api_key(text: str) -> bool:
    value = text.strip()
    return len(value) > 20 and value.startswith("sk-")


def _overlay_status_update(text: str) -> None:
    overlay = runtime.active_overlay
    if overlay is not None:
        overlay.update_thinking_status(text)


def _is_cancelled(cancel_token: Any) -> bool:
    return bool(cancel_token is not None and hasattr(cancel_token, "is_set") and cancel_token.is_set())


def _estimate_tokens(text: str) -> int:
    # Fast approximation suitable for relative telemetry.
    return max(1, len(text) // 4) if text else 0


def _evict_stale_cache_entries(rt: AppRuntime, *, now_mono: float) -> None:
    url_ttl = max(1, int(rt.cfg.get("url_cache_ttl_sec", 300)))
    ocr_ttl = max(1, int(rt.cfg.get("ocr_cache_ttl_sec", 300)))
    rt.url_cache = {
        k: v for k, v in rt.url_cache.items() if (now_mono - v[0]) <= url_ttl
    }
    rt.ocr_cache = {
        k: v for k, v in rt.ocr_cache.items() if (now_mono - v[0]) <= ocr_ttl
    }


def _get_cached_url_page(rt: AppRuntime, url: str, *, now_mono: float) -> FetchedPage | None:
    entry = rt.url_cache.get(url)
    if entry is None:
        return None
    ts, page = entry
    ttl = max(1, int(rt.cfg.get("url_cache_ttl_sec", 300)))
    if (now_mono - ts) > ttl:
        return None
    return page


def _set_cached_url_page(rt: AppRuntime, url: str, page: FetchedPage, *, now_mono: float) -> None:
    rt.url_cache[url] = (now_mono, page)


def _image_cache_key(image) -> str:
    if image is None or not hasattr(image, "convert"):
        return ""
    gray = image.convert("L")
    small = gray.resize((128, 128))
    return hashlib.blake2b(small.tobytes(), digest_size=16).hexdigest()


def _get_cached_ocr_text(rt: AppRuntime, key: str, *, now_mono: float) -> str:
    if not key:
        return ""
    entry = rt.ocr_cache.get(key)
    if entry is None:
        return ""
    ts, text = entry
    ttl = max(1, int(rt.cfg.get("ocr_cache_ttl_sec", 300)))
    if (now_mono - ts) > ttl:
        return ""
    return text


def _set_cached_ocr_text(rt: AppRuntime, key: str, text: str, *, now_mono: float) -> None:
    if key and text:
        rt.ocr_cache[key] = (now_mono, text)


def _pack_context_blocks(
    *,
    blocks: list[dict[str, Any]],
    question: str,
    max_chars: int,
) -> tuple[list[tuple[str, str]], list[dict[str, Any]], bool]:
    """Token-budgeted context packing with priority and trimming."""
    max_chars = max(1000, int(max_chars))
    question_len = len(question)
    remaining = max(max_chars - question_len, 0)
    selected: list[tuple[int, int, str, str]] = []
    dropped: list[dict[str, Any]] = []
    used = 0

    for idx, block in enumerate(blocks):
        name = str(block.get("name", f"block_{idx}"))
        text = str(block.get("text", ""))
        if not text:
            continue
        priority = int(block.get("priority", 50))
        required = bool(block.get("required", False))
        allow_trim = bool(block.get("allow_trim", False))
        length = len(text)

        if required and (used + length) > remaining:
            if remaining <= used:
                dropped.append({"name": name, "reason": "no_budget"})
                continue
            if allow_trim:
                trim_to = max(0, remaining - used)
                if trim_to > 0:
                    trimmed = text[:trim_to]
                    selected.append((priority, idx, name, trimmed))
                    used += len(trimmed)
                else:
                    dropped.append({"name": name, "reason": "no_budget"})
            else:
                dropped.append({"name": name, "reason": "no_budget"})
            continue

        selected.append((priority, idx, name, text))
        used += length

    # Keep higher-priority blocks first, then stable order.
    selected.sort(key=lambda item: (-item[0], item[1]))
    packed: list[tuple[str, str]] = []
    final_used = 0
    trimmed = False
    for priority, idx, name, text in selected:
        text_len = len(text)
        if (final_used + text_len) <= remaining:
            packed.append((name, text))
            final_used += text_len
            continue
        # Attempt tail-trim only for lower-priority long blocks.
        if text_len > 0 and priority < 60:
            remain = remaining - final_used
            if remain > 0:
                packed.append((name, text[:remain]))
                final_used += remain
                trimmed = True
            else:
                dropped.append({"name": name, "reason": "budget_pruned"})
            continue
        dropped.append({"name": name, "reason": "budget_pruned"})

    return packed, dropped, trimmed


def _log_context_telemetry(
    *,
    enabled: bool,
    included_blocks: list[tuple[str, str]],
    dropped: list[dict[str, Any]],
    question: str,
) -> None:
    if not enabled:
        return
    token_by_block = {name: _estimate_tokens(text) for name, text in included_blocks}
    token_by_block["question"] = _estimate_tokens(question)
    total = sum(token_by_block.values())
    dropped_desc = ",".join(f"{d['name']}:{d['reason']}" for d in dropped) if dropped else "none"
    block_desc = ",".join(f"{k}:{v}" for k, v in token_by_block.items())
    logger.info(
        "event=CONTEXT_PACK flow=submit result=packed est_tokens_total=%d blocks=%s dropped=%s",
        total,
        block_desc,
        dropped_desc,
    )


def on_screen_change(frame, distance):
    ts = time.strftime("%H:%M:%S")
    logger.info('[%s] Screen changed (distance=%d) window="%s"', ts, distance, frame.title)

    rt = runtime
    if rt.proactive_controller is None or rt.active_overlay is None:
        return
    if not bool(rt.cfg.get("proactive_hints", False)):
        return

    decision = rt.proactive_controller.should_show_hint(
        distance=distance,
        onboarding_needed=rt.onboarding_needed,
        overlay_resting=rt.active_overlay.can_show_proactive(),
    )
    if not decision.allowed:
        logger.info(
            "event=PROACTIVE_HINT_SKIPPED flow=monitor result=skipped reason=%s distance=%d threshold=%d",
            decision.reason,
            distance,
            rt.proactive_controller.threshold,
        )
        return

    title = "Screen changed"
    body = frame.title.strip() if getattr(frame, "title", "") else "Something changed on your screen."
    rt.active_overlay.show_alert(
        title=title,
        body=body,
        hint="Click to ask - Esc dismiss",
        priority="proactive",
    )
    rt.proactive_controller.mark_shown()
    logger.info(
        "event=PROACTIVE_HINT_TRIGGERED flow=monitor result=shown reason=%s distance=%d threshold=%d",
        decision.reason,
        distance,
        rt.proactive_controller.threshold,
    )


def _begin_activation(rt: AppRuntime) -> bool:
    with rt.activation_lock:
        if rt.activation_in_progress:
            return False
        rt.activation_in_progress = True
        return True


def _end_activation(rt: AppRuntime) -> None:
    with rt.activation_lock:
        rt.activation_in_progress = False


def _validate_and_save_onboarding_key(rt: AppRuntime, backend: str, text: str) -> str:
    if backend == "openai":
        ai_temp = _build_ai_instance(
            cfg_override=rt.cfg,
            openai_api_key_override=text,
        )
        valid, err = ai_temp.validate_key()
        if not valid:
            return f"That key did not validate. {err}\nPlease paste a valid key and try again."
        save_user_config(
            {
                "openai_api_key": text,
                "onboarding_done": True,
                "force_onboarding": False,
            }
        )
        rt.cfg["openai_api_key"] = text
        rt.cfg["onboarding_done"] = True
        rt.cfg["force_onboarding"] = False
        rt.onboarding_needed = False
        rt.ai = _build_ai_instance(cfg_override=rt.cfg)
        _refresh_runtime_after_ai_change(rt)
        return "Nice. OpenAI API key saved. Wake me again and ask anything."

    ai_temp = _build_ai_instance(api_key=text, cfg_override=rt.cfg)
    valid, err = ai_temp.validate_key()
    if not valid:
        return f"That key did not validate. {err}\nPlease paste a valid key and try again."
    save_user_config({"api_key": text, "onboarding_done": True, "force_onboarding": False})
    rt.cfg["api_key"] = text
    rt.cfg["onboarding_done"] = True
    rt.cfg["force_onboarding"] = False
    rt.onboarding_needed = False
    rt.ai = _build_ai_instance(api_key=text, cfg_override=rt.cfg)
    _refresh_runtime_after_ai_change(rt)
    return "Nice. API key saved. Wake me again and ask anything."


def _handle_onboarding_submit(rt: AppRuntime, question: str) -> str:
    backend = _backend_name(rt.cfg)
    text = question.strip()
    if text.lower() in {"skip", "later", "not now"}:
        if backend == "openai":
            return (
                "No problem. To finish setup later, add your key to either "
                "`config.json` (`openai_api_key`) or `.env` (`OPENAI_API_KEY`), "
                "then wake me again."
            )
        if backend == "ollama":
            return (
                "No problem. To finish setup later, set `ollama_base_url` in "
                "`config.json`, then wake me again."
            )
        return (
            "No problem. To finish setup later, add your key to either "
            "`config.json` (`api_key`) or `.env` (`ANTHROPIC_API_KEY`), "
            "then wake me again."
        )

    if backend == "ollama":
        ai_temp = _build_ai_instance(cfg_override=rt.cfg)
        valid, err = ai_temp.validate_key()
        if not valid:
            return f"Could not connect to Ollama. {err}\nPlease check ollama_base_url and try again."
        save_user_config({"onboarding_done": True, "force_onboarding": False})
        rt.cfg["onboarding_done"] = True
        rt.cfg["force_onboarding"] = False
        rt.onboarding_needed = False
        rt.ai = _build_ai_instance(cfg_override=rt.cfg)
        _refresh_runtime_after_ai_change(rt)
        return "Nice. Ollama connectivity validated. Wake me again and ask anything."

    if not _looks_like_api_key(text):
        if backend == "openai":
            return (
                "I still need your OpenAI API key. Paste it here and press Enter.\n"
                "Tip: it usually starts with `sk-`.\n"
                "If you want to set it manually, type `skip`."
            )
        return (
            "I still need your Anthropic API key. Paste it here and press Enter.\n"
            "Tip: it usually starts with `sk-`.\n"
            "If you want to set it manually, type `skip`."
        )

    return _validate_and_save_onboarding_key(rt, backend, text)


def on_submit(question, image, *, cancel_token=None):
    """Called by overlay UI when user submits a question."""
    rt = runtime
    if rt.onboarding_needed:
        return _handle_onboarding_submit(rt, question)

    now_mono = time.monotonic()
    _evict_stale_cache_entries(rt, now_mono=now_mono)
    rt.turn_counter += 1

    with rt.state_lock:
        hwnd = rt.target_hwnd
        app = rt.current_app
        clip_text = rt.clipboard_context_text
        clip_pending = rt.clipboard_context_pending
        if rt.clipboard_context_pending and rt.clipboard_context_text:
            rt.clipboard_context_pending = False

    if image is None and hwnd:
        raw = capture_window(hwnd)
        if raw and app:
            image = filter_content(raw, app)

    app_type = app.app_type.value if app else ""
    response_mode = classify_response_mode(question=question, app_type=app_type, ai=rt.ai)
    if _is_cancelled(cancel_token):
        return AssistantTurnResult(text="Request cancelled.", response_mode=response_mode)
    mode_hint = "focus on actionable, task-oriented response"
    if response_mode == ResponseMode.CASUAL:
        mode_hint = "light conversational response"

    urls = extract_urls(question)
    browse_context = ""
    browse_warning = ""
    if urls:
        _overlay_status_update("Fetching links...")
        pages = []
        failures: list[str] = []
        allow_private_urls = bool(rt.cfg.get("allow_private_url_browse", True))
        deadline = time.monotonic() + URL_GLOBAL_TIMEOUT
        for url in urls:
            if _is_cancelled(cancel_token):
                return AssistantTurnResult(text="Request cancelled.", response_mode=response_mode)
            cached_page = _get_cached_url_page(rt, url, now_mono=time.monotonic())
            if cached_page is not None:
                logger.info("URL browse cache hit: url=%s ok=%s", url, cached_page.ok)
                page = cached_page
            else:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    failures.append(f"{url} (global_timeout)")
                    continue
                timeout_sec = min(URL_PER_TIMEOUT, remaining)
                logger.info("URL browse start: url=%s timeout=%.2fs", url, timeout_sec)
                page = fetch_public_page(
                    url=url,
                    timeout_sec=timeout_sec,
                    max_bytes=URL_MAX_BYTES,
                    allow_private=allow_private_urls,
                )
                _set_cached_url_page(rt, url, page, now_mono=time.monotonic())
                if not page.ok:
                    logger.info("URL browse failed: url=%s reason=%s", url, page.error)
                else:
                    logger.info(
                        "URL browse success: url=%s type=%s duration_ms=%d",
                        url,
                        page.content_type,
                        page.duration_ms,
                    )
            if not page.ok:
                failures.append(f"{url} ({page.error})")
                continue
            pages.append(page)

        if failures and not pages:
            msg = (
                "I could not browse those links right now. "
                "Please retry or send accessible public pages only.\n"
                f"Failed links: {'; '.join(failures)}"
            )
            return AssistantTurnResult(text=msg, response_mode=response_mode)
        if failures:
            browse_warning = (
                "[Direct URL browse note]\n"
                f"Some links could not be fetched: {'; '.join(failures)}"
            )
        if pages:
            browse_context = build_browse_context(
                pages=pages,
                max_chars_per_url=URL_MAX_CHARS_PER_URL,
                max_total_chars=URL_MAX_TOTAL_CHARS,
            )

    app_context_block = ""
    if app:
        app_context_block = f"[{build_context_prompt(app)}]"

    clipboard_block = ""
    if clip_pending and clip_text:
        clipboard_block = f"[Clipboard context]\n{clip_text}"

    ocr_text = ""
    ocr_block = ""
    if image is not None and should_use_ocr(
        app_type=app_type,
        enabled=bool(rt.cfg.get("enable_ocr_fallback", False)),
        preferred_apps=rt.cfg.get("ocr_preferred_apps", list(DEFAULT_PREFERRED_APPS)),
    ):
        if _is_cancelled(cancel_token):
            return AssistantTurnResult(text="Request cancelled.", response_mode=response_mode)
        _overlay_status_update("Running OCR...")
        ocr_key = _image_cache_key(image)
        ocr_text = _get_cached_ocr_text(rt, ocr_key, now_mono=time.monotonic())
        if not ocr_text:
            ocr_text = extract_ocr_text(
                image,
                max_chars=int(rt.cfg.get("ocr_max_chars", 3000)),
                timeout_sec=int(rt.cfg.get("ocr_timeout_sec", 5)),
                tesseract_cmd=str(rt.cfg.get("tesseract_cmd", "")),
            )
            _set_cached_ocr_text(rt, ocr_key, ocr_text, now_mono=time.monotonic())
        else:
            logger.info("OCR cache hit: key=%s chars=%d", ocr_key[:8], len(ocr_text))
        if _is_cancelled(cancel_token):
            return AssistantTurnResult(text="Request cancelled.", response_mode=response_mode)
        if ocr_text:
            ocr_block = f"[Screen text (OCR)]\n{ocr_text}"

    browse_context_block = f"[Direct URL browse context]\n{browse_context}" if browse_context else ""
    browse_warning_block = browse_warning or ""

    static_payload = "\n\n".join(
        [
            part
            for part in [
                app_context_block,
                clipboard_block,
                ocr_block,
                browse_context_block,
                browse_warning_block,
            ]
            if part
        ]
    )
    static_hash = hashlib.sha1(static_payload.encode("utf-8")).hexdigest() if static_payload else ""
    refresh_turns = max(1, int(rt.cfg.get("context_reference_refresh_turns", 3)))
    use_context_reference = (
        bool(static_hash)
        and static_hash == rt.static_context_hash
        and bool(rt.static_context_id)
        and (rt.turn_counter - rt.static_context_last_full_turn) < refresh_turns
    )
    if static_hash and not use_context_reference:
        rt.static_context_hash = static_hash
        rt.static_context_id = static_hash[:8]
        rt.static_context_last_full_turn = rt.turn_counter
    if not static_hash:
        rt.static_context_hash = ""
        rt.static_context_id = ""
        rt.static_context_last_full_turn = 0

    context_blocks: list[dict[str, Any]] = []
    if use_context_reference:
        context_blocks.append(
            {
                "name": "context_ref",
                "text": (
                    f"[Context reference id: {rt.static_context_id}]\n"
                    "Reuse static context from previous turn in this session."
                ),
                "priority": 95,
                "required": True,
            }
        )
    else:
        if rt.static_context_id:
            context_blocks.append(
                {
                    "name": "context_id",
                    "text": f"[Context id: {rt.static_context_id}]",
                    "priority": 95,
                    "required": False,
                }
            )
        if app_context_block:
            context_blocks.append(
                {"name": "app", "text": app_context_block, "priority": 90, "required": False}
            )
        if clipboard_block:
            context_blocks.append(
                {
                    "name": "clipboard",
                    "text": clipboard_block,
                    "priority": 85,
                    "required": False,
                    "allow_trim": True,
                }
            )
        if ocr_block:
            context_blocks.append(
                {
                    "name": "ocr",
                    "text": ocr_block,
                    "priority": 82,
                    "required": False,
                    "allow_trim": True,
                }
            )
        if browse_warning_block:
            context_blocks.append(
                {
                    "name": "browse_warning",
                    "text": browse_warning_block,
                    "priority": 65,
                    "required": False,
                }
            )
        if browse_context_block:
            context_blocks.append(
                {
                    "name": "browse_context",
                    "text": browse_context_block,
                    "priority": 20,
                    "required": False,
                    "allow_trim": True,
                }
            )

    context_blocks.append(
        {
            "name": "mode_hint",
            "text": f"[Mode hint] {mode_hint}",
            "priority": 75,
            "required": True,
        }
    )

    packed_blocks, dropped, _trimmed = _pack_context_blocks(
        blocks=context_blocks,
        question=question,
        max_chars=int(rt.cfg.get("context_max_chars", 9000)),
    )
    full_question = "\n\n".join([text for _name, text in packed_blocks] + [question])

    _log_context_telemetry(
        enabled=bool(rt.cfg.get("context_telemetry", True)),
        included_blocks=packed_blocks,
        dropped=dropped,
        question=question,
    )

    if _is_cancelled(cancel_token):
        return AssistantTurnResult(text="Request cancelled.", response_mode=response_mode)
    _overlay_status_update("Thinking...")
    answer = rt.ai.ask(
        full_question,
        image=image,
        search_hint_question=question,
        ocr_text=ocr_text,
    )
    return AssistantTurnResult(text=answer, response_mode=response_mode)


def _deliver_daily_news_slot(slot_id: str, overlay: OverlayWindow, now_local: datetime | None = None) -> bool:
    """Generate and show one slot. Returns True when delivered."""
    rt = runtime
    if rt.daily_chat_source is None or rt.notification_manager is None:
        return False

    now_local = now_local or datetime.now()
    notification = rt.daily_chat_source.generate_for_slot(slot_id, now_local, rt.notification_manager.state)
    if notification is None:
        return False

    rt.ai.set_app_context("")
    with rt.state_lock:
        rt.target_hwnd = 0
        rt.current_app = None
    rt.static_context_hash = ""
    rt.static_context_id = ""
    rt.static_context_last_full_turn = 0
    rt.turn_counter = 0
    overlay.show_notice(
        notification.text,
        hint=notification.hint,
        status=notification.status,
        pet_state=notification.pet_state,
    )
    return True


def _run_scheduled_news_loop(overlay: OverlayWindow) -> None:
    rt = runtime
    while True:
        time.sleep(SCHEDULE_POLL_SECONDS)
        if rt.onboarding_needed:
            continue
        if rt.daily_chat_source is None or not rt.daily_chat_source.is_enabled():
            continue

        try:
            now_local = datetime.now()
            pending_slots = rt.daily_chat_source.pending_timed_slots(rt.notification_manager.state, now_local)  # type: ignore[union-attr]
            if pending_slots and not overlay.can_show_proactive():
                logger.info(
                    "event=SLOT_DEFERRED_BUSY flow=scheduled_proactive slot_id=%s result=deferred reason=overlay_not_resting pet_state=%s",
                    pending_slots[0],
                    overlay.pet_state_name,
                )
                continue

            with rt.news_lock:
                if rt.onboarding_needed:
                    continue
                if rt.daily_chat_source is None or not rt.daily_chat_source.is_enabled():
                    continue
                if not overlay.can_show_proactive():
                    continue
                pending_slots = rt.daily_chat_source.pending_timed_slots(
                    rt.notification_manager.state,  # type: ignore[union-attr]
                    now_local,
                )
                if not pending_slots:
                    continue

                slot_id = pending_slots[0]
                delivered = _deliver_daily_news_slot(slot_id, overlay, now_local)
                if delivered:
                    logger.info(
                        "event=SLOT_DELIVERED flow=scheduled_proactive slot_id=%s result=delivered",
                        slot_id,
                    )
                else:
                    logger.info(
                        "event=SLOT_SKIPPED_NO_UNIQUE_TOPIC flow=scheduled_proactive slot_id=%s result=skipped reason=no_unique_topic",
                        slot_id,
                    )
        except Exception:
            logger.exception("Scheduled daily news loop failed")


def on_activate(overlay):
    """Called when wake-up action is triggered."""
    rt = runtime
    if not _begin_activation(rt):
        logger.info("event=ACTIVATE flow=manual_activation result=ignored reason=activation_in_progress")
        return

    try:
        if not overlay.can_show_proactive():
            logger.info(
                "event=ACTIVATE flow=manual_activation result=ignored reason=overlay_not_resting pet_state=%s",
                overlay.pet_state_name,
            )
            return

        if rt.onboarding_needed:
            with rt.state_lock:
                rt.clipboard_context_text = ""
                rt.clipboard_context_pending = False
            rt.static_context_hash = ""
            rt.static_context_id = ""
            rt.static_context_last_full_turn = 0
            rt.turn_counter = 0
            overlay.show(image=None, window_title="BuddyGPT Onboarding")
            overlay.show_notice(
                _onboarding_prompt(rt.cfg),
                hint=_onboarding_hint(rt.cfg),
                status="Setup: API key needed",
                pet_state=PetState.GREETING,
            )
            return

        with rt.news_lock:
            try:
                if (
                    rt.daily_chat_source
                    and rt.notification_manager
                    and rt.daily_chat_source.should_trigger_slot(
                        rt.notification_manager.state,
                        WAKE_FIRST_SLOT,
                        datetime.now(),
                    )
                ):
                    if _deliver_daily_news_slot(WAKE_FIRST_SLOT, overlay, datetime.now()):
                        with rt.state_lock:
                            rt.clipboard_context_text = ""
                            rt.clipboard_context_pending = False
                        rt.static_context_hash = ""
                        rt.static_context_id = ""
                        rt.static_context_last_full_turn = 0
                        rt.turn_counter = 0
                        logger.info(
                            "event=SLOT_DELIVERED flow=wake_activation slot_id=%s result=delivered",
                            WAKE_FIRST_SLOT,
                        )
                        return
                    logger.info(
                        "event=SLOT_SKIPPED_NO_UNIQUE_TOPIC flow=wake_activation slot_id=%s result=skipped reason=no_unique_topic",
                        WAKE_FIRST_SLOT,
                    )
            except Exception:
                logger.exception("Daily chat wake check failed, falling back to normal activation")

            hwnd = get_active_hwnd(skip_hwnd=overlay.hwnd)
            app = detect_app(hwnd)
            with rt.state_lock:
                rt.target_hwnd = hwnd
                rt.current_app = app
                rt.clipboard_context_text = ""
                rt.clipboard_context_pending = False
            rt.static_context_hash = ""
            rt.static_context_id = ""
            rt.static_context_last_full_turn = 0
            rt.turn_counter = 0
            img = capture_window(hwnd)
            if img and app:
                img = filter_content(img, app)

            rt.ai.clear_history()
            if app:
                rt.ai.set_app_context(app.app_type.value)
                logger.info("Activated: %s (%s) hwnd=%d", app.label, app.process_name, hwnd)
                window_title = f"{app.label} - {app.window_title}"
            else:
                rt.ai.set_app_context("")
                logger.info("Activated: unknown app hwnd=%d", hwnd)
                window_title = "BuddyGPT"
            overlay.show(image=img, window_title=window_title)
    finally:
        _end_activation(rt)


def on_activate_clipboard(overlay):
    """Wake BuddyGPT with clipboard context instead of screenshot context."""
    rt = runtime
    if not _begin_activation(rt):
        logger.info(
            "event=ACTIVATE_CLIPBOARD flow=manual_activation result=ignored reason=activation_in_progress"
        )
        return

    try:
        if not overlay.can_show_proactive():
            logger.info(
                "event=ACTIVATE_CLIPBOARD flow=manual_activation result=ignored reason=overlay_not_resting pet_state=%s",
                overlay.pet_state_name,
            )
            return

        if rt.onboarding_needed:
            rt.static_context_hash = ""
            rt.static_context_id = ""
            rt.static_context_last_full_turn = 0
            rt.turn_counter = 0
            overlay.show(image=None, window_title="BuddyGPT Onboarding")
            overlay.show_notice(
                _onboarding_prompt(rt.cfg),
                hint=_onboarding_hint(rt.cfg),
                status="Setup: API key needed",
                pet_state=PetState.GREETING,
            )
            return

        hwnd = get_active_hwnd(skip_hwnd=overlay.hwnd)
        app = detect_app(hwnd)
        clip_text, clip_error = get_clipboard_text(max_chars=6000)
        if clip_error:
            overlay.show_notice(
                "Clipboard mode could not read text content.\nCopy some text first, then press Ctrl+Shift+V again.",
                hint="Esc dismiss",
                status="Clipboard empty",
                pet_state=PetState.GREETING,
            )
            logger.info(
                "event=CLIPBOARD_CAPTURE flow=clipboard_activation result=skipped reason=%s",
                clip_error,
            )
            return

        with rt.state_lock:
            rt.target_hwnd = 0
            rt.current_app = app
            rt.clipboard_context_text = clip_text
            rt.clipboard_context_pending = True
        rt.static_context_hash = ""
        rt.static_context_id = ""
        rt.static_context_last_full_turn = 0
        rt.turn_counter = 0

        rt.ai.clear_history()
        rt.ai.set_app_context(app.app_type.value if app else "")
        overlay.show(image=None, window_title="BuddyGPT - Clipboard Mode")
        overlay.show_notice(
            "Clipboard captured. Ask your question and I will use it as context.",
            hint="Enter ask - Esc dismiss",
            status="Clipboard mode",
            pet_state=None,
        )
        logger.info(
            "event=CLIPBOARD_CAPTURE flow=clipboard_activation result=captured chars=%d",
            len(clip_text),
        )
    finally:
        _end_activation(rt)


def _start_monitor_if_enabled(config: dict) -> bool:
    rt = runtime
    if not bool(config.get("enable_monitor", False)):
        print("Screen monitor disabled.")
        return False

    rt.proactive_controller = ProactiveHintController(
        base_threshold=int(config.get("hash_threshold", 12)),
        sensitivity=str(config.get("proactive_sensitivity", "medium")),
        cooldown_sec=int(config.get("proactive_cooldown_sec", 90)),
        max_per_hour=int(config.get("proactive_max_per_hour", 8)),
        quiet_hours_enabled=bool(config.get("proactive_quiet_hours_enabled", False)),
        quiet_start=str(config.get("proactive_quiet_start", "22:00")),
        quiet_end=str(config.get("proactive_quiet_end", "08:00")),
    )
    monitor_config = MonitorConfig(
        interval=config["screenshot_interval"],
        hash_threshold=config["hash_threshold"],
    )
    monitor = ScreenMonitor(monitor_config)
    monitor.on_change(on_screen_change)
    monitor_thread = threading.Thread(target=monitor.run, daemon=True)
    monitor_thread.start()
    print("Screen monitor started.")
    return True


def _setup_tray_icon(overlay: OverlayWindow, hk: HotkeyManager):
    try:
        import pystray
    except Exception as exc:
        logger.warning("Tray mode requested but pystray import failed: %s", exc)
        return None

    icon_candidates = [
        Path(__file__).resolve().parent / "assets" / "shiba" / "states" / "state-awake.png",
        Path(__file__).resolve().parent / "assets" / "shiba" / "states" / "state-resting.png",
    ]
    icon_image = None
    for path in icon_candidates:
        if path.exists():
            try:
                icon_image = Image.open(path).convert("RGBA").resize((64, 64), Image.LANCZOS)
                break
            except Exception:
                continue
    if icon_image is None:
        icon_image = Image.new("RGBA", (64, 64), (255, 140, 0, 255))

    def _on_wake(_icon, _item):
        on_activate(overlay)

    def _on_clipboard(_icon, _item):
        on_activate_clipboard(overlay)

    def _on_quit(icon, _item):
        hk.request_quit()
        icon.stop()

    def _on_snooze(_icon, _item):
        ctl = runtime.proactive_controller
        if ctl is None:
            return
        ctl.snooze(minutes=60)
        logger.info(
            "event=PROACTIVE_HINT_SNOOZE flow=tray result=applied reason=user_action minutes=60"
        )

    def _on_resume(_icon, _item):
        ctl = runtime.proactive_controller
        if ctl is None:
            return
        ctl.clear_snooze()
        logger.info("event=PROACTIVE_HINT_SNOOZE flow=tray result=cleared reason=user_action")

    menu = pystray.Menu(
        pystray.MenuItem("Wake (Ctrl+Shift+Space)", _on_wake, default=True),
        pystray.MenuItem("Clipboard Wake (Ctrl+Shift+V)", _on_clipboard),
        pystray.MenuItem("Snooze Hints 1h", _on_snooze),
        pystray.MenuItem("Resume Hints", _on_resume),
        pystray.MenuItem("Quit", _on_quit),
    )
    tray_icon = pystray.Icon("BuddyGPT", icon_image, "BuddyGPT", menu)
    threading.Thread(target=tray_icon.run, daemon=True, name="tray-icon").start()
    return tray_icon


def main():
    rt = runtime
    activate_keys = rt.cfg["hotkey_activate"]
    clipboard_keys = rt.cfg.get("hotkey_clipboard", "ctrl+shift+v")
    quit_keys = rt.cfg["hotkey_quit"]
    tray_mode = bool(rt.cfg.get("tray_mode", False))
    backend = _backend_name(rt.cfg)

    print("=" * 50)
    print("  BuddyGPT - Screen AI Assistant")
    print("=" * 50)
    print(f"  {activate_keys} = Wake buddy")
    print(f"  {clipboard_keys} = Wake with clipboard")
    print(f"  {quit_keys} = Quit")
    print(f"  backend: {backend}")
    print(f"  model: {rt.cfg['model']}")
    print(f"  personality: {rt.cfg.get('personality', 'buddy')}")
    print("=" * 50)

    overlay = OverlayWindow(
        on_submit=on_submit,
        on_activate=lambda: on_activate(overlay),
        tray_mode=tray_mode,
        show_token_cost=bool(rt.cfg.get("show_token_cost", False)),
        usage_provider=lambda: rt.ai.get_last_usage(),
    )
    rt.active_overlay = overlay
    print("UI ready.")
    tray_icon = None

    scheduler_thread = threading.Thread(
        target=_run_scheduled_news_loop,
        args=(overlay,),
        daemon=True,
        name="daily-news-scheduler",
    )
    scheduler_thread.start()
    print("Daily news scheduler started.")

    _start_monitor_if_enabled(rt.cfg)

    hk = HotkeyManager(
        hotkey=parse_hotkey(activate_keys),
        clipboard_hotkey=parse_hotkey(clipboard_keys),
        quit_hotkey=parse_hotkey(quit_keys),
    )
    hk.on_activate(lambda: on_activate(overlay))
    hk.on_clipboard(lambda: on_activate_clipboard(overlay))
    hk.start()
    print("Hotkey listener started.")

    if tray_mode:
        tray_icon = _setup_tray_icon(overlay, hk)
        print("System tray mode enabled.")

    print(f"\nWaiting for wake-up... ({activate_keys})\n")

    hk.wait()
    hk.stop()
    if tray_icon:
        try:
            tray_icon.stop()
        except Exception:
            logger.exception("Failed to stop tray icon cleanly")
    print("BuddyGPT exited.")


if __name__ == "__main__":
    main()
