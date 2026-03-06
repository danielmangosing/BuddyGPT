"""BuddyGPT - Screen AI Assistant main program."""

from __future__ import annotations

import logging
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
from src.content_filter import filter_content
from src.context_state import ContextAvailability, ContextControls, RecoveryViewState
from src.hotkey import HotkeyManager, parse_hotkey
from src.log_events import log_event
from src.monitor import MonitorConfig, ScreenMonitor
from src.notifications import DailyChatSource, NotificationManager
from src.overlay import OverlayWindow
from src.pet import PetState
from src.proactive import ProactiveHintController
from src.prompts import PERSONALITIES
from src.recovery import SessionRecoverySnapshot
from src.screenshot import capture_window, get_active_hwnd
from src.settings import AppSettings
from src.turn_pipeline import TurnPipeline
from src.url_browse import FetchedPage

WAKE_FIRST_SLOT = "wake_first"
SCHEDULE_POLL_SECONDS = 20


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
    settings = AppSettings.from_dict(cfg_override or cfg)
    backend_name = settings.backend.strip().lower()
    personality = settings.personality.lower()
    max_tokens = int(settings.max_tokens)
    buddy_default = int(PERSONALITIES["buddy"]["max_tokens"])

    max_tokens_override = max_tokens
    if personality != "buddy" and max_tokens == buddy_default:
        max_tokens_override = None

    return AIAssistant(
        api_key=api_key if api_key is not None else settings.api_key,
        model=settings.model,
        personality=personality,
        max_tokens=max_tokens_override,
        history_window_turns=settings.history_window_turns,
        history_summary_every_turns=settings.history_summary_every_turns,
        history_summary_max_chars=settings.history_summary_max_chars,
        search_cache_ttl_sec=settings.search_cache_ttl_sec,
        backend=backend_name,
        openai_api_key=(
            openai_api_key_override
            if openai_api_key_override is not None
            else settings.openai_api_key
        ),
        ollama_base_url=settings.ollama_base_url,
        openai_base_url=settings.openai_base_url,
        backend_timeout_sec=settings.backend_timeout_sec,
    )


@dataclass
class AppRuntime:
    cfg: dict
    ai: AIAssistant
    turn_pipeline: TurnPipeline | None = None
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
    previous_context_blocks: dict[str, str] = field(default_factory=dict)
    url_cache: dict[str, tuple[float, FetchedPage]] = field(default_factory=dict)
    url_summary_cache: dict[str, tuple[float, str, str]] = field(default_factory=dict)
    ocr_cache: dict[str, tuple[float, str]] = field(default_factory=dict)
    recovery_snapshot: SessionRecoverySnapshot | None = None


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
        turn_pipeline=TurnPipeline(status_callback=lambda text: _overlay_status_update(text)),
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


def _settings(config: dict) -> AppSettings:
    return AppSettings.from_dict(config)


def _reset_session_context(rt: AppRuntime) -> None:
    rt.static_context_hash = ""
    rt.static_context_id = ""
    rt.static_context_last_full_turn = 0
    rt.turn_counter = 0
    rt.previous_context_blocks = {}
    rt.recovery_snapshot = None


def _default_context_availability(*, has_screenshot: bool, has_clipboard: bool) -> ContextAvailability:
    return ContextAvailability(
        screenshot=has_screenshot,
        clipboard=has_clipboard,
        urls=True,
        ocr=has_screenshot,
    )


def _store_recovery_snapshot(view: RecoveryViewState) -> None:
    rt = runtime
    if rt.onboarding_needed:
        return
    if not (view.answer_text or view.draft_text or rt.ai.history):
        return
    with rt.state_lock:
        target_hwnd = rt.target_hwnd
        current_app = rt.current_app
        clip_text = rt.clipboard_context_text
        clip_pending = rt.clipboard_context_pending
    rt.recovery_snapshot = SessionRecoverySnapshot(
        created_at=time.monotonic(),
        view=view,
        assistant_state=rt.ai.snapshot_session_state(),
        window_title=getattr(rt.active_overlay, "_window_title", "BuddyGPT"),
        target_hwnd=target_hwnd,
        current_app=current_app,
        clipboard_context_text=clip_text,
        clipboard_context_pending=clip_pending,
        static_context_hash=rt.static_context_hash,
        static_context_id=rt.static_context_id,
        static_context_last_full_turn=rt.static_context_last_full_turn,
        turn_counter=rt.turn_counter,
        previous_context_blocks=dict(rt.previous_context_blocks),
        image=getattr(rt.active_overlay, "_image", None),
    )
    log_event(
        logger,
        "SESSION_SNAPSHOT",
        "dismiss",
        "stored",
        has_answer=int(bool(view.answer_text)),
        has_draft=int(bool(view.draft_text)),
        history_turns=len(rt.ai.history),
    )


def _restore_recent_session(rt: AppRuntime, overlay: OverlayWindow) -> bool:
    snapshot = rt.recovery_snapshot
    if snapshot is None:
        return False
    ttl_sec = max(10, _settings(rt.cfg).session_recovery_ttl_sec)
    if (time.monotonic() - snapshot.created_at) > ttl_sec:
        rt.recovery_snapshot = None
        return False

    rt.ai.restore_session_state(snapshot.assistant_state)
    with rt.state_lock:
        rt.target_hwnd = snapshot.target_hwnd
        rt.current_app = snapshot.current_app
        rt.clipboard_context_text = snapshot.clipboard_context_text
        rt.clipboard_context_pending = snapshot.clipboard_context_pending
    rt.static_context_hash = snapshot.static_context_hash
    rt.static_context_id = snapshot.static_context_id
    rt.static_context_last_full_turn = snapshot.static_context_last_full_turn
    rt.turn_counter = snapshot.turn_counter
    rt.previous_context_blocks = dict(snapshot.previous_context_blocks)
    overlay.restore_session(
        answer_text=snapshot.view.answer_text,
        draft_text=snapshot.view.draft_text,
        response_mode=snapshot.view.response_mode,
        controls=snapshot.view.controls,
        availability=snapshot.view.availability,
        image=snapshot.image,
        window_title=f"{snapshot.window_title} (Recovered)",
    )
    rt.recovery_snapshot = None
    log_event(logger, "SESSION_RECOVERY", "manual_activation", "restored")
    return True


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


def on_submit(question, image, *, cancel_token=None, controls: ContextControls | None = None):
    """Called by overlay UI when user submits a question."""
    rt = runtime
    if rt.onboarding_needed:
        return _handle_onboarding_submit(rt, question)
    if rt.turn_pipeline is None:
        rt.turn_pipeline = TurnPipeline(status_callback=lambda text: _overlay_status_update(text))
    return rt.turn_pipeline.submit(
        rt,
        question,
        image,
        cancel_token=cancel_token,
        controls=controls,
    )


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
    _reset_session_context(rt)
    overlay.show_notice(
        notification.text,
        hint=notification.hint,
        status=notification.status,
        pet_state=notification.pet_state,
    )
    overlay.set_context_availability(_default_context_availability(has_screenshot=False, has_clipboard=False))
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
            _reset_session_context(rt)
            overlay.show(image=None, window_title="BuddyGPT Onboarding")
            overlay.show_notice(
                _onboarding_prompt(rt.cfg),
                hint=_onboarding_hint(rt.cfg),
                status="Setup: API key needed",
                pet_state=PetState.GREETING,
            )
            overlay.set_context_availability(
                _default_context_availability(has_screenshot=False, has_clipboard=False)
            )
            return

        if _restore_recent_session(rt, overlay):
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
                        _reset_session_context(rt)
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
            _reset_session_context(rt)
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
            overlay.set_context_availability(
                _default_context_availability(has_screenshot=img is not None, has_clipboard=False)
            )
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
            _reset_session_context(rt)
            overlay.show(image=None, window_title="BuddyGPT Onboarding")
            overlay.show_notice(
                _onboarding_prompt(rt.cfg),
                hint=_onboarding_hint(rt.cfg),
                status="Setup: API key needed",
                pet_state=PetState.GREETING,
            )
            overlay.set_context_availability(
                _default_context_availability(has_screenshot=False, has_clipboard=False)
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
        _reset_session_context(rt)

        rt.ai.clear_history()
        rt.ai.set_app_context(app.app_type.value if app else "")
        overlay.show(image=None, window_title="BuddyGPT - Clipboard Mode")
        overlay.show_notice(
            "Clipboard captured. Ask your question and I will use it as context.",
            hint="Enter ask - Esc dismiss",
            status="Clipboard mode",
            pet_state=None,
        )
        overlay.set_context_availability(
            _default_context_availability(has_screenshot=False, has_clipboard=True)
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
    settings = _settings(rt.cfg)
    activate_keys = settings.hotkey_activate
    clipboard_keys = settings.hotkey_clipboard
    quit_keys = settings.hotkey_quit
    tray_mode = settings.tray_mode
    backend = settings.backend.strip().lower()

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
        on_dismiss=_store_recovery_snapshot,
        tray_mode=tray_mode,
        show_token_cost=settings.show_token_cost,
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
