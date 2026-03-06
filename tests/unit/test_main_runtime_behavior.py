"""Unit tests for main orchestration behavior changes."""

from __future__ import annotations

import logging
import time
from types import SimpleNamespace

import main as main_mod
import src.turn_pipeline as turn_pipeline_mod
from PIL import Image
from src.context_state import ContextAvailability, ContextControls, RecoveryViewState
from src.interaction_mode import ResponseMode
from src.recovery import SessionRecoverySnapshot
from src.url_browse import FetchedPage


class _FakeAI:
    def __init__(self, answer: str = "ok"):
        self.answer = answer
        self.calls: list[dict] = []
        self.max_tokens = 400

    def ask(self, question: str, image=None, **kwargs):
        self.calls.append({"question": question, "image": image, **kwargs})
        return self.answer


def test_on_submit_continues_when_some_urls_fail(monkeypatch):
    rt = main_mod.runtime
    fake_ai = _FakeAI(answer="assistant reply")
    monkeypatch.setattr(rt, "ai", fake_ai)
    monkeypatch.setattr(rt, "onboarding_needed", False)
    monkeypatch.setattr(rt, "target_hwnd", 0)
    monkeypatch.setattr(rt, "current_app", None)
    monkeypatch.setattr(turn_pipeline_mod, "classify_response_mode", lambda **_kwargs: ResponseMode.WORK)
    monkeypatch.setattr(turn_pipeline_mod, "extract_urls", lambda _text: ["https://ok", "https://bad"])
    monkeypatch.setitem(main_mod.cfg, "allow_private_url_browse", True)

    def _fake_fetch(url, timeout_sec, max_bytes, allow_private=True):
        if "ok" in url:
            return FetchedPage(
                url=url,
                ok=True,
                title="ok",
                text="ok body",
                content_type="text/html",
            )
        return FetchedPage(url=url, ok=False, error="timeout")

    monkeypatch.setattr(turn_pipeline_mod, "fetch_public_page", _fake_fetch)
    monkeypatch.setattr(
        turn_pipeline_mod.TurnPipeline,
        "_build_cached_browse_context",
        lambda self, runtime, settings, pages: "BROWSE_CTX",
    )

    result = main_mod.on_submit("Please check these links", image=None)

    assert result.text == "assistant reply"
    assert len(fake_ai.calls) == 1
    sent = fake_ai.calls[0]["question"]
    assert "[Direct URL browse context]" in sent
    assert "BROWSE_CTX" in sent
    assert "[Direct URL browse note]" in sent
    assert "https://bad (timeout)" in sent
    assert fake_ai.calls[0]["search_hint_question"] == "Please check these links"


def test_on_submit_returns_error_when_all_urls_fail(monkeypatch):
    rt = main_mod.runtime
    fake_ai = _FakeAI(answer="should not be used")
    monkeypatch.setattr(rt, "ai", fake_ai)
    monkeypatch.setattr(rt, "onboarding_needed", False)
    monkeypatch.setattr(rt, "target_hwnd", 0)
    monkeypatch.setattr(rt, "current_app", None)
    monkeypatch.setattr(turn_pipeline_mod, "classify_response_mode", lambda **_kwargs: ResponseMode.WORK)
    monkeypatch.setattr(turn_pipeline_mod, "extract_urls", lambda _text: ["https://bad"])
    monkeypatch.setitem(main_mod.cfg, "allow_private_url_browse", True)
    monkeypatch.setattr(
        turn_pipeline_mod,
        "fetch_public_page",
        lambda **_kwargs: FetchedPage(url="https://bad", ok=False, error="timeout"),
    )

    result = main_mod.on_submit("Only bad link", image=None)
    assert "could not browse those links" in result.text.lower()
    assert fake_ai.calls == []


def test_on_activate_ignored_when_overlay_not_resting(monkeypatch, caplog):
    rt = main_mod.runtime
    monkeypatch.setattr(rt, "onboarding_needed", False)
    monkeypatch.setattr(rt, "activation_in_progress", False)

    class _BusyOverlay:
        pet_state_name = "thinking"
        hwnd = 0

        def can_show_proactive(self):
            return False

    with caplog.at_level(logging.INFO):
        main_mod.on_activate(_BusyOverlay())

    assert "event=ACTIVATE" in caplog.text
    assert "reason=overlay_not_resting" in caplog.text


def test_onboarding_invalid_key_does_not_persist(monkeypatch):
    rt = main_mod.runtime
    monkeypatch.setattr(rt, "onboarding_needed", True)
    monkeypatch.setattr(main_mod, "_looks_like_api_key", lambda _text: True)
    monkeypatch.setattr(main_mod, "_refresh_runtime_after_ai_change", lambda _rt: None)
    saved = {"called": False}
    monkeypatch.setattr(
        main_mod,
        "save_user_config",
        lambda _updates: saved.__setitem__("called", True),
    )

    class _InvalidKeyAI:
        def __init__(self, *args, **kwargs):
            pass

        def validate_key(self):
            return False, "invalid"

    monkeypatch.setattr(main_mod, "AIAssistant", _InvalidKeyAI)
    msg = main_mod.on_submit("sk-test-invalid-key-1234567890", image=None)

    assert "did not validate" in msg
    assert saved["called"] is False
    assert rt.onboarding_needed is True


def test_onboarding_valid_key_persists_and_completes(monkeypatch):
    rt = main_mod.runtime
    monkeypatch.setattr(rt, "onboarding_needed", True)
    monkeypatch.setattr(main_mod, "_looks_like_api_key", lambda _text: True)
    monkeypatch.setattr(main_mod, "_refresh_runtime_after_ai_change", lambda _rt: None)

    saved = {}

    def _save(updates):
        saved.update(updates)

    monkeypatch.setattr(main_mod, "save_user_config", _save)

    class _ValidKeyAI:
        def __init__(self, *args, **kwargs):
            pass

        def validate_key(self):
            return True, ""

    monkeypatch.setattr(main_mod, "AIAssistant", _ValidKeyAI)
    msg = main_mod.on_submit("sk-test-valid-key-1234567890", image=None)

    assert "API key saved" in msg
    assert saved["onboarding_done"] is True
    assert rt.onboarding_needed is False


def test_start_monitor_if_enabled_false_skips_start(monkeypatch):
    monkeypatch.setattr(
        main_mod,
        "ScreenMonitor",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("should not construct")),
    )
    started = main_mod._start_monitor_if_enabled({"enable_monitor": False})
    assert started is False


def test_start_monitor_if_enabled_true_starts_thread(monkeypatch):
    calls = {"on_change": False, "thread_started": False}

    class _DummyMonitor:
        def __init__(self, _config):
            pass

        def on_change(self, _callback):
            calls["on_change"] = True

        def run(self):
            return None

    class _DummyThread:
        def __init__(self, target=None, daemon=None):
            self.target = target
            self.daemon = daemon

        def start(self):
            calls["thread_started"] = True

    monkeypatch.setattr(main_mod, "ScreenMonitor", _DummyMonitor)
    monkeypatch.setattr(main_mod.threading, "Thread", _DummyThread)

    started = main_mod._start_monitor_if_enabled(
        {"enable_monitor": True, "screenshot_interval": 3.0, "hash_threshold": 12}
    )

    assert started is True
    assert calls["on_change"] is True
    assert calls["thread_started"] is True


def test_build_ai_instance_non_buddy_with_default_max_uses_personality_default(monkeypatch):
    captured = {}

    class _CaptureAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(main_mod, "AIAssistant", _CaptureAI)
    monkeypatch.setitem(main_mod.cfg, "api_key", "sk-test")
    monkeypatch.setitem(main_mod.cfg, "model", "claude-sonnet-4-20250514")
    monkeypatch.setitem(main_mod.cfg, "personality", "detailed")
    monkeypatch.setitem(main_mod.cfg, "max_tokens", 400)
    monkeypatch.setitem(main_mod.cfg, "history_window_turns", 6)

    main_mod._build_ai_instance()
    assert captured["max_tokens"] is None
    assert captured["personality"] == "detailed"


def test_build_ai_instance_non_buddy_with_explicit_max_keeps_override(monkeypatch):
    captured = {}

    class _CaptureAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(main_mod, "AIAssistant", _CaptureAI)
    monkeypatch.setitem(main_mod.cfg, "api_key", "sk-test")
    monkeypatch.setitem(main_mod.cfg, "model", "claude-sonnet-4-20250514")
    monkeypatch.setitem(main_mod.cfg, "personality", "detailed")
    monkeypatch.setitem(main_mod.cfg, "max_tokens", 900)
    monkeypatch.setitem(main_mod.cfg, "history_window_turns", 6)

    main_mod._build_ai_instance()
    assert captured["max_tokens"] == 900
    assert captured["personality"] == "detailed"


def test_on_screen_change_proactive_hint_shown_when_allowed(monkeypatch):
    class _Overlay:
        def __init__(self):
            self.shown = False

        def can_show_proactive(self):
            return True

        def show_alert(self, title, body, hint, priority="normal"):
            self.shown = True

    class _Frame:
        title = "Terminal - Error"

    overlay = _Overlay()
    rt = main_mod.runtime
    monkeypatch.setattr(rt, "active_overlay", overlay)
    monkeypatch.setattr(
        rt,
        "proactive_controller",
        main_mod.ProactiveHintController(base_threshold=10, sensitivity="high"),
    )
    monkeypatch.setattr(rt, "onboarding_needed", False)
    monkeypatch.setitem(main_mod.cfg, "proactive_hints", True)

    main_mod.on_screen_change(_Frame(), 25)
    assert overlay.shown is True


def test_on_screen_change_proactive_hint_skipped_when_not_resting(monkeypatch):
    class _Overlay:
        def __init__(self):
            self.shown = False

        def can_show_proactive(self):
            return False

        def show_alert(self, title, body, hint, priority="normal"):
            self.shown = True

    class _Frame:
        title = "VS Code"

    overlay = _Overlay()
    rt = main_mod.runtime
    monkeypatch.setattr(rt, "active_overlay", overlay)
    monkeypatch.setattr(
        rt,
        "proactive_controller",
        main_mod.ProactiveHintController(base_threshold=10, sensitivity="high"),
    )
    monkeypatch.setattr(rt, "onboarding_needed", False)
    monkeypatch.setitem(main_mod.cfg, "proactive_hints", True)

    main_mod.on_screen_change(_Frame(), 25)
    assert overlay.shown is False


def test_on_submit_updates_thinking_status_for_url_and_thinking(monkeypatch):
    rt = main_mod.runtime
    fake_ai = _FakeAI(answer="ok")
    statuses = []

    class _Overlay:
        def update_thinking_status(self, text):
            statuses.append(text)

    monkeypatch.setattr(rt, "ai", fake_ai)
    monkeypatch.setattr(rt, "active_overlay", _Overlay())
    monkeypatch.setattr(rt, "onboarding_needed", False)
    monkeypatch.setattr(rt, "target_hwnd", 0)
    monkeypatch.setattr(rt, "current_app", None)
    monkeypatch.setattr(turn_pipeline_mod, "classify_response_mode", lambda **_kwargs: ResponseMode.WORK)
    monkeypatch.setattr(turn_pipeline_mod, "extract_urls", lambda _text: ["https://ok"])
    monkeypatch.setattr(turn_pipeline_mod, "should_use_ocr", lambda **_kwargs: False)
    monkeypatch.setattr(
        turn_pipeline_mod,
        "fetch_public_page",
        lambda **_kwargs: FetchedPage(
            url="https://ok",
            ok=True,
            title="ok",
            text="ok body",
            content_type="text/html",
        ),
    )
    monkeypatch.setattr(
        turn_pipeline_mod.TurnPipeline,
        "_build_cached_browse_context",
        lambda self, runtime, settings, pages: "CTX",
    )

    result = main_mod.on_submit("check https://ok", image=None)
    assert result.text == "ok"
    assert "Fetching links..." in statuses
    assert "Thinking..." in statuses


def test_on_submit_updates_thinking_status_for_ocr(monkeypatch):
    rt = main_mod.runtime
    fake_ai = _FakeAI(answer="ok")
    statuses = []

    class _Overlay:
        def update_thinking_status(self, text):
            statuses.append(text)

    monkeypatch.setattr(rt, "ai", fake_ai)
    monkeypatch.setattr(rt, "active_overlay", _Overlay())
    monkeypatch.setattr(rt, "onboarding_needed", False)
    monkeypatch.setattr(rt, "target_hwnd", 0)
    monkeypatch.setattr(rt, "current_app", None)
    monkeypatch.setattr(turn_pipeline_mod, "classify_response_mode", lambda **_kwargs: ResponseMode.WORK)
    monkeypatch.setattr(turn_pipeline_mod, "extract_urls", lambda _text: [])
    monkeypatch.setattr(turn_pipeline_mod, "should_use_ocr", lambda **_kwargs: True)
    monkeypatch.setattr(turn_pipeline_mod, "extract_ocr_text", lambda *_args, **_kwargs: "ocr result")
    monkeypatch.setitem(main_mod.cfg, "enable_ocr_fallback", True)

    result = main_mod.on_submit("summarize this", image=Image.new("RGB", (4, 4)))
    assert result.text == "ok"
    assert "Running OCR..." in statuses


def test_on_submit_honors_cancel_token(monkeypatch):
    rt = main_mod.runtime
    fake_ai = _FakeAI(answer="ok")

    class _Token:
        def is_set(self):
            return True

    monkeypatch.setattr(rt, "ai", fake_ai)
    monkeypatch.setattr(rt, "onboarding_needed", False)
    monkeypatch.setattr(rt, "target_hwnd", 0)
    monkeypatch.setattr(rt, "current_app", None)
    monkeypatch.setattr(turn_pipeline_mod, "classify_response_mode", lambda **_kwargs: ResponseMode.WORK)
    monkeypatch.setattr(turn_pipeline_mod, "extract_urls", lambda _text: [])

    result = main_mod.on_submit("cancel me", image=None, cancel_token=_Token())
    assert "cancelled" in result.text.lower()
    assert fake_ai.calls == []


def test_on_submit_reuses_url_cache(monkeypatch):
    rt = main_mod.runtime
    fake_ai = _FakeAI(answer="ok")
    fetch_calls = {"n": 0}
    rt.url_cache.clear()

    monkeypatch.setattr(rt, "ai", fake_ai)
    monkeypatch.setattr(rt, "onboarding_needed", False)
    monkeypatch.setattr(rt, "target_hwnd", 0)
    monkeypatch.setattr(rt, "current_app", None)
    monkeypatch.setattr(turn_pipeline_mod, "classify_response_mode", lambda **_kwargs: ResponseMode.WORK)
    monkeypatch.setattr(turn_pipeline_mod, "extract_urls", lambda _text: ["https://cache-me"])
    monkeypatch.setitem(main_mod.cfg, "allow_private_url_browse", True)
    monkeypatch.setitem(main_mod.cfg, "url_cache_ttl_sec", 600)

    def _fake_fetch(**_kwargs):
        fetch_calls["n"] += 1
        return FetchedPage(
            url="https://cache-me",
            ok=True,
            title="cached",
            text="cached body",
            content_type="text/html",
        )

    monkeypatch.setattr(turn_pipeline_mod, "fetch_public_page", _fake_fetch)
    monkeypatch.setattr(
        turn_pipeline_mod.TurnPipeline,
        "_build_cached_browse_context",
        lambda self, runtime, settings, pages: "CTX",
    )

    main_mod.on_submit("check https://cache-me", image=None)
    main_mod.on_submit("check https://cache-me again", image=None)
    assert fetch_calls["n"] == 1


def test_on_submit_uses_context_reference_on_followup(monkeypatch):
    rt = main_mod.runtime
    fake_ai = _FakeAI(answer="ok")
    rt.static_context_hash = ""
    rt.static_context_id = ""
    rt.static_context_last_full_turn = 0
    rt.turn_counter = 0

    class _App:
        app_type = SimpleNamespace(value="browser")

    monkeypatch.setattr(rt, "ai", fake_ai)
    monkeypatch.setattr(rt, "onboarding_needed", False)
    monkeypatch.setattr(rt, "target_hwnd", 0)
    monkeypatch.setattr(rt, "current_app", _App())
    monkeypatch.setattr(turn_pipeline_mod, "build_context_prompt", lambda _app: "Active app: Browser")
    monkeypatch.setattr(turn_pipeline_mod, "classify_response_mode", lambda **_kwargs: ResponseMode.WORK)
    monkeypatch.setattr(turn_pipeline_mod, "extract_urls", lambda _text: [])
    monkeypatch.setitem(main_mod.cfg, "context_reference_refresh_turns", 3)

    main_mod.on_submit("first", image=None)
    main_mod.on_submit("second", image=None)

    first_sent = fake_ai.calls[0]["question"]
    second_sent = fake_ai.calls[1]["question"]
    assert "[Context id:" in first_sent
    assert "[Context reference id:" in second_sent


def test_restore_recent_session_restores_overlay_and_image(monkeypatch):
    rt = main_mod.runtime
    screenshot = Image.new("RGB", (4, 4))
    restore_calls = []
    restored_states = []

    class _RecoverableAI:
        history = ["existing"]

        def restore_session_state(self, state):
            restored_states.append(state)

    class _Overlay:
        def restore_session(self, **kwargs):
            restore_calls.append(kwargs)

    monkeypatch.setattr(rt, "ai", _RecoverableAI())
    rt.recovery_snapshot = SessionRecoverySnapshot(
        created_at=time.monotonic(),
        view=RecoveryViewState(
            answer_text="Recovered answer",
            draft_text="follow-up",
            response_mode=ResponseMode.WORK,
            controls=ContextControls(screenshot=True, clipboard=False, urls=True, ocr=True),
            availability=ContextAvailability(screenshot=True, clipboard=False, urls=True, ocr=True),
        ),
        assistant_state=object(),
        window_title="BuddyGPT",
        target_hwnd=123,
        current_app=None,
        clipboard_context_text="",
        clipboard_context_pending=False,
        static_context_hash="hash",
        static_context_id="ctx12345",
        static_context_last_full_turn=2,
        turn_counter=2,
        previous_context_blocks={"app": "[Active app: browser]"},
        image=screenshot,
    )

    restored = main_mod._restore_recent_session(rt, _Overlay())

    assert restored is True
    assert restored_states
    assert restore_calls[0]["answer_text"] == "Recovered answer"
    assert restore_calls[0]["draft_text"] == "follow-up"
    assert restore_calls[0]["image"] is screenshot
    assert rt.recovery_snapshot is None


def test_turn_pipeline_pack_context_blocks_keeps_required_block_before_optional_trim():
    pipeline = turn_pipeline_mod.TurnPipeline()
    packed, dropped, trimmed = pipeline._pack_context_blocks(
        blocks=[
            turn_pipeline_mod.ContextBlock(
                name="browse_context",
                text="x" * 950,
                priority=20,
                source="url",
                allow_trim=True,
            ),
            turn_pipeline_mod.ContextBlock(
                name="mode_hint",
                text="[Mode hint] stay actionable",
                priority=75,
                source="mode_hint",
                required=True,
            ),
        ],
        question="q" * 50,
        max_chars=1000,
    )

    packed_names = [name for name, _text in packed]
    assert "mode_hint" in packed_names
    assert "browse_context" in packed_names
    assert trimmed is True
    assert dropped == []
