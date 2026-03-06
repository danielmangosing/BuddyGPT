"""Unit tests for content-aware response mode routing."""

from __future__ import annotations

from types import SimpleNamespace

from src.intent_router import classify_response_mode
from src.interaction_mode import ResponseMode


class DummyAI:
    """Minimal stand-in for classifier signature."""

    client = None
    model = "dummy"


class _BackendClassifierAI:
    model = "gpt-4o-mini"
    client = None

    def __init__(self, text: str):
        self.backend = SimpleNamespace(
            chat=lambda **_kwargs: SimpleNamespace(text=text),
        )


def test_work_app_neutral_text_defaults_work():
    mode = classify_response_mode(
        question="Can you help me draft this response?",
        app_type="gmail",
        ai=DummyAI(),
    )
    assert mode == ResponseMode.WORK


def test_casual_text_can_override_work_prior():
    mode = classify_response_mode(
        question="hi hello thanks haha joke",
        app_type="gmail",
        ai=DummyAI(),
    )
    assert mode == ResponseMode.CASUAL


def test_ambiguous_text_uses_model_fallback(monkeypatch):
    called = {"v": False}

    def _fake_model(question: str, app_type: str, ai):
        called["v"] = True
        return ResponseMode.CASUAL

    monkeypatch.setattr("src.intent_router._classify_with_model", _fake_model)
    mode = classify_response_mode(
        question="ok",
        app_type="browser",
        ai=DummyAI(),
    )
    assert called["v"] is True
    assert mode == ResponseMode.CASUAL


def test_ambiguous_text_uses_backend_chat_fallback_for_non_anthropic():
    mode = classify_response_mode(
        question="ok",
        app_type="browser",
        ai=_BackendClassifierAI("casual"),
    )
    assert mode == ResponseMode.CASUAL


def test_restored_work_keyword_chinese_routes_work():
    mode = classify_response_mode(
        question="\u8fd9\u4e2a\u9879\u76ee\u4eca\u5929\u9700\u8981\u4fee\u590d\u62a5\u9519",
        app_type="browser",
        ai=DummyAI(),
    )
    assert mode == ResponseMode.WORK


def test_restored_casual_keyword_chinese_routes_casual():
    mode = classify_response_mode(
        question="\u4f60\u597d \u8c22\u8c22 \u54c8\u54c8",
        app_type="browser",
        ai=DummyAI(),
    )
    assert mode == ResponseMode.CASUAL
