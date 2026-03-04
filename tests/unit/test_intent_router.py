"""Unit tests for content-aware response mode routing."""

from __future__ import annotations

from src.intent_router import classify_response_mode
from src.interaction_mode import ResponseMode


class DummyAI:
    """Minimal stand-in for classifier signature."""

    pass


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


def test_ambiguous_text_defaults_work():
    mode = classify_response_mode(
        question="ok",
        app_type="browser",
        ai=DummyAI(),
    )
    assert mode == ResponseMode.WORK
