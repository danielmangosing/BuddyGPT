"""Tests for daily chat slot scheduling, dedup, and skip logic."""

from __future__ import annotations

import json
from datetime import datetime, timedelta

import pytest

from src.notifications.daily_chat import (
    DAILY_NEWS_STATE_KEY,
    WAKE_FIRST_SLOT,
    DailyChatSource,
)


# ---------------------------------------------------------------------------
# Test 1: Slot due after time boundary
# ---------------------------------------------------------------------------

class TestSlotDueAfterTimeBoundary:
    """pending_timed_slots respects push_times boundaries."""

    def test_before_first_slot(self, daily_source, fake_state):
        now = datetime(2026, 2, 23, 14, 59)
        result = daily_source.pending_timed_slots(fake_state, now)
        assert result == []

    def test_after_first_slot(self, daily_source, fake_state):
        now = datetime(2026, 2, 23, 15, 1)
        result = daily_source.pending_timed_slots(fake_state, now)
        assert "afternoon_1500" in result

    def test_after_both_slots(self, daily_source, fake_state):
        now = datetime(2026, 2, 23, 20, 1)
        result = daily_source.pending_timed_slots(fake_state, now)
        assert "afternoon_1500" in result
        assert "evening_2000" in result

    def test_exactly_at_boundary(self, daily_source, fake_state):
        now = datetime(2026, 2, 23, 15, 0)
        result = daily_source.pending_timed_slots(fake_state, now)
        assert "afternoon_1500" in result


# ---------------------------------------------------------------------------
# Test 2: Same-day topic dedup across slots
# ---------------------------------------------------------------------------

class TestSameDayTopicDedupe:
    """A topic used by one slot cannot be reused by a later slot on the same day."""

    def test_dedupe_blocks_same_topic(self, fake_ai, default_config, fake_state):
        # AI always returns the same topic.
        fake_ai._responses = [
            '{"topic_key": "ai-breakthrough", "message": "AI made a big leap today."}',
            '{"topic_key": "ai-breakthrough", "message": "AI made a big leap today."}',
            '{"topic_key": "ai-breakthrough", "message": "AI made a big leap today."}',
            '{"topic_key": "ai-breakthrough", "message": "AI made a big leap today."}',
        ]
        source = DailyChatSource(ai_assistant=fake_ai, config=default_config)
        now = datetime(2026, 2, 23, 9, 0)

        # First slot delivers successfully.
        result1 = source.generate_for_slot(WAKE_FIRST_SLOT, now, fake_state)
        assert result1 is not None
        assert "ai-breakthrough" in result1.text.lower() or result1.text != ""

        # Verify topic is recorded.
        day = source._day_record(fake_state, now)
        assert "ai-breakthrough" in day["used_topics"]

        # Second slot with same topic should be skipped.
        now2 = datetime(2026, 2, 23, 15, 1)
        result2 = source.generate_for_slot("afternoon_1500", now2, fake_state)
        assert result2 is None


# ---------------------------------------------------------------------------
# Test 3: Skip when no unique topic after retries
# ---------------------------------------------------------------------------

class TestSkipNoUniqueTopic:
    """When all retries produce duplicate topics, slot is marked skipped."""

    def test_skip_after_exhausting_retries(self, fake_ai, fake_state):
        config = {
            "daily_chat": {
                "enabled": True,
                "push_times": ["15:00", "20:00"],
                "max_topic_retry": 2,
            },
        }
        # AI always returns the same topic.
        fake_ai._responses = [
            '{"topic_key": "ai-news", "message": "Some AI news."}',
            '{"topic_key": "ai-news", "message": "Some AI news."}',
            '{"topic_key": "ai-news", "message": "Some AI news."}',
        ]
        source = DailyChatSource(ai_assistant=fake_ai, config=config)
        now = datetime(2026, 2, 23, 9, 0)

        # Pre-populate state with the topic already used today.
        day_key = now.date().isoformat()
        fake_state.set(DAILY_NEWS_STATE_KEY, {
            day_key: {
                "delivered_slots": [],
                "used_topics": ["ai-news"],
                "slot_status": {},
            },
        })

        result = source.generate_for_slot(WAKE_FIRST_SLOT, now, fake_state)
        assert result is None

        # Verify slot was marked as skipped.
        day = source._day_record(fake_state, now)
        assert day["slot_status"].get(WAKE_FIRST_SLOT) == "skipped_no_unique_topic"


# ---------------------------------------------------------------------------
# Test 4: No cross-day replay
# ---------------------------------------------------------------------------

class TestNoCrossDayReplay:
    """Slots delivered yesterday do not block today's delivery."""

    def test_new_day_resets_slots(self, daily_source, fake_state):
        yesterday = datetime(2026, 2, 22, 9, 0)
        today = datetime(2026, 2, 23, 9, 0)

        # Mark yesterday's wake_first as delivered.
        yesterday_key = yesterday.date().isoformat()
        fake_state.set(DAILY_NEWS_STATE_KEY, {
            yesterday_key: {
                "delivered_slots": [WAKE_FIRST_SLOT],
                "used_topics": ["old-topic"],
                "slot_status": {WAKE_FIRST_SLOT: "delivered"},
            },
        })

        # Today's wake_first should still trigger.
        assert daily_source.should_trigger_slot(fake_state, WAKE_FIRST_SLOT, today) is True

    def test_new_day_timed_slots_available(self, daily_source, fake_state):
        yesterday = datetime(2026, 2, 22, 15, 30)
        today = datetime(2026, 2, 23, 15, 30)

        # Mark yesterday's afternoon slot as delivered.
        yesterday_key = yesterday.date().isoformat()
        fake_state.set(DAILY_NEWS_STATE_KEY, {
            yesterday_key: {
                "delivered_slots": ["afternoon_1500"],
                "used_topics": ["old-topic"],
                "slot_status": {"afternoon_1500": "delivered"},
            },
        })

        # Today's afternoon slot should be pending.
        pending = daily_source.pending_timed_slots(fake_state, today)
        assert "afternoon_1500" in pending


def test_generate_for_slot_passes_search_hint_query(default_config, fake_state):
    class _CapturingAI:
        def __init__(self):
            self.system_prompt = "fake-system"
            self.max_tokens = 300
            self.calls = []

        def spawn_with_overrides(self, **_kwargs):
            return self

        def clear_history(self):
            return None

        def ask(self, question: str, image=None, **kwargs) -> str:
            self.calls.append({"question": question, "image": image, **kwargs})
            return json.dumps(
                {
                    "topic_key": "daily-topic",
                    "message": "A current event opener.",
                }
            )

    ai = _CapturingAI()
    source = DailyChatSource(ai_assistant=ai, config=default_config)
    now = datetime(2026, 2, 23, 9, 0)

    result = source.generate_for_slot(WAKE_FIRST_SLOT, now, fake_state)

    assert result is not None
    assert ai.calls[0]["force_search_tool"] is True
    assert "latest major news headlines 2026-02-23" in ai.calls[0]["search_hint_question"]
