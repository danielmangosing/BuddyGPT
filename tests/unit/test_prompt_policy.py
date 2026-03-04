"""Tests for system prompt policy compliance and structured log output."""

from __future__ import annotations

import logging
from datetime import datetime

from src.notifications.daily_chat import DAILY_NEWS_STATE_KEY, WAKE_FIRST_SLOT
from src.prompts import SYSTEM_PROMPT


class TestPromptPolicyRules:
    """SYSTEM_PROMPT must contain required Soul.md policy phrases."""

    def test_concise_response_rule(self):
        assert "short and focused" in SYSTEM_PROMPT or "1-3 sentences" in SYSTEM_PROMPT

    def test_turn_by_turn_rule(self):
        assert "turn-by-turn" in SYSTEM_PROMPT

    def test_language_following_rule(self):
        assert "same language" in SYSTEM_PROMPT

    def test_time_awareness_rule(self):
        assert "web_search" in SYSTEM_PROMPT

    def test_creative_boundary_rule(self):
        assert "subjective" in SYSTEM_PROMPT.lower()

    def test_no_autonomous_execution_rule(self):
        prompt_lower = SYSTEM_PROMPT.lower()
        assert (
            ("not" in prompt_lower and "autonomous" in prompt_lower)
            or ("do not execute" in prompt_lower)
            or ("unless explicitly asked" in prompt_lower)
        )

    def test_prompt_is_english(self):
        for char in SYSTEM_PROMPT:
            assert ord(char) < 0x4E00 or ord(char) > 0x9FFF, (
                f"Found non-English character in SYSTEM_PROMPT: {char!r}"
            )


class TestStructuredLogFields:
    """Protocol events should emit structured key=value log fields."""

    def test_generation_logs_contain_required_fields(
        self, fake_ai, default_config, fake_state, caplog
    ):
        from src.notifications.daily_chat import DailyChatSource

        fake_ai._responses = ['{"topic_key": "test-topic", "message": "Test message."}']
        source = DailyChatSource(ai_assistant=fake_ai, config=default_config)
        now = datetime(2026, 2, 23, 9, 0)

        with caplog.at_level(logging.INFO, logger="src.notifications.daily_chat"):
            source.generate_for_slot(WAKE_FIRST_SLOT, now, fake_state)

        log_text = "\n".join(record.getMessage() for record in caplog.records)
        assert "event=SLOT_GENERATE_START" in log_text
        assert "event=SLOT_DELIVERED" in log_text
        assert "flow=daily_chat" in log_text
        assert f"slot_id={WAKE_FIRST_SLOT}" in log_text
        assert "result=delivered" in log_text

    def test_skip_logs_contain_required_fields(self, fake_ai, fake_state, caplog):
        config = {
            "daily_chat": {
                "enabled": True,
                "push_times": ["15:00", "20:00"],
                "max_topic_retry": 1,
            },
        }
        fake_ai._responses = ['{"topic_key": "dup-topic", "message": "Dup."}']
        from src.notifications.daily_chat import DailyChatSource

        source = DailyChatSource(ai_assistant=fake_ai, config=config)
        now = datetime(2026, 2, 23, 9, 0)
        day_key = now.date().isoformat()
        fake_state.set(
            DAILY_NEWS_STATE_KEY,
            {
                day_key: {
                    "delivered_slots": [],
                    "used_topics": ["dup-topic"],
                    "slot_status": {},
                },
            },
        )

        with caplog.at_level(logging.INFO, logger="src.notifications.daily_chat"):
            result = source.generate_for_slot(WAKE_FIRST_SLOT, now, fake_state)

        assert result is None
        log_text = "\n".join(record.getMessage() for record in caplog.records)
        assert "event=SLOT_SKIPPED_NO_UNIQUE_TOPIC" in log_text
        assert f"slot_id={WAKE_FIRST_SLOT}" in log_text
        assert "result=skipped" in log_text
        assert "reason=no_unique_topic" in log_text
