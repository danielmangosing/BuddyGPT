"""Phase 1 daily proactive chat source with scheduled pushes."""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from typing import Any

from src.pet import PetState

from .base import Notification, NotificationSource

logger = logging.getLogger(__name__)

DAILY_NEWS_STATE_KEY = "daily_news"
WAKE_FIRST_SLOT = "wake_first"
TIMED_SLOTS = ("afternoon_1500", "evening_2000")
DEFAULT_PUSH_TIMES = ("15:00", "20:00")
DEFAULT_MAX_TOPIC_RETRY = 3

DAILY_CHAT_SYSTEM_PROMPT = """
You generate proactive daily news openers for BuddyGPT.

Rules:
- Always use web_search first.
- Pick ONE distinct topic for the current slot.
- Topic must be different from previously used topics today.
- Reply with JSON only, no markdown fences:
  {
    "topic_key": "short-lowercase-topic-key",
    "message": "2-3 sentence casual opener ending with a conversation hook"
  }
- topic_key must describe the news subject, not the wording.
"""


class DailyChatSource(NotificationSource):
    source_id = "daily_chat"

    def __init__(self, ai_assistant, config: dict):
        self._ai = ai_assistant
        self._config = config

    # --- Public API used by manager/main ---

    def is_enabled(self) -> bool:
        settings = self._settings()
        return bool(settings.get("enabled", True))

    def check(self, state) -> bool:
        """Manager compatibility: first wake-up slot."""
        return self.should_trigger_slot(state, WAKE_FIRST_SLOT, datetime.now())

    def generate(self, state) -> Notification | None:
        """Manager compatibility: generate first wake-up slot."""
        return self.generate_for_slot(WAKE_FIRST_SLOT, datetime.now(), state)

    def pending_timed_slots(self, state, now_local: datetime | None = None) -> list[str]:
        """Return due timed slots that are still pending for today."""
        now_local = now_local or datetime.now()
        if not self.is_enabled():
            return []

        push_times = self._timed_slot_schedule()
        pending: list[str] = []
        for slot_id in TIMED_SLOTS:
            if not self.should_trigger_slot(state, slot_id, now_local):
                continue
            hhmm = push_times.get(slot_id, "")
            if not hhmm:
                continue
            if self._is_due(now_local, hhmm):
                pending.append(slot_id)
        return pending

    def should_trigger_slot(self, state, slot_id: str, now_local: datetime | None = None) -> bool:
        now_local = now_local or datetime.now()
        if not self.is_enabled():
            return False
        day = self._day_record(state, now_local)
        return not self._is_slot_done(day, slot_id)

    def generate_for_slot(self, slot_id: str, now_local: datetime, state) -> Notification | None:
        """Generate and persist one slot result.

        Returns a Notification when delivered.
        Returns None when skipped (e.g. no unique topic).
        """
        if not self.should_trigger_slot(state, slot_id, now_local):
            return None

        logger.info(
            "event=SLOT_GENERATE_START flow=daily_chat slot_id=%s date=%s time_local=%s",
            slot_id,
            now_local.date().isoformat(),
            now_local.strftime("%H:%M:%S"),
        )
        day = self._day_record(state, now_local)
        used_topics = {self._normalize_topic(t) for t in day["used_topics"]}
        used_topics.discard("")

        result = self._generate_unique_payload(slot_id, now_local, used_topics)
        if result is None:
            self._mark_slot_status(state, now_local, slot_id, "skipped_no_unique_topic")
            logger.info(
                "event=SLOT_SKIPPED_NO_UNIQUE_TOPIC flow=daily_chat slot_id=%s result=skipped reason=no_unique_topic date=%s time_local=%s",
                slot_id,
                now_local.date().isoformat(),
                now_local.strftime("%H:%M:%S"),
            )
            return None

        topic_key, message = result
        self._mark_slot_delivered(state, now_local, slot_id, topic_key)
        logger.info(
            "event=SLOT_DELIVERED flow=daily_chat slot_id=%s result=delivered topic_key=%s date=%s time_local=%s",
            slot_id,
            topic_key,
            now_local.date().isoformat(),
            now_local.strftime("%H:%M:%S"),
        )
        status = self._slot_status_text(slot_id)
        return Notification(
            source_id=f"{self.source_id}:{slot_id}",
            text=message,
            hint="Enter reply - Esc dismiss",
            status=status,
            priority="normal",
            pet_state=PetState.GREETING,
        )

    # --- Generation logic ---

    def _generate_unique_payload(
        self,
        slot_id: str,
        now_local: datetime,
        used_topics: set[str],
    ) -> tuple[str, str] | None:
        max_retry = int(self._settings().get("max_topic_retry", DEFAULT_MAX_TOPIC_RETRY))
        max_retry = max(1, max_retry)
        generation_system_prompt = f"{self._ai.system_prompt}\n\n{DAILY_CHAT_SYSTEM_PROMPT}"
        if hasattr(self._ai, "spawn_with_overrides"):
            temp_ai = self._ai.spawn_with_overrides(
                system_prompt=generation_system_prompt,
                max_tokens=320,
            )
            use_temp_instance = True
        else:
            temp_ai = self._ai
            use_temp_instance = False
            original_prompt = self._ai.system_prompt
            original_max_tokens = self._ai.max_tokens

        try:
            for _ in range(max_retry):
                temp_ai.clear_history()
                if not use_temp_instance:
                    temp_ai.system_prompt = generation_system_prompt
                    temp_ai.max_tokens = 320
                slot_prompt = self._build_slot_prompt(slot_id, now_local, used_topics)
                raw = temp_ai.ask(
                    slot_prompt,
                    force_search_tool=True,
                    search_hint_question=self._build_search_query(slot_id, now_local),
                ).strip()
                payload = self._extract_json_payload(raw)
                if not payload:
                    continue

                topic_key = self._normalize_topic(str(payload.get("topic_key", "")))
                message = str(payload.get("message", "")).strip()
                if not topic_key or not message:
                    continue
                if topic_key in used_topics:
                    continue
                return topic_key, message
            return None
        except Exception:
            logger.exception("Daily chat generation failed for slot '%s'", slot_id)
            temp_ai.clear_history()
            return None
        finally:
            if not use_temp_instance:
                self._ai.system_prompt = original_prompt
                self._ai.max_tokens = original_max_tokens

    def _build_slot_prompt(self, slot_id: str, now_local: datetime, used_topics: set[str]) -> str:
        slot_label = self._slot_label(slot_id)
        if used_topics:
            used = ", ".join(sorted(used_topics))
        else:
            used = "(none)"
        return (
            f"Today is {now_local.strftime('%Y-%m-%d')}. "
            f"Current slot is {slot_label}. "
            f"Already used topic keys today: {used}. "
            "Return JSON only."
        )

    def _build_search_query(self, slot_id: str, now_local: datetime) -> str:
        slot_label = self._slot_label(slot_id)
        return (
            f"latest major news headlines {now_local.strftime('%Y-%m-%d')} "
            f"for BuddyGPT {slot_label}"
        )

    def _extract_json_payload(self, text: str) -> dict[str, Any] | None:
        candidates = [text.strip()]
        fenced = re.findall(r"\{[\s\S]*\}", text)
        candidates.extend(fenced)

        for candidate in candidates:
            if not candidate:
                continue
            try:
                data = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            if isinstance(data, dict):
                return data
        return None

    # --- Slot schedule/state helpers ---

    def _settings(self) -> dict:
        daily_chat = self._config.get("daily_chat", {})
        if isinstance(daily_chat, dict):
            merged = {
                "enabled": True,
                "push_times": list(DEFAULT_PUSH_TIMES),
                "max_topic_retry": DEFAULT_MAX_TOPIC_RETRY,
            }
            merged.update(daily_chat)
            return merged
        return {
            "enabled": bool(daily_chat),
            "push_times": list(DEFAULT_PUSH_TIMES),
            "max_topic_retry": DEFAULT_MAX_TOPIC_RETRY,
        }

    def _timed_slot_schedule(self) -> dict[str, str]:
        values = self._settings().get("push_times", list(DEFAULT_PUSH_TIMES))
        if not isinstance(values, list):
            values = list(DEFAULT_PUSH_TIMES)
        if len(values) < 2:
            values = list(DEFAULT_PUSH_TIMES)
        return {
            "afternoon_1500": str(values[0]).strip(),
            "evening_2000": str(values[1]).strip(),
        }

    def _is_due(self, now_local: datetime, hhmm: str) -> bool:
        try:
            hh_s, mm_s = hhmm.split(":", 1)
            hh = int(hh_s)
            mm = int(mm_s)
        except (ValueError, AttributeError):
            return False
        if hh < 0 or hh > 23 or mm < 0 or mm > 59:
            return False
        return (now_local.hour, now_local.minute) >= (hh, mm)

    def _slot_label(self, slot_id: str) -> str:
        if slot_id == WAKE_FIRST_SLOT:
            return "first wake-up"
        if slot_id == "afternoon_1500":
            return "afternoon 15:00"
        if slot_id == "evening_2000":
            return "evening 20:00"
        return slot_id

    def _slot_status_text(self, slot_id: str) -> str:
        if slot_id == WAKE_FIRST_SLOT:
            return "Daily chat"
        if slot_id == "afternoon_1500":
            return "News update (3 PM)"
        if slot_id == "evening_2000":
            return "News update (8 PM)"
        return "News update"

    def _day_key(self, now_local: datetime) -> str:
        return now_local.date().isoformat()

    def _default_day_record(self) -> dict[str, Any]:
        return {
            "delivered_slots": [],
            "used_topics": [],
            "slot_status": {},
        }

    def _news_state(self, state) -> dict[str, Any]:
        raw = state.get(DAILY_NEWS_STATE_KEY, {})
        if isinstance(raw, dict):
            return raw
        return {}

    def _day_record(self, state, now_local: datetime) -> dict[str, Any]:
        news = self._news_state(state)
        day = news.get(self._day_key(now_local), {})
        if not isinstance(day, dict):
            day = {}
        merged = self._default_day_record()
        merged.update(day)
        if not isinstance(merged.get("delivered_slots"), list):
            merged["delivered_slots"] = []
        if not isinstance(merged.get("used_topics"), list):
            merged["used_topics"] = []
        if not isinstance(merged.get("slot_status"), dict):
            merged["slot_status"] = {}
        return merged

    def _is_slot_done(self, day_record: dict[str, Any], slot_id: str) -> bool:
        status = day_record.get("slot_status", {})
        return slot_id in status

    def _save_day_record(self, state, now_local: datetime, day_record: dict[str, Any]) -> None:
        news = self._news_state(state)
        key = self._day_key(now_local)
        news[key] = day_record

        # Keep recent records only.
        keep_keys = sorted(news.keys())[-7:]
        news = {k: news[k] for k in keep_keys}
        state.set(DAILY_NEWS_STATE_KEY, news)

    def _mark_slot_status(self, state, now_local: datetime, slot_id: str, status: str) -> None:
        day = self._day_record(state, now_local)
        day["slot_status"][slot_id] = status
        if status == "delivered" and slot_id not in day["delivered_slots"]:
            day["delivered_slots"].append(slot_id)
        self._save_day_record(state, now_local, day)

    def _mark_slot_delivered(self, state, now_local: datetime, slot_id: str, topic_key: str) -> None:
        day = self._day_record(state, now_local)
        if topic_key and topic_key not in day["used_topics"]:
            day["used_topics"].append(topic_key)
        day["slot_status"][slot_id] = "delivered"
        if slot_id not in day["delivered_slots"]:
            day["delivered_slots"].append(slot_id)
        self._save_day_record(state, now_local, day)

    def _normalize_topic(self, topic: str) -> str:
        lowered = topic.strip().lower()
        lowered = re.sub(r"[^a-z0-9]+", "-", lowered)
        lowered = re.sub(r"-{2,}", "-", lowered).strip("-")
        return lowered
