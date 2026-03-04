"""Proactive hint gating controller (threshold, cooldown, and rate limit)."""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime


_SENSITIVITY_FACTORS = {
    "low": 2.0,
    "medium": 1.25,
    "high": 0.67,
}


@dataclass
class HintDecision:
    allowed: bool
    reason: str


class ProactiveHintController:
    def __init__(
        self,
        *,
        base_threshold: int,
        sensitivity: str = "medium",
        cooldown_sec: int = 90,
        max_per_hour: int = 8,
        quiet_hours_enabled: bool = False,
        quiet_start: str = "22:00",
        quiet_end: str = "08:00",
    ):
        self.base_threshold = max(1, int(base_threshold))
        self.sensitivity = (sensitivity or "medium").strip().lower()
        self.cooldown_sec = max(1, int(cooldown_sec))
        self.max_per_hour = max(1, int(max_per_hour))
        self.quiet_hours_enabled = bool(quiet_hours_enabled)
        self._quiet_start_min = self._parse_hhmm_to_min(quiet_start)
        self._quiet_end_min = self._parse_hhmm_to_min(quiet_end)
        self._last_shown_ts: float = 0.0
        self._shown_ts: deque[float] = deque()
        self._snooze_until_ts: float = 0.0

    @property
    def threshold(self) -> int:
        factor = _SENSITIVITY_FACTORS.get(self.sensitivity, _SENSITIVITY_FACTORS["medium"])
        return max(1, int(round(self.base_threshold * factor)))

    def _prune_hour_window(self, now_ts: float) -> None:
        cutoff = now_ts - 3600.0
        while self._shown_ts and self._shown_ts[0] < cutoff:
            self._shown_ts.popleft()

    def _parse_hhmm_to_min(self, value: str) -> int | None:
        try:
            hh_s, mm_s = str(value).strip().split(":", 1)
            hh = int(hh_s)
            mm = int(mm_s)
        except (ValueError, TypeError):
            return None
        if hh < 0 or hh > 23 or mm < 0 or mm > 59:
            return None
        return hh * 60 + mm

    def _is_quiet_hour(self, now_ts: float) -> bool:
        if not self.quiet_hours_enabled:
            return False
        start = self._quiet_start_min
        end = self._quiet_end_min
        if start is None or end is None or start == end:
            return False

        now_local = datetime.fromtimestamp(now_ts)
        now_min = now_local.hour * 60 + now_local.minute
        if start < end:
            return start <= now_min < end
        return now_min >= start or now_min < end

    def snooze(self, *, minutes: int, now_ts: float | None = None) -> None:
        now_ts = now_ts if now_ts is not None else time.time()
        minutes = max(1, int(minutes))
        self._snooze_until_ts = now_ts + (minutes * 60.0)

    def clear_snooze(self) -> None:
        self._snooze_until_ts = 0.0

    @property
    def snooze_until_ts(self) -> float:
        return self._snooze_until_ts

    def should_show_hint(
        self,
        *,
        distance: int,
        onboarding_needed: bool,
        overlay_resting: bool,
        now_ts: float | None = None,
    ) -> HintDecision:
        now_ts = now_ts if now_ts is not None else time.time()
        if onboarding_needed:
            return HintDecision(False, "onboarding")
        if self._snooze_until_ts and now_ts < self._snooze_until_ts:
            return HintDecision(False, "snoozed")
        if self._is_quiet_hour(now_ts):
            return HintDecision(False, "quiet_hours")
        if not overlay_resting:
            return HintDecision(False, "overlay_not_resting")
        if distance < self.threshold:
            return HintDecision(False, "below_threshold")
        if self._last_shown_ts and (now_ts - self._last_shown_ts) < self.cooldown_sec:
            return HintDecision(False, "cooldown")
        self._prune_hour_window(now_ts)
        if len(self._shown_ts) >= self.max_per_hour:
            return HintDecision(False, "rate_limit")
        return HintDecision(True, "screen_change")

    def mark_shown(self, *, now_ts: float | None = None) -> None:
        now_ts = now_ts if now_ts is not None else time.time()
        self._last_shown_ts = now_ts
        self._shown_ts.append(now_ts)
        self._prune_hour_window(now_ts)
