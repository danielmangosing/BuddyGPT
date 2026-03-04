"""Unit tests for proactive hint gating policy."""

from datetime import datetime

from src.proactive import ProactiveHintController


def test_threshold_scales_with_sensitivity():
    low = ProactiveHintController(base_threshold=12, sensitivity="low")
    med = ProactiveHintController(base_threshold=12, sensitivity="medium")
    high = ProactiveHintController(base_threshold=12, sensitivity="high")
    assert low.threshold > med.threshold > high.threshold


def test_below_threshold_is_skipped():
    ctl = ProactiveHintController(base_threshold=12, sensitivity="medium")
    decision = ctl.should_show_hint(
        distance=5,
        onboarding_needed=False,
        overlay_resting=True,
        now_ts=100.0,
    )
    assert decision.allowed is False
    assert decision.reason == "below_threshold"


def test_cooldown_blocks_rapid_retrigger():
    ctl = ProactiveHintController(base_threshold=10, sensitivity="high", cooldown_sec=60)
    first = ctl.should_show_hint(
        distance=20,
        onboarding_needed=False,
        overlay_resting=True,
        now_ts=100.0,
    )
    assert first.allowed is True
    ctl.mark_shown(now_ts=100.0)
    second = ctl.should_show_hint(
        distance=20,
        onboarding_needed=False,
        overlay_resting=True,
        now_ts=130.0,
    )
    assert second.allowed is False
    assert second.reason == "cooldown"


def test_rate_limit_blocks_after_cap():
    ctl = ProactiveHintController(
        base_threshold=10,
        sensitivity="high",
        cooldown_sec=1,
        max_per_hour=2,
    )
    for ts in (0.0, 120.0):
        decision = ctl.should_show_hint(
            distance=20,
            onboarding_needed=False,
            overlay_resting=True,
            now_ts=ts,
        )
        assert decision.allowed is True
        ctl.mark_shown(now_ts=ts)

    blocked = ctl.should_show_hint(
        distance=20,
        onboarding_needed=False,
        overlay_resting=True,
        now_ts=180.0,
    )
    assert blocked.allowed is False
    assert blocked.reason == "rate_limit"


def test_quiet_hours_blocks_inside_window():
    ctl = ProactiveHintController(
        base_threshold=10,
        sensitivity="high",
        quiet_hours_enabled=True,
        quiet_start="22:00",
        quiet_end="08:00",
    )
    now = datetime(2026, 3, 4, 23, 0).timestamp()
    decision = ctl.should_show_hint(
        distance=20,
        onboarding_needed=False,
        overlay_resting=True,
        now_ts=now,
    )
    assert decision.allowed is False
    assert decision.reason == "quiet_hours"


def test_quiet_hours_allows_outside_window():
    ctl = ProactiveHintController(
        base_threshold=10,
        sensitivity="high",
        quiet_hours_enabled=True,
        quiet_start="22:00",
        quiet_end="08:00",
    )
    now = datetime(2026, 3, 4, 13, 0).timestamp()
    decision = ctl.should_show_hint(
        distance=20,
        onboarding_needed=False,
        overlay_resting=True,
        now_ts=now,
    )
    assert decision.allowed is True


def test_snooze_blocks_then_can_resume():
    ctl = ProactiveHintController(base_threshold=10, sensitivity="high")
    ctl.snooze(minutes=60, now_ts=100.0)
    blocked = ctl.should_show_hint(
        distance=20,
        onboarding_needed=False,
        overlay_resting=True,
        now_ts=120.0,
    )
    assert blocked.allowed is False
    assert blocked.reason == "snoozed"

    ctl.clear_snooze()
    allowed = ctl.should_show_hint(
        distance=20,
        onboarding_needed=False,
        overlay_resting=True,
        now_ts=120.0,
    )
    assert allowed.allowed is True
