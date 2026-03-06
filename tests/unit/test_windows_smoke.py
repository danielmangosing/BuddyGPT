"""Tests for the Windows smoke harness orchestration."""

from __future__ import annotations

from src.windows_smoke import SmokeCheckResult, main, run_hotkey_check


def test_run_hotkey_check_passes():
    result = run_hotkey_check()
    assert result.name == "hotkey"
    assert result.status == "pass"


def test_windows_smoke_main_returns_nonzero_on_failure(monkeypatch):
    monkeypatch.setattr(
        "src.windows_smoke.run_all_checks",
        lambda _selected=None: [SmokeCheckResult(name="tray", status="fail", details="boom")],
    )

    code = main(["--checks", "tray"])

    assert code == 1
