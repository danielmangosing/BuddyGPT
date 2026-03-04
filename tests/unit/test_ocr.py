"""Unit tests for OCR helper gating."""

from src.ocr import DEFAULT_PREFERRED_APPS, should_use_ocr


def test_should_use_ocr_disabled_returns_false():
    assert (
        should_use_ocr(
            app_type="vscode",
            enabled=False,
            preferred_apps=list(DEFAULT_PREFERRED_APPS),
        )
        is False
    )


def test_should_use_ocr_enabled_for_preferred_app():
    assert (
        should_use_ocr(
            app_type="vscode",
            enabled=True,
            preferred_apps=list(DEFAULT_PREFERRED_APPS),
        )
        is True
    )


def test_should_use_ocr_enabled_for_non_preferred_app_returns_false():
    assert (
        should_use_ocr(
            app_type="browser",
            enabled=True,
            preferred_apps=list(DEFAULT_PREFERRED_APPS),
        )
        is False
    )
