"""Windows clipboard helpers for text capture."""

from __future__ import annotations

import ctypes

CF_UNICODETEXT = 13

user32 = ctypes.windll.user32
kernel32 = ctypes.windll.kernel32


def get_clipboard_text(max_chars: int = 6000) -> tuple[str, str]:
    """Return (text, error). Empty text with non-empty error indicates failure."""
    if not user32.OpenClipboard(None):
        return "", "open_failed"
    try:
        if not user32.IsClipboardFormatAvailable(CF_UNICODETEXT):
            return "", "no_text"

        handle = user32.GetClipboardData(CF_UNICODETEXT)
        if not handle:
            return "", "get_data_failed"

        ptr = kernel32.GlobalLock(handle)
        if not ptr:
            return "", "lock_failed"
        try:
            text = ctypes.wstring_at(ptr)
        finally:
            kernel32.GlobalUnlock(handle)
    finally:
        user32.CloseClipboard()

    text = text.strip()
    if not text:
        return "", "empty"
    if len(text) > max_chars:
        text = text[:max_chars]
    return text, ""
