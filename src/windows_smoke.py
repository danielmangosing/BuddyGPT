"""Windows-oriented smoke checks for release validation."""

from __future__ import annotations

import argparse
import json
import platform
import tkinter as tk
from dataclasses import asdict, dataclass
from typing import Callable

from PIL import Image, ImageDraw, ImageFont

from .clipboard_utils import get_clipboard_text
from .hotkey import HotkeyManager, parse_hotkey
from .ocr import extract_ocr_text, is_ocr_available


@dataclass(slots=True)
class SmokeCheckResult:
    name: str
    status: str
    details: str


def _set_clipboard_text(text: str) -> None:
    root = tk.Tk()
    root.withdraw()
    try:
        root.clipboard_clear()
        root.clipboard_append(text)
        root.update()
    finally:
        root.destroy()


def run_hotkey_check() -> SmokeCheckResult:
    hits = {"activate": 0, "clipboard": 0, "quit": 0}
    hotkey = parse_hotkey("ctrl+shift+space")
    clipboard = parse_hotkey("ctrl+shift+v")
    quit_hotkey = parse_hotkey("ctrl+shift+q")
    hk = HotkeyManager(hotkey=hotkey, clipboard_hotkey=clipboard, quit_hotkey=quit_hotkey)
    hk.on_activate(lambda: hits.__setitem__("activate", hits["activate"] + 1))
    hk.on_clipboard(lambda: hits.__setitem__("clipboard", hits["clipboard"] + 1))
    hk.on_quit(lambda: hits.__setitem__("quit", hits["quit"] + 1))

    for key in hotkey:
        hk._on_press(key)
    for key in hotkey:
        hk._on_release(key)
    for key in clipboard:
        hk._on_press(key)
    for key in clipboard:
        hk._on_release(key)
    for key in quit_hotkey:
        hk._on_press(key)
    for key in quit_hotkey:
        hk._on_release(key)

    ok = hits == {"activate": 1, "clipboard": 1, "quit": 1}
    details = f"activate={hits['activate']} clipboard={hits['clipboard']} quit={hits['quit']}"
    return SmokeCheckResult("hotkey", "pass" if ok else "fail", details)


def run_clipboard_check() -> SmokeCheckResult:
    original_text, _original_error = get_clipboard_text(max_chars=6000)
    sample = "BuddyGPT smoke test clipboard"
    try:
        _set_clipboard_text(sample)
        text, err = get_clipboard_text(max_chars=6000)
    except Exception as exc:
        return SmokeCheckResult("clipboard", "fail", str(exc))
    finally:
        try:
            _set_clipboard_text(original_text)
        except Exception:
            pass

    if err:
        return SmokeCheckResult("clipboard", "fail", f"read_error={err}")
    if text != sample:
        return SmokeCheckResult("clipboard", "fail", f"unexpected_text={text!r}")
    return SmokeCheckResult("clipboard", "pass", f"chars={len(text)}")


def run_ocr_check() -> SmokeCheckResult:
    if not is_ocr_available():
        return SmokeCheckResult("ocr", "skip", "tesseract_unavailable")

    image = Image.new("RGB", (240, 80), "white")
    draw = ImageDraw.Draw(image)
    draw.text((12, 24), "BuddyGPT", fill="black", font=ImageFont.load_default())
    text = extract_ocr_text(image, max_chars=200, timeout_sec=5)
    if not text.strip():
        return SmokeCheckResult("ocr", "fail", "empty_text")
    return SmokeCheckResult("ocr", "pass", text.strip()[:80])


def run_tray_check() -> SmokeCheckResult:
    try:
        import main as buddy_main
    except Exception as exc:
        return SmokeCheckResult("tray", "fail", f"import_failed={exc}")

    class _Overlay:
        hwnd = 0

    class _Hotkeys:
        def request_quit(self):
            return None

    try:
        icon = buddy_main._setup_tray_icon(_Overlay(), _Hotkeys())
    except Exception as exc:
        return SmokeCheckResult("tray", "fail", f"build_failed={exc}")
    if icon is None:
        return SmokeCheckResult("tray", "fail", "tray_icon_not_created")
    menu_items = getattr(getattr(icon, "menu", None), "items", ())
    item_count = len(tuple(menu_items))
    try:
        icon.stop()
    except Exception:
        pass
    return SmokeCheckResult("tray", "pass", f"menu_items={item_count}")


def run_platform_check() -> SmokeCheckResult:
    if platform.system() != "Windows":
        return SmokeCheckResult("platform", "skip", f"platform={platform.system()}")
    return SmokeCheckResult("platform", "pass", "Windows")


def run_all_checks(selected: list[str] | None = None) -> list[SmokeCheckResult]:
    checks: dict[str, Callable[[], SmokeCheckResult]] = {
        "platform": run_platform_check,
        "hotkey": run_hotkey_check,
        "clipboard": run_clipboard_check,
        "ocr": run_ocr_check,
        "tray": run_tray_check,
    }
    order = selected or list(checks.keys())
    results: list[SmokeCheckResult] = []
    for name in order:
        check = checks[name]
        results.append(check())
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run BuddyGPT Windows smoke checks.")
    parser.add_argument(
        "--checks",
        nargs="*",
        choices=["platform", "hotkey", "clipboard", "ocr", "tray"],
        help="Subset of checks to run.",
    )
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    args = parser.parse_args(argv)

    results = run_all_checks(args.checks)
    if args.json:
        print(json.dumps([asdict(item) for item in results], ensure_ascii=False, indent=2))
    else:
        for item in results:
            print(f"[{item.status.upper()}] {item.name}: {item.details}")

    return 1 if any(item.status == "fail" for item in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
