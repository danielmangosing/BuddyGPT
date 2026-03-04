"""Optional OCR helpers for text-heavy screenshot workflows."""

from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from pathlib import Path

from PIL import Image

logger = logging.getLogger(__name__)

try:
    import pytesseract

    OCR_IMPORT_OK = True
except Exception:
    OCR_IMPORT_OK = False
    pytesseract = None  # type: ignore[assignment]

DEFAULT_MAX_CHARS = 3000
DEFAULT_TIMEOUT_SEC = 5
DEFAULT_PREFERRED_APPS = (
    "terminal",
    "vscode",
    "gmail",
    "outlook",
    "word",
    "pdf_reader",
)


def _common_windows_tesseract_paths() -> list[Path]:
    return [
        Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe"),
        Path(r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"),
    ]


def resolve_tesseract_cmd(configured_cmd: str = "") -> str:
    """Resolve tesseract executable path from config, common install paths, or PATH."""
    configured = configured_cmd.strip()
    if configured:
        candidate = Path(configured)
        if candidate.exists():
            return str(candidate)
        return configured

    for candidate in _common_windows_tesseract_paths():
        if candidate.exists():
            return str(candidate)

    return os.environ.get("TESSERACT_CMD", "").strip()


def is_ocr_available(configured_cmd: str = "") -> bool:
    if not OCR_IMPORT_OK or pytesseract is None:
        return False
    resolved = resolve_tesseract_cmd(configured_cmd)
    if resolved:
        pytesseract.pytesseract.tesseract_cmd = resolved
    try:
        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False


def should_use_ocr(app_type: str, *, enabled: bool, preferred_apps: list[str] | tuple[str, ...]) -> bool:
    if not enabled:
        return False
    app = (app_type or "").strip().lower()
    return app in {a.strip().lower() for a in preferred_apps}


def _run_tesseract(gray: Image.Image, timeout_sec: int) -> str:
    if pytesseract is None:
        return ""
    return str(pytesseract.image_to_string(gray, timeout=timeout_sec))


def extract_ocr_text(
    image: Image.Image,
    *,
    max_chars: int = DEFAULT_MAX_CHARS,
    timeout_sec: int = DEFAULT_TIMEOUT_SEC,
    tesseract_cmd: str = "",
) -> str:
    if not is_ocr_available(tesseract_cmd):
        return ""

    gray = image.convert("L")
    timeout_sec = max(1, int(timeout_sec))
    max_chars = max(1, int(max_chars))

    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(_run_tesseract, gray, timeout_sec)
            text = future.result(timeout=timeout_sec + 1)
    except TimeoutError:
        logger.warning("OCR timed out after %ss", timeout_sec)
        return ""
    except Exception as exc:
        logger.warning("OCR failed: %s", exc)
        return ""

    cleaned = text.strip()
    if not cleaned:
        return ""
    if len(cleaned) > max_chars:
        cleaned = f"{cleaned[:max_chars]}\n[... OCR truncated]"
    logger.info("OCR extracted %d characters", len(cleaned))
    return cleaned
