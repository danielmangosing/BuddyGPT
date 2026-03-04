"""Detect which application is in the foreground based on window title and process name."""

import ctypes
from ctypes import wintypes
from dataclasses import dataclass
from enum import Enum

user32 = ctypes.windll.user32
kernel32 = ctypes.windll.kernel32
psapi = ctypes.windll.psapi

PROCESS_QUERY_LIMITED_INFORMATION = 0x1000


class AppType(Enum):
    GMAIL = "gmail"
    OUTLOOK = "outlook"
    BROWSER = "browser"
    VSCODE = "vscode"
    TERMINAL = "terminal"
    SLACK = "slack"
    DISCORD = "discord"
    EXCEL = "excel"
    WORD = "word"
    POWERPOINT = "powerpoint"
    PDF_READER = "pdf_reader"
    FILE_EXPLORER = "file_explorer"
    UNKNOWN = "unknown"


# Friendly display names
APP_LABELS = {
    AppType.GMAIL: "Gmail",
    AppType.OUTLOOK: "Outlook",
    AppType.BROWSER: "Browser",
    AppType.VSCODE: "VS Code",
    AppType.TERMINAL: "Terminal",
    AppType.SLACK: "Slack",
    AppType.DISCORD: "Discord",
    AppType.EXCEL: "Excel",
    AppType.WORD: "Word",
    AppType.POWERPOINT: "PowerPoint",
    AppType.PDF_READER: "PDF Reader",
    AppType.FILE_EXPLORER: "File Explorer",
    AppType.UNKNOWN: "Unknown",
}


@dataclass
class AppInfo:
    app_type: AppType
    label: str
    process_name: str
    window_title: str
    url_hint: str  # extracted from title if browser


def get_process_name(hwnd: int) -> str:
    """Get the executable name for a window handle."""
    pid = wintypes.DWORD()
    user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
    handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid.value)
    if not handle:
        return ""
    try:
        buf = ctypes.create_unicode_buffer(512)
        size = wintypes.DWORD(512)
        kernel32.QueryFullProcessImageNameW(handle, 0, buf, ctypes.byref(size))
        # Return just the filename, e.g. "chrome.exe"
        return buf.value.rsplit("\\", 1)[-1].lower()
    finally:
        kernel32.CloseHandle(handle)


# Detection rules
# Each rule: (process_names, title_keywords) -> AppType
# Process names checked first, then title keywords refine the result.

_PROCESS_MAP = {
    "code.exe": AppType.VSCODE,
    "slack.exe": AppType.SLACK,
    "discord.exe": AppType.DISCORD,
    "excel.exe": AppType.EXCEL,
    "winword.exe": AppType.WORD,
    "powerpnt.exe": AppType.POWERPOINT,
    "acrord32.exe": AppType.PDF_READER,
    "foxitreader.exe": AppType.PDF_READER,
    "sumatrapdf.exe": AppType.PDF_READER,
    "explorer.exe": AppType.FILE_EXPLORER,
}

_TERMINAL_PROCESSES = {
    "cmd.exe", "powershell.exe", "pwsh.exe",
    "windowsterminal.exe", "wt.exe",
    "conhost.exe", "mintty.exe", "alacritty.exe",
    "wezterm-gui.exe",
}

_BROWSER_PROCESSES = {
    "chrome.exe", "msedge.exe", "firefox.exe",
    "brave.exe", "opera.exe", "vivaldi.exe", "arc.exe",
}


def _extract_url_hint(title: str) -> str:
    """Try to extract a domain or URL hint from a browser window title."""
    # Common patterns:
    # - "Page Title - Site Name - Browser" -> Site Name
    # - "Page Title - Browser" -> Page Title
    # - "SingleTitle" -> SingleTitle
    parts = [part.strip() for part in title.split(" - ") if part.strip()]
    if len(parts) >= 3:
        return parts[-2]
    if len(parts) == 2:
        return parts[0]
    return parts[0] if parts else ""


def _detect_browser_subtype(title: str) -> AppType:
    """Check if a browser tab is showing a known web app."""
    t = title.lower()
    if "gmail" in t or "mail.google" in t:
        return AppType.GMAIL
    if "outlook" in t or "mail.live" in t or "outlook.office" in t:
        return AppType.OUTLOOK
    if "slack" in t:
        return AppType.SLACK
    if "discord" in t:
        return AppType.DISCORD
    return AppType.BROWSER


def detect_app(hwnd: int = 0) -> AppInfo:
    """Detect the application for a given window handle (0 = foreground)."""
    if not hwnd:
        hwnd = user32.GetForegroundWindow()

    # Get process name
    proc = get_process_name(hwnd)

    # Get window title
    length = user32.GetWindowTextLengthW(hwnd)
    buf = ctypes.create_unicode_buffer(length + 1)
    user32.GetWindowTextW(hwnd, buf, length + 1)
    title = buf.value

    # Detect by process
    app_type = AppType.UNKNOWN
    url_hint = ""

    if proc in _TERMINAL_PROCESSES:
        app_type = AppType.TERMINAL
    elif proc in _BROWSER_PROCESSES:
        app_type = _detect_browser_subtype(title)
        url_hint = _extract_url_hint(title)
    elif proc in _PROCESS_MAP:
        app_type = _PROCESS_MAP[proc]
    else:
        # Fallback: check title keywords
        t = title.lower()
        if "visual studio code" in t:
            app_type = AppType.VSCODE
        elif "gmail" in t:
            app_type = AppType.GMAIL
        elif "slack" in t:
            app_type = AppType.SLACK

    label = APP_LABELS.get(app_type, "Unknown")

    return AppInfo(
        app_type=app_type,
        label=label,
        process_name=proc,
        window_title=title,
        url_hint=url_hint,
    )
