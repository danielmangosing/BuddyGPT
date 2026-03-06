"""Session recovery state."""

from __future__ import annotations

from dataclasses import dataclass

from PIL import Image

from .ai_assistant import AssistantSessionState
from .app_detector import AppInfo
from .context_state import RecoveryViewState


@dataclass(slots=True)
class SessionRecoverySnapshot:
    created_at: float
    view: RecoveryViewState
    assistant_state: AssistantSessionState
    window_title: str
    target_hwnd: int
    current_app: AppInfo | None
    clipboard_context_text: str
    clipboard_context_pending: bool
    static_context_hash: str
    static_context_id: str
    static_context_last_full_turn: int
    turn_counter: int
    previous_context_blocks: dict[str, str]
    image: Image.Image | None = None
