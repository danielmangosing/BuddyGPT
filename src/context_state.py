"""Shared context-control and recovery state types."""

from __future__ import annotations

from dataclasses import dataclass, field

from .interaction_mode import ResponseMode


@dataclass(slots=True)
class ContextControls:
    screenshot: bool = True
    clipboard: bool = True
    urls: bool = True
    ocr: bool = True


@dataclass(slots=True)
class ContextAvailability:
    screenshot: bool = False
    clipboard: bool = False
    urls: bool = True
    ocr: bool = False

    def default_controls(self) -> ContextControls:
        return ContextControls(
            screenshot=self.screenshot,
            clipboard=self.clipboard,
            urls=self.urls,
            ocr=self.ocr,
        )


@dataclass(slots=True)
class RecoveryViewState:
    answer_text: str = ""
    draft_text: str = ""
    response_mode: ResponseMode = ResponseMode.WORK
    controls: ContextControls = field(default_factory=ContextControls)
    availability: ContextAvailability = field(default_factory=ContextAvailability)
