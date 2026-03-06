"""Helpers for consistent structured runtime logging."""

from __future__ import annotations

import logging
from typing import Any


def log_event(logger: logging.Logger, event: str, flow: str, result: str, **fields: Any) -> None:
    parts = [f"event={event}", f"flow={flow}", f"result={result}"]
    for key, value in fields.items():
        if value is None:
            continue
        parts.append(f"{key}={value}")
    logger.info(" ".join(parts))
