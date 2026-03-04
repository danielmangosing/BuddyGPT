"""Classify each user turn into work or casual response mode."""

from __future__ import annotations

import logging
import re

from .interaction_mode import ResponseMode

logger = logging.getLogger(__name__)

WORK_PRIOR_APPS = {
    "gmail",
    "outlook",
    "slack",
    "vscode",
    "terminal",
    "excel",
    "word",
    "powerpoint",
}

WORK_KEYWORDS = {
    "email",
    "deadline",
    "review",
    "task",
    "ticket",
    "issue",
    "bug",
    "fix",
    "meeting",
    "report",
    "proposal",
    "client",
    "deploy",
    "production",
    "error",
    "work",
}

CASUAL_KEYWORDS = {
    "hi",
    "hello",
    "thanks",
    "thank you",
    "how are you",
    "joke",
    "chat",
    "haha",
    "lol",
    "good morning",
    "good night",
}

RULE_GAP_THRESHOLD = 2


def _keyword_hits(text_lower: str, keywords: set[str]) -> int:
    hits = 0
    for kw in keywords:
        if kw in text_lower:
            hits += 1
    return hits


def classify_response_mode(question: str, app_type: str, ai) -> ResponseMode:
    """Rule-based routing for one user turn.

    Ambiguous cases default to WORK mode to avoid an extra model call.
    """
    _ = ai  # kept for call-site compatibility
    text = question.strip()
    lowered = text.lower()

    work_score = 0
    casual_score = 0

    if app_type in WORK_PRIOR_APPS:
        work_score += 2

    work_score += _keyword_hits(lowered, WORK_KEYWORDS)
    casual_score += _keyword_hits(lowered, CASUAL_KEYWORDS)

    if re.search(r"(?:^|\s)(lol|haha|hehe)(?:\s|$)", lowered):
        casual_score += 1

    gap = work_score - casual_score
    if abs(gap) >= RULE_GAP_THRESHOLD:
        mode = ResponseMode.WORK if gap > 0 else ResponseMode.CASUAL
        logger.info(
            "Mode route: source=rule app=%s work_score=%d casual_score=%d mode=%s",
            app_type,
            work_score,
            casual_score,
            mode.value,
        )
        return mode

    logger.info(
        "Mode route: source=rule_default app=%s work_score=%d casual_score=%d mode=%s",
        app_type,
        work_score,
        casual_score,
        ResponseMode.WORK.value,
    )
    return ResponseMode.WORK
