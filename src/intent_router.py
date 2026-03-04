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
    "项目",
    "工作",
    "邮件",
    "修复",
    "上线",
    "报错",
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
    "闲聊",
    "你好",
    "谢谢",
    "哈哈",
}

RULE_GAP_THRESHOLD = 2


def _keyword_hits(text_lower: str, keywords: set[str]) -> int:
    hits = 0
    for kw in keywords:
        if kw in text_lower:
            hits += 1
    return hits


def _normalize_output_to_mode(raw: str) -> ResponseMode:
    text = raw.strip().lower()
    if "casual" in text:
        return ResponseMode.CASUAL
    if "work" in text:
        return ResponseMode.WORK
    return ResponseMode.WORK


def _classify_with_model(question: str, app_type: str, ai) -> ResponseMode:
    """Fallback classifier using model output `work` or `casual`."""
    client = getattr(ai, "client", None)
    model = getattr(ai, "model", None)
    if client is None or not model:
        return ResponseMode.WORK
    messages_api = getattr(client, "messages", None)
    create_fn = getattr(messages_api, "create", None)
    if create_fn is None:
        return ResponseMode.WORK

    prompt = (
        "Classify the user's message into exactly one label: work or casual.\n"
        "Rules:\n"
        "- work: task-oriented, problem solving, email/code/productivity intent\n"
        "- casual: social chat, greetings, jokes, gratitude, small talk\n"
        f"App context: {app_type or 'unknown'}\n"
        f"Message: {question}\n"
        "Output exactly one word: work or casual."
    )

    try:
        response = create_fn(
            model=model,
            max_tokens=8,
            system=(
                "You are a strict classifier. Return one tokenized word only: "
                "work or casual."
            ),
            messages=[{"role": "user", "content": [{"type": "text", "text": prompt}]}],
        )
        text_parts = [getattr(block, "text", "") for block in getattr(response, "content", [])]
        return _normalize_output_to_mode(" ".join(text_parts))
    except Exception:
        logger.exception("Intent fallback classification failed")
        return ResponseMode.WORK


def classify_response_mode(question: str, app_type: str, ai) -> ResponseMode:
    """Hybrid rule + model fallback classification for one user turn."""
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

    mode = _classify_with_model(question=text, app_type=app_type, ai=ai)
    logger.info(
        "Mode route: source=model app=%s work_score=%d casual_score=%d mode=%s",
        app_type,
        work_score,
        casual_score,
        mode.value,
    )
    return mode
