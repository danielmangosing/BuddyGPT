"""AI assistant with multi-backend support, vision, and optional web search."""

from __future__ import annotations

import base64
import io
import logging
import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from PIL import Image

from .backends import (
    DEFAULT_BACKEND,
    OLLAMA_BASE_URL,
    OPENAI_BASE_URL,
    BackendResponse,
    build_backend,
)
from .prompts import APP_PROMPTS, PERSONALITIES
from .web_search import format_results, search

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-sonnet-4-20250514"
DEFAULT_HISTORY_WINDOW_TURNS = 6
DEFAULT_HISTORY_SUMMARY_EVERY_TURNS = 6
DEFAULT_HISTORY_SUMMARY_MAX_CHARS = 1800

_IMAGE_PRESETS = {
    "terminal": {"max_size": 800, "quality": 60},
    "vscode": {"max_size": 900, "quality": 65},
    "excel": {"max_size": 900, "quality": 65},
}
_DEFAULT_PRESET = {"max_size": 1024, "quality": 70}
_OCR_PRESET = {"max_size": 512, "quality": 40}

_SEARCH_KEYWORDS = {
    "latest",
    "current",
    "recent",
    "today",
    "news",
    "update",
    "updated",
    "release",
    "version",
    "as of",
    "who is",
    "what is",
    "when did",
    "where can i find",
    "documentation",
    "docs",
    "search",
    "look up",
    "lookup",
    "find online",
    "verify",
    "fact-check",
}
_SCREEN_SIGNALS = [
    re.compile(r"\bon screen\b"),
    re.compile(r"\bthis (error|code|line|issue|page)\b"),
    re.compile(r"\bthat (error|code|line|issue|page)\b"),
    re.compile(r"\bline\s+\d+\b"),
    re.compile(r"\bhere\b"),
]

_PRICING_PER_MILLION = {
    "claude-haiku-4-5-20251001": {"input": 0.80, "output": 4.00},
    "claude-sonnet-4-20250514": {"input": 3.00, "output": 15.00},
    "claude-sonnet-4-5-20250929": {"input": 3.00, "output": 15.00},
    "claude-opus-4-6": {"input": 15.00, "output": 75.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o": {"input": 2.50, "output": 10.00},
}
_DEFAULT_PRICING = {"input": 3.00, "output": 15.00}

SEARCH_TOOL = {
    "name": "web_search",
    "description": (
        "Search the web for current information. Use this when you need to "
        "verify facts, look up documentation, find recent news, or when the "
        "user's question requires information you're not sure about. "
        "Always use web search for questions about current events, recent news, "
        "or anything that may have changed after your knowledge cutoff. "
        "Include the current year in search queries when looking for recent information."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": (
                    "The search query. Include the year or date when searching "
                    "for recent or time-sensitive information."
                ),
            },
        },
        "required": ["query"],
    },
}


@dataclass
class ChatMessage:
    role: str
    text: str
    image: Image.Image | None = None


@dataclass
class UsageStats:
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
    estimated_cost_usd: float = 0.0
    session_total_usd: float = 0.0


class AIAssistant:
    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
        system_prompt: str | None = None,
        max_tokens: int | None = None,
        personality: str = "buddy",
        history_window_turns: int = DEFAULT_HISTORY_WINDOW_TURNS,
        history_summary_every_turns: int = DEFAULT_HISTORY_SUMMARY_EVERY_TURNS,
        history_summary_max_chars: int = DEFAULT_HISTORY_SUMMARY_MAX_CHARS,
        backend: str = DEFAULT_BACKEND,
        openai_api_key: str | None = None,
        ollama_base_url: str = OLLAMA_BASE_URL,
        openai_base_url: str = OPENAI_BASE_URL,
        backend_timeout_sec: int = 45,
    ):
        self.backend_name = str(backend or DEFAULT_BACKEND).lower()
        self._configured_model = model
        self.personality = str(personality or "buddy").lower()
        persona = PERSONALITIES.get(self.personality, PERSONALITIES["buddy"])

        self.system_prompt = system_prompt if system_prompt is not None else persona["prompt"]
        if max_tokens is None:
            max_tokens = int(persona["max_tokens"])
        self.max_tokens = max(1, int(max_tokens))
        self.history_window_turns = max(1, int(history_window_turns or DEFAULT_HISTORY_WINDOW_TURNS))
        self.history_summary_every_turns = max(
            1, int(history_summary_every_turns or DEFAULT_HISTORY_SUMMARY_EVERY_TURNS)
        )
        self.history_summary_max_chars = max(
            256, int(history_summary_max_chars or DEFAULT_HISTORY_SUMMARY_MAX_CHARS)
        )

        anthro_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        openai_key = openai_api_key or os.environ.get("OPENAI_API_KEY", "")
        timeout_sec = max(5, int(backend_timeout_sec))
        self._anthropic_api_key = anthro_key
        self._openai_api_key = openai_key
        self._ollama_base_url = ollama_base_url
        self._openai_base_url = openai_base_url
        self._backend_timeout_sec = timeout_sec

        self.backend = build_backend(
            backend_name=self.backend_name,
            model=model,
            anthropic_api_key=anthro_key,
            openai_api_key=openai_key,
            ollama_base_url=ollama_base_url,
            openai_base_url=openai_base_url,
            timeout_sec=timeout_sec,
        )
        self.backend_name = self.backend.backend_name
        self.model = getattr(self.backend, "model", model)

        self.history: list[ChatMessage] = []
        self._app_type: str = ""
        self._ocr_active_for_turn = False
        self._history_summary: str = ""

        self._session_cost: float = 0.0
        self._last_usage: UsageStats | None = None

    @property
    def client(self):
        # Backward compatibility for tests/legacy modules.
        return getattr(self.backend, "client", None)

    @client.setter
    def client(self, value):
        if hasattr(self.backend, "client"):
            self.backend.client = value

    def set_app_context(self, app_type: str):
        self._app_type = app_type

    def get_last_usage(self) -> UsageStats | None:
        return self._last_usage

    def validate_key(self) -> tuple[bool, str]:
        return self.backend.validate()

    def _get_full_system_prompt(self) -> str:
        prompt = self.system_prompt

        now = datetime.now()
        time_str = now.strftime("%Y-%m-%d %H:%M %A")
        prompt += f"\n\n## Current date and time\n{time_str}"

        app_addition = APP_PROMPTS.get(self._app_type, "")
        if app_addition:
            prompt += f"\n\n## Current context\n{app_addition}"
        if self._history_summary:
            prompt += f"\n\n## Session summary\n{self._history_summary}"
        return prompt

    def _get_system_payload(self) -> list[dict[str, Any]] | str:
        prompt = self._get_full_system_prompt()
        if self.backend_name == "anthropic":
            return [
                {
                    "type": "text",
                    "text": prompt,
                    "cache_control": {"type": "ephemeral"},
                }
            ]
        return prompt

    def _image_to_base64(self, img: Image.Image) -> str:
        preset = _OCR_PRESET if self._ocr_active_for_turn else _IMAGE_PRESETS.get(
            self._app_type, _DEFAULT_PRESET
        )
        max_size = preset["max_size"]
        quality = preset["quality"]

        w, h = img.size
        if max(w, h) > max_size:
            scale = max_size / max(w, h)
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        size_kb = buf.tell() / 1024
        logger.info(
            "Image: %dx%d -> %dx%d, %.0fKB (quality=%d, ocr_mode=%s)",
            w,
            h,
            img.size[0],
            img.size[1],
            size_kb,
            quality,
            self._ocr_active_for_turn,
        )
        return base64.standard_b64encode(buf.getvalue()).decode("utf-8")

    def _build_user_content(self, question: str, image: Image.Image | None = None) -> list[dict[str, Any]]:
        content: list[dict[str, Any]] = []
        if image is not None:
            b64 = self._image_to_base64(image)
            content.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": b64,
                    },
                }
            )
        content.append({"type": "text", "text": question})
        return content

    def _trim_history(self) -> None:
        user_indices = [i for i, msg in enumerate(self.history) if msg.role == "user"]
        if len(user_indices) <= self.history_window_turns:
            return

        keep_start = user_indices[-self.history_window_turns]
        dropped = self.history[:keep_start]
        self._update_history_summary(dropped)
        trimmed = self.history[keep_start:]

        image_user_indices = [
            i for i, msg in enumerate(self.history) if msg.role == "user" and msg.image is not None
        ]
        if image_user_indices:
            image_idx = image_user_indices[-1]
            if image_idx < keep_start:
                seed = [self.history[image_idx]]
                if image_idx + 1 < len(self.history):
                    maybe_assistant = self.history[image_idx + 1]
                    if maybe_assistant.role == "assistant":
                        seed.append(maybe_assistant)
                trimmed = seed + trimmed

        self.history = trimmed

    def _summarize_messages(self, messages: list[ChatMessage]) -> str:
        lines: list[str] = []
        for msg in messages:
            text = msg.text.strip().replace("\n", " ")
            if not text:
                continue
            text = text[:180]
            if msg.role == "user":
                lines.append(f"User: {text}")
            else:
                lines.append(f"Assistant: {text}")
            if len(lines) >= 12:
                break
        return " | ".join(lines).strip()

    def _update_history_summary(self, dropped_messages: list[ChatMessage]) -> None:
        if not dropped_messages:
            return
        user_count = sum(1 for m in dropped_messages if m.role == "user")
        if user_count <= 0:
            return
        # Refresh summary only on configured cadence to avoid excessive prompt churn.
        existing_users = sum(1 for m in self.history if m.role == "user")
        if (existing_users % self.history_summary_every_turns) != 0:
            return

        segment = self._summarize_messages(dropped_messages)
        if not segment:
            return
        if not self._history_summary:
            merged = segment
        else:
            merged = f"{self._history_summary} | {segment}"
        if len(merged) > self.history_summary_max_chars:
            merged = merged[-self.history_summary_max_chars :]
        self._history_summary = merged

    def _build_messages(self, *, include_images: bool = True) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []
        for i, msg in enumerate(self.history):
            if msg.role == "user":
                is_latest_user = (i == len(self.history) - 1) or all(
                    m.role == "assistant" for m in self.history[i + 1 :]
                )
                img = msg.image if (is_latest_user and include_images) else None
                messages.append(
                    {
                        "role": "user",
                        "content": self._build_user_content(msg.text, img),
                    }
                )
            else:
                messages.append({"role": "assistant", "content": msg.text})
        return messages

    def _should_include_search(self, question: str) -> bool:
        q_lower = question.lower()
        if any(pattern.search(q_lower) for pattern in _SCREEN_SIGNALS):
            return False
        return any(keyword in q_lower for keyword in _SEARCH_KEYWORDS)

    def _extract_text_blocks(self, response: BackendResponse | Any) -> str:
        if isinstance(response, BackendResponse):
            return response.text.strip()
        if isinstance(response, str):
            return response.strip()
        text_parts: list[str] = []
        for block in getattr(response, "content", []):
            block_text = getattr(block, "text", "")
            if block_text:
                text_parts.append(block_text)
        return "\n".join(text_parts).strip()

    def _estimate_cost(
        self,
        input_tokens: int,
        output_tokens: int,
        cached_tokens: int = 0,
    ) -> float:
        if self.backend_name == "ollama":
            return 0.0
        pricing = _PRICING_PER_MILLION.get(self.model, _DEFAULT_PRICING)
        normal_input = max(input_tokens - cached_tokens, 0)
        return (
            (normal_input / 1_000_000) * pricing["input"]
            + (cached_tokens / 1_000_000) * pricing["input"] * 0.1
            + (output_tokens / 1_000_000) * pricing["output"]
        )

    def _accumulate_usage(self, response: BackendResponse, turn_usage: dict[str, int]) -> None:
        turn_usage["input"] += int(response.input_tokens)
        turn_usage["output"] += int(response.output_tokens)
        turn_usage["cached"] += int(response.cached_tokens)
        logger.info(
            "Tokens - input: %d (cached: %d), output: %d",
            int(response.input_tokens),
            int(response.cached_tokens),
            int(response.output_tokens),
        )

    def _finalize_usage(self, turn_usage: dict[str, int]) -> None:
        cost = self._estimate_cost(
            input_tokens=turn_usage["input"],
            output_tokens=turn_usage["output"],
            cached_tokens=turn_usage["cached"],
        )
        self._session_cost += cost
        self._last_usage = UsageStats(
            input_tokens=turn_usage["input"],
            output_tokens=turn_usage["output"],
            cached_tokens=turn_usage["cached"],
            estimated_cost_usd=cost,
            session_total_usd=self._session_cost,
        )

    def _handle_tool_call(
        self,
        response: BackendResponse,
        messages: list[dict[str, Any]],
        include_search_tool: bool,
        turn_usage: dict[str, int],
    ) -> str:
        max_rounds = 3

        for round_num in range(max_rounds):
            tool_results: list[dict[str, Any]] = []
            assistant_tool_blocks: list[dict[str, Any]] = []
            for call in response.tool_calls:
                logger.info("Tool call [round %d]: %s(%s)", round_num + 1, call.name, call.input)
                if call.name != "web_search":
                    continue
                query = str(call.input.get("query", "")).strip()
                results = search(query)
                assistant_tool_blocks.append(
                    {
                        "type": "tool_use",
                        "id": call.id,
                        "name": call.name,
                        "input": call.input,
                    }
                )
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": call.id,
                        "content": format_results(results),
                    }
                )

            if not tool_results:
                break

            messages.append({"role": "assistant", "content": assistant_tool_blocks})
            messages.append({"role": "user", "content": tool_results})

            response = self.backend.chat(
                messages=messages,
                max_tokens=self.max_tokens,
                system=self._get_system_payload(),
                tools=[SEARCH_TOOL] if include_search_tool else None,
            )
            self._accumulate_usage(response, turn_usage)

            if response.stop_reason != "tool_use":
                break

        final_text = self._extract_text_blocks(response)
        return final_text if final_text else "(No response after search)"

    def ask(
        self,
        question: str,
        image: Image.Image | None = None,
        *,
        search_hint_question: str | None = None,
        force_search_tool: bool = False,
        ocr_text: str = "",
    ) -> str:
        self._ocr_active_for_turn = bool(ocr_text.strip())
        self.history.append(ChatMessage(role="user", text=question, image=image))
        self._trim_history()

        include_search_tool = (
            self.backend.supports_tools
            and (force_search_tool or self._should_include_search(search_hint_question or question))
        )
        turn_usage = {"input": 0, "output": 0, "cached": 0}

        try:
            messages = self._build_messages(include_images=self.backend.supports_vision)
            if image is not None and not self.backend.supports_vision:
                logger.info("Backend '%s' does not support vision; dropping image.", self.backend_name)

            logger.info(
                "Sending request (backend=%s, model=%s, max_tokens=%d, history=%d, tools=%s)",
                self.backend_name,
                self.model,
                self.max_tokens,
                len(self.history),
                "web_search" if include_search_tool else "none",
            )

            response = self.backend.chat(
                messages=messages,
                max_tokens=self.max_tokens,
                system=self._get_system_payload(),
                tools=[SEARCH_TOOL] if include_search_tool else None,
            )
            self._accumulate_usage(response, turn_usage)

            if response.stop_reason == "tool_use" and include_search_tool:
                answer = self._handle_tool_call(
                    response,
                    messages,
                    include_search_tool,
                    turn_usage,
                )
            else:
                answer = self._extract_text_blocks(response)

            self._finalize_usage(turn_usage)
            self.history.append(ChatMessage(role="assistant", text=answer))
            return answer
        except Exception:
            logger.exception("Assistant request failed")
            if self.history and self.history[-1].role == "user":
                self.history.pop()
            raise
        finally:
            self._ocr_active_for_turn = False

    def clear_history(self):
        self.history.clear()
        self._history_summary = ""

    def spawn_with_overrides(
        self,
        *,
        system_prompt: str | None = None,
        max_tokens: int | None = None,
    ) -> "AIAssistant":
        return AIAssistant(
            api_key=self._anthropic_api_key,
            openai_api_key=self._openai_api_key,
            backend=self.backend_name,
            model=self._configured_model,
            personality=self.personality,
            system_prompt=self.system_prompt if system_prompt is None else system_prompt,
            max_tokens=self.max_tokens if max_tokens is None else max_tokens,
            history_window_turns=self.history_window_turns,
            history_summary_every_turns=self.history_summary_every_turns,
            history_summary_max_chars=self.history_summary_max_chars,
            ollama_base_url=self._ollama_base_url,
            openai_base_url=self._openai_base_url,
            backend_timeout_sec=self._backend_timeout_sec,
        )
