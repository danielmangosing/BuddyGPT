"""AI assistant with multi-backend support, vision, and optional web search."""

from __future__ import annotations

import base64
import copy
import io
import logging
import os
import re
from dataclasses import dataclass, field
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
_ARTIFACT_PATTERN = re.compile(r"(https?://\S+|[A-Za-z]:\\[^\s]+|[\w./\\-]+\.[A-Za-z0-9]{1,8})")

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


@dataclass
class PreferenceMemory:
    answer_style: str = ""
    language: str = ""
    notes: list[str] = field(default_factory=list)


@dataclass
class StructuredSessionMemory:
    goal: str = ""
    current_task: str = ""
    decisions: list[str] = field(default_factory=list)
    artifacts: list[str] = field(default_factory=list)
    open_questions: list[str] = field(default_factory=list)


@dataclass
class AssistantSessionState:
    history: list[ChatMessage]
    app_type: str
    history_summary: str
    summary_backlog: list[ChatMessage]
    summary_backlog_user_count: int
    preference_memory: PreferenceMemory
    structured_memory: StructuredSessionMemory
    session_cost: float
    last_usage: UsageStats | None


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
        search_cache_ttl_sec: int = 90,
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
        self.search_cache_ttl_sec = max(0, int(search_cache_ttl_sec))

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
        self._summary_backlog: list[ChatMessage] = []
        self._summary_backlog_user_count: int = 0
        self._preference_memory = PreferenceMemory()
        self._structured_memory = StructuredSessionMemory()

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
        preference_text = self._render_preference_memory()
        if preference_text:
            prompt += f"\n\n## User preferences\n{preference_text}"
        structured_memory = self._render_structured_memory()
        if structured_memory:
            prompt += f"\n\n## Working memory\n{structured_memory}"
        session_summary = self._get_session_summary_text()
        if session_summary:
            prompt += f"\n\n## Session summary\n{session_summary}"
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

    def _merge_history_summary(self, segment: str) -> None:
        if not segment:
            return
        if not self._history_summary:
            merged = segment
        else:
            merged = f"{self._history_summary} | {segment}"
        if len(merged) > self.history_summary_max_chars:
            merged = merged[-self.history_summary_max_chars :]
        self._history_summary = merged

    def _remember_preference_from_question(self, question: str) -> None:
        lowered = question.strip().lower()
        if not lowered:
            return
        if any(phrase in lowered for phrase in ("keep it short", "be concise", "brief answer", "terse")):
            self._preference_memory.answer_style = "concise"
        elif any(phrase in lowered for phrase in ("step by step", "give steps", "numbered steps")):
            self._preference_memory.answer_style = "steps"
        elif any(phrase in lowered for phrase in ("more detail", "be detailed", "go deeper")):
            self._preference_memory.answer_style = "detailed"
        elif any(phrase in lowered for phrase in ("plain language", "simple words", "explain simply")):
            self._preference_memory.answer_style = "plain"

        if any(phrase in lowered for phrase in ("in chinese", "reply in chinese", "中文")):
            self._preference_memory.language = "Chinese"
        elif any(phrase in lowered for phrase in ("in english", "reply in english", "英文")):
            self._preference_memory.language = "English"

        for phrase in ("just the answer", "no preamble", "show code", "use bullets"):
            if phrase in lowered and phrase not in self._preference_memory.notes:
                self._preference_memory.notes.append(phrase)
        self._preference_memory.notes = self._preference_memory.notes[-4:]

    def _append_memory_item(self, items: list[str], value: str, *, limit: int = 4) -> None:
        value = value.strip()
        if not value:
            return
        if value in items:
            items.remove(value)
        items.append(value)
        del items[:-limit]

    def _update_structured_memory(self, messages: list[ChatMessage]) -> None:
        for msg in messages:
            text = msg.text.strip().replace("\n", " ")
            if not text:
                continue
            compact = text[:180]
            lowered = compact.lower()
            if msg.role == "user":
                if not self._structured_memory.goal and any(
                    phrase in lowered for phrase in ("help me", "i need", "i want", "i'm trying", "can you")
                ):
                    self._structured_memory.goal = compact
                if any(
                    phrase in lowered
                    for phrase in ("working on", "trying to", "need to", "fix", "debug", "draft", "write")
                ):
                    self._structured_memory.current_task = compact
                if compact.endswith("?"):
                    self._append_memory_item(self._structured_memory.open_questions, compact, limit=5)
            else:
                if any(phrase in lowered for phrase in ("use ", "set ", "keep ", "try ", "next ", "recommend")):
                    self._append_memory_item(self._structured_memory.decisions, compact, limit=5)

            for match in _ARTIFACT_PATTERN.findall(compact):
                self._append_memory_item(self._structured_memory.artifacts, match, limit=6)

    def _render_preference_memory(self) -> str:
        lines: list[str] = []
        if self._preference_memory.answer_style:
            lines.append(f"- answer_style: {self._preference_memory.answer_style}")
        if self._preference_memory.language:
            lines.append(f"- preferred_language: {self._preference_memory.language}")
        for note in self._preference_memory.notes:
            lines.append(f"- note: {note}")
        return "\n".join(lines).strip()

    def _render_structured_memory(self) -> str:
        lines: list[str] = []
        if self._structured_memory.goal:
            lines.append(f"- goal: {self._structured_memory.goal}")
        if self._structured_memory.current_task:
            lines.append(f"- current_task: {self._structured_memory.current_task}")
        if self._structured_memory.decisions:
            lines.append(f"- decisions: {' | '.join(self._structured_memory.decisions[-3:])}")
        if self._structured_memory.artifacts:
            lines.append(f"- artifacts: {' | '.join(self._structured_memory.artifacts[-4:])}")
        if self._structured_memory.open_questions:
            lines.append(f"- open_questions: {' | '.join(self._structured_memory.open_questions[-3:])}")
        return "\n".join(lines).strip()

    def _get_session_summary_text(self) -> str:
        parts: list[str] = []
        if self._history_summary:
            parts.append(self._history_summary)
        if self._summary_backlog:
            pending = self._summarize_messages(self._summary_backlog)
            if pending:
                parts.append(pending)
        merged = " | ".join(parts).strip()
        if len(merged) > self.history_summary_max_chars:
            merged = merged[-self.history_summary_max_chars :]
        return merged

    def _update_history_summary(self, dropped_messages: list[ChatMessage]) -> None:
        if not dropped_messages:
            return
        user_count = sum(1 for m in dropped_messages if m.role == "user")
        if user_count <= 0:
            return
        self._update_structured_memory(dropped_messages)
        self._summary_backlog.extend(dropped_messages)
        self._summary_backlog_user_count += user_count
        # Keep newly trimmed turns visible immediately through the backlog summary,
        # but only merge them into the stable summary on the configured cadence.
        if self._summary_backlog_user_count < self.history_summary_every_turns:
            return

        segment = self._summarize_messages(self._summary_backlog)
        if segment:
            self._merge_history_summary(segment)
        self._summary_backlog = []
        self._summary_backlog_user_count = 0

    def _inject_text_context_into_latest_user_message(
        self,
        messages: list[dict[str, Any]],
        *context_blocks: str,
    ) -> None:
        extra_blocks = [block.strip() for block in context_blocks if block and block.strip()]
        if not extra_blocks:
            return

        for msg in reversed(messages):
            if msg.get("role") != "user":
                continue
            content = msg.get("content", "")
            if isinstance(content, str):
                msg["content"] = "\n\n".join(extra_blocks + [content])
                return
            if not isinstance(content, list):
                return

            insert_at = len(content)
            for idx in range(len(content) - 1, -1, -1):
                if content[idx].get("type") == "text":
                    insert_at = idx
                    break
            blocks = [{"type": "text", "text": block} for block in extra_blocks]
            content[insert_at:insert_at] = blocks
            return

    def _build_direct_search_context(self, query: str) -> str:
        query = query.strip()
        if not query:
            return ""
        results = search(query, cache_ttl_sec=self.search_cache_ttl_sec)
        logger.info("Direct search context: query='%s' results=%d", query, len(results))
        return f"[Web search results]\n{format_results(results)}"

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
                results = search(query, cache_ttl_sec=self.search_cache_ttl_sec)
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
        max_tokens_override: int | None = None,
    ) -> str:
        self._ocr_active_for_turn = bool(ocr_text.strip())
        self._remember_preference_from_question(question)
        self._update_structured_memory([ChatMessage(role="user", text=question)])
        self.history.append(ChatMessage(role="user", text=question, image=image))
        self._trim_history()

        search_requested = force_search_tool or self._should_include_search(search_hint_question or question)
        include_search_tool = self.backend.supports_tools and search_requested
        turn_usage = {"input": 0, "output": 0, "cached": 0}
        request_max_tokens = max(1, int(max_tokens_override or self.max_tokens))
        direct_search_augmentation = bool(
            getattr(getattr(self.backend, "capabilities", None), "direct_search_augmentation", True)
        )

        try:
            messages = self._build_messages(include_images=self.backend.supports_vision)
            if image is not None and not self.backend.supports_vision:
                logger.info("Backend '%s' does not support vision; dropping image.", self.backend_name)
            if search_requested and direct_search_augmentation:
                direct_search_context = self._build_direct_search_context(search_hint_question or question)
                self._inject_text_context_into_latest_user_message(messages, direct_search_context)

            logger.info(
                "Sending request (backend=%s, model=%s, max_tokens=%d, history=%d, tools=%s)",
                self.backend_name,
                self.model,
                request_max_tokens,
                len(self.history),
                "web_search" if include_search_tool else "none",
            )

            response = self.backend.chat(
                messages=messages,
                max_tokens=request_max_tokens,
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
        self._summary_backlog = []
        self._summary_backlog_user_count = 0
        self._preference_memory = PreferenceMemory()
        self._structured_memory = StructuredSessionMemory()

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
            search_cache_ttl_sec=self.search_cache_ttl_sec,
            ollama_base_url=self._ollama_base_url,
            openai_base_url=self._openai_base_url,
            backend_timeout_sec=self._backend_timeout_sec,
        )

    def snapshot_session_state(self) -> AssistantSessionState:
        history = [copy.deepcopy(msg) for msg in self.history]
        backlog = [copy.deepcopy(msg) for msg in self._summary_backlog]
        last_usage = copy.deepcopy(self._last_usage)
        return AssistantSessionState(
            history=history,
            app_type=self._app_type,
            history_summary=self._history_summary,
            summary_backlog=backlog,
            summary_backlog_user_count=self._summary_backlog_user_count,
            preference_memory=copy.deepcopy(self._preference_memory),
            structured_memory=copy.deepcopy(self._structured_memory),
            session_cost=self._session_cost,
            last_usage=last_usage,
        )

    def restore_session_state(self, state: AssistantSessionState) -> None:
        self.history = [copy.deepcopy(msg) for msg in state.history]
        self._app_type = state.app_type
        self._history_summary = state.history_summary
        self._summary_backlog = [copy.deepcopy(msg) for msg in state.summary_backlog]
        self._summary_backlog_user_count = state.summary_backlog_user_count
        self._preference_memory = copy.deepcopy(state.preference_memory)
        self._structured_memory = copy.deepcopy(state.structured_memory)
        self._session_cost = float(state.session_cost)
        self._last_usage = copy.deepcopy(state.last_usage)
