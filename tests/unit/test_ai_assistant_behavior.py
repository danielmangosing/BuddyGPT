"""Unit tests for AIAssistant behavior changes."""

from __future__ import annotations

from types import SimpleNamespace

import anthropic
from PIL import Image

from src.ai_assistant import AIAssistant, ChatMessage, SEARCH_TOOL


class _Block:
    def __init__(self, text: str = "", block_type: str = "text"):
        self.type = block_type
        self.text = text


class _Response:
    def __init__(self, content, stop_reason: str = "end_turn"):
        self.content = content
        self.stop_reason = stop_reason
        self.usage = SimpleNamespace(input_tokens=10, output_tokens=5)


class _MessagesAPI:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self._responses.pop(0)


class _Client:
    def __init__(self, responses):
        self.messages = _MessagesAPI(responses)


def test_non_tool_response_joins_all_text_blocks():
    ai = AIAssistant(api_key="sk-test")
    ai.client = _Client(
        [
            _Response(
                [
                    _Block("First line"),
                    _Block("Second line"),
                ],
                stop_reason="end_turn",
            )
        ]
    )

    answer = ai.ask("Summarize this function.")
    assert answer == "First line\nSecond line"
    assert ai.client.messages.calls[0]["tools"] is anthropic.NOT_GIVEN


def test_search_tool_included_for_time_sensitive_question():
    ai = AIAssistant(api_key="sk-test")
    ai.client = _Client([_Response([_Block("Done")], stop_reason="end_turn")])

    ai.ask("What is the latest Python release today?")
    assert ai.client.messages.calls[0]["tools"] == [SEARCH_TOOL]


def test_search_tool_suppressed_for_screen_reference():
    ai = AIAssistant(api_key="sk-test")
    ai.client = _Client([_Response([_Block("Done")], stop_reason="end_turn")])

    ai.ask("What is this error on screen?")
    assert ai.client.messages.calls[0]["tools"] is anthropic.NOT_GIVEN


def test_history_trim_keeps_recent_turns_and_latest_image_turn():
    ai = AIAssistant(api_key="sk-test", history_window_turns=2)
    screenshot = Image.new("RGB", (4, 4))

    ai.history = [
        ChatMessage(role="user", text="old0"),
        ChatMessage(role="assistant", text="old0-reply"),
        ChatMessage(role="user", text="seed-image", image=screenshot),
        ChatMessage(role="assistant", text="seed-image-reply"),
        ChatMessage(role="user", text="recent1"),
        ChatMessage(role="assistant", text="recent1-reply"),
        ChatMessage(role="user", text="recent2"),
    ]

    ai._trim_history()

    assert [m.text for m in ai.history] == [
        "seed-image",
        "seed-image-reply",
        "recent1",
        "recent1-reply",
        "recent2",
    ]


def test_personality_sets_default_max_tokens():
    ai = AIAssistant(api_key="sk-test", personality="detailed")
    assert ai.max_tokens == 1500
    assert "knowledgeable technical partner" in ai.system_prompt


def test_explicit_max_tokens_overrides_personality_default():
    ai = AIAssistant(api_key="sk-test", personality="detailed", max_tokens=256)
    assert ai.max_tokens == 256


def test_history_summary_added_to_system_prompt_after_trim():
    ai = AIAssistant(
        api_key="sk-test",
        history_window_turns=1,
        history_summary_every_turns=1,
        history_summary_max_chars=600,
    )
    ai.history = [
        ChatMessage(role="user", text="old user 1"),
        ChatMessage(role="assistant", text="old assistant 1"),
        ChatMessage(role="user", text="old user 2"),
        ChatMessage(role="assistant", text="old assistant 2"),
        ChatMessage(role="user", text="latest"),
    ]

    ai._trim_history()
    prompt = ai._get_full_system_prompt()
    assert "Session summary" in prompt
    assert "old user" in prompt.lower()
