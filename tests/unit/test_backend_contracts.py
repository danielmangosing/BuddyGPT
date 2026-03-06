"""Contract tests for provider-level behavior parity."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime

from PIL import Image

from src.ai_assistant import AIAssistant, SEARCH_TOOL
from src.backends import BackendResponse
from src.intent_router import classify_response_mode
from src.interaction_mode import ResponseMode
from src.notifications.daily_chat import DAILY_NEWS_STATE_KEY, DailyChatSource, WAKE_FIRST_SLOT


@dataclass(frozen=True)
class ProviderCase:
    name: str
    supports_tools: bool


PROVIDER_CASES = [
    ProviderCase("anthropic", True),
    ProviderCase("openai", False),
    ProviderCase("ollama", False),
]


class _CaptureBackend:
    def __init__(self, case: ProviderCase, response_text: str):
        self.backend_name = case.name
        self.supports_tools = case.supports_tools
        self.supports_vision = True
        self.model = f"{case.name}-model"
        self._response_text = response_text
        self.calls: list[dict] = []

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        return BackendResponse(text=self._response_text, stop_reason="end_turn", tool_calls=[])

    def validate(self):
        return True, ""


def _build_ai(monkeypatch, case: ProviderCase, *, response_text: str) -> tuple[AIAssistant, list[_CaptureBackend]]:
    created: list[_CaptureBackend] = []

    def _fake_build_backend(**_kwargs):
        backend = _CaptureBackend(case, response_text=response_text)
        created.append(backend)
        return backend

    monkeypatch.setattr("src.ai_assistant.build_backend", _fake_build_backend)
    ai = AIAssistant(api_key="sk-test", backend=case.name, model="test-model")
    return ai, created


def _latest_user_text_blocks(call: dict) -> list[str]:
    messages = call["messages"]
    latest_user = next(msg for msg in reversed(messages) if msg["role"] == "user")
    content = latest_user["content"]
    if isinstance(content, str):
        return [content]
    return [str(block.get("text", "")) for block in content if block.get("type") == "text"]


def test_mode_routing_contract_uses_backend_fallback(monkeypatch):
    for case in PROVIDER_CASES:
        ai, backends = _build_ai(monkeypatch, case, response_text="casual")
        mode = classify_response_mode(question="ok", app_type="browser", ai=ai)

        assert mode == ResponseMode.CASUAL
        assert backends[0].calls
        assert backends[0].calls[0]["max_tokens"] == 8


def test_current_info_contract_is_provider_consistent(monkeypatch):
    monkeypatch.setattr(
        "src.ai_assistant.search",
        lambda query, cache_ttl_sec=90: [{"title": query, "url": "https://example.com", "snippet": "snippet"}],
    )
    monkeypatch.setattr("src.ai_assistant.format_results", lambda results: "SEARCH_RESULT_BLOCK")

    for case in PROVIDER_CASES:
        ai, backends = _build_ai(monkeypatch, case, response_text="answer")
        answer = ai.ask("What is the latest Python release today?")

        assert answer == "answer"
        first_call = backends[0].calls[0]
        if case.supports_tools:
            assert first_call["tools"] == [SEARCH_TOOL]
        else:
            assert first_call["tools"] is None
            text_blocks = _latest_user_text_blocks(first_call)
            assert any("[Web search results]" in block for block in text_blocks)
            assert any("SEARCH_RESULT_BLOCK" in block for block in text_blocks)


def test_vision_contract_passes_image_to_all_backends(monkeypatch):
    screenshot = Image.new("RGB", (8, 8), "white")

    for case in PROVIDER_CASES:
        ai, backends = _build_ai(monkeypatch, case, response_text="vision")
        ai.ask("Describe this image.", image=screenshot)

        text_blocks = backends[0].calls[0]["messages"][-1]["content"]
        assert any(block.get("type") == "image" for block in text_blocks)


def test_daily_chat_contract_keeps_current_info_behavior(monkeypatch, default_config, fake_state):
    monkeypatch.setattr(
        "src.ai_assistant.search",
        lambda query, cache_ttl_sec=90: [{"title": query, "url": "https://example.com", "snippet": "snippet"}],
    )
    monkeypatch.setattr("src.ai_assistant.format_results", lambda results: "SEARCH_RESULT_BLOCK")

    payload = json.dumps(
        {
            "topic_key": "provider-contract-topic",
            "message": "Provider contract daily chat opener.",
        }
    )

    for case in PROVIDER_CASES:
        fake_state.set(DAILY_NEWS_STATE_KEY, {})
        ai, backends = _build_ai(monkeypatch, case, response_text=payload)
        source = DailyChatSource(ai_assistant=ai, config=default_config)
        result = source.generate_for_slot(WAKE_FIRST_SLOT, datetime(2026, 3, 5, 9, 0), fake_state)

        assert result is not None
        temp_backend = backends[-1]
        first_call = temp_backend.calls[0]
        if case.supports_tools:
            assert first_call["tools"] == [SEARCH_TOOL]
        else:
            assert first_call["tools"] is None
            text_blocks = _latest_user_text_blocks(first_call)
            assert any("[Web search results]" in block for block in text_blocks)
            assert any("SEARCH_RESULT_BLOCK" in block for block in text_blocks)
