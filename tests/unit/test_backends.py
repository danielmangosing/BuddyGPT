"""Unit tests for backend model helpers and HTTP adapter behavior."""

import json

from src.backends import OllamaBackend, OpenAIBackend, resolve_model_for_backend


class _FakeResponse:
    def __init__(self, payload: dict):
        self._data = json.dumps(payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self):
        return self._data


def test_resolve_model_keeps_compatible_anthropic_model():
    assert (
        resolve_model_for_backend("anthropic", "claude-sonnet-4-20250514")
        == "claude-sonnet-4-20250514"
    )


def test_resolve_model_falls_back_for_openai_when_given_claude():
    assert resolve_model_for_backend("openai", "claude-sonnet-4-20250514") == "gpt-4o-mini"


def test_resolve_model_falls_back_for_ollama_when_given_openai():
    assert resolve_model_for_backend("ollama", "gpt-4o-mini") == "llava:13b"


def test_openai_backend_chat_parses_response(monkeypatch):
    captured = {}

    def _fake_urlopen(req, timeout):
        captured["url"] = req.full_url
        captured["timeout"] = timeout
        captured["payload"] = json.loads(req.data.decode("utf-8"))
        return _FakeResponse(
            {
                "choices": [
                    {
                        "message": {"content": "hello from openai"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 11, "completion_tokens": 7},
            }
        )

    monkeypatch.setattr("src.backends.urlopen", _fake_urlopen)
    backend = OpenAIBackend(
        api_key="sk-openai-test",
        model="gpt-4o-mini",
        base_url="https://api.openai.com/v1",
        timeout_sec=9,
    )
    result = backend.chat(
        messages=[{"role": "user", "content": [{"type": "text", "text": "ping"}]}],
        system="sys",
        max_tokens=12,
    )
    assert captured["url"].endswith("/chat/completions")
    assert captured["timeout"] == 9
    assert captured["payload"]["model"] == "gpt-4o-mini"
    assert result.text == "hello from openai"
    assert result.input_tokens == 11
    assert result.output_tokens == 7


def test_ollama_backend_chat_parses_response(monkeypatch):
    captured = {}

    def _fake_urlopen(req, timeout):
        captured["url"] = req.full_url
        captured["timeout"] = timeout
        captured["payload"] = json.loads(req.data.decode("utf-8"))
        return _FakeResponse(
            {
                "message": {"content": "hello from ollama"},
                "done_reason": "stop",
                "prompt_eval_count": 20,
                "eval_count": 9,
            }
        )

    monkeypatch.setattr("src.backends.urlopen", _fake_urlopen)
    backend = OllamaBackend(model="llava:13b", base_url="http://127.0.0.1:11434", timeout_sec=7)
    result = backend.chat(
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "ping"},
                    {
                        "type": "image",
                        "source": {"type": "base64", "media_type": "image/jpeg", "data": "abc123"},
                    },
                ],
            }
        ],
        system="sys",
        max_tokens=15,
    )
    assert captured["url"].endswith("/api/chat")
    assert captured["timeout"] == 7
    sent = captured["payload"]["messages"][1]
    assert sent["role"] == "user"
    assert sent["images"] == ["abc123"]
    assert result.text == "hello from ollama"
    assert result.input_tokens == 20
    assert result.output_tokens == 9
