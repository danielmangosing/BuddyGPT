"""Model backend abstraction for Anthropic, OpenAI, and Ollama."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import anthropic

logger = logging.getLogger(__name__)

DEFAULT_BACKEND = "anthropic"
DEFAULT_MODELS = {
    "anthropic": "claude-sonnet-4-20250514",
    "openai": "gpt-4o-mini",
    "ollama": "llava:13b",
}

OPENAI_BASE_URL = "https://api.openai.com/v1"
OLLAMA_BASE_URL = "http://127.0.0.1:11434"


@dataclass(slots=True)
class ToolCall:
    id: str
    name: str
    input: dict[str, Any]


@dataclass(slots=True)
class BackendResponse:
    text: str
    stop_reason: str
    tool_calls: list[ToolCall]
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0


class ModelBackend:
    supports_tools: bool = False
    supports_vision: bool = False
    backend_name: str = ""

    def chat(
        self,
        *,
        messages: list[dict[str, Any]],
        system: list[dict[str, Any]] | str,
        max_tokens: int,
        tools: list[dict[str, Any]] | None = None,
    ) -> BackendResponse:
        raise NotImplementedError

    def validate(self) -> tuple[bool, str]:
        raise NotImplementedError


def _normalize_backend_name(name: str | None) -> str:
    value = (name or DEFAULT_BACKEND).strip().lower()
    if value in {"anthropic", "openai", "ollama"}:
        return value
    logger.warning("Unknown backend '%s'; falling back to anthropic", name)
    return DEFAULT_BACKEND


def _is_model_compatible(backend: str, model: str) -> bool:
    model_l = (model or "").strip().lower()
    if not model_l:
        return False
    if backend == "anthropic":
        return model_l.startswith("claude-")
    if backend == "openai":
        return model_l.startswith(("gpt-", "o1", "o3", "o4", "chatgpt-"))
    if backend == "ollama":
        # Most Ollama model tags look like name:tag.
        return not model_l.startswith(("claude-", "gpt-", "o1", "o3", "o4"))
    return False


def resolve_model_for_backend(backend: str, configured_model: str) -> str:
    if _is_model_compatible(backend, configured_model):
        return configured_model
    fallback = DEFAULT_MODELS[backend]
    logger.warning(
        "Model '%s' does not look compatible with backend '%s'; using '%s'",
        configured_model,
        backend,
        fallback,
    )
    return fallback


def _coerce_system_text(system: list[dict[str, Any]] | str) -> str:
    if isinstance(system, str):
        return system
    parts: list[str] = []
    for block in system:
        text = str(block.get("text", "")).strip()
        if text:
            parts.append(text)
    return "\n".join(parts).strip()


def _extract_text_blocks(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    text = str(block.get("text", "")).strip()
                    if text:
                        parts.append(text)
                elif "text" in block:
                    text = str(block.get("text", "")).strip()
                    if text:
                        parts.append(text)
        return "\n".join(parts).strip()
    return str(content or "").strip()


class AnthropicBackend(ModelBackend):
    backend_name = "anthropic"
    supports_tools = True
    supports_vision = True

    def __init__(self, api_key: str | None, model: str):
        self.model = model
        self.client = anthropic.Anthropic(api_key=api_key)

    def _to_anthropic_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        converted: list[dict[str, Any]] = []
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "assistant" and isinstance(content, str):
                converted.append({"role": "assistant", "content": content})
                continue
            if role == "assistant" and isinstance(content, list):
                blocks: list[dict[str, Any]] = []
                for block in content:
                    block_type = block.get("type")
                    if block_type == "text":
                        blocks.append({"type": "text", "text": block.get("text", "")})
                    elif block_type == "tool_use":
                        blocks.append(
                            {
                                "type": "tool_use",
                                "id": block.get("id", ""),
                                "name": block.get("name", ""),
                                "input": block.get("input", {}),
                            }
                        )
                converted.append({"role": "assistant", "content": blocks})
                continue

            if role != "user":
                continue
            if isinstance(content, str):
                converted.append({"role": "user", "content": [{"type": "text", "text": content}]})
                continue

            blocks: list[dict[str, Any]] = []
            for block in content:
                block_type = block.get("type")
                if block_type == "text":
                    blocks.append({"type": "text", "text": block.get("text", "")})
                elif block_type == "image":
                    source = block.get("source", {})
                    media_type = source.get("media_type") or block.get("media_type") or "image/jpeg"
                    data = source.get("data") or block.get("data", "")
                    blocks.append(
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": data,
                            },
                        }
                    )
                elif block_type == "tool_result":
                    blocks.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.get("tool_use_id", ""),
                            "content": block.get("content", ""),
                        }
                    )
            converted.append({"role": "user", "content": blocks})
        return converted

    def chat(
        self,
        *,
        messages: list[dict[str, Any]],
        system: list[dict[str, Any]] | str,
        max_tokens: int,
        tools: list[dict[str, Any]] | None = None,
    ) -> BackendResponse:
        tools_arg = tools if tools else anthropic.NOT_GIVEN
        system_arg = system if isinstance(system, list) else [{"type": "text", "text": system}]
        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system_arg,
            tools=tools_arg,
            messages=self._to_anthropic_messages(messages),
        )

        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        for block in getattr(response, "content", []):
            block_type = getattr(block, "type", "")
            if block_type == "text":
                block_text = getattr(block, "text", "")
                if block_text:
                    text_parts.append(block_text)
            elif block_type == "tool_use":
                tool_calls.append(
                    ToolCall(
                        id=getattr(block, "id", ""),
                        name=getattr(block, "name", ""),
                        input=getattr(block, "input", {}) or {},
                    )
                )

        usage = getattr(response, "usage", None)
        input_tokens = int(getattr(usage, "input_tokens", 0) or 0)
        output_tokens = int(getattr(usage, "output_tokens", 0) or 0)
        cached_tokens = int(getattr(usage, "cache_read_input_tokens", 0) or 0)

        return BackendResponse(
            text="\n".join(text_parts).strip(),
            stop_reason=str(getattr(response, "stop_reason", "end_turn") or "end_turn"),
            tool_calls=tool_calls,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_tokens=cached_tokens,
        )

    def validate(self) -> tuple[bool, str]:
        try:
            self.client.messages.create(
                model=self.model,
                max_tokens=1,
                messages=[{"role": "user", "content": "ping"}],
            )
            return True, ""
        except anthropic.AuthenticationError:
            return False, "Invalid Anthropic API key."
        except anthropic.PermissionDeniedError as exc:
            return False, f"Permission denied: {exc}"
        except anthropic.APIError as exc:
            return False, f"API error: {exc}"
        except Exception as exc:
            return False, f"Unexpected error while validating key: {exc}"


class OpenAIBackend(ModelBackend):
    backend_name = "openai"
    supports_tools = False
    supports_vision = True

    def __init__(
        self,
        api_key: str | None,
        model: str,
        *,
        base_url: str = OPENAI_BASE_URL,
        timeout_sec: int = 45,
    ):
        self.api_key = api_key or ""
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout_sec = max(1, int(timeout_sec))

    def _request(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        if not self.api_key:
            raise RuntimeError("Missing OpenAI API key.")
        url = f"{self.base_url}{path}"
        body = json.dumps(payload).encode("utf-8")
        req = Request(
            url,
            data=body,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        )
        with urlopen(req, timeout=self.timeout_sec) as resp:
            data = resp.read()
        return json.loads(data.decode("utf-8", errors="replace"))

    def _to_openai_messages(
        self,
        messages: list[dict[str, Any]],
        system: list[dict[str, Any]] | str,
    ) -> list[dict[str, Any]]:
        converted: list[dict[str, Any]] = []
        system_text = _coerce_system_text(system)
        if system_text:
            converted.append({"role": "system", "content": system_text})

        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "assistant":
                converted.append({"role": "assistant", "content": _extract_text_blocks(content)})
                continue
            if role != "user":
                continue

            if isinstance(content, str):
                converted.append({"role": "user", "content": content})
                continue

            blocks: list[dict[str, Any]] = []
            for block in content:
                block_type = block.get("type")
                if block_type == "text":
                    blocks.append({"type": "text", "text": block.get("text", "")})
                elif block_type == "image":
                    source = block.get("source", {})
                    media_type = source.get("media_type", "image/jpeg")
                    data = source.get("data", "")
                    data_url = f"data:{media_type};base64,{data}"
                    blocks.append({"type": "image_url", "image_url": {"url": data_url}})
                elif block_type == "tool_result":
                    blocks.append({"type": "text", "text": str(block.get("content", ""))})

            if not blocks:
                converted.append({"role": "user", "content": ""})
            elif len(blocks) == 1 and blocks[0]["type"] == "text":
                converted.append({"role": "user", "content": blocks[0]["text"]})
            else:
                converted.append({"role": "user", "content": blocks})
        return converted

    def chat(
        self,
        *,
        messages: list[dict[str, Any]],
        system: list[dict[str, Any]] | str,
        max_tokens: int,
        tools: list[dict[str, Any]] | None = None,
    ) -> BackendResponse:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": self._to_openai_messages(messages, system),
            "max_tokens": max_tokens,
            "temperature": 0.2,
        }
        if tools:
            logger.info("OpenAI backend currently ignores tool definitions.")

        data = self._request("/chat/completions", payload)
        choices = data.get("choices", [])
        if not choices:
            return BackendResponse(text="", stop_reason="end_turn", tool_calls=[])

        first = choices[0]
        message = first.get("message", {})
        content = message.get("content", "")
        text = _extract_text_blocks(content)
        usage = data.get("usage", {})
        cached_tokens = int(
            (
                usage.get("prompt_tokens_details", {}) or {}
            ).get("cached_tokens", 0)
            or 0
        )
        return BackendResponse(
            text=text,
            stop_reason=str(first.get("finish_reason") or "end_turn"),
            tool_calls=[],
            input_tokens=int(usage.get("prompt_tokens", 0) or 0),
            output_tokens=int(usage.get("completion_tokens", 0) or 0),
            cached_tokens=cached_tokens,
        )

    def validate(self) -> tuple[bool, str]:
        if not self.api_key:
            return False, "OpenAI API key is required."
        try:
            self.chat(messages=[{"role": "user", "content": "ping"}], system="", max_tokens=1)
            return True, ""
        except HTTPError as exc:
            if exc.code in {401, 403}:
                return False, "Invalid OpenAI API key."
            return False, f"OpenAI HTTP error: {exc.code}"
        except URLError as exc:
            return False, f"OpenAI connection error: {exc.reason}"
        except Exception as exc:
            return False, f"Unexpected OpenAI validation error: {exc}"


class OllamaBackend(ModelBackend):
    backend_name = "ollama"
    supports_tools = False
    supports_vision = True

    def __init__(
        self,
        model: str,
        *,
        base_url: str = OLLAMA_BASE_URL,
        timeout_sec: int = 45,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout_sec = max(1, int(timeout_sec))

    def _request(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        body = json.dumps(payload).encode("utf-8")
        req = Request(
            url,
            data=body,
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        with urlopen(req, timeout=self.timeout_sec) as resp:
            data = resp.read()
        return json.loads(data.decode("utf-8", errors="replace"))

    def _to_ollama_messages(self, messages: list[dict[str, Any]], system: list[dict[str, Any]] | str):
        converted: list[dict[str, Any]] = []
        system_text = _coerce_system_text(system)
        if system_text:
            converted.append({"role": "system", "content": system_text})

        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "assistant":
                converted.append({"role": "assistant", "content": _extract_text_blocks(content)})
                continue
            if role != "user":
                continue

            if isinstance(content, str):
                converted.append({"role": "user", "content": content})
                continue

            text_parts: list[str] = []
            images: list[str] = []
            for block in content:
                block_type = block.get("type")
                if block_type == "text":
                    text = str(block.get("text", "")).strip()
                    if text:
                        text_parts.append(text)
                elif block_type == "tool_result":
                    text = str(block.get("content", "")).strip()
                    if text:
                        text_parts.append(text)
                elif block_type == "image":
                    source = block.get("source", {})
                    data = source.get("data") or block.get("data")
                    if data:
                        images.append(data)
            item: dict[str, Any] = {
                "role": "user",
                "content": "\n".join(text_parts).strip(),
            }
            if images:
                item["images"] = images
            converted.append(item)

        return converted

    def chat(
        self,
        *,
        messages: list[dict[str, Any]],
        system: list[dict[str, Any]] | str,
        max_tokens: int,
        tools: list[dict[str, Any]] | None = None,
    ) -> BackendResponse:
        if tools:
            logger.info("Ollama backend currently ignores tool definitions.")
        payload = {
            "model": self.model,
            "messages": self._to_ollama_messages(messages, system),
            "stream": False,
            "options": {"num_predict": max_tokens},
        }
        data = self._request("/api/chat", payload)
        message = data.get("message", {})
        text = str(message.get("content", "")).strip()
        return BackendResponse(
            text=text,
            stop_reason=str(data.get("done_reason", "end_turn") or "end_turn"),
            tool_calls=[],
            input_tokens=int(data.get("prompt_eval_count", 0) or 0),
            output_tokens=int(data.get("eval_count", 0) or 0),
            cached_tokens=0,
        )

    def validate(self) -> tuple[bool, str]:
        try:
            # /api/tags gives fast connectivity + model list check.
            url = f"{self.base_url}/api/tags"
            req = Request(url, method="GET")
            with urlopen(req, timeout=self.timeout_sec) as resp:
                data = json.loads(resp.read().decode("utf-8", errors="replace"))
            models = data.get("models", []) or []
            available = {str(item.get("name", "")) for item in models if isinstance(item, dict)}
            if available and self.model not in available:
                return (
                    False,
                    f"Ollama reachable, but model '{self.model}' is not installed.",
                )
            return True, ""
        except HTTPError as exc:
            return False, f"Ollama HTTP error: {exc.code}"
        except URLError as exc:
            return False, f"Ollama connection error: {exc.reason}"
        except Exception as exc:
            return False, f"Unexpected Ollama validation error: {exc}"


def build_backend(
    *,
    backend_name: str,
    model: str,
    anthropic_api_key: str | None,
    openai_api_key: str | None,
    ollama_base_url: str,
    openai_base_url: str,
    timeout_sec: int,
) -> ModelBackend:
    backend = _normalize_backend_name(backend_name)
    resolved_model = resolve_model_for_backend(backend, model)

    if backend == "anthropic":
        return AnthropicBackend(api_key=anthropic_api_key, model=resolved_model)
    if backend == "openai":
        return OpenAIBackend(
            api_key=openai_api_key,
            model=resolved_model,
            base_url=openai_base_url,
            timeout_sec=timeout_sec,
        )
    return OllamaBackend(
        model=resolved_model,
        base_url=ollama_base_url,
        timeout_sec=timeout_sec,
    )
