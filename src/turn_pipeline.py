"""Turn assembly and execution pipeline."""

from __future__ import annotations

import difflib
import hashlib
import logging
import time
from dataclasses import dataclass
from typing import Any

from PIL import Image

from .app_detector import AppInfo
from .content_filter import build_context_prompt, filter_content
from .context_state import ContextControls
from .interaction_mode import AssistantTurnResult, ResponseMode
from .intent_router import classify_response_mode
from .log_events import log_event
from .ocr import DEFAULT_PREFERRED_APPS, extract_ocr_text, should_use_ocr
from .settings import AppSettings
from .screenshot import capture_window
from .url_browse import (
    DEFAULT_GLOBAL_TIMEOUT,
    DEFAULT_MAX_BYTES,
    DEFAULT_MAX_CHARS_PER_URL,
    DEFAULT_MAX_TOTAL_CHARS,
    DEFAULT_PER_URL_TIMEOUT,
    FetchedPage,
    extract_urls,
    fetch_public_page,
)

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ContextBlock:
    name: str
    text: str
    priority: int
    source: str
    required: bool = False
    allow_trim: bool = False
    score: float = 0.0


class TurnPipeline:
    def __init__(self, *, status_callback=None):
        self._status_callback = status_callback or (lambda _text: None)

    def submit(
        self,
        runtime,
        question: str,
        image: Image.Image | None,
        *,
        cancel_token=None,
        controls: ContextControls | None = None,
    ) -> AssistantTurnResult:
        settings = AppSettings.from_dict(runtime.cfg)
        controls = controls or ContextControls()
        now_mono = time.monotonic()
        self._evict_stale_cache_entries(runtime, settings=settings, now_mono=now_mono)
        runtime.turn_counter += 1

        with runtime.state_lock:
            hwnd = runtime.target_hwnd
            app = runtime.current_app
            clip_text = runtime.clipboard_context_text
            clip_pending = runtime.clipboard_context_pending
            if controls.clipboard and runtime.clipboard_context_pending and runtime.clipboard_context_text:
                runtime.clipboard_context_pending = False

        if not controls.screenshot:
            image = None
        elif image is None and hwnd:
            raw = capture_window(hwnd)
            if raw and app:
                image = filter_content(raw, app)

        app_type = app.app_type.value if app else ""
        response_mode = classify_response_mode(question=question, app_type=app_type, ai=runtime.ai)
        if self._is_cancelled(cancel_token):
            return AssistantTurnResult(text="Request cancelled.", response_mode=response_mode)

        mode_hint = "focus on actionable, task-oriented response"
        if response_mode == ResponseMode.CASUAL:
            mode_hint = "light conversational response"

        urls = extract_urls(question) if controls.urls else []
        browse_context = ""
        browse_warning = ""
        if urls:
            self._status_callback("Fetching links...")
            pages: list[FetchedPage] = []
            failures: list[str] = []
            deadline = time.monotonic() + DEFAULT_GLOBAL_TIMEOUT
            for url in urls:
                if self._is_cancelled(cancel_token):
                    return AssistantTurnResult(text="Request cancelled.", response_mode=response_mode)
                page = self._get_or_fetch_url_page(runtime, settings=settings, url=url, deadline=deadline)
                if not page.ok:
                    failures.append(f"{url} ({page.error})")
                    continue
                pages.append(page)

            if failures and not pages:
                msg = (
                    "I could not browse those links right now. "
                    "Please retry or send accessible public pages only.\n"
                    f"Failed links: {'; '.join(failures)}"
                )
                return AssistantTurnResult(text=msg, response_mode=response_mode)
            if failures:
                browse_warning = (
                    "[Direct URL browse note]\n"
                    f"Some links could not be fetched: {'; '.join(failures)}"
                )
            if pages:
                browse_context = self._build_cached_browse_context(runtime, settings=settings, pages=pages)

        clipboard_block = ""
        if controls.clipboard and clip_pending and clip_text:
            clipboard_block = f"[Clipboard context]\n{clip_text}"

        ocr_text = ""
        ocr_block = ""
        allow_ocr = (
            controls.ocr
            and image is not None
            and should_use_ocr(
                app_type=app_type,
                enabled=settings.enable_ocr_fallback,
                preferred_apps=settings.ocr_preferred_apps or list(DEFAULT_PREFERRED_APPS),
            )
        )
        if allow_ocr:
            if self._is_cancelled(cancel_token):
                return AssistantTurnResult(text="Request cancelled.", response_mode=response_mode)
            self._status_callback("Running OCR...")
            ocr_key = self._image_cache_key(image)
            ocr_text = self._get_cached_ocr_text(runtime, settings=settings, key=ocr_key, now_mono=time.monotonic())
            if not ocr_text:
                ocr_text = extract_ocr_text(
                    image,
                    max_chars=settings.ocr_max_chars,
                    timeout_sec=settings.ocr_timeout_sec,
                    tesseract_cmd=settings.tesseract_cmd,
                )
                self._set_cached_ocr_text(runtime, key=ocr_key, text=ocr_text, now_mono=time.monotonic())
            else:
                logger.info("OCR cache hit: key=%s chars=%d", ocr_key[:8], len(ocr_text))
            if self._is_cancelled(cancel_token):
                return AssistantTurnResult(text="Request cancelled.", response_mode=response_mode)
            if ocr_text:
                ocr_block = f"[Screen text (OCR)]\n{ocr_text}"

        app_context_block = ""
        if app:
            try:
                app_context_block = f"[{build_context_prompt(app)}]"
            except Exception:
                app_context_block = f"[Active app: {app_type or 'unknown'}]"
        browse_context_block = f"[Direct URL browse context]\n{browse_context}" if browse_context else ""
        browse_warning_block = browse_warning or ""
        static_payload = "\n\n".join(
            part
            for part in [
                app_context_block,
                clipboard_block,
                ocr_block,
                browse_context_block,
                browse_warning_block,
            ]
            if part
        )
        current_context_blocks = {
            "app": app_context_block,
            "clipboard": clipboard_block,
            "ocr": ocr_block,
            "browse_context": browse_context_block,
            "browse_warning": browse_warning_block,
        }
        static_hash = hashlib.sha1(static_payload.encode("utf-8")).hexdigest() if static_payload else ""
        use_context_reference = self._should_use_context_reference(runtime, settings=settings, static_hash=static_hash)
        if static_hash and not use_context_reference:
            runtime.static_context_hash = static_hash
            runtime.static_context_id = static_hash[:8]
            runtime.static_context_last_full_turn = runtime.turn_counter
        if not static_hash:
            runtime.static_context_hash = ""
            runtime.static_context_id = ""
            runtime.static_context_last_full_turn = 0

        blocks: list[ContextBlock] = []
        if use_context_reference:
            blocks.append(
                ContextBlock(
                    name="context_ref",
                    text=(
                        f"[Context reference id: {runtime.static_context_id}]\n"
                        "Reuse static context from previous turn in this session."
                    ),
                    priority=95,
                    source="context_ref",
                    required=True,
                )
            )
        else:
            if runtime.static_context_id:
                blocks.append(
                    ContextBlock(
                        name="context_id",
                        text=f"[Context id: {runtime.static_context_id}]",
                        priority=95,
                        source="context_ref",
                    )
                )
            blocks.extend(
                self._build_context_blocks(
                    runtime=runtime,
                    question=question,
                    current_blocks=current_context_blocks,
                )
            )
        blocks.append(
            ContextBlock(
                name="mode_hint",
                text=f"[Mode hint] {mode_hint}",
                priority=75,
                source="mode_hint",
                required=True,
            )
        )

        self._score_blocks(blocks, question=question)
        packed_blocks, dropped, trimmed = self._pack_context_blocks(
            blocks=blocks,
            question=question,
            max_chars=settings.context_max_chars,
        )
        full_question = "\n\n".join([text for _name, text in packed_blocks] + [question])
        estimated_context_tokens = self._estimate_tokens(full_question)
        output_cap = self._compute_output_cap(
            base_max=int(getattr(runtime.ai, "max_tokens", settings.max_tokens)),
            response_mode=response_mode,
            question=question,
            estimated_context_tokens=estimated_context_tokens,
            adaptive_enabled=settings.adaptive_output_caps,
        )
        self._log_context_preview(
            settings=settings,
            blocks=packed_blocks,
            question=question,
            dropped=dropped,
            output_cap=output_cap,
            trimmed=trimmed,
            controls=controls,
        )

        if self._is_cancelled(cancel_token):
            return AssistantTurnResult(text="Request cancelled.", response_mode=response_mode)
        if settings.show_token_cost:
            self._status_callback(f"Thinking... (~{estimated_context_tokens} ctx tok, cap {output_cap})")
        else:
            self._status_callback("Thinking...")
        answer = runtime.ai.ask(
            full_question,
            image=image,
            search_hint_question=question,
            ocr_text=ocr_text,
            max_tokens_override=output_cap,
        )
        runtime.previous_context_blocks = current_context_blocks
        return AssistantTurnResult(text=answer, response_mode=response_mode)

    def _should_use_context_reference(self, runtime, *, settings: AppSettings, static_hash: str) -> bool:
        refresh_turns = max(1, settings.context_reference_refresh_turns)
        return (
            bool(static_hash)
            and static_hash == runtime.static_context_hash
            and bool(runtime.static_context_id)
            and (runtime.turn_counter - runtime.static_context_last_full_turn) < refresh_turns
        )

    def _build_context_blocks(self, *, runtime, question: str, current_blocks: dict[str, str]) -> list[ContextBlock]:
        blocks: list[ContextBlock] = []
        previous_blocks = getattr(runtime, "previous_context_blocks", {}) or {}
        specs = [
            ("app", current_blocks.get("app", ""), 90, False, False),
            ("clipboard", current_blocks.get("clipboard", ""), 85, False, True),
            ("ocr", current_blocks.get("ocr", ""), 82, False, True),
            ("browse_warning", current_blocks.get("browse_warning", ""), 65, False, False),
            ("browse_context", current_blocks.get("browse_context", ""), 20, False, True),
        ]
        for name, text, priority, required, allow_trim in specs:
            if not text:
                continue
            reduced = self._reduce_with_delta(name=name, current=text, previous=previous_blocks.get(name, ""))
            blocks.append(
                ContextBlock(
                    name=name,
                    text=reduced,
                    priority=priority,
                    source=name,
                    required=required,
                    allow_trim=allow_trim,
                )
            )
        return blocks

    def _reduce_with_delta(self, *, name: str, current: str, previous: str) -> str:
        current = current.strip()
        previous = previous.strip()
        if not current:
            return ""
        if not previous:
            return current
        if current == previous:
            return f"[{name} reused]\nUnchanged from previous turn."

        ratio = difflib.SequenceMatcher(a=previous, b=current).ratio()
        if ratio < 0.55 or len(current) < 220:
            return current

        matcher = difflib.SequenceMatcher(a=previous, b=current)
        pieces: list[str] = []
        for tag, _i1, _i2, j1, j2 in matcher.get_opcodes():
            if tag in {"insert", "replace"}:
                chunk = current[j1:j2].strip()
                if chunk:
                    pieces.append(chunk)
        delta = "\n".join(piece for piece in pieces if piece).strip()
        if not delta:
            return f"[{name} reused]\nMostly unchanged from previous turn."
        candidate = (
            f"[{name} delta since last turn]\n"
            "Mostly unchanged from previous turn. Focus on these updates:\n"
            f"{delta[:700]}"
        )
        return candidate if len(candidate) < len(current) else current

    def _score_blocks(self, blocks: list[ContextBlock], *, question: str) -> None:
        question_terms = {term for term in question.lower().split() if len(term) > 3}
        for block in blocks:
            text_lower = block.text.lower()
            overlap = sum(1 for term in question_terms if term in text_lower)
            size_penalty = min(len(block.text) / 500.0, 12.0)
            short_bonus = 6.0 if len(block.text) < 600 else 0.0
            source_bonus = 0.0
            if block.source in {"app", "clipboard", "ocr"}:
                source_bonus += 4.0
            if block.required:
                source_bonus += 100.0
            block.score = float(block.priority) + (overlap * 8.0) + short_bonus + source_bonus - size_penalty

    def _pack_context_blocks(
        self,
        *,
        blocks: list[ContextBlock],
        question: str,
        max_chars: int,
    ) -> tuple[list[tuple[str, str]], list[dict[str, Any]], bool]:
        max_chars = max(1000, int(max_chars))
        remaining = max(max_chars - len(question), 0)
        dropped: list[dict[str, Any]] = []
        packed: list[tuple[str, str]] = []
        used = 0
        trimmed = False

        ordered = sorted(
            blocks,
            key=lambda block: (0 if block.required else 1, -block.score, -block.priority, block.name),
        )
        for block in ordered:
            text = block.text
            text_len = len(text)
            if (used + text_len) <= remaining:
                packed.append((block.name, text))
                used += text_len
                continue
            remain = remaining - used
            if remain <= 0:
                dropped.append(
                    {"name": block.name, "reason": "no_budget" if block.required else "budget_pruned"}
                )
                continue
            if block.allow_trim:
                packed.append((block.name, text[:remain]))
                used += remain
                trimmed = True
                continue
            dropped.append({"name": block.name, "reason": "no_budget" if block.required else "budget_pruned"})

        return packed, dropped, trimmed

    def _compute_output_cap(
        self,
        *,
        base_max: int,
        response_mode: ResponseMode,
        question: str,
        estimated_context_tokens: int,
        adaptive_enabled: bool,
    ) -> int:
        if not adaptive_enabled:
            return max(1, int(base_max))
        lowered = question.lower()
        if response_mode == ResponseMode.CASUAL:
            cap = min(base_max, 220)
        else:
            cap = min(base_max, 320)
        if any(
            phrase in lowered
            for phrase in ("explain", "walk me through", "step", "compare", "why", "detailed", "rewrite")
        ):
            cap = min(base_max, max(cap, 480))
        if any(phrase in lowered for phrase in ("summarize", "tl;dr", "just the answer", "one sentence")):
            cap = min(cap, 180)
        if estimated_context_tokens > 2500:
            cap = max(160, cap - 60)
        if len(question) > 800:
            cap = min(base_max, max(cap, 520))
        return max(120, int(cap))

    def _log_context_preview(
        self,
        *,
        settings: AppSettings,
        blocks: list[tuple[str, str]],
        question: str,
        dropped: list[dict[str, Any]],
        output_cap: int,
        trimmed: bool,
        controls: ContextControls,
    ) -> None:
        if not settings.context_telemetry:
            return
        token_by_block = {name: self._estimate_tokens(text) for name, text in blocks}
        token_by_block["question"] = self._estimate_tokens(question)
        total = sum(token_by_block.values())
        log_event(
            logger,
            "CONTEXT_PREVIEW",
            "submit",
            "prepared",
            est_tokens_total=total,
            output_cap=output_cap,
            blocks=",".join(f"{key}:{value}" for key, value in token_by_block.items()),
            dropped="none" if not dropped else ",".join(f"{item['name']}:{item['reason']}" for item in dropped),
            trimmed=str(trimmed).lower(),
            controls=(
                f"screenshot={int(controls.screenshot)},clipboard={int(controls.clipboard)},"
                f"urls={int(controls.urls)},ocr={int(controls.ocr)}"
            ),
        )

    def _build_cached_browse_context(
        self,
        runtime,
        *,
        settings: AppSettings,
        pages: list[FetchedPage],
    ) -> str:
        separator = "\n\n---\n\n"
        max_total = DEFAULT_MAX_TOTAL_CHARS
        parts: list[str] = []
        total = 0
        for page in pages:
            if not page.ok:
                continue
            chunk = self._get_or_build_url_summary(runtime, settings=settings, page=page)
            prefix_len = len(separator) if parts else 0
            if total + prefix_len + len(chunk) > max_total:
                remain = max_total - total - prefix_len
                if remain <= 0:
                    break
                chunk = chunk[:remain]
            parts.append(chunk)
            total += prefix_len + len(chunk)
            if total >= max_total:
                break
        return separator.join(parts)

    def _get_or_build_url_summary(self, runtime, *, settings: AppSettings, page: FetchedPage) -> str:
        now_mono = time.monotonic()
        ttl = max(1, settings.url_cache_ttl_sec)
        cache = getattr(runtime, "url_summary_cache", {})
        fingerprint = hashlib.sha1(f"{page.title}\n{page.text}".encode("utf-8")).hexdigest()
        entry = cache.get(page.url)
        if entry is not None:
            ts, cached_fingerprint, chunk = entry
            if (now_mono - ts) <= ttl and cached_fingerprint == fingerprint:
                logger.info("URL summary cache hit: url=%s", page.url)
                return chunk

        chunk = (
            f"URL: {page.url}\n"
            f"Title: {page.title or '(no title)'}\n"
            f"Content: {page.text[:DEFAULT_MAX_CHARS_PER_URL]}"
        )
        cache[page.url] = (now_mono, fingerprint, chunk)
        runtime.url_summary_cache = cache
        return chunk

    def _get_or_fetch_url_page(
        self,
        runtime,
        *,
        settings: AppSettings,
        url: str,
        deadline: float,
    ) -> FetchedPage:
        cached = self._get_cached_url_page(runtime, settings=settings, url=url, now_mono=time.monotonic())
        if cached is not None:
            logger.info("URL browse cache hit: url=%s ok=%s", url, cached.ok)
            return cached

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return FetchedPage(url=url, ok=False, error="global_timeout")
        timeout_sec = min(DEFAULT_PER_URL_TIMEOUT, remaining)
        logger.info("URL browse start: url=%s timeout=%.2fs", url, timeout_sec)
        page = fetch_public_page(
            url=url,
            timeout_sec=timeout_sec,
            max_bytes=DEFAULT_MAX_BYTES,
            allow_private=settings.allow_private_url_browse,
        )
        self._set_cached_url_page(runtime, url=url, page=page, now_mono=time.monotonic())
        return page

    def _evict_stale_cache_entries(self, runtime, *, settings: AppSettings, now_mono: float) -> None:
        url_ttl = max(1, settings.url_cache_ttl_sec)
        ocr_ttl = max(1, settings.ocr_cache_ttl_sec)
        runtime.url_cache = {
            key: value for key, value in runtime.url_cache.items() if (now_mono - value[0]) <= url_ttl
        }
        runtime.ocr_cache = {
            key: value for key, value in runtime.ocr_cache.items() if (now_mono - value[0]) <= ocr_ttl
        }
        runtime.url_summary_cache = {
            key: value
            for key, value in getattr(runtime, "url_summary_cache", {}).items()
            if (now_mono - value[0]) <= url_ttl
        }

    def _get_cached_url_page(self, runtime, *, settings: AppSettings, url: str, now_mono: float) -> FetchedPage | None:
        entry = runtime.url_cache.get(url)
        if entry is None:
            return None
        ts, page = entry
        if (now_mono - ts) > max(1, settings.url_cache_ttl_sec):
            return None
        return page

    def _set_cached_url_page(self, runtime, *, url: str, page: FetchedPage, now_mono: float) -> None:
        runtime.url_cache[url] = (now_mono, page)

    def _image_cache_key(self, image: Image.Image | None) -> str:
        if image is None or not hasattr(image, "convert"):
            return ""
        gray = image.convert("L")
        small = gray.resize((128, 128))
        return hashlib.blake2b(small.tobytes(), digest_size=16).hexdigest()

    def _get_cached_ocr_text(self, runtime, *, settings: AppSettings, key: str, now_mono: float) -> str:
        if not key:
            return ""
        entry = runtime.ocr_cache.get(key)
        if entry is None:
            return ""
        ts, text = entry
        if (now_mono - ts) > max(1, settings.ocr_cache_ttl_sec):
            return ""
        return text

    def _set_cached_ocr_text(self, runtime, *, key: str, text: str, now_mono: float) -> None:
        if key and text:
            runtime.ocr_cache[key] = (now_mono, text)

    def _is_cancelled(self, cancel_token: Any) -> bool:
        return bool(cancel_token is not None and hasattr(cancel_token, "is_set") and cancel_token.is_set())

    def _estimate_tokens(self, text: str) -> int:
        return max(1, len(text) // 4) if text else 0
