"""Desktop pet overlay with animated Shiba and chat UI."""

import ctypes
import threading
import tkinter as tk
import tkinter.font as tkfont

from PIL import Image, ImageTk

from .context_state import ContextAvailability, ContextControls, RecoveryViewState
from .interaction_mode import AssistantTurnResult, ResponseMode
from .pet import Pet, PetState
from .sprites import SpriteManager

user32 = ctypes.windll.user32

CHROMA = "#00ff00"
CHROMA_RGB = (0, 255, 0)

BUBBLE_BG = "#E9E9EB"
BUBBLE_FG = "#000000"
INPUT_BG = "#FFFFFF"
INPUT_BORDER = "#C7C7CC"
ACCENT = "#007AFF"
STATUS_FG = "#FFFFFF"
STATUS_BG = "#8E8E93"
HINT_FG = "#8E8E93"

SPRITE_SIZE = 128
FRAME_MS = 160
AUTO_REST_MS = 15000
ALERT_DISMISS_MS = 30000
ALERT_DISMISS_PROACTIVE_MS = 15000
ALERT_DISMISS_MANUAL_MS = 45000
DRAG_THRESHOLD_PX = 6

WINDOW_W = 320
H_RESTING = 200
H_AWAKE = 260

BUBBLE_TAIL_H = 15
CORNER_R = 15
BUBBLE_PAD_TOP = 16
BUBBLE_PAD_BOTTOM = 16
BUBBLE_MIN_TEXT_H = 28
BUBBLE_MAX_TEXT_H = 280

INPUT_H = 34
INPUT_R = 17
INPUT_PAD_X = 28
CONTROL_ROW_H = 30

BASE_H = SPRITE_SIZE + 50
INPUT_AREA_H = INPUT_H + 16


def _create_rounded_rect(canvas, x1, y1, x2, y2, r, **kwargs):
    points = [
        x1 + r,
        y1,
        x2 - r,
        y1,
        x2,
        y1,
        x2,
        y1 + r,
        x2,
        y2 - r,
        x2,
        y2,
        x2 - r,
        y2,
        x1 + r,
        y2,
        x1,
        y2,
        x1,
        y2 - r,
        x1,
        y1 + r,
        x1,
        y1,
    ]
    return canvas.create_polygon(points, smooth=True, **kwargs)


def _resolve_response_mode(answer, chat_mode: bool) -> tuple[str, str, ResponseMode]:
    """Resolve reply event/text/mode from legacy str or AssistantTurnResult."""
    response_mode = ResponseMode.CASUAL if chat_mode else ResponseMode.WORK
    answer_text = answer if isinstance(answer, str) else str(answer)
    if isinstance(answer, AssistantTurnResult):
        answer_text = answer.text
        response_mode = answer.response_mode

    reply_event = "chat_answer" if response_mode == ResponseMode.CASUAL else "answer"
    return reply_event, answer_text, response_mode


class OverlayWindow:
    def __init__(
        self,
        on_submit,
        on_activate=None,
        on_dismiss=None,
        tray_mode: bool = False,
        show_token_cost: bool = False,
        usage_provider=None,
    ):
        self._on_submit = on_submit
        self._on_activate = on_activate
        self._on_dismiss = on_dismiss
        self._tray_mode = tray_mode
        self._show_token_cost = show_token_cost
        self._usage_provider = usage_provider

        self._root: tk.Tk | None = None
        self._hwnd: int = 0
        self._image: Image.Image | None = None
        self._window_title: str = "BuddyGPT"
        self._pet = Pet()
        self._sprites = SpriteManager(frame_size=SPRITE_SIZE, chroma=CHROMA_RGB)
        self._ready = threading.Event()
        self._drag_data = {"x": 0, "y": 0}
        self._drag_start_root = {"x": 0, "y": 0}
        self._drag_moved = False
        self._photo: ImageTk.PhotoImage | None = None
        self._bubble_total_h = 0
        self._idle_after_id: str | None = None
        self._alert_after_id: str | None = None
        self._chat_mode = False
        self._cancel_event = threading.Event()
        self._request_counter = 0
        self._active_request_id = 0
        self._last_answer_text = ""
        self._last_response_mode = ResponseMode.WORK
        self._alert_dismiss_ms = ALERT_DISMISS_MS
        self._context_availability = ContextAvailability()
        self._context_controls = ContextControls()

        self._pet.on_state_change(
            lambda old, new: self._root.after(0, self._on_pet_state_change)
            if self._root
            else None
        )

        self._tk_thread = threading.Thread(target=self._run_tk, daemon=True)
        self._tk_thread.start()
        self._ready.wait()

    @property
    def hwnd(self) -> int:
        return self._hwnd

    @property
    def pet_state(self) -> PetState:
        return self._pet.state

    @property
    def pet_state_name(self) -> str:
        return self._pet.state.value

    def can_show_proactive(self) -> bool:
        return self._pet.state == PetState.RESTING

    def _run_tk(self):
        self._root = tk.Tk()
        self._root.title("BuddyGPT")
        self._root.attributes("-topmost", True)
        self._root.overrideredirect(True)
        self._root.configure(bg=CHROMA)
        self._root.attributes("-transparentcolor", CHROMA)
        self._root.attributes("-alpha", 1.0)

        sw = self._root.winfo_screenwidth()
        sh = self._root.winfo_screenheight()
        self._root.geometry(
            f"{WINDOW_W}x{H_RESTING}+{sw - WINDOW_W - 20}+{sh - H_RESTING - 60}"
        )

        self._root.update_idletasks()
        self._hwnd = user32.GetParent(self._root.winfo_id()) or self._root.winfo_id()
        self._measure_font = tkfont.Font(family="Segoe UI", size=10)

        self._frame = tk.Frame(self._root, bg=CHROMA)
        self._frame.pack(fill=tk.BOTH, expand=True)

        canvas_w = WINDOW_W - 16
        self._canvas_w = canvas_w
        self._bubble_canvas = tk.Canvas(
            self._frame,
            bg=CHROMA,
            highlightthickness=0,
            width=canvas_w,
            height=0,
        )
        self._bubble_text = tk.Text(
            self._bubble_canvas,
            wrap=tk.WORD,
            font=("Segoe UI", 10),
            bg=BUBBLE_BG,
            fg=BUBBLE_FG,
            relief=tk.FLAT,
            height=1,
            width=30,
            state=tk.DISABLED,
            cursor="arrow",
            borderwidth=0,
            highlightthickness=0,
        )
        self._bubble_hint = tk.Label(
            self._bubble_canvas,
            text="",
            font=("Segoe UI", 8),
            fg=HINT_FG,
            bg=BUBBLE_BG,
            anchor=tk.E,
        )
        self._bubble_canvas.bind("<Button-1>", self._drag_start)
        self._bubble_canvas.bind("<B1-Motion>", self._drag_move)

        self._pet_label = tk.Label(
            self._frame,
            bg=CHROMA,
            cursor="hand2",
            width=SPRITE_SIZE,
            height=SPRITE_SIZE,
        )
        self._pet_label.pack(pady=(4, 0))
        self._pet_label.bind("<Button-1>", self._drag_start)
        self._pet_label.bind("<B1-Motion>", self._drag_move)
        self._pet_label.bind("<ButtonRelease-1>", self._on_pet_click)

        self._status_font = tkfont.Font(family="Segoe UI", size=9, weight="bold")
        self._status_canvas = tk.Canvas(
            self._frame,
            bg=CHROMA,
            highlightthickness=0,
            height=26,
        )
        self._status_canvas.pack(pady=(0, 4))
        self._status_canvas.bind("<Button-1>", self._drag_start)
        self._status_canvas.bind("<B1-Motion>", self._drag_move)
        self._update_status("zzZ")

        self._cost_label = tk.Label(
            self._frame,
            text="",
            font=("Segoe UI", 8),
            fg=HINT_FG,
            bg=CHROMA,
            anchor=tk.CENTER,
        )

        self._quick_action_row = tk.Frame(self._frame, bg=CHROMA)
        self._quick_simple_btn = tk.Button(
            self._quick_action_row,
            text="Explain simpler",
            font=("Segoe UI", 8),
            bg=BUBBLE_BG,
            fg=BUBBLE_FG,
            relief=tk.FLAT,
            borderwidth=1,
            command=self._quick_explain_simpler,
        )
        self._quick_steps_btn = tk.Button(
            self._quick_action_row,
            text="Give steps",
            font=("Segoe UI", 8),
            bg=BUBBLE_BG,
            fg=BUBBLE_FG,
            relief=tk.FLAT,
            borderwidth=1,
            command=self._quick_give_steps,
        )
        self._quick_copy_btn = tk.Button(
            self._quick_action_row,
            text="Copy answer",
            font=("Segoe UI", 8),
            bg=BUBBLE_BG,
            fg=BUBBLE_FG,
            relief=tk.FLAT,
            borderwidth=1,
            command=self._quick_copy_answer,
        )
        self._quick_simple_btn.pack(side=tk.LEFT, padx=3)
        self._quick_steps_btn.pack(side=tk.LEFT, padx=3)
        self._quick_copy_btn.pack(side=tk.LEFT, padx=3)

        self._context_row = tk.Frame(self._frame, bg=CHROMA)
        self._context_vars = {
            "screenshot": tk.BooleanVar(value=True),
            "clipboard": tk.BooleanVar(value=False),
            "urls": tk.BooleanVar(value=True),
            "ocr": tk.BooleanVar(value=False),
        }
        self._context_buttons: dict[str, tk.Checkbutton] = {}
        for key, label in (
            ("screenshot", "Use screenshot"),
            ("clipboard", "Use clipboard"),
            ("urls", "Use URLs"),
            ("ocr", "Use OCR"),
        ):
            btn = tk.Checkbutton(
                self._context_row,
                text=label,
                variable=self._context_vars[key],
                bg=CHROMA,
                fg=BUBBLE_FG,
                activebackground=CHROMA,
                activeforeground=BUBBLE_FG,
                selectcolor=BUBBLE_BG,
                relief=tk.FLAT,
                font=("Segoe UI", 8),
                borderwidth=0,
                highlightthickness=0,
                command=self._on_context_control_changed,
            )
            btn.pack(side=tk.LEFT, padx=2)
            self._context_buttons[key] = btn

        self._input_canvas = tk.Canvas(
            self._frame,
            bg=CHROMA,
            highlightthickness=0,
            width=canvas_w,
            height=INPUT_H,
        )

        pill_x1 = INPUT_PAD_X
        pill_x2 = canvas_w - INPUT_PAD_X
        pill_mid = (pill_x1 + pill_x2) // 2
        _create_rounded_rect(
            self._input_canvas,
            pill_x1,
            2,
            pill_x2,
            INPUT_H - 2,
            r=INPUT_R,
            fill=INPUT_BG,
            outline=INPUT_BORDER,
            width=1,
        )

        pill_inner_w = pill_x2 - pill_x1
        self._entry = tk.Entry(
            self._input_canvas,
            font=("Segoe UI", 9),
            bg=INPUT_BG,
            fg="#000000",
            insertbackground="#000000",
            relief=tk.FLAT,
            borderwidth=0,
            highlightthickness=0,
        )
        self._input_canvas.create_window(
            pill_mid - 14,
            INPUT_H // 2,
            window=self._entry,
            width=pill_inner_w - 62,
            height=INPUT_H - 12,
        )
        self._entry.bind("<Return>", self._on_enter)
        self._entry.bind("<KeyPress>", self._on_entry_activity)

        self._send_btn = tk.Button(
            self._input_canvas,
            text="\u2191",
            font=("Segoe UI", 9, "bold"),
            bg=ACCENT,
            fg="#FFFFFF",
            relief=tk.FLAT,
            activebackground="#005EC4",
            activeforeground="#FFFFFF",
            cursor="hand2",
            borderwidth=0,
            highlightthickness=0,
            command=lambda: self._on_enter(None),
        )
        self._input_canvas.create_window(
            pill_x2 - 20,
            INPUT_H // 2,
            window=self._send_btn,
            width=26,
            height=26,
        )

        self._root.bind("<Escape>", self._on_escape)

        self._sprites.pick_random("resting")
        self._tick()

        if self._tray_mode:
            self._root.withdraw()

        self._ready.set()
        self._root.mainloop()

    def _measure_text_height(self, text):
        available_w = self._canvas_w - 40
        line_h = self._measure_font.metrics("linespace")

        total_lines = 0
        for line in text.split("\n"):
            if not line:
                total_lines += 1
                continue
            line_px = self._measure_font.measure(line)
            total_lines += max(1, -(-line_px // available_w))

        return max(total_lines * line_h + 8, BUBBLE_MIN_TEXT_H)

    def _update_bubble(self, text, hint=""):
        canvas_w = self._canvas_w
        available_w = canvas_w - 40

        text_h = min(self._measure_text_height(text), BUBBLE_MAX_TEXT_H)
        hint_h = 22 if hint else 0
        bubble_h = BUBBLE_PAD_TOP + text_h + hint_h + BUBBLE_PAD_BOTTOM
        total_h = bubble_h + BUBBLE_TAIL_H

        self._bubble_canvas.config(height=total_h)
        self._bubble_canvas.delete("all")

        _create_rounded_rect(
            self._bubble_canvas,
            4,
            4,
            canvas_w - 4,
            bubble_h,
            r=CORNER_R,
            fill=BUBBLE_BG,
            outline="",
        )

        mid = canvas_w // 2
        self._bubble_canvas.create_polygon(
            mid - 10,
            bubble_h - 2,
            mid,
            bubble_h + BUBBLE_TAIL_H - 2,
            mid + 10,
            bubble_h - 2,
            fill=BUBBLE_BG,
            outline="",
            smooth=False,
        )

        self._bubble_text.config(state=tk.NORMAL)
        self._bubble_text.delete("1.0", tk.END)
        self._bubble_text.insert("1.0", text)
        self._bubble_text.config(state=tk.DISABLED)

        text_center_y = BUBBLE_PAD_TOP + text_h // 2 + 4
        self._bubble_canvas.create_window(
            canvas_w // 2,
            text_center_y,
            window=self._bubble_text,
            width=available_w,
            height=text_h,
        )

        if hint:
            self._bubble_hint.config(text=hint)
            self._bubble_canvas.create_window(
                canvas_w // 2,
                bubble_h - BUBBLE_PAD_BOTTOM // 2 - 2,
                window=self._bubble_hint,
                width=available_w,
            )

        if not self._bubble_canvas.winfo_ismapped():
            self._bubble_canvas.pack(fill=tk.X, padx=8, before=self._pet_label)

        self._bubble_total_h = total_h
        return total_h

    def _update_status(self, text):
        self._status_canvas.delete("all")
        text_w = self._status_font.measure(text)
        pill_w = text_w + 24
        pill_h = 22
        cx = self._canvas_w // 2
        x1 = cx - pill_w // 2
        x2 = cx + pill_w // 2
        y1 = 2
        y2 = y1 + pill_h
        _create_rounded_rect(
            self._status_canvas,
            x1,
            y1,
            x2,
            y2,
            r=pill_h // 2,
            fill=STATUS_BG,
            outline="",
        )
        self._status_canvas.create_text(
            cx,
            y1 + pill_h // 2,
            text=text,
            font=self._status_font,
            fill=STATUS_FG,
        )
        self._status_canvas.config(width=self._canvas_w)

    def _set_window_height(self, bubble_h=0, with_input=False):
        x = self._root.winfo_x()
        bottom = self._root.winfo_y() + self._root.winfo_height()

        new_h = BASE_H
        if bubble_h > 0:
            new_h += bubble_h + 8
        if with_input:
            new_h += INPUT_AREA_H
            if self._context_row.winfo_ismapped():
                new_h += CONTROL_ROW_H

        self._root.geometry(f"{WINDOW_W}x{new_h}+{x}+{bottom - new_h}")

    def _hide_cost_label(self):
        self._cost_label.pack_forget()

    def _update_cost_label(self):
        if not self._show_token_cost or not self._usage_provider:
            self._hide_cost_label()
            return
        usage = self._usage_provider()
        if usage is None:
            self._hide_cost_label()
            return
        self._cost_label.config(
            text=f"~${usage.estimated_cost_usd:.4f} - session: ${usage.session_total_usd:.4f}"
        )
        self._cost_label.pack(pady=(0, 2), before=self._input_canvas)

    def _tick(self):
        self._pet.tick()
        anim = self._pet.get_animation()
        frame = self._sprites.get_frame(anim.state.value, anim.frame_index)
        if frame:
            self._photo = ImageTk.PhotoImage(frame)
            self._pet_label.config(image=self._photo)
        self._root.after(FRAME_MS, self._tick)

    def _on_pet_state_change(self):
        state = self._pet.get_animation().state
        x = self._root.winfo_x()
        bottom = self._root.winfo_y() + self._root.winfo_height()
        self._sprites.pick_random(state.value)

        if state == PetState.RESTING:
            self._cancel_auto_rest()
            self._cancel_alert_dismiss()
            self._chat_mode = False
            self._hide_cost_label()
            self._hide_quick_actions()
            self._bubble_canvas.pack_forget()
            self._hide_context_controls()
            self._input_canvas.pack_forget()
            self._root.geometry(f"{WINDOW_W}x{H_RESTING}+{x}+{bottom - H_RESTING}")
            self._update_status("zzZ")
            return

        if state == PetState.ALERT:
            self._cancel_auto_rest()
            self._hide_cost_label()
            self._hide_quick_actions()
            self._hide_context_controls()
            self._input_canvas.pack_forget()
            self._schedule_alert_dismiss()
            return

        if state == PetState.GREETING:
            self._cancel_auto_rest()
            self._hide_cost_label()
            self._hide_quick_actions()
            self._show_context_controls()
            self._input_canvas.pack(pady=(4, 8))
            if self._bubble_total_h > 0:
                self._set_window_height(bubble_h=self._bubble_total_h, with_input=True)
            self._update_status("Daily chat")
            return

        if state == PetState.AWAKE:
            self._cancel_auto_rest()
            self._hide_cost_label()
            self._hide_quick_actions()
            self._bubble_canvas.pack_forget()
            self._show_context_controls()
            self._input_canvas.pack(pady=(4, 8))
            self._root.geometry(f"{WINDOW_W}x{H_AWAKE}+{x}+{bottom - H_AWAKE}")
            self._entry.delete(0, tk.END)
            self._entry.config(state=tk.NORMAL)
            self._send_btn.config(state=tk.NORMAL)
            self._update_status("Ask me anything!")
            self._root.after(100, lambda: self._entry.focus_set())
            return

        if state == PetState.THINKING:
            self._cancel_auto_rest()
            self._hide_cost_label()
            self._hide_quick_actions()
            self._hide_context_controls()
            self._input_canvas.pack_forget()
            bubble_h = self._update_bubble("Thinking... (Esc cancel)")
            self._set_window_height(bubble_h=bubble_h)
            self._update_status("Hmm...")
            return

        if state == PetState.REPLY:
            self._update_status("Ask more, or Esc to close")
            self._show_quick_actions()
            self._show_context_controls()
            return

        if state == PetState.IDLE_CHAT:
            self._cancel_auto_rest()
            self._show_context_controls()
            self._input_canvas.pack(pady=(4, 8))
            if self._bubble_total_h > 0:
                self._set_window_height(bubble_h=self._bubble_total_h, with_input=True)
            self._update_status("Keep chatting")

    def show(self, image: Image.Image | None = None, window_title: str = ""):
        self._image = image
        self._window_title = window_title
        if self._root:
            self._root.after(0, self._do_show)

    def show_notice(
        self,
        text: str,
        hint: str = "",
        status: str = "",
        pet_state: PetState | None = None,
    ):
        if self._root:
            self._root.after(150, lambda: self._do_show_notice(text, hint, status, pet_state))

    def update_thinking_status(self, text: str):
        """Update interim THINKING-phase status text."""
        if self._root:
            self._root.after(0, lambda: self._do_update_thinking_status(text))

    def _do_show(self):
        self._chat_mode = False
        self._cancel_event.clear()
        self._pet.trigger("activate")
        self._hide_cost_label()
        self._hide_quick_actions()
        self._bubble_text.config(state=tk.NORMAL)
        self._bubble_text.delete("1.0", tk.END)
        self._bubble_text.config(state=tk.DISABLED)
        self._set_context_availability(
            ContextAvailability(
                screenshot=self._image is not None,
                clipboard=False,
                urls=True,
                ocr=self._image is not None,
            )
        )
        self._root.deiconify()
        self._root.focus_force()

    def _do_show_notice(
        self,
        text: str,
        hint: str,
        status: str,
        pet_state: PetState | None,
    ):
        self._image = None
        self._root.deiconify()
        self._root.focus_force()
        self._hide_cost_label()
        self._hide_quick_actions()
        self._set_context_availability(
            ContextAvailability(
                screenshot=False,
                clipboard=False,
                urls=True,
                ocr=False,
            )
        )

        if self._pet.state == PetState.RESTING:
            if pet_state == PetState.GREETING:
                self._pet.trigger("greet")
            elif pet_state == PetState.ALERT:
                self._pet.trigger("alert")
            else:
                self._pet.trigger("activate")

        self._chat_mode = pet_state == PetState.GREETING
        bubble_h = self._update_bubble(text, hint=hint)
        self._input_canvas.pack(pady=(4, 8))
        self._entry.delete(0, tk.END)
        self._entry.config(state=tk.NORMAL)
        self._send_btn.config(state=tk.NORMAL)
        self._set_window_height(bubble_h=bubble_h, with_input=True)
        if status:
            self._update_status(status)
        self._root.after(100, lambda: self._entry.focus_set())

    def _do_update_thinking_status(self, text: str):
        if self._pet.state != PetState.THINKING:
            return
        bubble_h = self._update_bubble(text)
        self._set_window_height(bubble_h=bubble_h)
        self._update_status(text)

    def show_alert(
        self,
        title: str,
        body: str,
        hint: str = "Click to respond - Esc dismiss",
        priority: str = "normal",
    ):
        if self._root:
            self._root.after(0, lambda: self._do_show_alert(title, body, hint, priority))

    def _do_show_alert(self, title: str, body: str, hint: str, priority: str):
        if self._pet.state != PetState.RESTING:
            return
        self._alert_dismiss_ms = ALERT_DISMISS_MS
        if priority == "proactive":
            self._alert_dismiss_ms = ALERT_DISMISS_PROACTIVE_MS
        elif priority == "manual":
            self._alert_dismiss_ms = ALERT_DISMISS_MANUAL_MS
        bubble_h = self._update_bubble(body, hint=hint)
        self._input_canvas.pack_forget()
        self._set_window_height(bubble_h=bubble_h)
        self._update_status(title)
        self._hide_cost_label()
        self._root.deiconify()
        self._pet.trigger("alert")

    def _schedule_alert_dismiss(self):
        self._cancel_alert_dismiss()
        if self._root:
            self._alert_after_id = self._root.after(self._alert_dismiss_ms, self._dismiss)

    def _cancel_alert_dismiss(self):
        if self._root and self._alert_after_id:
            self._root.after_cancel(self._alert_after_id)
            self._alert_after_id = None

    def _dismiss(self):
        if self._on_dismiss:
            try:
                self._on_dismiss(self.get_view_state())
            except Exception:
                pass
        self._cancel_event.set()
        self._active_request_id += 1
        self._cancel_auto_rest()
        self._cancel_alert_dismiss()
        self._chat_mode = False
        self._pet.trigger("dismiss")
        self._hide_cost_label()
        self._hide_quick_actions()
        self._hide_context_controls()
        if self._tray_mode and self._root:
            self._root.after(300, self._root.withdraw)

    def _on_enter(self, event):
        question = self._entry.get().strip()
        if not question:
            return
        self._submit_question(question)

    def _submit_question(self, question: str):
        self._cancel_event.clear()
        self._request_counter += 1
        self._active_request_id = self._request_counter
        request_id = self._active_request_id
        controls = self.get_context_controls()

        self._cancel_auto_rest()
        self._entry.delete(0, tk.END)
        self._entry.config(state=tk.DISABLED)
        self._send_btn.config(state=tk.DISABLED)
        for btn in self._context_buttons.values():
            btn.config(state=tk.DISABLED)
        self._set_quick_actions_enabled(False)
        self._pet.trigger("submit")
        threading.Thread(
            target=self._ask_async,
            args=(question, request_id, controls),
            daemon=True,
        ).start()

    def _ask_async(self, question, request_id: int, controls: ContextControls):
        try:
            answer = self._on_submit(
                question,
                self._image,
                cancel_token=self._cancel_event,
                controls=controls,
            )
            self._image = None
            self._root.after(0, lambda: self._show_answer_if_current(request_id, answer))
        except Exception as exc:
            self._root.after(0, lambda: self._show_answer_if_current(request_id, f"Error: {exc}"))

    def _show_answer_if_current(self, request_id: int, answer):
        if request_id != self._active_request_id:
            return
        self._show_answer(answer)

    def _show_answer(self, answer):
        reply_event, answer_text, response_mode = _resolve_response_mode(answer, self._chat_mode)
        self._pet.trigger(reply_event)
        self._chat_mode = response_mode == ResponseMode.CASUAL
        self._last_answer_text = answer_text
        self._last_response_mode = response_mode
        self._set_quick_actions_enabled(True)

        hint = "Esc close - Enter follow-up"
        if response_mode == ResponseMode.CASUAL:
            hint = "Esc close - Enter continue chatting"
        bubble_h = self._update_bubble(answer_text, hint=hint)

        self._input_canvas.pack(pady=(4, 8))
        self._update_cost_label()
        self._entry.delete(0, tk.END)
        self._entry.config(state=tk.NORMAL)
        self._send_btn.config(state=tk.NORMAL)
        for key, btn in self._context_buttons.items():
            btn.config(
                state=tk.NORMAL
                if getattr(self._context_availability, key if key != "urls" else "urls")
                else tk.DISABLED
            )

        self._set_window_height(bubble_h=bubble_h, with_input=True)
        self._root.after(100, lambda: self._entry.focus_set())
        self._schedule_auto_rest()

    def _on_entry_activity(self, _event):
        self._cancel_auto_rest()

    def _on_escape(self, _event):
        if self._pet.state == PetState.THINKING:
            self._cancel_event.set()
            self._dismiss()
            return
        self._dismiss()

    def _schedule_auto_rest(self):
        self._cancel_auto_rest()
        if self._root:
            self._idle_after_id = self._root.after(AUTO_REST_MS, self._dismiss)

    def _cancel_auto_rest(self):
        if self._root and self._idle_after_id:
            self._root.after_cancel(self._idle_after_id)
            self._idle_after_id = None

    def _drag_start(self, event):
        self._drag_moved = False
        self._drag_start_root["x"] = event.x_root
        self._drag_start_root["y"] = event.y_root
        self._drag_data["x"] = event.x_root - self._root.winfo_x()
        self._drag_data["y"] = event.y_root - self._root.winfo_y()

    def _drag_move(self, event):
        dx = abs(event.x_root - self._drag_start_root["x"])
        dy = abs(event.y_root - self._drag_start_root["y"])
        if not self._drag_moved and dx < DRAG_THRESHOLD_PX and dy < DRAG_THRESHOLD_PX:
            return
        self._drag_moved = True
        x = event.x_root - self._drag_data["x"]
        y = event.y_root - self._drag_data["y"]
        self._root.geometry(f"+{x}+{y}")

    def _on_pet_click(self, _event):
        if self._drag_moved:
            return

        state = self._pet.get_animation().state
        if state == PetState.ALERT:
            self._cancel_alert_dismiss()
            self._pet.trigger("activate")
            return
        if state != PetState.RESTING:
            return

        if self._on_activate:
            threading.Thread(target=self._on_activate, daemon=True).start()
        else:
            self._do_show()

    def _show_quick_actions(self):
        if not self._quick_action_row.winfo_ismapped():
            self._quick_action_row.pack(pady=(0, 4), before=self._input_canvas)

    def _hide_quick_actions(self):
        self._quick_action_row.pack_forget()

    def _set_quick_actions_enabled(self, enabled: bool):
        state = tk.NORMAL if enabled else tk.DISABLED
        for btn in (self._quick_simple_btn, self._quick_steps_btn, self._quick_copy_btn):
            btn.config(state=state)

    def _quick_explain_simpler(self):
        if not self._last_answer_text:
            return
        prompt = (
            "Explain this more simply in plain language and keep it short:\n\n"
            f"{self._last_answer_text}"
        )
        self._submit_question(prompt)

    def _quick_give_steps(self):
        if not self._last_answer_text:
            return
        prompt = (
            "Convert this into short numbered steps I can follow:\n\n"
            f"{self._last_answer_text}"
        )
        self._submit_question(prompt)

    def _quick_copy_answer(self):
        if not self._last_answer_text or not self._root:
            return
        try:
            self._root.clipboard_clear()
            self._root.clipboard_append(self._last_answer_text)
            self._update_status("Copied")
        except Exception:
            self._update_status("Copy failed")

    def _show_context_controls(self):
        if any(
            (
                self._context_availability.screenshot,
                self._context_availability.clipboard,
                self._context_availability.urls,
                self._context_availability.ocr,
            )
        ):
            if not self._context_row.winfo_ismapped():
                self._context_row.pack(pady=(0, 4), before=self._input_canvas)

    def _hide_context_controls(self):
        self._context_row.pack_forget()

    def _on_context_control_changed(self):
        if self._bubble_total_h > 0 and self._input_canvas.winfo_ismapped():
            self._set_window_height(bubble_h=self._bubble_total_h, with_input=True)

    def _set_context_availability(
        self,
        availability: ContextAvailability,
        controls: ContextControls | None = None,
    ):
        self._context_availability = availability
        next_controls = controls or availability.default_controls()
        self._context_controls = next_controls
        values = {
            "screenshot": availability.screenshot and next_controls.screenshot,
            "clipboard": availability.clipboard and next_controls.clipboard,
            "urls": availability.urls and next_controls.urls,
            "ocr": availability.ocr and next_controls.ocr,
        }
        for key, value in values.items():
            self._context_vars[key].set(bool(value))
            self._context_buttons[key].config(
                state=tk.NORMAL if getattr(availability, key if key != "urls" else "urls") else tk.DISABLED
            )

    def set_context_availability(
        self,
        availability: ContextAvailability,
        controls: ContextControls | None = None,
    ):
        if self._root:
            self._root.after(0, lambda: self._set_context_availability(availability, controls))

    def get_context_controls(self) -> ContextControls:
        return ContextControls(
            screenshot=bool(self._context_vars["screenshot"].get()),
            clipboard=bool(self._context_vars["clipboard"].get()),
            urls=bool(self._context_vars["urls"].get()),
            ocr=bool(self._context_vars["ocr"].get()),
        )

    def get_view_state(self) -> RecoveryViewState:
        draft_text = self._entry.get().strip() if self._entry else ""
        answer_text = self._last_answer_text
        return RecoveryViewState(
            answer_text=answer_text,
            draft_text=draft_text,
            response_mode=self._last_response_mode,
            controls=self.get_context_controls(),
            availability=self._context_availability,
        )

    def restore_session(
        self,
        *,
        answer_text: str,
        draft_text: str,
        response_mode: ResponseMode,
        controls: ContextControls,
        availability: ContextAvailability,
        image=None,
        window_title: str = "BuddyGPT (Recovered)",
    ):
        if self._root:
            self._root.after(
                0,
                lambda: self._do_restore_session(
                    answer_text=answer_text,
                    draft_text=draft_text,
                    response_mode=response_mode,
                    controls=controls,
                    availability=availability,
                    image=image,
                    window_title=window_title,
                ),
            )

    def _do_restore_session(
        self,
        *,
        answer_text: str,
        draft_text: str,
        response_mode: ResponseMode,
        controls: ContextControls,
        availability: ContextAvailability,
        image,
        window_title: str,
    ):
        self._window_title = window_title
        self._image = image
        self._cancel_event.clear()
        self._root.deiconify()
        self._root.focus_force()
        self._hide_cost_label()
        self._set_context_availability(availability, controls)
        if self._pet.state != PetState.RESTING:
            self._pet.trigger("dismiss")
        self._pet.trigger("activate")
        self._pet.trigger("submit")
        self._pet.trigger("chat_answer" if response_mode == ResponseMode.CASUAL else "answer")
        self._chat_mode = response_mode == ResponseMode.CASUAL
        self._last_response_mode = response_mode
        self._last_answer_text = answer_text
        self._set_quick_actions_enabled(True)
        self._show_quick_actions()
        self._show_context_controls()
        hint = "Recovered session - Enter continue"
        bubble_h = self._update_bubble(answer_text, hint=hint)
        self._input_canvas.pack(pady=(4, 8))
        self._entry.config(state=tk.NORMAL)
        self._send_btn.config(state=tk.NORMAL)
        self._entry.delete(0, tk.END)
        if draft_text:
            self._entry.insert(0, draft_text)
        self._update_status("Recovered session")
        self._update_cost_label()
        self._set_window_height(bubble_h=bubble_h, with_input=True)
        self._root.after(100, lambda: self._entry.focus_set())
        self._schedule_auto_rest()
