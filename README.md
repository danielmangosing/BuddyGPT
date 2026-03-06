# BuddyGPT

## What's New

- Added scheduled daily news pushes at **3:00 PM** and **8:00 PM** (local time).
- Kept the first wake-up daily news flow, so you now have up to 3 daily news slots:
  - first wake-up
  - 3:00 PM update
  - 8:00 PM update
- Enforced same-day topic dedupe across all slots:
  - if a generated topic repeats, BuddyGPT retries generation
  - if no unique topic is found, that slot is skipped (no duplicate push)
- Added "catch up when available today" behavior:
  - if a scheduled time is missed while you are busy in chat, BuddyGPT pushes it when it can safely show.

A tiny Shiba that lives in your screen corner and helps you unstuck.

![Real Shiba reference](assets/shiba/states/image0.jpg)

BuddyGPT is not a "do-it-for-me" agent. It is more like a friendly coworker who leans over, takes a quick look at your screen, and gives you a short, practical answer.

The original pain point was simple: I was asking ChatGPT a lot of practical questions, but each time I had to manually take screenshots or copy/paste email context before I could ask.

BuddyGPT removes that repetitive overhead. Instead of "screenshot -> copy context -> paste -> ask", you wake the dog and ask directly, so the whole flow feels smoother and more natural.

## What BuddyGPT Is

BuddyGPT is a desktop companion for lightweight help:
- "What does this email actually want?"
- "Why is this error happening?"
- "Is this page worth reading?"

It stays out of your way, then helps when you ask.

## Quick Install (Windows)

1. Download the latest installer: `BuddyGPT-Setup.exe` (from Releases).
2. Double-click the installer and complete setup.
3. Launch BuddyGPT from Start Menu.
4. On first wake-up, paste your Anthropic API key when the dog asks.

That is it - install, wake, ask.

## Product Soul

- `Soul.md`: product identity, boundaries, and behavior principles.
- `docs/soul-roadmap.md`: implementation status and future milestones aligned with `Soul.md`.

## How It Works

1. **Resting mode**: the Shiba hangs out in the corner.
2. **Wake up**: press `Ctrl+Shift+Space` or click the dog.
   - On first run (when no API key is configured), BuddyGPT enters onboarding and asks you to paste your API key.
3. **Context capture**: BuddyGPT captures your **last active window before wake-up**.
4. **Ask**: type your question and press `Enter`.
5. **Thinking -> Reply**: it analyzes your context and gives a short answer.
6. **Back to rest**:
- Press `Esc` to dismiss immediately.
- If there is no user response after a reply, it auto-returns to `resting` after **15 seconds**.

## Last Active Window Behavior

BuddyGPT tries to answer based on what you were just working on.

Examples:
- If you were writing an email, it captures that email window.
- If you were on a browser tab, it captures that tab window.

Implementation note:
- Wake-up from hotkey and wake-up from mouse click share the same activation pipeline.
- If overlay is foreground during activation, the code attempts to skip overlay and recover your previous real window.

## Pet States

| Resting (`zzZ`) | Awake (`Ask me anything!`) |
|---|---|
| ![Resting state](assets/shiba/states/state-resting.png) | ![Awake state](assets/shiba/states/state-awake.png) |
| Thinking (`Hmm...`) | Reply (`Ask more, or Esc to close`) |
| ![Thinking state](assets/shiba/states/state-thinking.png) | ![Reply state](assets/shiba/states/state-reply.png) |

Pet character attribution: from **Nudaeng** (`@nudaengdotbonk`).

## Hotkeys and Use Cases

| Hotkey / Action | Typical Use Case |
|---|---|
| `Ctrl+Shift+Space` | Wake BuddyGPT while you are reading an email, ticket, or docs page and want quick help. |
| `Ctrl+Shift+V` | Wake BuddyGPT with current clipboard text as first-turn context (no screenshot required). |
| Click the dog | Same as hotkey wake-up when your hand is already on the mouse. |
| `Enter` | Send your question after BuddyGPT captures context. |
| `Esc` | Close the current session immediately and send the dog back to rest. |
| `Ctrl+Shift+Q` | Quit BuddyGPT completely when you are done for the day. |

## Features

- Animated Shiba states: `resting` / `greeting` / `awake` / `thinking` / `idle_chat` / `reply`
- Active-window screenshot understanding
- App-aware context filtering
- Optional web lookup (DuckDuckGo) when needed
- Proactive daily news:
  - first wake-up push
  - scheduled pushes at 15:00 and 20:00 (local time)
  - same-day topic dedupe (no repeated topic in one day)
- Short, colleague-style answers
- Language matching (ask in Chinese, get Chinese; ask in English, get English)

## Setup

```bash
pip install -r requirements.txt
py main.py
```

## Windows Installer

Build a Windows app + installer from this repo:

```powershell
# from repository root
.\scripts\build_windows.ps1
```

Build only the app (skip Setup.exe):

```powershell
.\scripts\build_windows.ps1 -SkipInstaller
```

If your default Python is not the build target, set it explicitly:

```powershell
.\scripts\build_windows.ps1 -PythonCmd "py -3.12"
```

If your network requires a custom package index:

```powershell
.\scripts\build_windows.ps1 -PipIndexUrl "https://pypi.org/simple"
```

Outputs:
- App folder: `dist\BuddyGPT\`
- Single EXE: `dist\BuddyGPT\BuddyGPT.exe`
- Installer (when Inno Setup is available): `dist\installer\BuddyGPT-Setup.exe`

Notes:
- The script uses PyInstaller for EXE packaging.
- Setup builder uses Inno Setup (`iscc.exe`) via `packaging\BuddyGPT.iss`.
- If `pyinstaller` cannot be resolved, provide a reachable index URL with `-PipIndexUrl`.

## Uninstall

You can uninstall BuddyGPT in three ways:
- Windows Settings -> Apps -> Installed apps -> BuddyGPT -> Uninstall
- Start Menu -> BuddyGPT -> `Uninstall BuddyGPT`
- Silent uninstall (for scripts/IT):

```cmd
"C:\Program Files\BuddyGPT\unins000.exe" /VERYSILENT /SUPPRESSMSGBOXES /NORESTART
```

## Configuration

Use `config.json`:

```json
{
  "api_key": "",
  "openai_api_key": "",
  "backend": "anthropic",
  "model": "claude-sonnet-4-20250514",
  "openai_base_url": "https://api.openai.com/v1",
  "ollama_base_url": "http://127.0.0.1:11434",
  "backend_timeout_sec": 45,
  "personality": "buddy",
  "hotkey_activate": "ctrl+shift+space",
  "hotkey_clipboard": "ctrl+shift+v",
  "hotkey_quit": "ctrl+shift+q",
  "screenshot_interval": 3.0,
  "hash_threshold": 12,
  "max_tokens": 400,
  "history_window_turns": 6,
  "history_summary_every_turns": 6,
  "history_summary_max_chars": 1800,
  "enable_monitor": false,
  "allow_private_url_browse": true,
  "context_max_chars": 9000,
  "context_reference_refresh_turns": 3,
  "url_cache_ttl_sec": 300,
  "ocr_cache_ttl_sec": 300,
  "search_cache_ttl_sec": 90,
  "context_telemetry": true,
  "tray_mode": false,
  "show_token_cost": false,
  "adaptive_output_caps": true,
  "session_recovery_ttl_sec": 120,
  "enable_ocr_fallback": false,
  "ocr_max_chars": 3000,
  "ocr_timeout_sec": 5,
  "ocr_preferred_apps": ["terminal", "vscode", "gmail", "outlook", "word", "pdf_reader"],
  "tesseract_cmd": "",
  "proactive_hints": false,
  "proactive_sensitivity": "medium",
  "proactive_cooldown_sec": 90,
  "proactive_max_per_hour": 8,
  "proactive_quiet_hours_enabled": false,
  "proactive_quiet_start": "22:00",
  "proactive_quiet_end": "08:00",
  "daily_chat": {
    "enabled": true,
    "push_times": ["15:00", "20:00"],
    "max_topic_retry": 3
  }
}
```

Or use `.env`:

```bash
ANTHROPIC_API_KEY=sk-ant-xxx
OPENAI_API_KEY=sk-proj-xxx
```

Config priority in current code:
1. Backend is selected via `config.json -> backend`.
2. Anthropic key: `config.json -> api_key` then `.env -> ANTHROPIC_API_KEY`.
3. OpenAI key: `config.json -> openai_api_key` then `.env -> OPENAI_API_KEY`.
4. Ollama does not require an API key.

Additional config notes:
- `backend`: `anthropic`, `openai`, or `ollama`.
- `model`: provider model name; if incompatible with selected backend, BuddyGPT falls back to a backend default model.
- `openai_api_key`: used when `backend=openai`.
- `ollama_base_url`: used when `backend=ollama` (default local endpoint).
- `personality`: `buddy` (short default), `detailed`, or `terse`.
- For `detailed` or `terse`, if `max_tokens` remains at default `400`, BuddyGPT uses the personality token default automatically.
- `history_window_turns`: max number of recent user turns retained per session.
- `history_summary_every_turns`: cadence for rolling history summary updates when older turns are trimmed.
- `history_summary_max_chars`: cap for the rolling summary block added to system context.
- `hotkey_clipboard`: wake with clipboard text context.
- `enable_monitor`: controls whether background `ScreenMonitor` starts at app launch.
- `proactive_hints`: if enabled (and monitor enabled), show non-LLM proactive alert nudges when significant screen changes are detected.
- `proactive_sensitivity`: `low`, `medium`, `high`; threshold scales from `hash_threshold`.
- `proactive_cooldown_sec` and `proactive_max_per_hour`: anti-noise controls for proactive hints.
- `proactive_quiet_hours_enabled`: suppress proactive hints during quiet window.
- `proactive_quiet_start` / `proactive_quiet_end`: quiet-hours window (`HH:MM` local time, supports overnight windows).
- `allow_private_url_browse`: allows or blocks localhost/private-network URLs in direct URL browse mode.
- `context_max_chars`: character budget used for token-aware context packing before each ask.
- `context_reference_refresh_turns`: how often static context is resent in full vs reference-only.
- `url_cache_ttl_sec` / `ocr_cache_ttl_sec` / `search_cache_ttl_sec`: cache TTLs for URL fetch, OCR reuse, and short-term web search reuse.
- `context_telemetry`: enables per-turn context token estimate logging by block.
- `tray_mode`: hide pet to system tray between interactions.
- `show_token_cost`: display per-turn and session token cost estimate in the overlay.
- `adaptive_output_caps`: shrink or expand reply caps by turn type instead of always using the same `max_tokens`.
- `session_recovery_ttl_sec`: how long a dismissed reply/draft stays recoverable on the next manual activation.
- `enable_ocr_fallback`: optional local OCR extraction for text-heavy app contexts.
- `tesseract_cmd`: optional absolute path to `tesseract.exe`; if empty, BuddyGPT checks common Windows paths and PATH.

UX notes:
- Press `Esc` while BuddyGPT is thinking to cancel the current request.
- Each turn lets you toggle `Use screenshot`, `Use clipboard`, `Use URLs`, and `Use OCR` before sending.
- If you dismiss BuddyGPT accidentally, the most recent reply/draft can be restored on the next manual wake within the recovery TTL.
- Reply quick actions are available after each answer: `Explain simpler`, `Give steps`, and `Copy answer`.
- In tray mode, you can `Snooze Hints 1h` or `Resume Hints` from the tray menu.

## Windows Smoke Checks

Run the local smoke harness before packaging or after dependency changes:

```powershell
python scripts/windows_smoke.py
```

Useful variants:

```powershell
python scripts/windows_smoke.py --checks hotkey clipboard tray
python scripts/windows_smoke.py --json
```

What it covers:
- hotkey callback wiring
- clipboard capture/readback
- OCR availability + basic extraction
- tray icon construction

Installed app config location:
- `%APPDATA%\BuddyGPT\config.json`
- `%APPDATA%\BuddyGPT\.env`

## Token Usage (Detailed)

### When tokens are NOT used

No model tokens are consumed for:
- idle animation
- wake-up UI itself
- local window detection
- local screenshot capture/filtering
- drag/move UI actions
- auto-return to resting

### When tokens ARE used

Model tokens are consumed when:
- you send a question (`Enter`)
- you send follow-up questions
- tool-use triggers extra model rounds (for web lookup flow)

### What contributes to token count

Per request, usage is roughly:
- **Input tokens**: system prompt + your question + conversation history + attached context
- **Output tokens**: assistant reply

Important details:
- First turn commonly includes the captured window image.
- Image content can significantly increase input token usage.
- Longer follow-up chains increase history size and input tokens.

### What `max_tokens` actually does

- `max_tokens` limits **output tokens per model call**.
- It does **not** limit input tokens.
- If tool-use causes multiple model calls, each call has its own output cap.

### How to inspect token usage

Runtime logs already print:
- `input_tokens`
- `output_tokens`

You can use this to identify high-cost workflows.

### Practical ways to reduce token cost

- Ask more specific questions in one turn.
- Keep sessions shorter when possible.
- Avoid unnecessary web-search style prompts.
- Wake and ask from cleaner, less noisy screens.

## Privacy and Data Boundaries (Detailed)

### What stays local

These happen locally on your machine:
- hotkey listening
- mouse click wake handling
- active window detection
- screenshot capture and filtering
- UI rendering and pet state machine

Conversation history is held in memory and cleared on each new wake-up session.

### What may be sent externally

When you submit a question, request payload may include:
- your question text
- captured screenshot context
- prompt/context strings

If web lookup is triggered, additional query/result text may be exchanged in the tool-use flow.

### API key handling

- API key can come from `config.json` or `.env`.
- Both `.env` and `config.json` are gitignored in this project.
- Best practice: keep a single source of truth (usually `.env`).

### Main privacy risks

- screenshots may contain sensitive info (email content, customer data, internal links)
- terminal windows may expose secrets
- logs/screenshots shared externally can leak data

### Privacy best practices

- Check the foreground window before waking BuddyGPT.
- Mask sensitive content before asking.
- Rotate API keys immediately if exposure is suspected.
- Do not share logs/screenshots/configs that may contain secrets.

## Project Structure

```text
BuddyGPT/
|-- main.py
|-- config.example.json
|-- requirements.txt
|-- src/
|   |-- overlay.py
|   |-- pet.py
|   |-- ai_assistant.py
|   |-- screenshot.py
|   |-- content_filter.py
|   |-- app_detector.py
|   |-- web_search.py
|   '-- ...
'-- assets/
```

## Requirements

- Windows 10/11
- Python 3.12+
- Anthropic or OpenAI API key for cloud backends, or local Ollama for keyless local mode

If you see the little Shiba napping, everything is working as intended.

## FAQ

### 1) Is DuckDuckGo web search free?

Yes, search itself is free in this project (no separate search API key).  
If search is used, total model token usage can still increase because extra rounds are needed to read and summarize search results.

### 2) Does BuddyGPT keep spending tokens while idle?

No. Idle animation, wake-up, and local UI actions do not consume model tokens.  
Tokens are used only when you submit a question (and optional tool-use rounds).

### 3) Why does BuddyGPT sometimes answer using the wrong window?

BuddyGPT captures the last active window at wake-up.  
If focus changes right before activation, context may be off. Wake it again while the correct app is focused.

### 4) Do I need both `config.json` and `.env` API keys?

No. Use one source of truth.  
For Anthropic: `config.json -> api_key` overrides `.env -> ANTHROPIC_API_KEY`.  
For OpenAI: `config.json -> openai_api_key` overrides `.env -> OPENAI_API_KEY`.
