<div align="center">

# AARMA

**Adaptive AI Voice-Controlled Desktop Agent**

A desktop assistant that understands what you ask, checks what it is allowed to do,
and then does only that.

`Phase 6 — Sessions, address and voice profiles`  ·  `Python 3.10+`  ·  `Windows`  ·  `103 actions`  ·  `1932 tests passing`

</div>

---

> **Status: Phases 1 to 6 implemented.**
> Text and voice commands, a two-stage wake word, nearest-intent language
> understanding, 103 registered actions, the permission and risk model,
> reminders that actually fire, window/browser/workspace control, the
> conversation transcript and male/female voice profiles are implemented and
> tested. Speech recognition runs on your computer; audio is never uploaded and
> never written to disk.
>
> **Not implemented:** a language-model provider (the interface exists and
> returns "no model configured"), long-term memory, phishing detection, persona
> mode, non-English interface text, and an installer. Where this README
> describes those, it is describing the roadmap, and says so.

---

## Contents

- [Overview](#overview)
- [Problem statement](#problem-statement)
- [Objectives](#objectives)
- [Key features in this build](#key-features-in-this-build)
- [Architecture](#architecture)
- [Technology stack](#technology-stack)
- [Repository structure](#repository-structure)
- [Installation](#installation)
- [Environment configuration](#environment-configuration)
- [Running AARMA](#running-aarma)
- [Supported commands](#supported-commands)
- [Testing](#testing)
- [Security model](#security-model)
- [Multilingual design](#multilingual-design)
- [Interface](#interface)
- [Offline and online modes](#offline-and-online-modes)
- [Public distribution plan](#public-distribution-plan)
- [What is and is not implemented](#what-is-and-is-not-implemented)
- [Roadmap](#roadmap)
- [Development rules](#development-rules)
- [License](#license)

---

## Overview

AARMA is a desktop AI agent for Windows. You type (and, in a later phase, say)
what you want in ordinary language; AARMA works out which of its registered
capabilities that maps to, checks whether you have granted it permission,
weighs how risky the action is, asks you first if it is dangerous, performs it,
and tells you what happened.

Two assistant identities ship with it — **Asma** (female) and **Aariz** (male) —
and either can be renamed, revoiced and reconfigured. The assistant's name,
language, wake word, permissions and security settings are all user-controlled
and persist across restarts.

The design principle running through the whole codebase is that **the part of
the system that understands language is never the part that touches the
machine.** Understanding produces a structured request; a separate,
security-checked layer decides whether to honour it.

---

## Problem statement

Desktop assistants tend to fail in one of two directions.

Some are **too weak to be useful**: they match a fixed list of exact phrases, so
anything the author did not anticipate simply fails, and they cannot be extended
without editing the parser.

Others are **too dangerous to trust**: they hand a language model a shell, or an
"execute this command" tool, and rely on the model's judgement not to do
something destructive. That design has no floor. A model that is wrong — through
a mistake, an ambiguous instruction, or text injected into a document it was
asked to read — can delete files, exfiltrate data or install software, and
nothing in the architecture stands in the way.

AARMA is built around the position that **capability must be enumerated, not
inferred**. The assistant can do exactly the things someone deliberately
registered, with the permissions that were deliberately granted, and nothing
else — however the request is phrased, and whoever (or whatever) phrased it.

---

## Objectives

1. Understand natural language rather than fixed command strings.
2. Never allow arbitrary command execution, including from a future language model.
3. Put the user in explicit control of every capability, visibly and revocably.
4. Grade actions by risk and require human confirmation for destructive ones.
5. Support multiple languages as an architectural concern, not a bolt-on.
6. Work offline for local tasks, and be honest when it cannot.
7. Look and behave like a real desktop product, not a demonstration script.
8. Stay modular enough that voice, AI reasoning and packaging can be added
   without rewriting what already works.

---

## Key features in this build

| Area | What works today |
|---|---|
| **Voice output** | Natural neural voices in 10 languages online, Windows voices offline, with explicit fallback — see [`docs/VOICE.md`](docs/VOICE.md) |
| **Voice input** | On-device speech recognition; never fabricates a transcription; audio never leaves the machine; length-aware confidence gate so short commands are not refused for being short — see [`docs/SPEECH-INPUT.md`](docs/SPEECH-INPUT.md) |
| **Speech diagnostics** | Raw recogniser output shown *before* validation, exact rejection reason for every refusal, and a real-voice accuracy test that never executes what it hears |
| **Natural language** | Rule-based intent engine handling phrasing variety, wake-word prefixes and politeness filler |
| **Action registry** | 21 registered capabilities, each with a typed parameter schema, permission list and risk grade |
| **Permissions** | 15 user-controlled capability switches, least-privilege defaults, live toggles in the GUI |
| **Risk model** | Low / Medium / High grading; high-risk actions always require explicit confirmation |
| **Filesystem sandbox** | Every path confined to the user's own content folders, resolved via the Windows known-folder API |
| **Security** | Optional passphrase, PBKDF2-HMAC-SHA256, lock/unlock, secrets redacted from logs |
| **Identity** | Two assistant profiles, renameable by voice-style command or in Settings, persisted |
| **Language** | Interface / input / output languages configured independently, 10 languages in the catalogue |
| **Persistence** | SQLite with a migration ladder: transcript, action audit trail, permission decisions |
| **Interfaces** | Styled PySide6 dashboard (5 pages, dark/light) and a full-featured text mode |
| **First-run setup** | Eight-step wizard in the GUI, compact equivalent in the CLI |
| **Testing** | 792 tests covering identity, language, registry, security and the end-to-end pipeline |

---

## Architecture

Every command — typed today, spoken later, model-proposed in Phase 2 — travels
the same path. There is no shortcut around the middle of it.

```
                    USER
                      │
        ┌─────────────┴─────────────┐
        │  wake word  →  speech-to-text │   ← Phase 2 (interfaces exist, no engine)
        └─────────────┬─────────────┘
                      │
              LANGUAGE MANAGER              language/manager.py
                      │
         ┌────────────┴────────────┐
         │      INTENT ENGINE      │        ai/intent.py     (rules today)
         │   └ LLM PROVIDER        │        ai/llm.py        (Phase 2)
         └────────────┬────────────┘
                      │  Intent  (structured, not text)
               TASK PLANNER                 ai/planner.py
                      │  ActionRequest
        ══════════════╪══════════════   ← nothing below trusts anything above
                      │
            1.  REGISTRY LOOKUP             actions/registry.py
                      │                     unregistered name → refused
            2.  PARAMETER VALIDATION        typed schema; undeclared keys refused
                      │
            3.  PERMISSION CHECK            security/permissions.py
                      │                     not granted → refused + audited
            4.  CONNECTIVITY CHECK          core/connectivity.py
                      │                     online-only action offline → refused
            5.  PRE-CHECK                   refuse before asking, not after
                      │
            6.  RISK / CONFIRMATION         security/validator.py
                      │                     HIGH → parked until the human says yes
            7.  PASSPHRASE GATE             security/passphrase.py
                      │
              TOOL EXECUTOR                 core/engine.py
                      │
        ┌─────────────┼─────────────┐
     apps        filesystem       browser        ← actions/*.py, the only OS access
        └─────────────┼─────────────┘
                      │
                  AUDIT TRAIL                database/db.py
                      │
             RESPONSE GENERATION             language/manager.py
                      │
                 text-to-speech              ← Phase 2 (interface exists)
                      │
                    USER
```

### The rule this enforces

`ai/llm.py` defines the boundary a language model will sit behind. A provider
may return a `ToolProposal` — an action *name* and a dictionary of arguments.
It cannot return a command, a script, or a path to execute, because that is not
a shape the interface can express.

The proposal then enters the pipeline at exactly the same point as a typed
sentence. A model that invents `run_shell_command` gets
`UnknownActionError` from the registry. A model that adds a `shell` argument to
`get_time` gets `ValidationFailed`. A model that proposes a real, destructive
action still has to wait for a human to click Yes.

Tests pin this down: `tests/test_registry.py::TestLookup::test_a_model_style_hallucinated_name_is_refused`.

### Layering

Dependencies point strictly downward. `utils` depends on nothing; `config`
depends on `utils`; `security`, `language` and `database` depend on `config`;
`actions` depends on those; `ai` and `core` sit above; `ui` sits above
everything and contains no security logic at all.

---

## Technology stack

| Layer | Choice | Why |
|---|---|---|
| Language | Python 3.10+ | Type hints, dataclasses, mature standard library |
| Desktop UI | PySide6 (Qt 6) | Official Qt binding, LGPL, full control over styling, packages cleanly |
| Styling | Generated Qt stylesheet from design tokens | One palette definition drives dark and light themes |
| Persistence | SQLite (stdlib `sqlite3`) | Local, zero-configuration, no server |
| Config | JSON | User-readable and user-repairable |
| Passphrase | `hashlib.pbkdf2_hmac` (stdlib) | No third-party crypto dependency to audit |
| Voice (online) | `edge-tts` — Microsoft neural voices | Genuinely natural, covers every language AARMA declares, no key required |
| Voice (offline) | Windows SAPI 5 via `pywin32` | Always present on Windows, no download; honestly labelled *standard* quality |
| Speech recognition | Whisper via `faster-whisper` | Runs on-device — audio never leaves the machine — and the only local option covering AARMA's declared languages |
| Microphone | `sounddevice` (PortAudio) | Device enumeration and clean capture; bundled wheels |
| Audio playback | Windows MCI (`winmm.dll`) | Decodes MP3 natively — no codec to bundle |
| Screen capture | Pillow *(optional)* | Widely available; degrades honestly if absent |
| Deletion | Send2Trash *(optional)* | Recycle Bin, so deletion is recoverable |
| Testing | pytest | Fixtures make sandboxed, non-destructive tests straightforward |

The core engine runs on the **standard library alone**. Every third-party
package is tied to one capability and its absence produces a clear message
rather than a crash.

---

## Repository structure

```
AARMA/
├── main.py                     Entry point: --cli, --setup, --doctor
├── README.md
├── requirements.txt
├── pytest.ini
├── .gitignore
├── .env.example
│
├── config/
│   └── settings.py             All user-adjustable settings; atomic JSON persistence
│
├── ai/
│   ├── intent.py               Rule-based natural-language understanding
│   ├── planner.py              Intent → registry-backed ActionRequest
│   ├── llm.py                  Language-model interface (no model in Phase 1)
│   └── memory.py               Short-term conversation context
│
├── speech/
│   ├── engine.py               Voice orchestration: queue, worker, fallback
│   ├── speaker.py              Text-to-speech provider interface
│   ├── voices.py               Voice profiles, quality tiers, language mapping
│   ├── text.py                 Speech safety filtering and natural phrasing
│   ├── audio.py                Playback (Windows MCI)
│   ├── providers/
│   │   ├── edge.py             Online neural voices
│   │   └── sapi.py             Offline Windows voices
│   ├── listener.py             Speech-to-text provider interface
│   ├── capture.py              Microphone capture, devices, endpointing
│   ├── input_engine.py         Listening state machine and worker
│   ├── providers/whisper_local.py  On-device recognition
│   └── wakeword.py             Wake-word interface (Phase 2C)
│
├── language/
│   └── manager.py              Language catalogue, resolution, all user-facing strings
│
├── actions/
│   ├── registry.py             The allow-list: ActionSpec, validation, lookup
│   ├── system.py               get_time, get_date, get_status
│   ├── applications.py         Approved-application launcher and closer
│   ├── browser.py              open_url, search_web
│   ├── filesystem.py           Sandboxed file and folder operations
│   ├── screenshot.py           Screen capture
│   ├── assistant.py            Rename, profile, language, wake word
│   ├── keyboard.py             Phase 3 — registers nothing
│   ├── mouse.py                Phase 3 — registers nothing
│   └── reminders.py            Phase 3 — registers nothing
│
├── security/
│   ├── permissions.py          Capability switches and checks
│   ├── validator.py            Risk levels, path sandbox, URL safety
│   ├── confirmation.py         Token-based human-in-the-loop approval
│   └── passphrase.py           PBKDF2 hashing, lock/unlock
│
├── core/
│   ├── engine.py               The pipeline — the only caller of action handlers
│   └── connectivity.py         Online/offline state
│
├── database/
│   └── db.py                   SQLite with a forward-only migration ladder
│
├── ui/
│   ├── dashboard.py            Main window: 5 pages, threaded engine calls
│   ├── setup_wizard.py         Eight-step first-run setup
│   ├── console.py              Text interface
│   ├── responsive.py           Screen/DPI/font-aware sizing, flow + reflow layouts
│   ├── theme.py                Design tokens, stylesheet, generated icon/avatar
│   └── widgets.py              Reusable presentation components
│
├── utils/
│   ├── logger.py               Structured logging with secret redaction
│   └── paths.py                Data root and Windows known-folder resolution
│
├── tests/                      792 tests, none destructive
├── docs/                       Architecture, security, setup, roadmap, screenshots
└── data/                       Portable data root (empty in version control)
```

> **Two additions to the specified layout, both deliberate.** `core/` holds the
> orchestrator, which cannot live in `ui/` (both interfaces need it) or in `ai/`
> (it is a security boundary, not a language concern). `actions/assistant.py`
> routes "change your name to Nova" through the registry like any other action,
> instead of special-casing it in the parser.

---

## Installation

**Requires Python 3.10 or newer on Windows.**

```bash
git clone <repository-url>
```

```bash
cd AARMA
```

```bash
python -m venv .venv
```

```bash
.venv\Scripts\activate
```

```bash
pip install -r requirements.txt
```

Confirm the installation:

```bash
python main.py --doctor
```

That prints the Python version, where AARMA will store its data, how many
actions are registered, and which optional components are present.

### Running without the optional packages

The engine and the text interface need nothing beyond the standard library. If
PySide6 is unavailable, `python main.py --cli` still works and `python main.py`
explains what to install. If Pillow or Send2Trash are missing, those two actions
report it rather than failing obscurely — and AARMA will **not** fall back to
permanent deletion when the Recycle Bin component is absent.

---

## Environment configuration

Copy `.env.example` to `.env` and edit it. Nothing in it is required.

| Variable | Purpose | Default |
|---|---|---|
| `AARMA_HOME` | Where config, database, logs and screenshots are written | `%APPDATA%\AARMA` |

Set `AARMA_HOME=./data` for a fully portable checkout that writes nothing
outside the project folder.

**Phase 1 reads no API keys.** The names reserved for Phase 2 providers are
listed in `.env.example`. `.env` is git-ignored; never commit a real key.

---

## Running AARMA

```bash
python main.py
```

| Command | What it does |
|---|---|
| `python main.py` | Desktop dashboard (first run opens the setup wizard) |
| `python main.py --cli` | Text interface in the terminal |
| `python main.py --setup` | Re-run first-run setup, then continue |
| `python main.py --doctor` | Environment report, then exit |
| `python main.py --log-level DEBUG` | Verbose logging for this run |

### Text mode

```
you > what time is it
Asma > The current time is 9:10 AM.

you > create a folder called AI Project in Documents
Asma > Created 'AI Project' in C:\Users\you\OneDrive\Documents.

you > delete AI Project
Asma > 'delete_item' is a high-risk action. Reply 'yes' to confirm or 'no' to cancel.
         Send a file or folder to the Recycle Bin. (path: AI Project)

you > yes
Asma > Sent 'AI Project' to the Recycle Bin.

you > exit
Asma > Goodbye.
```

---

## Supported commands

21 registered actions. `~` changes something on your machine; `!` is high risk
and always confirmed first.

| Action | Risk | Needs | Say something like |
|---|---|---|---|
| `get_time` | Low | — | "what time is it" |
| `get_date` | Low | — | "what's the date" |
| `get_status` | Low | — | "status", "are you online" |
| `open_application` | Low | Launch applications | "open chrome", "launch notepad" |
| `close_application` | ~ Medium | Launch apps + System control | "close notepad" |
| `list_applications` | Low | — | "what apps can you open" |
| `open_url` | Low | Browser + internet | "open youtube.com" |
| `search_web` | Low | Browser + Internet + internet | "search the internet for NVIDIA" |
| `list_files` | Low | Files and folders | "list files in Documents" |
| `list_allowed_folders` | Low | — | "which folders can you access" |
| `create_folder` | ~ Medium | Files and folders | "create a folder called AI Project in Documents" |
| `open_file` | ~ Medium | Files and folders | "open resume.pdf" |
| `rename_item` | ~ Medium | Files and folders | "rename draft.txt to final.txt" |
| `move_item` | ~ Medium | Files and folders | "move report.pdf to Desktop" |
| `delete_item` | **! High** | Files and folders | "delete old draft.txt" |
| `take_screenshot` | Low | Screenshots | "take a screenshot" |
| `set_assistant_name` | ~ Medium | — | "change your name to Nova" |
| `set_assistant_profile` | ~ Medium | — | "switch to the Aariz profile" |
| `set_language` | Low | — | "change language to Hindi" |
| `list_languages` | Low | — | "what languages do you speak" |
| `set_wake_word` | ~ Medium | — | "set the wake word to hey nova" |

Plus `help`, `hello`, `thanks` and `exit`. Type `help` in either interface for
the live, grouped list.

**Phrasing is not fixed.** "open chrome", "launch chrome", "start chrome",
"could you open chrome please" and "hey asma, open chrome" all reach the same
action. An unrecognised sentence produces "I didn't understand that yet" — never
a guess.

---

## Testing

```bash
python -m pytest
```

```
238 passed
```

| File | Covers |
|---|---|
| `test_identity.py` | Profile defaults, rename, wake-word tracking, persistence, corrupt-config recovery |
| `test_language.py` | Catalogue, alias resolution, scoped changes, fallback honesty, persistence |
| `test_registry.py` | Registration, lookup, unknown-action refusal, parameter schema, risk grading |
| `test_security.py` | Permission defaults and checks, risk policy, path sandbox, URL safety, confirmations, passphrase, log redaction |
| `test_intent.py` | Phrasing variety, parameter extraction, rule ordering, unknown input, fuzz safety |
| `test_engine.py` | End-to-end pipeline, permission enforcement, offline refusal, safe execution, confirmation flow, audit trail |
| `test_ui_responsive.py` | Window placement across 6 resolutions × 5 scaling factors, minimize/maximize/restore, scrollable pages, reflow, eliding, DPI-aware art |
| `test_speech.py` | Voice profiles, provider selection and fallback, language refusal, credential filtering, delivery controls, stop/replace, persistence |
| `test_speech_input.py` | Recognition contract, permission enforcement, cancellation, confidence gating, **never-fabricate refusals**, privacy state, engine integration |
| `test_speech_pipeline.py` | Capture endpointing, voice activity detection, resampling, device resolution and recovery |
| `test_microphone_diagnostics.py` | Windows permission detection, per-device probing, the eight capture verdicts, level formatting |
| `test_speech_reliability.py` | **Raw recogniser output preserved before validation**, exact rejection reasons, 20 consecutive cycles, failure isolation, name priming and its prompt limits, diagnostic audio off by default, accuracy scoring |

**No test is destructive.** Every test redirects `AARMA_HOME` into a pytest
temporary directory and roots the filesystem sandbox in a scratch folder.
Application launching, browser opening and deletion are patched — the tests
assert on *what was requested*, and the out-of-sandbox tests assert that the
real file is still there afterwards.

---

## Security model

### Permissions

15 capabilities, each independently switchable in Settings → Permissions and
effective on the very next command.

**Granted by default** — Speaker, Browser, Internet, Launch applications,
Files and folders, Reminders, Memory.

**Denied by default** — Microphone, Camera, Screenshots, Keyboard control,
Mouse control, System control, Calendar, Email.

The split is deliberate: anything that can **observe you** or **control the
machine** requires a decision, not an oversight.

### Risk levels

| Level | Meaning | Gate |
|---|---|---|
| **Low** | Read-only or trivially reversible | Permission check only |
| **Medium** | Changes something, not destructively | Permission check, logged and audited |
| **High** | Destructive, irreversible or security-sensitive | **Always** requires explicit confirmation |

The high-risk confirmation floor is **not user-adjustable**. There is no setting
that turns it off, and a test asserts that relaxing every other security setting
does not remove it.

### Filesystem sandbox

Every path argument is resolved by `SafePathResolver` before use, and confined
to the user's own content folders. `..`, absolute system paths, path separators
inside a name and Windows reserved device names (`CON`, `NUL`, `COM1`…) are all
refused rather than sanitised — quietly rewriting `../secrets` to `secrets`
would hide an attempted escape instead of surfacing it.

Folder locations come from the Windows known-folder API, not from guessing
`~/Documents`. On a profile with OneDrive Backup enabled — the default on many
machines — the guess is simply the wrong folder.

The sandbox holds **after** confirmation too: approving a deletion outside the
allowed roots still refuses at execution.

### Command execution

There is none. No registered action accepts a command line, and shells and
terminals are deliberately absent from the approved-application catalogue —
launching `cmd` would turn a Low-risk action into arbitrary code execution.
`open_file` refuses executable extensions for the same reason.

### Secrets

The passphrase is stored only as PBKDF2-HMAC-SHA256 (210,000 iterations,
16-byte random salt) and verified with `hmac.compare_digest`. It is never
written to the configuration file — a test reads the file back and asserts the
plaintext is absent — and never to a log. As defence in depth, every log record
passes through a redacting filter that scrubs anything keyed as a password,
passphrase, token, API key or secret, in the message *and* in the structured
payload.

Security mode starts **locked** after a restart. Action parameters are redacted
before they are written to the audit trail.

### Audit trail

Every action is recorded with its risk, source, whether it was allowed, whether
it succeeded and how long it took. Every permission denial is recorded
separately. Both are visible in the Activity page.

More detail: [`docs/SECURITY.md`](docs/SECURITY.md).

---

## Multilingual design

Language is a dedicated layer (`language/manager.py`), not a concern sprinkled
through modules. No other package hardcodes user-facing English text.

Three settings are configured **independently**, so you can read one language
and speak another:

- interface language
- speech input language
- speech output language

The catalogue distinguishes **translated** from **declared**:

| Status | Meaning | Languages |
|---|---|---|
| Translated | Interface strings exist | English |
| Declared | Selectable, persisted and routed through the whole system; strings fall back to English | Hindi, Kannada, Telugu, Tamil, Malayalam (Phase 2); Urdu, Arabic, Spanish, French (Phase 3) |

Selecting a declared language works and persists, and AARMA says so plainly:

> "Language set to Hindi. My interface text is still English in this build —
> full Hindi support is planned for a later phase."

That honesty is the point. Adding a language later means adding a message table
and a catalogue entry — no other module changes.

---

## Interface

Two front ends, both thin. Neither parses commands, checks permissions or
touches the operating system; both drive the same engine.

### Dashboard

Five pages behind a sidebar: **Conversation**, **Activity**, **Permissions**,
**Settings**, **About**. A header bar carries the assistant's generated avatar,
name, tagline and live status pills (state, language, voice, connectivity).

<div align="center">

![Conversation](docs/screenshots/dashboard-conversation.png)

*Conversation, with a high-risk action waiting for approval*

![Permissions](docs/screenshots/dashboard-permissions.png)

*Live capability switches*

![Voice settings](docs/screenshots/settings-voice.png)

*Voice settings — the status line always states what you would actually hear*

![Setup wizard](docs/screenshots/wizard-permissions.png)

*First-run setup, permissions step*

![Narrow window](docs/screenshots/dashboard-narrow.png)

*The same dashboard at 740×470 — sidebar narrowed, secondary status pills
dropped, chat bubbles reflowed, every control still reachable*

![Cramped wizard](docs/screenshots/wizard-small-window.png)

*The wizard at 660×430 — the step rail steps aside, content scrolls, and Back
and Next stay pinned outside the scroll area*

</div>

Design notes:

- **Engine calls run on a worker thread.** Launching an application or probing
  connectivity blocks for a moment; doing that on the GUI thread produces the
  frozen-window look that makes desktop software feel amateur.
- **Confirmations are inline, not modal.** A high-risk action appears as a card
  in the conversation with its own Yes/Cancel buttons, so you can read the
  surrounding context before deciding — and can simply ignore it.
- **Every colour comes from a design token.** Dark and light themes are the same
  stylesheet with a different palette.
- The microphone button is present and **disabled**, with a tooltip explaining
  that voice arrives in Phase 2. It does not pretend to listen.

### Display and window behaviour

AARMA is built for machines that look nothing like the developer's. Nothing in
the interface carries a hardcoded size.

| Variable | How it is handled |
|---|---|
| **Screen size** | Window size is *derived from* the work area — never chosen and then clamped. Centred, with room for the taskbar and the window frame. |
| **Display scaling** | Layouts are written in logical pixels and Qt maps them, with `PassThrough` rounding so 125% and 150% are honoured exactly rather than rounded. Generated art is painted at the device pixel ratio. |
| **Font scaling** | Every font size and spacing value derives from the user's own system font, so Windows' "make text bigger" enlarges the whole interface instead of clipping it. |

Both windows minimize, maximize, restore, resize and close like ordinary
Windows applications. Every page scrolls; navigation and confirmation controls
sit outside the scroll area and are always reachable. When space runs short the
interface reflows (grids drop to one column, chip rows wrap), rescales (sidebar
and chat bubbles take a share of the window) and sheds decorative chrome — it
never shrinks text or hides a control the user needs.

**Minimum supported size:** 720 × 460 logical pixels for the dashboard,
620 × 420 for the wizard. On a work area smaller than that the minimum yields
and content scrolls, so there is no screen on which AARMA opens a window bigger
than the space available. Verified down to a 640 × 336 logical work area.

The rules every future screen must follow are in
[`docs/UI-GUIDELINES.md`](docs/UI-GUIDELINES.md).

### Text mode

The same engine at a prompt — the fastest way to demonstrate the security model,
since the confirmation flow appears in the terminal exactly as it does in the GUI.

---

## Offline and online modes

AARMA states its mode and never pretends. Two things decide it: your
`network_mode` setting (`auto`, `online`, `offline` — where `offline` is a hard
override) and, in `auto`, a short TCP reachability probe cached for 20 seconds.

**Works offline** — conversation, time and date, application launching and
closing, all file operations, screenshots, settings, permissions, security,
memory, the database and the audit trail.

**Needs the internet** — `open_url` and `search_web`. Both declare
`requires_internet`, and the engine refuses them while offline with
"That needs an internet connection and you're currently offline" rather than
opening a browser to a failure page.

---

## Public distribution plan

The intended release flow:

```
source repository → automated build → PyInstaller bundle
   → Inno Setup installer → AARMA-Setup.exe → GitHub Release
   → download → install → first-run setup → ready
```

The installer will create the application and data directories, initialise the
database, create shortcuts and launch first-run setup. Users will not need
Python, pip or a terminal.

**None of this is built yet** — it is Phase 7. The groundwork that makes it
addable without rewriting anything is: code and user data are already strictly
separated (`utils/paths.py`), no path is hardcoded, the GUI has no compiled
assets to ship (the icon and avatar are drawn at runtime), and the dependency
list is short and all-Python.

---

## What is and is not implemented

### Implemented and working

Repository structure · configuration system · structured logging with redaction ·
assistant identity and conversational rename · language layer · permission
system · action registry · risk classification · confirmation flow ·
passphrase security · filesystem sandbox · rule-based intent engine ·
nearest-intent resolver · multi-step plans · **103 actions** · SQLite
persistence and audit trail · connectivity detection · text interface · desktop
dashboard · first-run wizard · neural and offline speech output · on-device
speech recognition · two-stage wake word · one-time session bootstrap ·
assistant-address extraction · reminders and scheduling · calendar and email
drafts · window, browser and workspace control · keyboard and mouse ·
male/female voice profiles · conversation transcript · **1932 tests** ·
documentation.

### Not implemented — and does not pretend to be

| Not implemented | Where the seam is |
|---|---|
| Language model | `ai/llm.py` — interface defined, `NullLLMProvider` returns "no model configured". Command understanding is deterministic rules plus the nearest-intent resolver, not a model |
| Long-term memory | `ai/memory.py` — short-term context only; the store waits for the view/edit/delete controls |
| Bare-name wake word | Only `hey aariz` has a validated acoustic model. A bare name wakes the assistant only once a detector has been enrolled for it — see *Wake word* below |
| Phishing detection | Not started |
| Persona mode | Not started |
| Non-English interface text | Selectable and persisted; strings fall back to English |
| Installer | Not started |

### Wake word

The shipped detector is a two-stage cascade trained, evaluated and frozen on the
phrase **"hey aariz"**. It hears that phrase and no other. A session starts
locked and is activated once by saying it; after that the assistant is addressed
by its current name in ordinary speech — *"Aariz, open Chrome"* — which is text
handling, not a second acoustic model.

Giving a renamed assistant a bare-name wake word therefore requires enrolling
one, which records your own samples locally:

```bash
python tools/enrol_wakeword.py
```

Until that is done the status line says so. Nothing claims the frozen model
recognises a phrase it was not trained on.

Modules for unimplemented features exist so the capability has an owner in the
architecture, but they **register nothing** and return no fabricated data. A
request for one of them gets the ordinary "unknown action" refusal, not a
plausible-looking lie.

---

## Roadmap

| Phase | Scope |
|---|---|
| **1 — Foundation** ✅ | Architecture, security model, registry, text mode, GUI shell, tests |
| **2A — Voice output** ✅ | Neural + offline TTS, voice profiles, speech safety filtering, GUI voice settings |
| **2B — Voice input** ✅ | On-device speech-to-text, microphone handling, listening states |
| **2C — Wake word** ✅ | Two-stage cascade detector, hands-free activation |
| **3 — Desktop depth** ✅ | Reminders that fire, keyboard/mouse, window and browser control, screenshots |
| **4 — Natural language** ✅ | Nearest-intent resolution; exact command strings no longer required |
| **5 — Workspace and system** ✅ | Repository/application/file discovery, bulk actions, gated shell execution |
| **6 — Sessions and voice** ✅ | Session bootstrap, assistant-address extraction, conversation transcript, male/female voice profiles |
| **Next — AI** | LLM provider and tool calling (`ai/llm.py` is the seam) |
| **Next — Memory** | Explicit long-term memories with view/edit/delete controls |
| **Next — Trust and persona** | Phishing/website risk analysis, persona simulation mode |
| **7 — Distribution** | PyInstaller build, `AARMA-Setup.exe`, GitHub Releases, auto-update |

Fuller detail: [`docs/ROADMAP.md`](docs/ROADMAP.md).

---

## Development rules

These are enforced by review and, where possible, by tests.

1. **Register every capability.** No code path executes anything not in the registry.
2. **Never execute arbitrary commands.** No action takes a command line. Argument
   lists are fixed; `shell=True` appears nowhere.
3. **Check permissions in one place.** Only `PermissionManager` answers whether a
   capability is allowed.
4. **Grade every action.** A new action declares its risk honestly. Destructive
   means High, and High always confirms.
5. **Validate at the boundary.** Parameters are typed and schema-checked;
   undeclared keys are refused, not ignored.
6. **Sandbox every path.** Filesystem arguments go through `SafePathResolver`.
7. **Never log a secret.** Never store one in plaintext.
8. **No misleading implementations.** An unimplemented feature reports that it is
   unimplemented. It does not return plausible fake data.
9. **Keep language in the language layer.** No user-facing English elsewhere.
10. **Keep configuration in the config layer.** No hardcoded names, paths or defaults.
11. **Keep the UI thin.** Interfaces render; they do not decide.
12. **Never hardcode a screen dimension.** Window sizes derive from the work
    area; font sizes and spacing derive from the user's font. Every page
    scrolls. See [`docs/UI-GUIDELINES.md`](docs/UI-GUIDELINES.md).
13. **Fail gracefully.** Users get a sentence; developers get the log.
14. **Test without destruction.** Temporary directories and mocks, always.
15. **Build in phases.** Do not start the next one before the current one is solid.

---

## License

Not yet selected. MIT is the intended choice at first public release.

---

<div align="center">

**AARMA** · Adaptive AI Voice-Controlled Desktop Agent · Phase 1 Foundation

</div>
