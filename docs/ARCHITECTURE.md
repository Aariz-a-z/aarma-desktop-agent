# Architecture

This document explains *why* AARMA is put together the way it is. The README
covers what it does; this covers the reasoning a reviewer would otherwise have
to reconstruct from the code.

---

## 1. The central decision

Most of the design follows from one choice:

> **The component that understands language is never the component that touches
> the machine.**

Understanding produces a structured `Intent`. A planner turns that into an
`ActionRequest`. A separate, security-checked executor decides whether to honour
it. Those three things live in different packages and communicate through
dataclasses, not function calls into each other.

This costs more code than letting a parser call `os.system` directly. It buys
one thing: when a language model is introduced in Phase 2, it enters the
pipeline at the *same* point a typed sentence does, and every check that
protects the user from a typo also protects them from a hallucination.

---

## 2. Layering

Dependencies point strictly downward. Nothing at a lower layer imports from a
higher one.

```
  ui/                    dashboard, wizard, console, theme, widgets
   │
  core/                  engine, connectivity
   │
  ai/                    intent, planner, llm, memory
   │
  actions/               registry + capability modules
   │
  security/   language/   database/          speech/
   │             │           │                 │
  config/  ← the single source of truth for every setting
   │
  utils/   ← paths, logging. Imports nothing from AARMA.
```

### Cycles that had to be designed around

**`config` ↔ `security`.** Permissions are user settings, so they belong in
`config`; the `Permission` enum belongs in `security`. Storing permissions as
`{name: bool}` strings keeps `config` free of any `security` import, and
`PermissionManager` does the enum↔string mapping.

**`actions.registry` ↔ capability modules.** The registry defines `ActionSpec`,
which every capability module imports. `build_default_registry()` therefore
imports those modules *inside the function body* rather than at module scope.

**`registry` ↔ `validator`.** `RiskLevel` lives in `security.validator`;
`ParameterSpec` and its validation live in `actions.registry`. The dependency
runs one way only.

---

## 3. The pipeline, step by step

`core/engine.py::AarmaEngine.execute` is the whole security model in one
readable function. The order is load-bearing.

| # | Step | Failure mode it prevents |
|---|---|---|
| 1 | Registry lookup | Executing a name nobody registered |
| 2 | Platform check | Running a Windows-only action elsewhere and failing obscurely |
| 3 | Parameter validation | Undeclared arguments reaching a handler |
| 4 | Permission check | Using a capability the user never granted |
| 5 | Connectivity check | Pretending an online feature works offline |
| 6 | Pre-check | Asking someone to approve something that will be refused anyway |
| 7 | Risk / confirmation | Destroying something without a human saying yes |
| 8 | Passphrase gate | A high-risk action running while the assistant is locked |

Step 6 exists because of a real behaviour observed during development: asking
"delete `C:\Windows\...`?", getting a yes, and *then* refusing. Confirming
things that get refused anyway teaches people to click through prompts, which
is exactly the habit the confirmation flow depends on them not having.

`execute()` is the only caller of `spec.handler`. Every interface goes through
it, so no front end can accidentally skip a step.

---

## 4. Where each concern lives

### Configuration (`config/settings.py`)

Frozen-ish dataclasses, one per concern, serialised to JSON. Writes are atomic
(temp file + `os.replace`), so a crash mid-save cannot truncate the file.
`from_dict` tolerates missing *and* unknown keys, so a config written by a
different build still loads. A corrupt file logs an error and falls back to
defaults rather than preventing startup.

Listeners fire after every successful mutation. That is how "change your name to
Nova" typed into the CLI updates the dashboard header — the GUI subscribes and
marshals the notification onto its own thread with a Qt signal.

### Language (`language/manager.py`)

Holds the catalogue, the alias table, and every user-facing string. The split
between `TRANSLATED` and `DECLARED` is what lets a language be *selectable and
fully routed* before it is *translated*, without lying about which it is.

Interface, input and output are three separate settings because they are three
separate decisions. Retrofitting that later is precisely how language logic ends
up leaking into unrelated modules.

### Actions (`actions/`)

An `ActionSpec` carries its own permissions, risk grade and typed parameter
schema. Risk travels with the *capability*, not with the caller — so there is no
code path that can execute an action while forgetting to check it.

Capability modules expose `get_actions()`. Adding a capability is: write a
handler, declare a spec, add the module to `build_default_registry`. Nothing
else changes.

Modules for unimplemented features (`keyboard`, `mouse`, `reminders`) return an
empty list. They exist so the capability has an owner and the planned surface is
written down, but they contribute nothing executable.

### Security (`security/`)

Four files, four questions:

- `permissions.py` — may AARMA touch this capability at all?
- `validator.py` — are the arguments safe, and how risky is this?
- `confirmation.py` — does a human have to say yes first?
- `passphrase.py` — does a human have to prove who they are?

`ConfirmationBroker` is token-based and non-blocking. The engine parks a request
and returns; the interface resolves it by token whenever the user decides. That
keeps one flow working across the CLI, the GUI and the future voice loop, and
means a user can always walk away without approving anything. Tokens are
single-use and expire.

### Database (`database/db.py`)

Connections are per-thread — the GUI thread and the engine's worker both touch
the database, and a `sqlite3.Connection` is not safe to share.

The schema is deliberately small. Tables for memories, reminders and calendar
sync are *not* pre-created; they arrive in the migration that ships with the
feature that uses them. An unused table is a promise the code has not kept.

### UI (`ui/`)

`theme.py` holds design tokens; nothing else names a colour. `responsive.py`
holds sizing tokens; nothing else names a dimension. The window icon and the
assistant avatar are drawn with `QPainter` at runtime, so they always match the
current palette and the current name — and there are no binary assets to ship or
to keep in sync with a rename.

`dashboard.py` moves the engine onto a `QThread`. Communication is by signal in
both directions, which is also what makes cross-thread config notifications safe.

#### The responsive layer

Three environment variables move independently, and conflating any two of them
produces clipped controls on somebody's laptop:

* **Screen size** — `compute_window_placement` derives a window rectangle from
  the work area. It is a pure function over a rectangle, which is what lets the
  whole resolution × scaling matrix be tested without owning the monitors.
* **Display scaling** — Qt maps logical pixels to physical ones, so layout code
  never mentions DPI. `PassThrough` rounding keeps fractional factors exact.
  Manually painted pixmaps are allocated at the device pixel ratio.
* **Font scaling** — `Metrics` derives every font size and spacing value from
  the user's own font. Stylesheet font sizes are emitted in points rather than
  pixels, because a QSS `px` ignores the user's text-size preference.

Above those sit three layout tools: `scroll_wrap` (a page that shrinks to
nothing and scrolls), `flow_container` (a row that wraps instead of
overflowing), and `ReflowGrid` (a grid that drops columns as it narrows).

The rules every screen must follow are in [`UI-GUIDELINES.md`](UI-GUIDELINES.md).
They are enforced by `tests/test_ui_responsive.py`, which drives both windows
through real resizes and asserts that navigation never leaves the frame.

---

## 5. Seams left open for later phases

These are the places where the next phases attach, and the reason the
architecture is shaped as it is:

| Seam | Phase | What plugs in |
|---|---|---|
| `LLMProvider` | 2 | A real model. `tool_schema_from_registry()` already describes the registry in tool-calling shape. |
| `SpeechListener` / `SpeechSpeaker` | 2 | Real engines. Both are already language-aware in their signatures. |
| `WakeWordDetector` | 2 | Any engine. `supports_phrase()` lets Settings tell the user whether their chosen phrase is detectable *before* they save it. |
| `Plan.steps` | 2 | Multi-step plans. Already a sequence; the engine already loops. |
| `ActionSpec` | 3+ | New capabilities, no engine changes. |
| Migration ladder | 3+ | New tables alongside the features that need them. |
| `Palette` | any | New themes. |
| `MESSAGES` | 2 | New languages. |

---

## 6. Things deliberately not done

- **No plugin system.** Loading third-party action modules would undermine the
  allow-list, which is the whole security argument.
- **No shell action, at any risk level.** Registering "run a command" with a
  confirmation prompt would reduce the entire model to a habit of clicking Yes.
- **No embedded browser.** URLs go to the user's own browser, with its own
  sandbox, extensions and settings.
- **No pre-created tables for unbuilt features.** See above.
- **No cloud sync of configuration.** Everything is local. There is no account,
  no telemetry and no network call AARMA makes on its own behalf.
