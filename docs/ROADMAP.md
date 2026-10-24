# Roadmap

Phase 1 is complete. Each later phase lists what it adds, the seam it attaches
to, and what it must not break.

---

## Phase 1 — Foundation ✅

Repository structure, configuration, logging, identity, language layer,
permissions, action registry, risk model, confirmation flow, passphrase
security, filesystem sandbox, intent engine, planner, 21 actions, SQLite
persistence, connectivity, text interface, desktop dashboard, first-run wizard,
792 tests, documentation.

---

## Phase 2A — Voice output ✅

**Delivered:** a two-tier text-to-speech subsystem — natural neural voices
online (10 languages), Windows voices offline, with explicit fallback rather
than silent downgrade. Voice profiles for Asma and Aariz, delivery controls
(style, speed, volume, pitch), a speech-safety filter that keeps credentials and
tool payloads out of audio, a worker-thread engine with stop/replace semantics,
and GUI voice settings with a live availability readout and Test voice.

Full detail and limitations: [`VOICE.md`](VOICE.md).

---

## Phase 2B — Voice input ✅

**Delivered:** microphone capture with device selection and adaptive silence
endpointing, on-device speech recognition (audio never leaves the machine or
touches disk), a confidence gate that refuses rather than guesses, visible
listening states, permission enforcement, cancellation, and Settings.
Transcribed text enters the ordinary engine, so spoken commands are audited
exactly like typed ones.

Full detail and limitations: [`SPEECH-INPUT.md`](SPEECH-INPUT.md).

---

## Phase 2C — Wake word

**Adds:** hands-free activation.

**Attaches to:** `WakeWordDetector`.

**Constraint.** A wake word means the microphone is open continuously, which is
a much stronger privacy claim than press-to-talk. Detection must run on-device,
the open microphone must be visible at all times, and the `microphone`
permission must gate it exactly as it gates Phase 2B.

---

## Phase 2D — AI

**Adds:** a real language model, tool calling, multi-step planning, Hindi
interface translation.

**Attaches to:** `LLMProvider`, `Plan.steps`, `MESSAGES`.

**Must not break:** the model proposes `ToolProposal`s and nothing else. It
enters at `AarmaEngine.execute`, exactly where a typed command does. Every check
in `docs/SECURITY.md` applies to model proposals unchanged. If the model is
unavailable, the rule engine still handles everything it handles today.

**New risk to design for:** prompt injection. Once AARMA can read a web page or
a document, text inside it may try to issue instructions. The registry, the
permission checks and the High-risk confirmation are the existing defences; the
addition is treating any content AARMA *reads* as data, never as instructions.

---

## Phase 3 — Desktop depth

**Adds:** local reminders with natural-language time parsing, a scheduler that
survives restarts, desktop notifications, keyboard and mouse automation, screen
understanding.

**Attaches to:** `actions/reminders.py`, `actions/keyboard.py`,
`actions/mouse.py`, a new database migration.

**Constraints.** Reminders must work fully offline; a reminder the user believes
was set but which never fires is worse than no reminder. Keyboard automation can
type into whatever window has focus — including a password box — so it needs the
`keyboard` permission (off by default), a focused-window check and careful risk
grading. Mouse automation waits for screen understanding: blind coordinate
clicking is not a safe approximation of seeing.

---

## Phase 4 — Memory and personalisation

**Adds:** explicit memories ("remember that my presentation is on Monday"),
learned preferences, task history, and the UI to view, edit and delete all of it.

**Attaches to:** `ai/memory.py`, a new migration.

**Constraint.** Build the management UI *first*. Storing things a user cannot
inspect or remove is how an assistant quietly accumulates data nobody can get
back out — which is why Phase 1 ships the short-term buffer and nothing more.
Memory is already a permission and is already revocable.

---

## Phase 5 — Web intelligence

**Adds:** research and summarisation, optional Google Calendar and Outlook
integration, optional email.

**Constraints.** Cloud integrations are opt-in, per-service, and revocable.
OAuth tokens go to the OS credential store, never to `config.json`. Calendar and
Email permissions exist and stay off by default. Sending anything on the user's
behalf is High risk and always confirmed.

---

## Phase 6 — Trust and persona

**Adds:** website risk analysis (URL structure, domain similarity,
typosquatting, TLS, redirects, reputation signals) and persona simulation mode.

**Constraints.** Risk analysis is **probabilistic and must say so**. AARMA never
says "this website is safe" — it says "no obvious risk detected" or "this looks
suspicious, because…". Persona mode is labelled as simulation in the interface;
it must never imply a real person is communicating. Persona data is
user-authored, user-visible and deletable.

---

## Phase 7 — Distribution

**Adds:** PyInstaller build, Inno Setup installer producing `AARMA-Setup.exe`,
GitHub Releases, update checking.

**Already in place:** code and user data are strictly separated, no path is
hardcoded, the icon and avatar are drawn at runtime so there are no assets to
bundle, and the dependency list is short and all-Python.

**Constraints.** No bundled secrets. No telemetry without explicit opt-in.
Uninstall must remove the application and offer to remove user data — never
silently delete it.

---

## Standing rules across phases

1. A phase ships only when its tests pass and its documentation is updated.
2. No feature claims to work before it does. Unimplemented means "reports that
   it is unimplemented", never fabricated data.
3. New capabilities are registered actions with honest risk grades.
4. Security checks are never bypassed for convenience, including for a model.
5. The user can always see, and switch off, what AARMA is allowed to do.
