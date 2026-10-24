# Security

What AARMA protects against, how, and what it does **not** protect against.

---

## Threat model

AARMA is a desktop agent that runs with the user's own privileges. It is not a
sandbox and it does not try to defend the machine from its own user. What it
defends against is narrower and more realistic:

| Threat | Defence |
|---|---|
| A misunderstood command destroys something | Risk grading; High always confirms |
| A capability is used the user never agreed to | Per-capability permissions, denied by default where it matters |
| A path argument escapes the intended folder | `SafePathResolver`, refusal rather than sanitisation |
| A link opens something that is not a web page | Scheme allow-list (`http`, `https` only) |
| An action becomes arbitrary code execution | No action accepts a command line; no shells in the app catalogue |
| **A future language model proposes something destructive** | Registry allow-list, schema validation, permission check, confirmation — all applied to model proposals identically |
| A passphrase leaks | PBKDF2 storage, no plaintext anywhere, log redaction |
| An action runs unnoticed | Full audit trail with permission decisions |

### Explicitly out of scope

- Defending against a user who deliberately misuses their own machine.
- Privilege escalation, sandbox escape, or malware defence.
- Protecting data from other software running as the same user.
- Network-level attacks.

---

## The allow-list

`ActionRegistry` is the enforcement point. `require()` raises
`UnknownActionError` for any name that was never registered. There is no
fallback, no dynamic import and no "if it looks like a shell command, run it".

Three properties are tested:

1. Unregistered names cannot run — including the obvious ones a model might
   invent (`run_shell_command`, `execute_shell`, `eval_python`, `sudo`).
2. Undeclared parameters cannot be smuggled in. `spec.validate()` rejects keys
   the schema does not declare rather than dropping them silently.
3. No registered action's name contains `shell`, `command`, `exec`, `eval`,
   `powershell`, `cmd` or `script`.

`tests/test_registry.py`, `tests/test_engine.py::TestUnregisteredActions`.

---

## Permissions

15 capabilities, each independently switchable, read fresh on every command so a
toggle takes effect immediately.

**Granted by default:** Speaker, Browser, Internet, Launch applications, Files
and folders, Reminders, Memory.

**Denied by default:** Microphone, Camera, Screenshots, Keyboard control, Mouse
control, System control, Calendar, Email.

The dividing line is whether the capability can **observe the user** or
**control the machine**. Those require a deliberate decision.

Denials are recorded in `permission_audit` with the action that triggered them.

---

## Risk levels

| Level | Examples | Gate |
|---|---|---|
| **Low** | `get_time`, `open_url`, `list_files`, `take_screenshot` | Permission check |
| **Medium** | `create_folder`, `rename_item`, `move_item`, `close_application`, `set_assistant_name` | Permission check, audited |
| **High** | `delete_item` | **Always** confirmed |

`RiskPolicy.requires_confirmation` returns `True` for `HIGH` unconditionally.
There is no setting that disables it, and
`test_confirmation_floor_cannot_be_switched_off_by_settings` asserts that
relaxing every other security setting leaves it intact.

Enabling the passphrase and `require_passphrase_for_high_risk` adds a second
gate on top.

---

## Confirmation flow

1. The engine detects a High-risk action and parks it with `ConfirmationBroker`.
2. It returns `CONFIRMATION_REQUIRED` with an unguessable token. **Nothing has
   run.**
3. The interface presents the choice — inline card in the GUI, prompt in the CLI.
4. The user approves or declines; the engine re-executes with `pre_approved=True`
   or records the refusal.

Properties:

- Tokens are `secrets.token_urlsafe(12)` and single-use.
- Requests expire after 120 seconds, so an unanswered prompt cannot be approved
  later by an unrelated "yes".
- The user can ignore it entirely — nothing blocks.
- **Every other check still applies after approval.** Approving a deletion
  outside the sandbox still refuses at execution.

---

## Filesystem sandbox

`SafePathResolver` confines every path argument to the user's own content
folders: Documents, Desktop, Downloads, Pictures, Music, Videos.

Locations come from the Windows known-folder API (`SHGetKnownFolderPath`), not
from assuming `~/Documents`. On a profile with OneDrive Backup enabled — the
default on many machines — that assumption resolves to the wrong folder
entirely. This was confirmed during development: on the test machine, Documents
is `…\OneDrive\Documents`, and the naive path pointed somewhere else.

### What is refused

| Input | Result |
|---|---|
| `../escape`, `..\escape`, `..` | Refused — traversal |
| `sub/dir`, `sub\dir` in a *name* | Refused — separator in a name |
| `C:\Windows\System32` | Refused — outside allowed roots |
| `CON`, `NUL`, `COM1`, `LPT9.txt` | Refused — Windows reserved device names |
| `a<b`, `a>b`, `a:b`, `a"b`, `a\|b`, `a?b`, `a*b` | Refused — illegal filename characters |
| Names over 120 characters | Refused |

Refused, not cleaned up: silently rewriting `../secrets` to `secrets` would hide
an attempted escape instead of surfacing it.

Symlinks are resolved before the containment check, so a link inside the sandbox
pointing outside it does not widen the boundary.

### Executable files

`open_file` refuses roughly 24 executable extensions (`.exe`, `.bat`, `.cmd`,
`.ps1`, `.vbs`, `.js`, `.lnk`, `.reg`, `.msi`, `.jar`, `.hta`, …). Handing one
of those to the shell's default-open verb is equivalent to executing it, which
no Medium-risk action may do.

### Deletion

`delete_item` sends to the **Recycle Bin**, never a permanent unlink. If the
Send2Trash component is missing, AARMA refuses to delete rather than falling
back to permanent deletion.

---

## Applications

`open_application` never receives a command line. It receives a *name*, looks it
up in `APPLICATION_CATALOG`, and executes only the path it resolved itself, with
`shell=False` and an explicit argument vector.

**Shells and terminals are deliberately absent from the catalogue.** Launching
`cmd` or `powershell` would turn a Low-risk action into arbitrary code
execution.

`close_application` uses a fixed argument list with a whitelisted image name
taken from the same catalogue.

---

## URLs

Only `http` and `https` reach a browser. `file:`, `javascript:`, `data:`,
`vbscript:` and custom application schemes are refused — passing those to the
default handler is how "just open a link" becomes local code execution.

Search queries are `quote_plus`-encoded into a fixed template, so a query can
never alter the URL's shape.

---

## Passphrase

- **Algorithm:** PBKDF2-HMAC-SHA256
- **Iterations:** 210,000 (stored per record, so it can be raised later without
  invalidating existing passphrases)
- **Salt:** 16 random bytes per user
- **Comparison:** `hmac.compare_digest`
- **Minimum length:** 8 characters

The plaintext exists only inside `security/passphrase.py`, only long enough to
be hashed. It is never written to `config.json` — `test_plaintext_is_never_written_to_the_config`
reads the file back and asserts its absence — and never to a log.

Changing or disabling requires the current passphrase. Security mode starts
**locked** after a restart; while locked, the only input the engine accepts is
an unlock attempt.

---

## Logging

Every record passes through `SecretRedactingFilter`, which scrubs both the
message and the structured payload. Keys containing `password`, `passphrase`,
`secret`, `api_key`, `token`, `credential`, `authorization`, `salt` or `hash`
are replaced with `***REDACTED***`, as are inline patterns like
`password=hunter2` in free text.

Redaction is defence in depth — callers are still expected not to pass secrets
to the logger. Action parameters are redacted again before being written to the
audit trail.

Logs rotate at 2 MB, 5 backups, and stay local. Nothing is transmitted anywhere.

---

## Audit trail

`action_history` records: timestamp, session, action, redacted parameters, risk,
source, whether it was allowed, whether it succeeded, the message, and duration.

`permission_audit` records every denial with its reason.

Both are visible in the dashboard's Activity page and survive restarts.

---

## Reporting a vulnerability

This is a university capstone project, not production software. Open a GitHub
issue with reproduction steps. Do not include real credentials, API keys or
personal data in an issue or a screenshot.
