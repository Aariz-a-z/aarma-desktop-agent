# Setup and operation

---

## Requirements

- Windows 10 or 11
- Python 3.10 or newer
- About 150 MB of disk space (mostly PySide6)

### Displays

AARMA adapts to the screen it finds. It has no preferred resolution and no
assumed display scaling.

| | Dashboard | Setup wizard |
|---|---|---|
| Comfortable | 1240 × 820 logical px | 980 × 680 logical px |
| Minimum | 720 × 460 logical px | 620 × 420 logical px |

Those are *logical* pixels, so the physical requirement falls as display
scaling rises — a 1920×1080 screen at 150% has a logical work area of about
1280 × 672.

On a work area smaller than the minimum, the minimum yields and content scrolls
rather than the window overflowing the screen. Verified down to a 640 × 336
logical work area. Both windows minimize, maximize, restore, resize and close
normally, and remember nothing about your monitor between runs — each launch
sizes itself against whatever screen it opens on.

If Windows text scaling ("Make text bigger") is raised, the whole interface
scales with it rather than clipping.

Other platforms: the engine, the text interface and the test suite run on macOS
and Linux. `close_application` is Windows-only and declares it; folder
resolution falls back to `~/Documents` and friends.

---

## Install

```bash
python -m venv .venv
```

```bash
.venv\Scripts\activate
```

```bash
pip install -r requirements.txt
```

```bash
python main.py --doctor
```

`--doctor` prints the Python version, the data locations, the number of
registered actions, and which optional components are installed. Run it first
whenever something behaves unexpectedly.

### Minimal install

The core engine needs nothing beyond the standard library:

```bash
python main.py --cli
```

Without PySide6 the dashboard is unavailable and says so. Without Pillow,
`take_screenshot` reports the missing component. Without Send2Trash,
`delete_item` **refuses** rather than deleting permanently.

---

## First run

The first launch opens the eight-step setup wizard: welcome, assistant,
language, voice, name, permissions, optional security, finish. Nothing is
written to disk until you press **Finish**, so backing out leaves your
configuration untouched.

`python main.py --cli` runs a compact equivalent in the terminal. Re-run either
at any time with `--setup`; everything is also editable later in Settings.

---

## Where AARMA keeps things

| What | Default location |
|---|---|
| Configuration | `%APPDATA%\AARMA\config.json` |
| Database | `%APPDATA%\AARMA\aarma.db` |
| Logs | `%APPDATA%\AARMA\logs\aarma.log` |
| Screenshots | `%APPDATA%\AARMA\screenshots\` |

Set `AARMA_HOME` to move all of it. `AARMA_HOME=./data` gives a fully portable
checkout that writes nothing outside the project folder:

```bash
set AARMA_HOME=./data
```

Everything AARMA writes is git-ignored.

---

## Running

| Command | Purpose |
|---|---|
| `python main.py` | Desktop dashboard |
| `python main.py --cli` | Text interface |
| `python main.py --setup` | Re-run first-run setup |
| `python main.py --doctor` | Environment report |
| `python main.py --log-level DEBUG` | Verbose logging for one run |
| `python main.py --version` | Version |

---

## Tests

```bash
python -m pytest
```

Verbose, or a single file:

```bash
python -m pytest -v tests/test_security.py
```

No test writes outside a pytest temporary directory, launches an application,
opens a browser or deletes a real file.

---

## Troubleshooting

**The dashboard will not start.** Run `python main.py --doctor` and check
whether PySide6 is installed. `pip install -r requirements.txt` fixes it;
`python main.py --cli` works meanwhile.

**"I don't currently have permission to do that."** Working as designed — the
capability is switched off. Settings → Permissions, toggle it, ask again. No
restart needed.

**"That needs an internet connection and you're currently offline."** Either
there is no connectivity, or Settings → Network mode is set to *Offline only*.

**"'x' isn't in my approved application list."** The catalogue is deliberately
short and excludes shells. Ask "what apps can you open" for the list. Extending
it means adding an entry to `APPLICATION_CATALOG` in `actions/applications.py`.

**"That path is outside the folders I'm allowed to use."** The sandbox is
Documents, Desktop, Downloads, Pictures, Music and Videos. Ask "which folders
can you access" for the resolved paths on your machine.

**AARMA says it is locked.** Security mode is on and it starts locked after
every restart. Type the passphrase as your first message. If you have lost it,
delete `config.json` — you lose your settings, not your files.

**A control looks cut off, or the window does not fit my screen.** It should not
— please report it with your resolution and Windows scaling setting. As a
workaround the window is freely resizable and maximizable, and every page
scrolls. To reproduce a different scaling factor without changing your display
settings:

```bash
set QT_SCALE_FACTOR=1.5 && python main.py
```

**Where do errors go?** `%APPDATA%\AARMA\logs\aarma.log` (or `$AARMA_HOME/logs/`).
Users get a sentence; the log gets the detail. Passphrases, tokens and API keys
are redacted from it.

---

## Uninstalling

Delete the repository folder and the data folder shown by `--doctor`. AARMA
writes nothing to the registry, installs no services, and makes no network calls
on its own behalf.
