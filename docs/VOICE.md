# Voice output

What AARMA sounds like, which technology produces it, and what that technology
can and cannot do. Phase 2A implements **speech output only** — the assistant
speaks, it does not listen.

---

## The two providers

| | Natural online voice | Windows built-in voice |
|---|---|---|
| Module | `speech/providers/edge.py` | `speech/providers/sapi.py` |
| Technology | Microsoft Edge neural voices (`edge-tts`) | Windows SAPI 5, including OneCore |
| Quality tier | **Neural** | **Standard** |
| Needs internet | Yes | No |
| Languages | All 10 in AARMA's catalogue | Whatever Windows has installed |
| Cost / account | None | None |

The engine prefers the neural voice whenever there is a connection, and falls
back to the Windows voice **with an explicit notice** — never silently. A user
who is told "using the standard offline voice because you're not connected"
understands what they are hearing; a user who is not simply concludes the
product sounds bad.

### Why these two

The requirement was a voice that is natural, warm and comfortable for long
listening. Nothing available offline on a stock Windows machine meets that bar:
this development machine has only David and Zira (classic SAPI) plus the
OneCore set (Mark, and the Indian English pair Heera and Ravi). They are clear
and perfectly usable, but audibly synthetic.

The Edge neural voices genuinely are natural, and they cover Hindi, Kannada,
Telugu, Tamil, Malayalam and Urdu — the exact roadmap in Phase 1's language
catalogue. That combination is why they were chosen over the alternatives:

- **pyttsx3** — a wrapper over the same SAPI voices, so no quality gain, plus
  known threading problems. SAPI is used directly instead.
- **Piper / Coqui (local neural)** — genuinely offline *and* natural, but needs
  a 60 MB+ model per voice and an ONNX runtime. A strong future addition for the
  offline tier; too heavy to be the Phase 2A default.
- **Azure Speech** — the same voices as Edge under a supported contract, but
  requires an account, a key and billing. The right answer for a commercial
  release.

### The caveat, stated plainly

`edge-tts` uses the endpoint that Microsoft Edge's Read Aloud feature uses. It
needs no key and no account, but it is **not a documented public API** and
Microsoft could change or restrict it. For a capstone and for personal use that
is a reasonable trade. For a commercial release, swap it for Azure Speech (the
same voices) or a local neural engine — which costs one file, because everything
else depends only on the `SpeechSpeaker` interface.

---

## Voices

AARMA stores an **abstract** voice id (`female_warm`, `male_calm`), never a
provider's voice name. The same saved configuration therefore survives a change
of provider, language or machine.

| Profile | Gender | Default for | Character |
|---|---|---|---|
| `female_warm` | female | **Asma** | Gentle, friendly, unhurried |
| `female_clear` | female | — | Brighter and more precise |
| `male_calm` | male | **Aariz** | Steady, low, measured |
| `male_bright` | male | — | Crisper, more energetic |

Resolved to concrete voices per language in `speech/voices.py`. English defaults
to Indian English (`en-IN-NeerjaNeural` / `en-IN-PrabhatNeural`) — AARMA's
assistants are Asma and Aariz and its language roadmap is Indian languages, so
that is the better default for its users than a US voice.

### Delivery styles

Warm, Clear, Calm, Professional, Energetic. A style adjusts rate and pitch, and
where a language offers more than one voice per gender it also selects between
them. **Most languages ship one voice per gender**, so for those the style is
prosody only — `ResolvedVoice.style_matched_voice` reports which happened rather
than implying a choice that was not made.

Adjustments are deliberately modest (±10% rate, ±2 semitones). Large prosody
shifts are what make synthetic speech tiring, which is the opposite of the goal.

---

## Languages

Neural voices exist for every language in AARMA's catalogue:

English, Hindi, Kannada, Telugu, Tamil, Malayalam, Urdu, Arabic, Spanish, French.

Offline coverage depends entirely on what Windows has installed — typically
English only. **AARMA never substitutes a different language.** Asked for
Kannada with no Kannada voice available, it says:

> There's no offline Kannada voice installed on this computer.
> Connect to the internet to use a natural Kannada voice, or install the
> language's speech pack in Windows Settings.

An English voice reading Kannada text is not a fallback; it is a different and
worse failure.

---

## What is never spoken

`speech/text.py` is the last boundary before audio, and it removes:

- **Credentials** — labelled (`password: …`, `api_key=…`), conversational
  ("the password is …"), and recognisable shapes (`sk-…`, `ghp_…`, `AKIA…`,
  JWTs, long hex digests). A reply containing one is replaced wholesale with
  *"I've left that on screen rather than reading it out."* — a partially
  redacted sentence is safe but meaningless to hear.
- **Tool payloads** — JSON blobs, code fences, log lines.
- **Full paths** — `C:\Users\me\OneDrive\Documents` becomes "Documents".

Speech leaves the machine through a speaker, where anyone in the room hears it
and no redaction is possible afterwards. This is the third independent filter on
the same data, after the logger and the audit trail.

It also makes speech listenable: multi-line output is reduced to its first line
(the summary), replies are capped at 3 sentences / 320 characters, `item(s)`
becomes "items", `→` becomes "then", and markdown is stripped. The full text
always remains on screen.

---

## Threading and control

Synthesis and playback run on one worker thread inside `SpeechEngine`:

- the interface never waits for audio;
- one worker means two replies can never talk over each other;
- `speak()` replaces the queue by default — a newer answer supersedes an older
  one rather than queueing behind it;
- `stop()` is safe from any thread and interrupts mid-utterance;
- listeners fire on the worker, and the GUI marshals them onto its own thread
  with a Qt signal (`SpeechBridge`).

**Escape stops the voice** anywhere in the dashboard. Being able to silence an
assistant instantly, without hunting for a control, is the difference between a
voice people leave switched on and one they turn off.

Playback uses Windows' MCI interface (`winmm.dll`), which decodes MP3 natively —
so the natural voice adds no codec to the eventual installer.

---

## Settings

Settings → Voice offers: voice profile, delivery style, voice source
(automatic / online / offline), speaking speed, volume, pitch, "speak replies
out loud", "speak reminders and alerts", and **Test voice**.

The status line always states what the user would actually hear —
`Asma · English · Natural · Online` — and, when unavailable, why and what to do
about it. Test voice says *"Hello, I'm Asma. I'm ready to help."* using the
**configured** name, so a renamed assistant introduces itself correctly.

---

## Known limitations

1. **The natural voice needs a connection.** Offline falls back to the Windows
   voice, which is honestly labelled *Standard*.
2. **`edge-tts` uses an undocumented endpoint** (see above).
3. **Offline non-English voices are usually absent.** AARMA reports this rather
   than substituting.
4. **Pitch is approximate.** SAPI applies it through XML markup and neural
   voices through a hertz offset; the two do not match exactly.
5. **No local neural option yet.** Piper is the obvious Phase 2B+ addition for
   offline natural quality.
6. **No caching.** Every utterance is synthesised fresh; repeated phrases cost a
   round trip.
7. **Windows only for offline speech.** The SAPI provider reports itself
   unavailable elsewhere; the neural provider is cross-platform.
8. **Voice input is not implemented.** No microphone, no wake word, no
   transcription. `speech/listener.py` and `speech/wakeword.py` still hold only
   interfaces.
