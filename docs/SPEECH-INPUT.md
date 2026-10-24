# Speech input

How AARMA listens, what it does with the audio, and what it refuses to do.

> **This document describes the rebuilt pipeline.** The first speech-input build
> failed in real use — it worked once and then stopped, missed distant speech,
> and was defeated by ordinary room noise. The root causes were found,
> reproduced and fixed; they are recorded below rather than quietly patched,
> because the same mistakes are easy to make again.

---

## What went wrong the first time

Each cause was reproduced with a scripted audio stream before anything was
changed, and each now has a regression test in `tests/test_speech_pipeline.py`.

| Root cause | Symptom | Fix |
|---|---|---|
| **Noise calibration was poisoned by speech.** The first 250 ms of every recording was averaged as "room noise". Speaking immediately made the threshold `speech_level × 3.5`, which nothing could then exceed. | *"Works once, then fails."* Users who waited succeeded; users who spoke straight away never did. | No calibration in the capture path at all. Speech detection is neural. |
| **A fixed level floor of 0.006 RMS.** | Speech beyond ~1.5 m was silently ignored. | No absolute floor. |
| **Threshold was noise floor × 3.5.** In a room with any hum, that exceeded normal speech. | Failed in noisy rooms. | Neural detection is unaffected by a constant noise floor. |
| **No early exit.** | Every failure cost the full 15 s and felt like a hang. | Gives up ~6 s after the button if nobody has started speaking. |
| **The whole recording was sent to the recogniser.** | 4.5 s of mostly silence for 2 s of speech — slow, and a hallucination risk. | Speech is trimmed out before recognition. |
| **A new capture object per call.** `cancel()` acted on an instance that was never recording. | Pressing stop did nothing. | One stream, explicit lifecycle. |

Measured before and after, on identical scripted audio:

| Case | Before | After |
|---|---|---|
| Speaking immediately | **failed** (15 s) | works (2.6 s) |
| Speech at 1/10 amplitude ("2 m") | **failed** | works |
| Speech at 1/50 amplitude | **failed** | works |
| Room noise 0.01 RMS | **failed** | works |
| Room noise 0.04 RMS | **failed** | works |
| Silence | 15.8 s to report | 6.1 s to report |

---

## The pipeline

```
microphone → stream (negotiated rate, mono)
           → level metering
           → neural voice activity detection  ── every 250 ms
           → endpointing (speech end / no-speech / hard limit)
           → trim to speech, with padding
           → speech-to-text (language forced)
           → validation: script, repetition, confidence
           → text
                ↓
              the ordinary AARMA engine
                ↓
   intent → permission → risk → registry → action → spoken reply
```

The listening layer **executes nothing**. It produces text; the interface calls
`handle_text(text, source="voice")`, the same method the text box uses. There is
no second execution path, so every permission check, risk grade and confirmation
applies to a spoken command unchanged, and the audit trail records
`source=voice`.

---

## Voice activity detection

**Silero VAD v6**, which ships inside `faster-whisper` — no new dependency and
no extra download. It judges whether audio *sounds like speech* rather than
whether it is loud, which is what fixes distance and noise together.

Measured on this machine, on real speech embedded in silence:

- speech start detected within **30 ms** of truth — the first word survives;
- speech found at **every** noise level tested up to 0.05 RMS, where the energy
  threshold found none;
- silence and noise at 0.02 and 0.05 RMS correctly yield **no speech**;
- **7–18 ms** to scan a three-second window, cheap enough to run live.

Tuned away from Silero's defaults: threshold 0.35 (rather than 0.5, to hear
quieter speech), minimum utterance 120 ms (so single words like "stop"
register), and 300 ms of padding at both ends so neither the first nor the last
word is trimmed. The **Hearing sensitivity** slider moves the threshold between
0.60 and 0.20.

An energy detector remains as a fallback if the model is ever unavailable. It is
the approach that failed, and it is labelled as the weaker option rather than
presented as equivalent.

---

## Model selection — measured, not assumed

`tools/benchmark_stt.py` measures accuracy, latency and memory for each
candidate. Run on **this** machine: 2-core Intel i3-1115G4, 20 GB RAM, **no
CUDA GPU** (Intel UHD only, so GPU inference is not available).

### English (word error rate, lower is better)

| Model | clean | quiet | noisy | far | latency | RTF | RAM | download |
|---|---|---|---|---|---|---|---|---|
| tiny | 2% | 2% | 17% | 59% | 0.7 s | 0.18 | 310 MB | 75 MB |
| base | 2% | 2% | 15% | 36% | 1.1 s | 0.28 | 355 MB | 145 MB |
| **small** | **2%** | **2%** | **15%** | **23%** | **3.4 s** | **0.89** | **540 MB** | **484 MB** |
| medium | 2% | 2% | 2% | 15% | 11.0 s | 2.86 | 1504 MB | 1500 MB |

### Other languages (character error rate; *n/m* = phrases in the correct script)

| Model | Hindi | Kannada | Telugu | Tamil | Malayalam |
|---|---|---|---|---|---|
| tiny | 118% 0/3 | 131% 0/3 | 114% 0/3 | 32% 3/3 | 107% 0/3 |
| base | 107% 0/3 | 110% 0/3 | 111% 0/3 | 31% 3/3 | 100% 0/3 |
| **small** | **16% 3/3** | 110% 0/3 | 114% 0/3 | **8% 3/3** | 120% 0/3 |
| medium | 8% 3/3 | 59% 3/3 | 118% 2/3 | 10% 3/3 | 254% 0/3 |

### Why `small`

`medium` is more accurate — notably in noise (2% vs 15%) — but takes **11
seconds** per command on this CPU, an RTF of 2.86. That is not a voice
assistant. `base` is quick but loses distant speech (36% WER at range).

`small` is the only size that is simultaneously usable in latency (3.4 s),
competent at distance (23% WER), and able to handle Hindi and Tamil at all. It
loads in 1.9 s once cached, and is preloaded in the background at startup so the
first command is not the slowest.

All four remain selectable in Settings, with their measured latency shown, so
someone on a faster machine can choose `medium`.

> **These figures come from synthetic speech.** Reference audio is generated with
> the Phase 2A neural voices, which are cleaner and more consistent than a person
> in a room. Treat them as an **upper bound** and a way to compare models against
> each other — not as a prediction of your own accuracy.

---

## Languages

The configured **speech input** language is passed explicitly to the recogniser
every time. Automatic detection is deliberately not used, and the task is
transcription, never translation — Kannada speech should become Kannada text.

| Status | Languages | Evidence |
|---|---|---|
| **Verified** | English, Hindi, Tamil | Benchmarked at `small`: CER 1%, 16%, 8%, correct script every time |
| **Supported but failing** | Kannada, Telugu, Malayalam | Whisper answers in the wrong script at every size tested. Kannada improves at `medium` (CER 59%, correct script) but is still not usable |
| **Untested** | Urdu, Arabic, Spanish, French | Accepted by the model; not benchmarked here |

Kannada, Telugu and Malayalam do not silently produce wrong text: the script
check below rejects it, and the user is asked to try again or change language.

---

## Never fabricated

Four independent mechanisms, each measured rather than assumed.

**1. Voice activity detection.** Silence and noise produce no speech regions, so
there is nothing to transcribe and nothing to invent from.

**2. Confidence.** `exp(avg_logprob)` — the model's own mean per-token
probability, not a number invented to look reassuring. Where a provider gives no
confidence, the field stays `None` and the gate is skipped rather than fabricated.
Base threshold **0.35**, adjustable, and **scaled by utterance length** — see
[Short commands are not penalised](#short-commands-are-not-penalised).

**3. Script validation.** Asked for Kannada, the model returned Devanagari
("इगर समया एश्टूः") or Latin ("atility"). Output whose script disagrees with the
requested language is refused — it is transliteration or hallucination, not
transcription.

**4. Repetition detection.** Degenerate decoder output ("the the the the",
long runs of one character) is refused.

Every rejection produces an honest message and no command:

| Situation | What the user hears |
|---|---|
| Silence, noise, too short | "I didn't hear anything. Please try again." |
| Below the confidence threshold | "I couldn't understand that clearly. Please try again." |
| Wrong script for the language | "That didn't come through as Kannada. Please try again, or change the speech language in Settings." |
| Degenerate output | "I couldn't understand that clearly. Please try again." |
| Microphone failed | "I couldn't record from the microphone." |

---

## Short commands are not penalised

A user reported that pressing the microphone and speaking returned *"I couldn't
understand that clearly"* every time, while the microphone check said the device
was fine. The microphone was fine. **The confidence gate was discarding correct
transcriptions.**

`avg_logprob` is a *mean over tokens*. A short utterance carries few tokens, and
the first tokens of any decode are the least certain — so brief commands score
systematically lower than sentences even when they are transcribed perfectly.
Measured on `small` across four acoustic conditions:

| Words | Confidence observed | Correct at the low end? |
|---|---|---|
| 1–2 | 0.366 – 0.638 | yes, including 0.366 |
| 4 | 0.637 – 0.742 | yes |
| 7 | 0.699 – 0.717 | yes |

The old flat **0.45** gate therefore rejected correct readings of "Asma" (0.379),
"Help." (0.447) and "Open Chrome." (0.400), while passing every long sentence —
penalising precisely the short commands a voice assistant exists to handle. Even
where it did not reject, the margin was thin: "Help." at distance measured 0.469,
0.019 above the threshold.

`speech.listener.confidence_floor` now scales the requirement by length: ×0.55 at
one to two words, ×0.80 at three to four, full threshold beyond. The gate is
unchanged where a low score genuinely means trouble.

---

## Hearing its own name

Measured on `small` with the phrase set above: **"Asma" was transcribed as
"Asthma" in 8 of 8 attempts**, across all four acoustic conditions. So was
"Hello Asma." The assistant could not hear its own name.

The fix is to prime the decoder with the name via Whisper's `initial_prompt`.
Measured on the same 36 utterances:

| | Exact transcriptions |
|---|---|
| Without priming | 28 / 36 |
| With priming | **36 / 36** |

The other 28 utterances were unchanged, so the prompt recovers the name without
distorting ordinary commands.

The prompt carries **names only** — the assistant's name and the words of its
wake word, capped at 12 single-word tokens by
`speech.providers.whisper_local.build_prompt`. That limit is a safety property,
not a formality: a decoder prompt biases every word of the next decode, so
conversation history in it would make the model repeat itself, and user content
in it would put private text inside the recognition of whatever is said next in
the room. `condition_on_previous_text` stays `False` for the same reason.

---

## The assistant does not hear itself

Before the microphone opens, AARMA stops speaking and waits out a short
post-speech delay (0.25 s by default). Without this the microphone hears
*"Opening Chrome."* and transcribes the assistant's own reply as a command.

---

## Reliability

`tools/reliability_microphone.py` runs the lifecycle the way a person does.
Measured over 20 consecutive cycles on the real device:

```
median cycle   : 1.53s
first half avg : 1.53s     second half avg: 1.53s
latency drift  : +0.00s (OK)
leaked threads : 0         memory growth  : +0.0 MB
hard failures  : 0
```

Plus the awkward orderings, all passing: cancel while idle, cancel mid-capture,
capture immediately after a cancel, close while idle, and capture after close
(which reopens the device).

The stream is **reused** across recordings rather than reopened per command, and
closed explicitly on shutdown so no device is held.

---

## Privacy

Unchanged, and still the default:

- **Processed entirely on this computer.** Nothing is uploaded. There is no
  cloud fallback, and one will never be added silently.
- **Never written to disk by default** — no temporary WAV, no cache. The
  recordings folder is not even created unless you switch on **Save diagnostic
  audio**, which is off by default, keeps only the last 20 recordings, and
  uploads nothing.
- **Never logged.** The log records character counts, language, confidence and
  duration; never audio, never the transcription.
- The microphone is open only during a session and is closed when it ends.
- Where audio is processed is stated on the Settings screen.

---

## Permission

Requires the **microphone** permission, off by default. With it off the button
is disabled with an explanatory tooltip, and `start_listening()` returns `False`
without opening any device. Revoking it while running disables the control
immediately.

---

## Interface

| State | Microphone button | Indicator | Status pill |
|---|---|---|---|
| Idle | Normal, "Click to speak" | Hidden | Idle |
| Listening | Highlighted, "click to stop" | "Listening…" + level meter | Listening |
| Processing | Disabled | "Understanding…" | Understanding |
| Error / disabled | Tooltip explains why | Hidden | Idle |

Never colour alone. The level indicator appears **only** while capture is
running. Spoken turns are labelled `you · spoken` in the transcript, and a
marginal recognition reads `you · spoken, unclear` rather than being presented as
certain.

**Settings → Speech input** offers device, recognition quality (with measured
latency), maximum listening time, hearing sensitivity, required certainty, and
**Check microphone**.

### Microphone check

Two measurements — the room, then your voice — reported as four verdicts:
Microphone, Noise, Speech level, Clipping, with one line of advice. It changes
nothing: blind gain adjustment turns a quiet microphone into a loud noisy one,
and knowing *which* problem you have is the useful output.

### Advanced diagnostics

The microphone check answers *"is audio arriving?"*. This answers the next
question — *"what did AARMA do with it?"* — for the most recent attempt:

| Row | What it tells you |
|---|---|
| Microphone, Recording, Levels | Device, seconds captured vs. seconds of speech, RMS / peak / background, clipping |
| Voice detected | Whether the detector found speech at all |
| Model | Recognition quality, requested language, detected language when it differs |
| **Heard** | **The recogniser's raw output, before validation ran** |
| Certainty | Confidence, the threshold it had to clear, and the underlying `avg_logprob` |
| Decision | ACCEPTED or REJECTED, with the exact internal reason |
| Response time | End-to-end latency for the attempt |

**Heard** and **Decision** are the pair that matters. Together they separate
*"the recogniser misheard you"* from *"the recogniser was right and AARMA's own
rules threw it away"* — two failures that are indistinguishable from the outside
and have opposite fixes. The user still sees one friendly sentence; the exact
reason goes here and into the log.

### Run speech test

A real-voice accuracy test: nine phrases, spoken once each, scored against what
you actually said.

```
Expected          Detected        Confidence      Speech   Verdict
Hello Asma.       Hello Asma.     0.656 / 0.193   1.31s    ✓ PASS
Open Chrome.      open chrome     0.528 / 0.193   1.02s    ✓ PASS
Asma              Asthma          0.478 / 0.193   0.71s    ✗ REJECTED (misheard)
```

Case and punctuation are Whisper's formatting, not mishearing, so they are not
scored as errors. The five specified phrases are joined by four one- and two-word
commands, because short utterances score lowest and a test made only of sentences
would miss the failure this whole exercise began with.

The test **never executes what it hears**. It calls `listen_once()`, which
returns the result to the dialog without notifying the command pipeline — so
saying "Open Chrome." during a test opens nothing.

It reports two numbers, separately and with their denominators: how many phrases
became usable commands, and how many the recogniser got right regardless of
whether AARMA then accepted them. A gap between those two is a validation
problem, not a recognition problem.

---

## Known limitations

1. **Verified for English, Hindi and Tamil.** Kannada, Telugu and Malayalam fail
   and are rejected rather than guessed at.
2. **Benchmarks use synthetic speech** — an upper bound, not a prediction.
3. **`small` takes ~3.4 s per command** on a 2-core CPU. Faster machines can use
   `medium`; there is no GPU path on this hardware.
4. **First use needs an internet connection** for the one-time model download.
5. **No online recogniser** — deliberate.
6. **No wake word.** Listening is explicit.
7. **No streaming or partial results.** One utterance, then transcription.
8. **No automatic language detection.**
9. **Far-field is better, not solved.** 23% WER at low amplitude with `small`.
   Distance benchmarking with a real voice at 0.5/1/2/3 m has **not** been
   performed — see below.

---

## When AARMA reports no audio

`Settings → Speech input → Check microphone` answers five questions in order,
and reports each separately rather than collapsing them into one message:

1. **Windows permission** — Windows hands desktop apps a *silent stream* when
   microphone access is off, rather than failing, so this is checked explicitly.
2. **Capture** — did the stream open?
3. **Samples** — are they arriving, and are they non-zero?
4. **Noise / Speech** — measured levels.
5. **Overall** — with one line of advice.

**Levels are never printed as `0.0000`.** A microphone reading 0.000017 RMS is
*working and very quiet*; four decimal places rendered that as zero and made it
indistinguishable from a dead device. Small values are shown in scientific
notation, and a true zero reads `0 (exactly)`.

**Scan all devices** opens every input in turn and reports frames, RMS, peak and
non-zero count for each — so if the selected device is silent but another is
hearing you, the dialog names it.

### Device identity and recovery

Microphones are stored **by name**, not index. PortAudio indexes shift when
hardware is plugged in or removed, so a saved index can quietly come to mean a
different device. If the saved microphone disappears, AARMA falls back to the
Windows default and says so:

> The microphone 'Studio Condenser' is unavailable. I've switched to the Windows
> default microphone.

### Reset

`Reset speech input settings` restores the microphone, recognition quality,
sensitivity and timeouts to defaults. It does **not** touch the assistant's
identity, language, permissions, security or any stored data.

`Settings → Security → Reset all settings` restores every setting to first-run
defaults. It still does **not** delete the conversation history or activity log —
removing data is a separate, explicit action.

---

## Testing it yourself

Automated tests need no hardware:

```bash
python -m pytest tests/test_speech_input.py tests/test_speech_pipeline.py tests/test_speech_reliability.py
```

Reliability and benchmarking tools:

```bash
python tools/reliability_microphone.py --cycles 20
```

```bash
python tools/benchmark_stt.py --models base small
```

### With your own voice

```bash
python main.py
```

1. Settings → Permissions → enable **Microphone**.
2. Settings → Speech input → **Check microphone**. Stay quiet, then say
   "Hello Asma". Confirm Samples reads *Receiving audio* and Speech reads
   *Detected*. If Samples shows almost no signal, press **Scan all devices** —
   it will name any device that *is* hearing you — and pick that one from the
   dropdown.
3. Settings → Speech input → **Run speech test**. Say the nine phrases as
   prompted. This is the measurement; steps 4–8 are the things it cannot cover.
4. Conversation → click the microphone and say **"Open Chrome."**
5. Repeat **five times in a row**, speaking *immediately* after clicking — this
   is the case that used to fail.
6. Try **"What time is it?"**, then a longer sentence such as
   *"Open YouTube and search for Python tutorials."*
7. Step back to about 1 m, then 2 m, and repeat.
8. Click the microphone and say nothing: it must report *"I didn't hear
   anything"*, never a guessed command.
9. Click the microphone while AARMA is speaking: it should stop talking and
   listen, and must not transcribe its own words.

**If anything is refused,** open **Advanced diagnostics** and read the **Heard**
row. If it holds the right words, the recogniser was correct and the refusal came
from AARMA's own checks — press **Copy details** and that block says which check
and with what numbers.

> **Not verified by the developer.** A test with a human voice — including the
> far-field distances above and the accuracy target — has not been performed. I
> cannot speak, and the substitute (playing audio through the speakers into the
> microphone) is defeated by Windows' communications ducking, which drops speaker
> output by ~80% whenever a microphone stream is open. Everything reported here
> was measured either on scripted audio through the real capture and recognition
> code, or on the real device for lifecycle behaviour. **No accuracy figure in
> this document is a claim about real human speech**, and the ≥95% target for
> English short commands is stated as a target, not as a result.
