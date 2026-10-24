# Wake word

How AARMA decides which phrase wakes it, how a new one is taught, and the one
thing the whole design exists to prevent.

> **A wake-word model is trained for one phrase.** Renaming the assistant to
> Friday does not make it able to hear "Hey Friday". If the Aariz model stayed
> armed after that rename, the assistant would wake at *another person's name* —
> which is worse than not waking at all. Everything below follows from refusing
> to do that.

---

## The chain

    config.assistant.name              settings, persisted
      -> config.assistant.wake_word    derived by wake_phrase_for_name()
      -> normalise_wake_phrase()       trim / collapse / lowercase
      -> WakeWordRegistry              phrase -> provider
           |- shipped providers        models that come with AARMA
           `- CustomWakeWordStore      models the user enrolled
      -> detector                      or MissingWakeWordDetector
      -> mic -> onset gate -> stage 1 -> stage 2 -> wake event
      -> engine._run_voice_turn        listen_once -> handle_text -> speak
      -> resume wake listening

`config.assistant.wake_word` is the single source of truth. `core/engine.py`
names no detector class; adding a model does not touch it.

## The phrase is the name

You wake Friday by saying **"Friday"**. Names are what people actually call
each other, and requiring a prefix would make the assistant answer to something
nobody would say. `wake_phrase_for_name("Friday")` is `"friday"`,
`"My Assistant"` is `"my assistant"`.

One identity differs, and it is a fact about a model rather than a preference:
the shipped Phase-15 detector was trained, evaluated and frozen on
**"hey aariz"** and hears nothing else. Giving Aariz a bare-name phrase would
leave the only validated model in the project unreachable, so that one identity
keeps the phrase its model knows. The exception lives in one place —
`LEGACY_WAKE_PHRASES` in `config/settings.py` — and a future model trained on a
bare name simply does not appear there.

## Renaming by talking to it

    "Asma, from now on I'll call you Friday."
    "Got it. From now on I'm Friday. I haven't learned the Friday wake word
     yet, so voice wake-up is off until you teach me."

The intent rules all require the sentence to be *addressed to the assistant* —
"your name", "call you", "yourself". That is what separates an instruction from
a remark: "my friend's name is Friday" and "I met someone called Friday" match
nothing, because neither is about what to call the assistant. A second guard
rejects captures that are not plausibly names, so "I'll call you back" and
"I'll call you tomorrow" stay telephone calls.

The rename writes through `ConfigManager` like any other settings change, which
means the wake detector swaps, the window header updates and the change
survives a restart, all through machinery that already existed.

**Identity and model are separate, and after a rename they are usually out of
step.** Friday is the assistant; nothing has been trained to hear "Friday".
That is a valid state, not an error — and the previous identity's detector is
stopped, never left armed. An assistant called Friday that still woke to "Asma"
would be answering to a name it no longer has.


| Concern | Home |
|---|---|
| Name → phrase, phrase normalisation | `config/settings.py` |
| Phrase → detector | `speech/wakeword.py` |
| The two-stage detector | `speech/wakeword_cascade.py` |
| Enrolled models on disk | `speech/wakeword_store.py` |
| Recording, training, validating | `speech/wakeword_enrollment.py` |
| Runtime loop, identity following | `core/engine.py` |
| Console enrollment | `tools/enrol_wakeword.py` |
| Settings enrollment | `ui/wakeword_enrol.py` |

## The detector

Two stages, because one is either too deaf or too eager.

**Stage 1** is EfficientWord-Net (`resnet_50_arc`) at a frozen 0.70. It is
*few-shot*: it is not trained for a phrase, it is shown one. That is what makes
custom wake words possible at all — enrolling stage 1 is just embedding the
user's recordings.

**Stage 2** is a 77 k-parameter classifier trained from scratch: log-mel 40 →
conv 16/32/64/64 → mean+max temporal pooling → FC 128 → one logit. Stage 1
alone hears "Hey Aariz" 38 times in 40 but also fires on 21 of 120 other-name
clips; stage 2 removed all 21 while rejecting none of the true candidates.

Both thresholds for the shipped model are frozen from that evaluation
(`experiments/farfield_verifier/RESULTS.md`) and must not be re-tuned against
live behaviour. Custom models get their **own** calibrated threshold — 0.10 was
calibrated for one model of one phrase, and reusing it elsewhere would be a
guess dressed as a measurement.

## What happens with no model

`REGISTRY.create()` never returns `None` and never raises. An unknown phrase, or
a provider that fails to construct, yields a `MissingWakeWordDetector`, which
reports itself unavailable, says which model is missing, returns `False` from
`start()` and never invokes the callback. **There is no fallback to another
identity's model**, configured or otherwise.

AARMA ships as **Asma**, and no Asma model exists — so this is the
out-of-the-box state, not an edge case.

## Enrolling a custom wake word

    python main.py --enrol-wake-word
    python main.py --enrol-wake-word --name Friday --count 20

or Settings → Assistant → **Enrol wake word**. Both drive the same code; the
front ends prompt, `speech/wakeword_enrollment.py` decides.

It asks for 15 recordings by default and varies them on purpose — near, 0.5 m
and 1 m, at different volumes and speeds. Fifteen identical recordings teach the
model one delivery of one phrase, and it then fails the first time you say it
while walking away.

### The split, and why there are three parts

| Split | Used for |
|---|---|
| ~55% | training stage 2, and enrolling stage 1 |
| ~20% | choosing the threshold |
| ~25% | the quality gate, consulted once |

The middle one is not decoration. Training drives the loss to about 1e-4 on a
set this small: the model memorises its own examples and scores them 1.0, so a
threshold picked from them saturates at the top of the grid. Measured, on real
recordings, before this was fixed:

| Threshold chosen on | Threshold | Held-out recall | Margin |
|---|---:|---:|---:|
| training clips (memorised) | 0.95 | 2/4 | +0.001 |
| **calibration clips (unseen)** | **0.05** | **3/4** | **+0.516** |

### Judging a recording

Each recording is judged once, by `assess_clip`, and both front ends render the
same verdict — the console prints it, the dialog shows it, and neither decides
anything itself.

The first version of this check rejected anything peaking below an absolute
0.02, and on a low-gain microphone that meant *every* recording, forever, with
the unhelpful message "too quiet". It was wrong, and measurably so. Across the
320 wake-phrase recordings already in this machine's corpus, made by this same
recorder:

| | |
|---|---:|
| quietest usable clip, peak | **0.0141** |
| loudest *ambient* clip containing no speech, peak | **0.4716** |

Absolute level does not separate speech from silence. It measures how much gain
the sound card applies, which differs per machine and per Windows setting. So
every bound is now relative to the room measured at the start of the session,
and each is set below the **minimum** observed across those 320 recordings:

| Check | Measured minimum | Threshold | Why |
|---|---:|---:|---|
| peak ÷ room floor | 14.8 | **8.0** | the phrase must stand out from the room |
| SNR | 7.4 dB | **4.0 dB** | same test, in log space, when a floor reference exists |
| voiced speech | 0.92 s | **0.45 s** | the whole phrase, not half of it |
| peak (dead device only) | — | **0.002** | a silent room here measures ≈0.0001, so this catches a muted or absent input, not a quiet speaker |
| clipped samples | 0 of 320 | **>1% at ≥0.99** | distortion the model would otherwise learn |

Checks run most-diagnostic first, so a muted device is never reported as "too
quiet" and a clip with no speech in it is never reported as poor signal-to-noise.
Every rejection names something the user can act on.

Before prompting, `readiness()` samples the room once and refuses to start at
all if the device returns nothing, returns digital silence, or the room is too
loud — one clear sentence instead of fifteen rejected recordings.

### Negative data

Positives alone produce a model that learns "audio exists". The negatives are,
in order of value: the user's own recordings of *other* wake words, the 54
recorded distractors, then LibriSpeech (CC-BY 4.0) and MUSAN noise (CC-BY 4.0)
when a copy is already on the machine. Nothing is downloaded. If there is not
enough, enrollment **refuses to train** rather than producing something
plausible.

### The quality gate

Measured on recordings the model never saw, and reported with numerators and
denominators because the numbers are small.

| Check | Bar |
|---|---|
| `held_out_recall` | ≥ 70% of the held-back recordings |
| `other_names` | zero fires on other wake words |
| `ordinary_speech` | zero fires on the recorded distractors |
| `room_noise` | no worse than the shipped model, measured the same way |
| `margin` | median held-out score > 0.05 above the threshold |

Fail any one and **nothing is installed**; a previously working detector is left
exactly as it was.

**On the noise bar.** The first version used a plausible-looking 12 acceptances
per hour. The shipped Aariz verifier measures **98.1/h** under that same
harness, so that bar would have rejected the validated detector this project
already ships. It is now anchored to that measurement. The number is high
because it is the *verifier alone*: in the real cascade, stage 1 proposes about
1.3 candidates an hour, so almost none of those windows are ever scored — Phase
15 measured 280/h standalone against roughly 0.42/h in the cascade.

**Passing the gate is not a Phase-15 validation.** It means "not obviously
broken", which is the most a few minutes of enrollment can establish. The
cascade false-activation rate of an enrolled phrase is **unmeasured**.

## Storage

    <AARMA_HOME>/models/custom/hey_friday/
        verifier.pt      stage-2 weights (torch state_dict)
        reference.npz    stage-1 enrollment embeddings
        metadata.json    phrase, thresholds, validation, provenance

User data, not application files: they are recordings of one person's voice in
one room, they cannot be regenerated from the repository, and on a real install
the application directory may not be writable.

Two rules:

* **Paths are derived, never stored.** The directory comes from the phrase, so a
  metadata file cannot point the loader anywhere else, however it was edited.
  Metadata whose phrase disagrees with its folder is ignored.
* **Checkpoints are data, never code.** They load with `weights_only=True` and
  their tensor shapes, dtypes and parameter count are checked before use — at
  install *and* at every runtime load.

Re-enrollment stages the new model beside the old one and swaps atomically, so a
working detector is never destroyed by a failed or rejected replacement. The
version number increments.

## Adding a shipped model later

One registration in `speech/wakeword.py`:

```python
REGISTRY.register(WakeWordProvider(
    phrase="hey nova",
    factory=lambda **kw: CascadeWakeWordDetector(
        phrase="hey nova",
        model_path=_ROOT / "models" / "hey_nova_verifier.pt",
        reference_path=_ROOT / "models" / "hey_nova_reference.npz",
        **kw),
    label="Nova"))
```

Shipped providers win over enrolled ones for the same phrase: the validated
model is the one whose behaviour is known. Do not register a phrase without a
model evaluated the way Phase 15 was.

## Microphone ownership

One reader, always. The detector holds the device while listening; a voice turn
stops it before `listen_once()`; enrollment borrows it through
`engine.wake_paused()`, which restores the loop afterwards whether the caller
succeeded, failed or cancelled. Enrollment records through the same
`MicrophoneStream` and the same VAD, via `tools.wakeword_corpus.Recorder`.

The assistant does not hear itself: the detector is given the listening engine's
existing `_assistant_is_speaking` guard, and drops its buffer while the
assistant is talking so speech straddling the end of a reply cannot be stitched
into a false wake.

## Known limits

* **AMBIENT-GATE = PENDING** for the shipped model. One detection in 2.3611 h
  *is* the resolution floor of that corpus (0.4235/h), so ≤0.125/h can be
  neither confirmed nor refuted. ≥8 h of ambient audio recorded in the real
  environment is needed.
* **Cascade FA for enrolled phrases is unmeasured**, and the gate does not
  establish it.
* **Multi-speaker is unmeasured.** Every recording is one speaker.
* **Stage-1 licence is UNVERIFIED.** EfficientWord-Net 1.0.5 is Apache-2.0 in
  code and weights, but its distilled training corpus is hosted on private links
  and cannot be checked. Fine for local use; unresolved for redistribution, and
  it applies to enrolled models too, since they use the same stage 1.
* Enrollment takes about **8 minutes** on a 2-core i3, almost all of it training.
