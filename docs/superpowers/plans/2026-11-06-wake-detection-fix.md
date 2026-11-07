# Wake Detection Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** "Hey Hermes" wakes the app on a real laptop mic, and utterances end when the speaker stops.

**Root cause, measured on real mic recordings:**
- `webrtcvad` called 80% of frames speech in this room, so ten spoken lines became one 30 second utterance.
- `sherpa-onnx` gave 0 hits on real mic audio at every threshold, and 0 hits on clean speech at 10% volume.
- Synthetic files had digital silence, so tests never showed either problem.

**Architecture:** Silero speech detection, already bundled with faster-whisper, replaces `webrtcvad`. While asleep, finished utterances go to a small local Whisper model that looks for the wake word. Only an utterance with the wake word, or one spoken while awake, goes to the cloud transcriber. `sherpa-onnx`, `sentencepiece`, the model download, the ring buffer, and `webrtcvad` are removed.

**Tech Stack:** Existing stack, minus the removed packages. `onnxruntime` becomes a direct dependency.

## Decisions

- The promise is now: no audio or text leaves this computer until the phrase is heard. Audio does reach the segmenter and local Whisper while asleep.
- Wake model: `base.en`, chosen over `tiny.en`, which missed more phrases.
- The wake word is the last word of the phrase. It must appear within the first four words of a sentence, so Whisper's "Pay Hermes" and "Have a Hermes" still count, and "Hermes open the logs" works too.
- A matched utterance goes to the cloud as a whole, so the command keeps its first words. No ring buffer is needed.

## Global Constraints

- While asleep, no audio or text reaches Gemini, Hermes, or any network call.
- Audio is never written to disk.
- All tests run with no network, mic, Gemini, or Hermes, and do not read any `.env` file.
- Tests may load the real Silero model; it ships inside the faster-whisper package.
- macOS and Linux behave the same.

## Verified Facts

Silero asset: `<faster_whisper package dir>/assets/silero_vad_v6.onnx`. Working streaming use, 0.2 ms per frame:
```python
opts = onnxruntime.SessionOptions()
opts.inter_op_num_threads = 1
opts.intra_op_num_threads = 1
opts.log_severity_level = 4
session = onnxruntime.InferenceSession(path, providers=["CPUExecutionProvider"], sess_options=opts)
h = np.zeros((1, 1, 128), "float32"); c = np.zeros((1, 1, 128), "float32")
context = np.zeros(64, "float32")
# for each 512-sample float32 chunk x, scaled to +/-1:
out, h, c = session.run(None, {"input": np.r_[context, x][None, :], "h": h, "c": c})
context = x[-64:]
speech = float(np.squeeze(out)) >= threshold
```
Frames are 480 samples, so buffer to 512 and return the latest decision.

Wake transcription that worked, 0.5 to 0.9 s per utterance:
`model.transcribe(audio, beam_size=1, condition_on_previous_text=False, hotwords="Hermes", vad_filter=False)` with `WhisperModel("base.en", device="cpu", compute_type="int8")`. No `initial_prompt`, which risks inventing the phrase from noise.

## Review Focus

1. **Noisy room:** an utterance still ends about a second after speech stops, with steady background noise.
2. **Talk while asleep:** never reaches the cloud transcriber or Hermes, and does not delay a spoken reply.
3. **Phrase, pause, command:** the phrase alone opens the 8 second window, and the next utterance is sent.
4. **Next wake during cloud transcription:** "Hey Hermes" said while the last command is still transcribing is not lost.
5. **Wake model missing or failing to load:** a plain log line, `loading` stays up, nothing is sent, and it retries.

---

### Task 1: Replace detection

**Files:**
- Modify: `src/hermes_voice/audio.py`, `src/hermes_voice/wake.py`, `src/hermes_voice/stt.py`, `src/hermes_voice/app.py`, `src/hermes_voice/__main__.py`, `pyproject.toml`, `uv.lock`, `README.md`, `tests/test_audio.py`, `tests/test_wake.py`, `tests/test_stt.py`, `tests/test_app.py`, `tests/test_cli.py`
- Existing fixture: `tests/fixtures/hey_hermes.pcm`, 16 kHz mono int16 speech

**Interfaces:**
- In `hermes_voice.audio`: `SpeechDetector(threshold: float = 0.5)`, callable as `(frame: bytes) -> bool`, with `.reset() -> None`. `faster_whisper` and `onnxruntime` are imported in the constructor.
- In `hermes_voice.stt`:
  - `wake_match(text: str, phrase: str) -> str | None`: `None` when there is no wake word, else the text after it, which may be empty. Replaces `strip_wake`.
  - `Transcriber(model: str = "small.en", hotword: str | None = None)`: with `hotword`, passes `hotwords=hotword` and `vad_filter=False`, and no `initial_prompt`.
- In `hermes_voice.wake`: `WakeGate(window_s: float = 8, clock=time.monotonic)` with `.awake`, `.wake()`, `.touch()`, `.expire(busy: bool) -> bool`, `.sleep()`. `WakeError`, `WakeSpotter`, `ensure_model`, and `model_is_cached` are removed.
- `VoiceApp(..., wake: str | None = None, gate: WakeGate | None = None, wake_transcriber=None)`.
- CLI: remove `--vad-level` and `--wake-threshold`. Add `--vad-threshold` (float, default 0.5) and `--wake-model` (default `base.en`).

- [ ] **Step 1: Write the failing tests**

`tests/test_audio.py`:
```python
def _noise(seconds):  # seeded numpy gaussian noise as int16 bytes, rms about 0.007
def _feed(segmenter, pcm): ...  # pushes 960-byte frames, returns finished utterances

def test_detector_finds_real_speech(): ...          # fixture: True on some frames
def test_detector_ignores_silence_and_noise(): ...  # False on every frame of zeros and of _noise(3)
def test_detector_buffers_480_sample_frames(): ...  # any 960-byte frame works; no exception over 3 s
def test_detector_reset_clears_state(): ...
def test_utterance_ends_in_a_noisy_room():
    # _noise(1) + fixture mixed with noise + _noise(2) through Segmenter(SpeechDetector())
    # gives exactly one utterance, and it is under 5 seconds, not the 30 second cap
```

`tests/test_stt.py`:
```python
@pytest.mark.parametrize("text,expected", [
    ("Hey Hermes, open the logs", "open the logs"),
    ("hey hermes", ""),
    ("Hey Hermes.", ""),
    ("hey, hermes open the logs", "open the logs"),
    ("Hermes open the logs", "open the logs"),
    ("Pay Hermes, what time is it?", "what time is it?"),
    ("Have a Hermes", ""),
    ("Hey Hermès", ""),
    ("Thanks. Hey Hermes, go", "go"),
    ("hey hermes stop. hey hermes go", "go"),
    ("Hey Herman, can you open the door?", None),
    ("We should talk about the budget", None),
    ("We bought a very expensive Hermes scarf", None),
    ("open the hermescope", None),
])
def test_wake_match(text, expected):
    assert wake_match(text, "hey hermes") == expected

def test_wake_match_with_a_custom_phrase():
    assert wake_match("Computer, lights on", "computer") == "lights on"
    assert wake_match("Hey Jarvis lights on", "hey jarvis") == "lights on"
```

`tests/test_wake.py`: the window behavior of `WakeGate`: asleep at start, `wake`, `touch`, `expire` with a fake clock, `expire` blocked by `busy`, `sleep`.

`tests/test_app.py`, with a recording `FakeTranscriber` for each of `transcriber` and `wake_transcriber`:
```python
def test_status_is_waiting_while_asleep(): ...
def test_talk_while_asleep_never_reaches_the_cloud(): ...      # wake model heard it; cloud calls == 0; c.started == []
def test_wake_utterance_goes_to_the_cloud_and_strips_the_phrase(): ...  # c.started == ["open the logs"]
def test_wake_rings_the_bell(): ...
def test_phrase_only_opens_the_window_for_the_next_utterance(): ...     # cloud not called for the phrase-only utterance
def test_gate_closes_after_a_command(): ...
def test_next_wake_during_cloud_transcription_is_not_lost(): ...
def test_window_expiry_returns_to_waiting(): ...
def test_mute_closes_the_gate_and_drops_queued_audio(): ...
def test_speech_playback_closes_the_gate(): ...
def test_talk_while_asleep_does_not_delay_a_spoken_reply(): ...
def test_wake_model_failure_logs_and_sends_nothing(): ...
def test_no_gate_keeps_open_mic_behavior(): ...
```

`tests/test_cli.py`: default builds a gate and a `base.en` wake transcriber with hotword `Hermes`-cased from the phrase's last word; `--wake-model`, `--wake`, and `--vad-threshold` reach their constructors; `--open-mic` builds neither; `--local` still gates with the wake model.

Remove tests for deleted code. Run: `uv run pytest -q`. Expected: new tests FAIL.

- [ ] **Step 2: Update dependencies**

`uv remove sherpa-onnx sentencepiece webrtcvad-wheels` then `uv add onnxruntime`.

- [ ] **Step 3: Implement detection and matching**

`wake_match`: strip accents with `unicodedata` NFKD before matching. Key word is the phrase's last word. Pattern: `(?:^|[.!?]\s+)(?:[\w'’]+[\s,]+){0,3}?<word>\b\W*`, case-insensitive. Use the last match, and return the rest of the text, stripped of leading punctuation and spaces.

- [ ] **Step 4: Rewire `VoiceApp`**

- Capture: frames always go to the segmenter, whether awake or asleep. Remove the ring and spotter code. Show `hearing` only when awake or with no gate; asleep stays `waiting`.
- `_stt_loop`, for each utterance:
  1. Check the mute generation as today.
  2. With a gate that is asleep: `wake_transcriber.transcribe(pcm)`, then `wake_match`. `None` drops the utterance. Otherwise wake the gate under the lock, ring the bell on the UI thread, and sync status. An empty or `is_noise` remainder stops here, waiting for the command.
  3. Run the cloud `transcriber.transcribe(pcm)` and deliver as today. Check the mute generation again first.
- `on_transcript`: with a gate, drop the text if not awake. `wake_match` the text; use the remainder if it matched. If empty or `is_noise`, return and stay awake. Otherwise close the gate and send. Closing here does not drain `_utterances`, because queued audio now gets the local check. Mute, mic loss, and speech playback still drain it.
- `_gate_busy` and the wait before speech apply only when the gate is awake or absent, so room talk cannot delay a reply.
- Load `wake_transcriber` in the existing load step, with the same log-and-retry behavior.

- [ ] **Step 5: Update `__main__.py`**

Build `Segmenter(SpeechDetector(args.vad_threshold), end_silence_ms=args.end_silence_ms)`. Without `--open-mic`: `WakeGate()` and `Transcriber(args.wake_model, hotword=<last word of phrase, capitalized>)`.

- [ ] **Step 6: Update `README.md`**

Remove the wake model download and `--wake-threshold`. Document `--wake-model` and `--vad-threshold`. Replace the privacy note with:

> By default nothing leaves this computer until the wake phrase is heard. While waiting, a small Whisper model on this computer listens for the phrase. The utterance containing the phrase, and the next one if you pause, is then sent to Google for transcription. Reply text is sent to Google to be spoken. Transcripts also go to Hermes and its model provider. With `--open-mic`, every utterance is sent. With `--local`, nothing goes to Google.

- [ ] **Step 7: Run tests and commit**

Run: `uv run pytest -q`
Expected: all passed, no warnings.

```bash
git add .
git commit -m "fix: detect speech and the wake word on real microphones"
```
