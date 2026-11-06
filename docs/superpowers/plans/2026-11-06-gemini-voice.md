# Gemini Voice Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Transcribe speech with Gemini and speak Hermes replies through the speakers, with local Whisper still available behind `--local`.

**Architecture:** Local speech detection stays. Finished utterances go to Gemini as one inline WAV request. Finished replies are cleaned for speech, streamed from Gemini TTS, and played. The mic drops frames while the app talks, so it never hears itself. Hermes stays the agent.

**Tech Stack:** Existing stack plus the Gemini Interactions REST API through httpx. No new dependencies.

**Spec:** Approved in conversation; the decisions are below.

## Decisions

- Default mode is cloud. `--local` uses Whisper and no spoken replies, so nothing goes to Google.
- Transcribe with `gemini-3.5-transcribe`. Speak with `gemini-3.8-flash-lite-tts`, voice `Kore`.
- Read aloud plain text only. Code blocks and tables are skipped; the full reply stays on screen.
- Mic pauses while speaking. Esc cuts speech off. Voice cannot interrupt.

## Global Constraints

- The Gemini key comes from `GEMINI_API_KEY`, else `~/.config/hermes-voice/.env`. It travels only in the `x-goog-api-key` header, never a URL. Never print, log, or write it. Never read that file's contents yourself.
- All httpx clients use `trust_env=False`.
- Audio is never written to disk.
- Do not contact Gemini, Hermes, or the microphone from tests.
- macOS and Linux behave the same.

## Gemini API Contract

Verified with live calls on 2026-11-06.

Base URL `https://generativelanguage.googleapis.com/v1beta`. Both calls are `POST /interactions`.

Transcribe request:
```json
{"model": "models/gemini-3.5-transcribe",
 "input": [{"type": "audio", "data": "<base64 WAV>", "mime_type": "audio/wav"}],
 "generation_config": {"transcription_config": {"custom_vocabulary": ["Hermes"]}}}
```
Response: `{"status": "completed", "steps": [{"type": "model_output", "content": [{"type": "text", "text": "..."}]}]}`

Speech request:
```json
{"model": "gemini-3.8-flash-lite-tts", "stream": true,
 "input": [{"type": "user_input", "content": [{"type": "text", "text": "..."}]}],
 "response_format": {"type": "audio", "mime_type": "audio/l16", "sample_rate": 24000},
 "generation_config": {"speech_config": [{"voice": "Kore"}]}}
```
Response is SSE (`text/event-stream`) with `event:` and `data:` lines. Audio arrives as `data: {"event_type": "step.delta", "delta": {"type": "audio", "data": "<base64>", "mime_type": "audio/l16", "sample_rate": 24000, "channels": 1}}`. Other event types are `interaction.created`, `interaction.status_update`, `step.start`, `step.stop`, `interaction.completed`. Audio is 24 kHz mono 16-bit little-endian PCM.

## Review Focus

1. **The app hearing itself:** frames during speech, and for 300 ms after, must never reach the segmenter or Hermes.
2. **Reply arrives while the user is mid-sentence:** speech waits for them to finish, up to 30 s, so their words are not cut off.
3. **Bad key or no network:** a plain log line, no crash, and the app keeps listening. A failed reply speech never blocks the next one.
4. **Reply that is only code:** no speech and no error.
5. **Esc and quit during speech:** sound stops at once, and the mic returns to normal.

---

### Task 1: Gemini speech in and out

**Files:**
- Create: `src/hermes_voice/gemini.py`, `tests/test_gemini.py`
- Modify: `src/hermes_voice/audio.py`, `src/hermes_voice/app.py`, `src/hermes_voice/__main__.py`, `README.md`, `tests/test_audio.py`, `tests/test_app.py`, `tests/test_cli.py`

**Interfaces:**
- In `hermes_voice.gemini`:
  - `GeminiError(Exception)`: message shown to the user as-is. Includes the API's error message when present, never the key.
  - `load_gemini_key(env_path: Path | None = None) -> str`: raises `GeminiError` naming the file and saying a key comes from aistudio.google.com/apikey, and that `--local` needs none.
  - `speakable(text: str) -> str`
  - `GeminiTranscriber(api_key: str, model: str = "gemini-3.5-transcribe", transport=None)`
    - `.load() -> None`: `GET /models/gemini-3.5-transcribe` to check the key; raises `GeminiError` on failure.
    - `.transcribe(pcm: bytes) -> str`
  - `GeminiSpeaker(api_key: str, model: str = "gemini-3.8-flash-lite-tts", voice: str = "Kore", transport=None)`
    - `.stream(text: str) -> Iterator[bytes]`
    - `.say(text: str, stop: threading.Event) -> None`: `play(self.stream(text), stop)`.
- In `hermes_voice.audio`: `play(chunks: Iterable[bytes], stop: threading.Event) -> None`
- In `hermes_voice.app`: `VoiceApp(..., speaker=None)`, where `speaker` has `.say(text, stop)`. New status `speaking`.
- New CLI flags: `--local`, `--tts-model` (default `gemini-3.8-flash-lite-tts`), `--voice` (default `Kore`). `--stt-model` default becomes `gemini-3.5-transcribe`, or `small.en` with `--local`.

- [ ] **Step 1: Write the failing tests**

`tests/test_gemini.py`, using `httpx.MockTransport`:
```python
def test_transcribe_sends_inline_wav_with_key_in_header_only():
    # request URL has no key; x-goog-api-key is set; body model == "models/gemini-3.5-transcribe"
    # input[0].mime_type == "audio/wav"; base64-decoded data parses with `wave` as 16 kHz mono 16-bit
    # with frames equal to the pcm; custom_vocabulary == ["Hermes"]

def test_transcribe_joins_model_output_text():
    # two text parts in a model_output step, one step of another type ignored -> "hello there"

def test_transcribe_retries_once_on_transport_error_then_succeeds(): ...
def test_transcribe_retries_once_on_503_then_raises_gemini_error(): ...
def test_transcribe_bad_key_raises_gemini_error_without_the_key_in_message(): ...

def test_stream_yields_decoded_audio_and_ignores_other_events():
    # SSE body with interaction.created, two audio step.delta, one non-audio step.delta, interaction.completed

def test_stream_sends_voice_and_model(): ...
def test_stream_error_status_raises_gemini_error(): ...
def test_load_checks_the_key(): ...  # 400 response -> GeminiError
def test_clients_ignore_proxy_settings(): ...  # trust_env is False on both

@pytest.mark.parametrize("text,expected", [
    ("**Done** with `git status`.", "Done with git status."),
    ("# Title\n\nBody text.", "Title\n\nBody text."),
    ("See [the logs](http://x.test/a) now.", "See the logs now."),
    ("- one\n- two", "one\ntwo"),
    ("Before\n```py\nprint(1)\n```\nAfter", "Before\n\nAfter"),
    ("| a | b |\n|---|---|\n| 1 | 2 |\nAfter", "After"),
    ("```\ncode only\n```", ""),
])
def test_speakable(text, expected): ...

def test_speakable_caps_long_text_at_a_sentence_end():
    # 400 sentences -> len <= 1300, ends with "The rest is on screen."

def test_key_comes_from_env_then_file(): ...
def test_missing_key_names_the_file_and_local_flag(): ...
```

`tests/test_audio.py`, with a fake `sounddevice` holding a recording `RawOutputStream`:
```python
def test_play_writes_chunks_at_24khz_then_drains():  # samplerate 24000, stop() called, closed
def test_play_holds_back_an_odd_trailing_byte():     # chunks b"abc", b"d" -> written b"ab", b"cd"
def test_play_aborts_when_stop_is_set():             # abort() called, later chunks not written, closed
def test_play_closes_the_stream_when_chunks_raise(): ...
```

`tests/test_app.py`, with a `FakeSpeaker` that records text and can hold until released:
```python
def test_completed_reply_is_spoken_after_cleanup(): ...         # spoke == ["Done"] for output "**Done**"
def test_reply_with_only_code_is_not_spoken(): ...
def test_no_speaker_means_no_speech(): ...
def test_status_is_speaking_during_speech(): ...
def test_frames_during_speech_never_reach_the_segmenter(): ...
def test_frames_in_the_300ms_after_speech_are_dropped(): ...
def test_speech_waits_until_the_user_stops_talking(): ...
def test_esc_cuts_speech_before_it_stops_the_run(): ...         # first Esc sets stop, c.stops == []; next Esc stops run
def test_failed_speech_logs_and_next_reply_still_speaks(): ...
def test_quit_stops_speech(): ...
```

`tests/test_cli.py`:
```python
def test_default_mode_builds_gemini_clients(): ...
def test_local_uses_whisper_and_no_speaker_and_needs_no_key(): ...
def test_missing_gemini_key_exits_1_with_the_message(): ...
def test_voice_and_model_flags_reach_the_speaker(): ...
```

Run: `uv run pytest -q`
Expected: new tests FAIL.

- [ ] **Step 2: Implement `gemini.py`**

- WAV: stdlib `wave` into `io.BytesIO`.
- `httpx.Client(trust_env=False, timeout=30, transport=transport)`.
- `transcribe`: one retry on `httpx.TransportError`, 429, and 5xx. Then `GeminiError`.
- `stream`: `client.stream("POST", ...)`, read `data:` lines, skip lines that are not JSON, yield `base64.b64decode(delta["data"])` when `delta.type == "audio"`. Status >= 400 raises `GeminiError`; read the body first so the message is available.
- `speakable`: remove fenced code blocks and table rows, unwrap links, strip `#`, `*`, `_` emphasis, backticks, and list markers. Collapse 3+ newlines to 2 and strip. Cap at 1200 characters, cut at the last `.`, `!`, or `?`, and append ` The rest is on screen.` Mark the cap with a `ponytail:` comment. The cap bounds cost and listening time.

- [ ] **Step 3: Implement `play` in `audio.py`**

Use the existing lazy `sounddevice` helper. `RawOutputStream(samplerate=24000, channels=1, dtype="int16")`. Write whole frames, holding back an odd trailing byte. On `stop` set, call `abort()` and return. Normal end calls `stop()`, which drains. Always `close()`.

- [ ] **Step 4: Add speech to `VoiceApp`**

- A third daemon worker `hermes-speak`, started when `speaker` is set, reading a queue with short timeouts and checking the existing stop flag.
- `run.completed`: put `speakable(output)` on the queue when non-empty.
- Before speaking: wait while `segmenter.in_speech`, up to 30 s, checking stop.
- Then under `_audio_lock`: set `_speaking = True` and reset the segmenter. Clear the speech stop event. Sync status.
- Call `speaker.say`. Any exception is logged and the worker continues.
- Afterwards: wait 300 ms, then under the lock reset the segmenter and set `_speaking = False`. Sync status.
- Capture loop: while `_speaking`, read and discard frames before the segmenter, the level meter, and the silent-mic counter. Show level 0.
- Status priority: `mic lost`, `muted`, `approval`, `offline`, `speaking`, `thinking`, `loading`, `transcribing`, `hearing`, `listening`. Give `speaking` its own colour.
- `action_stop`: if speaking, set the speech stop event, empty the queue, and return. Otherwise stop the run as today.
- Quit sets the speech stop event first.

- [ ] **Step 5: Update `__main__.py`**

Without `--local`: `load_gemini_key()` before the UI; on `GeminiError` print and exit 1. Build `GeminiTranscriber` and `GeminiSpeaker`. With `--local`: `Transcriber` and `speaker=None`.

- [ ] **Step 6: Update `README.md`**

Cover the Gemini key setup, `--local`, `--tts-model`, `--voice`, and Esc cutting speech. Replace the privacy note with:

> By default, finished utterances are sent to Google for transcription, and reply text is sent to Google to be spoken. In open-mic mode that includes anything said in the room. Silence is never sent. Transcripts also go to Hermes and its model provider. With `--local`, nothing goes to Google and audio stays on this computer.

- [ ] **Step 7: Run tests and commit**

Run: `uv run pytest -q`
Expected: all passed, no warnings.

```bash
git add .
git commit -m "feat: add Gemini transcription and spoken replies"
```
