# Hermes Voice Agent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A terminal app for macOS and Linux that listens to the microphone continuously, transcribes each thing the user says on-device, sends it to the locally installed Hermes agent, and shows the conversation in a polished terminal UI.

**Architecture:** Microphone frames pass through a voice-activity segmenter, then local Whisper, then the Hermes Runs API on `127.0.0.1:8642`. Hermes streams events back over SSE into a Textual UI. One Hermes run is active at a time; anything said during a run is merged into the next one.

**Tech Stack:** Python 3.13, uv, Textual 8, sounddevice, webrtcvad-wheels, faster-whisper, httpx, python-dotenv, numpy, pytest. All versions resolved with binary wheels on macOS arm64, Linux x86_64, and Linux aarch64 on 2026-11-06.

**Spec:** No separate spec; the user skipped the design questions. The requirements are in "Requirements" below, and the choices made on the user's behalf are in "Decisions".

## Requirements

From the user:

- Runs in the terminal, on macOS and on Linux.
- Keeps listening while running.
- Gives what it hears directly to the Hermes installed on this computer, using the API or whichever service is best.
- Shows a UI in the terminal that looks good.

## Decisions

These were made for the user and are theirs to overrule.

| Decision | Choice | Reason |
|---|---|---|
| Hermes connection | Runs API (`POST /v1/runs` plus SSE) | Already running on port 8642. Unlike chat completions, it gives tool events, stop, and approvals. |
| What gets sent | Every utterance (open mic) | The user said "whatever it listens to". `--wake hermes` narrows this to utterances starting with the phrase. |
| Speech to text | faster-whisper `small.en`, on-device | No per-minute cost, and raw audio never leaves the Mac. |
| Replies | Shown on screen only | The request covers listening and handing off. Spoken replies would also need echo handling, so they are left out. |
| Conversation | One Hermes session per launch | Follow-ups keep context; a new launch starts clean. |
| Approvals | Keyboard only, `once` or `deny` | A voice "yes" from a TV must never approve a shell command. |
| Linux target | A machine with its own microphone and its own Hermes | Matches "installed on the computer". A headless server or SSH session has no local mic, so the app exits with a clear message. |

Hermes already has its own `/voice` continuous mode and "Hey Hermes" wake word (`website/docs/user-guide/features/voice-mode.md`, `wake-word.md`). They use push-to-start or a wake word and live inside the Hermes CLI. This project exists for the always-open mic and a dedicated listening UI.

## Global Constraints

- Connect only to `http://127.0.0.1:8642`, even though the gateway binds `0.0.0.0`.
- Read the key from `HERMES_API_KEY`, else `API_SERVER_KEY` in `$HERMES_HOME/.env` (default `~/.hermes/.env`). Never print, log, or write the key.
- Audio is 16 kHz mono int16 in 30 ms frames (480 samples, 960 bytes). Audio is never written to disk.
- Do not import from the Hermes install; its venv and Python differ. Talk to it over HTTP only.
- Do not change anything in the Hermes home directory.
- macOS and Linux behave the same. Platform-specific code is limited to `silence_alsa` and `MIC_SILENT_HINT`.
- Commit `uv.lock`; it is cross-platform, so both systems get the same versions.
- Tunable values stay CLI flags, because rooms and microphones differ: `--vad-level` (0-3, default 2), `--end-silence-ms` (default 1000), `--stt-model` (default `small.en`), `--device`, `--wake`.

## Hermes API Contract

Verified against Hermes v0.22.2 source in `gateway/platforms/api_server_runs.py`.

- Auth: `Authorization: Bearer <key>` on every call except `GET /health`.
- `POST /v1/runs` with `{"input": str, "session_id": str}` returns `{"run_id": str, "status": "started"}`. An unknown `session_id` is accepted and used as the session's id.
- `GET /v1/runs/{id}/events` is SSE. Frames are `id: <seq>` then `data: <json>`, and every payload has `event`, `run_id`, `timestamp`, `seq`. Lines starting with `:` are keepalives or the close marker.
- Events used:
  - `message.delta` (`delta`)
  - `tool.started` (`tool`, `preview`)
  - `tool.completed` (`tool`, `duration`, `error`, `preview`)
  - `approval.request` (`command`, `choices`, optional `request_id`)
  - `approval.responded`
  - `run.completed` (`output`)
  - `run.failed` (`error`)
  - `run.cancelled`
  - `run.interrupted`
- `POST /v1/runs/{id}/stop` returns `{"status": "stopping"}`.
- `POST /v1/runs/{id}/approval` takes `{"choice": "once"|"deny", "request_id": str}`, with `request_id` only when the event had one.
- A full concurrency cap answers 429.

## Linux Notes

Checked in a `python:3.13-slim` Linux container on 2026-11-06 unless marked otherwise.

- **PortAudio is a system package on Linux.** Without it, `import sounddevice` raises `OSError: PortAudio library not found`. Debian and Ubuntu need `libportaudio2`; Fedora and Arch need `portaudio`. The macOS wheel bundles it.
- **No input device:** `sd.query_devices(kind="input")` raises `PortAudioError: Error querying device -1`.
- **Textual draws on stderr** (`linux_driver.py`: `self._file = sys.__stderr__`). Anything a C library prints to stderr lands on top of the UI, and stderr must never be redirected while the app runs.
- **sounddevice silences stderr only during `_initialize()`.** ALSA and JACK can still print when a stream opens or overruns. This is known ALSA behaviour; it did not reproduce in the container, which has no audio hardware, so Task 6 confirms it on a real machine.
- **Raw `hw:` devices may reject 16 kHz.** The `default`, `pulse`, and `pipewire` devices resample. Not tested here. A rejected device surfaces as `MicError`, which points to `--list-devices`.

## UI

```
 HERMES VOICE   ● LISTENING            MacBook Air Microphone
 ▁▂▁▁▃▇█▅▂▁▁▁▁▂▁▁▁▁▁▂▅▇▆▃▁▁▁▁▁▁▁▂▁▁▁▁▁▁▃▂▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
──────────────────────────────────────────────────────────────
 You     check whether the orange housing build passed
   ⚙ terminal  gh run list --limit 1            1.2s
 Hermes  The latest build passed 12 minutes ago.

 You     open the logs
 Hermes  Pulling them up now▌
──────────────────────────────────────────────────────────────
 m Mute   esc Stop run   y Approve   n Deny   q Quit
```

Status values: `loading`, `listening`, `hearing`, `transcribing`, `thinking`, `approval`, `muted`, `offline`, `mic lost`. Each has its own colour. `approval`, `offline`, and `mic lost` must read clearly from across a room.

## Review Focus

1. **Speech during a long run** (TV, a meeting, the user continuing to talk): must not stack up dozens of runs. Everything said merges into one next run.
2. **Microphone unavailable:** macOS permission denial and a muted Linux source deliver silent frames with no error. Missing PortAudio, no input device, and a rejected sample rate raise errors. Each must produce a plain message with the fix, not a traceback or a UI that looks fine and hears nothing.
3. **Hermes down or restarted mid-run:** the app keeps listening, the spoken text is not lost, and a closed stream does not leave the UI stuck on `thinking`.
4. **Whisper inventing text from noise** ("Thank you.", "you"): must never reach Hermes.
5. **Approval while the mic is open:** saying "yes" must not approve. It becomes ordinary speech for the next run.

---

### Task 1: Project setup and speech segmenter

**Files:**
- Create: `pyproject.toml`, `.gitignore`, `src/hermes_voice/__init__.py`, `src/hermes_voice/audio.py`
- Test: `tests/test_audio.py`

**Interfaces:**
- Produces, in `hermes_voice.audio`:
  - Constants `SAMPLE_RATE = 16000`, `FRAME_BYTES = 960`.
  - `level(frame: bytes) -> float`: RMS scaled to 0.0-1.0.
  - `Segmenter(is_speech: Callable[[bytes], bool], end_silence_ms=1000, min_speech_ms=300, max_ms=30000, preroll_ms=300)`
    - `.push(frame: bytes) -> bytes | None`: returns a finished utterance, otherwise `None`.
    - `.reset() -> None`
    - `.in_speech: bool`
  - `MicError(Exception)`: its message is shown to the user as-is.
  - `MIC_SILENT_HINT: str`: chosen from `sys.platform`.
  - `silence_alsa() -> None`: no-op unless on Linux with `libasound` present.
  - `list_devices() -> str`
  - `Mic(device: str | int | None = None)`
    - `.check() -> None`: raises `MicError` if there is no usable input device.
    - `.start() -> None`: raises `MicError`. `.stop() -> None`
    - `.read(timeout: float = 0.1) -> bytes | None`
    - `.alive: bool`, `.name: str`

- [ ] **Step 1: Set up the project**

On Linux, install PortAudio first (`sudo apt install libportaudio2`). Run `git init`, then `uv init --lib --name hermes-voice --python 3.13`. Add runtime deps with `uv add textual sounddevice webrtcvad-wheels faster-whisper httpx python-dotenv numpy` and dev dep with `uv add --dev pytest`. Add `.venv/`, `__pycache__/`, `.pytest_cache/` to `.gitignore`.

Run: `uv run python -c "import webrtcvad, sounddevice, faster_whisper, textual; print('ok')"`
Expected: `ok`

- [ ] **Step 2: Write the failing tests**

```python
import sys
from types import SimpleNamespace as NS
import pytest
from hermes_voice.audio import FRAME_BYTES, Mic, MicError, Segmenter, level

SPEECH, QUIET = b"\x01" * FRAME_BYTES, b"\x00" * FRAME_BYTES

def seg(**kw):
    return Segmenter(lambda f: f == SPEECH, **kw)

def feed(s, frames):
    return [out for f in frames if (out := s.push(f)) is not None]

def test_utterance_ends_after_end_silence():
    # 20 frames speech (600 ms), then 34 quiet frames (1020 ms) closes it
    out = feed(seg(), [SPEECH] * 20 + [QUIET] * 34)
    assert len(out) == 1

def test_preroll_is_kept_so_first_word_is_not_clipped():
    s = seg(preroll_ms=90)
    out = feed(s, [QUIET] * 10 + [SPEECH] * 20 + [QUIET] * 34)
    assert len(out[0]) >= (3 + 20) * FRAME_BYTES

def test_short_blip_is_dropped():
    assert feed(seg(), [SPEECH] * 3 + [QUIET] * 34) == []

def test_long_speech_is_force_flushed_at_max():
    assert len(feed(seg(max_ms=3000), [SPEECH] * 101)) == 1

def test_reset_discards_partial_utterance():
    s = seg()
    feed(s, [SPEECH] * 20)
    s.reset()
    assert s.in_speech is False
    assert feed(s, [QUIET] * 34) == []

def test_level_is_zero_for_silence_and_positive_for_signal():
    assert level(QUIET) == 0.0
    assert 0.0 < level(b"\xff\x7f" * 480) <= 1.0

def test_missing_portaudio_gives_install_hint(monkeypatch):
    monkeypatch.setitem(sys.modules, "sounddevice", None)
    with pytest.raises(MicError, match="libportaudio2"):
        Mic().start()

def test_device_that_cannot_open_points_to_list_devices(monkeypatch):
    class PortAudioError(Exception):
        pass
    def refuse(**kw):
        raise PortAudioError("Invalid sample rate")
    fake = NS(PortAudioError=PortAudioError, RawInputStream=refuse, _terminate=lambda: None, _initialize=lambda: None)
    monkeypatch.setitem(sys.modules, "sounddevice", fake)
    with pytest.raises(MicError, match="--list-devices"):
        Mic("hw:0,0").start()
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/test_audio.py -v`
Expected: FAIL with `ImportError`

- [ ] **Step 4: Implement `level`, `Segmenter`, and `Mic` in `src/hermes_voice/audio.py`**

`Segmenter` keeps a `collections.deque` of preroll frames. Speech starts on the first speech frame. Count speech frames separately from total frames, so `min_speech_ms` measures actual speech. After a max-length flush, stay in the idle state.

`Mic` wraps `sounddevice.RawInputStream(samplerate=16000, channels=1, dtype="int16", blocksize=480, device=...)`. The callback does `put_nowait` into a `queue.Queue(maxsize=1000)` and drops the frame on `queue.Full`. `.stop()` closes the stream so the macOS orange mic indicator turns off. `.start()` calls `sd._terminate()` and `sd._initialize()` first, so a device that came back is seen. `.name` comes from `sd.query_devices(device, kind="input")["name"]`.

Import `sounddevice` inside one private helper, never at module level, because the import itself fails on Linux without PortAudio. The helper catches `OSError` and `ImportError` and raises `MicError` with the install command for Debian/Ubuntu (`libportaudio2`) and Fedora/Arch (`portaudio`). This also lets the test suite run on a machine with no audio stack. Convert `sd.PortAudioError` from `.start()` and `.check()` to `MicError`, naming `--list-devices`.

`.check()` calls `query_devices(device, kind="input")`. Its message says a local microphone is required, and that SSH sessions and headless machines have none.

`silence_alsa` passes a no-op `ctypes.CFUNCTYPE(None, c_char_p, c_int, c_char_p, c_int, c_char_p)` to `snd_lib_error_set_handler` in `libasound.so.2`. Keep the callback in a module global, or it is garbage-collected and crashes the process. Ignore `OSError` if the library is absent. Call it once from `Mic.__init__`.

`MIC_SILENT_HINT` is a one-line fix for silent audio. macOS: System Settings > Privacy & Security > Microphone, for the terminal app. Linux: check the input is not muted in sound settings, or choose another with `--device`.

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_audio.py -v`
Expected: 8 passed

- [ ] **Step 6: Commit**

```bash
git add .
git commit -m "feat: add project setup and speech segmenter"
```

---

### Task 2: Local transcription

**Files:**
- Create: `src/hermes_voice/stt.py`
- Test: `tests/test_stt.py`

**Interfaces:**
- Produces, in `hermes_voice.stt`:
  - `is_noise(text: str) -> bool`
  - `apply_wake(text: str, wake: str | None) -> str | None`: the text to send, or `None` to ignore it.
  - `join_segments(segments: Iterable) -> str`: takes objects with `.text` and `.no_speech_prob`.
  - `Transcriber(model: str = "small.en")`
    - `.load() -> None`
    - `.transcribe(pcm: bytes) -> str`

- [ ] **Step 1: Write the failing tests**

```python
from types import SimpleNamespace as NS
import pytest
from hermes_voice.stt import apply_wake, is_noise, join_segments

@pytest.mark.parametrize("text", ["", "  ", "Thank you.", "thanks for watching!", "you", "...", "[BLANK_AUDIO]", "(music)"])
def test_whisper_noise_is_dropped(text):
    assert is_noise(text)

@pytest.mark.parametrize("text", ["thank you for fixing the build", "stop the docker container"])
def test_real_requests_are_kept(text):
    assert not is_noise(text)

def test_no_wake_phrase_passes_everything():
    assert apply_wake("open the logs", None) == "open the logs"

@pytest.mark.parametrize("text", ["Hermes, open the logs", "hey hermes open the logs", "HERMES: open the logs"])
def test_wake_phrase_is_stripped(text):
    assert apply_wake(text, "hermes") == "open the logs"

def test_speech_without_wake_phrase_is_ignored():
    assert apply_wake("what's for dinner", "hermes") is None

def test_wake_phrase_alone_is_ignored():
    assert apply_wake("Hey Hermes.", "hermes") is None

def test_join_skips_likely_non_speech_segments():
    segs = [NS(text=" open logs", no_speech_prob=0.1), NS(text=" Thank you", no_speech_prob=0.9)]
    assert join_segments(segs) == "open logs"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_stt.py -v`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Implement `src/hermes_voice/stt.py`**

- `is_noise`: compare the lowercased text, stripped of punctuation, against a small frozenset of known Whisper outputs, and reject bracketed tags and text with no letters. Mark the list with a `ponytail:` comment naming its ceiling: known phrases only.
- `apply_wake`: regex `^\W*(hey\s+)?<escaped wake>\b\W*`, case-insensitive. Empty remainder returns `None`.
- `join_segments`: drop segments with `no_speech_prob > 0.6`.
- `Transcriber.load`: `WhisperModel(model, device="cpu", compute_type="int8")`.
- `Transcriber.transcribe`: convert with `np.frombuffer(pcm, np.int16).astype(np.float32) / 32768`, then call `.transcribe(audio, beam_size=1, condition_on_previous_text=False, initial_prompt="Hermes")`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_stt.py -v`
Expected: all passed

- [ ] **Step 5: Commit**

```bash
git add src/hermes_voice/stt.py tests/test_stt.py
git commit -m "feat: add local transcription with noise and wake filters"
```

---

### Task 3: Hermes client

**Files:**
- Create: `src/hermes_voice/hermes.py`
- Test: `tests/test_hermes.py`

**Interfaces:**
- Produces, in `hermes_voice.hermes`:
  - `RunEvent(name: str, data: dict)`: frozen dataclass.
  - `TERMINAL = {"run.completed", "run.failed", "run.cancelled", "run.interrupted"}`
  - Exceptions `HermesAuthError` (401/403) and `HermesUnavailable` (connection error, 429, 5xx).
  - `load_api_key(env_path: Path | None = None) -> str`: defaults to `$HERMES_HOME/.env`, else `~/.hermes/.env`. If no key is found, raises `HermesAuthError` naming the file and saying to set `API_SERVER_ENABLED=true` and `API_SERVER_KEY` there.
  - `HermesClient(api_key: str, base_url: str = "http://127.0.0.1:8642", transport: httpx.AsyncBaseTransport | None = None)`
    - `async start_run(text: str, session_id: str) -> str`
    - `async events(run_id: str) -> AsyncIterator[RunEvent]`
    - `async stop(run_id: str) -> None`
    - `async approve(run_id: str, choice: str, request_id: str | None = None) -> None`
    - `async aclose() -> None`

- [ ] **Step 1: Write the failing tests**

```python
import asyncio, json
import httpx, pytest
from hermes_voice.hermes import HermesAuthError, HermesClient, HermesUnavailable, load_api_key

def sse(*events):
    body = b": open\n\n"
    for i, e in enumerate(events):
        body += f"id: {i}\ndata: {json.dumps(e)}\n\n".encode()
    return body + b": stream closed\n\n"

def client(handler):
    return HermesClient("k" * 32, transport=httpx.MockTransport(handler))

def test_start_run_sends_input_session_and_bearer():
    seen = {}
    def h(req):
        seen.update(path=req.url.path, auth=req.headers["authorization"], body=json.loads(req.content))
        return httpx.Response(202, json={"run_id": "run_1", "status": "started"})
    assert asyncio.run(client(h).start_run("hi", "voice-1")) == "run_1"
    assert seen == {"path": "/v1/runs", "auth": "Bearer " + "k" * 32, "body": {"input": "hi", "session_id": "voice-1"}}

@pytest.mark.parametrize("status,exc", [(401, HermesAuthError), (429, HermesUnavailable), (503, HermesUnavailable)])
def test_error_statuses_map_to_exceptions(status, exc):
    with pytest.raises(exc):
        asyncio.run(client(lambda r: httpx.Response(status, json={})).start_run("hi", "s"))

def test_connection_refused_is_unavailable():
    def h(req):
        raise httpx.ConnectError("refused")
    with pytest.raises(HermesUnavailable):
        asyncio.run(client(h).start_run("hi", "s"))

async def collect(c):
    return [e async for e in c.events("run_1")]

def test_events_parse_and_stop_at_terminal():
    body = sse({"event": "message.delta", "delta": "He"}, {"event": "run.completed", "output": "Hello"})
    got = asyncio.run(collect(client(lambda r: httpx.Response(200, content=body))))
    assert [e.name for e in got] == ["message.delta", "run.completed"]
    assert got[1].data["output"] == "Hello"

def test_stream_closing_early_yields_interrupted():
    body = sse({"event": "message.delta", "delta": "He"})
    got = asyncio.run(collect(client(lambda r: httpx.Response(200, content=body))))
    assert got[-1].name == "run.interrupted"

def test_approve_omits_missing_request_id():
    seen = {}
    def h(req):
        seen.update(path=req.url.path, body=json.loads(req.content))
        return httpx.Response(200, json={})
    asyncio.run(client(h).approve("run_1", "deny"))
    assert seen == {"path": "/v1/runs/run_1/approval", "body": {"choice": "deny"}}

def test_key_file_with_inline_comments_parses(tmp_path, monkeypatch):
    monkeypatch.delenv("HERMES_API_KEY", raising=False)
    f = tmp_path / ".env"
    f.write_text("API_SERVER_HOST=0.0.0.0   # phone access\nAPI_SERVER_KEY=abc123   # gateway\n")
    assert load_api_key(f) == "abc123"

def test_missing_key_names_the_file(tmp_path, monkeypatch):
    monkeypatch.delenv("HERMES_API_KEY", raising=False)
    with pytest.raises(HermesAuthError, match=".env"):
        load_api_key(tmp_path / ".env")

def test_hermes_home_is_respected(tmp_path, monkeypatch):
    monkeypatch.delenv("HERMES_API_KEY", raising=False)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / ".env").write_text("API_SERVER_KEY=fromhome\n")
    assert load_api_key() == "fromhome"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_hermes.py -v`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Implement `src/hermes_voice/hermes.py`**

Use one `httpx.AsyncClient` with `timeout=httpx.Timeout(10, read=None)`; `read=None` is needed because runs can stay quiet for minutes between events.

`events` uses `client.stream("GET", ...)` and `aiter_lines()`. Parse only lines starting with `data: `. If the stream ends with no terminal event, or `httpx.TransportError` is raised mid-stream, yield `RunEvent("run.interrupted", {})`. Load the key with `dotenv_values`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_hermes.py -v`
Expected: all passed

- [ ] **Step 5: Commit**

```bash
git add src/hermes_voice/hermes.py tests/test_hermes.py
git commit -m "feat: add Hermes runs API client"
```

---

### Task 4: Terminal UI and conversation flow

**Files:**
- Create: `src/hermes_voice/app.py`
- Test: `tests/test_app.py`

**Interfaces:**
- Consumes: `HermesClient`, `RunEvent`, `TERMINAL`, `HermesAuthError`, `HermesUnavailable` (Task 3); `apply_wake` (Task 2).
- Produces, in `hermes_voice.app`:
  - `VoiceApp(client, session_id: str, wake: str | None = None, mic=None, transcriber=None, segmenter=None)`: a Textual `App`. Audio workers start only when `mic` is given (Task 5).
  - `.status: reactive[str]`, one of the values in the UI section.
  - `.on_transcript(text: str) -> None`: called on the UI thread.
  - `.set_level(value: float) -> None`: feeds the meter.
  - `.log_line(text: str) -> None`: appends a system line to the conversation.

- [ ] **Step 1: Write the failing tests**

Add a `FakeClient` to `tests/test_app.py`, with the same async methods as `HermesClient`. It records `started: list[str]`, `approvals: list[tuple]`, and `stops: list[str]`. It plays scripted `RunEvent` lists, holds a run open until `release()` is called, and can raise `HermesUnavailable` from `start_run` a set number of times.

```python
def run(coro_fn):
    asyncio.run(coro_fn())

def test_speech_during_run_merges_into_one_next_run():
    async def go():
        c = FakeClient(hold=True)
        app = VoiceApp(c, "voice-test")
        async with app.run_test() as pilot:
            app.on_transcript("first"); await pilot.pause()
            app.on_transcript("second"); app.on_transcript("third"); await pilot.pause()
            assert c.started == ["first"]
            c.release(); await pilot.pause()
            c.release(); await pilot.pause()
            assert c.started == ["first", "second\nthird"]
    run(go)

def test_offline_hermes_keeps_text_and_sends_it_later():
    async def go():
        c = FakeClient(fail_starts=1)
        app = VoiceApp(c, "voice-test", retry_seconds=0.01)
        async with app.run_test() as pilot:
            app.on_transcript("do not lose this"); await pilot.pause()
            assert app.status == "offline"
            await asyncio.sleep(0.05); await pilot.pause()
            assert c.started == ["do not lose this"]
    run(go)

def test_interrupted_stream_returns_to_listening():
    async def go():
        c = FakeClient(script=[RunEvent("run.interrupted", {})])
        app = VoiceApp(c, "voice-test")
        async with app.run_test() as pilot:
            app.on_transcript("hi"); await pilot.pause()
            assert app.status == "listening"
    run(go)

def test_saying_yes_cannot_approve_but_y_key_can():
    async def go():
        c = FakeClient(hold=True, script=[RunEvent("approval.request", {"command": "rm x", "choices": ["once", "deny"], "request_id": "a1"})])
        app = VoiceApp(c, "voice-test")
        async with app.run_test() as pilot:
            app.on_transcript("clean up"); await pilot.pause()
            assert app.status == "approval"
            app.on_transcript("yes"); await pilot.pause()
            assert c.approvals == []
            await pilot.press("y"); await pilot.pause()
            assert c.approvals == [("run_1", "once", "a1")]
    run(go)

def test_y_does_nothing_without_a_pending_approval():
    async def go():
        c = FakeClient()
        app = VoiceApp(c, "voice-test")
        async with app.run_test() as pilot:
            await pilot.press("y"); await pilot.pause()
            assert c.approvals == []
    run(go)

def test_escape_stops_the_active_run():
    async def go():
        c = FakeClient(hold=True)
        app = VoiceApp(c, "voice-test")
        async with app.run_test() as pilot:
            app.on_transcript("long job"); await pilot.pause()
            await pilot.press("escape"); await pilot.pause()
            assert c.stops == ["run_1"]
    run(go)

def test_wake_mode_ignores_speech_without_the_phrase():
    async def go():
        c = FakeClient()
        app = VoiceApp(c, "voice-test", wake="hermes")
        async with app.run_test() as pilot:
            app.on_transcript("what's for dinner")
            app.on_transcript("hermes open the logs"); await pilot.pause()
            assert c.started == ["open the logs"]
    run(go)
```

`VoiceApp` therefore also takes `retry_seconds: float = 5`.

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_app.py -v`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Implement `VoiceApp` in `src/hermes_voice/app.py`**

Layout follows the UI section:
- Header row: `Static` title, `Static` status, `Static` device name.
- Level meter: Textual `Sparkline`.
- Conversation: `RichLog` for finished lines, plus a `Static` for the reply still streaming.
- `Footer` with bindings.

Styles live in the `CSS` class attribute, with one CSS class per status.

Flow:
- `on_transcript` applies `apply_wake`, drops `None`, writes the `You` line, appends to `self._pending`, and starts the send worker unless a run is active.
- The send worker loops while `_pending` is non-empty. It joins with `"\n"`, calls `start_run`, clears `_pending` only after that succeeds, then consumes `events`.
- On `HermesUnavailable`, set `offline`, keep `_pending`, and retry with `set_timer(retry_seconds, ...)`.
- On `HermesAuthError`, write a line telling the user to check `API_SERVER_KEY`, and stop sending.

Events:
- `message.delta`: append to the streaming `Static`.
- `tool.started`, `tool.completed`: write a dim tool line.
- `approval.request`: store it, show the command, set `approval`, call `self.bell()`.
- `approval.responded`, `tool.completed`, and any `TERMINAL` event: clear the stored approval.
- `run.completed`: write `output` as `rich.markdown.Markdown` and clear the `Static`.
- `run.failed`: write the error.
- `run.cancelled`, `run.interrupted`: write a line saying the run ended early.

Bindings: `m` mute (Task 5), `escape` stop, `y` approve once, `n` deny, `q` quit. `y` and `n` act only when an approval is stored. Close the client on quit.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_app.py -v`
Expected: all passed

- [ ] **Step 5: Commit**

```bash
git add src/hermes_voice/app.py tests/test_app.py
git commit -m "feat: add terminal UI and conversation flow"
```

---

### Task 5: Live audio pipeline and mute

**Files:**
- Modify: `src/hermes_voice/app.py`
- Test: `tests/test_app.py`

**Interfaces:**
- Consumes: `Mic`, `MicError`, `MIC_SILENT_HINT`, `Segmenter`, `level` (Task 1); `Transcriber`, `is_noise` (Task 2).
- Produces: audio workers inside `VoiceApp`. The public interface is unchanged.

- [ ] **Step 1: Write the failing tests**

Add a `FakeMic` with the `Mic` interface. It plays a list of frames, then returns `None`. Its `.alive` is settable and it counts `start()` and `stop()` calls. Add a `FakeTranscriber` that returns scripted text.

```python
def test_utterance_flows_from_mic_to_hermes():
    # FakeMic plays speech frames then silence; fake segmenter emits one utterance
    # asserts c.started == ["open the logs"]

def test_noise_transcript_never_reaches_hermes():
    # FakeTranscriber returns "Thank you."; asserts c.started == []

def test_all_zero_audio_warns_about_mic_permission_once():
    # FakeMic plays 150 zero frames; the log contains MIC_SILENT_HINT exactly once
    # asserts status stays "listening"

def test_mute_closes_the_mic_and_unmute_reopens_it():
    # press m: status == "muted", mic.stops == 1, partial speech discarded
    # press m: status == "listening", mic.starts == 2

def test_lost_mic_shows_status_and_recovers():
    # mic.alive = False: status == "mic lost"; mic.alive = True: status == "listening"

def test_mic_that_cannot_open_shows_the_reason_and_keeps_retrying():
    # FakeMic.start raises MicError("no such device") once, then works
    # status == "mic lost", log contains "no such device" once, then status == "listening"
```

Give each test the same `go()` shape as Task 4, with these assertions.

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_app.py -v`
Expected: the 6 new tests FAIL; Task 4 tests still pass.

- [ ] **Step 3: Add two thread workers to `VoiceApp`**

Start both in `on_mount` when `mic` is set. Worker threads must use `self.call_from_thread(...)` for all UI updates.

- **Capture worker:** starts the mic, then loops on `mic.read()`. Each frame goes to `segmenter.push`; finished utterances go to a `queue.Queue`. Update the meter about every 3 frames, and set `hearing` or `listening` from `segmenter.in_speech` unless a run or approval owns the status.
  - After 100 consecutive all-zero frames (3 s), write `MIC_SILENT_HINT` to the log once.
  - If `mic.alive` is false or `mic.start()` raised `MicError`, and the app is not muted, set `mic lost` and retry `mic.start()` every 2 s. Log a given error message only once.
  - After every successful `mic.start()`, call `self.refresh(repaint=True)` on the UI thread, which paints over anything a sound library printed while opening the device.
- **STT worker:** calls `transcriber.load()` while status is `loading`, then loops on the queue. It sets `transcribing`, calls `transcribe`, drops `is_noise` text, and passes the rest to `on_transcript`.

Mute calls `mic.stop()` and `segmenter.reset()`, then unmute calls `mic.start()`. Make `status` a derived value in one method, so overlapping sources cannot leave it stale. Priority order: `mic lost`, `muted`, `approval`, `offline`, `thinking`, `transcribing`, `hearing`, `listening`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest -v`
Expected: all passed

- [ ] **Step 5: Commit**

```bash
git add src/hermes_voice/app.py tests/test_app.py
git commit -m "feat: connect microphone and transcription to the UI"
```

---

### Task 6: Command line and live check

**Files:**
- Create: `src/hermes_voice/__main__.py`, `README.md`
- Modify: `pyproject.toml`

**Interfaces:**
- Consumes: everything above.
- Produces: the `hermes-voice` command, with flags `--wake`, `--device`, `--list-devices`, `--stt-model`, `--vad-level`, `--end-silence-ms`.

- [ ] **Step 1: Add `main() -> None` in `src/hermes_voice/__main__.py`**

Use `argparse`. Add `[project.scripts] hermes-voice = "hermes_voice.__main__:main"`.

Before opening the UI:
- `--list-devices` prints `list_devices()` and exits.
- `Mic(args.device).check()` failure prints the `MicError` message and exits 1. `--list-devices` failing the same way does the same.
- `load_api_key()` failure prints the message and exits 1.
- `GET /health` failure prints `Hermes is not running. Start it with: hermes gateway` and exits 1.

Build the VAD as `webrtcvad.Vad(args.vad_level)`, and `is_speech` as `lambda f: vad.is_speech(f, SAMPLE_RATE)`. Session id is `voice-<YYYYmmdd-HHMMSS>`.

- [ ] **Step 2: Check the failure paths**

Run: `HERMES_API_KEY=x uv run python -c "from hermes_voice.hermes import load_api_key; print(load_api_key() == 'x')"`
Expected: `True`

Run: `uv run hermes-voice --list-devices`
Expected: lists at least one input device (`MacBook Air Microphone` on this Mac)

- [ ] **Step 3: Run the suite and the failure message on Linux**

```bash
docker run --rm -v "$PWD":/app -w /app -e UV_PROJECT_ENVIRONMENT=/tmp/venv python:3.13-slim \
  sh -c 'pip install -q uv && uv run pytest -q; uv run hermes-voice --list-devices; echo "exit=$?"'
```

Expected: all tests pass, then the `libportaudio2` install message with no traceback, then `exit=1`.

- [ ] **Step 4: Live check with the user, on real hardware, on each system**

The user runs `uv run hermes-voice` in their terminal. On macOS they approve the microphone prompt. On Linux they first install PortAudio and make sure the gateway is running (`hermes gateway`). The first launch downloads the Whisper model (about 480 MB), so `loading` stays up for a while.

Confirm each, on both systems:
- Header shows the expected microphone and the level meter moves with speech. On Mac, the iPhone mic can become the default; if so, use `--device`. On Linux the name is often `default`, so trust the meter.
- Saying "what time is it" produces a `You` line, then a `Hermes` reply.
- A follow-up referring to the previous answer works, proving the session continues.
- `m` closes the mic, and speech is ignored. On macOS the orange indicator turns off.
- `esc` stops a long request.
- Speaking "Hey Hermes" does not fire twice. `wake_word.enabled` is `true` in `~/.hermes/config.yaml`, so a running Hermes CLI or desktop app may also react. If it does, tell the user and let them decide; do not edit their config.
- If transcripts cut off mid-sentence, raise `--end-silence-ms`. If room noise triggers `hearing`, raise `--vad-level`.

Linux only:
- No ALSA or JACK text appears over the UI at startup or after a few minutes of listening. If it does, record the exact line; JACK has its own error hook that `silence_alsa` does not cover.
- Unplug and replug a USB or Bluetooth mic: status shows `mic lost`, then recovers.

- [ ] **Step 5: Write `README.md`**

Cover: what it does, setup on macOS and Linux (install uv; on Linux, PortAudio; Hermes installed with its API server enabled), `uv run hermes-voice`, the keys, the flags, and the privacy note below.

> Audio stays on this computer. Every transcript goes to Hermes, which sends it to whichever model provider Hermes is configured to use. In open-mic mode that includes anything said in the room.

- [ ] **Step 6: Commit**

```bash
git add .
git commit -m "feat: add hermes-voice command"
```
