import asyncio
import concurrent.futures
import queue
import threading
import time

import httpx
import pytest
from textual.widgets import RichLog, Sparkline, Static

from hermes_voice.app import VoiceApp
from hermes_voice.audio import FRAME_BYTES, MIC_SILENT_HINT, MicError, Segmenter
from hermes_voice.hermes import HermesAuthError, HermesUnavailable, RunEvent
from hermes_voice.stt import is_noise
from hermes_voice.wake import WakeGate


class FakeClient:
    """Scripted stand-in for HermesClient.

    A held run waits until release(), and also until aclose(), so a test that
    never releases still ends when the app shuts the client or cancels the worker.
    """

    def __init__(
        self,
        script=None,
        hold=False,
        fail_starts=0,
        fail_stop=False,
        fail_approve=False,
        stop_error=None,
    ):
        self.script = list(script or [])
        self.hold = hold
        self.fail_starts = fail_starts
        self.fail_stop = fail_stop
        self.fail_approve = fail_approve
        self.stop_error = stop_error
        self.started: list[str] = []
        self.approvals: list[tuple] = []
        self.stops: list[str] = []
        self._next_id = 0
        self._released = 0
        self._wake = asyncio.Event()
        self._closed = False
        # ponytail: pilot.pause() stays busy longer than the offline test's 0.01s retry.
        # The accepted retry waits here so the app is still "offline" when that test looks.
        # Cancel or aclose releases it, so the test can end without a second release().
        self._arm_delay = False
        self._delay_gate = asyncio.Event()

    def release(self) -> None:
        self._released += 1
        self._wake.set()

    async def start_run(self, text: str, session_id: str) -> str:
        del session_id
        if self.fail_starts:
            self.fail_starts -= 1
            self._arm_delay = True
            raise HermesUnavailable("down")
        self.started.append(text)
        self._next_id += 1
        run_id = f"run_{self._next_id}"
        if self._arm_delay:
            self._arm_delay = False
            await self._delay_gate.wait()
        return run_id

    async def events(self, run_id: str):
        del run_id
        for event in self.script:
            yield event
        if not self.hold:
            return
        need = self._released + 1
        while self._released < need and not self._closed:
            self._wake.clear()
            if self._released >= need or self._closed:
                return
            await self._wake.wait()

    async def stop(self, run_id: str) -> None:
        if self.stop_error is not None:
            raise self.stop_error
        if self.fail_stop:
            raise HermesUnavailable("down")
        self.stops.append(run_id)

    async def approve(self, run_id: str, choice: str, request_id: str | None = None) -> None:
        if self.fail_approve:
            raise HermesUnavailable("down")
        self.approvals.append((run_id, choice, request_id))

    async def aclose(self) -> None:
        self._closed = True
        self._wake.set()
        self._delay_gate.set()


def _log_text(app: VoiceApp) -> str:
    return "".join(strip.text for strip in app.query_one("#log", RichLog).lines)


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
        gate = WakeGate()
        app = VoiceApp(c, "voice-test", wake="hey hermes", gate=gate)
        async with app.run_test() as pilot:
            app.on_transcript("what's for dinner")
            await pilot.pause()
            assert c.started == []
            gate.wake()
            app.on_transcript("hey hermes open the logs")
            await pilot.pause()
            assert c.started == ["open the logs"]
    run(go)


def test_wake_phrase_noise_does_not_start_a_run():
    async def go():
        c = FakeClient()
        gate = WakeGate()
        gate.wake()
        app = VoiceApp(c, "voice-test", wake="hermes", gate=gate)
        async with app.run_test() as pilot:
            app.on_transcript("Hermes, thank you.")
            await pilot.pause()
            assert c.started == []
            assert "thank you" not in _log_text(app)
            assert app._pending == []
            assert gate.awake is True
    run(go)


def test_transcript_markup_is_shown_literally():
    async def go():
        c = FakeClient()
        app = VoiceApp(c, "voice-test")
        async with app.run_test() as pilot:
            app.on_transcript("[red] alert")
            await pilot.pause()
            assert "[red] alert" in _log_text(app)
    run(go)


def test_tool_preview_markup_is_shown_literally():
    async def go():
        c = FakeClient(script=[
            RunEvent("tool.started", {"tool": "terminal", "preview": "[red] rm x"}),
            RunEvent("run.completed", {"output": "done"}),
        ])
        app = VoiceApp(c, "voice-test")
        async with app.run_test() as pilot:
            app.on_transcript("go")
            await pilot.pause()
            assert "[red] rm x" in _log_text(app)
    run(go)


def test_failed_stop_is_logged():
    async def go():
        c = FakeClient(hold=True, fail_stop=True)
        app = VoiceApp(c, "voice-test")
        async with app.run_test() as pilot:
            app.on_transcript("long job")
            await pilot.pause()
            await pilot.press("escape")
            await pilot.pause()
            assert c.stops == []
            assert "unavailable" in _log_text(app).lower()
    run(go)


def test_failed_approve_is_logged():
    async def go():
        c = FakeClient(
            hold=True,
            fail_approve=True,
            script=[RunEvent("approval.request", {"command": "rm x", "request_id": "a1"})],
        )
        app = VoiceApp(c, "voice-test")
        async with app.run_test() as pilot:
            app.on_transcript("clean up")
            await pilot.pause()
            await pilot.press("n")
            await pilot.pause()
            assert c.approvals == []
            assert "unavailable" in _log_text(app).lower()
    run(go)


def test_deny_key_sends_deny():
    async def go():
        c = FakeClient(
            hold=True,
            script=[RunEvent("approval.request", {"command": "rm x", "request_id": "a1"})],
        )
        app = VoiceApp(c, "voice-test")
        async with app.run_test() as pilot:
            app.on_transcript("clean up")
            await pilot.pause()
            await pilot.press("n")
            await pilot.pause()
            assert c.approvals == [("run_1", "deny", "a1")]
    run(go)


def test_app_starts_listening():
    async def go():
        app = VoiceApp(FakeClient(), "voice-test")
        async with app.run_test() as pilot:
            await pilot.pause()
            assert app.status == "listening"
            app.set_level(0.4)
            await pilot.pause()
            assert app.query_one("#meter", Sparkline).data == [0.4]
    run(go)


_SPEECH = b"\x01" * FRAME_BYTES
_QUIET = b"\x00" * FRAME_BYTES


class FakeMic:
    """Plays queued frames, then returns None. `.alive` is settable."""

    def __init__(self, frames=None, fail_starts=0):
        self.frames = list(frames or [])
        self.fail_starts = fail_starts
        self.alive = False
        self.starts = 0
        self.stops = 0
        self.reads = 0
        self.name = "Test Microphone"

    def start(self) -> None:
        self.starts += 1
        if self.fail_starts:
            self.fail_starts -= 1
            self.alive = False
            raise MicError("no such device")
        self.alive = True

    def stop(self) -> None:
        self.stops += 1
        self.alive = False
        self.frames.clear()

    def clear(self) -> None:
        self.frames.clear()

    def read(self, timeout: float = 0.1) -> bytes | None:
        self.reads += 1
        if self.alive and self.frames:
            return self.frames.pop(0)
        time.sleep(0.01)
        return None


class FakeTranscriber:
    def __init__(self, texts, hold=False):
        self._texts = list(texts)
        self.hold = hold
        self.calls = 0
        self.heard: list[bytes] = []
        self.entered = threading.Event()
        self.release = threading.Event()

    def load(self) -> None:
        return None

    def transcribe(self, pcm: bytes) -> str:
        self.heard.append(pcm)
        self.calls += 1
        self.entered.set()
        if self.hold:
            self.release.wait(3)
        if not self._texts:
            return ""
        return self._texts.pop(0)


def _segmenter():
    return Segmenter(lambda frame: frame == _SPEECH)


async def until(check, timeout=2.0):
    deadline = asyncio.get_running_loop().time() + timeout
    while not check():
        if asyncio.get_running_loop().time() >= deadline:
            raise AssertionError("timed out")
        await asyncio.sleep(0.01)


def test_utterance_flows_from_mic_to_hermes():
    # FakeMic plays speech frames then silence; fake segmenter emits one utterance
    async def go():
        frames = [_SPEECH] * 20 + [_QUIET] * 34
        mic = FakeMic(frames)
        c = FakeClient()
        app = VoiceApp(c, "voice-test", mic=mic, transcriber=FakeTranscriber(["open the logs"]), segmenter=_segmenter())
        async with app.run_test() as pilot:
            await pilot.pause()
            await until(lambda: c.started == ["open the logs"])
    run(go)


def test_noise_transcript_never_reaches_hermes(monkeypatch):
    # FakeTranscriber returns "Thank you."; asserts c.started == []
    # calls increments on entry, so wait until is_noise has judged the text.
    decided: list[str] = []

    def watch(text: str) -> bool:
        result = is_noise(text)
        decided.append(text)
        return result

    monkeypatch.setattr("hermes_voice.app.is_noise", watch)

    async def go():
        frames = [_SPEECH] * 20 + [_QUIET] * 34
        mic = FakeMic(frames)
        heard = FakeTranscriber(["Thank you."])
        c = FakeClient()
        app = VoiceApp(c, "voice-test", mic=mic, transcriber=heard, segmenter=_segmenter())
        async with app.run_test() as pilot:
            await pilot.pause()
            await until(lambda: decided == ["Thank you."])
            await pilot.pause()
            assert c.started == []
    run(go)


def test_all_zero_audio_warns_about_mic_permission_once():
    # FakeMic plays 150 zero frames; the log contains MIC_SILENT_HINT exactly once
    async def go():
        mic = FakeMic([_QUIET] * 150)
        app = VoiceApp(
            FakeClient(),
            "voice-test",
            mic=mic,
            transcriber=FakeTranscriber(["open the logs"]),
            segmenter=_segmenter(),
        )
        async with app.run_test() as pilot:
            await pilot.pause()
            await until(lambda: not mic.frames and MIC_SILENT_HINT in _log_text(app))
            assert _log_text(app).count(MIC_SILENT_HINT) == 1
            assert app.status == "listening"
    run(go)


def test_mute_closes_the_mic_and_unmute_reopens_it():
    # press m: status == "muted", mic.stops == 1, partial speech discarded
    # press m: status == "listening", mic.starts == 2
    async def go():
        mic = FakeMic([_SPEECH] * 5)
        segmenter = _segmenter()
        c = FakeClient()
        app = VoiceApp(c, "voice-test", mic=mic, transcriber=FakeTranscriber(["open the logs"]), segmenter=segmenter)
        async with app.run_test() as pilot:
            await pilot.pause()
            await until(lambda: mic.starts >= 1 and not mic.frames and segmenter.in_speech)
            await pilot.press("m")
            await pilot.pause()
            assert app.status == "muted"
            assert mic.stops == 1
            assert segmenter.in_speech is False
            await asyncio.sleep(0.2)
            assert c.started == []
            await pilot.press("m")
            await pilot.pause()
            assert app.status == "listening"
            assert mic.starts == 2
    run(go)


def test_mute_discards_audio_waiting_for_transcription():
    async def go():
        mic = FakeMic([_SPEECH, _SPEECH])
        heard = FakeTranscriber(["open the logs"], hold=True)
        c = FakeClient()

        class Emit:
            def __init__(self):
                self.in_speech = False

            def push(self, frame):
                del frame
                self.in_speech = True
                return _SPEECH

            def reset(self):
                self.in_speech = False

        app = VoiceApp(c, "voice-test", mic=mic, transcriber=heard, segmenter=Emit())
        async with app.run_test() as pilot:
            await until(lambda: heard.calls == 1 and not mic.frames)
            await pilot.press("m")
            await pilot.pause()
            heard.release.set()
            await asyncio.sleep(0.2)
            assert c.started == []
            assert heard.calls == 1
            assert app.status == "muted"
    run(go)


def test_lost_mic_shows_status_and_recovers():
    # mic.alive = False: status == "mic lost"; mic.alive = True: status == "listening"
    async def go():
        mic = FakeMic()
        app = VoiceApp(FakeClient(), "voice-test", mic=mic, transcriber=FakeTranscriber([]), segmenter=_segmenter())
        async with app.run_test() as pilot:
            await pilot.pause()
            await until(lambda: mic.reads >= 1 and mic.starts >= 1 and app.status == "listening")
            mic.alive = False
            await until(lambda: app.status == "mic lost")
            mic.alive = True
            await until(lambda: app.status == "listening")
    run(go)


def test_mic_that_cannot_open_shows_the_reason_and_keeps_retrying():
    # FakeMic.start raises MicError("no such device") once, then works
    async def go():
        mic = FakeMic(fail_starts=1)
        app = VoiceApp(FakeClient(), "voice-test", mic=mic, transcriber=FakeTranscriber([]), segmenter=_segmenter())
        async with app.run_test() as pilot:
            await pilot.pause()
            await until(lambda: app.status == "mic lost" and "no such device" in _log_text(app))
            assert _log_text(app).count("no such device") == 1
            await until(lambda: app.status == "listening", timeout=3.5)
            assert _log_text(app).count("no such device") == 1
    run(go)


def test_stop_on_a_finished_run_does_not_crash():
    async def go():
        request = httpx.Request("POST", "http://127.0.0.1:8642/v1/runs/run_1/stop")
        response = httpx.Response(404, request=request)
        error = httpx.HTTPStatusError("not found", request=request, response=response)
        c = FakeClient(hold=True, stop_error=error)
        app = VoiceApp(c, "voice-test")
        async with app.run_test() as pilot:
            app.on_transcript("long job")
            await pilot.pause()
            await pilot.press("escape")
            await pilot.pause()
            assert app.is_running
            assert "already finished" in _log_text(app).lower()
    run(go)


def test_closing_the_app_stops_the_mic_and_workers():
    async def go():
        mic = FakeMic()
        app = VoiceApp(FakeClient(), "voice-test", mic=mic, transcriber=FakeTranscriber([]), segmenter=_segmenter())
        async with app.run_test() as pilot:
            await pilot.pause()
            await until(lambda: mic.starts >= 1)
        app._capture_thread.join(1)
        app._stt_thread.join(1)
        assert mic.stops >= 1
        assert not app._capture_thread.is_alive()
        assert not app._stt_thread.is_alive()
    run(go)


def test_stale_transcript_does_not_reach_hermes():
    async def go():
        c = FakeClient()
        app = VoiceApp(c, "voice-test")
        async with app.run_test() as pilot:
            await pilot.pause()
            app._mute_gen = 1
            app._deliver_transcript("open the logs", 0)
            await pilot.pause()
            assert c.started == []
    run(go)


def test_capture_error_is_logged_once_and_the_loop_keeps_running():
    class RaisingMic(FakeMic):
        def read(self, timeout: float = 0.1) -> bytes | None:
            del timeout
            self.reads += 1
            raise RuntimeError("capture blew up")

    async def go():
        mic = RaisingMic()
        app = VoiceApp(
            FakeClient(),
            "voice-test",
            mic=mic,
            transcriber=FakeTranscriber([]),
            segmenter=_segmenter(),
        )
        async with app.run_test() as pilot:
            await pilot.pause()
            await until(
                lambda: (
                    mic.reads >= 3
                    and _log_text(app).count("capture blew up") == 1
                    and app.status == "mic lost"
                )
                or not app._capture_thread.is_alive()
            )
            assert app._capture_thread.is_alive()
            assert mic.reads >= 3
            assert _log_text(app).count("capture blew up") == 1
            assert app.status == "mic lost"
    run(go)


def test_transcribe_error_drops_the_utterance_and_the_loop_continues():
    class Emit:
        def __init__(self):
            self.in_speech = False

        def push(self, frame):
            del frame
            self.in_speech = False
            return _SPEECH

        def reset(self):
            self.in_speech = False

    class Flaky(FakeTranscriber):
        def transcribe(self, pcm: bytes) -> str:
            del pcm
            self.calls += 1
            self.entered.set()
            if self.calls == 1:
                raise RuntimeError("stt failed")
            if not self._texts:
                return ""
            return self._texts.pop(0)

    async def go():
        mic = FakeMic([_SPEECH, _SPEECH])
        heard = Flaky(["open the logs"])
        c = FakeClient()
        app = VoiceApp(c, "voice-test", mic=mic, transcriber=heard, segmenter=Emit())
        async with app.run_test() as pilot:
            await pilot.pause()
            await until(lambda: c.started == ["open the logs"] or not app._stt_thread.is_alive())
            assert c.started == ["open the logs"]
            assert "stt failed" in _log_text(app)
            assert heard.calls == 2
    run(go)


def test_model_load_error_stays_loading_and_retries():
    class NeverLoads(FakeTranscriber):
        def __init__(self):
            super().__init__([])
            self.loads = 0
            self.times: list[float] = []

        def load(self) -> None:
            self.loads += 1
            self.times.append(time.monotonic())
            raise RuntimeError("model download failed")

    async def go():
        heard = NeverLoads()
        app = VoiceApp(
            FakeClient(),
            "voice-test",
            mic=FakeMic(),
            transcriber=heard,
            segmenter=_segmenter(),
        )
        async with app.run_test() as pilot:
            await pilot.pause()
            await until(
                lambda: heard.loads >= 2 or not app._stt_thread.is_alive(),
                timeout=8,
            )
            assert heard.loads >= 2
            assert app._stt_thread.is_alive()
            assert app.status == "loading"
            assert _log_text(app).count("model download failed") == 1
            assert heard.times[1] - heard.times[0] >= 5
            app._stop_audio.set()
            await asyncio.to_thread(app._stt_thread.join, 1.0)
            assert not app._stt_thread.is_alive()
    run(go)


def test_pending_speech_is_retried_when_send_exits():
    class DieInEvents(FakeClient):
        async def events(self, run_id: str):
            del run_id
            await asyncio.sleep(0.2)
            if self.started:
                raise RuntimeError("stream died")
            yield RunEvent("run.interrupted", {})

    async def go():
        c = DieInEvents()
        app = VoiceApp(c, "voice-test", retry_seconds=0.01)
        async with app.run_test() as pilot:
            app.on_transcript("first")
            await until(lambda: c.started == ["first"])
            app.on_transcript("second")
            await until(lambda: c.started == ["first", "second"])
            assert app._pending == []
    run(go)


@pytest.mark.parametrize("name", ["run.cancelled", "run.interrupted"])
def test_partial_reply_is_logged_when_the_run_ends_early(name):
    async def go():
        c = FakeClient(script=[
            RunEvent("message.delta", {"delta": "partial answer"}),
            RunEvent(name, {}),
        ])
        app = VoiceApp(c, "voice-test")
        async with app.run_test() as pilot:
            app.on_transcript("hi")
            await pilot.pause()
            assert "partial answer" in _log_text(app)
    run(go)


def test_bad_key_stays_offline_and_drops_later_speech():
    class RejectKey(FakeClient):
        async def start_run(self, text: str, session_id: str) -> str:
            del session_id
            self.started.append(text)
            raise HermesAuthError("bad key")

    async def go():
        c = RejectKey()
        app = VoiceApp(c, "voice-test", retry_seconds=0.01)
        async with app.run_test() as pilot:
            app.on_transcript("hello")
            await pilot.pause()
            app.on_transcript("again")
            await asyncio.sleep(0.05)
            await pilot.pause()
            assert c.started == ["hello"]
            assert app.status == "offline"
            assert app._pending == ["hello"]
            assert "You  again" not in _log_text(app)
            assert _log_text(app).count("not being sent") == 1
    run(go)


def test_quit_stops_the_active_run():
    class StopFails(FakeClient):
        async def stop(self, run_id: str) -> None:
            self.stops.append(run_id)
            raise RuntimeError("nope")

    async def go():
        c = StopFails(hold=True)
        app = VoiceApp(c, "voice-test")
        async with app.run_test() as pilot:
            app.on_transcript("long job")
            await pilot.pause()
            await pilot.press("q")
        assert c.stops == ["run_1"]
        assert c._closed
    run(go)


def test_mute_during_mic_lost_shows_muted():
    async def go():
        mic = FakeMic()
        app = VoiceApp(
            FakeClient(),
            "voice-test",
            mic=mic,
            transcriber=FakeTranscriber([]),
            segmenter=_segmenter(),
        )
        async with app.run_test() as pilot:
            await pilot.pause()
            await until(lambda: mic.reads >= 1 and mic.starts >= 1 and app.status == "listening")
            mic.alive = False
            await until(lambda: app.status == "mic lost")
            await pilot.press("m")
            await pilot.pause()
            await asyncio.sleep(0.2)
            assert app.status == "muted"
    run(go)


def test_cancelled_ui_callback_does_not_raise():
    async def go():
        app = VoiceApp(FakeClient(), "voice-test")
        async with app.run_test() as pilot:
            await pilot.pause()

            def cancelled(*_args, **_kwargs):
                raise concurrent.futures.CancelledError()

            app.call_from_thread = cancelled
            app._on_ui(app.log_line, "still shutting down")
            assert app.is_running
    run(go)


class FakeSpeaker:
    """Records spoken text. hold=True waits until release or the stop event."""

    def __init__(self, hold=False):
        self.hold = hold
        self.spoke: list[str] = []
        self.entered = threading.Event()
        self.release = threading.Event()
        self.stop_events: list[threading.Event] = []

    def say(self, text: str, stop: threading.Event) -> None:
        self.stop_events.append(stop)
        self.entered.set()
        self.spoke.append(text)
        if self.hold:
            while not stop.is_set() and not self.release.is_set():
                stop.wait(0.02)


class _CountingSegmenter:
    def __init__(self):
        self.in_speech = False
        self.pushes = 0

    def push(self, frame):
        del frame
        self.pushes += 1
        return None

    def reset(self):
        self.in_speech = False


class _PulseMic(FakeMic):
    def read(self, timeout: float = 0.1) -> bytes | None:
        del timeout
        self.reads += 1
        time.sleep(0.02)
        if not self.alive:
            return None
        return _SPEECH


def test_completed_reply_is_spoken_after_cleanup():
    async def go():
        c = FakeClient(script=[RunEvent("run.completed", {"output": "**Done**"})])
        speaker = FakeSpeaker()
        app = VoiceApp(c, "voice-test", speaker=speaker)
        async with app.run_test() as pilot:
            app.on_transcript("go")
            await until(lambda: speaker.spoke == ["Done"])
            assert app._reply == ""
            assert "Hermes" in _log_text(app)
            assert app._speak_thread.name == "hermes-speak"
            assert app._speak_thread.daemon
    run(go)


def test_reply_with_only_code_is_not_spoken():
    async def go():
        c = FakeClient(script=[RunEvent("run.completed", {"output": "```\ncode only\n```"})])
        speaker = FakeSpeaker()
        app = VoiceApp(c, "voice-test", speaker=speaker)
        async with app.run_test() as pilot:
            app.on_transcript("go")
            await pilot.pause()
            await asyncio.sleep(0.25)
            assert speaker.spoke == []
            assert app.status == "listening"
            assert "speaker" not in _log_text(app).lower()
    run(go)


def test_no_speaker_means_no_speech():
    async def go():
        c = FakeClient(script=[RunEvent("run.completed", {"output": "Done"})])
        app = VoiceApp(c, "voice-test")
        async with app.run_test() as pilot:
            app.on_transcript("go")
            await pilot.pause()
            assert app._speak_thread is None
            assert "Done" in _log_text(app)
            assert app.status == "listening"
    run(go)


def test_status_is_speaking_during_speech():
    async def go():
        c = FakeClient(script=[RunEvent("run.completed", {"output": "Hello."})])
        speaker = FakeSpeaker(hold=True)
        app = VoiceApp(c, "voice-test", speaker=speaker)
        async with app.run_test() as pilot:
            app.on_transcript("go")
            await until(lambda: app.status == "speaking")
            speaker.release.set()
            await until(lambda: app.status == "listening")
    run(go)


def test_frames_during_speech_never_reach_the_segmenter():
    async def go():
        mic = FakeMic()
        segmenter = _CountingSegmenter()
        speaker = FakeSpeaker(hold=True)
        c = FakeClient(script=[RunEvent("run.completed", {"output": "Hello."})])
        app = VoiceApp(
            c,
            "voice-test",
            mic=mic,
            transcriber=FakeTranscriber([]),
            segmenter=segmenter,
            speaker=speaker,
        )
        async with app.run_test() as pilot:
            app.on_transcript("go")
            await until(lambda: speaker.entered.is_set() and app.status == "speaking")
            seen = segmenter.pushes
            mic.frames.extend([_SPEECH] * 30)
            await until(lambda: not mic.frames and mic.reads >= 5)
            assert segmenter.pushes == seen
            await until(lambda: app._levels and app._levels[-1] == 0.0)
    run(go)


def test_frames_in_the_300ms_after_speech_are_dropped():
    async def go():
        mic = _PulseMic()
        segmenter = _CountingSegmenter()
        speaker = FakeSpeaker(hold=True)
        c = FakeClient(script=[RunEvent("run.completed", {"output": "Hello."})])
        app = VoiceApp(
            c,
            "voice-test",
            mic=mic,
            transcriber=FakeTranscriber([]),
            segmenter=segmenter,
            speaker=speaker,
        )
        async with app.run_test() as pilot:
            app.on_transcript("go")
            await until(lambda: app.status == "speaking")
            seen = segmenter.pushes
            await asyncio.sleep(0.12)
            assert segmenter.pushes == seen
            speaker.release.set()
            mark = time.monotonic()
            await asyncio.sleep(0.1)
            assert segmenter.pushes == seen
            await until(lambda: segmenter.pushes > seen, timeout=1.5)
            assert time.monotonic() - mark >= 0.3
    run(go)


def test_speech_waits_until_the_user_stops_talking():
    class Talking:
        def __init__(self):
            self.in_speech = True

        def push(self, frame):
            del frame
            return None

        def reset(self):
            return None

    async def go():
        segmenter = Talking()
        speaker = FakeSpeaker()
        c = FakeClient(script=[RunEvent("run.completed", {"output": "Ready."})])
        app = VoiceApp(c, "voice-test", speaker=speaker, segmenter=segmenter)
        async with app.run_test() as pilot:
            app.on_transcript("go")
            await pilot.pause()
            await asyncio.sleep(0.3)
            assert speaker.spoke == []
            segmenter.in_speech = False
            await until(lambda: speaker.spoke == ["Ready."])
    run(go)


def test_esc_cuts_speech_before_it_stops_the_run():
    class TwoRuns(FakeClient):
        def __init__(self):
            super().__init__(hold=True)
            self.calls = 0

        async def events(self, run_id: str):
            del run_id
            self.calls += 1
            if self.calls == 1:
                yield RunEvent("run.completed", {"output": "Hello there."})
                return
            need = self._released + 1
            while self._released < need and not self._closed:
                self._wake.clear()
                if self._released >= need or self._closed:
                    return
                await self._wake.wait()

    async def go():
        c = TwoRuns()
        speaker = FakeSpeaker(hold=True)
        app = VoiceApp(c, "voice-test", speaker=speaker)
        async with app.run_test() as pilot:
            app.on_transcript("first")
            await until(lambda: speaker.entered.is_set())
            app._speech.put("should not be spoken")
            app.on_transcript("second")
            await until(lambda: c.started == ["first", "second"] and app._run_id is not None)
            await pilot.press("escape")
            await pilot.pause()
            assert c.stops == []
            assert speaker.stop_events[-1].is_set()
            await until(lambda: app._speaking is False)
            await asyncio.sleep(0.2)
            assert speaker.spoke == ["Hello there."]
            await pilot.press("escape")
            await pilot.pause()
            assert c.stops == ["run_2"]
    run(go)


def test_failed_speech_logs_and_next_reply_still_speaks():
    class Flaky(FakeSpeaker):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def say(self, text: str, stop: threading.Event) -> None:
            del stop
            self.calls += 1
            self.entered.set()
            if self.calls == 1:
                raise RuntimeError("speaker failed")
            self.spoke.append(text)

    class Replies(FakeClient):
        def __init__(self):
            super().__init__()
            self.n = 0

        async def events(self, run_id: str):
            del run_id
            self.n += 1
            word = "First sentence." if self.n == 1 else "Second sentence."
            yield RunEvent("run.completed", {"output": word})

    async def go():
        speaker = Flaky()
        c = Replies()
        app = VoiceApp(c, "voice-test", speaker=speaker)
        async with app.run_test() as pilot:
            app.on_transcript("one")
            await until(lambda: "speaker failed" in _log_text(app))
            app.on_transcript("two")
            await until(lambda: speaker.spoke == ["Second sentence."])
            assert app._speak_thread.is_alive()
    run(go)


def test_quit_stops_speech():
    async def go():
        speaker = FakeSpeaker(hold=True)
        c = FakeClient(script=[RunEvent("run.completed", {"output": "Bye."})])
        app = VoiceApp(c, "voice-test", speaker=speaker)
        async with app.run_test() as pilot:
            app.on_transcript("go")
            await until(lambda: speaker.entered.is_set())
            await pilot.press("q")
        assert speaker.stop_events[0].is_set()
        await asyncio.to_thread(app._speak_thread.join, 1.0)
        assert not app._speak_thread.is_alive()
    run(go)


def test_esc_while_waiting_for_the_user_cancels_the_reply():
    class Talking:
        def __init__(self):
            self.in_speech = True

        def push(self, frame):
            del frame
            return None

        def reset(self):
            return None

    class TwoRuns(FakeClient):
        def __init__(self):
            super().__init__(hold=True)
            self.calls = 0

        async def events(self, run_id: str):
            del run_id
            self.calls += 1
            if self.calls == 1:
                yield RunEvent("run.completed", {"output": "Ready."})
                return
            need = self._released + 1
            while self._released < need and not self._closed:
                self._wake.clear()
                if self._released >= need or self._closed:
                    return
                await self._wake.wait()

    async def go():
        segmenter = Talking()
        speaker = FakeSpeaker()
        c = TwoRuns()
        app = VoiceApp(c, "voice-test", speaker=speaker, segmenter=segmenter)
        async with app.run_test() as pilot:
            app.on_transcript("first")
            await until(
                lambda: c.started == ["first"] and app._run_id is None and app._speech.empty()
            )
            await asyncio.sleep(0.15)
            assert speaker.spoke == []
            app.on_transcript("second")
            await until(lambda: app._run_id is not None)
            await pilot.press("escape")
            await pilot.pause()
            assert c.stops == []
            segmenter.in_speech = False
            await asyncio.sleep(0.4)
            assert speaker.spoke == []
    run(go)


def test_speech_keeps_waiting_if_talking_starts_before_playback():
    class Race:
        def __init__(self):
            self._talking = False
            self._armed = False
            self.reset_while_talking = False
            self.quiet_seen = threading.Event()
            self.release = threading.Event()

        @property
        def in_speech(self):
            if not self._armed:
                self._armed = True
                self.quiet_seen.set()
                self.release.wait(2)
                self._talking = True
                return False
            return self._talking

        def push(self, frame):
            del frame
            return None

        def reset(self):
            if self._talking:
                self.reset_while_talking = True
            self._talking = False

    async def go():
        segmenter = Race()
        speaker = FakeSpeaker()
        c = FakeClient(script=[RunEvent("run.completed", {"output": "Ready."})])
        app = VoiceApp(c, "voice-test", speaker=speaker, segmenter=segmenter)
        async with app.run_test() as pilot:
            del pilot
            app.on_transcript("go")
            await until(lambda: segmenter.quiet_seen.is_set())
            segmenter.release.set()
            await asyncio.sleep(0.2)
            assert speaker.spoke == []
            assert segmenter.reset_while_talking is False
            segmenter._talking = False
            await until(lambda: speaker.spoke == ["Ready."])
    run(go)


class _BacklogMic(FakeMic):
    def __init__(self):
        super().__init__()
        self.backlog: list[bytes] = []
        self.release = threading.Event()

    def read(self, timeout: float = 0.1) -> bytes | None:
        del timeout
        self.reads += 1
        if self.alive and self.frames:
            return self.frames.pop(0)
        if self.release.is_set() and self.backlog:
            return self.backlog.pop(0)
        time.sleep(0.01)
        return None

    def clear(self) -> None:
        super().clear()
        self.backlog.clear()


def test_frames_queued_before_the_tail_ends_never_reach_the_segmenter():
    async def go():
        mic = _BacklogMic()
        segmenter = _CountingSegmenter()
        speaker = FakeSpeaker(hold=True)
        c = FakeClient(script=[RunEvent("run.completed", {"output": "Hello."})])
        app = VoiceApp(
            c,
            "voice-test",
            mic=mic,
            transcriber=FakeTranscriber([]),
            segmenter=segmenter,
            speaker=speaker,
        )
        async with app.run_test() as pilot:
            del pilot
            app.on_transcript("go")
            await until(lambda: app._speaking)
            mic.backlog.extend([_SPEECH] * 10)
            seen = segmenter.pushes
            speaker.release.set()
            await until(lambda: app._speaking is False, timeout=2)
            mic.release.set()
            await asyncio.sleep(0.2)
            assert segmenter.pushes == seen
            assert mic.backlog == []
    run(go)


def test_discarded_frames_set_the_meter_to_zero_once():
    async def go():
        mic = FakeMic()
        speaker = FakeSpeaker(hold=True)
        c = FakeClient(script=[RunEvent("run.completed", {"output": "Hello."})])
        app = VoiceApp(
            c,
            "voice-test",
            mic=mic,
            transcriber=FakeTranscriber([]),
            segmenter=_CountingSegmenter(),
            speaker=speaker,
        )
        levels: list[float] = []

        def record(value):
            levels.append(value)
            VoiceApp.set_level(app, value)

        app.set_level = record
        async with app.run_test() as pilot:
            app.on_transcript("go")
            await until(lambda: app.status == "speaking" and levels == [0.0])
            mic.frames.extend([_SPEECH] * 20)
            await until(lambda: not mic.frames and mic.reads >= 5)
            await pilot.pause()
            assert levels == [0.0]
            speaker.release.set()
    run(go)


def test_quit_during_playback_start_does_not_clear_stop():
    class Segmenter:
        def __init__(self):
            self.in_speech = False
            self.app = None
            self.resets = 0

        def push(self, frame):
            del frame
            return None

        def reset(self):
            self.resets += 1
            if self.resets == 1 and self.app is not None:
                self.app._stop_speech.set()
                self.app._stop_audio.set()

    async def go():
        segmenter = Segmenter()
        speaker = FakeSpeaker()
        c = FakeClient(script=[RunEvent("run.completed", {"output": "Bye."})])
        app = VoiceApp(c, "voice-test", speaker=speaker, segmenter=segmenter)
        segmenter.app = app
        async with app.run_test() as pilot:
            del pilot
            app.on_transcript("go")
            await until(lambda: segmenter.resets >= 1)
            await asyncio.sleep(0.1)
            if speaker.stop_events:
                assert speaker.stop_events[0].is_set()
            else:
                assert app._stop_speech.is_set()
    run(go)


class _IdleSegmenter:
    """Stays out of speech. reset() drops anything recorded before the wake."""

    def __init__(self):
        self.in_speech = False
        self.frames: list[bytes] = []

    def push(self, frame):
        self.frames.append(frame)
        return None

    def reset(self):
        self.frames.clear()


class _FrameMic(FakeMic):
    def __init__(self):
        super().__init__()
        self.box: queue.Queue[bytes] = queue.Queue()

    def read(self, timeout: float = 0.1) -> bytes | None:
        del timeout
        self.reads += 1
        if not self.alive:
            time.sleep(0.01)
            return None
        try:
            return self.box.get_nowait()
        except queue.Empty:
            time.sleep(0.01)
            return None


class _Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def test_status_is_waiting_while_asleep():
    async def go():
        gate = WakeGate()
        app = VoiceApp(FakeClient(), "voice-test", wake="hey hermes", gate=gate)
        async with app.run_test() as pilot:
            await pilot.pause()
            assert app.status == "waiting"
            header = app.query_one("#status", Static).content
            assert header == '● WAITING FOR "HEY HERMES"'
    run(go)


def test_one_breath_command_is_sent_without_the_phrase():
    async def go():
        c = FakeClient()
        gate = WakeGate()
        gate.wake()
        app = VoiceApp(c, "voice-test", wake="hey hermes", gate=gate)
        async with app.run_test() as pilot:
            app.on_transcript("Hey Hermes, open the logs")
            await pilot.pause()
            assert c.started == ["open the logs"]
    run(go)


def test_phrase_only_transcript_keeps_the_gate_open():
    async def go():
        c = FakeClient()
        gate = WakeGate()
        gate.wake()
        app = VoiceApp(c, "voice-test", wake="hey hermes", gate=gate)
        async with app.run_test() as pilot:
            app.on_transcript("Hey Hermes.")
            await pilot.pause()
            assert c.started == []
            assert gate.awake is True
            app.on_transcript("open the logs")
            await pilot.pause()
            assert c.started == ["open the logs"]
    run(go)


def test_transcript_arriving_after_the_gate_closed_is_dropped():
    async def go():
        c = FakeClient()
        gate = WakeGate()
        gate.wake()
        app = VoiceApp(c, "voice-test", wake="hey hermes", gate=gate)
        async with app.run_test() as pilot:
            app.on_transcript("Hey Hermes, open the logs")
            await pilot.pause()
            app.on_transcript("and another thing")
            await pilot.pause()
            assert c.started == ["open the logs"]
    run(go)


class _EmitCount:
    """Returns one utterance per push, for the first `count` frames."""

    def __init__(self, count):
        self.in_speech = False
        self.count = count
        self.n = 0

    def push(self, frame):
        del frame
        self.n += 1
        if self.n <= self.count:
            return f"utt-{self.n}".encode()
        return None

    def reset(self):
        self.in_speech = False


def test_window_expiry_returns_to_waiting():
    async def go():
        clock = _Clock()
        gate = WakeGate(window_s=8, clock=clock)
        mic = _FrameMic()
        app = VoiceApp(
            FakeClient(),
            "voice-test",
            wake="hey hermes",
            gate=gate,
            mic=mic,
            transcriber=FakeTranscriber([]),
            wake_transcriber=FakeTranscriber(["Hey Hermes"]),
            segmenter=_EmitCount(1),
        )
        async with app.run_test() as pilot:
            del pilot
            await until(lambda: mic.starts >= 1 and app.status == "waiting")
            mic.box.put(_QUIET)
            await until(lambda: gate.awake and app.status == "listening")
            clock.now = 8
            mic.box.put(_QUIET)
            await until(lambda: app.status == "waiting" and gate.awake is False)
    run(go)


def test_speech_playback_closes_the_gate():
    async def go():
        gate = WakeGate()
        gate.wake()
        speaker = FakeSpeaker(hold=True)
        c = FakeClient(script=[RunEvent("run.completed", {"output": "Hello."})])
        app = VoiceApp(c, "voice-test", wake="hey hermes", gate=gate, speaker=speaker)
        async with app.run_test() as pilot:
            del pilot
            app._pending.append("open the logs")
            app._ensure_send()
            await until(lambda: app.status == "speaking")
            assert gate.awake is False
            assert app._utterances.empty()
            speaker.release.set()
    run(go)


def test_mic_loss_closes_the_gate():
    async def go():
        gate = WakeGate()
        gate.wake()
        mic = FakeMic()
        app = VoiceApp(
            FakeClient(),
            "voice-test",
            wake="hey hermes",
            gate=gate,
            mic=mic,
            transcriber=FakeTranscriber([]),
            segmenter=_segmenter(),
        )
        async with app.run_test() as pilot:
            del pilot
            await until(lambda: mic.starts >= 1 and app.status == "listening")
            app._utterances.put((b"queued", 0))
            mic.alive = False
            await until(lambda: app.status == "mic lost" and gate.awake is False)
            assert app._utterances.empty()
            mic.alive = True
            await until(lambda: app.status == "waiting", timeout=3.5)
    run(go)


def test_no_gate_keeps_open_mic_behavior():
    async def go():
        frames = [_SPEECH] * 20 + [_QUIET] * 34
        mic = FakeMic(frames)
        c = FakeClient()
        app = VoiceApp(
            c,
            "voice-test",
            mic=mic,
            transcriber=FakeTranscriber(["open the logs"]),
            segmenter=_segmenter(),
        )
        async with app.run_test() as pilot:
            del pilot
            await until(lambda: c.started == ["open the logs"])
            assert app.gate is None
    run(go)


def test_finished_reply_is_labelled_and_turns_are_separated():
    async def go():
        c = FakeClient(script=[RunEvent("run.completed", {"output": "Done."})])
        app = VoiceApp(c, "voice-test")
        async with app.run_test() as pilot:
            app.on_transcript("first")
            await pilot.pause()
            app.on_transcript("second")
            await pilot.pause()
            lines = [strip.text.strip() for strip in app.query_one("#log", RichLog).lines]
            assert lines[:2] == ["You  first", "Hermes"]
            assert "" in lines[2:lines.index("You  second")]

    run(go)


class _RecordingTranscriber:
    """Records each utterance. Holds the first call so a later one can sit in the queue."""

    def __init__(self, texts):
        self._texts = list(texts)
        self.heard: list[bytes] = []
        self.entered = threading.Event()
        self.release = threading.Event()

    def load(self) -> None:
        return None

    def transcribe(self, pcm: bytes) -> str:
        self.heard.append(pcm)
        self.entered.set()
        if len(self.heard) == 1:
            self.release.wait(3)
        if not self._texts:
            return ""
        return self._texts.pop(0)


def test_next_wake_during_cloud_transcription_is_not_lost():
    async def go():
        cloud = _RecordingTranscriber([
            "Hey Hermes, open the logs",
            "Hey Hermes, what time is it",
        ])
        wake = FakeTranscriber([
            "Hey Hermes, open the logs",
            "Hey Hermes, what time is it",
        ])
        gate = WakeGate()
        mic = FakeMic()
        c = FakeClient()
        app = VoiceApp(
            c,
            "voice-test",
            wake="hey hermes",
            gate=gate,
            mic=mic,
            transcriber=cloud,
            wake_transcriber=wake,
            segmenter=_IdleSegmenter(),
        )
        async with app.run_test() as pilot:
            del pilot
            await until(lambda: mic.starts >= 1)
            app._utterances.put((b"first", 0))
            try:
                await until(lambda: cloud.entered.is_set() and app._utterances.qsize() == 0)
                app._utterances.put((b"second", 0))
                cloud.release.set()
                await until(lambda: c.started == ["open the logs", "what time is it"])
            finally:
                cloud.release.set()
    run(go)


def test_reply_during_wake_transcription_plays_after_the_command_is_sent():
    async def go():
        heard = _RecordingTranscriber(["Hey Hermes, open the logs"])
        gate = WakeGate()
        gate.wake()
        speaker = FakeSpeaker()
        mic = FakeMic()
        c = FakeClient()
        app = VoiceApp(
            c,
            "voice-test",
            wake="hey hermes",
            gate=gate,
            mic=mic,
            transcriber=heard,
            segmenter=_IdleSegmenter(),
            speaker=speaker,
        )
        async with app.run_test() as pilot:
            del pilot
            await until(lambda: mic.starts >= 1)
            app._utterances.put((b"command-audio", 0))
            try:
                await until(lambda: heard.entered.is_set())
                app._speech.put("Ready.")
                await until(lambda: app._speech_waiting or speaker.spoke or not gate.awake)
                assert gate.awake is True
                assert speaker.spoke == []
                assert c.started == []
                heard.release.set()
                await until(lambda: c.started == ["open the logs"] and speaker.spoke == ["Ready."])
            finally:
                heard.release.set()
    run(go)


def test_accepted_command_returns_to_waiting_and_the_reply_plays_promptly():
    class Stuck:
        def __init__(self):
            self.in_speech = True

        def push(self, frame):
            del frame
            return None

        def reset(self):
            self.in_speech = False

    async def go():
        segmenter = Stuck()
        speaker = FakeSpeaker()
        c = FakeClient(script=[RunEvent("run.completed", {"output": "Ready."})])
        gate = WakeGate()
        gate.wake()
        app = VoiceApp(
            c,
            "voice-test",
            wake="hey hermes",
            gate=gate,
            segmenter=segmenter,
            speaker=speaker,
        )
        async with app.run_test() as pilot:
            del pilot
            app._hearing = True
            app._sync_status()
            assert app.status == "hearing"
            app.on_transcript("Hey Hermes, open the logs")
            assert app.status == "waiting"
            assert segmenter.in_speech is False
            await until(lambda: speaker.spoke == ["Ready."], timeout=2)
            assert c.started == ["open the logs"]
    run(go)


def test_talk_while_asleep_never_reaches_the_cloud():
    # The wake model hears the room. The cloud model and Hermes do not.
    async def go():
        frames = [_SPEECH] * 20 + [_QUIET] * 34
        mic = FakeMic(frames)
        wake = FakeTranscriber(["what's for dinner"])
        cloud = FakeTranscriber(["what's for dinner"])
        c = FakeClient()
        app = VoiceApp(
            c,
            "voice-test",
            wake="hey hermes",
            gate=WakeGate(),
            mic=mic,
            transcriber=cloud,
            wake_transcriber=wake,
            segmenter=_segmenter(),
        )
        async with app.run_test() as pilot:
            del pilot
            await until(lambda: wake.calls >= 1)
            await asyncio.sleep(0.2)
            assert wake.heard
            assert cloud.calls == 0
            assert c.started == []
            assert app.status == "waiting"
            assert app._hearing is False
    run(go)


def test_wake_utterance_goes_to_the_cloud_and_strips_the_phrase():
    async def go():
        frames = [_SPEECH] * 20 + [_QUIET] * 34
        mic = FakeMic(frames)
        wake = FakeTranscriber(["Hey Hermes, open the logs"])
        cloud = FakeTranscriber(["Hey Hermes, open the logs"])
        c = FakeClient()
        app = VoiceApp(
            c,
            "voice-test",
            wake="hey hermes",
            gate=WakeGate(),
            mic=mic,
            transcriber=cloud,
            wake_transcriber=wake,
            segmenter=_segmenter(),
        )
        async with app.run_test() as pilot:
            del pilot
            await until(lambda: c.started == ["open the logs"])
            assert cloud.calls == 1
            assert wake.calls == 1
    run(go)


def test_wake_rings_the_bell(monkeypatch):
    rang = []
    monkeypatch.setattr(VoiceApp, "bell", lambda self: rang.append(self))

    async def go():
        wake = FakeTranscriber(["Hey Hermes"])
        cloud = FakeTranscriber(["should not run"])
        gate = WakeGate()
        app = VoiceApp(
            FakeClient(),
            "voice-test",
            wake="hey hermes",
            gate=gate,
            mic=FakeMic([_SPEECH]),
            transcriber=cloud,
            wake_transcriber=wake,
            segmenter=_EmitCount(1),
        )
        async with app.run_test() as pilot:
            del pilot
            await until(lambda: len(rang) == 1 and app.status == "listening")
            await asyncio.sleep(0.2)
            assert cloud.calls == 0
            assert gate.awake is True
    run(go)


def test_phrase_only_opens_the_window_for_the_next_utterance():
    # The phrase-only utterance is not sent. The next one is.
    async def go():
        wake = FakeTranscriber(["Hey Hermes.", "open the logs"])
        cloud = FakeTranscriber(["open the logs"])
        c = FakeClient()
        app = VoiceApp(
            c,
            "voice-test",
            wake="hey hermes",
            gate=WakeGate(),
            mic=FakeMic([_SPEECH, _SPEECH]),
            transcriber=cloud,
            wake_transcriber=wake,
            segmenter=_EmitCount(2),
        )
        async with app.run_test() as pilot:
            del pilot
            await until(lambda: c.started == ["open the logs"])
            await asyncio.sleep(0.2)
            assert cloud.calls == 1
            assert wake.calls == 1
    run(go)


def test_gate_closes_after_a_command():
    async def go():
        frames = [_SPEECH] * 20 + [_QUIET] * 34
        gate = WakeGate()
        c = FakeClient()
        app = VoiceApp(
            c,
            "voice-test",
            wake="hey hermes",
            gate=gate,
            mic=FakeMic(frames),
            transcriber=FakeTranscriber(["Hey Hermes, open the logs"]),
            wake_transcriber=FakeTranscriber(["Hey Hermes, open the logs"]),
            segmenter=_segmenter(),
        )
        async with app.run_test() as pilot:
            del pilot
            await until(lambda: c.started == ["open the logs"])
            assert gate.awake is False
            await until(lambda: app.status == "waiting")
    run(go)


def test_mute_closes_the_gate_and_drops_queued_audio():
    async def go():
        gate = WakeGate()
        gate.wake()
        cloud = FakeTranscriber(["Hey Hermes, open the logs", "and the metrics"], hold=True)
        c = FakeClient()
        app = VoiceApp(
            c,
            "voice-test",
            wake="hey hermes",
            gate=gate,
            mic=FakeMic([_SPEECH, _SPEECH]),
            transcriber=cloud,
            segmenter=_EmitCount(2),
        )
        async with app.run_test() as pilot:
            try:
                await until(lambda: cloud.calls == 1 and app._utterances.qsize() >= 1)
                await pilot.press("m")
                await pilot.pause()
                assert app.status == "muted"
                assert gate.awake is False
                assert app._utterances.empty()
                cloud.release.set()
                await asyncio.sleep(0.2)
                assert c.started == []
            finally:
                cloud.release.set()
    run(go)


def test_talk_while_asleep_does_not_delay_a_spoken_reply():
    class Talking:
        def __init__(self):
            self.in_speech = True

        def push(self, frame):
            del frame
            return None

        def reset(self):
            return None

    async def go():
        speaker = FakeSpeaker()
        gate = WakeGate()
        app = VoiceApp(
            FakeClient(),
            "voice-test",
            wake="hey hermes",
            gate=gate,
            speaker=speaker,
            segmenter=Talking(),
        )
        async with app.run_test() as pilot:
            del pilot
            app._utterances.put((b"room-talk", 0))
            app._speech.put("Ready.")
            await until(lambda: speaker.spoke == ["Ready."], timeout=1)
            assert app._utterances.empty()
            assert gate.awake is False
    run(go)


def test_wake_model_failure_logs_and_sends_nothing():
    class NeverWakes(FakeTranscriber):
        def __init__(self):
            super().__init__([])
            self.loads = 0
            self.times: list[float] = []

        def load(self) -> None:
            self.loads += 1
            self.times.append(time.monotonic())
            raise RuntimeError("wake model missing")

    async def go():
        wake = NeverWakes()
        cloud = FakeTranscriber(["open the logs"])
        c = FakeClient()
        frames = [_SPEECH] * 20 + [_QUIET] * 34
        app = VoiceApp(
            c,
            "voice-test",
            wake="hey hermes",
            gate=WakeGate(),
            mic=FakeMic(frames),
            transcriber=cloud,
            wake_transcriber=wake,
            segmenter=_segmenter(),
        )
        async with app.run_test() as pilot:
            del pilot
            await until(
                lambda: wake.loads >= 2 or not app._stt_thread.is_alive(),
                timeout=8,
            )
            assert wake.loads >= 2
            assert app._stt_thread.is_alive()
            assert app.status == "loading"
            assert cloud.calls == 0
            assert c.started == []
            assert _log_text(app).count("wake model missing") == 1
            assert wake.times[1] - wake.times[0] >= 5
            app._stop_audio.set()
            await asyncio.to_thread(app._stt_thread.join, 1.0)
            assert not app._stt_thread.is_alive()
    run(go)


def test_show_heard_prints_what_the_wake_model_heard():
    async def go():
        mic = FakeMic([_SPEECH] * 20 + [_QUIET] * 34)
        wake = FakeTranscriber(["hey her mess"])
        cloud = FakeTranscriber([])
        app = VoiceApp(
            FakeClient(),
            "voice-test",
            wake="hey hermes",
            gate=WakeGate(),
            mic=mic,
            transcriber=cloud,
            wake_transcriber=wake,
            segmenter=_segmenter(),
            show_heard=True,
        )
        async with app.run_test() as pilot:
            await until(lambda: wake.calls >= 1)
            await asyncio.sleep(0.2)
            await pilot.pause()
            assert "heard: hey her mess" in _log_text(app)
            assert cloud.calls == 0

    run(go)


def test_heard_text_stays_hidden_by_default():
    async def go():
        mic = FakeMic([_SPEECH] * 20 + [_QUIET] * 34)
        wake = FakeTranscriber(["private room talk"])
        app = VoiceApp(
            FakeClient(),
            "voice-test",
            wake="hey hermes",
            gate=WakeGate(),
            mic=mic,
            transcriber=FakeTranscriber([]),
            wake_transcriber=wake,
            segmenter=_segmenter(),
        )
        async with app.run_test() as pilot:
            await until(lambda: wake.calls >= 1)
            await asyncio.sleep(0.2)
            await pilot.pause()
            assert "private room talk" not in _log_text(app)

    run(go)


def _wake_check_app(wake, cloud, client):
    return VoiceApp(
        client,
        "voice-test",
        wake="hey hermes",
        gate=WakeGate(),
        mic=FakeMic([_SPEECH] * 20 + [_QUIET] * 34),
        transcriber=cloud,
        wake_transcriber=wake,
        segmenter=_segmenter(),
    )


@pytest.mark.parametrize("flag", ["_mic_lost", "_speaking"])
def test_matching_utterance_is_not_uploaded_if_listening_stopped_during_the_check(flag):
    async def go():
        wake = FakeTranscriber(["hey hermes open the logs"], hold=True)
        cloud = FakeTranscriber(["open the logs"])
        c = FakeClient()
        app = _wake_check_app(wake, cloud, c)
        async with app.run_test() as pilot:
            await until(wake.entered.is_set)
            with app._audio_lock:
                setattr(app, flag, True)
                app._close_gate()
            with app._audio_lock:
                setattr(app, flag, False)  # a glitch that recovers before the check ends
            wake.release.set()
            await asyncio.sleep(0.3)
            await pilot.pause()
            assert cloud.calls == 0
            assert c.started == []
            assert app.gate.awake is False

    run(go)


def test_gate_closing_after_a_match_stops_the_upload():
    async def go():
        wake = FakeTranscriber(["hey hermes open the logs"])
        cloud = FakeTranscriber(["open the logs"])
        c = FakeClient()
        app = _wake_check_app(wake, cloud, c)

        def closes_the_gate():
            with app._audio_lock:
                app.gate.sleep()

        app.bell = closes_the_gate
        async with app.run_test() as pilot:
            await until(lambda: wake.calls >= 1)
            await asyncio.sleep(0.3)
            await pilot.pause()
            assert cloud.calls == 0
            assert c.started == []

    run(go)

