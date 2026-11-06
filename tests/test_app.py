import asyncio
import concurrent.futures
import threading
import time

import httpx
import pytest
from textual.widgets import RichLog, Sparkline

from hermes_voice.app import VoiceApp
from hermes_voice.audio import FRAME_BYTES, MIC_SILENT_HINT, MicError, Segmenter
from hermes_voice.hermes import HermesAuthError, HermesUnavailable, RunEvent
from hermes_voice.stt import is_noise


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
        app = VoiceApp(c, "voice-test", wake="hermes")
        async with app.run_test() as pilot:
            app.on_transcript("what's for dinner")
            app.on_transcript("hermes open the logs"); await pilot.pause()
            assert c.started == ["open the logs"]
    run(go)


def test_wake_phrase_noise_does_not_start_a_run():
    async def go():
        c = FakeClient()
        app = VoiceApp(c, "voice-test", wake="hermes")
        async with app.run_test() as pilot:
            app.on_transcript("Hermes, thank you.")
            await pilot.pause()
            assert c.started == []
            assert "thank you" not in _log_text(app)
            assert app._pending == []
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
        self.entered = threading.Event()
        self.release = threading.Event()

    def load(self) -> None:
        return None

    def transcribe(self, pcm: bytes) -> str:
        del pcm
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
