import asyncio

from textual.widgets import RichLog, Sparkline

from hermes_voice.app import VoiceApp
from hermes_voice.hermes import HermesUnavailable, RunEvent


class FakeClient:
    """Scripted stand-in for HermesClient.

    A held run waits until release(), and also until aclose(), so a test that
    never releases still ends when the app shuts the client or cancels the worker.
    """

    def __init__(self, script=None, hold=False, fail_starts=0, fail_stop=False, fail_approve=False):
        self.script = list(script or [])
        self.hold = hold
        self.fail_starts = fail_starts
        self.fail_stop = fail_stop
        self.fail_approve = fail_approve
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
