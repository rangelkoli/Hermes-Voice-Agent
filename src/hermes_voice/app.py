"""Terminal UI for a Hermes voice conversation."""

import asyncio
import queue
import threading
import time
from collections import deque
from concurrent.futures import CancelledError

import httpx
from rich.markdown import Markdown
from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal
from textual.reactive import reactive
from textual.widgets import Footer, RichLog, Sparkline, Static

from hermes_voice.audio import MIC_SILENT_HINT, MicError, level
from hermes_voice.hermes import TERMINAL, HermesAuthError, HermesUnavailable, RunEvent
from hermes_voice.stt import apply_wake, is_noise

# ponytail: the meter keeps the newest 120 samples. Older levels are dropped.
_METER_KEEP = 120
# ponytail: a dead mic is reopened after a fixed 2 seconds. Quit and mute return early.
_MIC_RETRY_SECONDS = 2.0
# ponytail: a failed model download is retried after a fixed 5 seconds. Quit returns early.
_LOAD_RETRY_SECONDS = 5.0
_AUDIO_WAIT = 0.1
_SILENT_FRAMES = 100


def _status_class(status: str) -> str:
    return "status-" + status.replace(" ", "-")


class VoiceApp(App[None]):
    """Conversation screen for one Hermes session."""

    ENABLE_COMMAND_PALETTE = False
    ESCAPE_TO_MINIMIZE = False

    BINDINGS = [
        Binding("m", "mute", "Mute"),
        Binding("escape", "stop", "Stop run", key_display="esc"),
        Binding("y", "approve", "Approve"),
        Binding("n", "deny", "Deny"),
        Binding("q", "quit", "Quit"),
    ]

    CSS = """
    Screen {
        background: $surface;
    }
    #header {
        height: 1;
        width: 1fr;
    }
    #title {
        width: auto;
        padding: 0 1;
        text-style: bold;
    }
    #status {
        width: auto;
        padding: 0 1;
        text-style: bold;
    }
    #device {
        width: 1fr;
        padding: 0 1;
        content-align: right middle;
        color: $text-muted;
    }
    #meter {
        height: 1;
        margin: 0 1;
    }
    #log {
        height: 1fr;
        border-top: solid $primary;
    }
    #stream {
        height: 1;
        padding: 0 1;
    }
    .status-loading {
        color: #8a8a8a;
    }
    .status-listening {
        color: #3dde7a;
    }
    .status-hearing {
        color: #7dffb2;
    }
    .status-transcribing {
        color: #6cb6ff;
    }
    .status-thinking {
        color: #5fd0e0;
    }
    .status-muted {
        color: #9aa0a6;
    }
    .status-approval {
        background: #ffbf00;
        color: #1a1a1a;
    }
    .status-offline {
        background: #ff2d2d;
        color: #ffffff;
    }
    .status-mic-lost {
        background: #ff4fd8;
        color: #1a1a1a;
    }
    """

    status: reactive[str] = reactive("listening")

    def __init__(
        self,
        client,
        session_id: str,
        wake: str | None = None,
        mic=None,
        transcriber=None,
        segmenter=None,
        retry_seconds: float = 5,
    ) -> None:
        super().__init__()
        self.client = client
        self.session_id = session_id
        self.wake = wake
        self.mic = mic
        self.transcriber = transcriber
        self.segmenter = segmenter
        self._retry_seconds = retry_seconds
        self._pending: list[str] = []
        self._levels: deque[float] = deque(maxlen=_METER_KEEP)
        self._run_id: str | None = None
        self._approval: dict | None = None
        self._reply = ""
        self._busy = False
        self._stop_sending = False
        self._quit = False
        self._muted = False
        self._mic_lost = False
        self._offline = False
        self._hearing = False
        self._transcribing = False
        self._loading = False
        self._retry_mic = False
        self._mute_gen = 0
        self._zero_frames = 0
        self._frame_count = 0
        self._logged_mic_errors: set[str] = set()
        self._utterances: queue.Queue[tuple[bytes, int]] = queue.Queue()
        self._stop_audio = threading.Event()
        self._audio_lock = threading.Lock()
        self._capture_thread: threading.Thread | None = None
        self._stt_thread: threading.Thread | None = None

    def compose(self) -> ComposeResult:
        device = "" if self.mic is None else str(self.mic.name)
        with Horizontal(id="header"):
            yield Static("HERMES VOICE", id="title")
            yield Static("● LISTENING", id="status", classes="status-listening")
            yield Static(device, id="device")
        yield Sparkline(id="meter")
        yield RichLog(id="log", markup=False, highlight=False, wrap=True)
        yield Static("", id="stream", markup=False)
        yield Footer()

    def watch_status(self, status: str) -> None:
        if not self.is_mounted:
            return
        widget = self.query_one("#status", Static)
        widget.set_classes(_status_class(status))
        widget.update(f"● {status.upper()}")

    def on_mount(self) -> None:
        if self.mic is None:
            return
        self._capture_thread = threading.Thread(
            target=self._capture_loop, name="hermes-capture", daemon=True
        )
        self._stt_thread = threading.Thread(
            target=self._stt_loop, name="hermes-stt", daemon=True
        )
        self._capture_thread.start()
        self._stt_thread.start()

    def on_unmount(self) -> None:
        self._quit = True
        self._stop_audio.set()
        if self.mic is not None:
            self.mic.stop()

    def _sync_status(self) -> None:
        if self._mic_lost:
            status = "mic lost"
        elif self._muted:
            status = "muted"
        elif self._approval is not None:
            status = "approval"
        elif self._offline:
            status = "offline"
        elif self._run_id is not None:
            status = "thinking"
        elif self._loading:
            status = "loading"
        elif self._transcribing:
            status = "transcribing"
        elif self._hearing:
            status = "hearing"
        else:
            status = "listening"
        self.status = status

    def _on_ui(self, callback, *args, **kwargs) -> None:
        # ponytail: call_from_thread raises if the loop is already gone. Drop that race.
        if self._quit or not self.is_running:
            return
        try:
            self.call_from_thread(callback, *args, **kwargs)
        except (RuntimeError, CancelledError):
            return
        except Exception:
            if self._quit or not self.is_running:
                return
            raise

    def _repaint(self) -> None:
        self.refresh(repaint=True)

    def _capture_loop(self) -> None:
        try:
            while not self._stop_audio.is_set():
                try:
                    self._capture_once()
                except Exception as exc:
                    self._capture_failed(exc)
        finally:
            self.mic.stop()

    def _capture_once(self) -> None:
        if self._muted:
            self._stop_audio.wait(_AUDIO_WAIT)
            return
        if not self.mic.alive:
            self._recover_mic()
            return
        frame = self.mic.read(_AUDIO_WAIT)
        if frame is None or self._muted or self._stop_audio.is_set():
            return
        self._take_frame(frame)

    def _capture_failed(self, exc: Exception) -> None:
        text = str(exc)
        with self._audio_lock:
            self._mic_lost = True
            fresh = bool(text) and text not in self._logged_mic_errors
            if fresh:
                self._logged_mic_errors.add(text)
        self._on_ui(self._sync_status)
        if fresh:
            self._on_ui(self.log_line, text)
        self._stop_audio.wait(_AUDIO_WAIT)

    def _recover_mic(self) -> None:
        if self._muted or self._stop_audio.is_set():
            return
        if self._retry_mic:
            with self._audio_lock:
                if not self._muted and not self._stop_audio.is_set():
                    self._mic_lost = True
            self._on_ui(self._sync_status)
            if not self._wait_to_retry():
                return
        outcome, message = self._try_start()
        if outcome == "started":
            self._on_ui(self._sync_status)
            self._on_ui(self._repaint)
        elif outcome == "alive":
            self._on_ui(self._sync_status)
        elif outcome == "lost":
            self._on_ui(self._sync_status)
            if message:
                self._on_ui(self.log_line, message)

    def _wait_to_retry(self) -> bool:
        deadline = time.monotonic() + _MIC_RETRY_SECONDS
        while time.monotonic() < deadline:
            if self._stop_audio.is_set() or self._muted:
                return False
            if self.mic.alive:
                with self._audio_lock:
                    if not self._muted:
                        self._mic_lost = False
                self._on_ui(self._sync_status)
                return False
            self._stop_audio.wait(_AUDIO_WAIT)
        return not self._stop_audio.is_set() and not self._muted

    def _try_start(self) -> tuple[str, str | None]:
        message = None
        with self._audio_lock:
            if self._muted or self._stop_audio.is_set():
                return "skip", None
            if self.mic.alive:
                if not self._mic_lost:
                    return "skip", None
                self._mic_lost = False
                return "alive", None
            try:
                self.mic.start()
            except MicError as exc:
                text = str(exc)
                if text not in self._logged_mic_errors:
                    self._logged_mic_errors.add(text)
                    message = text
                self._retry_mic = True
                self._mic_lost = True
                return "lost", message
            self._retry_mic = True
            self._mic_lost = False
            return "started", None

    def _take_frame(self, frame: bytes) -> None:
        self._frame_count += 1
        if self._frame_count % 3 == 0:
            self._on_ui(self.set_level, level(frame))
        if not any(frame):
            self._zero_frames += 1
            if self._zero_frames == _SILENT_FRAMES:
                self._on_ui(self.log_line, MIC_SILENT_HINT)
        else:
            self._zero_frames = 0
        with self._audio_lock:
            if self._muted or self.segmenter is None:
                return
            utterance = self.segmenter.push(frame)
            hearing = bool(self.segmenter.in_speech)
            changed = hearing != self._hearing
            self._hearing = hearing
            if utterance is not None:
                self._utterances.put((utterance, self._mute_gen))
        if changed:
            self._on_ui(self._sync_status)

    def _stt_loop(self) -> None:
        self._load_transcriber()
        while not self._stop_audio.is_set():
            try:
                utterance, generation = self._utterances.get(timeout=_AUDIO_WAIT)
            except queue.Empty:
                continue
            if self._stop_audio.is_set():
                break
            self._transcribing = True
            self._on_ui(self._sync_status)
            failed = False
            try:
                text = "" if self.transcriber is None else self.transcriber.transcribe(utterance)
            except Exception as exc:
                failed = True
                self._on_ui(self.log_line, str(exc))
            finally:
                self._transcribing = False
                self._on_ui(self._sync_status)
            if failed or self._stop_audio.is_set() or is_noise(text):
                continue
            self._on_ui(self._deliver_transcript, text, generation)

    def _load_transcriber(self) -> None:
        self._loading = True
        self._on_ui(self._sync_status)
        logged: str | None = None
        try:
            if self.transcriber is None:
                return
            while not self._stop_audio.is_set():
                try:
                    self.transcriber.load()
                except Exception as exc:
                    text = str(exc)
                    if text != logged:
                        logged = text
                        self._on_ui(self.log_line, text)
                    if not self._wait_for_stop(_LOAD_RETRY_SECONDS):
                        return
                    continue
                return
        finally:
            self._loading = False
            self._on_ui(self._sync_status)

    def _wait_for_stop(self, seconds: float) -> bool:
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            if self._stop_audio.is_set():
                return False
            remaining = deadline - time.monotonic()
            self._stop_audio.wait(min(_AUDIO_WAIT, remaining))
        return not self._stop_audio.is_set()

    def _deliver_transcript(self, text: str, generation: int) -> None:
        if generation != self._mute_gen:
            return
        self.on_transcript(text)

    def _drop_waiting_audio(self) -> None:
        while True:
            try:
                self._utterances.get_nowait()
            except queue.Empty:
                return

    def on_transcript(self, text: str) -> None:
        if self._stop_sending:
            return
        spoken = apply_wake(text, self.wake)
        if spoken is None or is_noise(spoken):
            return
        self.log_line(f"You  {spoken}")
        self._pending.append(spoken)
        if self._run_id is None:
            self._ensure_send()

    def set_level(self, value: float) -> None:
        self._levels.append(float(value))
        if self.is_mounted:
            self.query_one("#meter", Sparkline).data = list(self._levels)

    def log_line(self, text: str) -> None:
        self.query_one("#log", RichLog).write(text)

    def _ensure_send(self) -> None:
        if self._quit or self._stop_sending or self._busy or not self._pending:
            return
        self._busy = True
        self.run_worker(
            self._send,
            name="send",
            group="send",
            exit_on_error=False,
        )

    async def _send(self) -> None:
        try:
            while self._pending and not self._quit and not self._stop_sending:
                batch = list(self._pending)
                text = "\n".join(batch)
                try:
                    run_id = await self.client.start_run(text, self.session_id)
                except HermesUnavailable:
                    self._offline = True
                    self._sync_status()
                    return
                self._offline = False
                del self._pending[: len(batch)]
                self._run_id = run_id
                self._approval = None
                self._reply = ""
                self.query_one("#stream", Static).update("")
                self._sync_status()
                try:
                    async for event in self.client.events(run_id):
                        self._on_event(event)
                finally:
                    self._run_id = None
                    self._approval = None
                    self._sync_status()
        except asyncio.CancelledError:
            raise
        except HermesAuthError:
            self._stop_sending = True
            self._offline = True
            self._sync_status()
            self.log_line(
                "Speech is not being sent until the API key is fixed and the app is restarted."
            )
        except Exception as exc:
            self.log_line(f"Send failed: {exc}")
        finally:
            self._busy = False
            if self._pending and not self._stop_sending and not self._quit:
                self.set_timer(self._retry_seconds, self._ensure_send)

    def _on_event(self, event: RunEvent) -> None:
        name = event.name
        data = event.data
        if name == "message.delta":
            self._reply += str(data.get("delta", ""))
            self.query_one("#stream", Static).update(f"Hermes  {self._reply}")
        elif name in ("tool.started", "tool.completed"):
            self._write_tool(data)
        elif name == "approval.request":
            self._approval = data
            self.log_line(f"Approval  {data.get('command', '')}")
            self.bell()
        elif name == "run.completed":
            output = data.get("output", "")
            self.query_one("#log", RichLog).write(Markdown("" if output is None else str(output)))
            self._clear_reply()
        elif name == "run.failed":
            self.log_line(str(data.get("error", "The run failed.")))
            self._clear_reply()
        elif name in ("run.cancelled", "run.interrupted"):
            if self._reply:
                self.log_line(self._reply)
            self.log_line("The run ended early.")
            self._clear_reply()

        if name in ("approval.responded", "tool.completed") or name in TERMINAL:
            self._approval = None
        if name in TERMINAL:
            self._run_id = None
        self._sync_status()

    def _write_tool(self, data: dict) -> None:
        parts = [str(data.get("tool", ""))]
        preview = data.get("preview")
        if preview:
            parts.append(str(preview))
        if data.get("duration") is not None:
            parts.append(str(data["duration"]))
        if data.get("error"):
            parts.append(str(data["error"]))
        line = "  ".join(part for part in parts if part)
        self.query_one("#log", RichLog).write(Text(f"⚙ {line}", style="dim"))

    def _clear_reply(self) -> None:
        self._reply = ""
        self.query_one("#stream", Static).update("")

    def action_mute(self) -> None:
        if self.mic is None:
            return
        message = None
        started = False
        with self._audio_lock:
            self._muted = not self._muted
            if self._muted:
                self._mute_gen += 1
                self._mic_lost = False
                self.mic.stop()
                if self.segmenter is not None:
                    self.segmenter.reset()
                self._hearing = False
                self._drop_waiting_audio()
            else:
                try:
                    self.mic.start()
                except MicError as exc:
                    self._mic_lost = True
                    text = str(exc)
                    if text not in self._logged_mic_errors:
                        self._logged_mic_errors.add(text)
                        message = text
                else:
                    self._mic_lost = False
                    self._retry_mic = True
                    started = True
        if message:
            self.log_line(message)
        if started:
            self.refresh(repaint=True)
        self._sync_status()

    async def action_stop(self) -> None:
        run_id = self._run_id
        if run_id is None:
            return
        try:
            await self.client.stop(run_id)
        except HermesUnavailable:
            self.log_line("Could not stop the run. Hermes is unavailable.")
        except httpx.HTTPStatusError:
            self.log_line("Could not stop the run. It had already finished.")
        except HermesAuthError:
            self.log_line("Could not stop the run. Hermes rejected the API key.")

    async def action_approve(self) -> None:
        await self._choose("once")

    async def action_deny(self) -> None:
        await self._choose("deny")

    async def _choose(self, choice: str) -> None:
        if self._approval is None or self._run_id is None:
            return
        request_id = self._approval.get("request_id")
        try:
            await self.client.approve(self._run_id, choice, request_id)
        except HermesUnavailable:
            self.log_line("Could not send the approval. Hermes is unavailable.")
        except httpx.HTTPStatusError:
            self.log_line("Could not send the approval. It had already finished.")
        except HermesAuthError:
            self.log_line("Could not send the approval. Hermes rejected the API key.")

    async def action_quit(self) -> None:
        self._quit = True
        self._stop_audio.set()
        if self.mic is not None:
            self.mic.stop()
        await self._join_audio()
        run_id = self._run_id
        if run_id is not None:
            try:
                await self.client.stop(run_id)
            except Exception:
                pass
        try:
            await self.client.aclose()
        except Exception as exc:
            self.log_line(f"Could not close Hermes: {exc}")
        finally:
            self.exit()

    async def _join_audio(self) -> None:
        for thread in (self._capture_thread, self._stt_thread):
            if thread is not None:
                await asyncio.to_thread(thread.join, 1.0)
