"""Terminal UI for a Hermes voice conversation."""

import asyncio
from collections import deque

from rich.markdown import Markdown
from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal
from textual.reactive import reactive
from textual.widgets import Footer, RichLog, Sparkline, Static

from hermes_voice.hermes import TERMINAL, HermesAuthError, HermesUnavailable, RunEvent
from hermes_voice.stt import apply_wake

# ponytail: the meter keeps the newest 120 samples. Older levels are dropped.
_METER_KEEP = 120


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
        # ponytail: stored for Task 5. No audio workers run in this task.
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

    def on_transcript(self, text: str) -> None:
        spoken = apply_wake(text, self.wake)
        if spoken is None:
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
                    self.status = "offline"
                    self.set_timer(self._retry_seconds, self._ensure_send)
                    return
                except HermesAuthError:
                    self._stop_sending = True
                    self.log_line("Hermes rejected the API key. Check API_SERVER_KEY.")
                    return
                del self._pending[: len(batch)]
                self._run_id = run_id
                self._approval = None
                self._reply = ""
                self.query_one("#stream", Static).update("")
                self.status = "thinking"
                try:
                    async for event in self.client.events(run_id):
                        self._on_event(event)
                finally:
                    self._run_id = None
                    self._approval = None
                    if self.status in ("thinking", "approval"):
                        self.status = "listening"
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.log_line(f"Send failed: {exc}")
        finally:
            self._busy = False

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
            self.status = "approval"
            self.bell()
        elif name == "run.completed":
            output = data.get("output", "")
            self.query_one("#log", RichLog).write(Markdown("" if output is None else str(output)))
            self._clear_reply()
        elif name == "run.failed":
            self.log_line(str(data.get("error", "The run failed.")))
            self._clear_reply()
        elif name in ("run.cancelled", "run.interrupted"):
            self.log_line("The run ended early.")
            self._clear_reply()

        if name in ("approval.responded", "tool.completed") or name in TERMINAL:
            self._approval = None
            if name not in TERMINAL and self.status == "approval" and self._run_id is not None:
                self.status = "thinking"
        if name in TERMINAL:
            self.status = "listening"

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
        pass

    async def action_stop(self) -> None:
        run_id = self._run_id
        if run_id is None:
            return
        try:
            await self.client.stop(run_id)
        except HermesUnavailable:
            self.log_line("Could not stop the run. Hermes is unavailable.")

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

    async def action_quit(self) -> None:
        self._quit = True
        try:
            await self.client.aclose()
        except Exception as exc:
            self.log_line(f"Could not close Hermes: {exc}")
        finally:
            self.exit()
