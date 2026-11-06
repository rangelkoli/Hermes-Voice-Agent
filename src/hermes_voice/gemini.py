"""Gemini transcription and spoken replies."""

import base64
import io
import json
import os
import re
import threading
import wave
from collections.abc import Iterator
from pathlib import Path

import httpx
from dotenv import dotenv_values

from hermes_voice.audio import play

_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
# ponytail: 1200 characters bounds TTS cost and listening time.
_SPEAK_LIMIT = 1200
_SPEAK_NOTE = " The rest is on screen."
_FENCE = re.compile(r"```.*?```", re.DOTALL)
_LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_LIST = re.compile(r"^[-*+]\s+")
_HEADING = re.compile(r"^#+\s*")
# Emphasis underscores only. snake_case keeps its underscores.
_LOOSE_UNDERSCORE = re.compile(r"(?<![A-Za-z0-9])_|_(?![A-Za-z0-9])")


class GeminiError(Exception):
    pass


def load_gemini_key(env_path: Path | None = None) -> str:
    key = os.environ.get("GEMINI_API_KEY")
    if key:
        return key
    if env_path is None:
        env_path = Path.home() / ".config" / "hermes-voice" / ".env"
    key = dotenv_values(env_path).get("GEMINI_API_KEY")
    if not key:
        raise GeminiError(
            f"No Gemini API key in {env_path}. "
            "Create one at aistudio.google.com/apikey. --local needs none."
        )
    return key


def speakable(text: str) -> str:
    cleaned = _LINK.sub(r"\1", _FENCE.sub("", text))
    lines = []
    for line in cleaned.split("\n"):
        stripped = line.strip()
        if stripped.startswith("|"):
            continue
        stripped = _HEADING.sub("", _LIST.sub("", stripped))
        stripped = stripped.replace("*", "").replace("`", "")
        stripped = _LOOSE_UNDERSCORE.sub("", stripped)
        lines.append(stripped)
    cleaned = re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()
    if len(cleaned) <= _SPEAK_LIMIT:
        return cleaned
    cut = cleaned[:_SPEAK_LIMIT]
    end = max(cut.rfind("."), cut.rfind("!"), cut.rfind("?"))
    head = cut if end < 0 else cut[: end + 1]
    return head.rstrip() + _SPEAK_NOTE


def _client(api_key: str, transport) -> httpx.Client:
    return httpx.Client(
        base_url=_BASE_URL,
        headers={"x-goog-api-key": api_key},
        trust_env=False,
        timeout=30,
        transport=transport,
    )


def _scrub(message: str, api_key: str) -> str:
    if api_key:
        message = message.replace(api_key, "")
    message = message.strip()
    return message or "Gemini request failed"


def _error_text(response: httpx.Response, api_key: str) -> str:
    message = ""
    try:
        payload = response.json()
    except Exception:
        payload = None
    if isinstance(payload, dict):
        error = payload.get("error")
        if isinstance(error, dict) and error.get("message"):
            message = str(error["message"])
        elif isinstance(error, str):
            message = error
    if not message:
        try:
            message = response.text.strip()
        except Exception:
            message = ""
    if not message:
        message = f"Gemini request failed ({response.status_code})"
    return _scrub(message, api_key)


def _wav(pcm: bytes) -> str:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(pcm)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _model_name(model: str) -> str:
    if model.startswith("models/"):
        return model
    return f"models/{model}"


def _transcript(payload: dict) -> str:
    parts: list[str] = []
    for step in payload.get("steps") or []:
        if step.get("type") != "model_output":
            continue
        for item in step.get("content") or []:
            if item.get("type") == "text":
                parts.append(str(item.get("text") or ""))
    return "".join(parts)


class GeminiTranscriber:
    def __init__(self, api_key: str, model: str = "gemini-3.5-transcribe", transport=None):
        self.model = model
        self._api_key = api_key
        self._client = _client(api_key, transport)

    def load(self) -> None:
        name = self.model.removeprefix("models/")
        try:
            response = self._client.get(f"/models/{name}")
        except httpx.HTTPError as exc:
            raise GeminiError(_scrub(str(exc), self._api_key)) from exc
        if response.status_code >= 400:
            raise GeminiError(_error_text(response, self._api_key))

    def transcribe(self, pcm: bytes) -> str:
        body = {
            "model": _model_name(self.model),
            "input": [{"type": "audio", "data": _wav(pcm), "mime_type": "audio/wav"}],
            "generation_config": {"transcription_config": {"custom_vocabulary": ["Hermes"]}},
        }
        for attempt in range(2):
            try:
                response = self._client.post("/interactions", json=body)
            except httpx.TransportError as exc:
                if attempt == 0:
                    continue
                raise GeminiError(_scrub(str(exc), self._api_key)) from exc
            if response.status_code == 429 or response.status_code >= 500:
                if attempt == 0:
                    continue
                raise GeminiError(_error_text(response, self._api_key))
            if response.status_code >= 400:
                raise GeminiError(_error_text(response, self._api_key))
            return _transcript(response.json())
        raise GeminiError("Gemini request failed")


class GeminiSpeaker:
    def __init__(
        self,
        api_key: str,
        model: str = "gemini-3.8-flash-lite-tts",
        voice: str = "Kore",
        transport=None,
    ):
        self.model = model
        self.voice = voice
        self._api_key = api_key
        self._client = _client(api_key, transport)
        self._stop: threading.Event | None = None
        self._response: httpx.Response | None = None
        self._response_lock = threading.Lock()

    def stream(self, text: str) -> Iterator[bytes]:
        if self._stop is not None and self._stop.is_set():
            return
        body = {
            "model": self.model,
            "stream": True,
            "input": [{"type": "user_input", "content": [{"type": "text", "text": text}]}],
            "response_format": {
                "type": "audio",
                "mime_type": "audio/l16",
                "sample_rate": 24000,
            },
            "generation_config": {"speech_config": [{"voice": self.voice}]},
        }
        with self._client.stream("POST", "/interactions", json=body) as response:
            self._attach(response)
            try:
                if response.status_code >= 400:
                    response.read()
                    raise GeminiError(_error_text(response, self._api_key))
                for line in response.iter_lines():
                    if not line.startswith("data:"):
                        continue
                    raw = line[5:].strip()
                    if not raw:
                        continue
                    try:
                        event = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    delta = event.get("delta")
                    if not isinstance(delta, dict) or delta.get("type") != "audio":
                        continue
                    data = delta.get("data")
                    if not data:
                        continue
                    yield base64.b64decode(data)
            finally:
                self._detach(response)

    def say(self, text: str, stop: threading.Event) -> None:
        if stop.is_set():
            return
        done = threading.Event()
        self._stop = stop
        watcher = threading.Thread(
            target=self._close_when_stopped,
            args=(stop, done),
            name="gemini-stop",
            daemon=True,
        )
        watcher.start()
        try:
            play(self.stream(text), stop)
        except Exception:
            if not stop.is_set():
                raise
        finally:
            done.set()
            self._stop = None
            self._detach(None)
            watcher.join(0.2)

    def _attach(self, response: httpx.Response) -> None:
        with self._response_lock:
            self._response = response
            stopped = self._stop is not None and self._stop.is_set()
        if stopped:
            response.close()

    def _detach(self, response: httpx.Response | None) -> None:
        with self._response_lock:
            if response is None or self._response is response:
                self._response = None

    def _close_when_stopped(self, stop: threading.Event, done: threading.Event) -> None:
        while not done.is_set():
            if stop.wait(0.05):
                break
        if done.is_set() or not stop.is_set():
            return
        with self._response_lock:
            if done.is_set():
                return
            response = self._response
        if response is None:
            return
        try:
            response.close()
        except Exception:
            return
