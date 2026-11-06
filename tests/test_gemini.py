import base64
import io
import json
import sys
import threading
import wave
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from hermes_voice.gemini import (
    GeminiError,
    GeminiSpeaker,
    GeminiTranscriber,
    load_gemini_key,
    speakable,
)

_KEY = "super-secret-gemini-key"
_PCM = b"\x01\x02\x03\x04" * 40


def _completed(text: str) -> dict:
    return {
        "status": "completed",
        "steps": [
            {
                "type": "model_output",
                "content": [{"type": "text", "text": text}],
            }
        ],
    }


def _transcriber(handler, model="gemini-3.5-transcribe"):
    return GeminiTranscriber(_KEY, model=model, transport=httpx.MockTransport(handler))


def _speaker(handler, model="gemini-3.8-flash-lite-tts", voice="Kore"):
    return GeminiSpeaker(_KEY, model=model, voice=voice, transport=httpx.MockTransport(handler))


def _wav_frames(data: bytes) -> bytes:
    with wave.open(io.BytesIO(data), "rb") as handle:
        assert handle.getnchannels() == 1
        assert handle.getsampwidth() == 2
        assert handle.getframerate() == 16000
        return handle.readframes(handle.getnframes())


def test_transcribe_sends_inline_wav_with_key_in_header_only():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["header"] = request.headers["x-goog-api-key"]
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=_completed("noted"))

    assert _transcriber(handler).transcribe(_PCM) == "noted"
    body = seen["body"]
    audio = body["input"][0]
    assert _KEY not in seen["url"]
    assert seen["header"] == _KEY
    assert body["model"] == "models/gemini-3.5-transcribe"
    assert audio["mime_type"] == "audio/wav"
    assert _wav_frames(base64.b64decode(audio["data"])) == _PCM
    assert body["generation_config"]["transcription_config"]["custom_vocabulary"] == ["Hermes"]


def test_transcribe_joins_model_output_text():
    payload = {
        "status": "completed",
        "steps": [
            {
                "type": "model_output",
                "content": [
                    {"type": "text", "text": "hello"},
                    {"type": "text", "text": " there"},
                ],
            },
            {
                "type": "other",
                "content": [{"type": "text", "text": "ignored"}],
            },
        ],
    }

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload)

    assert _transcriber(handler).transcribe(_PCM) == "hello there"


def test_transcribe_retries_once_on_transport_error_then_succeeds():
    calls = {"n": 0}

    def handler(_request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            raise httpx.ConnectError("refused")
        return httpx.Response(200, json=_completed("ok"))

    assert _transcriber(handler).transcribe(_PCM) == "ok"
    assert calls["n"] == 2


def test_transcribe_retries_once_on_503_then_raises_gemini_error():
    calls = {"n": 0}

    def handler(_request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(503, json={"error": {"message": "unavailable"}})

    with pytest.raises(GeminiError, match="unavailable"):
        _transcriber(handler).transcribe(_PCM)
    assert calls["n"] == 2


def test_transcribe_bad_key_raises_gemini_error_without_the_key_in_message():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["x-goog-api-key"] == _KEY
        assert _KEY not in str(request.url)
        return httpx.Response(
            400, json={"error": {"message": f"API key {_KEY} not valid"}}
        )

    with pytest.raises(GeminiError) as exc:
        _transcriber(handler).transcribe(_PCM)
    message = str(exc.value)
    assert _KEY not in message
    assert "not valid" in message


def _sse(*chunks: bytes) -> bytes:
    lines = [
        "event: interaction.created",
        'data: {"event_type": "interaction.created"}',
        "",
        "data: not-json",
        "",
    ]
    for chunk in chunks:
        encoded = base64.b64encode(chunk).decode()
        lines += [
            "event: step.delta",
            "data: "
            + json.dumps(
                {
                    "event_type": "step.delta",
                    "delta": {"type": "audio", "data": encoded},
                }
            ),
            "",
        ]
    lines += [
        "event: step.delta",
        'data: {"event_type": "step.delta", "delta": {"type": "text", "text": "nope"}}',
        "",
        "event: interaction.completed",
        'data: {"event_type": "interaction.completed"}',
        "",
    ]
    return "\n".join(lines).encode()


def test_stream_yields_decoded_audio_and_ignores_other_events():
    first, second = b"\x01\x02", b"\x03\x04"

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=_sse(first, second))

    assert list(_speaker(handler).stream("Hello")) == [first, second]


def test_stream_sends_voice_and_model():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["header"] = request.headers["x-goog-api-key"]
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, content=_sse(b"\x00\x00"))

    list(_speaker(handler, model="gemini-3.8-flash-lite-tts", voice="Kore").stream("Hello"))
    body = seen["body"]
    assert _KEY not in seen["url"]
    assert seen["header"] == _KEY
    assert body["model"] == "gemini-3.8-flash-lite-tts"
    assert body["stream"] is True
    assert body["input"] == [
        {"type": "user_input", "content": [{"type": "text", "text": "Hello"}]}
    ]
    assert body["response_format"] == {
        "type": "audio",
        "mime_type": "audio/l16",
        "sample_rate": 24000,
    }
    assert body["generation_config"]["speech_config"] == [{"voice": "Kore"}]


def test_stream_error_status_raises_gemini_error():
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"error": {"message": "bad voice"}})

    with pytest.raises(GeminiError, match="bad voice"):
        list(_speaker(handler).stream("Hello"))


def test_load_checks_the_key():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["method"] = request.method
        seen["path"] = request.url.path
        seen["url"] = str(request.url)
        return httpx.Response(400, json={"error": {"message": "key invalid"}})

    with pytest.raises(GeminiError, match="key invalid"):
        _transcriber(handler).load()
    assert seen["method"] == "GET"
    assert seen["path"].endswith("/models/gemini-3.5-transcribe")
    assert _KEY not in seen["url"]


def test_clients_ignore_proxy_settings(monkeypatch):
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:9")
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:9")
    monkeypatch.setenv("ALL_PROXY", "http://127.0.0.1:9")
    transcriber = _transcriber(lambda _request: httpx.Response(200))
    speaker = _speaker(lambda _request: httpx.Response(200))
    assert transcriber._client.trust_env is False
    assert speaker._client.trust_env is False


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("**Done** with `git status`.", "Done with git status."),
        ("# Title\n\nBody text.", "Title\n\nBody text."),
        ("See [the logs](http://x.test/a) now.", "See the logs now."),
        ("- one\n- two", "one\ntwo"),
        ("Before\n```py\nprint(1)\n```\nAfter", "Before\n\nAfter"),
        ("| a | b |\n|---|---|\n| 1 | 2 |\nAfter", "After"),
        ("```\ncode only\n```", ""),
    ],
)
def test_speakable(text, expected):
    assert speakable(text) == expected


def test_speakable_caps_long_text_at_a_sentence_end():
    text = " ".join(["This is one sentence."] * 400)
    result = speakable(text)
    assert len(result) <= 1300
    assert result.endswith("The rest is on screen.")
    assert result[: -len(" The rest is on screen.")].endswith(".")


def test_key_comes_from_env_then_file(tmp_path, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text("GEMINI_API_KEY=from-file\n")
    assert load_gemini_key(env_file) == "from-file"
    monkeypatch.setenv("GEMINI_API_KEY", "from-env")
    assert load_gemini_key(env_file) == "from-env"


def test_missing_key_names_the_file_and_local_flag(tmp_path, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    missing = tmp_path / ".env"
    with pytest.raises(GeminiError) as exc:
        load_gemini_key(missing)
    message = str(exc.value)
    assert str(missing) in message
    assert "aistudio.google.com/apikey" in message
    assert "--local" in message
    assert _KEY not in message


def test_default_key_file_is_under_config_home(tmp_path, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    with pytest.raises(GeminiError) as exc:
        load_gemini_key()
    assert str(tmp_path / ".config" / "hermes-voice" / ".env") in str(exc.value)


class _Out:
    def start(self):
        return None

    def write(self, data):
        del data

    def stop(self):
        return None

    def abort(self):
        return None

    def close(self):
        return None


@pytest.mark.parametrize("stall", ["request", "body"])
def test_stop_ends_speech_during_a_stalled_network_call(monkeypatch, stall):
    # The stall ignores close(), like a blocked TLS read on macOS.
    started = threading.Event()
    release = threading.Event()

    class _Stream(httpx.SyncByteStream):
        def __iter__(self):
            started.set()
            release.wait(5)
            yield b""

    class _Transport(httpx.BaseTransport):
        def handle_request(self, request):
            del request
            if stall == "request":
                started.set()
                release.wait(5)
            return httpx.Response(200, stream=_Stream())

    monkeypatch.setitem(
        sys.modules,
        "sounddevice",
        SimpleNamespace(RawOutputStream=lambda **_kwargs: _Out()),
    )
    speaker = GeminiSpeaker(_KEY, transport=_Transport())
    stop = threading.Event()
    errors: list[BaseException] = []

    def run():
        try:
            speaker.say("Hello", stop)
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    assert started.wait(2)
    stop.set()
    thread.join(1)
    alive = thread.is_alive()
    release.set()
    assert not alive
    assert errors == []


def test_say_raises_a_speech_failure(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "sounddevice",
        SimpleNamespace(RawOutputStream=lambda **_kwargs: _Out()),
    )
    transport = httpx.MockTransport(lambda request: httpx.Response(403, json={"error": {"message": "denied"}}))
    speaker = GeminiSpeaker(_KEY, transport=transport)
    with pytest.raises(GeminiError, match="denied"):
        speaker.say("Hello", threading.Event())
