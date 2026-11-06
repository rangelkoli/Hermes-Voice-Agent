import os
import re
import subprocess
import sys
from types import SimpleNamespace as NS

import httpx
import pytest
import webrtcvad

from hermes_voice.__main__ import main
from hermes_voice.audio import FRAME_BYTES, SAMPLE_RATE



def _silence_health(monkeypatch, status=200):
    seen = {"count": 0}
    answers = status if isinstance(status, list) else None

    def get(url, timeout=None, **kwargs):
        seen["call"] = (url, timeout, kwargs)
        seen["count"] += 1
        answer = answers.pop(0) if answers else status
        if answer == "connect":
            raise httpx.ConnectError("refused")
        if answer == "timeout":
            raise httpx.ReadTimeout("slow")
        return httpx.Response(answer, request=httpx.Request("GET", url))

    monkeypatch.setattr("hermes_voice.__main__.httpx.get", get)
    monkeypatch.setattr("hermes_voice.__main__.time.sleep", lambda _s: None)
    return seen


def _ready(monkeypatch, tmp_path, *, health=200):
    monkeypatch.delenv("HERMES_API_KEY", raising=False)
    monkeypatch.setenv("HERMES_API_KEY", "test-key")
    monkeypatch.setenv("GEMINI_API_KEY", "gem-test-key")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr("hermes_voice.__main__.Mic.check", lambda self: None)
    return _silence_health(monkeypatch, health)


def _stub_wake(monkeypatch, tmp_path):
    model = tmp_path / "wake-model"
    model.mkdir(exist_ok=True)
    seen = {"ensure": 0, "phrase": None, "threshold": None, "model_dir": None}

    def ensure_model(root=None, download=None):
        seen["ensure"] += 1
        seen["root"] = root
        seen["download"] = download
        return model

    class Spotter:
        def __init__(self, phrase, model_dir, threshold=0.25):
            seen["phrase"] = phrase
            seen["model_dir"] = model_dir
            seen["threshold"] = threshold

        def feed(self, frame):
            del frame
            return False

        def reset(self):
            return None

    monkeypatch.setattr("hermes_voice.__main__.ensure_model", ensure_model, raising=False)
    monkeypatch.setattr("hermes_voice.__main__.WakeSpotter", Spotter, raising=False)
    return seen


def _capture_app(monkeypatch, tmp_path):
    modes = []
    rates = []

    class RecordingVad(webrtcvad.Vad):
        def __init__(self, mode=None):
            modes.append(mode)
            super().__init__(mode)

        def is_speech(self, buf, sample_rate, length=None):
            rates.append(sample_rate)
            return super().is_speech(buf, sample_rate, length)

    monkeypatch.setattr("hermes_voice.__main__.webrtcvad.Vad", RecordingVad)
    found = {}

    def run(self):
        found["app"] = self

    monkeypatch.setattr("hermes_voice.__main__.VoiceApp.run", run)
    _stub_wake(monkeypatch, tmp_path)
    return found, modes, rates


def test_list_devices_prints_mic_error_and_exits_1(monkeypatch, capsys):
    class PortAudioError(Exception):
        pass

    def query_devices(*_args, **_kwargs):
        raise PortAudioError("Host error -9996")

    monkeypatch.setitem(
        sys.modules,
        "sounddevice",
        NS(PortAudioError=PortAudioError, query_devices=query_devices),
    )
    called = _silence_health(monkeypatch)

    with pytest.raises(SystemExit) as exc:
        main(["--list-devices"])

    assert exc.value.code == 1
    captured = capsys.readouterr()
    text = captured.out + captured.err
    assert "Host error -9996" in text
    assert "local microphone" in text
    assert "--list-devices" not in text
    assert "Traceback" not in text
    assert "call" not in called


def test_missing_key_exits_1_without_traceback(monkeypatch, capsys, tmp_path):
    monkeypatch.delenv("HERMES_API_KEY", raising=False)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr("hermes_voice.__main__.Mic.check", lambda self: None)
    called = _silence_health(monkeypatch)

    with pytest.raises(SystemExit) as exc:
        main([])

    assert exc.value.code == 1
    captured = capsys.readouterr()
    text = captured.out + captured.err
    assert "API_SERVER_KEY" in text
    assert "Traceback" not in text
    assert "call" not in called


def test_refused_connection_retries_then_exits_1_with_the_fix(monkeypatch, capsys, tmp_path):
    seen = _ready(monkeypatch, tmp_path, health="connect")

    with pytest.raises(SystemExit) as exc:
        main([])

    assert exc.value.code == 1
    assert seen["count"] == 3
    captured = capsys.readouterr()
    text = captured.out + captured.err
    assert "hermes gateway" in text
    assert "API_SERVER_ENABLED" in text
    assert "refused" in text
    assert "Traceback" not in text
    assert "test-key" not in text


def test_refused_once_then_up_starts_the_app(monkeypatch, tmp_path):
    seen = _ready(monkeypatch, tmp_path, health=["connect", 200])
    found, _modes, _rates = _capture_app(monkeypatch, tmp_path)

    main([])

    assert seen["count"] == 2
    assert "app" in found


@pytest.mark.parametrize("answer", ["timeout", 503])
def test_busy_hermes_does_not_block_startup(monkeypatch, tmp_path, answer):
    # The gateway is there but slow. The app already retries sends, so opening it is safe.
    seen = _ready(monkeypatch, tmp_path, health=answer)
    found, _modes, _rates = _capture_app(monkeypatch, tmp_path)

    main([])

    assert seen["count"] == 1
    assert "app" in found


def test_default_flags_reach_constructors(monkeypatch, tmp_path):
    seen = _ready(monkeypatch, tmp_path)
    found, modes, rates = _capture_app(monkeypatch, tmp_path)

    main([])

    app = found["app"]
    assert modes == [2]
    assert app.segmenter._end_silence_ms == 1000
    assert app.transcriber.model == "gemini-3.5-transcribe"
    assert app.mic._device is None
    assert app.wake == "hey hermes"
    assert re.fullmatch(r"voice-\d{8}-\d{6}", app.session_id)
    assert seen["call"][0] == "http://127.0.0.1:8642/health"
    assert seen["call"][1] == 3
    assert seen["call"][2] == {"trust_env": False}
    assert app.client._client.headers["Authorization"] == "Bearer test-key"
    assert str(app.client._client.base_url).rstrip("/") == "http://127.0.0.1:8642"
    assert app.segmenter._is_speech(b"\x00" * FRAME_BYTES) in (True, False)
    assert rates == [SAMPLE_RATE]
    assert app.segmenter._is_speech.__code__.co_varnames[0] == "f"


def test_device_name_reaches_the_mic(monkeypatch, tmp_path):
    _ready(monkeypatch, tmp_path)
    found, _modes, _rates = _capture_app(monkeypatch, tmp_path)

    main(["--device", "hw:1,0"])

    assert found["app"].mic._device == "hw:1,0"


def test_explicit_flags_reach_constructors(monkeypatch, tmp_path):
    _ready(monkeypatch, tmp_path)
    found, modes, rates = _capture_app(monkeypatch, tmp_path)

    main([
        "--wake", "hermes",
        "--device", "3",
        "--stt-model", "tiny.en",
        "--vad-level", "1",
        "--end-silence-ms", "1500",
    ])

    app = found["app"]
    assert modes == [1]
    assert app.segmenter._end_silence_ms == 1500
    assert app.transcriber.model == "tiny.en"
    assert app.mic._device == 3
    assert app.wake == "hermes"
    app.segmenter._is_speech(b"\x00" * FRAME_BYTES)
    assert rates == [SAMPLE_RATE]


def test_health_check_ignores_http_proxy(monkeypatch, tmp_path):
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:9")
    seen = _ready(monkeypatch, tmp_path)
    found, _modes, _rates = _capture_app(monkeypatch, tmp_path)

    main([])

    assert seen["call"][2]["trust_env"] is False
    assert found["app"].client._client.trust_env is False


@pytest.mark.parametrize("value", ["0", "-1"])
def test_end_silence_must_be_positive(monkeypatch, tmp_path, value):
    _ready(monkeypatch, tmp_path)
    _capture_app(monkeypatch, tmp_path)

    with pytest.raises(SystemExit) as exc:
        main(["--end-silence-ms", value])

    assert exc.value.code == 2


_LOCK_INSIDE_UI = """
import asyncio, multiprocessing, sys
import httpx
from textual.app import App
from hermes_voice import __main__ as cli

def run_inside_textual(self):
    async def go():
        async with App().run_test():
            multiprocessing.RLock()  # what tqdm does while Whisper loads
    asyncio.run(go())

class _Spotter:
    def __init__(self, phrase, model_dir, threshold=0.25):
        pass
    def feed(self, frame):
        return False
    def reset(self):
        pass

cli.Mic.check = lambda self: None
cli.httpx.get = lambda *a, **k: httpx.Response(200, request=httpx.Request("GET", "http://x"))
cli.VoiceApp.run = run_inside_textual
cli.ensure_model = lambda root=None, download=None: __import__("pathlib").Path(".")
cli.WakeSpotter = _Spotter
sys.argv = ["hermes-voice"]
cli.main()
"""


def test_default_mode_builds_gemini_clients(monkeypatch, tmp_path):
    from hermes_voice.gemini import GeminiSpeaker, GeminiTranscriber

    _ready(monkeypatch, tmp_path)
    found, _modes, _rates = _capture_app(monkeypatch, tmp_path)

    main([])

    app = found["app"]
    assert isinstance(app.transcriber, GeminiTranscriber)
    assert isinstance(app.speaker, GeminiSpeaker)
    assert app.transcriber.model == "gemini-3.5-transcribe"
    assert app.speaker.model == "gemini-3.8-flash-lite-tts"
    assert app.speaker.voice == "Kore"
    assert app.transcriber._client.headers["x-goog-api-key"] == "gem-test-key"
    assert "gem-test-key" not in str(app.transcriber._client.base_url)
    assert app.speaker._client.trust_env is False


def test_local_uses_whisper_and_no_speaker_and_needs_no_key(monkeypatch, tmp_path):
    import pathlib

    from hermes_voice.stt import Transcriber

    monkeypatch.setattr(pathlib.Path, "home", lambda: tmp_path)
    _ready(monkeypatch, tmp_path)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    found, _modes, _rates = _capture_app(monkeypatch, tmp_path)

    main(["--local"])

    app = found["app"]
    assert isinstance(app.transcriber, Transcriber)
    assert app.transcriber.model == "small.en"
    assert app.speaker is None


def test_missing_gemini_key_exits_1_with_the_message(monkeypatch, capsys, tmp_path):
    import pathlib

    monkeypatch.setattr(pathlib.Path, "home", lambda: tmp_path)
    _ready(monkeypatch, tmp_path)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    _capture_app(monkeypatch, tmp_path)

    with pytest.raises(SystemExit) as exc:
        main([])

    assert exc.value.code == 1
    text = capsys.readouterr().err
    expected = tmp_path / ".config" / "hermes-voice" / ".env"
    assert str(expected) in text
    assert "aistudio.google.com/apikey" in text
    assert "--local" in text
    assert "Traceback" not in text
    assert "gem-test-key" not in text


def test_voice_and_model_flags_reach_the_speaker(monkeypatch, tmp_path):
    _ready(monkeypatch, tmp_path)
    found, _modes, _rates = _capture_app(monkeypatch, tmp_path)

    main([
        "--tts-model", "gemini-custom-tts",
        "--voice", "Puck",
        "--stt-model", "gemini-custom-stt",
    ])

    app = found["app"]
    assert app.transcriber.model == "gemini-custom-stt"
    assert app.speaker.model == "gemini-custom-tts"
    assert app.speaker.voice == "Puck"


def test_multiprocessing_lock_works_inside_the_ui():
    # Needs a fresh process: the tracker is global, so an earlier test could hide the bug.
    env = {**os.environ, "HERMES_API_KEY": "test-key", "GEMINI_API_KEY": "test-gemini-key"}
    done = subprocess.run(
        [sys.executable, "-c", _LOCK_INSIDE_UI], env=env, capture_output=True, text=True, timeout=60
    )
    assert done.returncode == 0, done.stderr[-500:]


def test_default_builds_a_gate_for_hey_hermes(monkeypatch, capsys, tmp_path):
    from hermes_voice.wake import MODEL_DIR, WakeGate

    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    _ready(monkeypatch, tmp_path)
    found, _modes, _rates = _capture_app(monkeypatch, tmp_path)
    seen = _stub_wake(monkeypatch, tmp_path)

    main([])

    out = capsys.readouterr().out
    assert "Preparing the wake word model..." in out
    app = found["app"]
    assert app.wake == "hey hermes"
    assert isinstance(app.gate, WakeGate)
    assert seen["phrase"] == "hey hermes"
    assert seen["threshold"] == 0.25
    assert seen["ensure"] == 1

    cached = tmp_path / "cache" / "hermes-voice" / "wake" / MODEL_DIR
    cached.mkdir(parents=True)
    (cached / "tokens.txt").write_text("tok", encoding="utf-8")
    main([])
    assert "Preparing the wake word model..." not in capsys.readouterr().out


def test_wake_flag_and_threshold_reach_the_spotter(monkeypatch, tmp_path):
    _ready(monkeypatch, tmp_path)
    found, _modes, _rates = _capture_app(monkeypatch, tmp_path)
    seen = _stub_wake(monkeypatch, tmp_path)

    main(["--wake", "computer", "--wake-threshold", "0.4"])

    app = found["app"]
    assert app.wake == "computer"
    assert seen["phrase"] == "computer"
    assert seen["threshold"] == 0.4
    assert seen["model_dir"] == tmp_path / "wake-model"
    assert app.gate is not None


def test_open_mic_builds_no_gate_and_downloads_no_model(monkeypatch, capsys, tmp_path):
    _ready(monkeypatch, tmp_path)
    found, _modes, _rates = _capture_app(monkeypatch, tmp_path)
    seen = _stub_wake(monkeypatch, tmp_path)

    main(["--open-mic"])

    app = found["app"]
    assert app.gate is None
    assert app.wake is None
    assert seen["ensure"] == 0
    assert "Preparing the wake word model..." not in capsys.readouterr().out


def test_model_failure_exits_1_with_the_message(monkeypatch, capsys, tmp_path):
    from hermes_voice.wake import WakeError

    _ready(monkeypatch, tmp_path)
    found, _modes, _rates = _capture_app(monkeypatch, tmp_path)
    message = "The wake word model could not be prepared. Run with --open-mic, which needs no model."

    def boom(root=None, download=None):
        del root, download
        raise WakeError(message)

    monkeypatch.setattr("hermes_voice.__main__.ensure_model", boom)

    with pytest.raises(SystemExit) as exc:
        main([])

    assert exc.value.code == 1
    captured = capsys.readouterr()
    assert message in captured.err
    assert "Traceback" not in captured.err
    assert "app" not in found
