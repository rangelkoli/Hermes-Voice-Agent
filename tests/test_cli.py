import os
import re
import subprocess
import sys
from types import SimpleNamespace as NS

import httpx
import pytest

from hermes_voice.__main__ import main



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


def _capture_app(monkeypatch, tmp_path):
    del tmp_path
    thresholds = []

    class RecordingDetector:
        def __init__(self, threshold=0.5):
            thresholds.append(threshold)
            self.threshold = threshold

        def __call__(self, frame):
            del frame
            return False

        def reset(self):
            return None

    monkeypatch.setattr("hermes_voice.__main__.SpeechDetector", RecordingDetector)
    found = {}

    def run(self):
        found["app"] = self

    monkeypatch.setattr("hermes_voice.__main__.VoiceApp.run", run)
    return found, thresholds


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
    found, _thresholds = _capture_app(monkeypatch, tmp_path)

    main([])

    assert seen["count"] == 2
    assert "app" in found


@pytest.mark.parametrize("answer", ["timeout", 503])
def test_busy_hermes_does_not_block_startup(monkeypatch, tmp_path, answer):
    # The gateway is there but slow. The app already retries sends, so opening it is safe.
    seen = _ready(monkeypatch, tmp_path, health=answer)
    found, _thresholds = _capture_app(monkeypatch, tmp_path)

    main([])

    assert seen["count"] == 1
    assert "app" in found


def test_default_flags_reach_constructors(monkeypatch, tmp_path):
    seen = _ready(monkeypatch, tmp_path)
    found, thresholds = _capture_app(monkeypatch, tmp_path)

    main([])

    app = found["app"]
    assert thresholds == [0.5]
    assert app.segmenter._end_silence_ms == 1000
    assert app.transcriber.model == "gemini-3.5-transcribe"
    assert app.mic._device is None
    from hermes_voice.wake import WakeGate

    assert app.wake == "hey hermes"
    assert isinstance(app.gate, WakeGate)
    assert app.wake_transcriber.model == "base.en"
    assert app.wake_transcriber.hotword == "Hermes"
    assert re.fullmatch(r"voice-\d{8}-\d{6}", app.session_id)
    assert seen["call"][0] == "http://127.0.0.1:8642/health"
    assert seen["call"][1] == 3
    assert seen["call"][2] == {"trust_env": False}
    assert app.client._client.headers["Authorization"] == "Bearer test-key"
    assert str(app.client._client.base_url).rstrip("/") == "http://127.0.0.1:8642"
    assert app.segmenter._is_speech.threshold == 0.5


def test_device_name_reaches_the_mic(monkeypatch, tmp_path):
    _ready(monkeypatch, tmp_path)
    found, _thresholds = _capture_app(monkeypatch, tmp_path)

    main(["--device", "hw:1,0"])

    assert found["app"].mic._device == "hw:1,0"


def test_explicit_flags_reach_constructors(monkeypatch, tmp_path):
    _ready(monkeypatch, tmp_path)
    found, thresholds = _capture_app(monkeypatch, tmp_path)

    main([
        "--wake", "computer",
        "--device", "3",
        "--stt-model", "tiny.en",
        "--vad-threshold", "0.8",
        "--wake-model", "tiny.en",
        "--end-silence-ms", "1500",
    ])

    app = found["app"]
    assert thresholds == [0.8]
    assert app.segmenter._end_silence_ms == 1500
    assert app.transcriber.model == "tiny.en"
    assert app.mic._device == 3
    assert app.wake == "computer"
    assert app.wake_transcriber.model == "tiny.en"
    assert app.wake_transcriber.hotword == "Computer"


def test_health_check_ignores_http_proxy(monkeypatch, tmp_path):
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:9")
    seen = _ready(monkeypatch, tmp_path)
    found, _thresholds = _capture_app(monkeypatch, tmp_path)

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

cli.Mic.check = lambda self: None
cli.httpx.get = lambda *a, **k: httpx.Response(200, request=httpx.Request("GET", "http://x"))
cli.VoiceApp.run = run_inside_textual
sys.argv = ["hermes-voice"]
cli.main()
"""


def test_default_mode_builds_gemini_clients(monkeypatch, tmp_path):
    from hermes_voice.gemini import GeminiSpeaker, GeminiTranscriber

    _ready(monkeypatch, tmp_path)
    found, _thresholds = _capture_app(monkeypatch, tmp_path)

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
    found, _thresholds = _capture_app(monkeypatch, tmp_path)

    main(["--local"])

    from hermes_voice.wake import WakeGate

    app = found["app"]
    assert isinstance(app.transcriber, Transcriber)
    assert app.transcriber.model == "small.en"
    assert app.transcriber.hotword is None
    assert app.speaker is None
    assert isinstance(app.gate, WakeGate)
    assert isinstance(app.wake_transcriber, Transcriber)
    assert app.wake_transcriber.model == "base.en"
    assert app.wake_transcriber.hotword == "Hermes"


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
    found, _thresholds = _capture_app(monkeypatch, tmp_path)

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


def test_default_builds_a_gate_and_wake_model(monkeypatch, capsys, tmp_path):
    from hermes_voice.stt import Transcriber
    from hermes_voice.wake import WakeGate

    _ready(monkeypatch, tmp_path)
    found, thresholds = _capture_app(monkeypatch, tmp_path)

    main([])

    out = capsys.readouterr().out
    assert "Preparing the wake word model..." not in out
    app = found["app"]
    assert isinstance(app.gate, WakeGate)
    assert isinstance(app.wake_transcriber, Transcriber)
    assert app.wake_transcriber.model == "base.en"
    assert app.wake_transcriber.hotword == "Hermes"
    assert thresholds == [0.5]


def test_wake_model_and_threshold_reach_constructors(monkeypatch, tmp_path):
    _ready(monkeypatch, tmp_path)
    found, thresholds = _capture_app(monkeypatch, tmp_path)

    main(["--wake", "hey jarvis", "--wake-model", "tiny.en", "--vad-threshold", "0.4"])

    app = found["app"]
    assert app.wake == "hey jarvis"
    assert app.wake_transcriber.model == "tiny.en"
    assert app.wake_transcriber.hotword == "Jarvis"
    assert thresholds == [0.4]
    assert app.gate is not None


def test_open_mic_builds_no_gate_and_no_wake_model(monkeypatch, capsys, tmp_path):
    _ready(monkeypatch, tmp_path)
    found, _thresholds = _capture_app(monkeypatch, tmp_path)

    main(["--open-mic"])

    app = found["app"]
    assert app.gate is None
    assert app.wake is None
    assert app.wake_transcriber is None
    assert "Preparing the wake word model..." not in capsys.readouterr().out


def test_show_heard_reaches_the_app(monkeypatch, tmp_path):
    _ready(monkeypatch, tmp_path)
    found, _thresholds = _capture_app(monkeypatch, tmp_path)

    main(["--show-heard"])

    assert found["app"].show_heard is True
