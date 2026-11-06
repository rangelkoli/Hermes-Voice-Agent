import re
import sys
from types import SimpleNamespace as NS

import httpx
import pytest
import webrtcvad

from hermes_voice.__main__ import main
from hermes_voice.audio import FRAME_BYTES, SAMPLE_RATE

_GATEWAY = "Hermes is not running. Start it with: hermes gateway"


def _silence_health(monkeypatch, status=200):
    seen = {}

    def get(url, timeout=None, **kwargs):
        seen["call"] = (url, timeout, kwargs)
        if status == "connect":
            raise httpx.ConnectError("refused")
        return httpx.Response(status, request=httpx.Request("GET", url))

    monkeypatch.setattr("hermes_voice.__main__.httpx.get", get)
    return seen


def _ready(monkeypatch, tmp_path, *, health=200):
    monkeypatch.delenv("HERMES_API_KEY", raising=False)
    monkeypatch.setenv("HERMES_API_KEY", "test-key")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr("hermes_voice.__main__.Mic.check", lambda self: None)
    return _silence_health(monkeypatch, health)


def _capture_app(monkeypatch):
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


@pytest.mark.parametrize("status", ["connect", 503])
def test_health_failure_exits_1_with_gateway_message(monkeypatch, capsys, tmp_path, status):
    _ready(monkeypatch, tmp_path, health=status)

    with pytest.raises(SystemExit) as exc:
        main([])

    assert exc.value.code == 1
    captured = capsys.readouterr()
    text = captured.out + captured.err
    assert _GATEWAY in text
    assert "Traceback" not in text
    assert "test-key" not in text


def test_default_flags_reach_constructors(monkeypatch, tmp_path):
    seen = _ready(monkeypatch, tmp_path)
    found, modes, rates = _capture_app(monkeypatch)

    main([])

    app = found["app"]
    assert modes == [2]
    assert app.segmenter._end_silence_ms == 1000
    assert app.transcriber.model == "small.en"
    assert app.mic._device is None
    assert app.wake is None
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
    found, _modes, _rates = _capture_app(monkeypatch)

    main(["--device", "hw:1,0"])

    assert found["app"].mic._device == "hw:1,0"


def test_explicit_flags_reach_constructors(monkeypatch, tmp_path):
    _ready(monkeypatch, tmp_path)
    found, modes, rates = _capture_app(monkeypatch)

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
    found, _modes, _rates = _capture_app(monkeypatch)

    main([])

    assert seen["call"][2]["trust_env"] is False
    assert found["app"].client._client.trust_env is False


@pytest.mark.parametrize("value", ["0", "-1"])
def test_end_silence_must_be_positive(monkeypatch, tmp_path, value):
    _ready(monkeypatch, tmp_path)
    _capture_app(monkeypatch)

    with pytest.raises(SystemExit) as exc:
        main(["--end-silence-ms", value])

    assert exc.value.code == 2
