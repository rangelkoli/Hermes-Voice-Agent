import sys
from types import SimpleNamespace as NS
import pytest
from hermes_voice.audio import FRAME_BYTES, Mic, MicError, Segmenter, level

SPEECH, QUIET = b"\x01" * FRAME_BYTES, b"\x00" * FRAME_BYTES

def seg(**kw):
    return Segmenter(lambda f: f == SPEECH, **kw)

def feed(s, frames):
    return [out for f in frames if (out := s.push(f)) is not None]

def test_utterance_ends_after_end_silence():
    # 20 frames speech (600 ms), then 34 quiet frames (1020 ms) closes it
    out = feed(seg(), [SPEECH] * 20 + [QUIET] * 34)
    assert len(out) == 1

def test_preroll_is_kept_so_first_word_is_not_clipped():
    s = seg(preroll_ms=90)
    out = feed(s, [QUIET] * 10 + [SPEECH] * 20 + [QUIET] * 34)
    assert len(out[0]) >= (3 + 20) * FRAME_BYTES

def test_short_blip_is_dropped():
    assert feed(seg(), [SPEECH] * 3 + [QUIET] * 34) == []

def test_long_speech_is_force_flushed_at_max():
    assert len(feed(seg(max_ms=3000), [SPEECH] * 101)) == 1

def test_reset_discards_partial_utterance():
    s = seg()
    feed(s, [SPEECH] * 20)
    s.reset()
    assert s.in_speech is False
    assert feed(s, [QUIET] * 34) == []

def test_level_is_zero_for_silence_and_positive_for_signal():
    assert level(QUIET) == 0.0
    assert 0.0 < level(b"\xff\x7f" * 480) <= 1.0

def test_missing_portaudio_gives_install_hint(monkeypatch):
    monkeypatch.setitem(sys.modules, "sounddevice", None)
    with pytest.raises(MicError, match="libportaudio2"):
        Mic().start()

def test_stop_discards_audio_still_in_the_queue(monkeypatch):
    held = {}

    class Stream:
        active = True

        def start(self):
            self.active = True

        def close(self):
            self.active = False

    def open_stream(**kwargs):
        held["callback"] = kwargs["callback"]
        return Stream()

    fake = NS(
        PortAudioError=type("PortAudioError", (Exception,), {}),
        RawInputStream=open_stream,
        _terminate=lambda: None,
        _initialize=lambda: None,
        query_devices=lambda device=None, kind=None: {"name": "Fake Mic", "max_input_channels": 1},
    )
    monkeypatch.setitem(sys.modules, "sounddevice", fake)
    mic = Mic()
    mic.start()
    frame = b"\x01" * FRAME_BYTES
    held["callback"](frame, 480, None, None)
    assert mic.read(timeout=0) == frame
    held["callback"](frame, 480, None, None)
    mic.stop()
    assert mic.read(timeout=0) is None


def test_alive_is_false_when_the_stream_check_fails(monkeypatch):
    class PortAudioError(Exception):
        pass

    class Stream:
        def __init__(self):
            self.fail = False

        def start(self):
            return None

        def close(self):
            return None

        @property
        def active(self):
            if self.fail:
                raise PortAudioError("stream closed")
            return True

    stream = Stream()

    def open_stream(**kwargs):
        del kwargs
        return stream

    fake = NS(
        PortAudioError=PortAudioError,
        RawInputStream=open_stream,
        _terminate=lambda: None,
        _initialize=lambda: None,
        query_devices=lambda device=None, kind=None: {"name": "Fake Mic", "max_input_channels": 1},
    )
    monkeypatch.setitem(sys.modules, "sounddevice", fake)
    mic = Mic()
    mic.start()
    assert mic.alive is True
    stream.fail = True
    assert mic.alive is False
    mic.stop()


def test_device_that_cannot_open_points_to_list_devices(monkeypatch):
    class PortAudioError(Exception):
        pass
    def refuse(**kw):
        raise PortAudioError("Invalid sample rate")
    fake = NS(PortAudioError=PortAudioError, RawInputStream=refuse, _terminate=lambda: None, _initialize=lambda: None)
    monkeypatch.setitem(sys.modules, "sounddevice", fake)
    with pytest.raises(MicError, match="--list-devices"):
        Mic("hw:0,0").start()
