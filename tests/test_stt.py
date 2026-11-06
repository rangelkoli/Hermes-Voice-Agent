from types import SimpleNamespace as NS
import pytest
from hermes_voice.stt import is_noise, join_segments, strip_wake

@pytest.mark.parametrize("text", ["", "  ", "Thank you.", "thanks for watching!", "you", "...", "[BLANK_AUDIO]", "(music)"])
def test_whisper_noise_is_dropped(text):
    assert is_noise(text)

@pytest.mark.parametrize("text", ["thank you for fixing the build", "stop the docker container"])
def test_real_requests_are_kept(text):
    assert not is_noise(text)

@pytest.mark.parametrize("text,expected", [
    ("Hey Hermes, open the logs", "open the logs"),
    ("hey, hermes open the logs", "open the logs"),
    ("HEY HERMES: open the logs", "open the logs"),
    ("and then hey hermes open the logs", "open the logs"),
    ("Hermes open the logs", "open the logs"),
    ("open the logs", "open the logs"),
    ("Hey Hermes.", ""),
    ("hey hermes stop hey hermes go", "go"),
])
def test_strip_wake(text, expected):
    assert strip_wake(text, "hey hermes") == expected

def test_strip_wake_with_a_custom_phrase():
    assert strip_wake("Computer, lights on", "computer") == "lights on"

def test_prompt_word_alone_is_noise():
    assert is_noise("Hermes.")

def test_join_skips_likely_non_speech_segments():
    segs = [NS(text=" open logs", no_speech_prob=0.1), NS(text=" Thank you", no_speech_prob=0.9)]
    assert join_segments(segs) == "open logs"
