from types import SimpleNamespace as NS
import pytest
from hermes_voice.stt import apply_wake, is_noise, join_segments

@pytest.mark.parametrize("text", ["", "  ", "Thank you.", "thanks for watching!", "you", "...", "[BLANK_AUDIO]", "(music)"])
def test_whisper_noise_is_dropped(text):
    assert is_noise(text)

@pytest.mark.parametrize("text", ["thank you for fixing the build", "stop the docker container"])
def test_real_requests_are_kept(text):
    assert not is_noise(text)

def test_no_wake_phrase_passes_everything():
    assert apply_wake("open the logs", None) == "open the logs"

@pytest.mark.parametrize("text", ["Hermes, open the logs", "hey hermes open the logs", "HERMES: open the logs"])
def test_wake_phrase_is_stripped(text):
    assert apply_wake(text, "hermes") == "open the logs"

def test_speech_without_wake_phrase_is_ignored():
    assert apply_wake("what's for dinner", "hermes") is None

def test_wake_phrase_alone_is_ignored():
    assert apply_wake("Hey Hermes.", "hermes") is None

def test_hey_with_punctuation_is_a_wake_phrase():
    assert apply_wake("hey, hermes open the logs", "hermes") == "open the logs"

def test_prompt_word_alone_is_noise():
    assert is_noise("Hermes.")

def test_join_skips_likely_non_speech_segments():
    segs = [NS(text=" open logs", no_speech_prob=0.1), NS(text=" Thank you", no_speech_prob=0.9)]
    assert join_segments(segs) == "open logs"
