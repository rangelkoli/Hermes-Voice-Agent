from types import SimpleNamespace as NS
import pytest
from hermes_voice.stt import is_noise, join_segments, wake_match

@pytest.mark.parametrize("text", ["", "  ", "Thank you.", "thanks for watching!", "you", "...", "[BLANK_AUDIO]", "(music)"])
def test_whisper_noise_is_dropped(text):
    assert is_noise(text)

@pytest.mark.parametrize("text", ["thank you for fixing the build", "stop the docker container"])
def test_real_requests_are_kept(text):
    assert not is_noise(text)

@pytest.mark.parametrize("text,expected", [
    ("Hey Hermes, open the logs", "open the logs"),
    ("hey hermes", ""),
    ("Hey Hermes.", ""),
    ("hey, hermes open the logs", "open the logs"),
    ("Hermes open the logs", "open the logs"),
    ("Pay Hermes, what time is it?", "what time is it?"),
    ("Have a Hermes", ""),
    ("and the new Hermes, what time is it?", "what time is it?"),
    ("Hey Hermès", ""),
    ("Thanks. Hey Hermes, go", "go"),
    ("hey hermes stop. hey hermes go", "go"),
    ("Hey, almies!", ""),
    ("Hey almies, what time is it?", "what time is it?"),
    ("Hey her mess", ""),
    ("hey hermies open the logs", "open the logs"),
    ("Hi armies.", ""),
    ("Hey her message came through", None),
    ("Hey James, are you coming?", None),
    ("Hey Amy's here", None),
    ("Almies are great", None),
    ("Hey Herman, can you open the door?", None),
    ("We should talk about the budget", None),
    ("We bought a very expensive Hermes scarf", None),
    ("open the hermescope", None),
])
def test_wake_match(text, expected):
    assert wake_match(text, "hey hermes") == expected

def test_wake_match_with_a_custom_phrase():
    assert wake_match("Computer, lights on", "computer") == "lights on"
    assert wake_match("Hey Jarvis lights on", "hey jarvis") == "lights on"

def test_prompt_word_alone_is_noise():
    assert is_noise("Hermes.")

def test_join_skips_likely_non_speech_segments():
    segs = [NS(text=" open logs", no_speech_prob=0.1), NS(text=" Thank you", no_speech_prob=0.9)]
    assert join_segments(segs) == "open logs"


def test_soundalikes_are_only_for_hermes():
    assert wake_match("Hey almies", "hey jarvis") is None


def test_soundalikes_work_with_a_capitalized_phrase():
    assert wake_match("Hey almies", "Hey Hermes") == ""

