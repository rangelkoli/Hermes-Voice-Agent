"""Local speech-to-text and filters for Whisper output."""

import re
import string
import unicodedata
from collections.abc import Iterable

# ponytail: known phrases only.
_NOISE = frozenset({
    "thank you",
    "thanks for watching",
    "you",
    "hermes",
})

# ponytail: how Whisper mishears Hermes, such as "almies" and "her mess". Only trusted right
# after a greeting, and only for this word. Add spellings here as --show-heard reveals them.
_SOUNDS_LIKE = {"hermes": r"h?[aeiou]{1,2}[lr]\s?m(?:e|ee|ea|ie|i|y)[sz]+"}
_GREETING = r"(?:hey|hay|hi|heh)"

_PUNCTUATION = str.maketrans("", "", string.punctuation)
_BRACKETS = {("[", "]"), ("(", ")")}


def is_noise(text: str) -> bool:
    stripped = text.strip()
    if len(stripped) >= 2 and (stripped[0], stripped[-1]) in _BRACKETS:
        return True
    normalized = " ".join(stripped.lower().translate(_PUNCTUATION).split())
    if not any(char.isalpha() for char in normalized):
        return True
    return normalized in _NOISE


def _fold(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(char for char in normalized if not unicodedata.combining(char))


def wake_match(text: str, phrase: str) -> str | None:
    """Return the text after the wake word, or None when it was not said.

    The key word is the phrase's last word. An empty string means the phrase
    was the whole utterance.
    """
    words = _fold(phrase).split()
    if not words:
        return None
    folded = _fold(text)
    word = re.escape(words[-1])
    # ponytail: the key word has to be one of the first four words, and a
    # sentence break is only . ! or ?. A Hermes later in the sentence is ignored.
    pattern = rf"(?:^|[.!?]\s+)(?:[\w'’]+[\s,]+){{0,3}}?{word}\b\W*"
    soundalike = _SOUNDS_LIKE.get(words[-1].lower())
    if soundalike:
        pattern += rf"|(?:^|[.!?]\s+){_GREETING}\W+{soundalike}\b\W*"
    matches = list(re.finditer(pattern, folded, flags=re.IGNORECASE))
    if not matches:
        return None
    remainder = folded[matches[-1].end() :]
    return remainder.lstrip(string.punctuation + string.whitespace)


def join_segments(segments: Iterable) -> str:
    kept = (
        segment.text
        for segment in segments
        if segment.no_speech_prob <= 0.6
    )
    return "".join(kept).strip()


class Transcriber:
    def __init__(self, model: str = "small.en", hotword: str | None = None) -> None:
        self.model = model
        self.hotword = hotword
        self._model = None

    def load(self) -> None:
        from faster_whisper import WhisperModel

        self._model = WhisperModel(self.model, device="cpu", compute_type="int8")

    def transcribe(self, pcm: bytes) -> str:
        import numpy as np

        audio = np.frombuffer(pcm, np.int16).astype(np.float32) / 32768
        kwargs = {
            "beam_size": 1,
            "condition_on_previous_text": False,
        }
        if self.hotword:
            kwargs["hotwords"] = self.hotword
            kwargs["vad_filter"] = False
        else:
            kwargs["initial_prompt"] = "Hermes"
        segments, _info = self._model.transcribe(audio, **kwargs)
        return join_segments(segments)
