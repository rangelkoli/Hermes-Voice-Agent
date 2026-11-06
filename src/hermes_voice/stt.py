"""Local speech-to-text and filters for Whisper output."""

import re
import string
from collections.abc import Iterable

# ponytail: known phrases only.
_NOISE = frozenset({
    "thank you",
    "thanks for watching",
    "you",
    "hermes",
})

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


def apply_wake(text: str, wake: str | None) -> str | None:
    if wake is None:
        return text
    match = re.match(
        rf"^\W*(?:hey\W+)?{re.escape(wake)}\b\W*",
        text,
        re.IGNORECASE,
    )
    if match is None:
        return None
    remainder = text[match.end() :].strip()
    return remainder or None


def join_segments(segments: Iterable) -> str:
    kept = (
        segment.text
        for segment in segments
        if segment.no_speech_prob <= 0.6
    )
    return "".join(kept).strip()


class Transcriber:
    def __init__(self, model: str = "small.en"):
        self.model = model
        self._model = None

    def load(self) -> None:
        from faster_whisper import WhisperModel

        self._model = WhisperModel(self.model, device="cpu", compute_type="int8")

    def transcribe(self, pcm: bytes) -> str:
        import numpy as np

        audio = np.frombuffer(pcm, np.int16).astype(np.float32) / 32768
        segments, _info = self._model.transcribe(
            audio,
            beam_size=1,
            condition_on_previous_text=False,
            initial_prompt="Hermes",
        )
        return join_segments(segments)
