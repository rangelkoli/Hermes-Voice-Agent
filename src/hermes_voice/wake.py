"""On-device wake phrase. Audio stays in memory; only the model archive touches disk."""

import hashlib
import os
import shutil
import tarfile
import tempfile
import time
import urllib.request
import uuid
from collections import deque
from collections.abc import Callable
from pathlib import Path

# ponytail: one fixed English model. A different phrase still uses this file.
MODEL_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/kws-models/"
    "sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01.tar.bz2"
)
MODEL_SHA256 = "f170013b4716e41b62b9bfd809687c207cef798ef9bc6534d524e17af9b6561a"
MODEL_DIR = "sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01"
_ENCODER = "encoder-epoch-12-avg-2-chunk-16-left-64.onnx"
_DECODER = "decoder-epoch-12-avg-2-chunk-16-left-64.onnx"
_JOINER = "joiner-epoch-12-avg-2-chunk-16-left-64.onnx"

# ponytail: capture frames are 30 ms. A different size would mis-count the ring.
_FRAME_MS = 30
_SAMPLE_RATE = 16000

_OPEN_MIC = "Run with --open-mic, which needs no model."
_DOWNLOAD_ERROR = (
    "Could not download the wake word model. Check the network and try again. " + _OPEN_MIC
)
_HASH_ERROR = (
    "The wake word model download failed its integrity check. Try again. " + _OPEN_MIC
)
_UNPACK_ERROR = (
    "The wake word model archive could not be unpacked. Try again. " + _OPEN_MIC
)


class WakeError(Exception):
    """Shown to the user as-is."""


def _cache_root() -> Path:
    base = os.environ.get("XDG_CACHE_HOME")
    if base:
        return Path(base) / "hermes-voice" / "wake"
    return Path.home() / ".cache" / "hermes-voice" / "wake"


def model_is_cached(root: Path | None = None) -> bool:
    directory = _cache_root() if root is None else Path(root)
    return (directory / MODEL_DIR / "tokens.txt").is_file()


def _download(url: str, dest: Path) -> None:
    urllib.request.urlretrieve(url, dest)


def _remove_partial(root: Path, before: set[str], archive: Path) -> None:
    archive.unlink(missing_ok=True)
    for child in list(root.iterdir()):
        if child.name in before and child.name != MODEL_DIR:
            continue
        if child.is_symlink() or not child.is_dir():
            child.unlink(missing_ok=True)
        else:
            shutil.rmtree(child, ignore_errors=True)


def ensure_model(
    root: Path | None = None,
    download: Callable[[str, Path], None] | None = None,
) -> Path:
    """Return the model directory, downloading it into the cache when it is missing."""
    directory = _cache_root() if root is None else Path(root)
    directory.mkdir(parents=True, exist_ok=True)
    model = directory / MODEL_DIR
    if (model / "tokens.txt").is_file():
        return model
    fetch = _download if download is None else download
    archive = directory / f".wake-{uuid.uuid4().hex}.tar.bz2"
    before = {path.name for path in directory.iterdir()}
    try:
        _install_model(directory, model, archive, fetch)
    except Exception:
        _remove_partial(directory, before, archive)
        raise
    archive.unlink(missing_ok=True)
    return model


def _install_model(
    directory: Path,
    model: Path,
    archive: Path,
    fetch: Callable[[str, Path], None],
) -> None:
    try:
        fetch(MODEL_URL, archive)
    except WakeError:
        raise
    except Exception as exc:
        raise WakeError(_DOWNLOAD_ERROR) from exc
    if not archive.is_file():
        raise WakeError(_DOWNLOAD_ERROR)
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    if digest != MODEL_SHA256:
        raise WakeError(_HASH_ERROR)
    try:
        with tarfile.open(archive, "r:bz2") as tar:
            tar.extractall(directory, filter="data")
    except Exception as exc:
        raise WakeError(_UNPACK_ERROR) from exc
    if not (model / "tokens.txt").is_file():
        raise WakeError(_UNPACK_ERROR)


class WakeSpotter:
    """Local keyword spotter. sherpa and sentencepiece load only when this is built."""

    def __init__(self, phrase: str, model_dir: Path, threshold: float = 0.25) -> None:
        import sentencepiece
        import sherpa_onnx

        model_dir = Path(model_dir)
        processor = sentencepiece.SentencePieceProcessor(
            model_file=str(model_dir / "bpe.model")
        )
        tokens = processor.encode(phrase.upper(), out_type=str)
        line = " ".join(tokens) + " @" + phrase.upper().replace(" ", "_") + "\n"
        # The spotter reads this file while it is constructed and keeps the tokens in memory.
        fd, keywords_path = tempfile.mkstemp(prefix="hermes-wake-keywords-", suffix=".txt")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(line)
            self._spotter = sherpa_onnx.KeywordSpotter(
                tokens=str(model_dir / "tokens.txt"),
                encoder=str(model_dir / _ENCODER),
                decoder=str(model_dir / _DECODER),
                joiner=str(model_dir / _JOINER),
                keywords_file=keywords_path,
                keywords_threshold=threshold,
                num_threads=1,
            )
            self._stream = self._spotter.create_stream()
        finally:
            os.unlink(keywords_path)

    def feed(self, frame: bytes) -> bool:
        import numpy as np

        samples = np.frombuffer(frame, dtype=np.int16).astype(np.float32) / 32768.0
        self._stream.accept_waveform(_SAMPLE_RATE, samples)
        hit = False
        while self._spotter.is_ready(self._stream):
            self._spotter.decode_stream(self._stream)
            if self._spotter.get_result(self._stream):
                hit = True
                self._spotter.reset_stream(self._stream)
        return hit

    def reset(self) -> None:
        self._spotter.reset_stream(self._stream)


class WakeGate:
    """Passes audio to the segmenter only after the spotter hears the phrase."""

    def __init__(
        self,
        spotter,
        ring_ms: int = 1500,
        window_s: float = 8,
        clock=time.monotonic,
    ) -> None:
        self._spotter = spotter
        self._window_s = window_s
        self._clock = clock
        self._ring: deque[bytes] = deque(maxlen=ring_ms // _FRAME_MS)
        self.awake = False
        self._deadline: float | None = None

    def feed(self, frame: bytes) -> list[bytes]:
        if self.awake:
            return [frame]
        self._ring.append(frame)
        if not self._spotter.feed(frame):
            return []
        self.awake = True
        self._deadline = self._clock() + self._window_s
        return list(self._ring)

    def touch(self) -> None:
        if self.awake:
            self._deadline = self._clock() + self._window_s

    def expire(self, busy: bool) -> bool:
        if not self.awake or busy or self._deadline is None:
            return False
        if self._clock() < self._deadline:
            return False
        self.sleep()
        return True

    def sleep(self) -> None:
        self.awake = False
        self._ring.clear()
        self._deadline = None
        self._spotter.reset()
