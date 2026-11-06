"""Microphone capture and utterance segmentation."""

import array
import queue
import sys
import threading
from collections import deque
from collections.abc import Callable

SAMPLE_RATE = 16000
FRAME_BYTES = 960
# ponytail: every frame is 30 ms (480 int16 samples). Thresholds that are not
# a multiple of 30 ms take effect on the frame that reaches them.
_FRAME_MS = 30

if sys.platform == "darwin":
    MIC_SILENT_HINT = (
        "No audio is arriving. Allow this terminal app in "
        "System Settings > Privacy & Security > Microphone."
    )
else:
    MIC_SILENT_HINT = (
        "No audio is arriving. Check the input is not muted in sound settings, "
        "or choose another with --device."
    )

_alsa_error_handler = None


class MicError(Exception):
    pass


def level(frame: bytes) -> float:
    """RMS of int16 samples, scaled so full scale is 1.0."""
    count = len(frame) // 2
    if count == 0:
        return 0.0
    samples = array.array("h")
    samples.frombytes(frame[: count * 2])
    total = sum(sample * sample for sample in samples)
    rms = (total / count) ** 0.5
    return min(1.0, rms / 32768.0)


class Segmenter:
    def __init__(
        self,
        is_speech: Callable[[bytes], bool],
        end_silence_ms: int = 1000,
        min_speech_ms: int = 300,
        max_ms: int = 30000,
        preroll_ms: int = 300,
    ):
        self._is_speech = is_speech
        self._end_silence_ms = end_silence_ms
        self._min_speech_ms = min_speech_ms
        self._max_ms = max_ms
        self._preroll: deque[bytes] = deque(maxlen=preroll_ms // _FRAME_MS)
        self._frames: list[bytes] = []
        self._speech_frames = 0
        self._silence_frames = 0
        self.in_speech = False

    def reset(self) -> None:
        self._preroll.clear()
        self._frames.clear()
        self._speech_frames = 0
        self._silence_frames = 0
        self.in_speech = False

    def push(self, frame: bytes) -> bytes | None:
        if not self.in_speech:
            if not self._is_speech(frame):
                self._preroll.append(frame)
                return None
            self.in_speech = True
            self._frames = list(self._preroll)
            self._frames.append(frame)
            self._preroll.clear()
            self._speech_frames = 1
            self._silence_frames = 0
        else:
            self._frames.append(frame)
            if self._is_speech(frame):
                self._speech_frames += 1
                self._silence_frames = 0
            else:
                self._silence_frames += 1

        if self._silence_frames * _FRAME_MS >= self._end_silence_ms:
            keep = self._speech_frames * _FRAME_MS >= self._min_speech_ms
            return self._finish(keep)
        if len(self._frames) * _FRAME_MS >= self._max_ms:
            return self._finish(True)
        return None

    def _finish(self, keep: bool) -> bytes | None:
        audio = b"".join(self._frames) if keep else None
        self._frames = []
        self._speech_frames = 0
        self._silence_frames = 0
        self.in_speech = False
        return audio


def silence_alsa() -> None:
    """Drop ALSA stderr chatter. No-op except on Linux with libasound."""
    global _alsa_error_handler
    if sys.platform != "linux" or _alsa_error_handler is not None:
        return
    import ctypes

    try:
        lib = ctypes.CDLL("libasound.so.2")
        # ponytail: ALSA's handler is variadic; this no-op covers the five
        # fixed arguments and does not touch JACK.
        handler = ctypes.CFUNCTYPE(
            None,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
        )(lambda *_args: None)
        _alsa_error_handler = handler
        lib.snd_lib_error_set_handler(handler)
    except OSError:
        _alsa_error_handler = None


def list_devices() -> str:
    sd = _sounddevice()
    try:
        devices = sd.query_devices()
    except (sd.PortAudioError, ValueError) as exc:
        raise _device_error(exc) from exc
    lines = [
        f"{index}: {info['name']}"
        for index, info in enumerate(devices)
        if info["max_input_channels"] > 0
    ]
    if not lines:
        raise MicError(
            "A local microphone is required. SSH sessions and headless "
            "machines have none."
        )
    return "\n".join(lines)


class Mic:
    def __init__(self, device: str | int | None = None):
        silence_alsa()
        self._device = device
        self._queue: queue.Queue[bytes] = queue.Queue(maxsize=1000)
        self._stream = None
        self._name: str | None = None
        self._lock = threading.Lock()

    def check(self) -> None:
        sd = _sounddevice()
        try:
            sd.query_devices(self._device, kind="input")
        except (sd.PortAudioError, ValueError) as exc:
            raise _device_error(exc) from exc

    def start(self) -> None:
        sd = _sounddevice()
        self.stop()
        stream = None
        try:
            sd._terminate()
            sd._initialize()
            stream = sd.RawInputStream(
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="int16",
                blocksize=FRAME_BYTES // 2,
                device=self._device,
                callback=self._on_audio,
            )
            stream.start()
            with self._lock:
                self._stream = stream
                self._name = sd.query_devices(self._device, kind="input")["name"]
        except (sd.PortAudioError, ValueError) as exc:
            with self._lock:
                published = self._stream is stream
            if stream is not None and not published:
                stream.close()
            self.stop()
            raise MicError(
                f"Could not open the microphone: {exc}. Run with --list-devices."
            ) from exc

    def stop(self) -> None:
        with self._lock:
            stream = self._stream
            self._stream = None
        if stream is not None:
            stream.close()
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                return

    def read(self, timeout: float = 0.1) -> bytes | None:
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    @property
    def alive(self) -> bool:
        with self._lock:
            stream = self._stream
        return stream is not None and bool(stream.active)

    @property
    def name(self) -> str:
        if self._name is None:
            sd = _sounddevice()
            self._name = sd.query_devices(self._device, kind="input")["name"]
        return self._name

    def _on_audio(self, indata, _frames, _time, _status) -> None:
        try:
            self._queue.put_nowait(bytes(indata))
        except queue.Full:
            return


def _sounddevice():
    try:
        import sounddevice as sd
    except (OSError, ImportError) as exc:
        raise MicError(
            "PortAudio is missing. On Debian/Ubuntu install libportaudio2 "
            "(sudo apt install libportaudio2). On Fedora/Arch install portaudio."
        ) from exc
    return sd


def _device_error(exc: Exception) -> MicError:
    return MicError(
        "A local microphone is required. SSH sessions and headless machines "
        f"have none. ({exc}) Run with --list-devices."
    )
