"""Command line for the Hermes voice client."""

import argparse
import sys
import time
from datetime import datetime
from multiprocessing import resource_tracker

import httpx

from hermes_voice.app import VoiceApp
from hermes_voice.audio import Mic, MicError, Segmenter, SpeechDetector, list_devices
from hermes_voice.gemini import GeminiError, GeminiSpeaker, GeminiTranscriber, load_gemini_key
from hermes_voice.hermes import HermesAuthError, HermesClient, load_api_key
from hermes_voice.stt import Transcriber
from hermes_voice.wake import WakeGate

_HEALTH_URL = "http://127.0.0.1:8642/health"
_HEALTH_ATTEMPTS = 3


def _device(value: str) -> int | str:
    try:
        return int(value)
    except ValueError:
        return value


def _end_silence_ms(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("--end-silence-ms must be greater than 0")
    return number


def _hotword(phrase: str) -> str | None:
    words = phrase.split()
    if not words:
        return None
    return words[-1].capitalize()


def _check_hermes() -> None:
    """Exit only if nothing is listening. A busy gateway is fine: the app retries sends."""
    for attempt in range(_HEALTH_ATTEMPTS):
        try:
            httpx.get(_HEALTH_URL, timeout=3, trust_env=False)
            return
        except httpx.ConnectError as exc:
            reason = exc
        except httpx.HTTPError:
            return
        if attempt + 1 < _HEALTH_ATTEMPTS:
            time.sleep(1)
    print(
        f"Nothing is answering at {_HEALTH_URL} ({reason}).\n"
        "Start Hermes with: hermes gateway\n"
        "If it is already running, check API_SERVER_ENABLED=true and API_SERVER_PORT=8642 "
        "in its .env.",
        file=sys.stderr,
    )
    raise SystemExit(1)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="hermes-voice")
    parser.add_argument("--wake", default="hey hermes")
    parser.add_argument("--open-mic", action="store_true")
    parser.add_argument("--wake-model", default="base.en")
    parser.add_argument("--device", type=_device)
    parser.add_argument("--list-devices", action="store_true")
    parser.add_argument("--local", action="store_true")
    parser.add_argument("--stt-model")
    parser.add_argument("--tts-model", default="gemini-3.8-flash-lite-tts")
    parser.add_argument("--voice", default="Kore")
    parser.add_argument("--vad-threshold", type=float, default=0.5)
    parser.add_argument("--end-silence-ms", type=_end_silence_ms, default=1000)
    args = parser.parse_args(argv)

    if args.list_devices:
        try:
            print(list_devices())
        except MicError as exc:
            print(exc, file=sys.stderr)
            raise SystemExit(1) from None
        return

    try:
        Mic(args.device).check()
    except MicError as exc:
        print(exc, file=sys.stderr)
        raise SystemExit(1) from None

    try:
        api_key = load_api_key()
    except HermesAuthError as exc:
        print(exc, file=sys.stderr)
        raise SystemExit(1) from None

    _check_hermes()

    if args.local:
        stt_model = args.stt_model or "small.en"
        transcriber = Transcriber(stt_model)
        speaker = None
    else:
        stt_model = args.stt_model or "gemini-3.5-transcribe"
        try:
            gemini_key = load_gemini_key()
        except GeminiError as exc:
            print(exc, file=sys.stderr)
            raise SystemExit(1) from None
        transcriber = GeminiTranscriber(gemini_key, model=stt_model)
        speaker = GeminiSpeaker(gemini_key, model=args.tts_model, voice=args.voice)
    if args.open_mic:
        gate = None
        wake = None
        wake_transcriber = None
    else:
        gate = WakeGate()
        wake = args.wake
        wake_transcriber = Transcriber(args.wake_model, hotword=_hotword(args.wake))
    # Textual replaces sys.stderr with an object whose fileno() is -1. Loading Whisper makes
    # tqdm start this tracker, which passes that fd to its child and fails with "bad value(s)
    # in fds_to_keep". Starting it here, while stderr is real, avoids that.
    resource_tracker.ensure_running()
    VoiceApp(
        HermesClient(api_key),
        datetime.now().strftime("voice-%Y%m%d-%H%M%S"),
        wake=wake,
        mic=Mic(args.device),
        transcriber=transcriber,
        wake_transcriber=wake_transcriber,
        segmenter=Segmenter(
            SpeechDetector(args.vad_threshold), end_silence_ms=args.end_silence_ms
        ),
        speaker=speaker,
        gate=gate,
    ).run()
