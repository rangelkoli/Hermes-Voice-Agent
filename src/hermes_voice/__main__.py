"""Command line for the Hermes voice client."""

import argparse
import sys
from datetime import datetime
from multiprocessing import resource_tracker

import httpx
import webrtcvad

from hermes_voice.app import VoiceApp
from hermes_voice.audio import SAMPLE_RATE, Mic, MicError, Segmenter, list_devices
from hermes_voice.hermes import HermesAuthError, HermesClient, load_api_key
from hermes_voice.stt import Transcriber

_HEALTH_URL = "http://127.0.0.1:8642/health"
_GATEWAY = "Hermes is not running. Start it with: hermes gateway"


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


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="hermes-voice")
    parser.add_argument("--wake")
    parser.add_argument("--device", type=_device)
    parser.add_argument("--list-devices", action="store_true")
    parser.add_argument("--stt-model", default="small.en")
    parser.add_argument("--vad-level", type=int, choices=(0, 1, 2, 3), default=2)
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

    try:
        httpx.get(_HEALTH_URL, timeout=3, trust_env=False).raise_for_status()
    except httpx.HTTPError:
        print(_GATEWAY, file=sys.stderr)
        raise SystemExit(1) from None

    vad = webrtcvad.Vad(args.vad_level)
    is_speech = lambda f: vad.is_speech(f, SAMPLE_RATE)
    # Textual replaces sys.stderr with an object whose fileno() is -1. Loading Whisper makes
    # tqdm start this tracker, which passes that fd to its child and fails with "bad value(s)
    # in fds_to_keep". Starting it here, while stderr is real, avoids that.
    resource_tracker.ensure_running()
    VoiceApp(
        HermesClient(api_key),
        datetime.now().strftime("voice-%Y%m%d-%H%M%S"),
        wake=args.wake,
        mic=Mic(args.device),
        transcriber=Transcriber(args.stt_model),
        segmenter=Segmenter(is_speech, end_silence_ms=args.end_silence_ms),
    ).run()
