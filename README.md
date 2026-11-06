# Hermes Voice

Hermes Voice listens on the microphone, transcribes speech on this computer, and sends the text to a Hermes agent at `http://127.0.0.1:8642`. The conversation stays in the terminal.

## Setup

Install [uv](https://docs.astral.sh/uv/). Install Hermes with its API server enabled (`API_SERVER_ENABLED=true`).

On Linux, install PortAudio first. Debian and Ubuntu need `libportaudio2` (`sudo apt install libportaudio2`). Fedora and Arch need `portaudio`. The macOS install bundles it.

## Run

```bash
uv run hermes-voice
```

The key is read from `HERMES_API_KEY`. If that is unset, it is read from `API_SERVER_KEY` in `$HERMES_HOME/.env`, or `~/.hermes/.env` when `HERMES_HOME` is unset.

`m` mutes the microphone. `esc` stops the current request. `q` quits.

## Flags

- `--wake` sends speech only after this word.
- `--device` picks an input by index or name.
- `--list-devices` prints input devices and exits.
- `--stt-model` chooses the Whisper model. Default `small.en`. The first launch downloads it.
- `--vad-level` is 0 to 3. Default `2`. Raise it when room noise shows as hearing.
- `--end-silence-ms` is the pause that ends a phrase. Default `1000`. Raise it when transcripts cut off mid-sentence.

## Privacy

Audio stays on this computer. Every transcript goes to Hermes, which sends it to whichever model provider Hermes is configured to use. In open-mic mode that includes anything said in the room.
