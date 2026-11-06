# Hermes Voice

Hermes Voice listens on the microphone, transcribes speech, and sends the text to a Hermes agent at `http://127.0.0.1:8642`. The conversation stays in the terminal. Replies are spoken aloud.

## Setup

Install [uv](https://docs.astral.sh/uv/). Install Hermes with its API server enabled (`API_SERVER_ENABLED=true`).

On Linux, install PortAudio first. Debian and Ubuntu need `libportaudio2` (`sudo apt install libportaudio2`). Fedora and Arch need `portaudio`. The macOS install bundles it.

Create a Gemini API key at [aistudio.google.com/apikey](https://aistudio.google.com/apikey). Set `GEMINI_API_KEY`, or put `GEMINI_API_KEY` in `~/.config/hermes-voice/.env`.

## Run

```bash
uv run hermes-voice
```

The Hermes key is read from `HERMES_API_KEY`. If that is unset, it is read from `API_SERVER_KEY` in `$HERMES_HOME/.env`, or `~/.hermes/.env` when `HERMES_HOME` is unset.

`m` mutes the microphone. `esc` cuts speech, and a second press stops the current request. `q` quits.

`--local` transcribes with Whisper on this computer and does not speak replies. It needs no Gemini key.

## Flags

- `--wake` sends speech only after this word.
- `--device` picks an input by index or name.
- `--list-devices` prints input devices and exits.
- `--local` transcribes with Whisper and does not speak replies.
- `--stt-model` chooses the transcription model. Default `gemini-3.5-transcribe`, or `small.en` with `--local`.
- `--tts-model` chooses the speech model. Default `gemini-3.8-flash-lite-tts`.
- `--voice` chooses the spoken voice. Default `Kore`.
- `--vad-level` is 0 to 3. Default `2`. Raise it when room noise shows as hearing.
- `--end-silence-ms` is the pause that ends a phrase. Default `1000`. Raise it when transcripts cut off mid-sentence.

## Privacy

> By default, finished utterances are sent to Google for transcription, and reply text is sent to Google to be spoken. In open-mic mode that includes anything said in the room. Silence is never sent. Transcripts also go to Hermes and its model provider. With `--local`, nothing goes to Google and audio stays on this computer.
