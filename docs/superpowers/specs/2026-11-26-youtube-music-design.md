# YouTube music playback by voice

Approved 2026-11-26.

## Intent

Say "hey John, play One More Time on YouTube" or "hey John, play some music" and hear the audio from the speaker of the machine running the voice agent. Audio only: no video, no browser window.

## Goals

- Play a named song from YouTube Music.
- Play something reasonable when no song is named, and keep playing after the first track ends.
- Control playback by voice: pause, resume, next, stop, volume.
- Keep the assistant usable while music plays: the wake phrase and spoken replies stay audible.

## Non-goals

- The user's own YouTube Music library, liked songs, or any login.
- Queue editing ("play this after that").
- A now-playing panel on screen.
- Video.

## Anti-goals

- Music that drowns out the assistant's reply or stops the wake phrase from working.
- A feature that only works from the voice agent. Other Hermes clients should be able to use it too.

## Constraints

- Hermes runs on the same host as the voice agent (`http://127.0.0.1:8642`), so a process Hermes starts plays on that host's speaker. This holds on the Mac and on the Raspberry Pi.
- The Pi has 1 GB of memory, which rules out a browser-based player.
- Hermes registers MCP tools as `mcp_{server}_{tool}`. The Hermes venv has `mcp` 2.0.0, whose server class is `MCPServer` from `mcp.server.mcpserver`.
- `yt-dlp` and `mpv` are installed on the Mac. They are not yet confirmed on the Pi.
- `yt-dlp` must be current. Tested 2026-11-26: version 2026.07.04 finds songs but YouTube rejects its stream links with HTTP 403. Version 2026.11.23 streams correctly.

## Approaches

1. **MCP server wrapping `yt-dlp` and `mpv` (chosen).** Hermes calls typed tools directly. Works from any Hermes client. Chosen by the user in the Nov 21 session and confirmed here.
2. **Hermes skill that shells out through the terminal tool.** No new process, but Hermes composes shell commands on every request, which is slower and more error-prone.
3. **Playback built into the Rust voice binary.** Fastest "stop", but needs its own intent parsing and duplicates what Hermes already does. Voice-only.
4. **YouTube Music in a browser.** The only route to the user's own library, but too heavy for the Pi.

## Design

```
"hey John, play X" -> transcription -> Hermes -> mcp_youtube_music_play("X")
    -> yt-dlp: top YouTube Music song for X
    -> mpv --no-video: that song, then its auto-mix
    -> Hermes replies "Playing X"
```

### MCP server

`music/ytm_mcp.py` in this repo, a stdio server named `youtube_music`, registered under `mcp_servers` in `~/.hermes/config.yaml` and run with the Hermes venv's Python.

Tools:

| Tool | Behaviour |
|------|-----------|
| `play(query)` | Stops any current playback. Searches YouTube Music songs for `query`, takes the top result, and starts `mpv --no-video` on that song's auto-mix so playback continues after the first track. Returns the title. Returns "Nothing found" when the search is empty. |
| `pause` / `resume` | Sets mpv's `pause` property. |
| `next` | Skips to the next track in the mix. |
| `stop` | Ends mpv. |
| `set_volume(percent)` | Sets mpv's `volume` property, 0 to 100. |

Search uses `https://music.youtube.com/search?q=<query>#songs` with `--flat-playlist`, which returns songs rather than music videos. The mix URL is `https://music.youtube.com/watch?v=<id>&list=RDAMVM<id>`. Both were tested on the Mac: search took about 1.2 s. mpv only expands the mix into a playlist when started with `--ytdl-raw-options=yes-playlist=`. Without it, mpv plays the one song and exits.

Control goes over mpv's JSON IPC socket at `~/.config/hermes-voice/mpv.sock`, with `HERMES_VOICE_MPV_SOCKET` as the override. The socket is the only record of a running player, so a player started before Hermes restarted can still be controlled. Control calls return "Nothing is playing." at once when the socket does not answer.

`play` replies only after mpv reports that audio has started, waiting up to 15 s. If mpv exits or the wait runs out, it replies that playback could not start and that `yt-dlp` may need updating.

"Play some music" needs no special tool: Hermes chooses a seed query and calls `play`.

### Ducking in the voice agent

The voice agent lowers mpv's volume over the same IPC socket when the wake phrase fires and while a reply is being spoken, then restores it. If the socket is absent, this is a no-op. The duck level is a command-line flag, because the right level depends on the room and speaker.

The server and the voice agent read the same socket path and the same override.

### Prompt

One sentence is added to the per-run instructions the voice agent sends to Hermes: music requests go to the `mcp_youtube_music_*` tools, and the spoken reply is a short "Playing <title>".

## Risks

- `yt-dlp` breaks when YouTube changes and needs occasional updating. This is unofficial streaming, for personal use.
- Wake-phrase detection over music on the Pi depends on the PipeWire echo-cancel setup, which has not been measured: the microphone read silent in the last session.
- Voice "stop" goes through transcription and Hermes, so it takes a few seconds.

## Left to the builder

- Installing `mpv` and a current `yt-dlp` on the Pi.

The socket path, the default duck level (10), and how `play` confirms playback were settled while writing the plan and are recorded above and in the plan.
