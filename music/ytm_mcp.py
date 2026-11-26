"""MCP server that plays YouTube Music audio through mpv.

Hermes starts it over stdio. Register it as `youtube_music` (see the README).
The mpv control socket is the only record of a running player, so a player started before
Hermes restarted can still be paused or stopped.
"""

import json
import os
import shutil
import socket
import subprocess
import time
from pathlib import Path
from urllib.parse import quote_plus

from mcp.server.mcpserver import MCPServer

SOCKET = os.environ.get("HERMES_VOICE_MPV_SOCKET") or str(Path.home() / ".config/hermes-voice/mpv.sock")
YT_DLP = os.environ.get("YT_DLP") or shutil.which("yt-dlp")
MPV = os.environ.get("MPV") or shutil.which("mpv")
# Seconds play waits for audio. YouTube holds a new stream back about 6 s, and a start takes
# about 12 s on a Mac. A Raspberry Pi is slower.
PLAY_WAIT = 45.0
# The full mix is about 480 songs and takes 45 s to list. 50 lists in about 2 s.
MIX_SONGS = 50
FAILED = "Could not start playback. yt-dlp may need updating."

volume = 100  # last volume the user asked for, reused by the next play

mcp = MCPServer("youtube_music")


def search_url(query: str) -> str:
    return f"https://music.youtube.com/search?q={quote_plus(query)}#songs"


def mix_url(video_id: str) -> str:
    return f"https://music.youtube.com/watch?v={video_id}&list=RDAMVM{video_id}"


def find_song(query: str) -> tuple[str, str] | None:
    """Top YouTube Music song for the query, as (video id, title)."""
    argv = [YT_DLP, "--no-update", "--no-warnings", "--flat-playlist", "--playlist-end", "1",
            "--print", "%(id)s\t%(title)s", search_url(query)]
    try:
        done = subprocess.run(argv, capture_output=True, text=True, timeout=30)
    except (subprocess.TimeoutExpired, OSError):
        return None
    video_id, tab, title = done.stdout.strip().partition("\t")
    if done.returncode != 0 or not tab or not video_id:
        return None
    return video_id, title


def mpv_args(url: str, volume: int) -> list[str]:
    return [
        MPV,
        "--no-video",
        "--no-terminal",
        "--input-ipc-server=" + SOCKET,
        "--ytdl-format=bestaudio",
        # Without yes-playlist mpv plays the one song and ignores the rest of the mix.
        f"--ytdl-raw-options=yes-playlist=,playlist-end={MIX_SONGS}",
        "--script-opts=ytdl_hook-ytdl_path=" + YT_DLP,
        f"--volume={volume}",
        url,
    ]


def send(command: list) -> object:
    """One request to the running player. RuntimeError when it fails or nothing is playing."""
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as conn:
        conn.settimeout(2)
        try:
            conn.connect(SOCKET)
        except (FileNotFoundError, ConnectionRefusedError):
            raise RuntimeError("Nothing is playing.") from None
        try:
            conn.sendall((json.dumps({"command": command}) + "\n").encode())
            with conn.makefile() as lines:
                for line in lines:
                    try:
                        reply = json.loads(line)
                    except ValueError:
                        continue
                    # Event lines carry no "error" field. Only the reply does.
                    if isinstance(reply, dict) and "error" in reply:
                        if reply["error"] != "success":
                            raise RuntimeError(reply["error"])
                        return reply.get("data")
        except OSError:
            pass
    raise RuntimeError("The player is not responding.")


def _control(command: list, done: str) -> str:
    try:
        send(command)
    except RuntimeError as e:
        return str(e)
    return done


def _quit_current() -> None:
    """Stop the running player and wait for its socket to go, so it cannot remove the next one's."""
    try:
        send(["quit"])
    except RuntimeError:
        pass
    else:
        deadline = time.monotonic() + 2
        while os.path.exists(SOCKET) and time.monotonic() < deadline:
            time.sleep(0.05)
    try:
        os.unlink(SOCKET)  # left behind by a crash
    except FileNotFoundError:
        pass


@mcp.tool()
def play(query: str) -> str:
    """Play a song from YouTube Music by name, then keep playing similar songs. For a request with no song named, pass a genre, mood, or artist."""
    query = query.strip()
    if not query:
        return "Say what to play."
    if not YT_DLP:
        return "yt-dlp is not installed."
    if not MPV:
        return "mpv is not installed."
    song = find_song(query)
    if song is None:
        return f"Nothing found for {query}."
    video_id, title = song
    _quit_current()
    os.makedirs(os.path.dirname(SOCKET), exist_ok=True)
    proc = subprocess.Popen(
        mpv_args(mix_url(video_id), volume),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    deadline = time.monotonic() + PLAY_WAIT
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            return FAILED
        try:
            if isinstance(send(["get_property", "playback-time"]), (int, float)):
                return f"Playing {title}"
        except RuntimeError:
            pass  # not yet
        time.sleep(0.25)
    proc.terminate()
    return FAILED


@mcp.tool()
def pause() -> str:
    """Pause the music."""
    return _control(["set_property", "pause", True], "Paused")


@mcp.tool()
def resume() -> str:
    """Resume paused music."""
    return _control(["set_property", "pause", False], "Resumed")


@mcp.tool(name="next")
def next_track() -> str:
    """Skip to the next song."""
    return _control(["playlist-next", "force"], "Skipped")


@mcp.tool()
def stop() -> str:
    """Stop the music."""
    return _control(["quit"], "Stopped")


@mcp.tool()
def set_volume(percent: int) -> str:
    """Set the music volume from 0 to 100."""
    global volume
    volume = max(0, min(100, int(percent)))
    _control(["set_property", "volume", volume], "")
    return f"Volume {volume}"


if __name__ == "__main__":
    mcp.run()
