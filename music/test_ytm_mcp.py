"""Tests for the YouTube Music MCP server.

Run: ~/.hermes/hermes-agent/venv/bin/python3 -m unittest discover -s music -p 'test_*.py' -v
"""

import json
import os
import shutil
import socket
import subprocess
import tempfile
import threading
import time
import unittest
from unittest import mock

import ytm_mcp


class FakeMpv:
    """Answers mpv's JSON IPC on a Unix socket and patches ytm_mcp.SOCKET to it."""

    def __init__(self, properties=None, listening=True):
        self.properties = properties or {}
        self.commands = []
        self.listening = listening
        self.server = None

    def __enter__(self):
        # /tmp keeps the path under the 104 byte limit for Unix sockets.
        self.dir = tempfile.mkdtemp(dir="/tmp")
        self.path = os.path.join(self.dir, "mpv.sock")
        self.patch = mock.patch.object(ytm_mcp, "SOCKET", self.path)
        self.patch.start()
        if self.listening:
            self.listen()
        return self

    def __exit__(self, *exc):
        self.patch.stop()
        if self.server:
            self.server.close()
        shutil.rmtree(self.dir, ignore_errors=True)

    def listen(self):
        """A player starts: the socket appears, as when mpv launches."""
        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        server.bind(self.path)
        server.listen()
        self.server = server
        threading.Thread(target=self._accept, args=(server,), daemon=True).start()

    def _accept(self, server):
        while True:
            try:
                conn, _ = server.accept()
            except OSError:
                return
            threading.Thread(target=self._serve, args=(conn, server), daemon=True).start()

    def _serve(self, conn, server):
        with conn, conn.makefile("rw") as stream:
            for line in stream:
                command = json.loads(line)["command"]
                self.commands.append(command)
                stream.write('{"event":"playback-restart"}\n')
                if command[0] == "get_property" and command[1] not in self.properties:
                    reply = {"error": "property unavailable"}
                elif command[0] == "get_property":
                    reply = {"data": self.properties[command[1]], "error": "success"}
                else:
                    reply = {"error": "success"}
                stream.write(json.dumps(reply) + "\n")
                stream.flush()
                if command == ["quit"]:
                    # Like mpv, the player takes its socket with it.
                    server.close()
                    os.unlink(self.path)
                    return


class YtmTest(unittest.TestCase):
    def setUp(self):
        # No test may touch the real socket: play sends quit to whatever listens there.
        self.dir = tempfile.mkdtemp(dir="/tmp")
        self.addCleanup(shutil.rmtree, self.dir, ignore_errors=True)
        for name, value in (
            ("SOCKET", os.path.join(self.dir, "none.sock")),
            ("YT_DLP", "yt-dlp"),
            ("MPV", "mpv"),
            ("volume", 100),
        ):
            patch = mock.patch.object(ytm_mcp, name, value)
            patch.start()
            self.addCleanup(patch.stop)

    def fake_popen(self, poll, on_start=None):
        proc = mock.Mock()
        proc.poll.return_value = poll

        def start(*args, **kwargs):
            if on_start:
                on_start()
            return proc

        patch = mock.patch.object(ytm_mcp.subprocess, "Popen", side_effect=start)
        self.addCleanup(patch.stop)
        return proc, patch.start()

    def test_search_url_quotes_query(self):
        self.assertEqual(
            ytm_mcp.search_url("AC/DC back in black"),
            "https://music.youtube.com/search?q=AC%2FDC+back+in+black#songs",
        )
        self.assertEqual(
            ytm_mcp.search_url("what's up?"),
            "https://music.youtube.com/search?q=what%27s+up%3F#songs",
        )
        self.assertTrue(ytm_mcp.search_url("-ish").startswith("https://"))

    def test_mix_url(self):
        self.assertEqual(
            ytm_mcp.mix_url("wU26xVT_vBU"),
            "https://music.youtube.com/watch?v=wU26xVT_vBU&list=RDAMVMwU26xVT_vBU",
        )

    def test_find_song_parses_id_and_title(self):
        done = subprocess.CompletedProcess([], 0, "wU26xVT_vBU\tOne More Time (Radio Edit)\n", "")
        with mock.patch.object(ytm_mcp.subprocess, "run", return_value=done) as run:
            self.assertEqual(
                ytm_mcp.find_song("one more time"),
                ("wU26xVT_vBU", "One More Time (Radio Edit)"),
            )
        argv = run.call_args.args[0]
        self.assertIsInstance(argv, list)
        self.assertIn("--flat-playlist", argv)
        self.assertIn("--no-update", argv)
        self.assertEqual(argv[-1], ytm_mcp.search_url("one more time"))

    def test_find_song_none_on_failure(self):
        for outcome in (
            subprocess.CompletedProcess([], 1, "", "boom"),
            subprocess.CompletedProcess([], 0, "", ""),
            subprocess.TimeoutExpired("yt-dlp", 30),
        ):
            effect = {"side_effect": outcome} if isinstance(outcome, Exception) else {"return_value": outcome}
            with mock.patch.object(ytm_mcp.subprocess, "run", **effect):
                self.assertIsNone(ytm_mcp.find_song("x"))

    def test_mpv_args_expand_the_mix(self):
        args = ytm_mcp.mpv_args("URL", 40)
        for flag in (
            "--no-video",
            "--ytdl-raw-options=yes-playlist=",
            "--volume=40",
            "--input-ipc-server=" + ytm_mcp.SOCKET,
        ):
            self.assertIn(flag, args)
        self.assertEqual(args[-1], "URL")

    def test_send_skips_event_lines_and_returns_data(self):
        with FakeMpv({"volume": 55.0}):
            self.assertEqual(ytm_mcp.send(["get_property", "volume"]), 55.0)

    def test_controls_without_player_say_nothing_is_playing(self):
        started = time.monotonic()
        for tool in (ytm_mcp.pause, ytm_mcp.resume, ytm_mcp.next_track, ytm_mcp.stop):
            self.assertEqual(tool(), "Nothing is playing.")
        self.assertLess(time.monotonic() - started, 1)

    def test_stale_socket_file_says_nothing_is_playing(self):
        dead = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        dead.bind(ytm_mcp.SOCKET)
        dead.close()
        self.assertTrue(os.path.exists(ytm_mcp.SOCKET))
        self.assertEqual(ytm_mcp.pause(), "Nothing is playing.")

    def test_stop_reaches_player_this_process_did_not_start(self):
        with FakeMpv() as fake:
            self.assertEqual(ytm_mcp.stop(), "Stopped")
            self.assertIn(["quit"], fake.commands)

    def test_set_volume_clamps_and_is_remembered(self):
        with FakeMpv() as fake:
            self.assertEqual(ytm_mcp.set_volume(150), "Volume 100")
            self.assertEqual(ytm_mcp.set_volume(-5), "Volume 0")
            self.assertEqual(ytm_mcp.set_volume(40), "Volume 40")
            self.assertIn(["set_property", "volume", 40], fake.commands)
            self.assertEqual(ytm_mcp.volume, 40)

    def test_play_empty_query_asks_what_to_play(self):
        with mock.patch.object(ytm_mcp, "find_song") as find_song:
            self.assertEqual(ytm_mcp.play("  "), "Say what to play.")
        find_song.assert_not_called()

    def test_play_nothing_found(self):
        with mock.patch.object(ytm_mcp, "find_song", return_value=None):
            self.assertEqual(ytm_mcp.play("zzzz"), "Nothing found for zzzz.")

    def test_play_reports_failure_when_mpv_exits(self):
        self.fake_popen(poll=2)
        with mock.patch.object(ytm_mcp, "find_song", return_value=("id", "Title")):
            self.assertEqual(ytm_mcp.play("x"), "Could not start playback. yt-dlp may need updating.")

    def test_play_reports_failure_on_timeout(self):
        with FakeMpv(listening=False) as fake, mock.patch.object(ytm_mcp, "PLAY_WAIT", 0.5), mock.patch.object(
            ytm_mcp, "find_song", return_value=("id", "Title")
        ):
            proc, _ = self.fake_popen(poll=None, on_start=fake.listen)
            self.assertEqual(ytm_mcp.play("x"), "Could not start playback. yt-dlp may need updating.")
        proc.terminate.assert_called_once()

    def test_play_returns_title_once_audio_starts(self):
        # A player is already running. The new one binds the same path when it launches.
        with FakeMpv({"playback-time": 0.4}) as fake, mock.patch.object(
            ytm_mcp, "find_song", return_value=("id", "One More Time")
        ):
            _, popen = self.fake_popen(poll=None, on_start=fake.listen)
            self.assertEqual(ytm_mcp.play("one more time"), "Playing One More Time")
            self.assertIn(["quit"], fake.commands)
            self.assertEqual(popen.call_args.args[0], ytm_mcp.mpv_args(ytm_mcp.mix_url("id"), ytm_mcp.volume))
        self.assertTrue(popen.call_args.kwargs["start_new_session"])


if __name__ == "__main__":
    unittest.main()
