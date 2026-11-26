# YouTube Music Playback Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Saying "hey John, play <song>" or "play some music" plays YouTube Music audio through the speaker, with voice control and automatic ducking while the assistant listens or speaks.

**Architecture:** A Python stdio MCP server (`music/ytm_mcp.py`) gives Hermes six tools. `play` finds a song with `yt-dlp` and starts `mpv --no-video` on that song's auto-mix. Everything else talks to mpv over its JSON IPC socket. The Rust voice agent talks to the same socket to lower the volume while it is listening, working, or speaking, and restore it afterwards.

**Tech Stack:** Python 3.11 with `mcp` 2.0.0 (`mcp.server.mcpserver.MCPServer`), stdlib `unittest`; `yt-dlp` 2026.11.23 or newer; `mpv`; Rust (std `UnixStream`, `serde_json`, no new crates).

**Spec:** `docs/superpowers/specs/2026-11-26-youtube-music-design.md`

## Global Constraints

- Work on branch `feat/youtube-music`, created from `main`. The first commit adds the spec and this plan.
- No new Rust crates and no new Python packages. The server runs under `~/.hermes/hermes-agent/venv/bin/python3`.
- `yt-dlp` must be 2026.11.23 or newer. Version 2026.07.04 gets HTTP 403 on every stream.
- Socket path: `HERMES_VOICE_MPV_SOCKET` when set and non-empty, otherwise `$HOME/.config/hermes-voice/mpv.sock`. Python and Rust must resolve it identically.
- MCP server name is `youtube_music`, so Hermes exposes the tools as `mcp_youtube_music_play`, `_pause`, `_resume`, `_next`, `_stop`, `_set_volume`.
- Song search URL: `https://music.youtube.com/search?q=<quote_plus(query)>#songs`. Mix URL: `https://music.youtube.com/watch?v=<id>&list=RDAMVM<id>`.
- mpv must be started with `--ytdl-raw-options=yes-playlist=`. Without it the mix does not expand and playback ends after one song.
- External commands are run as argument lists, never through a shell.
- Default duck level is 10 (mpv volume percent).
- Rust checks: `cargo test --release` and `cargo clippy --release -- -D warnings` both pass.
- README and code comments are plain prose with no em dashes, matching the existing README.

## Review Focus

1. **Stale `yt-dlp`: the song is found but the stream is refused.** The user must hear "could not start playback", not "Playing X" followed by silence. Pinned in Task 1 (`test_play_reports_failure_when_mpv_exits`, `test_play_reports_failure_on_timeout`).
2. **Hermes restarts while music plays.** "Stop the music" must still stop it. Pinned in Task 1 (`test_stop_reaches_player_this_process_did_not_start`).
3. **Volume changed by voice while ducked, or set below the duck level.** Restoring must give the user's volume, and ducking must never make music louder. Pinned in Task 2 (`volume_set_while_ducked_is_what_gets_restored`, `volume_below_duck_level_is_left_alone`).
4. **mpv stops answering.** The voice agent must keep running. Pinned in Task 2 (`silent_socket_does_not_stall`).
5. **Song names with punctuation ("AC/DC", "what's up?", "-ish").** They must reach the search intact and never be read as a command-line option. Pinned in Task 1 (`test_search_url_quotes_query`).

Not testable in code, checked by hand in Task 4: lyrics leaking into the microphone during the 2 second follow-up window and being sent as a command.

---

### Task 1: MCP server

**Files:**
- Create: `music/ytm_mcp.py`
- Create: `music/test_ytm_mcp.py`

**Interfaces:**
- Consumes: nothing.
- Produces: the stdio server `youtube_music` with tools `play(query: str)`, `pause()`, `resume()`, `next()`, `stop()`, `set_volume(percent: int)`, each returning a short sentence for Hermes to speak. An mpv IPC socket at the Global Constraints path while a player is running.

Module-level names the tests rely on:

```python
SOCKET: str      # resolved once at import, per Global Constraints
YT_DLP: str | None   # os.environ.get("YT_DLP") or shutil.which("yt-dlp")
MPV: str | None      # os.environ.get("MPV") or shutil.which("mpv")
PLAY_WAIT = 15.0     # seconds play waits for audio to start
volume = 100         # last volume the user asked for, reused by the next play

def search_url(query: str) -> str
def mix_url(video_id: str) -> str
def find_song(query: str) -> tuple[str, str] | None      # (video id, title)
def mpv_args(url: str, volume: int) -> list[str]
def send(command: list) -> object                        # one mpv IPC request
```

`send`, `mpv_args`, and the tools read `SOCKET`, `YT_DLP`, `MPV` as module globals at call time so tests can patch them.

- [ ] **Step 1: Create the branch and commit the docs**

```bash
git checkout -b feat/youtube-music
git add docs/superpowers/specs/2026-11-26-youtube-music-design.md docs/superpowers/plans/2026-11-26-youtube-music.md
git commit -m "docs: design and plan for YouTube music playback"
```

- [ ] **Step 2: Write the failing tests in `music/test_ytm_mcp.py`**

Use `unittest` and `unittest.mock`. Add one helper, `FakeMpv`: a context manager that binds a Unix socket in `tempfile.mkdtemp(dir="/tmp")` (short path, under the 104 byte limit), patches `ytm_mcp.SOCKET` to it, and serves each connection on a thread. For every request line it first writes `{"event":"playback-restart"}\n`, then a reply. It records each received `command` list in `self.commands` and answers `get_property` from a dict `self.properties`, with `{"error":"property unavailable"}` for a missing key and `{"data":<value>,"error":"success"}` otherwise.

The test class's `setUp` patches `ytm_mcp.SOCKET` to a path that does not exist inside a fresh temp directory, and `ytm_mcp.YT_DLP` and `ytm_mcp.MPV` to `"yt-dlp"` and `"mpv"`. No test may touch the real socket, because `play` sends `quit` to whatever is listening there.

Tests and their assertions:

```python
def test_search_url_quotes_query(self):
    self.assertEqual(ytm_mcp.search_url("AC/DC back in black"),
                     "https://music.youtube.com/search?q=AC%2FDC+back+in+black#songs")
    self.assertEqual(ytm_mcp.search_url("what's up?"),
                     "https://music.youtube.com/search?q=what%27s+up%3F#songs")
    self.assertTrue(ytm_mcp.search_url("-ish").startswith("https://"))

def test_mix_url(self):
    self.assertEqual(ytm_mcp.mix_url("wU26xVT_vBU"),
                     "https://music.youtube.com/watch?v=wU26xVT_vBU&list=RDAMVMwU26xVT_vBU")

def test_find_song_parses_id_and_title(self):
    # subprocess.run patched: returncode 0, stdout "wU26xVT_vBU\tOne More Time (Radio Edit)\n"
    self.assertEqual(ytm_mcp.find_song("one more time"),
                     ("wU26xVT_vBU", "One More Time (Radio Edit)"))
    # the argv passed to subprocess.run is a list, contains "--flat-playlist" and "--no-update",
    # and ends with search_url("one more time")

def test_find_song_none_on_failure(self):
    # returncode 1 -> None; returncode 0 with empty stdout -> None;
    # subprocess.TimeoutExpired raised -> None

def test_mpv_args_expand_the_mix(self):
    args = ytm_mcp.mpv_args("URL", 40)
    for flag in ("--no-video", "--ytdl-raw-options=yes-playlist=", "--volume=40",
                 "--input-ipc-server=" + ytm_mcp.SOCKET):
        self.assertIn(flag, args)
    self.assertEqual(args[-1], "URL")

def test_send_skips_event_lines_and_returns_data(self):
    # FakeMpv with properties {"volume": 55.0}
    self.assertEqual(ytm_mcp.send(["get_property", "volume"]), 55.0)

def test_controls_without_player_say_nothing_is_playing(self):
    # SOCKET patched to a path that does not exist; the four calls together finish in under 1 s
    for tool in (ytm_mcp.pause, ytm_mcp.resume, ytm_mcp.next_track, ytm_mcp.stop):
        self.assertEqual(tool(), "Nothing is playing.")

def test_stale_socket_file_says_nothing_is_playing(self):
    # bind a Unix socket at the path, close it, leave the file behind
    self.assertEqual(ytm_mcp.pause(), "Nothing is playing.")

def test_stop_reaches_player_this_process_did_not_start(self):
    # FakeMpv only, no Popen anywhere in this process
    self.assertEqual(ytm_mcp.stop(), "Stopped")
    self.assertIn(["quit"], fake.commands)

def test_set_volume_clamps_and_is_remembered(self):
    # FakeMpv running
    self.assertEqual(ytm_mcp.set_volume(150), "Volume 100")
    self.assertEqual(ytm_mcp.set_volume(-5), "Volume 0")
    self.assertEqual(ytm_mcp.set_volume(40), "Volume 40")
    self.assertIn(["set_property", "volume", 40], fake.commands)
    self.assertEqual(ytm_mcp.volume, 40)

def test_play_empty_query_asks_what_to_play(self):
    # find_song patched; play("  ") returns "Say what to play." and find_song is not called

def test_play_nothing_found(self):
    # find_song patched to return None
    self.assertEqual(ytm_mcp.play("zzzz"), "Nothing found for zzzz.")

def test_play_reports_failure_when_mpv_exits(self):
    # find_song -> ("id", "Title"); subprocess.Popen patched, its poll() returns 2
    self.assertEqual(ytm_mcp.play("x"), "Could not start playback. yt-dlp may need updating.")

def test_play_reports_failure_on_timeout(self):
    # PLAY_WAIT patched to 0.5; Popen.poll() returns None; FakeMpv has no "playback-time"
    self.assertEqual(ytm_mcp.play("x"), "Could not start playback. yt-dlp may need updating.")
    # the fake Popen's terminate() was called

def test_play_returns_title_once_audio_starts(self):
    # find_song -> ("id", "One More Time"); Popen.poll() returns None;
    # FakeMpv properties {"playback-time": 0.4}
    self.assertEqual(ytm_mcp.play("one more time"), "Playing One More Time")
    # Popen was called with mpv_args(mix_url("id"), ytm_mcp.volume) and start_new_session=True
```

- [ ] **Step 3: Run the tests to see them fail**

Run: `~/.hermes/hermes-agent/venv/bin/python3 -m unittest discover -s music -p 'test_*.py' -v`
Expected: `ModuleNotFoundError: No module named 'ytm_mcp'`

- [ ] **Step 4: Implement `music/ytm_mcp.py`**

Decisions the tests do not fully determine:

- `mcp = MCPServer("youtube_music")`. Tools are plain synchronous functions under `@mcp.tool()`. The skip tool is the Python function `next_track` registered as `@mcp.tool(name="next")`. End the file with `if __name__ == "__main__": mcp.run()`.
- Every tool docstring is one sentence, because Hermes shows it to the model. `play`'s docstring: "Play a song from YouTube Music by name, then keep playing similar songs. For a request with no song named, pass a genre, mood, or artist."
- `find_song` argv: `[YT_DLP, "--no-update", "--no-warnings", "--flat-playlist", "--playlist-end", "1", "--print", "%(id)s\t%(title)s", search_url(query)]`, `capture_output=True, text=True, timeout=30`.
- `mpv_args` returns exactly: `[MPV, "--no-video", "--no-terminal", "--input-ipc-server=" + SOCKET, "--ytdl-format=bestaudio", "--ytdl-raw-options=yes-playlist=", "--script-opts=ytdl_hook-ytdl_path=" + YT_DLP, f"--volume={volume}", url]`.
- `send`: connect an `AF_UNIX` stream socket with a 2 second timeout, write `json.dumps({"command": command}) + "\n"`, read lines until one parses to an object with an `"error"` key, skipping all others. Return its `"data"` when `"error" == "success"`, otherwise raise `RuntimeError(<error text>)`. `FileNotFoundError` and `ConnectionRefusedError` on connect become `RuntimeError("Nothing is playing.")`.
- `pause`, `resume`, `next_track`, `stop` send `["set_property", "pause", True]`, `["set_property", "pause", False]`, `["playlist-next", "force"]`, `["quit"]` and return `"Paused"`, `"Resumed"`, `"Skipped"`, `"Stopped"`. On `RuntimeError` they return its message.
- `set_volume` clamps to 0..100, stores the module global `volume`, tries `["set_property", "volume", n]`, and returns `f"Volume {n}"` whether or not a player is running.
- `play`, in order: strip the query, empty gives `"Say what to play."`; `YT_DLP` or `MPV` missing gives `"yt-dlp is not installed."` or `"mpv is not installed."`; `find_song` returning `None` gives `f"Nothing found for {query}."`; send `["quit"]` to any current player and ignore `RuntimeError`; remove a leftover socket file; create the socket's parent directory; `subprocess.Popen(mpv_args(mix_url(id), volume), stdin=DEVNULL, stdout=DEVNULL, stderr=DEVNULL, start_new_session=True)`; then every 0.25 s until `PLAY_WAIT`: if `proc.poll() is not None` fail, if `send(["get_property", "playback-time"])` returns a number succeed with `f"Playing {title}"`, and treat a `RuntimeError` from it as "not yet". On timeout call `proc.terminate()`. Both failures return `"Could not start playback. yt-dlp may need updating."`.

- [ ] **Step 5: Run the tests to see them pass**

Run: `~/.hermes/hermes-agent/venv/bin/python3 -m unittest discover -s music -p 'test_*.py' -v`
Expected: `OK`, 15 tests.

- [ ] **Step 6: Confirm the server lists its tools**

Run: `cd music && ~/.hermes/hermes-agent/venv/bin/python3 -c "import asyncio, ytm_mcp; print(sorted(t.name for t in asyncio.run(ytm_mcp.mcp.list_tools())))"`
Expected: `['next', 'pause', 'play', 'resume', 'set_volume', 'stop']`

- [ ] **Step 7: Commit**

```bash
git add music/ytm_mcp.py music/test_ytm_mcp.py
git commit -m "feat: add YouTube Music MCP server"
```

---

### Task 2: Ducker

**Files:**
- Create: `src/music.rs`
- Modify: `src/lib.rs` (add `pub mod music;` in alphabetical position)

**Interfaces:**
- Consumes: the mpv IPC socket from Task 1. Request `{"command":["get_property","volume"]}\n`, reply line `{"data":<number>,"error":"success"}`. Request `{"command":["set_property","volume",<number>]}\n`. mpv may write `{"event":...}` lines before the reply.
- Produces:

```rust
/// Sentence added to every Hermes run so music requests reach the MCP tools.
pub const INSTRUCTIONS: &str;
/// HERMES_VOICE_MPV_SOCKET when set and non-empty, else $HOME/.config/hermes-voice/mpv.sock.
pub fn socket_path() -> PathBuf;
pub struct Ducker { /* path, level, saved: Option<f64> */ }
impl Ducker {
    pub fn new(path: PathBuf, level: u8) -> Self;
    /// Call a few times a second. Never blocks longer than one socket timeout.
    pub fn tick(&mut self, engaged: bool);
}
```

`INSTRUCTIONS` is exactly:

```
When the user asks to play, pause, resume, skip, or stop music, or to change its volume, call the `mcp_youtube_music_*` tools. For a request that names no song, choose a fitting search query yourself. After `mcp_youtube_music_play` succeeds, reply with only `Playing <title>`.
```

- [ ] **Step 1: Write the failing tests in `src/music.rs`**

Add a test helper `FakeMpv`: binds a `UnixListener` at `/tmp/hv-mpv-<pid>-<name>.sock`, serves each connection on a thread, holds `volume: Arc<Mutex<f64>>` and `sets: Arc<Mutex<Vec<f64>>>`. For each request line it writes `{"event":"playback-restart"}\n` first. `get_property volume` replies with the current volume. `set_property volume X` stores X as the volume, pushes X to `sets`, replies `{"error":"success"}`. It removes the socket file on drop.

```rust
#[test]
fn duck_saves_volume_and_restores_it() {
    let mpv = FakeMpv::new("restore", 80.0);
    let mut d = Ducker::new(mpv.path(), 10);
    d.tick(true);
    d.tick(true); // already ducked: no second write
    assert_eq!(mpv.sets(), [10.0]);
    d.tick(false);
    d.tick(false); // already restored: no second write
    assert_eq!(mpv.sets(), [10.0, 80.0]);
}

#[test]
fn volume_set_while_ducked_is_what_gets_restored() {
    let mpv = FakeMpv::new("changed", 80.0);
    let mut d = Ducker::new(mpv.path(), 10);
    d.tick(true);
    mpv.set_volume(30.0); // "set the volume to 30" arrives through the MCP tool
    d.tick(true);
    d.tick(false);
    assert_eq!(mpv.sets(), [10.0, 10.0, 30.0]);
}

#[test]
fn volume_below_duck_level_is_left_alone() {
    let mpv = FakeMpv::new("quiet", 5.0);
    let mut d = Ducker::new(mpv.path(), 10);
    d.tick(true);
    d.tick(false);
    assert!(mpv.sets().is_empty());
}

#[test]
fn missing_socket_is_a_noop() {
    let mut d = Ducker::new(PathBuf::from("/tmp/hv-mpv-does-not-exist.sock"), 10);
    d.tick(true);
    d.tick(false);
}

#[test]
fn silent_socket_does_not_stall() {
    // A listener that accepts and never replies.
    let started = Instant::now();
    d.tick(true);
    assert!(started.elapsed() < Duration::from_secs(1));
}

#[test]
fn socket_path_honours_the_override() {
    // One test for both cases, since it changes process environment.
    // HERMES_VOICE_MPV_SOCKET=/tmp/x.sock -> PathBuf::from("/tmp/x.sock")
    // unset -> ends_with(".config/hermes-voice/mpv.sock")
}

#[test]
fn instructions_name_the_tools() {
    assert!(INSTRUCTIONS.contains("mcp_youtube_music_"));
}
```

- [ ] **Step 2: Run the tests to see them fail**

Run: `cargo test --release music::`
Expected: compile error, `Ducker` not found.

- [ ] **Step 3: Implement `src/music.rs`**

- One private helper does a single request: connect `std::os::unix::net::UnixStream`, set read and write timeouts of 200 ms, write the JSON line, read lines until one has an `"error"` field, return its `"data"` when the error is `"success"`. Any failure returns `None`.
- `tick(true)`: read the volume. If it is more than `level + 0.5`, store it in `saved` and set the volume to `level`. Otherwise do nothing.
- `tick(false)`: if `saved` holds a value, take it and set the volume to it. A failed write is ignored and `saved` stays cleared.
- Add this comment on `tick`: `// ponytail: a volume the user sets at or below the duck level while ducked is not remembered. Track the user's volume in the MCP server if that matters.`

- [ ] **Step 4: Run the tests to see them pass**

Run: `cargo test --release music::`
Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add src/music.rs src/lib.rs
git commit -m "feat: duck mpv volume over its control socket"
```

---

### Task 3: Wire ducking and the prompt into the voice agent

**Files:**
- Modify: `src/app.rs` (`Shared` impl near line 138, `Config` at 271, `Core` at 284, `send_loop` call at 687, `run` at 1191)
- Modify: `src/main.rs` (`USAGE`, `Args`, `parse_args_from`, the `Config` literal near line 177)
- Modify: `README.md`

**Interfaces:**
- Consumes: `music::{Ducker, socket_path, INSTRUCTIONS}` from Task 2.
- Produces: the `--duck-volume N` flag and `Config::duck_volume: u8`.

- [ ] **Step 1: Write the failing tests**

In `src/app.rs` tests:

```rust
#[test]
fn engaged_while_awake_or_busy_and_not_while_asleep() {
    let mut s = Shared { gate: Some(WakeGate::new(Duration::from_secs(8))), ..Shared::default() };
    s.hearing = true; // room talk, or the music itself, while asleep
    assert!(!s.engaged());
    s.gate.as_mut().unwrap().wake();
    assert!(s.engaged());
    s.gate.as_mut().unwrap().sleep();
    s.speaking = true;
    assert!(s.engaged());
    s.speaking = false;
    s.run_id = Some("r".into());
    assert!(s.engaged());
}

#[test]
fn open_mic_is_engaged_only_while_busy() {
    let mut s = Shared::default(); // no gate
    assert!(!s.engaged());
    s.transcribing = true;
    assert!(s.engaged());
}
```

In `src/main.rs` tests:

```rust
#[test]
fn duck_volume_defaults_to_ten_and_rejects_out_of_range() {
    assert_eq!(parse_args_from(std::iter::empty::<&str>()).unwrap().duck_volume, 10);
    assert_eq!(parse_args_from(["--duck-volume", "0"]).unwrap().duck_volume, 0);
    assert_eq!(parse_args_from(["--duck-volume=25"]).unwrap().duck_volume, 25);
    for bad in ["101", "-1", "loud"] {
        let err = parse_args_from(["--duck-volume", bad]).err().unwrap();
        assert_eq!(err, "--duck-volume must be a whole number from 0 to 100");
    }
}
```

- [ ] **Step 2: Run the tests to see them fail**

Run: `cargo test --release -- engaged duck_volume`
Expected: compile errors, `engaged` and `duck_volume` not found.

- [ ] **Step 3: Implement**

- `Shared::engaged(&self) -> bool`: the gate is awake, or `conversation_busy()`. Doc comment: "Listening, working, or speaking, so music must stay quiet."
- `Args::duck_volume: u8`, default 10, parsed in `parse_args_from` with the error text from the test. Add `[--duck-volume N]` to `USAGE`. Pass it into `Config::duck_volume`.
- `Core` gains `ducker: Mutex<Ducker>`, built in `run` as `Ducker::new(music::socket_path(), config.duck_volume)`.
- New `fn duck_loop(core: &Core)` spawned as `hermes-duck`: while `core.pause(Duration::from_millis(250), |_| false)` is true, read `core.lock().engaged()`, release that lock, then call `tick` on the ducker. The state lock is never held during socket I/O.
- In `run`, after the main loop sets `quit`, call `tick(false)` on the ducker once so quitting mid-conversation does not leave the music quiet.
- Rename `Core::todo_instructions` to `instructions`. Its value is `format!("{} {}", hermes_instructions(&config.todo_path), music::INSTRUCTIONS)`.

- [ ] **Step 4: Run all Rust checks**

Run: `cargo test --release && cargo clippy --release -- -D warnings`
Expected: all tests pass, no warnings.

- [ ] **Step 5: Update `README.md`**

Add a "Music" section after the todo section covering, in the README's existing voice: what to say ("hey john, play One More Time", "play some music", "pause", "next", "stop the music", "set the volume to 40"); that it needs `mpv` and a current `yt-dlp`, with `pip install -U "yt-dlp[default]"` as the fix when playback will not start; the `hermes mcp add` command from Task 4; that music is turned down to `--duck-volume` while it listens and speaks; that `--open-mic` is not suited to music because lyrics are sent as commands; and one line in the privacy note that song searches and streams go to YouTube. Add `--duck-volume` to the flag list with its default of 10 and the advice to use 0 if lyrics get picked up as commands.

- [ ] **Step 6: Commit**

```bash
git add src/app.rs src/main.rs README.md
git commit -m "feat: duck music while listening and route music requests to Hermes tools"
```

---

### Task 4: Register with Hermes on the Mac and check it end to end

This task changes files outside the repo (`~/.hermes/config.yaml`, the system `yt-dlp`). Steps 5 and 6 need a person in the room.

**Files:**
- Modify: `~/.hermes/config.yaml` (through `hermes mcp add`)

**Interfaces:**
- Consumes: `music/ytm_mcp.py` from Task 1, the built voice agent from Task 3.
- Produces: a working Mac setup.

- [ ] **Step 1: Update `yt-dlp`**

Run: `/Library/Frameworks/Python.framework/Versions/3.11/bin/python3 -m pip install -U "yt-dlp[default]" && yt-dlp --version`
Expected: `2026.11.23` or newer.

- [ ] **Step 2: Back up the config and register the server**

```bash
cp ~/.hermes/config.yaml ~/.hermes/config.yaml.bak-$(date -u +%Y%m%dT%H%M%SZ)
hermes mcp add youtube_music \
  --command ~/.hermes/hermes-agent/venv/bin/python3 \
  --env YT_DLP=/Library/Frameworks/Python.framework/Versions/3.11/bin/yt-dlp MPV=/opt/homebrew/bin/mpv \
  --args "$(pwd)/music/ytm_mcp.py"
```

If `hermes mcp add` insists on an interactive prompt, add the same entry by hand under `mcp_servers:` in `~/.hermes/config.yaml`, following the shape of the existing `agent-reach` entry, with `timeout: 120` and `connect_timeout: 30`.

- [ ] **Step 3: Test the connection and restart Hermes**

Run: `hermes mcp test youtube_music`
Expected: the six tools listed.
Run: `hermes gateway restart && sleep 5 && curl -s http://127.0.0.1:8642/health`
Expected: a healthy response. If the API server on this machine is started another way, restart it that way instead.

- [ ] **Step 4: Play through the Hermes API**

With the key from `~/.hermes/.env` (`API_SERVER_KEY`) in `$KEY`, start a run with `{"input": "play one more time by daft punk", "session_id": "ytm-check"}` on `POST /v1/runs`, then read `GET /v1/runs/<run_id>/events`.
Expected: a tool event naming `mcp_youtube_music_play`, a final reply of "Playing One More Time ...", and audible music. Then send `"stop the music"` in the same session. Expected: silence, and `pgrep -f input-ipc-server` prints nothing.

- [ ] **Step 5: Check by voice (needs a person)**

Run `cargo run --release`. Say "hey john, play some music". Check: music starts after the spoken reply and returns to full volume about 2 seconds after it. Say "hey john" over the music: it turns down at the chime. Say "next", then "set the volume to 40", then "stop the music".

- [ ] **Step 6: Check for lyrics leaking in (needs a person)**

Play a song with vocals at normal volume and stay silent through three wake and reply cycles. If the transcript shows lyrics sent as a command, rerun with `--duck-volume 0` and note the result in the README's flag description.

- [ ] **Step 7: Commit any README change from Step 6**

```bash
git add README.md
git commit -m "docs: note duck level found in testing"
```

---

### Task 5: Raspberry Pi rollout

Needs the Pi reachable over SSH and the branch pushed or merged. Skip until then.

**Files:**
- Modify on the Pi: `~/.hermes/config.yaml`, the kiosk launcher only if `--duck-volume` needs a non-default value.

**Interfaces:**
- Consumes: everything above.
- Produces: the same behaviour on the Pi.

- [ ] **Step 1: Install the players**

Run on the Pi: `sudo apt install -y mpv pipx && pipx install "yt-dlp[default]" && yt-dlp --version && mpv --version | head -1`
Expected: `yt-dlp` 2026.11.23 or newer, and an mpv version line.

- [ ] **Step 2: Update and build**

Run on the Pi: `cd ~/Hermes-Voice-Agent && git pull && cargo build --release`
Expected: a finished release build.

- [ ] **Step 3: Register the server**

Same as Task 4 Step 2, with `YT_DLP=$(which yt-dlp)`, `MPV=$(which mpv)`, and `--args /home/rangelkoli/Hermes-Voice-Agent/music/ytm_mcp.py`. Then `hermes mcp test youtube_music` and restart Hermes.

- [ ] **Step 4: Confirm mpv plays into the echo-cancelled sink**

Start a song by voice, then run `pw-link -l | grep -i -A2 mpv`.
Expected: mpv linked to `ec-sink`, not directly to the headphone jack. If not, add `--audio-device=pipewire/ec-sink` to `mpv_args` behind an `MPV_AUDIO_DEVICE` environment variable set in the Pi's MCP entry.

- [ ] **Step 5: Check the wake phrase over music (needs a person)**

Restart `hermes-voice` with `--show-scores`. With music at normal volume, say "hey john" five times from the usual distance.
Expected: five wakes. If it misses, read the scores and lower `--wake-threshold`, or lower the music with "set the volume to 60". Record the working values in the kiosk launcher.
