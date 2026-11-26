//! Music played by the Hermes `youtube_music` tools runs in mpv. This turns it down over mpv's
//! control socket while the assistant listens, works, or speaks.

use std::ffi::OsString;
use std::io::{BufRead, BufReader, Write};
use std::os::unix::net::UnixStream;
use std::path::{Path, PathBuf};
use std::time::Duration;

use serde_json::{json, Value};

/// Sentence added to every Hermes run so music requests reach the MCP tools.
pub const INSTRUCTIONS: &str = "When the user asks to play, pause, resume, skip, or stop music, or to change its volume, \
    call the `mcp_youtube_music_*` tools. For a request that names no song, choose a fitting search query yourself. \
    After `mcp_youtube_music_play` succeeds, reply with only `Playing <title>`.";

/// A stuck player must not hold up the voice loop.
const TIMEOUT: Duration = Duration::from_millis(200);

/// Where mpv listens. `music/ytm_mcp.py` resolves the same path.
pub fn socket_path() -> PathBuf {
    socket_path_from(std::env::var_os("HERMES_VOICE_MPV_SOCKET"), std::env::var_os("HOME"))
}

fn socket_path_from(overridden: Option<OsString>, home: Option<OsString>) -> PathBuf {
    match overridden.filter(|p| !p.is_empty()) {
        Some(path) => PathBuf::from(path),
        None => PathBuf::from(home.unwrap_or_default()).join(".config/hermes-voice/mpv.sock"),
    }
}

/// One mpv IPC request. The reply's data, or None when no player answers.
fn request(path: &Path, command: Value) -> Option<Value> {
    let mut stream = UnixStream::connect(path).ok()?;
    stream.set_read_timeout(Some(TIMEOUT)).ok()?;
    stream.set_write_timeout(Some(TIMEOUT)).ok()?;
    writeln!(stream, "{}", json!({ "command": command })).ok()?;
    for line in BufReader::new(stream).lines() {
        let reply: Value = serde_json::from_str(&line.ok()?).ok()?;
        // Event lines carry no "error" field. Only the reply does.
        if let Some(error) = reply.get("error") {
            return (error == "success").then(|| reply["data"].clone());
        }
    }
    None
}

/// Lowers the music while the assistant is engaged and puts it back afterwards.
pub struct Ducker {
    path: PathBuf,
    level: f64,
    /// The volume to return to. Some only while ducked.
    saved: Option<f64>,
}

impl Ducker {
    pub fn new(path: PathBuf, level: u8) -> Self {
        Self { path, level: f64::from(level), saved: None }
    }

    /// Call a few times a second. Never blocks longer than one socket timeout per request.
    // ponytail: a volume the user sets at or below the duck level while ducked is not remembered.
    // Track the user's volume in the MCP server if that matters.
    pub fn tick(&mut self, engaged: bool) {
        if !engaged {
            if let Some(volume) = self.saved.take() {
                request(&self.path, json!(["set_property", "volume", volume]));
            }
            return;
        }
        // Read every time: a song started mid-conversation, or a spoken volume change, comes in loud.
        let volume = request(&self.path, json!(["get_property", "volume"])).and_then(|v| v.as_f64());
        if let Some(volume) = volume.filter(|v| *v > self.level + 0.5) {
            self.saved = Some(volume);
            request(&self.path, json!(["set_property", "volume", self.level]));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{BufRead, BufReader, Write};
    use std::os::unix::net::UnixListener;
    use std::sync::{Arc, Mutex};
    use std::time::{Duration, Instant};

    /// Answers volume requests the way mpv does, with an event line before each reply.
    struct FakeMpv {
        path: PathBuf,
        volume: Arc<Mutex<f64>>,
        sets: Arc<Mutex<Vec<f64>>>,
    }

    impl FakeMpv {
        fn new(name: &str, volume: f64) -> Self {
            let path = PathBuf::from(format!("/tmp/hv-mpv-{}-{name}.sock", std::process::id()));
            let _ = std::fs::remove_file(&path);
            let listener = UnixListener::bind(&path).unwrap();
            let volume = Arc::new(Mutex::new(volume));
            let sets = Arc::new(Mutex::new(Vec::new()));
            let (v, s) = (volume.clone(), sets.clone());
            std::thread::spawn(move || {
                for conn in listener.incoming() {
                    let Ok(mut conn) = conn else { return };
                    let mut line = String::new();
                    if BufReader::new(&conn).read_line(&mut line).is_err() {
                        continue;
                    }
                    let Ok(request) = serde_json::from_str::<Value>(&line) else { continue };
                    let command = &request["command"];
                    let reply = if command[0] == "get_property" {
                        json!({"data": *v.lock().unwrap(), "error": "success"})
                    } else {
                        let value = command[2].as_f64().unwrap();
                        *v.lock().unwrap() = value;
                        s.lock().unwrap().push(value);
                        json!({"error": "success"})
                    };
                    let _ = write!(conn, "{{\"event\":\"playback-restart\"}}\n{reply}\n");
                }
            });
            Self { path, volume, sets }
        }

        fn path(&self) -> PathBuf {
            self.path.clone()
        }

        fn sets(&self) -> Vec<f64> {
            self.sets.lock().unwrap().clone()
        }

        fn set_volume(&self, volume: f64) {
            *self.volume.lock().unwrap() = volume;
        }
    }

    impl Drop for FakeMpv {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.path);
        }
    }

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
        let path = PathBuf::from(format!("/tmp/hv-mpv-{}-silent.sock", std::process::id()));
        let _ = std::fs::remove_file(&path);
        // Accepts into its backlog and never replies.
        let _listener = UnixListener::bind(&path).unwrap();
        let mut d = Ducker::new(path.clone(), 10);
        let started = Instant::now();
        d.tick(true);
        assert!(started.elapsed() < Duration::from_secs(1));
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn socket_path_honours_the_override() {
        let home = Some(OsString::from("/home/me"));
        assert_eq!(socket_path_from(Some("/tmp/x.sock".into()), home.clone()), PathBuf::from("/tmp/x.sock"));
        let default = PathBuf::from("/home/me/.config/hermes-voice/mpv.sock");
        assert_eq!(socket_path_from(None, home.clone()), default);
        assert_eq!(socket_path_from(Some(OsString::new()), home), default);
    }

    #[test]
    fn instructions_name_the_tools() {
        assert!(INSTRUCTIONS.contains("mcp_youtube_music_"));
    }
}
