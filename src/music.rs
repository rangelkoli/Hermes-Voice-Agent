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
    call the `youtube_music` tools: play, pause, resume, next, stop, set_volume. For a request that names no song, \
    choose a fitting search query yourself. Starting a song can take 15 seconds. After play succeeds, reply with only \
    `Playing <title>`. If play fails, say so once and do not retry.";

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
///
/// It adds an mpv audio filter labelled `duck` and removes it again. The player's own volume is
/// never written, so a volume the user asks for while the music is ducked is kept. Whether the
/// music is ducked is read from the player each time, so a song started mid-conversation is
/// ducked, and a player left ducked by a killed voice agent is restored by the next one.
pub struct Ducker {
    path: PathBuf,
    /// Share of the music's volume left while ducked, 0 to 1.
    factor: f64,
}

impl Ducker {
    pub fn new(path: PathBuf, level: u8) -> Self {
        Self { path, factor: f64::from(level.min(100)) / 100.0 }
    }

    /// Call a few times a second. Never blocks longer than one socket timeout per request.
    pub fn tick(&self, engaged: bool) {
        let Some(filters) = request(&self.path, json!(["get_property", "af"])) else { return };
        let ducked = filters.as_array().is_some_and(|f| f.iter().any(|f| f["label"] == "duck"));
        if engaged && !ducked {
            request(&self.path, json!(["af", "add", format!("@duck:lavfi=[volume={:.2}]", self.factor)]));
        } else if !engaged && ducked {
            request(&self.path, json!(["af", "remove", "@duck"]));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::net::UnixListener;
    use std::sync::{Arc, Mutex};
    use std::time::Instant;

    /// Answers audio filter requests the way mpv does, with an event line before each reply.
    struct FakeMpv {
        path: PathBuf,
        ducked: Arc<Mutex<bool>>,
        /// Every command that was not a read, as JSON.
        writes: Arc<Mutex<Vec<String>>>,
    }

    impl FakeMpv {
        fn new(name: &str, ducked: bool) -> Self {
            let path = PathBuf::from(format!("/tmp/hv-mpv-{}-{name}.sock", std::process::id()));
            let _ = std::fs::remove_file(&path);
            let listener = UnixListener::bind(&path).unwrap();
            let ducked = Arc::new(Mutex::new(ducked));
            let writes = Arc::new(Mutex::new(Vec::new()));
            let (d, w) = (ducked.clone(), writes.clone());
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
                        let filters = if *d.lock().unwrap() { json!([{"name": "lavfi", "label": "duck"}]) } else { json!([]) };
                        json!({"data": filters, "error": "success"})
                    } else {
                        if command[0] == "af" {
                            *d.lock().unwrap() = command[1] == "add";
                        }
                        w.lock().unwrap().push(command.to_string());
                        json!({"data": null, "error": "success"})
                    };
                    let _ = write!(conn, "{{\"event\":\"playback-restart\"}}\n{reply}\n");
                }
            });
            Self { path, ducked, writes }
        }

        fn path(&self) -> PathBuf {
            self.path.clone()
        }

        fn writes(&self) -> Vec<String> {
            self.writes.lock().unwrap().clone()
        }

        /// A new song replaced the player: it starts with no filters.
        fn new_player(&self) {
            *self.ducked.lock().unwrap() = false;
        }
    }

    impl Drop for FakeMpv {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.path);
        }
    }

    const ADD: &str = r#"["af","add","@duck:lavfi=[volume=0.10]"]"#;
    const REMOVE: &str = r#"["af","remove","@duck"]"#;

    #[test]
    fn duck_adds_the_filter_once_and_removes_it_once() {
        let mpv = FakeMpv::new("once", false);
        let d = Ducker::new(mpv.path(), 10);
        d.tick(true);
        d.tick(true);
        assert_eq!(mpv.writes(), [ADD]);
        d.tick(false);
        d.tick(false);
        assert_eq!(mpv.writes(), [ADD, REMOVE]);
    }

    #[test]
    fn ducking_never_writes_the_volume() {
        // "Set the volume to 5" while ducked must still be 5 afterwards, so the volume is not ours.
        let mpv = FakeMpv::new("volume", false);
        let d = Ducker::new(mpv.path(), 10);
        d.tick(true);
        d.tick(false);
        assert!(!mpv.writes().iter().any(|c| c.contains("set_property")), "{:?}", mpv.writes());
    }

    #[test]
    fn a_song_started_while_engaged_is_ducked() {
        let mpv = FakeMpv::new("newsong", false);
        let d = Ducker::new(mpv.path(), 10);
        d.tick(true);
        mpv.new_player();
        d.tick(true);
        assert_eq!(mpv.writes(), [ADD, ADD]);
    }

    #[test]
    fn a_player_left_ducked_is_restored() {
        // The voice agent was killed mid-conversation and started again.
        let mpv = FakeMpv::new("leftover", true);
        Ducker::new(mpv.path(), 10).tick(false);
        assert_eq!(mpv.writes(), [REMOVE]);
    }

    #[test]
    fn duck_volume_sets_the_filter_strength() {
        let mpv = FakeMpv::new("strength", false);
        Ducker::new(mpv.path(), 25).tick(true);
        mpv.new_player();
        Ducker::new(mpv.path(), 0).tick(true);
        assert_eq!(
            mpv.writes(),
            [r#"["af","add","@duck:lavfi=[volume=0.25]"]"#, r#"["af","add","@duck:lavfi=[volume=0.00]"]"#]
        );
    }

    #[test]
    fn missing_socket_is_a_noop() {
        let d = Ducker::new(PathBuf::from("/tmp/hv-mpv-does-not-exist.sock"), 10);
        d.tick(true);
        d.tick(false);
    }

    #[test]
    fn silent_socket_does_not_stall() {
        let path = PathBuf::from(format!("/tmp/hv-mpv-{}-silent.sock", std::process::id()));
        let _ = std::fs::remove_file(&path);
        // Accepts into its backlog and never replies.
        let _listener = UnixListener::bind(&path).unwrap();
        let d = Ducker::new(path.clone(), 10);
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
        // Hermes prefixes MCP tool names differently between versions, so only the server is named.
        assert!(INSTRUCTIONS.contains("`youtube_music` tools"));
        assert!(!INSTRUCTIONS.contains("mcp_"));
    }
}
