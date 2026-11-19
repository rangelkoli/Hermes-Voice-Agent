//! Persistent Markdown checklist todos shared between the UI and Hermes.

use std::fmt::Write as _;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

static TMP_COUNTER: AtomicU64 = AtomicU64::new(0);

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TodoItem {
    pub text: String,
    pub done: bool,
}

/// Default todo list file: `~/.config/hermes-voice/todos.md`.
pub fn default_todo_path() -> PathBuf {
    PathBuf::from(std::env::var_os("HOME").unwrap_or_default()).join(".config/hermes-voice/todos.md")
}

/// Resolves `--todo-file` (expanding a leading `~/` and turning relative paths into absolute ones),
/// falling back to `default_todo_path()`.
pub fn resolve_todo_path(explicit: Option<&str>) -> PathBuf {
    let path = match explicit {
        Some(raw) => {
            if let Some(rest) = raw.strip_prefix("~/") {
                PathBuf::from(std::env::var_os("HOME").unwrap_or_default()).join(rest)
            } else {
                PathBuf::from(raw)
            }
        }
        None => default_todo_path(),
    };
    if path.is_absolute() {
        path
    } else {
        std::env::current_dir().map(|cwd| cwd.join(&path)).unwrap_or(path)
    }
}

/// Ensures the todo file and its parent directory exist so Hermes can read it without error.
pub fn ensure_todo_file(path: &Path) -> io::Result<()> {
    if path.exists() {
        return Ok(());
    }
    if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
        fs::create_dir_all(parent)?;
    }
    match fs::OpenOptions::new().write(true).create_new(true).open(path) {
        Ok(_) => Ok(()),
        Err(e) if e.kind() == io::ErrorKind::AlreadyExists => Ok(()),
        Err(e) => Err(e),
    }
}

/// Parses standard Markdown checklist lines (`- [ ]`, `- [x]`, `- [X]`, `* [ ]`, `* [x]`, `* [X]`).
pub fn parse_todos(input: &str) -> Vec<TodoItem> {
    input.lines().filter_map(parse_line).collect()
}

fn parse_line(line: &str) -> Option<TodoItem> {
    let trimmed = line.trim_start_matches([' ', '\t']);
    let rest = trimmed.strip_prefix("- ").or_else(|| trimmed.strip_prefix("* "))?;
    let (done, text) = if let Some(text) = rest.strip_prefix("[ ] ") {
        (false, text)
    } else if let Some(text) = rest.strip_prefix("[x] ").or_else(|| rest.strip_prefix("[X] ")) {
        (true, text)
    } else {
        return None;
    };
    let text = text.trim();
    if text.is_empty() {
        return None;
    }
    Some(TodoItem { text: text.to_string(), done })
}

/// Formats items as Markdown checklist lines (`- [ ] <text>\n` / `- [x] <text>\n`).
pub fn format_todos(items: &[TodoItem]) -> String {
    let mut out = String::new();
    for item in items {
        let mark = if item.done { 'x' } else { ' ' };
        let _ = writeln!(out, "- [{mark}] {}", item.text);
    }
    out
}

/// Loads todos from `path`. Missing files return an empty list.
pub fn load_todos(path: &Path) -> io::Result<Vec<TodoItem>> {
    match fs::read_to_string(path) {
        Ok(text) => Ok(parse_todos(&text)),
        Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(Vec::new()),
        Err(e) => Err(e),
    }
}

/// Atomically writes `items` to `path`, creating missing parent directories as needed.
pub fn save_todos(path: &Path, items: &[TodoItem]) -> io::Result<()> {
    if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
        fs::create_dir_all(parent)?;
    }
    let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("todos");
    let seq = TMP_COUNTER.fetch_add(1, Ordering::Relaxed);
    let tmp = path.with_file_name(format!(".{name}.tmp.{}.{seq}", std::process::id()));
    let content = format_todos(items);
    if let Err(e) = fs::write(&tmp, content) {
        let _ = fs::remove_file(&tmp);
        return Err(e);
    }
    if let Err(e) = fs::rename(&tmp, path) {
        let _ = fs::remove_file(&tmp);
        return Err(e);
    }
    Ok(())
}

/// Lightweight poller that reports parsed todos only when the backing file changes.
pub struct TodoWatcher {
    path: PathBuf,
    last: Option<String>,
}

impl TodoWatcher {
    pub fn new(path: PathBuf) -> Self {
        Self { path, last: None }
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Re-reads the file and returns updated items only when its raw contents changed.
    /// Non-`NotFound` I/O errors are ignored so a concurrent write does not wipe the list.
    pub fn poll(&mut self) -> Option<Vec<TodoItem>> {
        let current = match fs::read_to_string(&self.path) {
            Ok(text) => Some(text),
            Err(e) if e.kind() == io::ErrorKind::NotFound => None,
            Err(_) => return None,
        };
        if current == self.last {
            return None;
        }
        let items = current.as_deref().map_or_else(Vec::new, parse_todos);
        self.last = current;
        Some(items)
    }
}

/// System prompt instructions directing Hermes to read and edit the on-screen todo file.
pub fn hermes_instructions(path: &Path) -> String {
    let path = path.display();
    format!(
        "The user's persistent on-screen todo list is stored at `{path}` in Markdown checklist format. \
         When the user asks to view, add, complete, uncheck, edit, reorder, or remove todos, \
         use `read_file` and `write_file` directly on `{path}` without calling `skill_view`, `terminal`, \
         or the built-in `todo_list` tool. \
         Always call `read_file` on `{path}` before writing so external edits are kept. \
         Format the file strictly as one item per line using `- [ ] <task>` for open items and `- [x] <task>` \
         for completed items, with no headings or extra prose."
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    struct TempDir(PathBuf);

    impl TempDir {
        fn new(label: &str) -> Self {
            static COUNTER: AtomicU64 = AtomicU64::new(0);
            let seq = COUNTER.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir().join(format!("hv-todo-{label}-{}-{seq}", std::process::id()));
            fs::create_dir_all(&path).unwrap();
            Self(path)
        }

        fn path(&self) -> &Path {
            &self.0
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn default_path_is_under_config_dir() {
        let path = default_todo_path();
        assert!(path.ends_with(".config/hermes-voice/todos.md"), "{}", path.display());
    }

    #[test]
    fn resolve_path_expands_tilde_and_relative_paths() {
        assert_eq!(resolve_todo_path(None), default_todo_path());
        assert_eq!(resolve_todo_path(Some("/tmp/custom.md")), PathBuf::from("/tmp/custom.md"));
        let tilde = resolve_todo_path(Some("~/notes/todos.md"));
        assert!(tilde.ends_with("notes/todos.md"), "{}", tilde.display());
        assert!(!tilde.starts_with("~"), "{}", tilde.display());
        let rel = resolve_todo_path(Some("local-todos.md"));
        assert!(rel.is_absolute(), "{}", rel.display());
        assert!(rel.ends_with("local-todos.md"), "{}", rel.display());
    }

    #[test]
    fn ensure_todo_file_creates_empty_file_and_preserves_existing() {
        let dir = TempDir::new("ensure");
        let path = dir.path().join("nested/todos.md");
        ensure_todo_file(&path).unwrap();
        assert!(path.is_file());
        assert_eq!(fs::read_to_string(&path).unwrap(), "");

        fs::write(&path, "- [ ] Keep me\n").unwrap();
        ensure_todo_file(&path).unwrap();
        assert_eq!(fs::read_to_string(&path).unwrap(), "- [ ] Keep me\n");
    }

    #[test]
    fn parse_handles_markers_indentation_and_non_items() {
        let md = [
            "# Today",
            "",
            "- [ ] Buy milk",
            "  - [x] Walk the dog",
            "\t- [X] Send report  ",
            "* [ ]   Read chapter 4",
            "  * [x] Fix bug #8",
            "* [X] Clean desk",
            "- [ ]    ",
            "- [x]",
            "- Plain bullet",
            "Just a note",
        ]
        .join("\n");
        assert_eq!(
            parse_todos(&md),
            vec![
                TodoItem { text: "Buy milk".into(), done: false },
                TodoItem { text: "Walk the dog".into(), done: true },
                TodoItem { text: "Send report".into(), done: true },
                TodoItem { text: "Read chapter 4".into(), done: false },
                TodoItem { text: "Fix bug #8".into(), done: true },
                TodoItem { text: "Clean desk".into(), done: true },
            ]
        );
    }

    #[test]
    fn format_and_parse_roundtrip() {
        let items = vec![
            TodoItem { text: "Call Alex".into(), done: false },
            TodoItem { text: "Ship release".into(), done: true },
        ];
        let formatted = format_todos(&items);
        assert_eq!(formatted, "- [ ] Call Alex\n- [x] Ship release\n");
        assert_eq!(parse_todos(&formatted), items);
    }

    #[test]
    fn load_missing_file_returns_empty() {
        let dir = TempDir::new("missing");
        let path = dir.path().join("does-not-exist.md");
        assert_eq!(load_todos(&path).unwrap(), Vec::new());
    }

    #[test]
    fn save_and_load_roundtrip_creates_parent_dirs() {
        let dir = TempDir::new("roundtrip");
        let path = dir.path().join("nested/sub/todos.md");
        let items = vec![
            TodoItem { text: "First".into(), done: false },
            TodoItem { text: "Second".into(), done: true },
        ];
        save_todos(&path, &items).unwrap();
        assert_eq!(load_todos(&path).unwrap(), items);
    }

    #[test]
    fn watcher_detects_initial_edit_and_delete() {
        let dir = TempDir::new("watcher");
        let path = dir.path().join("todos.md");

        // Pre-existing file is picked up on first poll.
        let initial = vec![TodoItem { text: "Initial".into(), done: false }];
        save_todos(&path, &initial).unwrap();

        let mut watcher = TodoWatcher::new(path.clone());
        assert_eq!(watcher.path(), path.as_path());
        assert_eq!(watcher.poll(), Some(initial));
        assert_eq!(watcher.poll(), None);

        // Editing the file triggers a new poll result.
        let updated = vec![
            TodoItem { text: "Initial".into(), done: true },
            TodoItem { text: "Added".into(), done: false },
        ];
        save_todos(&path, &updated).unwrap();
        assert_eq!(watcher.poll(), Some(updated));
        assert_eq!(watcher.poll(), None);

        // Deleting the file yields an empty list once, then stays quiet.
        fs::remove_file(&path).unwrap();
        assert_eq!(watcher.poll(), Some(Vec::new()));
        assert_eq!(watcher.poll(), None);
    }

    #[test]
    fn watcher_ignores_missing_file_until_created_and_ignores_io_errors() {
        let dir = TempDir::new("watcher-missing");
        let path = dir.path().join("todos.md");

        let mut watcher = TodoWatcher::new(path.clone());
        assert_eq!(watcher.poll(), None);

        let items = vec![TodoItem { text: "Created".into(), done: false }];
        save_todos(&path, &items).unwrap();
        assert_eq!(watcher.poll(), Some(items));

        // Invalid UTF-8 triggers a non-NotFound I/O error; poll must not wipe items.
        fs::write(&path, [0xff, 0xfe]).unwrap();
        assert_eq!(watcher.poll(), None);
    }

    #[test]
    fn hermes_instructions_mentions_path_and_checklist_format() {
        let path = Path::new("/tmp/custom/todos.md");
        let prompt = hermes_instructions(path);
        assert!(prompt.contains("/tmp/custom/todos.md"), "{prompt}");
        assert!(prompt.contains("- [ ] "), "{prompt}");
        assert!(prompt.contains("- [x] "), "{prompt}");
        assert!(prompt.contains("todo_list"), "{prompt}");
    }

    #[test]
    #[ignore = "requires a running local Hermes gateway at http://127.0.0.1:8642"]
    fn live_hermes_can_edit_todo_file() {
        let key = crate::hermes::load_api_key().expect("HERMES_API_KEY or ~/.hermes/.env API_SERVER_KEY");
        let client = crate::hermes::HermesClient::new(&key, "http://127.0.0.1:8642");
        let dir = TempDir::new("live-e2e");
        let path = dir.path().join("todos.md");
        ensure_todo_file(&path).unwrap();

        let instructions = hermes_instructions(&path);
        let session = format!("voice-e2e-todo-{}", std::process::id());
        let mut watcher = TodoWatcher::new(path.clone());
        let _ = watcher.poll();

        // Turn 1: add two items
        let run1 = client
            .start_run_with_instructions(
                "Add 'Buy milk' and 'Call dentist' to my todo list.",
                &session,
                Some(&instructions),
            )
            .unwrap();
        let mut tools1 = Vec::new();
        let mut approval_seen = false;
        client
            .events(&run1, |ev| {
                if ev.name == "tool.started" {
                    if let Some(t) = ev.data["tool"].as_str() {
                        tools1.push(t.to_string());
                    }
                }
                if ev.name == "approval.request" {
                    approval_seen = true;
                }
            })
            .unwrap();
        assert!(!approval_seen, "unexpected approval prompt on turn 1");
        let after1 = watcher.poll().expect("turn 1 should modify todo file");
        eprintln!("Turn 1 tools: {tools1:?}, items: {after1:?}");
        assert!(after1.iter().any(|t| !t.done && t.text.eq_ignore_ascii_case("Buy milk")), "{after1:?}");
        assert!(after1.iter().any(|t| !t.done && t.text.eq_ignore_ascii_case("Call dentist")), "{after1:?}");

        // Turn 2: mark one done and add a third
        let run2 = client
            .start_run_with_instructions(
                "Mark 'Buy milk' as done and add 'Walk the dog' to my todo list.",
                &session,
                Some(&instructions),
            )
            .unwrap();
        let mut tools2 = Vec::new();
        client
            .events(&run2, |ev| {
                if ev.name == "tool.started" {
                    if let Some(t) = ev.data["tool"].as_str() {
                        tools2.push(t.to_string());
                    }
                }
                if ev.name == "approval.request" {
                    approval_seen = true;
                }
            })
            .unwrap();
        assert!(!approval_seen, "unexpected approval prompt on turn 2");
        let after2 = watcher.poll().expect("turn 2 should modify todo file");
        eprintln!("Turn 2 tools: {tools2:?}, items: {after2:?}");
        assert!(after2.iter().any(|t| t.done && t.text.eq_ignore_ascii_case("Buy milk")), "{after2:?}");
        assert!(after2.iter().any(|t| !t.done && t.text.eq_ignore_ascii_case("Call dentist")), "{after2:?}");
        assert!(after2.iter().any(|t| !t.done && t.text.eq_ignore_ascii_case("Walk the dog")), "{after2:?}");
    }
}
