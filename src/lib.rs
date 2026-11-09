pub mod app;
pub mod audio;
pub mod gemini;
pub mod hermes;
pub mod orb;
pub mod stt;
pub mod wake;

use std::path::Path;

/// Value of `key` in a dotenv file. Last assignment wins, like python-dotenv.
// ponytail: KEY=VALUE lines, optional `export` and quotes. No interpolation or multi-line values.
pub fn dotenv_get(path: &Path, key: &str) -> Option<String> {
    let text = std::fs::read_to_string(path).ok()?;
    let mut found = None;
    for line in text.lines() {
        let line = line.trim();
        let line = line.strip_prefix("export ").unwrap_or(line);
        let Some((name, value)) = line.split_once('=') else { continue };
        if name.trim() != key {
            continue;
        }
        let value = value.trim();
        let unquoted = ['"', '\'']
            .iter()
            .find_map(|q| value.strip_prefix(*q).and_then(|v| v.strip_suffix(*q)));
        let value = match unquoted {
            Some(v) => v,
            None => value.split(" #").next().unwrap_or("").trim(),
        };
        found = Some(value.to_string());
    }
    found.filter(|v| !v.is_empty())
}

#[cfg(test)]
mod tests {
    #[test]
    fn dotenv_reads_quoted_and_plain_values() {
        let path = std::env::temp_dir().join(format!("hv-env-{}", std::process::id()));
        std::fs::write(&path, "# c\nA=1\nexport B=\"two\" \nC=x # note\nA=3\n").unwrap();
        assert_eq!(super::dotenv_get(&path, "A").as_deref(), Some("3"));
        assert_eq!(super::dotenv_get(&path, "B").as_deref(), Some("two"));
        assert_eq!(super::dotenv_get(&path, "C").as_deref(), Some("x"));
        assert_eq!(super::dotenv_get(&path, "D"), None);
        std::fs::remove_file(path).unwrap();
    }
}
