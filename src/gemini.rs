//! Gemini transcription and spoken replies.

use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{channel, RecvTimeoutError};
use std::sync::Arc;
use std::time::Duration;

use base64::{engine::general_purpose::STANDARD, Engine};
use regex::Regex;
use serde_json::{json, Value};

use crate::audio::play;
use crate::stt::Stt;

const BASE_URL: &str = "https://generativelanguage.googleapis.com/v1beta";
// ponytail: 1200 characters bounds TTS cost and listening time.
const SPEAK_LIMIT: usize = 1200;
const SPEAK_NOTE: &str = " The rest is on screen.";

pub fn load_gemini_key() -> Result<String, String> {
    if let Ok(key) = std::env::var("GEMINI_API_KEY") {
        if !key.is_empty() {
            return Ok(key);
        }
    }
    let path = PathBuf::from(std::env::var_os("HOME").unwrap_or_default()).join(".config/hermes-voice/.env");
    crate::dotenv_get(&path, "GEMINI_API_KEY").ok_or_else(|| {
        format!(
            "No Gemini API key in {}. Create one at aistudio.google.com/apikey. --local needs none.",
            path.display()
        )
    })
}

/// Emphasis underscores only. snake_case keeps its underscores.
fn drop_loose_underscores(line: &str) -> String {
    let chars: Vec<char> = line.chars().collect();
    let alnum = |i: Option<usize>| i.and_then(|i| chars.get(i)).is_some_and(|c| c.is_ascii_alphanumeric());
    chars
        .iter()
        .enumerate()
        .filter(|&(i, &c)| c != '_' || (alnum(i.checked_sub(1)) && alnum(Some(i + 1))))
        .map(|(_, c)| *c)
        .collect()
}

pub fn speakable(text: &str) -> String {
    let fence = Regex::new(r"(?s)```.*?```").unwrap();
    let link = Regex::new(r"\[([^\]]*)\]\([^)]*\)").unwrap();
    let list = Regex::new(r"^[-*+]\s+").unwrap();
    let heading = Regex::new(r"^#+\s*").unwrap();
    let cleaned = link.replace_all(&fence.replace_all(text, ""), "$1").into_owned();
    let lines: Vec<String> = cleaned
        .split('\n')
        .map(str::trim)
        .filter(|line| !line.starts_with('|'))
        .map(|line| {
            let line = heading.replace(&list.replace(line, ""), "").replace(['*', '`'], "");
            drop_loose_underscores(&line)
        })
        .collect();
    let cleaned = Regex::new(r"\n{3,}").unwrap().replace_all(&lines.join("\n"), "\n\n").trim().to_string();
    if cleaned.chars().count() <= SPEAK_LIMIT {
        return cleaned;
    }
    let cut: String = cleaned.chars().take(SPEAK_LIMIT).collect();
    let head = match cut.rfind(['.', '!', '?']) {
        Some(end) => &cut[..=end],
        None => &cut,
    };
    format!("{}{SPEAK_NOTE}", head.trim_end())
}

fn agent() -> ureq::Agent {
    ureq::AgentBuilder::new().timeout(Duration::from_secs(30)).build()
}

fn scrub(message: &str, key: &str) -> String {
    let message = if key.is_empty() { message.to_string() } else { message.replace(key, "") };
    let message = message.trim();
    if message.is_empty() { "Gemini request failed".into() } else { message.into() }
}

fn error_text(error: ureq::Error, key: &str) -> String {
    let message = match error {
        ureq::Error::Status(code, response) => {
            let body = response.into_string().unwrap_or_default();
            let parsed: Value = serde_json::from_str(&body).unwrap_or(Value::Null);
            let message = match &parsed["error"] {
                Value::Object(e) => e.get("message").and_then(Value::as_str).unwrap_or("").to_string(),
                Value::String(s) => s.clone(),
                _ => String::new(),
            };
            if !message.is_empty() {
                message
            } else if !body.trim().is_empty() {
                body.trim().to_string()
            } else {
                format!("Gemini request failed ({code})")
            }
        }
        other => other.to_string(),
    };
    scrub(&message, key)
}

fn wav_base64(pcm: &[i16]) -> String {
    let data_len = (pcm.len() * 2) as u32;
    let mut wav = Vec::with_capacity(44 + data_len as usize);
    wav.extend_from_slice(b"RIFF");
    wav.extend_from_slice(&(36 + data_len).to_le_bytes());
    wav.extend_from_slice(b"WAVEfmt ");
    // PCM, mono, 16 kHz, 32000 bytes per second, 2-byte frames, 16 bits.
    for field in [16u32.to_le_bytes().as_slice(), &1u16.to_le_bytes(), &1u16.to_le_bytes(),
                  &16000u32.to_le_bytes(), &32000u32.to_le_bytes(), &2u16.to_le_bytes(), &16u16.to_le_bytes()] {
        wav.extend_from_slice(field);
    }
    wav.extend_from_slice(b"data");
    wav.extend_from_slice(&data_len.to_le_bytes());
    wav.extend(pcm.iter().flat_map(|s| s.to_le_bytes()));
    STANDARD.encode(wav)
}

fn transcript(payload: &Value) -> String {
    let empty = vec![];
    payload["steps"]
        .as_array()
        .unwrap_or(&empty)
        .iter()
        .filter(|step| step["type"] == "model_output")
        .flat_map(|step| step["content"].as_array().unwrap_or(&empty).iter())
        .filter(|item| item["type"] == "text")
        .map(|item| item["text"].as_str().unwrap_or(""))
        .collect()
}

pub struct GeminiTranscriber {
    pub model: String,
    key: String,
    agent: ureq::Agent,
}

impl GeminiTranscriber {
    pub fn new(key: &str, model: &str) -> Self {
        Self { model: model.into(), key: key.into(), agent: agent() }
    }
}

impl Stt for GeminiTranscriber {
    fn load(&mut self) -> Result<(), String> {
        let name = self.model.trim_start_matches("models/");
        self.agent
            .get(&format!("{BASE_URL}/models/{name}"))
            .set("x-goog-api-key", &self.key)
            .call()
            .map(drop)
            .map_err(|e| error_text(e, &self.key))
    }

    fn transcribe(&mut self, pcm: &[i16]) -> Result<String, String> {
        let model = if self.model.starts_with("models/") { self.model.clone() } else { format!("models/{}", self.model) };
        let body = json!({
            "model": model,
            "input": [{"type": "audio", "data": wav_base64(pcm), "mime_type": "audio/wav"}],
            "generation_config": {"transcription_config": {"custom_vocabulary": ["Hermes"]}},
        });
        let mut last = String::new();
        for _attempt in 0..2 {
            let response = self
                .agent
                .post(&format!("{BASE_URL}/interactions"))
                .set("x-goog-api-key", &self.key)
                .send_json(&body);
            match response {
                Ok(r) => return r.into_json::<Value>().map(|v| transcript(&v)).map_err(|e| e.to_string()),
                // One retry for a dropped connection, 429, or 5xx.
                Err(ureq::Error::Status(code, r)) if code == 429 || code >= 500 => {
                    last = error_text(ureq::Error::Status(code, r), &self.key)
                }
                Err(e @ ureq::Error::Transport(_)) => last = scrub(&e.to_string(), &self.key),
                Err(e) => return Err(error_text(e, &self.key)),
            }
        }
        Err(last)
    }
}

pub struct GeminiSpeaker {
    model: String,
    voice: String,
    key: String,
    agent: ureq::Agent,
}

impl GeminiSpeaker {
    pub fn new(key: &str, model: &str, voice: &str) -> Self {
        Self { model: model.into(), voice: voice.into(), key: key.into(), agent: agent() }
    }

    /// Streams 24 kHz int16 audio for `text`, calling `on_chunk` per piece until it returns false.
    fn stream(&self, text: &str, mut on_chunk: impl FnMut(Vec<u8>) -> bool) -> Result<(), String> {
        let body = json!({
            "model": self.model,
            "stream": true,
            "input": [{"type": "user_input", "content": [{"type": "text", "text": text}]}],
            "response_format": {"type": "audio", "mime_type": "audio/l16", "sample_rate": 24000},
            "generation_config": {"speech_config": [{"voice": self.voice}]},
        });
        let response = self
            .agent
            .post(&format!("{BASE_URL}/interactions"))
            .set("x-goog-api-key", &self.key)
            .send_json(body)
            .map_err(|e| error_text(e, &self.key))?;
        for line in BufReader::new(response.into_reader()).lines() {
            let line = line.map_err(|e| scrub(&e.to_string(), &self.key))?;
            let Some(raw) = line.strip_prefix("data:").map(str::trim) else { continue };
            let Ok(event) = serde_json::from_str::<Value>(raw) else { continue };
            let delta = &event["delta"];
            if delta["type"] != "audio" {
                continue;
            }
            let Some(data) = delta["data"].as_str().filter(|d| !d.is_empty()) else { continue };
            let Ok(bytes) = STANDARD.decode(data) else { continue };
            if !on_chunk(bytes) {
                return Ok(());
            }
        }
        Ok(())
    }

    pub fn say(self: &Arc<Self>, text: &str, stop: &Arc<AtomicBool>) -> Result<(), String> {
        if stop.load(Ordering::Relaxed) {
            return Ok(());
        }
        // Network reads run on their own thread because nothing can reliably wake a blocked
        // read. On stop this returns at once.
        // ponytail: an abandoned read lingers until its 30 s client timeout, then exits.
        let (tx, rx) = channel::<Result<Vec<u8>, String>>();
        let (me, text, flag) = (self.clone(), text.to_string(), stop.clone());
        std::thread::spawn(move || {
            let result = me.stream(&text, |chunk| !flag.load(Ordering::Relaxed) && tx.send(Ok(chunk)).is_ok());
            if let Err(e) = result {
                let _ = tx.send(Err(e));
            }
        });
        let chunks = std::iter::from_fn(|| loop {
            if stop.load(Ordering::Relaxed) {
                return None;
            }
            match rx.recv_timeout(Duration::from_millis(50)) {
                Ok(item) => return Some(item),
                Err(RecvTimeoutError::Timeout) => continue,
                Err(RecvTimeoutError::Disconnected) => return None,
            }
        });
        play(chunks, stop)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn speakable_strips_markdown() {
        let cases = [
            ("**Done** with `git status`.", "Done with git status."),
            ("# Title\n\nBody text.", "Title\n\nBody text."),
            ("See [the logs](http://x.test/a) now.", "See the logs now."),
            ("- one\n- two", "one\ntwo"),
            ("Before\n```py\nprint(1)\n```\nAfter", "Before\n\nAfter"),
            ("| a | b |\n|---|---|\n| 1 | 2 |\nAfter", "After"),
            ("```\ncode only\n```", ""),
            ("_note_ my_var", "note my_var"),
        ];
        for (text, expected) in cases {
            assert_eq!(speakable(text), expected, "{text:?}");
        }
    }

    #[test]
    fn speakable_caps_long_text_at_a_sentence_end() {
        let text = vec!["This is one sentence."; 400].join(" ");
        let result = speakable(&text);
        assert!(result.chars().count() <= 1300);
        assert!(result.strip_suffix(SPEAK_NOTE).unwrap().ends_with('.'));
    }

    #[test]
    fn wav_header_is_44_bytes() {
        let decoded = STANDARD.decode(wav_base64(&[0, 1])).unwrap();
        assert_eq!(decoded.len(), 48);
        assert_eq!(&decoded[..4], b"RIFF");
    }
}
