//! Local speech-to-text (whisper.cpp) and filters for Whisper output.

use std::path::PathBuf;

use regex::RegexBuilder;
use unicode_normalization::{char::is_combining_mark, UnicodeNormalization};
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

/// Anything that turns 16 kHz mono int16 audio into text.
pub trait Stt: Send {
    fn load(&mut self) -> Result<(), String>;
    fn transcribe(&mut self, pcm: &[i16]) -> Result<String, String>;
}

// ponytail: known phrases only.
const NOISE: [&str; 4] = ["thank you", "thanks for watching", "you", "hermes"];
// ponytail: how Whisper mishears Hermes, such as "almies" and "her mess". Only trusted right
// after a greeting, and only for this word.
const HERMES_SOUNDALIKE: &str = r"h?[aeiou]{1,2}[lr]\s?m(?:e|ee|ea|ie|i|y)[sz]+";
const GREETING: &str = r"(?:hey|hay|hi|heh)";

/// Other spellings Whisper writes for a wake word. Unlike soundalikes, these are the same word,
/// so they count anywhere the word itself does.
// ponytail: add a word here as --show-heard reveals its spellings.
fn spellings(word: &str) -> Option<&'static str> {
    match word.to_lowercase().as_str() {
        "john" => Some("joh?n"),
        _ => None,
    }
}

fn strip_edges(text: &str) -> &str {
    text.trim_start_matches(|c: char| c.is_ascii_punctuation() || c.is_whitespace())
}

pub fn is_noise(text: &str) -> bool {
    let stripped = text.trim();
    if let (Some(first), Some(last)) = (stripped.chars().next(), stripped.chars().last()) {
        if stripped.chars().count() >= 2 && matches!((first, last), ('[', ']') | ('(', ')')) {
            return true;
        }
    }
    let lowered: String = stripped
        .to_lowercase()
        .chars()
        .filter(|c| !c.is_ascii_punctuation())
        .collect();
    let normalized = lowered.split_whitespace().collect::<Vec<_>>().join(" ");
    if !normalized.chars().any(char::is_alphabetic) {
        return true;
    }
    NOISE.contains(&normalized.as_str())
}

fn fold(text: &str) -> String {
    text.nfkd().filter(|c| !is_combining_mark(*c)).collect()
}

/// The text after the wake word, or None when it was not said. The key word is the phrase's
/// last word. An empty string means the phrase was the whole utterance.
pub fn wake_match(text: &str, phrase: &str) -> Option<String> {
    let folded_phrase = fold(phrase);
    let word = folded_phrase.split_whitespace().last()?;
    let folded = fold(text);
    // ponytail: the key word has to be one of the first four words, and a sentence break is
    // only . ! or ?. A Hermes later in the sentence is ignored.
    let key = spellings(word).map_or_else(|| regex::escape(word), str::to_string);
    let mut pattern = format!(r"(?:^|[.!?]\s+)(?:[\w'’]+[\s,]+){{0,3}}?{key}\b\W*");
    if word.eq_ignore_ascii_case("hermes") {
        pattern += &format!(r"|(?:^|[.!?]\s+){GREETING}\W+{HERMES_SOUNDALIKE}\b\W*");
    }
    let re = RegexBuilder::new(&pattern).case_insensitive(true).build().ok()?;
    let end = re.find_iter(&folded).last()?.end();
    Some(strip_edges(&folded[end..]).to_string())
}

/// Whisper segments with their no-speech probability, joined into one transcript.
pub fn join_segments<'a>(segments: impl IntoIterator<Item = (f32, &'a str)>) -> String {
    segments
        .into_iter()
        .filter(|(no_speech, _)| *no_speech <= 0.6)
        .map(|(_, text)| text)
        .collect::<String>()
        .trim()
        .to_string()
}

pub fn models_dir() -> PathBuf {
    std::env::var_os("HOME")
        .map(PathBuf::from)
        .unwrap_or_default()
        .join(".cache/hermes-voice-rs")
}

/// A local file, or a model name such as `base.en` that maps to an 8-bit ggml file.
// ponytail: q8_0 only, the closest match to faster-whisper's int8. Pass a path for others.
fn model_path(model: &str) -> Result<PathBuf, String> {
    let direct = PathBuf::from(model);
    if direct.is_file() {
        return Ok(direct);
    }
    let file = format!("ggml-{model}-q8_0.bin");
    let path = models_dir().join(&file);
    if path.is_file() {
        return Ok(path);
    }
    std::fs::create_dir_all(models_dir()).map_err(|e| e.to_string())?;
    let url = format!("https://huggingface.co/ggerganov/whisper.cpp/resolve/main/{file}");
    let response = ureq::get(&url)
        .call()
        .map_err(|e| format!("Could not download {file}: {e}"))?;
    let partial = path.with_extension("part");
    let mut out = std::fs::File::create(&partial).map_err(|e| e.to_string())?;
    std::io::copy(&mut response.into_reader(), &mut out)
        .map_err(|e| format!("Could not download {file}: {e}"))?;
    std::fs::rename(&partial, &path).map_err(|e| e.to_string())?;
    Ok(path)
}

pub struct Transcriber {
    pub model: String,
    hotword: Option<String>,
    ctx: Option<WhisperContext>,
}

impl Transcriber {
    pub fn new(model: &str, hotword: Option<String>) -> Self {
        Self { model: model.to_string(), hotword, ctx: None }
    }
}

impl Stt for Transcriber {
    fn load(&mut self) -> Result<(), String> {
        whisper_rs::install_logging_hooks();
        let path = model_path(&self.model)?;
        let mut params = WhisperContextParameters::default();
        // Flash attention skips the full attention matrices: about 70 to 100 MB less per model.
        params.flash_attn(true);
        let ctx = WhisperContext::new_with_params(path.to_str().ok_or("model path is not UTF-8")?, params)
        .map_err(|e| e.to_string())?;
        self.ctx = Some(ctx);
        Ok(())
    }

    fn transcribe(&mut self, pcm: &[i16]) -> Result<String, String> {
        // ponytail: a fresh state per call frees its compute buffers (about 70 to 160 MB) between
        // utterances, like CTranslate2. Costs a few ms per call. Keep one if latency matters more.
        let mut state = self.ctx.as_ref().ok_or("model not loaded")?.create_state().map_err(|e| e.to_string())?;
        let audio: Vec<f32> = pcm.iter().map(|&s| s as f32 / 32768.0).collect();
        let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
        let threads = std::thread::available_parallelism().map_or(4, |n| n.get().min(4));
        params.set_n_threads(threads as i32);
        params.set_language(Some("en"));
        params.set_no_context(true);
        params.set_print_progress(false);
        params.set_print_realtime(false);
        params.set_print_special(false);
        params.set_print_timestamps(false);
        // faster-whisper takes the hotword as a prompt prefix; whisper.cpp's prompt does the same job.
        params.set_initial_prompt(self.hotword.as_deref().unwrap_or("Hermes"));
        state.full(params, &audio).map_err(|e| e.to_string())?;
        let segments: Vec<(f32, String)> = state
            .as_iter()
            .map(|s| (s.no_speech_probability(), s.to_str_lossy().map(|t| t.into_owned()).unwrap_or_default()))
            .collect();
        Ok(join_segments(segments.iter().map(|(p, t)| (*p, t.as_str()))))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn noise() {
        for text in ["", "  ", "Thank you.", "thanks for watching!", "you", "...", "[BLANK_AUDIO]", "(music)", "Hermes."] {
            assert!(is_noise(text), "{text:?}");
        }
        for text in ["thank you for fixing the build", "stop the docker container"] {
            assert!(!is_noise(text), "{text:?}");
        }
    }

    #[test]
    fn wake_phrases() {
        let cases: &[(&str, Option<&str>)] = &[
            ("Hey Hermes, open the logs", Some("open the logs")),
            ("hey hermes", Some("")),
            ("Hey Hermes.", Some("")),
            ("hey, hermes open the logs", Some("open the logs")),
            ("Hermes open the logs", Some("open the logs")),
            ("Pay Hermes, what time is it?", Some("what time is it?")),
            ("Have a Hermes", Some("")),
            ("and the new Hermes, what time is it?", Some("what time is it?")),
            ("Hey Hermès", Some("")),
            ("Thanks. Hey Hermes, go", Some("go")),
            ("hey hermes stop. hey hermes go", Some("go")),
            ("Hey, almies!", Some("")),
            ("Hey almies, what time is it?", Some("what time is it?")),
            ("Hey her mess", Some("")),
            ("hey hermies open the logs", Some("open the logs")),
            ("Hi armies.", Some("")),
            ("Hey her message came through", None),
            ("Hey James, are you coming?", None),
            ("Hey Amy's here", None),
            ("Almies are great", None),
            ("Hey Herman, can you open the door?", None),
            ("We should talk about the budget", None),
            ("We bought a very expensive Hermes scarf", None),
            ("open the hermescope", None),
        ];
        for (text, expected) in cases {
            assert_eq!(wake_match(text, "hey hermes").as_deref(), *expected, "{text:?}");
        }
        assert_eq!(wake_match("Computer, lights on", "computer").as_deref(), Some("lights on"));
        assert_eq!(wake_match("Hey Jarvis lights on", "hey jarvis").as_deref(), Some("lights on"));
        assert_eq!(wake_match("Hey almies", "hey jarvis"), None);
        assert_eq!(wake_match("Hey almies", "Hey Hermes").as_deref(), Some(""));
    }

    #[test]
    fn john_also_wakes_on_jon() {
        let cases: &[(&str, Option<&str>)] = &[
            ("Hey John, lights on", Some("lights on")),
            ("Hey Jon, lights on", Some("lights on")),
            ("Jon open the logs", Some("open the logs")),
            ("hey jon", Some("")),
            ("Hey Joan, lights on", None),
            ("Jonathan is here", None),
            ("We talked about the budget with Jon", None),
        ];
        for (text, expected) in cases {
            assert_eq!(wake_match(text, "hey john").as_deref(), *expected, "{text:?}");
        }
        assert_eq!(wake_match("Hey Jon", "hey hermes"), None);
    }

    #[test]
    fn join_skips_non_speech() {
        assert_eq!(join_segments([(0.1, " open logs"), (0.9, " Thank you")]), "open logs");
    }
}
