//! On-device wake word: openWakeWord models run through ONNX Runtime.
//!
//! Three models run in a chain every 80 ms. A melspectrogram model turns audio into 32-band
//! frames. Google's speech embedding model turns the last 76 frames into a 96-number feature.
//! The wake model, trained for one phrase, scores the last 16 features from 0 to 1.

use std::collections::VecDeque;
use std::path::{Path, PathBuf};

use ort::session::Session;
use ort::value::Tensor;

use crate::stt::{download, models_dir};

/// Audio per step: 80 ms at 16 kHz.
const CHUNK: usize = 1280;
/// Earlier samples the melspectrogram needs to make 8 frames from one chunk.
const OVERLAP: usize = 480;
const MEL_BANDS: usize = 32;
const MEL_WINDOW: usize = 76;
const FEATURES: usize = 96;
const RELEASES: &str = "https://github.com/dscripka/openWakeWord/releases/download/v0.5.1";
/// Ready-made models from openWakeWord, for phrases that need no training.
// ponytail: these are licensed CC BY-NC-SA 4.0, fine for personal use.
const PRETRAINED: [(&str, &str); 3] = [
    ("hey jarvis", "hey_jarvis_v0.1.onnx"),
    ("alexa", "alexa_v0.1.onnx"),
    ("hey mycroft", "hey_mycroft_v0.1.onnx"),
];

/// Where a trained model for `phrase` lives: `~/.config/hermes-voice/hey_john.onnx`.
pub fn default_model_path(phrase: &str) -> PathBuf {
    let name = phrase.to_lowercase().split_whitespace().collect::<Vec<_>>().join("_");
    PathBuf::from(std::env::var_os("HOME").unwrap_or_default())
        .join(".config/hermes-voice")
        .join(format!("{name}.onnx"))
}

/// The wake model for `phrase`: `explicit` if given, else the trained file, else a ready-made
/// model downloaded on first use.
pub fn resolve_model(phrase: &str, explicit: Option<&str>) -> Result<PathBuf, String> {
    if let Some(path) = explicit {
        let path = PathBuf::from(path);
        return if path.is_file() { Ok(path) } else { Err(format!("No wake model at {}.", path.display())) };
    }
    let trained = default_model_path(phrase);
    if trained.is_file() {
        return Ok(trained);
    }
    let wanted = phrase.to_lowercase().split_whitespace().collect::<Vec<_>>().join(" ");
    match PRETRAINED.iter().find(|(p, _)| *p == wanted) {
        Some((_, file)) => fetch(file),
        None => Err(format!(
            "No wake model for \"{phrase}\" at {}. Train one with training/train_wake_word.ipynb \
             (see README), or pass --wake-model.",
            trained.display()
        )),
    }
}

fn fetch(file: &str) -> Result<PathBuf, String> {
    let path = models_dir().join(file);
    if !path.is_file() {
        download(&format!("{RELEASES}/{file}"), &path)?;
    }
    Ok(path)
}

fn session(path: &Path) -> Result<Session, String> {
    Session::builder()
        .and_then(|b| b.with_intra_threads(1))
        .and_then(|b| b.with_inter_threads(1))
        .and_then(|b| b.commit_from_file(path))
        .map_err(|e| format!("Could not load {}: {e}", path.display()))
}

pub struct WakeWord {
    mel: Session,
    embed: Session,
    model: Session,
    model_input: String,
    /// Features the wake model reads at once, 16 for openWakeWord models.
    window: usize,
    /// The last OVERLAP samples, then the chunk being filled.
    audio: Vec<f32>,
    mels: VecDeque<[f32; MEL_BANDS]>,
    features: VecDeque<Vec<f32>>,
}

impl WakeWord {
    pub fn load(model: &Path) -> Result<Self, String> {
        let mel = session(&fetch("melspectrogram.onnx")?)?;
        let embed = session(&fetch("embedding_model.onnx")?)?;
        let model = session(model)?;
        let input = model.inputs.first().ok_or("The wake model has no input.")?;
        // [batch, 16, 96]
        let window = input
            .input_type
            .tensor_shape()
            .and_then(|s| s.get(1).copied())
            .filter(|&n| n > 0)
            .map_or(16, |n| n as usize);
        let model_input = input.name.clone();
        let mut detector = Self { mel, embed, model, model_input, window, audio: vec![], mels: VecDeque::new(), features: VecDeque::new() };
        detector.reset();
        Ok(detector)
    }

    /// Forget past audio. Scores start again about 2 s later, once the buffers hold only new audio.
    pub fn reset(&mut self) {
        self.audio = vec![0.0; OVERLAP];
        self.mels.clear();
        self.features.clear();
    }

    /// Adds 16 kHz audio. Returns the newest score each time a full 80 ms has arrived.
    pub fn feed(&mut self, samples: &[i16]) -> Option<f32> {
        let mut score = None;
        for &sample in samples {
            // openWakeWord's melspectrogram takes int16 values as floats, not scaled to +/-1.
            self.audio.push(sample as f32);
            if self.audio.len() == OVERLAP + CHUNK {
                score = self.step().ok().flatten().or(score);
                self.audio.drain(..CHUNK);
            }
        }
        score
    }

    fn step(&mut self) -> Result<Option<f32>, ort::Error> {
        let outputs = self.mel.run(ort::inputs![
            "input" => Tensor::from_array(([1usize, self.audio.len()], self.audio.clone()))?,
        ])?;
        let (_, spec) = outputs[0].try_extract_tensor::<f32>()?;
        for frame in spec.chunks_exact(MEL_BANDS) {
            let mut row = [0.0; MEL_BANDS];
            // The same rescaling openWakeWord applies, to match Google's TensorFlow features.
            for (out, value) in row.iter_mut().zip(frame) {
                *out = value / 10.0 + 2.0;
            }
            self.mels.push_back(row);
        }
        drop(outputs);
        while self.mels.len() > MEL_WINDOW {
            self.mels.pop_front();
        }
        if self.mels.len() < MEL_WINDOW {
            return Ok(None);
        }
        let flat: Vec<f32> = self.mels.iter().flatten().copied().collect();
        let outputs = self.embed.run(ort::inputs![
            "input_1" => Tensor::from_array(([1usize, MEL_WINDOW, MEL_BANDS, 1], flat))?,
        ])?;
        self.features.push_back(outputs[0].try_extract_tensor::<f32>()?.1.to_vec());
        drop(outputs);
        while self.features.len() > self.window {
            self.features.pop_front();
        }
        if self.features.len() < self.window {
            return Ok(None);
        }
        let flat: Vec<f32> = self.features.iter().flatten().copied().collect();
        let outputs = self.model.run(ort::inputs![
            self.model_input.as_str() => Tensor::from_array(([1usize, self.window, FEATURES], flat))?,
        ])?;
        Ok(outputs[0].try_extract_tensor::<f32>()?.1.first().copied())
    }
}

/// Turns scores into wake events: one per time the score rises past the threshold.
#[derive(Default)]
pub struct Trigger {
    high: bool,
    peak: f32,
}

/// What one score means.
#[derive(Debug, PartialEq)]
pub struct Reading {
    pub fired: bool,
    /// A finished run of scores worth showing with --show-scores, as its highest value.
    pub peak: Option<f32>,
}

/// Scores below this are not shown by --show-scores.
const SHOW_FROM: f32 = 0.1;

impl Trigger {
    pub fn read(&mut self, score: f32, threshold: f32) -> Reading {
        let fired = score >= threshold && !self.high;
        self.high = score >= threshold;
        let mut peak = None;
        if score >= SHOW_FROM {
            self.peak = self.peak.max(score);
        } else if self.peak > 0.0 {
            peak = Some(std::mem::take(&mut self.peak));
        }
        Reading { fired, peak }
    }

    pub fn reset(&mut self) {
        *self = Self::default();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trigger_fires_once_per_rise() {
        let mut t = Trigger::default();
        let fired: Vec<bool> = [0.1, 0.6, 0.9, 0.7, 0.2, 0.8].iter().map(|&s| t.read(s, 0.5).fired).collect();
        assert_eq!(fired, [false, true, false, false, false, true]);
    }

    #[test]
    fn trigger_reports_each_run_peak_once() {
        let mut t = Trigger::default();
        let peaks: Vec<Option<f32>> = [0.05, 0.2, 0.4, 0.3, 0.0, 0.0].iter().map(|&s| t.read(s, 0.5).peak).collect();
        assert_eq!(peaks, [None, None, None, None, Some(0.4), None]);
    }

    #[test]
    fn model_path_comes_from_the_phrase() {
        let path = default_model_path("Hey  John");
        assert!(path.ends_with(".config/hermes-voice/hey_john.onnx"), "{}", path.display());
    }

    #[test]
    fn missing_custom_model_says_how_to_get_one() {
        let error = resolve_model("hey zebra", None).unwrap_err();
        assert!(error.contains("train_wake_word.ipynb"), "{error}");
        assert!(resolve_model("hey zebra", Some("/nonexistent/x.onnx")).is_err());
    }
}
