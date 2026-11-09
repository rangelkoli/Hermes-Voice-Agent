//! Microphone capture, speech detection, utterance segmentation, and playback.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{sync_channel, Receiver, SyncSender};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{SampleFormat, SizedSample, Stream};
use ort::session::Session;
use ort::value::Tensor;

pub const SAMPLE_RATE: u32 = 16000;
/// Every frame is 30 ms: 480 int16 samples.
pub const FRAME_SAMPLES: usize = 480;
const FRAME_MS: u32 = 30;
const SPEECH_RATE: u32 = 24000;
// The same Silero v6 file faster-whisper ships, built in.
const SILERO: &[u8] = include_bytes!("../assets/silero_vad_v6.onnx");

pub const MIC_SILENT_HINT: &str = if cfg!(target_os = "macos") {
    "No audio is arriving. Allow this terminal app in System Settings > Privacy & Security > Microphone."
} else {
    "No audio is arriving. Check the input is not muted in sound settings, or choose another with --device."
};

/// RMS of int16 samples, scaled so full scale is 1.0.
pub fn level(frame: &[i16]) -> f32 {
    if frame.is_empty() {
        return 0.0;
    }
    let total: f64 = frame.iter().map(|&s| (s as f64) * (s as f64)).sum();
    ((total / frame.len() as f64).sqrt() / 32768.0).min(1.0) as f32
}

pub trait Vad: Send {
    fn is_speech(&mut self, frame: &[i16]) -> bool;
    fn reset(&mut self) {}
}

/// True when a frame is speech. Silero v6, the same ONNX file faster-whisper ships.
pub struct SpeechDetector {
    session: Session,
    threshold: f32,
    h: Vec<f32>,
    c: Vec<f32>,
    context: Vec<f32>,
    pending: Vec<f32>,
    speech: bool,
}

impl SpeechDetector {
    pub fn new(threshold: f32) -> Result<Self, String> {
        let session = Session::builder()
            .and_then(|b| b.with_intra_threads(1))
            .and_then(|b| b.with_inter_threads(1))
            .and_then(|b| b.commit_from_memory(SILERO))
            .map_err(|e| format!("Could not load speech detection: {e}"))?;
        let mut detector = Self {
            session,
            threshold,
            h: vec![],
            c: vec![],
            context: vec![],
            pending: vec![],
            speech: false,
        };
        detector.reset();
        Ok(detector)
    }

    fn run(&mut self, chunk: &[f32]) -> Result<f32, ort::Error> {
        // ponytail: 16 kHz Silero v6 only. State is (1, 1, 128) and the context is 64 samples.
        let mut input = self.context.clone();
        input.extend_from_slice(chunk);
        let outputs = self.session.run(ort::inputs![
            "input" => Tensor::from_array(([1usize, input.len()], input))?,
            "h" => Tensor::from_array(([1usize, 1, 128], self.h.clone()))?,
            "c" => Tensor::from_array(([1usize, 1, 128], self.c.clone()))?,
        ])?;
        let probability = outputs[0].try_extract_tensor::<f32>()?.1[0];
        self.h = outputs[1].try_extract_tensor::<f32>()?.1.to_vec();
        self.c = outputs[2].try_extract_tensor::<f32>()?.1.to_vec();
        self.context = chunk[chunk.len() - 64..].to_vec();
        Ok(probability)
    }
}

impl Vad for SpeechDetector {
    fn is_speech(&mut self, frame: &[i16]) -> bool {
        // Capture frames are 480 samples. Silero wants 512, so the latest decision stands until the buffer fills.
        self.pending.extend(frame.iter().map(|&s| s as f32 / 32768.0));
        while self.pending.len() >= 512 {
            let chunk: Vec<f32> = self.pending.drain(..512).collect();
            if let Ok(p) = self.run(&chunk) {
                self.speech = p >= self.threshold;
            }
        }
        self.speech
    }

    fn reset(&mut self) {
        self.h = vec![0.0; 128];
        self.c = vec![0.0; 128];
        self.context = vec![0.0; 64];
        self.pending.clear();
        self.speech = false;
    }
}

pub struct Segmenter {
    vad: Box<dyn Vad>,
    end_silence_ms: u32,
    min_speech_ms: u32,
    max_ms: u32,
    preroll_frames: usize,
    preroll: VecDeque<Vec<i16>>,
    frames: Vec<Vec<i16>>,
    speech_frames: u32,
    silence_frames: u32,
    pub in_speech: bool,
}

impl Segmenter {
    pub fn new(vad: Box<dyn Vad>, end_silence_ms: u32) -> Self {
        Self {
            vad,
            end_silence_ms,
            min_speech_ms: 300,
            max_ms: 30000,
            preroll_frames: (300 / FRAME_MS) as usize,
            preroll: VecDeque::new(),
            frames: vec![],
            speech_frames: 0,
            silence_frames: 0,
            in_speech: false,
        }
    }

    pub fn reset(&mut self) {
        self.preroll.clear();
        self.frames.clear();
        self.speech_frames = 0;
        self.silence_frames = 0;
        self.in_speech = false;
        self.vad.reset();
    }

    pub fn push(&mut self, frame: Vec<i16>) -> Option<Vec<i16>> {
        let speech = self.vad.is_speech(&frame);
        if !self.in_speech {
            if !speech {
                if self.preroll.len() == self.preroll_frames {
                    self.preroll.pop_front();
                }
                self.preroll.push_back(frame);
                return None;
            }
            self.in_speech = true;
            self.frames = self.preroll.drain(..).collect();
            self.frames.push(frame);
            self.speech_frames = 1;
            self.silence_frames = 0;
        } else {
            self.frames.push(frame);
            if speech {
                self.speech_frames += 1;
                self.silence_frames = 0;
            } else {
                self.silence_frames += 1;
            }
        }
        if self.silence_frames * FRAME_MS >= self.end_silence_ms {
            let keep = self.speech_frames * FRAME_MS >= self.min_speech_ms;
            return self.finish(keep);
        }
        if self.frames.len() as u32 * FRAME_MS >= self.max_ms {
            return self.finish(true);
        }
        None
    }

    fn finish(&mut self, keep: bool) -> Option<Vec<i16>> {
        let frames = std::mem::take(&mut self.frames);
        self.speech_frames = 0;
        self.silence_frames = 0;
        self.in_speech = false;
        keep.then(|| frames.concat())
    }
}

#[derive(Clone, Debug)]
pub enum DeviceSpec {
    Index(usize),
    Name(String),
}

impl DeviceSpec {
    pub fn parse(value: &str) -> Self {
        value.parse().map(Self::Index).unwrap_or_else(|_| Self::Name(value.to_string()))
    }
}

const NO_MIC: &str = "A local microphone is required. SSH sessions and headless machines have none.";

fn input_device(spec: Option<&DeviceSpec>) -> Result<cpal::Device, String> {
    let host = cpal::default_host();
    let found = match spec {
        None => host.default_input_device(),
        Some(DeviceSpec::Index(i)) => host.input_devices().ok().and_then(|mut d| d.nth(*i)),
        Some(DeviceSpec::Name(name)) => {
            let wanted = name.to_lowercase();
            host.input_devices().ok().and_then(|mut devices| {
                devices.find(|d| d.name().is_ok_and(|n| n.to_lowercase().contains(&wanted)))
            })
        }
    };
    found.ok_or_else(|| format!("{NO_MIC} Run with --list-devices."))
}

pub fn list_devices() -> Result<String, String> {
    let devices = cpal::default_host()
        .input_devices()
        .map_err(|e| format!("{NO_MIC} ({e})"))?;
    let lines: Vec<String> = devices
        .enumerate()
        .map(|(i, d)| format!("{i}: {}", d.name().unwrap_or_default()))
        .collect();
    if lines.is_empty() {
        return Err(NO_MIC.to_string());
    }
    Ok(lines.join("\n"))
}

/// Box-average resampler to 16 kHz, so a 48 kHz mic averages every 3 samples.
// ponytail: no proper low-pass. Fine for speech from rates at or above 16 kHz.
struct Downsampler {
    ratio: f64,
    position: f64,
    sum: f32,
    count: u32,
}

impl Downsampler {
    fn push(&mut self, sample: f32, out: &mut Vec<i16>) {
        self.sum += sample;
        self.count += 1;
        self.position += 1.0;
        while self.position >= self.ratio {
            let mean = self.sum / self.count.max(1) as f32;
            out.push((mean.clamp(-1.0, 1.0) * 32767.0) as i16);
            self.position -= self.ratio;
            self.sum = 0.0;
            self.count = 0;
        }
    }
}

/// The microphone stream. cpal streams are not Send on macOS, so one thread owns this.
pub struct Mic {
    spec: Option<DeviceSpec>,
    stream: Option<Stream>,
    dead: Arc<AtomicBool>,
    frames: Option<Receiver<Vec<i16>>>,
    pub name: String,
}

impl Mic {
    pub fn new(spec: Option<DeviceSpec>) -> Result<Self, String> {
        let name = input_device(spec.as_ref())?.name().unwrap_or_default();
        Ok(Self { spec, stream: None, dead: Arc::new(AtomicBool::new(false)), frames: None, name })
    }

    pub fn start(&mut self) -> Result<(), String> {
        self.stop();
        let fail = |e: String| format!("Could not open the microphone: {e}. Run with --list-devices.");
        let device = input_device(self.spec.as_ref()).map_err(fail)?;
        let config = device.default_input_config().map_err(|e| fail(e.to_string()))?;
        let (tx, rx) = sync_channel(1000);
        let dead = Arc::new(AtomicBool::new(false));
        let stream = match config.sample_format() {
            SampleFormat::F32 => build_input::<f32>(&device, &config.config(), tx, dead.clone()),
            SampleFormat::I16 => build_input::<i16>(&device, &config.config(), tx, dead.clone()),
            SampleFormat::I32 => build_input::<i32>(&device, &config.config(), tx, dead.clone()),
            SampleFormat::U16 => build_input::<u16>(&device, &config.config(), tx, dead.clone()),
            other => Err(format!("unsupported sample format {other}")),
        }
        .map_err(fail)?;
        stream.play().map_err(|e| fail(e.to_string()))?;
        self.name = device.name().unwrap_or_default();
        self.stream = Some(stream);
        self.dead = dead;
        self.frames = Some(rx);
        Ok(())
    }

    pub fn stop(&mut self) {
        self.stream = None;
        self.frames = None;
    }

    pub fn alive(&self) -> bool {
        self.stream.is_some() && !self.dead.load(Ordering::Relaxed)
    }

    pub fn read(&self, timeout: Duration) -> Option<Vec<i16>> {
        self.frames.as_ref()?.recv_timeout(timeout).ok()
    }
}

fn build_input<T>(
    device: &cpal::Device,
    config: &cpal::StreamConfig,
    tx: SyncSender<Vec<i16>>,
    dead: Arc<AtomicBool>,
) -> Result<Stream, String>
where
    T: SizedSample,
    f32: cpal::FromSample<T>,
{
    let channels = config.channels as usize;
    let mut down = Downsampler {
        ratio: config.sample_rate.0 as f64 / SAMPLE_RATE as f64,
        position: 0.0,
        sum: 0.0,
        count: 0,
    };
    let mut out = Vec::with_capacity(FRAME_SAMPLES * 2);
    device
        .build_input_stream(
            config,
            move |data: &[T], _: &_| {
                for frame in data.chunks(channels) {
                    let mono = frame.iter().map(|&s| f32::from_sample_(s)).sum::<f32>() / channels as f32;
                    down.push(mono, &mut out);
                }
                while out.len() >= FRAME_SAMPLES {
                    // A full queue drops the frame, like the Python client.
                    let _ = tx.try_send(out.drain(..FRAME_SAMPLES).collect());
                }
            },
            move |_err| dead.store(true, Ordering::Relaxed),
            None,
        )
        .map_err(|e| e.to_string())
}

use cpal::FromSample;

/// Play 24 kHz mono int16 chunks until they run out or `stop` is set.
pub fn play(
    chunks: impl Iterator<Item = Result<Vec<u8>, String>>,
    stop: &AtomicBool,
) -> Result<(), String> {
    let device = cpal::default_host().default_output_device().ok_or("No audio output device.")?;
    let supported = device.default_output_config().map_err(|e| e.to_string())?;
    let config = supported.config();
    let queue: Arc<Mutex<VecDeque<i16>>> = Arc::default();
    let stream = match supported.sample_format() {
        SampleFormat::F32 => build_output::<f32>(&device, &config, queue.clone()),
        SampleFormat::I16 => build_output::<i16>(&device, &config, queue.clone()),
        other => Err(format!("unsupported output format {other}")),
    }?;
    stream.play().map_err(|e| e.to_string())?;
    let mut odd: Option<u8> = None;
    for chunk in chunks {
        if stop.load(Ordering::Relaxed) {
            return Ok(());
        }
        let mut bytes = chunk?;
        if let Some(b) = odd.take() {
            bytes.insert(0, b);
        }
        if bytes.len() % 2 == 1 {
            odd = bytes.pop();
        }
        // Hold at most 2 s ahead of the speaker, like a blocking write.
        while queue.lock().unwrap().len() > SPEECH_RATE as usize * 2 {
            if stop.load(Ordering::Relaxed) {
                return Ok(());
            }
            std::thread::sleep(Duration::from_millis(20));
        }
        queue
            .lock()
            .unwrap()
            .extend(bytes.chunks_exact(2).map(|b| i16::from_le_bytes([b[0], b[1]])));
    }
    while !queue.lock().unwrap().is_empty() {
        if stop.load(Ordering::Relaxed) {
            return Ok(());
        }
        std::thread::sleep(Duration::from_millis(20));
    }
    // Let the device play out what it already pulled.
    std::thread::sleep(Duration::from_millis(150));
    Ok(())
}

fn build_output<T>(
    device: &cpal::Device,
    config: &cpal::StreamConfig,
    queue: Arc<Mutex<VecDeque<i16>>>,
) -> Result<Stream, String>
where
    T: SizedSample + FromSample<f32>,
{
    let channels = config.channels as usize;
    // Linear interpolation from 24 kHz to the device rate.
    let step = SPEECH_RATE as f32 / config.sample_rate.0 as f32;
    let (mut t, mut prev, mut cur) = (1.0f32, 0.0f32, 0.0f32);
    device
        .build_output_stream(
            config,
            move |data: &mut [T], _: &_| {
                let mut queue = queue.lock().unwrap();
                for frame in data.chunks_mut(channels) {
                    while t >= 1.0 {
                        prev = cur;
                        cur = queue.pop_front().map_or(0.0, |s| s as f32 / 32768.0);
                        t -= 1.0;
                    }
                    let sample = T::from_sample_(prev + (cur - prev) * t);
                    frame.fill(sample);
                    t += step;
                }
            },
            |_err| {},
            None,
        )
        .map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Loud;
    impl Vad for Loud {
        fn is_speech(&mut self, frame: &[i16]) -> bool {
            frame[0] != 0
        }
    }

    #[test]
    fn segmenter_keeps_speech_with_preroll_and_drops_blips() {
        let mut seg = Segmenter::new(Box::new(Loud), 300);
        let quiet = vec![0i16; FRAME_SAMPLES];
        let loud = vec![1i16; FRAME_SAMPLES];
        for _ in 0..20 {
            assert!(seg.push(quiet.clone()).is_none());
        }
        for _ in 0..10 {
            assert!(seg.push(loud.clone()).is_none());
        }
        assert!(seg.in_speech);
        let mut out = None;
        for _ in 0..10 {
            out = out.or(seg.push(quiet.clone()));
        }
        // 10 preroll + 10 speech + 10 silence frames.
        assert_eq!(out.unwrap().len(), 30 * FRAME_SAMPLES);
        // Under 300 ms of speech is dropped.
        seg.push(loud.clone());
        let blip = (0..10).filter_map(|_| seg.push(quiet.clone())).next();
        assert!(blip.is_none());
        assert!(!seg.in_speech);
    }

    #[test]
    fn downsampler_averages_48k_to_16k() {
        let mut d = Downsampler { ratio: 3.0, position: 0.0, sum: 0.0, count: 0 };
        let mut out = vec![];
        for s in [0.0, 0.3, 0.6, 0.5, 0.5, 0.5] {
            d.push(s, &mut out);
        }
        assert_eq!(out.len(), 2);
        assert!((out[0] as f32 / 32767.0 - 0.3).abs() < 1e-3);
    }

    #[test]
    fn level_of_full_scale_is_one() {
        assert_eq!(level(&[]), 0.0);
        assert!((level(&[i16::MIN; 4]) - 1.0).abs() < 1e-6);
    }
}
