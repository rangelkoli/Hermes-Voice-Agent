//! Load VAD + Whisper models, transcribe a 16 kHz int16 file with each, then idle 8 s so the
//! caller can sample steady memory. Usage: bench <file.pcm> <model>...
use hermes_voice::audio::{SpeechDetector, Vad, FRAME_SAMPLES};
use hermes_voice::stt::{Stt, Transcriber};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let bytes = std::fs::read(&args[1]).unwrap();
    let pcm: Vec<i16> = bytes.chunks_exact(2).map(|b| i16::from_le_bytes([b[0], b[1]])).collect();
    let mut vad = SpeechDetector::new(0.5).unwrap();
    let speech = pcm.chunks_exact(FRAME_SAMPLES).filter(|f| vad.is_speech(f)).count();
    let mut models: Vec<Transcriber> = args[2..].iter().map(|m| Transcriber::new(m, Some("Hermes".into()))).collect();
    for m in &mut models {
        m.load().unwrap();
    }
    for m in &mut models {
        let start = std::time::Instant::now();
        let text = m.transcribe(&pcm).unwrap();
        println!("{} speech_frames={speech} text={text:?} secs={:.2}", m.model, start.elapsed().as_secs_f32());
    }
    println!("ready");
    std::thread::sleep(std::time::Duration::from_secs(8));
}
