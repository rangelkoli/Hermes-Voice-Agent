//! Prints the wake score for every 80 ms of a 16 kHz mono int16 file, fed in 30 ms frames like
//! the microphone. Usage: wake_scores <model.onnx> <file.pcm>...
use hermes_voice::audio::FRAME_SAMPLES;
use hermes_voice::wakeword::WakeWord;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut detector = WakeWord::load(args[1].as_ref()).unwrap();
    for file in &args[2..] {
        detector.reset();
        let bytes = std::fs::read(file).unwrap();
        let pcm: Vec<i16> = bytes.chunks_exact(2).map(|b| i16::from_le_bytes([b[0], b[1]])).collect();
        let start = std::time::Instant::now();
        // (seconds into the file, score)
        let scores: Vec<(f32, f32)> = pcm
            .chunks(FRAME_SAMPLES)
            .enumerate()
            .filter_map(|(i, frame)| detector.feed(frame).map(|s| ((i + 1) as f32 * 0.03, s)))
            .collect();
        let took = start.elapsed().as_secs_f32();
        let (at, max) = scores.iter().fold((0.0, 0.0f32), |best, &(t, s)| if s > best.1 { (t, s) } else { best });
        let audio = pcm.len() as f32 / 16000.0;
        println!("{file} max={max:.4} at={at:.2}s cpu={:.1}% of real time", 100.0 * took / audio);
        if std::env::var_os("ALL").is_some() {
            println!("{}", scores.iter().map(|(_, s)| format!("{s:.4}")).collect::<Vec<_>>().join(" "));
        }
    }
}
