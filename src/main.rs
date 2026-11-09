//! Command line for the Hermes voice client.

use std::process::exit;
use std::time::Duration;

use hermes_voice::app::{self, Config};
use hermes_voice::audio::{list_devices, DeviceSpec, Mic, Segmenter, SpeechDetector};
use hermes_voice::gemini::{load_gemini_key, GeminiSpeaker, GeminiTranscriber};
use hermes_voice::hermes::{load_api_key, HermesClient};
use hermes_voice::stt::{Stt, Transcriber};

const BASE_URL: &str = "http://127.0.0.1:8642";
const USAGE: &str = "usage: hermes-voice [--wake PHRASE] [--open-mic] [--wake-model M] [--show-heard]
                    [--device D] [--list-devices] [--local] [--stt-model M]
                    [--tts-model M] [--voice V] [--vad-threshold F] [--end-silence-ms N]";

struct Args {
    wake: String,
    open_mic: bool,
    wake_model: String,
    show_heard: bool,
    device: Option<DeviceSpec>,
    list_devices: bool,
    local: bool,
    stt_model: Option<String>,
    tts_model: String,
    voice: String,
    vad_threshold: f32,
    end_silence_ms: u32,
}

fn fail(message: impl std::fmt::Display) -> ! {
    eprintln!("{message}");
    exit(1)
}

fn parse_args() -> Args {
    let mut args = Args {
        wake: "hey hermes".into(),
        open_mic: false,
        wake_model: "base.en".into(),
        show_heard: false,
        device: None,
        list_devices: false,
        local: false,
        stt_model: None,
        tts_model: "gemini-3.8-flash-lite-tts".into(),
        voice: "Kore".into(),
        vad_threshold: 0.5,
        end_silence_ms: 1000,
    };
    let mut it = std::env::args().skip(1);
    while let Some(flag) = it.next() {
        let (flag, inline) = match flag.split_once('=') {
            Some((f, v)) => (f.to_string(), Some(v.to_string())),
            None => (flag, None),
        };
        let mut value = || inline.clone().or_else(|| it.next()).unwrap_or_else(|| fail(format!("{flag} needs a value\n{USAGE}")));
        match flag.as_str() {
            "--wake" => args.wake = value(),
            "--open-mic" => args.open_mic = true,
            "--wake-model" => args.wake_model = value(),
            "--show-heard" => args.show_heard = true,
            "--device" => args.device = Some(DeviceSpec::parse(&value())),
            "--list-devices" => args.list_devices = true,
            "--local" => args.local = true,
            "--stt-model" => args.stt_model = Some(value()),
            "--tts-model" => args.tts_model = value(),
            "--voice" => args.voice = value(),
            "--vad-threshold" => args.vad_threshold = value().parse().unwrap_or_else(|_| fail("--vad-threshold must be a number")),
            "--end-silence-ms" => {
                args.end_silence_ms = value()
                    .parse()
                    .ok()
                    .filter(|&n| n > 0)
                    .unwrap_or_else(|| fail("--end-silence-ms must be greater than 0"))
            }
            "-h" | "--help" => {
                println!("{USAGE}");
                exit(0)
            }
            other => fail(format!("unknown argument {other}\n{USAGE}")),
        }
    }
    args
}

/// Exit only if nothing is listening. A busy gateway is fine: the app retries sends.
fn check_hermes() {
    let url = format!("{BASE_URL}/health");
    let mut reason = String::new();
    for attempt in 0..3 {
        match ureq::get(&url).timeout(Duration::from_secs(3)).call() {
            Ok(_) | Err(ureq::Error::Status(..)) => return,
            Err(e) => reason = e.to_string(),
        }
        if attempt < 2 {
            std::thread::sleep(Duration::from_secs(1));
        }
    }
    fail(format!(
        "Nothing is answering at {url} ({reason}).\nStart Hermes with: hermes gateway\n\
         If it is already running, check API_SERVER_ENABLED=true and API_SERVER_PORT=8642 in its .env."
    ))
}

fn session_id() -> String {
    let tm = app::local_now();
    format!(
        "voice-{:04}{:02}{:02}-{:02}{:02}{:02}",
        tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday, tm.tm_hour, tm.tm_min, tm.tm_sec
    )
}

fn main() {
    let args = parse_args();
    if args.list_devices {
        match list_devices() {
            Ok(list) => println!("{list}"),
            Err(e) => fail(e),
        }
        return;
    }
    let device_name = Mic::new(args.device.clone()).unwrap_or_else(|e| fail(e)).name;
    let api_key = load_api_key().unwrap_or_else(|e| fail(e));
    check_hermes();

    let detector = SpeechDetector::new(args.vad_threshold).unwrap_or_else(|e| fail(e));
    let (transcriber, speaker): (Box<dyn Stt>, _) = if args.local {
        (Box::new(Transcriber::new(args.stt_model.as_deref().unwrap_or("small.en"), None)), None)
    } else {
        let key = load_gemini_key().unwrap_or_else(|e| fail(e));
        let model = args.stt_model.as_deref().unwrap_or("gemini-3.5-transcribe");
        (
            Box::new(GeminiTranscriber::new(&key, model)),
            Some(GeminiSpeaker::new(&key, &args.tts_model, &args.voice)),
        )
    };
    let (wake, wake_transcriber) = if args.open_mic {
        (None, None)
    } else {
        let hotword = args.wake.split_whitespace().last().map(|w| {
            let mut c = w.chars();
            c.next().map(|f| f.to_uppercase().chain(c.flat_map(char::to_lowercase)).collect()).unwrap_or_default()
        });
        let wake_stt: Box<dyn Stt> = Box::new(Transcriber::new(&args.wake_model, hotword));
        (Some(args.wake.clone()), Some(wake_stt))
    };
    let config = Config {
        client: HermesClient::new(&api_key, BASE_URL),
        session_id: session_id(),
        wake,
        device: args.device,
        device_name,
        transcriber: Some(transcriber),
        wake_transcriber,
        show_heard: args.show_heard,
        segmenter: Segmenter::new(Box::new(detector), args.end_silence_ms),
        speaker,
    };
    if let Err(e) = app::run(config) {
        fail(e);
    }
    // Worker threads may sit in blocking reads; leave without joining them.
    exit(0)
}
