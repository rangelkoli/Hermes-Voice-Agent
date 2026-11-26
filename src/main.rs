//! Command line for the Hermes voice client.

use std::process::exit;
use std::time::Duration;

use hermes_voice::app::{self, Config};
use hermes_voice::audio::{list_devices, DeviceSpec, Mic, Segmenter, SpeechDetector};
use hermes_voice::gemini::{load_gemini_key, GeminiSpeaker, GeminiTranscriber};
use hermes_voice::hermes::{load_api_key, HermesClient};
use hermes_voice::stt::{Stt, Transcriber};
use hermes_voice::todo::resolve_todo_path;
use hermes_voice::wakeword::{resolve_model, WakeWord};

const BASE_URL: &str = "http://127.0.0.1:8642";
const USAGE: &str = "usage: hermes-voice [--wake PHRASE] [--open-mic] [--wake-model FILE]
                    [--wake-threshold F] [--show-scores] [--device D] [--list-devices] [--local] [--stt-model M]
                    [--tts-model M] [--voice V] [--vad-threshold F] [--end-silence-ms N] [--todo-file FILE]
                    [--duck-volume N]";

struct Args {
    wake: String,
    open_mic: bool,
    wake_model: Option<String>,
    wake_threshold: f32,
    show_scores: bool,
    device: Option<DeviceSpec>,
    list_devices: bool,
    local: bool,
    stt_model: Option<String>,
    tts_model: String,
    voice: String,
    vad_threshold: f32,
    end_silence_ms: u32,
    todo_file: Option<String>,
    duck_volume: u8,
    help: bool,
}

fn fail(message: impl std::fmt::Display) -> ! {
    eprintln!("{message}");
    exit(1)
}

fn parse_args_from<I, S>(iter: I) -> Result<Args, String>
where
    I: IntoIterator<Item = S>,
    S: Into<String>,
{
    let mut args = Args {
        wake: "hey john".into(),
        open_mic: false,
        wake_model: None,
        wake_threshold: 0.5,
        show_scores: false,
        device: None,
        list_devices: false,
        local: false,
        stt_model: None,
        tts_model: "gemini-3.8-flash-lite-tts".into(),
        voice: "Kore".into(),
        vad_threshold: 0.5,
        end_silence_ms: 1000,
        todo_file: None,
        duck_volume: 10,
        help: false,
    };
    let mut it = iter.into_iter().map(Into::into);
    while let Some(raw) = it.next() {
        let (flag, inline) = match raw.split_once('=') {
            Some((f, v)) => (f.to_string(), Some(v.to_string())),
            None => (raw, None),
        };
        let mut value = || {
            inline
                .clone()
                .or_else(|| it.next())
                .ok_or_else(|| format!("{flag} needs a value\n{USAGE}"))
        };
        match flag.as_str() {
            "--wake" => args.wake = value()?,
            "--open-mic" => args.open_mic = true,
            "--wake-model" => args.wake_model = Some(value()?),
            "--wake-threshold" => {
                args.wake_threshold = value()?
                    .parse()
                    .ok()
                    .filter(|t| (0.0..=1.0).contains(t))
                    .ok_or_else(|| "--wake-threshold must be a number from 0 to 1".to_string())?;
            }
            "--show-scores" => args.show_scores = true,
            "--device" => args.device = Some(DeviceSpec::parse(&value()?)),
            "--list-devices" => args.list_devices = true,
            "--local" => args.local = true,
            "--stt-model" => args.stt_model = Some(value()?),
            "--tts-model" => args.tts_model = value()?,
            "--voice" => args.voice = value()?,
            "--vad-threshold" => {
                args.vad_threshold = value()?
                    .parse()
                    .map_err(|_| "--vad-threshold must be a number".to_string())?;
            }
            "--end-silence-ms" => {
                args.end_silence_ms = value()?
                    .parse()
                    .ok()
                    .filter(|&n| n > 0)
                    .ok_or_else(|| "--end-silence-ms must be greater than 0".to_string())?;
            }
            "--todo-file" => args.todo_file = Some(value()?),
            "--duck-volume" => {
                args.duck_volume = value()?
                    .parse()
                    .ok()
                    .filter(|&n| n <= 100)
                    .ok_or_else(|| "--duck-volume must be a whole number from 0 to 100".to_string())?;
            }
            "-h" | "--help" => {
                args.help = true;
                return Ok(args);
            }
            other => return Err(format!("unknown argument {other}\n{USAGE}")),
        }
    }
    Ok(args)
}

fn parse_args() -> Args {
    let args = parse_args_from(std::env::args().skip(1)).unwrap_or_else(|e| fail(e));
    if args.help {
        println!("{USAGE}");
        exit(0)
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
    let (wake, wake_word) = if args.open_mic {
        (None, None)
    } else {
        let model = resolve_model(&args.wake, args.wake_model.as_deref()).unwrap_or_else(|e| fail(e));
        let detector = WakeWord::load(&model).unwrap_or_else(|e| fail(e));
        (Some(args.wake.clone()), Some(detector))
    };
    let config = Config {
        client: HermesClient::new(&api_key, BASE_URL),
        wake,
        device: args.device,
        device_name,
        transcriber: Some(transcriber),
        wake_word,
        wake_threshold: args.wake_threshold,
        show_scores: args.show_scores,
        segmenter: Segmenter::new(Box::new(detector), args.end_silence_ms),
        speaker,
        todo_path: resolve_todo_path(args.todo_file.as_deref()),
        duck_volume: args.duck_volume,
    };
    if let Err(e) = app::run(config) {
        fail(e);
    }
    // Worker threads may sit in blocking reads; leave without joining them.
    exit(0)
}

#[cfg(test)]
mod tests {
    use super::parse_args_from;

    #[test]
    fn duck_volume_defaults_to_ten_and_rejects_out_of_range() {
        assert_eq!(parse_args_from(std::iter::empty::<&str>()).unwrap().duck_volume, 10);
        assert_eq!(parse_args_from(["--duck-volume", "0"]).unwrap().duck_volume, 0);
        assert_eq!(parse_args_from(["--duck-volume=25"]).unwrap().duck_volume, 25);
        for bad in ["101", "-1", "loud"] {
            let err = parse_args_from(["--duck-volume", bad]).err().unwrap();
            assert_eq!(err, "--duck-volume must be a whole number from 0 to 100");
        }
    }

    #[test]
    fn todo_file_defaults_to_none() {
        let args = parse_args_from(std::iter::empty::<&str>()).unwrap();
        assert_eq!(args.todo_file, None);
        assert!(!args.help);
    }

    #[test]
    fn todo_file_parses_separate_and_inline_values() {
        let separate = parse_args_from(["--todo-file", "/tmp/my.md"]).unwrap();
        assert_eq!(separate.todo_file.as_deref(), Some("/tmp/my.md"));

        let inline = parse_args_from(["--todo-file=/tmp/my.md"]).unwrap();
        assert_eq!(inline.todo_file.as_deref(), Some("/tmp/my.md"));
    }

    #[test]
    fn todo_file_requires_value() {
        let err = parse_args_from(["--todo-file"]).err().unwrap();
        assert!(err.contains("--todo-file needs a value"));
    }

    #[test]
    fn help_flag_sets_help() {
        assert!(parse_args_from(["-h"]).unwrap().help);
        assert!(parse_args_from(["--help"]).unwrap().help);
    }
}
