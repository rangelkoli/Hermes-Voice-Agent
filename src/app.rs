//! Terminal UI and worker threads for one Hermes voice conversation.
//!
//! One mutex holds all shared state. Capture, transcription, sending and speech each run on
//! their own thread; the UI thread redraws from that state 12 times a second.

use std::collections::{HashSet, VecDeque};
use std::io::Write;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::time::{Duration, Instant};

use crossterm::event::{self, Event, KeyCode, KeyEventKind, KeyModifiers};
use ratatui::layout::Rect;
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Paragraph};
use ratatui::Frame;
use serde_json::Value;

use crate::audio::{level, DeviceSpec, Mic, Segmenter, MIC_SILENT_HINT, SAMPLE_RATE};
use crate::gemini::{speakable, GeminiSpeaker};
use crate::hermes::{HermesClient, HermesError, RunEvent, TERMINAL};
use crate::music::{self, Ducker};
use crate::orb::{Orb, FPS};
use crate::stt::{is_noise, wake_match, Stt};
use crate::todo::{ensure_todo_file, hermes_instructions, load_todos, TodoItem, TodoWatcher};
use crate::wake::WakeGate;
use crate::wakeword::{Trigger, WakeWord};

// ponytail: a dead mic is reopened after a fixed 2 seconds. Quit and mute return early.
const MIC_RETRY: Duration = Duration::from_secs(2);
// ponytail: a failed model download, or a failed send, is retried after a fixed 5 seconds.
const RETRY: Duration = Duration::from_secs(5);
const AUDIO_WAIT: Duration = Duration::from_millis(100);
// ponytail: poll the backing todo file every 500 ms so external edits appear live on screen.
const TODO_POLL: Duration = Duration::from_millis(500);
/// How often the music volume is matched to what the assistant is doing.
const DUCK_TICK: Duration = Duration::from_millis(250);
const SILENT_FRAMES: u32 = 100;
// ponytail: speech waits at most 30 seconds for the user to finish.
const SPEAK_WAIT: Duration = Duration::from_secs(30);
// ponytail: the mic drops audio for 300 ms after speech so it is not transcribed.
const SPEAK_TAIL: Duration = Duration::from_millis(300);
// ponytail: after a spoken reply the mic stays open 2 seconds for a follow-up, no wake phrase.
const FOLLOW_UP: Duration = Duration::from_secs(2);
// ponytail: after 60 seconds with nothing going on, the next request starts a new Hermes session.
// Add an --idle-reset flag if the Pi needs another length.
const IDLE_RESET: Duration = Duration::from_secs(60);
/// Audio kept from before the moment the wake word fired, enough for the phrase itself. Talk
/// further back in the same breath is not sent.
const WAKE_LEAD: usize = 3 * SAMPLE_RATE as usize / 2;
/// Less speech than this after the wake word fires is the end of the phrase, not a command.
// ponytail: from Piper voices with the "hey jarvis" model. The phrase alone left 60 to 120 ms;
// "stop" left 240 to 390 ms. Raise it if "hey john" alone gets sent; lower it if short
// commands are dropped.
const COMMAND_MIN_MS: u32 = 180;
const MAX_LOG: usize = 1000;
const MAX_WIDTH: u16 = 96;
const BODY_MAX_WIDTH: u16 = 128;
const MIN_TWO_COL_WIDTH: u16 = 72;
const BACKGROUND: [u8; 3] = [30, 30, 30];

enum Entry {
    Plain(String),
    You(String),
    HermesLabel,
    Tool(String),
}

#[derive(Default)]
struct Shared {
    muted: bool,
    mic_lost: bool,
    offline: bool,
    hearing: bool,
    transcribing: bool,
    loading: bool,
    speaking: bool,
    /// The wake phrase cut the reply being spoken. Mic frames count again from that moment.
    barged: bool,
    speech_waiting: bool,
    stop_sending: bool,
    retry_mic: bool,
    mute_gen: u64,
    run_id: Option<String>,
    approval: Option<Value>,
    reply: String,
    pending: Vec<String>,
    gate: Option<WakeGate>,
    segmenter: Option<Segmenter>,
    utterances: VecDeque<(Vec<i16>, u64)>,
    /// The utterance being heard had a frame while the gate was open.
    heard_awake: bool,
    /// Where the wake word fired in the utterance being heard: (samples, speech ms).
    wake_mark: Option<(usize, u32)>,
    trigger: Trigger,
    logged_mic_errors: HashSet<String>,
    log: VecDeque<Entry>,
    todos: Vec<TodoItem>,
    level: f32,
    speech: VecDeque<String>,
    bell: bool,
    zero_frames: u32,
    frame_count: u64,
    weather: Option<(String, String)>,
    /// The Hermes session the next run joins.
    session_id: String,
    /// When the current session last saw a wake, a request, a run, or speech. None until the
    /// session has its first request, so a fresh session is never reset again.
    last_activity: Option<Instant>,
}

impl Shared {
    fn status(&self) -> &'static str {
        if self.mic_lost {
            "mic lost"
        } else if self.muted {
            "muted"
        } else if self.approval.is_some() {
            "approval"
        } else if self.offline {
            "offline"
        } else if self.speaking {
            "speaking"
        } else if self.run_id.is_some() {
            "thinking"
        } else if self.loading {
            "loading"
        } else if self.transcribing {
            "transcribing"
        } else if self.hearing {
            "hearing"
        } else if self.gate.as_ref().is_some_and(|g| !g.awake) {
            "waiting"
        } else {
            "listening"
        }
    }

    fn asleep(&self) -> bool {
        self.gate.as_ref().is_some_and(|g| !g.awake)
    }

    /// Listening, working, or speaking, so music must stay quiet.
    fn engaged(&self) -> bool {
        self.gate.as_ref().is_some_and(|g| g.awake) || self.conversation_busy()
    }

    fn log_line(&mut self, text: impl Into<String>) {
        self.push_log(Entry::Plain(text.into()));
    }

    fn push_log(&mut self, entry: Entry) {
        if self.log.len() == MAX_LOG {
            self.log.pop_front();
        }
        self.log.push_back(entry);
    }

    fn log_once(&mut self, text: String) {
        if !text.is_empty() && self.logged_mic_errors.insert(text.clone()) {
            self.log_line(text);
        }
    }

    fn reset_segmenter(&mut self) {
        if let Some(seg) = &mut self.segmenter {
            seg.reset();
        }
        self.hearing = false;
        self.heard_awake = false;
        self.wake_mark = None;
        self.trigger.reset();
    }

    /// What to transcribe of a finished utterance. Talk heard only while asleep is dropped here,
    /// before it can reach the network. With the wake word in it, only the audio from just
    /// before the phrase is kept, and the phrase alone is not sent.
    fn finish_utterance(&mut self, utterance: Vec<i16>, speech_ms: u32) -> Option<Vec<i16>> {
        let heard_awake = std::mem::take(&mut self.heard_awake);
        let mark = self.wake_mark.take();
        if self.gate.is_none() {
            return Some(utterance);
        }
        if !heard_awake {
            return None;
        }
        match mark {
            None => Some(utterance),
            Some((_, at_ms)) if speech_ms.saturating_sub(at_ms) < COMMAND_MIN_MS => None,
            Some((at, _)) => {
                let start = at.saturating_sub(WAKE_LEAD).min(utterance.len());
                Some(utterance[start..].to_vec())
            }
        }
    }

    /// Drop the wake, partial speech, and all audio in flight.
    fn close_gate(&mut self) {
        let Some(gate) = &mut self.gate else { return };
        gate.sleep();
        self.mute_gen += 1;
        self.reset_segmenter();
        self.utterances.clear();
    }

    /// The wake phrase was heard over a spoken reply: cut it, drop queued replies, and listen.
    fn barge_in(&mut self, stop_speech: &AtomicBool) {
        stop_speech.store(true, Ordering::Relaxed);
        self.speech.clear();
        self.barged = true;
    }

    /// Playback is over. After a barge-in the wake window and the command already being heard
    /// stay. Otherwise the mic opens for a short follow-up.
    fn speech_ended(&mut self) {
        if !std::mem::take(&mut self.barged) {
            self.reset_segmenter();
            if !self.muted {
                if let Some(gate) = &mut self.gate {
                    gate.open_for(FOLLOW_UP);
                }
            }
        }
        self.speaking = false;
        self.speech_waiting = false;
    }

    /// A run, an approval, speech, or a request is in flight. The session must not change.
    fn conversation_busy(&self) -> bool {
        self.run_id.is_some()
            || self.approval.is_some()
            || self.speaking
            || self.speech_waiting
            || self.transcribing
            || !self.speech.is_empty()
            || !self.pending.is_empty()
    }

    /// Keeps a session that has had a request open for another `IDLE_RESET`.
    fn touch(&mut self, now: Instant) {
        if self.last_activity.is_some() {
            self.last_activity = Some(now);
        }
    }

    /// Moves to `new_id` once the session has been idle for `IDLE_RESET`, and clears the
    /// conversation on screen. Busy time counts as activity. True when it reset.
    fn expire_session(&mut self, now: Instant, new_id: impl FnOnce() -> String) -> bool {
        if self.conversation_busy() {
            self.touch(now);
            return false;
        }
        match self.last_activity {
            Some(last) if now.saturating_duration_since(last) >= IDLE_RESET => {}
            _ => return false,
        }
        self.session_id = new_id();
        self.last_activity = None;
        self.reply.clear();
        self.pending.clear();
        self.speech.clear();
        self.log.clear();
        self.log_line("New conversation");
        true
    }

    /// Room talk while asleep must not hold a reply. Once the gate is open, or when there is
    /// no gate, a result stays busy until it is delivered.
    fn gate_busy(&self) -> bool {
        if self.asleep() {
            return false;
        }
        self.segmenter.as_ref().is_some_and(|s| s.in_speech) || !self.utterances.is_empty() || self.transcribing
    }
}

pub struct Config {
    pub client: HermesClient,
    pub wake: Option<String>,
    pub device: Option<DeviceSpec>,
    pub device_name: String,
    pub transcriber: Option<Box<dyn Stt>>,
    pub wake_word: Option<WakeWord>,
    pub wake_threshold: f32,
    pub show_scores: bool,
    pub segmenter: Segmenter,
    pub speaker: Option<GeminiSpeaker>,
    pub todo_path: PathBuf,
    /// Percent of its volume the music keeps while the assistant is listening, working, or speaking.
    pub duck_volume: u8,
}

struct Core {
    state: Mutex<Shared>,
    changed: Condvar,
    quit: AtomicBool,
    stop_speech: Arc<AtomicBool>,
    client: HermesClient,
    wake: Option<String>,
    wake_threshold: f32,
    show_scores: bool,
    speaks: bool,
    todo_path: PathBuf,
    instructions: String,
    ducker: Mutex<Ducker>,
}

impl Core {
    fn lock(&self) -> MutexGuard<'_, Shared> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn quitting(&self) -> bool {
        self.quit.load(Ordering::Relaxed)
    }

    fn log(&self, text: impl Into<String>) {
        self.lock().log_line(text);
    }

    /// Matches the music volume to the conversation. `allow` false always restores it.
    /// The state lock is released before the player is spoken to.
    fn duck(&self, allow: bool) {
        let ducker = self.ducker.lock().unwrap_or_else(|e| e.into_inner());
        // Checked under the ducker lock, so no tick can duck again after the one made on quit.
        let engaged = allow && !self.quitting() && self.lock().engaged();
        ducker.tick(engaged);
    }

    fn wait<'a>(&self, guard: MutexGuard<'a, Shared>, timeout: Duration) -> MutexGuard<'a, Shared> {
        self.changed.wait_timeout(guard, timeout).unwrap_or_else(|e| e.into_inner()).0
    }

    /// Sleep for `duration`. False when quit, or `cancel`, cut it short.
    fn pause(&self, duration: Duration, cancel: impl Fn(&Shared) -> bool) -> bool {
        let deadline = Instant::now() + duration;
        let mut state = self.lock();
        loop {
            if self.quitting() || cancel(&state) {
                return false;
            }
            let now = Instant::now();
            if now >= deadline {
                return true;
            }
            state = self.wait(state, (deadline - now).min(AUDIO_WAIT));
        }
    }

    fn on_transcript(&self, s: &mut Shared, text: &str) {
        if s.stop_sending {
            return;
        }
        let spoken = match &mut s.gate {
            None if is_noise(text) => return,
            None => text.to_string(),
            Some(gate) => {
                if !gate.awake {
                    return;
                }
                let spoken = wake_match(text, self.wake.as_deref().unwrap_or("")).unwrap_or_else(|| text.to_string());
                if spoken.is_empty() || is_noise(&spoken) {
                    return;
                }
                // Queued audio stays. The local check gets the next wake.
                gate.sleep();
                s.reset_segmenter();
                spoken
            }
        };
        if !s.log.is_empty() {
            s.log_line("");
        }
        s.push_log(Entry::You(spoken.clone()));
        s.pending.push(spoken);
        s.last_activity = Some(Instant::now());
        self.changed.notify_all();
    }

    fn on_event(&self, s: &mut Shared, event: RunEvent) {
        let data = &event.data;
        match event.name.as_str() {
            "message.delta" => s.reply += &text(&data["delta"]),
            "tool.started" | "tool.completed" => {
                let mut parts = vec![text(&data["tool"])];
                match &data["duration"] {
                    Value::Null => parts.push(text(&data["preview"])),
                    duration => parts.push(format!("{duration}s")),
                }
                if truthy(&data["error"]) {
                    parts.push("failed".into());
                    parts.push(text(&data["preview"]));
                }
                parts.retain(|p| !p.is_empty());
                s.push_log(Entry::Tool(format!("⚙ {}", parts.join("  "))));
            }
            "approval.request" => {
                s.approval = Some(data.clone());
                s.log_line(format!("Approval  {}", text(&data["command"])));
                s.bell = true;
            }
            "run.completed" => {
                let output = text(&data["output"]);
                s.push_log(Entry::HermesLabel);
                // ponytail: Markdown is shown as plain text.
                for line in output.lines() {
                    s.log_line(line);
                }
                s.reply.clear();
                if self.speaks {
                    let spoken = speakable(&output);
                    if !spoken.is_empty() {
                        s.speech.push_back(spoken);
                        self.changed.notify_all();
                    }
                }
            }
            "run.failed" => {
                let error = data.get("error").map(text).unwrap_or_else(|| "The run failed.".into());
                s.log_line(error);
                s.reply.clear();
            }
            "run.cancelled" | "run.interrupted" => {
                if !s.reply.is_empty() {
                    let reply = std::mem::take(&mut s.reply);
                    s.log_line(reply);
                }
                s.log_line("The run ended early.");
            }
            _ => {}
        }
        let terminal = TERMINAL.contains(&event.name.as_str());
        if terminal || matches!(event.name.as_str(), "approval.responded" | "tool.completed") {
            s.approval = None;
        }
        if terminal {
            s.run_id = None;
        }
    }
}

fn text(value: &Value) -> String {
    match value {
        Value::String(s) => s.clone(),
        Value::Null => String::new(),
        other => other.to_string(),
    }
}

fn truthy(value: &Value) -> bool {
    match value {
        Value::Null | Value::Bool(false) => false,
        Value::String(s) => !s.is_empty(),
        _ => true,
    }
}

// ---------------------------------------------------------------- capture

fn capture_loop(core: &Core, device: Option<DeviceSpec>, mut wake_word: Option<WakeWord>) {
    let mut mic = match Mic::new(device) {
        Ok(mic) => mic,
        Err(e) => {
            let mut s = core.lock();
            s.mic_lost = true;
            s.log_line(e);
            return;
        }
    };
    // Whether the wake word model has heard the room since the last mute. It also runs during
    // replies, so "hey john" can cut one off.
    let mut listening = false;
    while !core.quitting() {
        if core.lock().muted {
            mic.stop();
            listening = false;
            core.pause(AUDIO_WAIT, |_| false);
            continue;
        }
        if !mic.alive() {
            recover_mic(core, &mut mic);
            continue;
        }
        if let Some(frame) = mic.read(AUDIO_WAIT) {
            // Runs outside the lock: a few ms every 80 ms on a Raspberry Pi.
            let score = wake_word.as_mut().and_then(|detector| {
                if !listening {
                    detector.reset();
                    listening = true;
                }
                detector.feed(&frame)
            });
            take_frame(core, frame, score);
        }
    }
}

fn recover_mic(core: &Core, mic: &mut Mic) {
    let retry = {
        let mut s = core.lock();
        if s.retry_mic {
            s.mic_lost = true;
            s.close_gate();
        }
        s.retry_mic
    };
    if retry && !core.pause(MIC_RETRY, |s| s.muted) {
        return;
    }
    let result = mic.start();
    let mut s = core.lock();
    if s.muted || core.quitting() {
        mic.stop();
        return;
    }
    s.retry_mic = true;
    match result {
        Ok(()) => s.mic_lost = false,
        Err(text) => {
            s.mic_lost = true;
            s.close_gate();
            s.log_once(text);
        }
    }
}

fn take_frame(core: &Core, frame: Vec<i16>, score: Option<f32>) {
    let mut guard = core.lock();
    let s = &mut *guard;
    if s.muted {
        return;
    }
    // Our own voice is not heard, except by the wake word model.
    // ponytail: no echo cancelling. If replies cut themselves off, raise --wake-threshold.
    if s.speaking && !s.barged {
        if score.is_some_and(|score| hear_wake(core, s, score)) {
            s.barge_in(&core.stop_speech);
            s.log_line("Cut off by the wake phrase.");
        }
        return;
    }
    s.frame_count += 1;
    if s.frame_count % 3 == 0 {
        s.level = level(&frame);
    }
    if frame.iter().all(|&x| x == 0) {
        s.zero_frames += 1;
        if s.zero_frames == SILENT_FRAMES {
            s.log_line(MIC_SILENT_HINT);
        }
    } else {
        s.zero_frames = 0;
    }
    if let Some(score) = score {
        hear_wake(core, s, score);
    }
    let awake = !s.asleep();
    let mut finished = None;
    if let Some(seg) = &mut s.segmenter {
        let speech_ms = seg.speech_ms();
        let utterance = seg.push(frame);
        if awake && (seg.in_speech || utterance.is_some()) {
            s.heard_awake = true;
        }
        finished = utterance.map(|u| (u, speech_ms));
    }
    if let Some((utterance, speech_ms)) = finished {
        if let Some(utterance) = s.finish_utterance(utterance, speech_ms) {
            s.utterances.push_back((utterance, s.mute_gen));
            core.changed.notify_all();
        }
        if let Some(gate) = &mut s.gate {
            gate.touch();
        }
    }
    let busy = s.gate_busy();
    if let Some(gate) = &mut s.gate {
        gate.expire(busy);
    }
    // Asleep stays "waiting" even while someone in the room is talking.
    s.hearing = s.segmenter.as_ref().is_some_and(|seg| seg.in_speech) && !s.asleep();
}

/// True when the wake phrase was just heard and the gate opened.
fn hear_wake(core: &Core, s: &mut Shared, score: f32) -> bool {
    let reading = s.trigger.read(score, core.wake_threshold);
    if let (true, Some(peak)) = (core.show_scores, reading.peak) {
        s.log_line(format!("wake score {peak:.2}"));
    }
    if !reading.fired {
        return false;
    }
    let Some(gate) = &mut s.gate else { return false };
    gate.wake();
    s.bell = true;
    s.touch(Instant::now());
    if let Some(seg) = s.segmenter.as_ref().filter(|seg| seg.in_speech) {
        s.wake_mark = Some((seg.samples(), seg.speech_ms()));
    }
    true
}

// ---------------------------------------------------------------- transcription

fn load_models(core: &Core, models: [&mut Option<Box<dyn Stt>>; 1]) -> bool {
    core.lock().loading = true;
    let mut logged: Option<String> = None;
    let mut ok = true;
    'models: for model in models.into_iter().flatten() {
        loop {
            if core.quitting() {
                ok = false;
                break 'models;
            }
            match model.load() {
                Ok(()) => break,
                Err(e) => {
                    if logged.as_ref() != Some(&e) {
                        core.log(e.clone());
                        logged = Some(e);
                    }
                    if !core.pause(RETRY, |_| false) {
                        ok = false;
                        break 'models;
                    }
                }
            }
        }
    }
    core.lock().loading = false;
    ok
}

fn stt_loop(core: &Core, mut transcriber: Option<Box<dyn Stt>>) {
    if !load_models(core, [&mut transcriber]) {
        return;
    }
    // Held aside so speech playback cannot drop a wake that arrived mid-transcription.
    let mut ahead: VecDeque<(Vec<i16>, u64)> = VecDeque::new();
    while !core.quitting() {
        let (utterance, generation) = match ahead.pop_front() {
            Some(item) => item,
            None => {
                let mut s = core.lock();
                match s.utterances.pop_front() {
                    Some(item) => item,
                    None => {
                        drop(core.wait(s, AUDIO_WAIT));
                        continue;
                    }
                }
            }
        };
        {
            let mut s = core.lock();
            // The gate can close after this utterance was queued, such as after a command.
            if generation != s.mute_gen || s.asleep() {
                continue;
            }
            s.transcribing = true;
        }
        let result = match transcriber.as_deref_mut() {
            Some(t) => t.transcribe(&utterance),
            None => Ok(String::new()),
        };
        let mut s = core.lock();
        s.transcribing = false;
        match result {
            Err(e) => s.log_line(e),
            Ok(text) if !core.quitting() && !is_noise(&text) => {
                let follow: Vec<_> = s.utterances.drain(..).collect();
                if generation == s.mute_gen {
                    core.on_transcript(&mut s, &text);
                }
                for item in follow.into_iter().rev() {
                    ahead.push_front(item);
                }
            }
            Ok(_) => {}
        }
    }
}

// ---------------------------------------------------------------- sending

fn auth_failed(core: &Core) {
    let mut s = core.lock();
    s.stop_sending = true;
    s.offline = true;
    s.log_line("Speech is not being sent until the API key is fixed and the app is restarted.");
}

fn send_loop(core: &Core) {
    loop {
        let (text, count, session_id) = {
            let mut s = core.lock();
            while s.pending.is_empty() || s.stop_sending {
                if core.quitting() {
                    return;
                }
                s = core.wait(s, AUDIO_WAIT);
            }
            (s.pending.join("\n"), s.pending.len(), s.session_id.clone())
        };
        let run_id = match core.client.start_run_with_instructions(
            &text,
            &session_id,
            Some(&core.instructions),
        ) {
            Ok(id) => id,
            Err(HermesError::Unavailable(_)) => {
                core.lock().offline = true;
                core.pause(RETRY, |_| false);
                continue;
            }
            Err(HermesError::Auth(_)) => return auth_failed(core),
            Err(e) => {
                core.log(format!("Send failed: {e}"));
                core.pause(RETRY, |_| false);
                continue;
            }
        };
        {
            let mut s = core.lock();
            s.offline = false;
            s.pending.drain(..count);
            s.run_id = Some(run_id.clone());
            s.approval = None;
            s.reply.clear();
        }
        let result = core.client.events(&run_id, |event| {
            let refresh_todos = matches!(event.name.as_str(), "tool.completed" | "run.completed");
            {
                let mut s = core.lock();
                core.on_event(&mut s, event);
            }
            if refresh_todos {
                if let Ok(items) = load_todos(&core.todo_path) {
                    core.lock().todos = items;
                }
            }
        });
        {
            let mut s = core.lock();
            s.run_id = None;
            s.approval = None;
        }
        match result {
            Ok(()) => {}
            Err(HermesError::Auth(_)) => return auth_failed(core),
            Err(e) => {
                core.log(format!("Send failed: {e}"));
                core.pause(RETRY, |_| false);
            }
        }
    }
}

// ---------------------------------------------------------------- speech

fn speak_loop(core: &Core, speaker: Arc<GeminiSpeaker>) {
    while !core.quitting() {
        let text = {
            let mut s = core.lock();
            match s.speech.pop_front() {
                Some(text) => text,
                None => {
                    drop(core.wait(s, AUDIO_WAIT));
                    continue;
                }
            }
        };
        if !claim_reply(core) {
            continue;
        }
        core.lock().level = 0.0;
        if let Err(e) = speaker.say(&text, &core.stop_speech) {
            core.log(e);
        }
        core.pause(SPEAK_TAIL, |s| s.barged);
        // Speech that starts within the follow-up window keeps the gate open until transcribed.
        core.lock().speech_ended();
    }
}

fn claim_reply(core: &Core) -> bool {
    let deadline = Instant::now() + SPEAK_WAIT;
    {
        let mut s = core.lock();
        if core.quitting() {
            return false;
        }
        core.stop_speech.store(false, Ordering::Relaxed);
        s.speech_waiting = true;
    }
    loop {
        // Wait for the user to finish, at most until the deadline.
        core.pause(deadline.saturating_duration_since(Instant::now()), |s| {
            core.stop_speech.load(Ordering::Relaxed) || !s.gate_busy()
        });
        let mut s = core.lock();
        if core.quitting() || core.stop_speech.load(Ordering::Relaxed) {
            s.speech_waiting = false;
            return false;
        }
        if s.gate_busy() && Instant::now() < deadline {
            continue;
        }
        s.speaking = true;
        s.speech_waiting = false;
        s.reset_segmenter();
        s.close_gate();
        return true;
    }
}

// ---------------------------------------------------------------- UI

fn status_style(status: &str) -> Style {
    let rgb = |r, g, b| Color::Rgb(r, g, b);
    let style = Style::default().add_modifier(Modifier::BOLD);
    match status {
        "loading" => style.fg(rgb(0x8a, 0x8a, 0x8a)),
        "waiting" => style.fg(rgb(0xd7, 0xba, 0x74)),
        "hearing" => style.fg(rgb(0x7d, 0xff, 0xb2)),
        "transcribing" => style.fg(rgb(0x6c, 0xb6, 0xff)),
        "thinking" => style.fg(rgb(0x5f, 0xd0, 0xe0)),
        "speaking" => style.fg(rgb(0xff, 0xb4, 0x54)),
        "muted" => style.fg(rgb(0x9a, 0xa0, 0xa6)),
        "approval" => style.bg(rgb(0xff, 0xbf, 0x00)).fg(rgb(0x1a, 0x1a, 0x1a)),
        "offline" => style.bg(rgb(0xff, 0x2d, 0x2d)).fg(Color::White),
        "mic lost" => style.bg(rgb(0xff, 0x4f, 0xd8)).fg(rgb(0x1a, 0x1a, 0x1a)),
        _ => style.fg(rgb(0x3d, 0xde, 0x7a)),
    }
}

/// Word wrap by characters.
// ponytail: one column per char. Wide CJK and emoji can overrun by a cell.
fn wrap(text: &str, width: usize) -> Vec<String> {
    let width = width.max(1);
    let mut lines = vec![];
    for paragraph in text.split('\n') {
        let mut line = String::new();
        for word in paragraph.split(' ') {
            let (line_len, word_len) = (line.chars().count(), word.chars().count());
            if line_len > 0 && line_len + 1 + word_len > width {
                lines.push(std::mem::take(&mut line));
            } else if line_len > 0 {
                line.push(' ');
            }
            let mut chars: Vec<char> = word.chars().collect();
            while chars.len() > width {
                lines.push(chars.drain(..width).collect());
            }
            line.extend(chars);
        }
        lines.push(line);
    }
    lines
}

/// The newest `max` wrapped log lines.
fn log_tail(log: &VecDeque<Entry>, width: usize, max: usize) -> Vec<Line<'static>> {
    let dim = Style::default().fg(Color::Rgb(0x80, 0x80, 0x80));
    let mut out: Vec<Line<'static>> = vec![];
    for entry in log.iter().rev() {
        if out.len() >= max {
            break;
        }
        let mut lines: Vec<Line<'static>> = match entry {
            Entry::Plain(t) => wrap(t, width).into_iter().map(Line::from).collect(),
            Entry::Tool(t) => wrap(t, width).into_iter().map(|l| Line::styled(l, dim)).collect(),
            Entry::HermesLabel => vec![Line::styled(
                "Hermes",
                Style::default().fg(Color::Rgb(0x5f, 0xd0, 0xe0)).add_modifier(Modifier::BOLD),
            )],
            Entry::You(t) => {
                let mut lines: Vec<Line<'static>> =
                    wrap(&format!("You  {t}"), width).into_iter().map(Line::from).collect();
                let rest = lines[0].spans[0].content.chars().skip(3).collect::<String>();
                lines[0] = Line::from(vec![
                    Span::styled("You", Style::default().fg(Color::Rgb(0x3d, 0xde, 0x7a)).add_modifier(Modifier::BOLD)),
                    Span::raw(rest),
                ]);
                lines
            }
        };
        lines.reverse();
        out.extend(lines);
    }
    out.truncate(max);
    out.reverse();
    out
}

fn body_columns(area: Rect, y: u16, height: u16) -> (Option<Rect>, Option<Rect>, Rect) {
    if area.width >= MIN_TWO_COL_WIDTH && height > 0 {
        let body_width = area.width.min(BODY_MAX_WIDTH);
        let body_x = area.x + (area.width - body_width) / 2;
        let todo_w = (body_width * 30 / 100).clamp(24, 38);
        let sep_w = 1;
        let chat_w = body_width.saturating_sub(todo_w + sep_w);
        (
            Some(Rect::new(body_x, y, todo_w, height)),
            Some(Rect::new(body_x + todo_w, y, sep_w, height)),
            Rect::new(body_x + todo_w + sep_w, y, chat_w, height),
        )
    } else {
        let chat_w = area.width.min(MAX_WIDTH);
        let chat_x = area.x + (area.width - chat_w) / 2;
        (None, None, Rect::new(chat_x, y, chat_w, height))
    }
}

fn todo_lines(todos: &[TodoItem], width: usize, max_lines: usize) -> Vec<Line<'static>> {
    if max_lines == 0 {
        return vec![];
    }
    let header_style = Style::default().fg(Color::Rgb(0xd7, 0xba, 0x74)).add_modifier(Modifier::BOLD);
    let dim_count = Style::default().fg(Color::Rgb(0x90, 0x90, 0x90));
    let header = if todos.is_empty() {
        Line::from(Span::styled("TODO", header_style))
    } else {
        let done = todos.iter().filter(|t| t.done).count();
        Line::from(vec![
            Span::styled("TODO", header_style),
            Span::styled(format!("  {done}/{}", todos.len()), dim_count),
        ])
    };
    let mut out = vec![header];
    if max_lines == 1 {
        return out;
    }
    if todos.is_empty() {
        out.push(Line::styled("No todos", Style::default().fg(Color::Rgb(0x70, 0x70, 0x70))));
        return out;
    }

    let text_width = width.saturating_sub(4).max(1);
    let open_box = Style::default().fg(Color::Rgb(0x5f, 0xd0, 0xe0)).add_modifier(Modifier::BOLD);
    let open_text = Style::default().fg(Color::Rgb(0xe0, 0xe0, 0xe0));
    let done_box = Style::default().fg(Color::Rgb(0x3d, 0xde, 0x7a));
    let done_text = Style::default().fg(Color::Rgb(0x78, 0x78, 0x78)).add_modifier(Modifier::CROSSED_OUT);

    let mut entries: Vec<(usize, Line<'static>)> = vec![];
    for (idx, item) in todos.iter().enumerate() {
        let (prefix, prefix_style, text_style) = if item.done {
            ("[x] ", done_box, done_text)
        } else {
            ("[ ] ", open_box, open_text)
        };
        for (i, line) in wrap(&item.text, text_width).into_iter().enumerate() {
            let rendered = if i == 0 {
                Line::from(vec![Span::styled(prefix, prefix_style), Span::styled(line, text_style)])
            } else {
                Line::from(vec![Span::raw("    "), Span::styled(line, text_style)])
            };
            entries.push((idx, rendered));
        }
    }

    let body_cap = max_lines - 1;
    if entries.len() <= body_cap {
        out.extend(entries.into_iter().map(|(_, line)| line));
    } else {
        let keep = body_cap - 1;
        let shown_items = entries.iter().take(keep).map(|(idx, _)| idx + 1).max().unwrap_or(0);
        let remaining = todos.len().saturating_sub(shown_items);
        out.extend(entries.into_iter().take(keep).map(|(_, line)| line));
        let label = if remaining > 0 { format!("+{remaining} more") } else { "…".into() };
        out.push(Line::styled(label, Style::default().fg(Color::Rgb(0x80, 0x80, 0x80))));
    }
    out
}

fn draw(frame: &mut Frame, s: &Shared, orb: &Orb, wake: Option<&str>, device: &str) {
    let area = frame.area();
    let bg = Color::Rgb(BACKGROUND[0], BACKGROUND[1], BACKGROUND[2]);
    frame.render_widget(Block::default().style(Style::default().bg(bg).fg(Color::Rgb(0xe0, 0xe0, 0xe0))), area);
    let width = area.width.min(MAX_WIDTH);
    let x = area.x + (area.width - width) / 2;
    let rows = area.height;
    let mut y = area.y;
    let mut room = rows;
    let mut take = |h: u16| {
        let h = h.min(room);
        let r = Rect::new(x, y, width, h);
        y += h;
        room -= h;
        r
    };

    let orb_area = take((rows * 35 / 100).clamp(7, 15));
    orb.render(orb_area, frame.buffer_mut(), BACKGROUND);
    draw_sides(frame, orb_area, s.weather.as_ref());

    let status = s.status();
    let label = if status == "waiting" {
        format!("WAITING FOR \"{}\"", wake.unwrap_or("").to_uppercase())
    } else {
        status.to_uppercase()
    };
    frame.render_widget(Paragraph::new(label).style(status_style(status)).centered(), take(1));
    frame.render_widget(
        Paragraph::new(device.to_string()).style(Style::default().fg(Color::Rgb(0x90, 0x90, 0x90))).centered(),
        take(1),
    );
    take(1);

    let (todo_area, divider_area, chat_area) = body_columns(area, y, room);
    if let Some(todo_area) = todo_area {
        let padded = pad(todo_area);
        let lines = todo_lines(&s.todos, padded.width as usize, padded.height as usize);
        frame.render_widget(Paragraph::new(lines), padded);
    }
    if let Some(div) = divider_area {
        let sep: Vec<Line> = (0..div.height).map(|_| Line::raw("│")).collect();
        frame.render_widget(
            Paragraph::new(sep).style(Style::default().fg(Color::Rgb(0x45, 0x45, 0x45))),
            div,
        );
    }

    let chat_inner = chat_area.width.saturating_sub(4) as usize;
    let stream_lines: Vec<Line> = if !s.reply.is_empty() && chat_area.height > 0 {
        // ponytail: shows the newest text only; the full reply lands in the log when done
        let tail: String = s.reply.chars().rev().take(400).collect::<Vec<_>>().into_iter().rev().collect();
        let lines = wrap(&format!("Hermes  {tail}"), chat_inner);
        let cap = 6.min(chat_area.height as usize);
        lines[lines.len().saturating_sub(cap)..].iter().cloned().map(Line::from).collect()
    } else {
        vec![]
    };
    let log_max = (chat_area.height as usize).saturating_sub(stream_lines.len());
    let log = log_tail(&s.log, chat_inner, log_max);
    let log_h = log.len() as u16;
    if log_h > 0 {
        let log_area = Rect::new(chat_area.x, chat_area.y, chat_area.width, log_h);
        frame.render_widget(Paragraph::new(log), pad(log_area));
    }
    if !stream_lines.is_empty() {
        let stream_h = stream_lines.len() as u16;
        let stream_area = Rect::new(chat_area.x, chat_area.y + log_h, chat_area.width, stream_h);
        frame.render_widget(
            Paragraph::new(stream_lines).style(Style::default().fg(Color::Rgb(0x5f, 0xd0, 0xe0))),
            pad(stream_area),
        );
    }

}

/// Clock to the left of the orb, weather to the right, each centred in its free column.
fn draw_sides(frame: &mut Frame, orb_area: Rect, weather: Option<&(String, String)>) {
    let side = orb_area.width.saturating_sub(orb_area.height * 2) / 2;
    if side < 10 || orb_area.height < 2 {
        return;
    }
    let dim = Style::default().fg(Color::Rgb(0x90, 0x90, 0x90));
    let bright = Style::default().fg(Color::Rgb(0xe0, 0xe0, 0xe0)).add_modifier(Modifier::BOLD);
    let mid = orb_area.y + orb_area.height / 2 - 1;
    let left = Rect::new(orb_area.x, mid, side, 2);
    let right = Rect::new(orb_area.right() - side, mid, side, 2);
    let now = local_now();
    let time = vec![Line::styled(clock(now.tm_hour, now.tm_min), bright), Line::styled(weekday(now.tm_wday), dim)];
    frame.render_widget(Paragraph::new(time).centered(), left);
    if let Some((temp, sky)) = weather {
        let lines = vec![Line::styled(temp.clone(), bright), Line::styled(sky.clone(), dim)];
        frame.render_widget(Paragraph::new(lines).centered(), right);
    }
}

/// A Hermes session id from the local time, such as `voice-20261119-093000`.
pub fn session_id() -> String {
    let tm = local_now();
    format!(
        "voice-{:04}{:02}{:02}-{:02}{:02}{:02}",
        tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday, tm.tm_hour, tm.tm_min, tm.tm_sec
    )
}

pub fn local_now() -> libc::tm {
    // SAFETY: localtime_r only writes the struct we pass.
    unsafe {
        let now = libc::time(std::ptr::null_mut());
        let mut tm: libc::tm = std::mem::zeroed();
        libc::localtime_r(&now, &mut tm);
        tm
    }
}

fn clock(hour: i32, min: i32) -> String {
    let h = if hour % 12 == 0 { 12 } else { hour % 12 };
    format!("{h}:{min:02} {}", if hour < 12 { "AM" } else { "PM" })
}

fn weekday(day: i32) -> &'static str {
    ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"][day.rem_euclid(7) as usize]
}

/// wttr.in answers `+51°F|Overcast`; anything else is an error page.
fn parse_weather(body: &str) -> Option<(String, String)> {
    let (temp, sky) = body.trim().split_once('|')?;
    if !temp.contains('°') || sky.len() > 40 {
        return None;
    }
    Some((temp.trim_start_matches('+').to_string(), sky.trim().to_string()))
}

// ponytail: location comes from the caller's IP via wttr.in. Add a --location flag if that guesses wrong.
fn weather_loop(core: &Core) {
    while !core.quit.load(Ordering::Relaxed) {
        let fetched = ureq::get("https://wttr.in/?format=%t|%C")
            .timeout(Duration::from_secs(10))
            .call()
            .ok()
            .and_then(|r| r.into_string().ok())
            .and_then(|b| parse_weather(&b));
        let ok = fetched.is_some();
        if ok {
            core.lock().weather = fetched;
        }
        // ponytail: refresh every 15 minutes, retry a failure after 1 minute
        std::thread::sleep(Duration::from_secs(if ok { 900 } else { 60 }));
    }
}

/// Keeps music quiet while the assistant is engaged, and restores it once it is not.
fn duck_loop(core: &Core) {
    while core.pause(DUCK_TICK, |_| false) {
        core.duck(true);
    }
}

fn todo_loop(core: &Core, mut watcher: TodoWatcher) {
    while core.pause(TODO_POLL, |_| false) {
        if let Some(items) = watcher.poll() {
            core.lock().todos = items;
        }
    }
}

fn pad(area: Rect) -> Rect {
    Rect::new(area.x + 2.min(area.width), area.y, area.width.saturating_sub(4), area.height)
}

fn bell() {
    let mut out = std::io::stdout();
    let _ = out.write_all(b"\x07");
    let _ = out.flush();
}

fn stop_or_cut(core: &Arc<Core>) {
    let run_id = {
        let mut s = core.lock();
        if s.speaking || s.speech_waiting {
            core.stop_speech.store(true, Ordering::Relaxed);
            s.speech.clear();
            return;
        }
        s.run_id.clone()
    };
    let Some(run_id) = run_id else { return };
    let core = core.clone();
    std::thread::spawn(move || {
        let message = match core.client.stop(&run_id) {
            Ok(()) => return,
            Err(HermesError::Unavailable(_)) => "Could not stop the run. Hermes is unavailable.".into(),
            Err(HermesError::Status(_)) => "Could not stop the run. It had already finished.".into(),
            Err(HermesError::Auth(_)) => "Could not stop the run. Hermes rejected the API key.".into(),
            Err(e) => format!("Could not stop the run. {e}"),
        };
        core.log(message);
    });
}

fn choose(core: &Arc<Core>, choice: &'static str) {
    let (run_id, request_id) = {
        let s = core.lock();
        match (&s.run_id, &s.approval) {
            (Some(run), Some(approval)) => (run.clone(), approval["request_id"].as_str().map(str::to_string)),
            _ => return,
        }
    };
    let core = core.clone();
    std::thread::spawn(move || {
        let message = match core.client.approve(&run_id, choice, request_id.as_deref()) {
            Ok(()) => return,
            Err(HermesError::Unavailable(_)) => "Could not send the approval. Hermes is unavailable.".into(),
            Err(HermesError::Status(_)) => "Could not send the approval. It had already finished.".into(),
            Err(HermesError::Auth(_)) => "Could not send the approval. Hermes rejected the API key.".into(),
            Err(e) => format!("Could not send the approval. {e}"),
        };
        core.log(message);
    });
}

fn toggle_mute(core: &Core) {
    let mut s = core.lock();
    s.muted = !s.muted;
    if s.muted {
        s.mute_gen += 1;
        s.mic_lost = false;
        s.reset_segmenter();
        s.utterances.clear();
        if let Some(gate) = &mut s.gate {
            gate.sleep();
        }
    } else {
        // Reopen at once rather than after the dead-mic delay.
        s.retry_mic = false;
    }
    core.changed.notify_all();
}

pub fn run(config: Config) -> std::io::Result<()> {
    let _ = ensure_todo_file(&config.todo_path);
    let mut todo_watcher = TodoWatcher::new(config.todo_path.clone());
    let initial_todos = todo_watcher.poll().unwrap_or_default();
    let instructions = format!("{} {}", hermes_instructions(&config.todo_path), music::INSTRUCTIONS);
    let core = Arc::new(Core {
        state: Mutex::new(Shared {
            gate: config.wake.as_ref().map(|_| WakeGate::new(Duration::from_secs(8))),
            segmenter: Some(config.segmenter),
            session_id: session_id(),
            todos: initial_todos,
            ..Shared::default()
        }),
        changed: Condvar::new(),
        quit: AtomicBool::new(false),
        stop_speech: Arc::new(AtomicBool::new(false)),
        client: config.client,
        wake: config.wake,
        wake_threshold: config.wake_threshold,
        show_scores: config.show_scores,
        speaks: config.speaker.is_some(),
        todo_path: config.todo_path,
        instructions,
        ducker: Mutex::new(Ducker::new(music::socket_path(), config.duck_volume)),
    });
    let spawn = |name: &str, f: Box<dyn FnOnce(&Core) + Send>| {
        let core = core.clone();
        std::thread::Builder::new().name(name.into()).spawn(move || f(&core)).expect("spawn thread");
    };
    let (device, wake_word) = (config.device, config.wake_word);
    spawn("hermes-capture", Box::new(move |c| capture_loop(c, device, wake_word)));
    let stt = config.transcriber;
    spawn("hermes-stt", Box::new(move |c| stt_loop(c, stt)));
    spawn("hermes-send", Box::new(send_loop));
    spawn("hermes-weather", Box::new(weather_loop));
    spawn("hermes-todos", Box::new(move |c| todo_loop(c, todo_watcher)));
    spawn("hermes-duck", Box::new(duck_loop));
    if let Some(speaker) = config.speaker {
        let speaker = Arc::new(speaker);
        spawn("hermes-speak", Box::new(move |c| speak_loop(c, speaker)));
    }

    let mut terminal = ratatui::init();
    let mut orb = Orb::default();
    let tick = Duration::from_secs_f32(1.0 / FPS);
    let mut next_tick = Instant::now();
    let result = loop {
        if Instant::now() >= next_tick {
            next_tick += tick;
            let mut s = core.lock();
            s.expire_session(Instant::now(), session_id);
            orb.tick(s.status(), s.level);
            if std::mem::take(&mut s.bell) {
                bell();
            }
            let drawn = terminal.draw(|f| draw(f, &s, &orb, core.wake.as_deref(), &config.device_name));
            if let Err(e) = drawn {
                break Err(e);
            }
        }
        match event::poll(next_tick.saturating_duration_since(Instant::now())) {
            Ok(false) => continue,
            Ok(true) => {}
            Err(e) => break Err(e),
        }
        let Ok(Event::Key(key)) = event::read() else { continue };
        if key.kind != KeyEventKind::Press {
            continue;
        }
        match key.code {
            KeyCode::Char('q') => break Ok(()),
            KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => break Ok(()),
            KeyCode::Char('m') => toggle_mute(&core),
            KeyCode::Esc => stop_or_cut(&core),
            KeyCode::Char('y') => choose(&core, "once"),
            KeyCode::Char('n') => choose(&core, "deny"),
            _ => {}
        }
    };
    core.quit.store(true, Ordering::Relaxed);
    core.stop_speech.store(true, Ordering::Relaxed);
    core.changed.notify_all();
    // Quitting mid-conversation must not leave the music quiet.
    core.duck(false);
    let run_id = core.lock().run_id.clone();
    if let Some(run_id) = run_id {
        let _ = core.client.stop(&run_id);
    }
    ratatui::restore();
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wrap_breaks_words_and_long_runs() {
        assert_eq!(wrap("one two three", 7), ["one two", "three"]);
        assert_eq!(wrap("abcdefghij", 4), ["abcd", "efgh", "ij"]);
        assert_eq!(wrap("a\n\nb", 10), ["a", "", "b"]);
    }

    #[test]
    fn clock_is_twelve_hour_and_weather_rejects_error_pages() {
        assert_eq!(clock(0, 5), "12:05 AM");
        assert_eq!(clock(12, 0), "12:00 PM");
        assert_eq!(clock(23, 59), "11:59 PM");
        assert_eq!(parse_weather("+51°F|Overcast \n"), Some(("51°F".into(), "Overcast".into())));
        assert_eq!(parse_weather("location not found: upstream error"), None);
    }

    #[test]
    fn engaged_while_awake_or_busy_and_not_while_asleep() {
        let mut s = Shared { gate: Some(WakeGate::new(Duration::from_secs(8))), ..Shared::default() };
        s.hearing = true; // room talk, or the music itself, while asleep
        assert!(!s.engaged());
        s.gate.as_mut().unwrap().wake();
        assert!(s.engaged());
        s.gate.as_mut().unwrap().sleep();
        s.speaking = true;
        assert!(s.engaged());
        s.speaking = false;
        s.run_id = Some("r".into());
        assert!(s.engaged());
    }

    #[test]
    fn open_mic_is_engaged_only_while_busy() {
        let mut s = Shared::default(); // no gate
        assert!(!s.engaged());
        s.transcribing = true;
        assert!(s.engaged());
    }

    #[test]
    fn talk_heard_only_while_asleep_is_dropped() {
        let mut s = Shared { gate: Some(WakeGate::new(Duration::from_secs(8))), ..Shared::default() };
        assert_eq!(s.finish_utterance(vec![1; 100], 2000), None);
        s.heard_awake = true;
        assert_eq!(s.finish_utterance(vec![1; 100], 2000), Some(vec![1; 100]));
        assert!(!s.heard_awake, "each utterance starts unheard");
        let mut open_mic = Shared::default();
        assert_eq!(open_mic.finish_utterance(vec![1; 100], 2000), Some(vec![1; 100]));
    }

    #[test]
    fn wake_phrase_alone_is_not_sent_and_earlier_talk_is_cut() {
        let mut s = Shared { gate: Some(WakeGate::new(Duration::from_secs(8))), ..Shared::default() };
        s.heard_awake = true;
        s.wake_mark = Some((5 * 16000, 1500));
        assert_eq!(s.finish_utterance(vec![1; 6 * 16000], 1500 + COMMAND_MIN_MS - 30), None);
        assert_eq!(s.wake_mark, None);
        s.heard_awake = true;
        s.wake_mark = Some((5 * 16000, 1500));
        let kept = s.finish_utterance(vec![1; 6 * 16000], 2500).unwrap();
        assert_eq!(kept.len(), 16000 + WAKE_LEAD);
        s.heard_awake = true;
        s.wake_mark = Some((8000, 300));
        assert_eq!(s.finish_utterance(vec![1; 32000], 1200).unwrap().len(), 32000);
    }

    #[test]
    fn wake_phrase_over_a_reply_cuts_it_and_keeps_listening() {
        let stop = AtomicBool::new(false);
        let mut s = Shared { gate: Some(WakeGate::new(Duration::from_secs(60))), speaking: true, ..Shared::default() };
        s.speech.push_back("next reply".into());
        s.gate.as_mut().unwrap().wake();
        s.barge_in(&stop);
        assert!(stop.load(Ordering::Relaxed));
        assert!(s.speech.is_empty(), "queued replies are dropped");
        s.speech_ended();
        assert!(!s.speaking && !s.barged);
        // The 60 s wake window stays, not the zero-length follow-up a normal end would give.
        assert!(!s.gate.as_mut().unwrap().expire(false));
        assert!(!s.asleep());
    }

    #[test]
    fn idle_session_resets_after_a_minute_but_never_while_busy() {
        let start = Instant::now();
        let fresh = || "fresh".to_string();
        let mut s = Shared { session_id: "old".into(), ..Shared::default() };
        // No request yet: nothing to reset.
        assert!(!s.expire_session(start + IDLE_RESET * 2, fresh));

        s.last_activity = Some(start);
        s.push_log(Entry::You("hello".into()));
        // A follow-up inside the minute keeps the session.
        assert!(!s.expire_session(start + IDLE_RESET - Duration::from_secs(1), fresh));
        assert_eq!(s.session_id, "old");

        // Busy past the minute: no reset, and the idle time restarts when it ends.
        s.run_id = Some("r".into());
        assert!(!s.expire_session(start + IDLE_RESET * 2, fresh));
        s.run_id = None;
        s.speaking = true;
        assert!(!s.expire_session(start + IDLE_RESET * 3, fresh));
        s.speaking = false;
        s.approval = Some(Value::Null);
        assert!(!s.expire_session(start + IDLE_RESET * 4, fresh));
        s.approval = None;
        assert!(!s.expire_session(start + IDLE_RESET * 5 - Duration::from_secs(1), fresh));
        assert_eq!(s.session_id, "old");

        assert!(s.expire_session(start + IDLE_RESET * 5, fresh));
        assert_eq!(s.session_id, "fresh");
        assert_eq!(s.last_activity, None);
        assert!(matches!(&s.log.iter().collect::<Vec<_>>()[..], [Entry::Plain(t)] if t == "New conversation"));
        // A fresh session is not reset again.
        assert!(!s.expire_session(start + IDLE_RESET * 10, || "again".into()));
    }

    #[test]
    fn status_priority_matches_the_python_client() {
        let mut s = Shared { gate: Some(WakeGate::new(Duration::ZERO)), ..Shared::default() };
        assert_eq!(s.status(), "waiting");
        s.hearing = true;
        assert_eq!(s.status(), "hearing");
        s.run_id = Some("r".into());
        assert_eq!(s.status(), "thinking");
        s.approval = Some(Value::Null);
        assert_eq!(s.status(), "approval");
        s.mic_lost = true;
        assert_eq!(s.status(), "mic lost");
    }

    fn line_text(line: &Line) -> String {
        line.spans.iter().map(|s| s.content.as_ref()).collect()
    }

    fn buffer_rect_text(buf: &ratatui::buffer::Buffer, rect: Rect) -> String {
        let mut out = String::new();
        for y in rect.y..rect.bottom() {
            for x in rect.x..rect.right() {
                out.push_str(buf[(x, y)].symbol());
            }
            out.push('\n');
        }
        out
    }

    #[test]
    fn todo_lines_wraps_indents_and_styles_open_vs_done() {
        assert!(todo_lines(&[], 24, 0).is_empty());
        let one = todo_lines(&[], 24, 1);
        assert_eq!(one.len(), 1);
        assert_eq!(line_text(&one[0]), "TODO");

        let empty = todo_lines(&[], 24, 4);
        assert_eq!(empty.len(), 2);
        assert_eq!(line_text(&empty[0]), "TODO");
        assert_eq!(line_text(&empty[1]), "No todos");
        assert_eq!(empty[1].style.fg, Some(Color::Rgb(0x70, 0x70, 0x70)));

        let items = vec![
            TodoItem { text: "buy fresh groceries from market".into(), done: false },
            TodoItem { text: "send quarterly report today".into(), done: true },
        ];
        let width = 18;
        let lines = todo_lines(&items, width, 12);

        assert_eq!(line_text(&lines[0]), "TODO  1/2");
        assert_eq!(lines[0].spans[0].style.fg, Some(Color::Rgb(0xd7, 0xba, 0x74)));
        assert!(lines[0].spans[0].style.add_modifier.contains(Modifier::BOLD));
        assert_eq!(lines[0].spans[1].style.fg, Some(Color::Rgb(0x90, 0x90, 0x90)));

        for line in &lines[1..] {
            assert!(
                line_text(line).chars().count() <= width,
                "line exceeded column width {width}: {:?}",
                line_text(line)
            );
        }

        // Open item: "[ ] " prefix + wrapped continuation indented by 4 spaces.
        assert_eq!(lines[1].spans[0].content, "[ ] ");
        assert_eq!(lines[1].spans[0].style.fg, Some(Color::Rgb(0x5f, 0xd0, 0xe0)));
        assert!(lines[1].spans[0].style.add_modifier.contains(Modifier::BOLD));
        assert_eq!(lines[1].spans[1].style.fg, Some(Color::Rgb(0xe0, 0xe0, 0xe0)));
        assert!(!lines[1].spans[1].style.add_modifier.contains(Modifier::CROSSED_OUT));

        assert_eq!(lines[2].spans[0].content, "    ");
        assert_eq!(lines[2].spans[1].style.fg, Some(Color::Rgb(0xe0, 0xe0, 0xe0)));

        // Done item: "[x] " prefix + crossed-out dim text and indented continuation.
        let done_idx = lines.iter().position(|l| l.spans[0].content == "[x] ").expect("done item line");
        assert_eq!(lines[done_idx].spans[0].style.fg, Some(Color::Rgb(0x3d, 0xde, 0x7a)));
        assert_eq!(lines[done_idx].spans[1].style.fg, Some(Color::Rgb(0x78, 0x78, 0x78)));
        assert!(lines[done_idx].spans[1].style.add_modifier.contains(Modifier::CROSSED_OUT));

        assert_eq!(lines[done_idx + 1].spans[0].content, "    ");
        assert!(lines[done_idx + 1].spans[1].style.add_modifier.contains(Modifier::CROSSED_OUT));
    }

    #[test]
    fn todo_lines_respects_max_lines_and_shows_overflow() {
        let items: Vec<TodoItem> =
            (1..=5).map(|i| TodoItem { text: format!("task {i}"), done: i % 2 == 0 }).collect();

        for max in 0..=8 {
            let lines = todo_lines(&items, 24, max);
            assert!(lines.len() <= max, "exceeded max_lines={max}: got {}", lines.len());
        }

        let cut = todo_lines(&items, 24, 4);
        assert_eq!(cut.len(), 4);
        assert_eq!(line_text(&cut[0]), "TODO  2/5");
        assert_eq!(line_text(&cut[1]), "[ ] task 1");
        assert_eq!(line_text(&cut[2]), "[x] task 2");
        assert_eq!(line_text(&cut[3]), "+3 more");

        let exact = todo_lines(&items, 24, 6);
        assert_eq!(exact.len(), 6);
        assert_eq!(line_text(&exact[5]), "[ ] task 5");

        let single_long = vec![TodoItem { text: "alpha beta gamma delta epsilon zeta".into(), done: false }];
        let truncated = todo_lines(&single_long, 14, 3);
        assert_eq!(truncated.len(), 3);
        assert_eq!(line_text(&truncated[2]), "…");
    }

    #[test]
    fn body_columns_switches_at_narrow_threshold_and_caps_width() {
        let (todo, div, chat) = body_columns(Rect::new(0, 0, MIN_TWO_COL_WIDTH - 1, 30), 10, 19);
        assert!(todo.is_none() && div.is_none());
        assert_eq!(chat, Rect::new(0, 10, MIN_TWO_COL_WIDTH - 1, 19));

        let (todo, div, chat) = body_columns(Rect::new(0, 0, MIN_TWO_COL_WIDTH, 30), 10, 19);
        let todo = todo.unwrap();
        let div = div.unwrap();
        assert_eq!(todo, Rect::new(0, 10, 24, 19));
        assert_eq!(div, Rect::new(24, 10, 1, 19));
        assert_eq!(chat, Rect::new(25, 10, 47, 19));
        assert_eq!(todo.right(), div.x);
        assert_eq!(div.right(), chat.x);

        let (todo, div, chat) = body_columns(Rect::new(0, 0, 160, 40), 14, 25);
        let todo = todo.unwrap();
        let div = div.unwrap();
        assert_eq!(todo, Rect::new(16, 14, 38, 25));
        assert_eq!(div, Rect::new(54, 14, 1, 25));
        assert_eq!(chat, Rect::new(55, 14, 89, 25));
        assert_eq!(todo.width + div.width + chat.width, BODY_MAX_WIDTH);

        let (todo, div, chat) = body_columns(Rect::new(0, 0, 120, 30), 10, 0);
        assert!(todo.is_none() && div.is_none());
        assert_eq!(chat.height, 0);
    }

    #[test]
    fn draw_wide_renders_todo_and_chat_side_by_side_without_overlap() {
        use ratatui::backend::TestBackend;
        use ratatui::Terminal;

        let mut s = Shared::default();
        s.todos = vec![
            TodoItem { text: "water plants".into(), done: false },
            TodoItem { text: "call dentist".into(), done: true },
        ];
        s.push_log(Entry::You("check my schedule".into()));
        s.push_log(Entry::HermesLabel);
        s.log_line("You have two meetings.");
        s.reply = "Working on the summary now".into();

        let mut terminal = Terminal::new(TestBackend::new(120, 32)).unwrap();
        let orb = Orb::default();
        terminal.draw(|f| draw(f, &s, &orb, Some("hey john"), "Built-in Mic")).unwrap();
        let buf = terminal.backend().buffer();

        let area = Rect::new(0, 0, 120, 32);
        let header_rows = (32u16 * 35 / 100).clamp(7, 15) + 3;
        let body_h = 32u16.saturating_sub(header_rows);
        let (todo_rect, div_rect, chat_rect) = body_columns(area, header_rows, body_h);
        let todo_rect = todo_rect.unwrap();
        let div_rect = div_rect.unwrap();

        let header_text = buffer_rect_text(buf, Rect::new(0, 0, 120, header_rows));
        assert!(!header_text.contains("TODO"));
        assert!(!header_text.contains("water plants"));
        assert!(!header_text.contains("check my schedule"));

        let todo_text = buffer_rect_text(buf, todo_rect);
        assert!(todo_text.contains("TODO  1/2"));
        assert!(todo_text.contains("[ ] water plants"));
        assert!(todo_text.contains("[x] call dentist"));
        assert!(!todo_text.contains("check my schedule"));
        assert!(!todo_text.contains("Working on the summary"));

        for y in div_rect.y..div_rect.bottom() {
            let cell = &buf[(div_rect.x, y)];
            assert_eq!(cell.symbol(), "│");
            assert_eq!(cell.fg, Color::Rgb(0x45, 0x45, 0x45));
        }

        let chat_text = buffer_rect_text(buf, chat_rect);
        assert!(chat_text.contains("You  check my schedule"));
        assert!(chat_text.contains("You have two meetings."));
        assert!(chat_text.contains("Hermes  Working on the summary now"));
        assert!(!chat_text.contains("TODO"));
        assert!(!chat_text.contains("water plants"));

        // No key hints: the device has no keyboard.
        let full = buffer_rect_text(buf, area);
        assert!(!full.contains("Mute"));
        assert!(!full.contains("Quit"));
    }

    #[test]
    fn draw_narrow_hides_todo_panel_and_renders_chat() {
        use ratatui::backend::TestBackend;
        use ratatui::Terminal;

        let mut s = Shared::default();
        s.todos = vec![TodoItem { text: "secret todo item".into(), done: false }];
        s.push_log(Entry::You("hello narrow screen".into()));
        s.reply = "replying in single column".into();

        let mut terminal = Terminal::new(TestBackend::new(60, 30)).unwrap();
        let orb = Orb::default();
        terminal.draw(|f| draw(f, &s, &orb, Some("hey john"), "USB Mic")).unwrap();
        let buf = terminal.backend().buffer();

        let full = buffer_rect_text(buf, Rect::new(0, 0, 60, 30));
        assert!(!full.contains("TODO"));
        assert!(!full.contains("secret todo item"));
        assert!(!full.contains('│'));
        assert!(full.contains("You  hello narrow screen"));
        assert!(full.contains("Hermes  replying in single column"));

        assert!(!full.contains("Mute"));
        assert!(!full.contains("Quit"));
    }

    #[test]
    fn draw_tiny_terminals_do_not_panic() {
        use ratatui::backend::TestBackend;
        use ratatui::Terminal;

        let mut s = Shared::default();
        s.todos = vec![
            TodoItem { text: "a very long todo item that wraps aggressively".into(), done: false },
            TodoItem { text: "finished task".into(), done: true },
        ];
        for i in 0..20 {
            s.push_log(Entry::You(format!("message number {i} with extra text")));
        }
        s.reply = "streaming text that is long enough to wrap across several lines".into();

        let orb = Orb::default();
        for (w, h) in [(20, 8), (10, 4), (4, 2), (80, 1), (0, 0)] {
            let mut terminal = Terminal::new(TestBackend::new(w, h)).unwrap();
            terminal.draw(|f| draw(f, &s, &orb, Some("hey john"), "Mic")).unwrap();
        }
    }

    #[test]
    fn expire_session_preserves_todos_while_clearing_chat() {
        let start = Instant::now();
        let mut s = Shared { session_id: "old".into(), last_activity: Some(start), ..Shared::default() };
        let expected_todos = vec![
            TodoItem { text: "keep this open task".into(), done: false },
            TodoItem { text: "keep this done task".into(), done: true },
        ];
        s.todos = expected_todos.clone();
        s.push_log(Entry::You("chat message".into()));
        s.reply = "partial reply".into();

        assert!(s.expire_session(start + IDLE_RESET, || "next".into()));
        assert_eq!(s.session_id, "next");
        assert!(s.reply.is_empty());
        assert!(s.pending.is_empty());
        assert!(s.speech.is_empty());
        assert!(matches!(&s.log.iter().collect::<Vec<_>>()[..], [Entry::Plain(t)] if t == "New conversation"));
        assert_eq!(s.todos, expected_todos);
    }
}
