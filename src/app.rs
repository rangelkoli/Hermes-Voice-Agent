//! Terminal UI and worker threads for one Hermes voice conversation.
//!
//! One mutex holds all shared state. Capture, transcription, sending and speech each run on
//! their own thread; the UI thread redraws from that state 12 times a second.

use std::collections::{HashSet, VecDeque};
use std::io::Write;
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

use crate::audio::{level, DeviceSpec, Mic, Segmenter, MIC_SILENT_HINT};
use crate::gemini::{speakable, GeminiSpeaker};
use crate::hermes::{HermesClient, HermesError, RunEvent, TERMINAL};
use crate::orb::{Orb, FPS};
use crate::stt::{is_noise, wake_match, Stt};
use crate::wake::WakeGate;

// ponytail: a dead mic is reopened after a fixed 2 seconds. Quit and mute return early.
const MIC_RETRY: Duration = Duration::from_secs(2);
// ponytail: a failed model download, or a failed send, is retried after a fixed 5 seconds.
const RETRY: Duration = Duration::from_secs(5);
const AUDIO_WAIT: Duration = Duration::from_millis(100);
const SILENT_FRAMES: u32 = 100;
// ponytail: speech waits at most 30 seconds for the user to finish.
const SPEAK_WAIT: Duration = Duration::from_secs(30);
// ponytail: the mic drops audio for 300 ms after speech so it is not transcribed.
const SPEAK_TAIL: Duration = Duration::from_millis(300);
// ponytail: after a spoken reply the mic stays open 2 seconds for a follow-up, no wake phrase.
const FOLLOW_UP: Duration = Duration::from_secs(2);
const MAX_LOG: usize = 1000;
const MAX_WIDTH: u16 = 96;
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
    logged_mic_errors: HashSet<String>,
    log: VecDeque<Entry>,
    level: f32,
    speech: VecDeque<String>,
    bell: bool,
    zero_frames: u32,
    frame_count: u64,
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
    }

    /// Drop the wake, partial speech, and all audio in flight.
    fn close_gate(&mut self) {
        let Some(gate) = &mut self.gate else { return };
        gate.sleep();
        self.mute_gen += 1;
        self.reset_segmenter();
        self.utterances.clear();
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
    pub session_id: String,
    pub wake: Option<String>,
    pub device: Option<DeviceSpec>,
    pub device_name: String,
    pub transcriber: Option<Box<dyn Stt>>,
    pub wake_transcriber: Option<Box<dyn Stt>>,
    pub show_heard: bool,
    pub segmenter: Segmenter,
    pub speaker: Option<GeminiSpeaker>,
}

struct Core {
    state: Mutex<Shared>,
    changed: Condvar,
    quit: AtomicBool,
    stop_speech: Arc<AtomicBool>,
    client: HermesClient,
    session_id: String,
    wake: Option<String>,
    show_heard: bool,
    speaks: bool,
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

fn capture_loop(core: &Core, device: Option<DeviceSpec>) {
    let mut mic = match Mic::new(device) {
        Ok(mic) => mic,
        Err(e) => {
            let mut s = core.lock();
            s.mic_lost = true;
            s.log_line(e);
            return;
        }
    };
    while !core.quitting() {
        if core.lock().muted {
            mic.stop();
            core.pause(AUDIO_WAIT, |_| false);
            continue;
        }
        if !mic.alive() {
            recover_mic(core, &mut mic);
            continue;
        }
        if let Some(frame) = mic.read(AUDIO_WAIT) {
            take_frame(core, frame);
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

fn take_frame(core: &Core, frame: Vec<i16>) {
    let mut guard = core.lock();
    let s = &mut *guard;
    if s.speaking || s.muted {
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
    if let Some(seg) = &mut s.segmenter {
        if let Some(utterance) = seg.push(frame) {
            s.utterances.push_back((utterance, s.mute_gen));
            if let Some(gate) = &mut s.gate {
                gate.touch();
            }
            core.changed.notify_all();
        }
    }
    let busy = s.gate_busy();
    if let Some(gate) = &mut s.gate {
        gate.expire(busy);
    }
    // Asleep stays "waiting" even while someone in the room is talking.
    s.hearing = s.segmenter.as_ref().is_some_and(|seg| seg.in_speech) && !s.asleep();
}

// ---------------------------------------------------------------- transcription

fn load_models(core: &Core, models: [&mut Option<Box<dyn Stt>>; 2]) -> bool {
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

fn stt_loop(core: &Core, mut transcriber: Option<Box<dyn Stt>>, mut wake_transcriber: Option<Box<dyn Stt>>) {
    if !load_models(core, [&mut transcriber, &mut wake_transcriber]) {
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
        let asleep = {
            let s = core.lock();
            if generation != s.mute_gen {
                continue;
            }
            s.asleep()
        };
        if asleep && !open_if_wake(core, &mut wake_transcriber, &utterance, generation) {
            continue;
        }
        {
            let mut s = core.lock();
            // The gate can close during the local check, after this utterance left the queue.
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

/// Hear one asleep utterance locally. True when the cloud should take it.
fn open_if_wake(core: &Core, wake_transcriber: &mut Option<Box<dyn Stt>>, utterance: &[i16], generation: u64) -> bool {
    let heard = match wake_transcriber.as_mut().map(|t| t.transcribe(utterance)) {
        None => String::new(),
        Some(Ok(heard)) => {
            if core.show_heard {
                core.log(format!("heard: {heard}"));
            }
            heard
        }
        Some(Err(e)) => {
            core.log(e);
            return false;
        }
    };
    let mut s = core.lock();
    if generation != s.mute_gen || !s.asleep() {
        return false;
    }
    let Some(remainder) = wake_match(&heard, core.wake.as_deref().unwrap_or("")) else { return false };
    if let Some(gate) = &mut s.gate {
        gate.wake();
    }
    s.bell = true;
    !remainder.is_empty() && !is_noise(&remainder)
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
        let (text, count) = {
            let mut s = core.lock();
            while s.pending.is_empty() || s.stop_sending {
                if core.quitting() {
                    return;
                }
                s = core.wait(s, AUDIO_WAIT);
            }
            (s.pending.join("\n"), s.pending.len())
        };
        let run_id = match core.client.start_run(&text, &core.session_id) {
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
            let mut s = core.lock();
            core.on_event(&mut s, event);
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
        core.pause(SPEAK_TAIL, |_| false);
        let mut s = core.lock();
        s.reset_segmenter();
        s.speaking = false;
        s.speech_waiting = false;
        // Speech that starts within the window keeps the gate open until it is transcribed.
        if !s.muted {
            if let Some(gate) = &mut s.gate {
                gate.open_for(FOLLOW_UP);
            }
        }
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

fn draw(frame: &mut Frame, s: &Shared, orb: &Orb, wake: Option<&str>, device: &str) {
    let area = frame.area();
    let bg = Color::Rgb(BACKGROUND[0], BACKGROUND[1], BACKGROUND[2]);
    frame.render_widget(Block::default().style(Style::default().bg(bg).fg(Color::Rgb(0xe0, 0xe0, 0xe0))), area);
    let width = area.width.min(MAX_WIDTH);
    let x = area.x + (area.width - width) / 2;
    let rows = area.height;
    let mut y = area.y;
    let mut room = rows.saturating_sub(1); // footer
    let mut take = |h: u16| {
        let h = h.min(room);
        let r = Rect::new(x, y, width, h);
        y += h;
        room -= h;
        r
    };

    let orb_area = take((rows * 35 / 100).clamp(7, 15));
    orb.render(orb_area, frame.buffer_mut(), BACKGROUND);

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

    let inner = width.saturating_sub(4) as usize;
    let log = log_tail(&s.log, inner, (rows * 35 / 100) as usize);
    let log_area = take(log.len() as u16);
    frame.render_widget(Paragraph::new(log), pad(log_area));

    if !s.reply.is_empty() {
        // ponytail: shows the newest text only; the full reply lands in the log when done
        let tail: String = s.reply.chars().rev().take(400).collect::<Vec<_>>().into_iter().rev().collect();
        let lines = wrap(&format!("Hermes  {tail}"), inner);
        let shown: Vec<Line> = lines[lines.len().saturating_sub(6)..].iter().cloned().map(Line::from).collect();
        let stream_area = take(shown.len() as u16);
        frame.render_widget(
            Paragraph::new(shown).style(Style::default().fg(Color::Rgb(0x5f, 0xd0, 0xe0))),
            pad(stream_area),
        );
    }

    let key = Style::default().fg(Color::Rgb(0xff, 0xa6, 0x2b)).add_modifier(Modifier::BOLD);
    let mut footer = vec![];
    for (k, label) in [("m", "Mute"), ("esc", "Stop run"), ("y", "Approve"), ("n", "Deny"), ("q", "Quit")] {
        footer.push(Span::styled(format!(" {k} "), key));
        footer.push(Span::raw(format!("{label} ")));
    }
    let footer_area = Rect::new(area.x, area.y + rows.saturating_sub(1), area.width, rows.min(1));
    frame.render_widget(Paragraph::new(Line::from(footer)).style(Style::default().bg(Color::Rgb(0x24, 0x24, 0x24))), footer_area);
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
    let core = Arc::new(Core {
        state: Mutex::new(Shared {
            gate: config.wake.as_ref().map(|_| WakeGate::new(Duration::from_secs(8))),
            segmenter: Some(config.segmenter),
            ..Shared::default()
        }),
        changed: Condvar::new(),
        quit: AtomicBool::new(false),
        stop_speech: Arc::new(AtomicBool::new(false)),
        client: config.client,
        session_id: config.session_id,
        wake: config.wake,
        show_heard: config.show_heard,
        speaks: config.speaker.is_some(),
    });
    let spawn = |name: &str, f: Box<dyn FnOnce(&Core) + Send>| {
        let core = core.clone();
        std::thread::Builder::new().name(name.into()).spawn(move || f(&core)).expect("spawn thread");
    };
    let device = config.device;
    spawn("hermes-capture", Box::new(move |c| capture_loop(c, device)));
    let (stt, wake_stt) = (config.transcriber, config.wake_transcriber);
    spawn("hermes-stt", Box::new(move |c| stt_loop(c, stt, wake_stt)));
    spawn("hermes-send", Box::new(send_loop));
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
}
