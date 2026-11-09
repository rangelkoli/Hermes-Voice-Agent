//! Wake window. A small local model hears the phrase; this only keeps the clock.

use std::time::{Duration, Instant};

/// Stays open for a few seconds after the wake phrase, then goes back to sleep.
pub struct WakeGate {
    window: Duration,
    pub awake: bool,
    deadline: Option<Instant>,
}

impl WakeGate {
    pub fn new(window: Duration) -> Self {
        Self { window, awake: false, deadline: None }
    }

    pub fn wake(&mut self) {
        self.awake = true;
        self.deadline = Some(Instant::now() + self.window);
    }

    pub fn touch(&mut self) {
        if self.awake {
            self.deadline = Some(Instant::now() + self.window);
        }
    }

    pub fn expire(&mut self, busy: bool) -> bool {
        match self.deadline {
            Some(deadline) if self.awake && !busy && Instant::now() >= deadline => {
                self.sleep();
                true
            }
            _ => false,
        }
    }

    pub fn sleep(&mut self) {
        self.awake = false;
        self.deadline = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gate_expires_only_when_idle() {
        let mut gate = WakeGate::new(Duration::ZERO);
        assert!(!gate.expire(false));
        gate.wake();
        assert!(!gate.expire(true));
        assert!(gate.awake);
        assert!(gate.expire(false));
        assert!(!gate.awake);
    }
}
