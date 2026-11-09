//! HTTP client for the local Hermes runs API.

use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use std::time::Duration;

use serde_json::{json, Value};

pub const TERMINAL: [&str; 4] = ["run.completed", "run.failed", "run.cancelled", "run.interrupted"];

#[derive(Debug)]
pub enum HermesError {
    Auth(String),
    Unavailable(String),
    /// A 4xx other than auth, such as stopping a run that already finished.
    Status(u16),
    Other(String),
}

impl std::fmt::Display for HermesError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Auth(m) | Self::Unavailable(m) | Self::Other(m) => f.write_str(m),
            Self::Status(code) => write!(f, "Hermes returned {code}"),
        }
    }
}

pub struct RunEvent {
    pub name: String,
    pub data: Value,
}

pub fn load_api_key() -> Result<String, String> {
    if let Ok(key) = std::env::var("HERMES_API_KEY") {
        if !key.is_empty() {
            return Ok(key);
        }
    }
    let path = match std::env::var_os("HERMES_HOME") {
        Some(home) => PathBuf::from(home).join(".env"),
        None => PathBuf::from(std::env::var_os("HOME").unwrap_or_default()).join(".hermes/.env"),
    };
    crate::dotenv_get(&path, "API_SERVER_KEY").ok_or_else(|| {
        format!(
            "No API key in {}. Set API_SERVER_ENABLED=true and API_SERVER_KEY there.",
            path.display()
        )
    })
}

fn status_error(code: u16) -> HermesError {
    match code {
        401 | 403 => HermesError::Auth(format!("Hermes rejected the API key ({code})")),
        429 | 500.. => HermesError::Unavailable(format!("Hermes unavailable ({code})")),
        _ => HermesError::Status(code),
    }
}

pub struct HermesClient {
    agent: ureq::Agent,
    base_url: String,
    auth: String,
}

impl HermesClient {
    pub fn new(api_key: &str, base_url: &str) -> Self {
        // ureq ignores proxy variables unless told to, matching trust_env=False.
        let agent = ureq::AgentBuilder::new().timeout_connect(Duration::from_secs(10)).build();
        Self { agent, base_url: base_url.to_string(), auth: format!("Bearer {api_key}") }
    }

    fn post(&self, path: &str, body: Option<Value>) -> Result<Value, HermesError> {
        let response = self
            .agent
            .post(&format!("{}{path}", self.base_url))
            .set("Authorization", &self.auth)
            .timeout(Duration::from_secs(30));
        let response = match body {
            Some(body) => response.send_json(body),
            None => response.call(),
        };
        match response {
            Ok(r) => Ok(r.into_json().unwrap_or(Value::Null)),
            Err(ureq::Error::Status(code, _)) => Err(status_error(code)),
            Err(e) => Err(HermesError::Unavailable(e.to_string())),
        }
    }

    pub fn start_run(&self, text: &str, session_id: &str) -> Result<String, HermesError> {
        let body = self.post("/v1/runs", Some(json!({"input": text, "session_id": session_id})))?;
        body["run_id"]
            .as_str()
            .map(str::to_string)
            .ok_or_else(|| HermesError::Other("Hermes returned no run_id".into()))
    }

    /// Calls `on_event` for each run event. An unfinished stream ends with run.interrupted.
    // ponytail: each data line is one JSON object. Multi-line SSE data fields are not joined.
    pub fn events(&self, run_id: &str, mut on_event: impl FnMut(RunEvent)) -> Result<(), HermesError> {
        let response = self
            .agent
            .get(&format!("{}/v1/runs/{run_id}/events", self.base_url))
            .set("Authorization", &self.auth)
            .call();
        let response = match response {
            Ok(r) => Some(r),
            Err(ureq::Error::Status(code @ (401 | 403), _)) => return Err(status_error(code)),
            Err(ureq::Error::Status(404 | 500, _)) => None,
            Err(ureq::Error::Status(code, _)) => return Err(status_error(code)),
            // Dropped connection. Becomes run.interrupted below.
            Err(ureq::Error::Transport(_)) => None,
        };
        let mut finished = false;
        if let Some(response) = response {
            for line in BufReader::new(response.into_reader()).lines() {
                let Ok(line) = line else { break };
                let Some(raw) = line.strip_prefix("data: ") else { continue };
                let Ok(data) = serde_json::from_str::<Value>(raw) else { break };
                let name = data["event"].as_str().unwrap_or_default().to_string();
                let terminal = TERMINAL.contains(&name.as_str());
                on_event(RunEvent { name, data });
                if terminal {
                    finished = true;
                    break;
                }
            }
        }
        if !finished {
            on_event(RunEvent { name: "run.interrupted".into(), data: json!({}) });
        }
        Ok(())
    }

    pub fn stop(&self, run_id: &str) -> Result<(), HermesError> {
        self.post(&format!("/v1/runs/{run_id}/stop"), None).map(drop)
    }

    pub fn approve(&self, run_id: &str, choice: &str, request_id: Option<&str>) -> Result<(), HermesError> {
        let mut body = json!({"choice": choice});
        if let Some(id) = request_id {
            body["request_id"] = json!(id);
        }
        self.post(&format!("/v1/runs/{run_id}/approval"), Some(body)).map(drop)
    }
}
