import asyncio, json
import httpx, pytest
from hermes_voice.hermes import HermesAuthError, HermesClient, HermesUnavailable, load_api_key

def sse(*events):
    body = b": open\n\n"
    for i, e in enumerate(events):
        body += f"id: {i}\ndata: {json.dumps(e)}\n\n".encode()
    return body + b": stream closed\n\n"

def client(handler):
    return HermesClient("k" * 32, transport=httpx.MockTransport(handler))

def test_start_run_sends_input_session_and_bearer():
    seen = {}
    def h(req):
        seen.update(path=req.url.path, auth=req.headers["authorization"], body=json.loads(req.content))
        return httpx.Response(202, json={"run_id": "run_1", "status": "started"})
    assert asyncio.run(client(h).start_run("hi", "voice-1")) == "run_1"
    assert seen == {"path": "/v1/runs", "auth": "Bearer " + "k" * 32, "body": {"input": "hi", "session_id": "voice-1"}}

@pytest.mark.parametrize("status,exc", [(401, HermesAuthError), (429, HermesUnavailable), (503, HermesUnavailable)])
def test_error_statuses_map_to_exceptions(status, exc):
    with pytest.raises(exc):
        asyncio.run(client(lambda r: httpx.Response(status, json={})).start_run("hi", "s"))

def test_connection_refused_is_unavailable():
    def h(req):
        raise httpx.ConnectError("refused")
    with pytest.raises(HermesUnavailable):
        asyncio.run(client(h).start_run("hi", "s"))

async def collect(c):
    return [e async for e in c.events("run_1")]

def test_events_parse_and_stop_at_terminal():
    body = sse({"event": "message.delta", "delta": "He"}, {"event": "run.completed", "output": "Hello"})
    got = asyncio.run(collect(client(lambda r: httpx.Response(200, content=body))))
    assert [e.name for e in got] == ["message.delta", "run.completed"]
    assert got[1].data["output"] == "Hello"

def test_stream_closing_early_yields_interrupted():
    body = sse({"event": "message.delta", "delta": "He"})
    got = asyncio.run(collect(client(lambda r: httpx.Response(200, content=body))))
    assert got[-1].name == "run.interrupted"

def test_approve_omits_missing_request_id():
    seen = {}
    def h(req):
        seen.update(path=req.url.path, body=json.loads(req.content))
        return httpx.Response(200, json={})
    asyncio.run(client(h).approve("run_1", "deny"))
    assert seen == {"path": "/v1/runs/run_1/approval", "body": {"choice": "deny"}}

def test_key_file_with_inline_comments_parses(tmp_path, monkeypatch):
    monkeypatch.delenv("HERMES_API_KEY", raising=False)
    f = tmp_path / ".env"
    f.write_text("API_SERVER_HOST=0.0.0.0   # phone access\nAPI_SERVER_KEY=abc123   # gateway\n")
    assert load_api_key(f) == "abc123"

def test_missing_key_names_the_file(tmp_path, monkeypatch):
    monkeypatch.delenv("HERMES_API_KEY", raising=False)
    with pytest.raises(HermesAuthError, match=".env"):
        load_api_key(tmp_path / ".env")

def test_hermes_home_is_respected(tmp_path, monkeypatch):
    monkeypatch.delenv("HERMES_API_KEY", raising=False)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / ".env").write_text("API_SERVER_KEY=fromhome\n")
    assert load_api_key() == "fromhome"
