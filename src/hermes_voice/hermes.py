"""HTTP client for the local Hermes runs API."""

import json
import os
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path

import httpx
from dotenv import dotenv_values

TERMINAL = {"run.completed", "run.failed", "run.cancelled", "run.interrupted"}


class HermesAuthError(Exception):
    pass


class HermesUnavailable(Exception):
    pass


@dataclass(frozen=True)
class RunEvent:
    name: str
    data: dict


def load_api_key(env_path: Path | None = None) -> str:
    key = os.environ.get("HERMES_API_KEY")
    if key:
        return key
    if env_path is None:
        home = os.environ.get("HERMES_HOME")
        env_path = Path(home) / ".env" if home else Path.home() / ".hermes" / ".env"
    key = dotenv_values(env_path).get("API_SERVER_KEY")
    if not key:
        raise HermesAuthError(
            f"No API key in {env_path}. Set API_SERVER_ENABLED=true and API_SERVER_KEY there."
        )
    return key


def _raise_for_status(response: httpx.Response) -> None:
    status = response.status_code
    if status in (401, 403):
        raise HermesAuthError(f"Hermes rejected the API key ({status})")
    if status == 429 or status >= 500:
        raise HermesUnavailable(f"Hermes unavailable ({status})")
    response.raise_for_status()


class HermesClient:
    def __init__(
        self,
        api_key: str,
        base_url: str = "http://127.0.0.1:8642",
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._client = httpx.AsyncClient(
            base_url=base_url,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=httpx.Timeout(10, read=None),
            transport=transport,
            trust_env=False,
        )

    async def start_run(self, text: str, session_id: str) -> str:
        response = await self._request(
            "POST", "/v1/runs", json={"input": text, "session_id": session_id}
        )
        return response.json()["run_id"]

    async def events(self, run_id: str) -> AsyncIterator[RunEvent]:
        # ponytail: each data line is one JSON object. Multi-line SSE data fields are not joined.
        finished = False
        try:
            async with self._client.stream(
                "GET",
                f"/v1/runs/{run_id}/events",
                timeout=httpx.Timeout(10, read=None),
            ) as response:
                if response.status_code in (401, 403):
                    raise HermesAuthError(
                        f"Hermes rejected the API key ({response.status_code})"
                    )
                if response.status_code not in (404, 500):
                    _raise_for_status(response)
                if response.is_success:
                    async for line in response.aiter_lines():
                        if not line.startswith("data: "):
                            continue
                        try:
                            payload = json.loads(line.removeprefix("data: "))
                        except json.JSONDecodeError:
                            break
                        event = RunEvent(payload["event"], payload)
                        yield event
                        if event.name in TERMINAL:
                            finished = True
                            break
        except HermesAuthError:
            raise
        except httpx.TransportError:
            # Dropped connection. An unfinished stream becomes run.interrupted below.
            pass
        if not finished:
            yield RunEvent("run.interrupted", {})

    async def stop(self, run_id: str) -> None:
        await self._request("POST", f"/v1/runs/{run_id}/stop")

    async def approve(self, run_id: str, choice: str, request_id: str | None = None) -> None:
        body = {"choice": choice}
        if request_id is not None:
            body["request_id"] = request_id
        await self._request("POST", f"/v1/runs/{run_id}/approval", json=body)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def _request(self, method: str, path: str, **kwargs) -> httpx.Response:
        try:
            response = await self._client.request(method, path, timeout=30, **kwargs)
        except httpx.TransportError as exc:
            raise HermesUnavailable(str(exc)) from exc
        _raise_for_status(response)
        return response
