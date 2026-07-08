"""EEG data sources. All of them yield (timestamp, {channel_name: value}) samples.

The rest of the Signal service (faa.py, service.py) doesn't care which of
these is plugged in - that's the point of the abstraction.
"""

from __future__ import annotations

import time
from typing import Iterator, Protocol

import numpy as np


class EEGSource(Protocol):
    def connect(self) -> None: ...
    def stream(self) -> Iterator[tuple[float, dict[str, float]]]: ...
    def close(self) -> None: ...


class MockEEGSource:
    """Synthetic 14-channel EEG for development without hardware.

    Alpha-band power at F3/F4 is modulated by `bias(t)` (defaults to slow
    random drift) so FAARewardComputer has something non-trivial to decode.
    Pass a custom `bias_fn(t) -> float in [-1, 1]` to script a known ground
    truth for tests (e.g. "reward should rise for the first 10s").
    """

    def __init__(
        self,
        channels: list[str],
        sample_rate_hz: int = 128,
        bias_fn=None,
        seed: int = 0,
    ):
        self.channels = channels
        self.fs = sample_rate_hz
        self._bias_fn = bias_fn or (lambda t: 0.0)
        self._rng = np.random.default_rng(seed)
        self._t = 0.0
        self._dt = 1.0 / sample_rate_hz

    def connect(self) -> None:
        self._t = 0.0

    def stream(self) -> Iterator[tuple[float, dict[str, float]]]:
        while True:
            self._t += self._dt
            bias = float(np.clip(self._bias_fn(self._t), -1.0, 1.0))
            sample = {}
            for ch in self.channels:
                # Base alpha oscillation (10 Hz) plus 1/f-ish noise.
                alpha = np.sin(2 * np.pi * 10.0 * self._t)
                noise = self._rng.normal(0, 0.3)
                gain = 1.0
                if ch == "F4":
                    gain = 1.0 + 0.6 * bias
                elif ch == "F3":
                    gain = 1.0 - 0.6 * bias
                sample[ch] = alpha * gain + noise
            yield self._t, sample

    def close(self) -> None:
        pass


class EmotivCortexSource:
    """EMOTIV Cortex API client for the EPOC X headset (WebSocket JSON-RPC).

    Flow: requestAccess -> authorize -> createSession -> subscribe("eeg").
    No official PyPI SDK exists; this talks to the Cortex websocket directly
    via `websocket-client`. Requires EMOTIV_CLIENT_ID / EMOTIV_CLIENT_SECRET.
    """

    CORTEX_URL = "wss://localhost:6868"

    def __init__(self, client_id: str | None = None, client_secret: str | None = None):
        self.client_id = client_id
        self.client_secret = client_secret
        self._ws = None
        self._cortex_token: str | None = None
        self._session_id: str | None = None
        self._req_id = 0

    def _next_id(self) -> int:
        self._req_id += 1
        return self._req_id

    def _call(self, method: str, params: dict) -> dict:
        import json

        assert self._ws is not None
        payload = {"jsonrpc": "2.0", "id": self._next_id(), "method": method, "params": params}
        self._ws.send(json.dumps(payload))
        response = json.loads(self._ws.recv())
        if "error" in response:
            raise RuntimeError(f"Cortex API error on {method}: {response['error']}")
        return response["result"]

    def connect(self) -> None:
        import websocket

        if not self.client_id or not self.client_secret:
            raise RuntimeError(
                "EMOTIV_CLIENT_ID / EMOTIV_CLIENT_SECRET are required to connect to Cortex"
            )
        self._ws = websocket.create_connection(self.CORTEX_URL)
        self._call("requestAccess", {"clientId": self.client_id, "clientSecret": self.client_secret})
        auth = self._call(
            "authorize", {"clientId": self.client_id, "clientSecret": self.client_secret}
        )
        self._cortex_token = auth["cortexToken"]
        session = self._call(
            "createSession",
            {"cortexToken": self._cortex_token, "status": "active"},
        )
        self._session_id = session["id"]
        self._call(
            "subscribe",
            {"cortexToken": self._cortex_token, "session": self._session_id, "streams": ["eeg"]},
        )

    def stream(self, channels: list[str]) -> Iterator[tuple[float, dict[str, float]]]:
        import json

        assert self._ws is not None, "call connect() first"
        while True:
            msg = json.loads(self._ws.recv())
            if "eeg" not in msg:
                continue
            values = msg["eeg"]
            # Cortex sends [time, ch0, ch1, ...]; caller's channel order must
            # match the subscribed montage from config.yaml.
            t = values[0]
            sample = dict(zip(channels, values[1:]))
            yield t, sample

    def close(self) -> None:
        if self._ws is not None:
            self._ws.close()
            self._ws = None


class BrainFlowLSLSource:
    """Pulls EEG from an LSL stream (e.g. BrainFlow's LSL output). Lazy-imports
    pylsl so the rest of the codebase works without it installed.
    """

    def __init__(self, channels: list[str], stream_name: str = "obci_eeg"):
        self.channels = channels
        self.stream_name = stream_name
        self._inlet = None

    def connect(self) -> None:
        from pylsl import StreamInlet, resolve_byprop

        streams = resolve_byprop("name", self.stream_name, timeout=5.0)
        if not streams:
            raise RuntimeError(f"No LSL stream named '{self.stream_name}' found")
        self._inlet = StreamInlet(streams[0])

    def stream(self) -> Iterator[tuple[float, dict[str, float]]]:
        assert self._inlet is not None, "call connect() first"
        while True:
            sample, timestamp = self._inlet.pull_sample()
            yield timestamp, dict(zip(self.channels, sample))

    def close(self) -> None:
        self._inlet = None


def wall_clock_pace(sample_iter, fs: float):
    """Wrap a sample iterator to yield in real time (for mock sources that
    would otherwise produce samples faster than realtime)."""
    period = 1.0 / fs
    next_tick = time.monotonic()
    for item in sample_iter:
        yield item
        next_tick += period
        sleep_for = next_tick - time.monotonic()
        if sleep_for > 0:
            time.sleep(sleep_for)
