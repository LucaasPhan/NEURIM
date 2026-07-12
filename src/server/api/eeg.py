"""API-owned EMOTIV connection and calibration lifecycle."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from typing import Any

from src.common.config import Config, emotiv_credentials
from src.signal_service.baseline import calibrate_baseline
from src.signal_service.eeg_sources import EmotivCortexSource
from src.signal_service.service import FAARewardSource, build_faa_service


SourceFactory = Callable[[], EmotivCortexSource]
Calibrator = Callable[[Any, Any, float], Any]


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(value: datetime | None) -> str | None:
    return value.isoformat() if value is not None else None


def _log(message: str) -> None:
    print(f"[eeg] {message}", flush=True)


class EEGConnectionManager:
    """Owns the real EEG source while the API process is alive."""

    def __init__(
        self,
        config: Config | None = None,
        source_factory: SourceFactory | None = None,
        calibrator: Calibrator = calibrate_baseline,
        calibration_seconds: float = 30.0,
        retry_interval_s: float = 60.0,
    ) -> None:
        self.config = config or Config.load()
        self.source_factory = source_factory or self._default_source
        self.calibrator = calibrator
        self.calibration_seconds = calibration_seconds
        self.retry_interval_s = retry_interval_s
        self._lock = threading.RLock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._source: EmotivCortexSource | None = None
        self._reward_source: FAARewardSource | None = None
        self._state = "disconnected"
        self._last_error: str | None = None
        self._last_connected_at: datetime | None = None
        self._last_calibrated_at: datetime | None = None
        self._next_retry_at: datetime | None = None
        self._last_health_check_monotonic = 0.0
        self._health_check_interval_s = 5.0
        self._calibrate_requested = threading.Event()
        self._rescan_requested = threading.Event()
        self._mode = "real"
        self._device: dict[str, Any] | None = None
        self._quality: dict[str, Any] = self._empty_quality()
        self._quality_valid_since: float | None = None

    def start(self) -> None:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop.clear()
            self._wake.set()
            self._thread = threading.Thread(target=self._run, name="neurim-eeg-connector", daemon=True)
            self._thread.start()
            _log("connector started")

    def close(self) -> None:
        _log("connector stopping")
        self._stop.set()
        self._wake.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=5)
        with self._lock:
            self._close_source_locked()
            self._state = "disconnected"
        _log("disconnected")

    def retry_now(self) -> dict[str, Any]:
        with self._lock:
            if self._state in {"device_validation", "fitting", "ready_to_calibrate"}:
                self._rescan_requested.set()
                self._wake.set()
                _log("headset rescan requested")
            elif self._state not in {"scanning", "calibrating", "ready"}:
                self._state = "disconnected"
                self._next_retry_at = _utc_now()
                self._wake.set()
                _log("manual retry requested")
            else:
                _log(f"manual retry ignored; state={self._state}")
        return self.status()

    def set_demo_mode(self, enabled: bool) -> dict[str, Any]:
        with self._lock:
            if self._state in {"calibrating", "ready"}:
                raise RuntimeError("Demo mode cannot change after calibration has started")
            if enabled:
                if not self._device or not self._device.get("is_virtual"):
                    raise RuntimeError("Demo mode requires a connected Cortex virtual headset")
                self._mode = "demo"
                if self._state == "device_validation":
                    self._state = "fitting"
            else:
                self._mode = "real"
                self._calibrate_requested.clear()
                if self._device and self._device.get("is_virtual"):
                    self._state = "device_validation"
                    self._quality_valid_since = None
            self._wake.set()
        return self.status()

    def start_calibration(self) -> dict[str, Any]:
        with self._lock:
            if not self._can_calibrate_locked():
                reasons = self._quality.get("blocking_reasons") or ["Fitting conditions are not ready"]
                raise RuntimeError("Cannot start calibration: " + "; ".join(map(str, reasons)))
            self._calibrate_requested.set()
            self._wake.set()
        return self.status()

    def status(self) -> dict[str, Any]:
        self._refresh_connection_health()
        with self._lock:
            return {
                "state": self._state,
                "connected": self._source is not None and self._state in {
                    "device_validation", "fitting", "ready_to_calibrate", "calibrating", "ready"
                },
                "calibrated": self._reward_source is not None and self._state == "ready",
                "calibration_seconds": self.calibration_seconds,
                "last_error": self._last_error,
                "last_connected_at": _iso(self._last_connected_at),
                "last_calibrated_at": _iso(self._last_calibrated_at),
                "next_retry_at": _iso(self._next_retry_at),
                "mode": self._mode,
                "device": dict(self._device) if self._device else None,
                "quality": self._quality_status_locked(),
                "can_calibrate": self._can_calibrate_locked(),
            }

    def require_ready_reward_source(self) -> FAARewardSource:
        with self._lock:
            if self._state != "ready" or self._reward_source is None:
                raise RuntimeError("EEG is not ready")
            return self._reward_source

    def _run(self) -> None:
        while not self._stop.is_set():
            self._wake.wait(timeout=self._seconds_until_retry())
            self._wake.clear()
            if self._stop.is_set():
                break
            if not self._retry_due():
                continue
            with self._lock:
                if self._state in {
                    "scanning", "device_validation", "fitting", "ready_to_calibrate",
                    "calibrating", "ready",
                }:
                    continue
            self._connect_and_fit()

    def _connect_and_fit(self) -> None:
        source: EmotivCortexSource | None = None
        try:
            with self._lock:
                self._state = "scanning"
                self._last_error = None
                self._next_retry_at = None
                self._device = None
                self._quality = self._empty_quality()
                self._quality_valid_since = None
                self._calibrate_requested.clear()
                self._rescan_requested.clear()
                self._close_source_locked()
            _log("scanning Cortex headsets")
            source = self.source_factory()
            source.connect()
            metadata = self._source_device_metadata(source)
            with self._lock:
                self._source = source
                self._device = metadata
                self._last_connected_at = _utc_now()
                if metadata["classification"] == "physical_epoc_x":
                    self._state = "fitting"
                else:
                    self._state = "device_validation"
            if metadata["classification"] == "physical_epoc_x":
                _log(f"physical EPOC X validated: {metadata['id']}")
            elif metadata["is_virtual"]:
                _log(f"virtual Brainwear detected: {metadata['id']}; waiting for demo mode")
            else:
                _log(f"unsupported physical headset detected: {metadata['model']}")

            while not self._stop.is_set():
                if self._rescan_requested.is_set():
                    source.close()
                    with self._lock:
                        if self._source is source:
                            self._source = None
                            self._reward_source = None
                        self._state = "disconnected"
                        self._next_retry_at = _utc_now()
                    self._wake.set()
                    return
                with self._lock:
                    unsupported = self._device and self._device["classification"] == "unsupported"
                    virtual_blocked = (
                        self._device and self._device["is_virtual"] and self._mode != "demo"
                    )
                if unsupported:
                    self._stop.wait(0.5)
                    continue
                quality = self._read_quality(source)
                self._update_quality(quality)
                with self._lock:
                    if virtual_blocked and self._mode != "demo":
                        self._state = "device_validation"
                    elif self._can_calibrate_locked():
                        self._state = "ready_to_calibrate"
                    else:
                        self._state = "fitting"
                    requested = self._calibrate_requested.is_set() and self._can_calibrate_locked()
                if requested:
                    break

            if self._stop.is_set():
                return
            signal_service = build_faa_service(self.config, source)
            reward_source = signal_service.reward_source
            with self._lock:
                self._reward_source = reward_source  # type: ignore[assignment]
                self._state = "calibrating"
            _log(f"calibrating baseline for {self.calibration_seconds:.1f}s")
            self.calibrator(reward_source.computer, source.stream(), self.calibration_seconds)
            with self._lock:
                self._state = "ready"
                self._last_calibrated_at = _utc_now()
                self._next_retry_at = None
            _log(f"{'simulation' if self._mode == 'demo' else 'physical EPOC X'} ready")
        except Exception as exc:  # noqa: BLE001
            if source is not None:
                try:
                    source.close()
                except Exception:
                    pass
            with self._lock:
                self._source = None
                self._reward_source = None
                self._state = "error"
                self._last_error = str(exc)
                self._next_retry_at = _utc_now() + timedelta(seconds=self.retry_interval_s)
                retry_at = self._next_retry_at
            _log(f"EPOC X connection failed: {exc}; retry at {_iso(retry_at)}")

    # Compatibility entrypoint retained for focused tests and older imports.
    def _connect_and_calibrate(self) -> None:
        self._connect_and_fit()

    def _seconds_until_retry(self) -> float:
        with self._lock:
            if self._next_retry_at is None:
                return self.retry_interval_s
            return max(0.0, (self._next_retry_at - _utc_now()).total_seconds())

    def _retry_due(self) -> bool:
        with self._lock:
            return self._next_retry_at is None or _utc_now() >= self._next_retry_at

    def _close_source_locked(self) -> None:
        if self._source is not None:
            try:
                self._source.close()
            except Exception:
                pass
        self._source = None
        self._reward_source = None

    def _refresh_connection_health(self) -> None:
        with self._lock:
            if self._state not in {
                "device_validation", "fitting", "ready_to_calibrate", "calibrating", "ready"
            } or self._source is None:
                return
            now = time.monotonic()
            if now - self._last_health_check_monotonic < self._health_check_interval_s:
                return
            self._last_health_check_monotonic = now
            source = self._source
            if not hasattr(source, "is_headset_connected"):
                return
        try:
            connected = source.is_headset_connected()
        except Exception as exc:  # noqa: BLE001
            connected = False
            error = str(exc)
        else:
            error = "Cortex no longer reports the headset as connected"
        if connected:
            return
        with self._lock:
            if self._source is source:
                self._close_source_locked()
                self._state = "error"
                self._last_error = error
                self._next_retry_at = _utc_now() + timedelta(seconds=self.retry_interval_s)
        _log(f"EPOC X connection lost: {error}")

    def _source_device_metadata(self, source: EmotivCortexSource) -> dict[str, Any]:
        if hasattr(source, "headset_info"):
            headset = source.headset_info()
            classification = EmotivCortexSource.classify_headset(headset)
            model = EmotivCortexSource.headset_model(headset)
        else:
            headset = {
                "id": "EPOCX-TEST",
                "customName": "",
                "isVirtual": False,
                "connectedBy": "dongle",
                "sensors": list(self.config.eeg.channels),
            }
            classification = "physical_epoc_x"
            model = "EPOC X"
        return {
            "id": str(headset.get("id", "")),
            "name": str(headset.get("customName", "") or ""),
            "model": model,
            "is_virtual": bool(headset.get("isVirtual")),
            "connected_by": headset.get("connectedBy"),
            "battery_percent": None,
            "signal": None,
            "sensors": list(headset.get("sensors") or []),
            "classification": classification,
        }

    def _read_quality(self, source: EmotivCortexSource) -> dict[str, Any]:
        if hasattr(source, "read_device_quality"):
            return source.read_device_quality()
        channels = self.config.eeg.channels or list(self._required_faa_channels())
        self._stop.wait(0.05)
        return {
            "battery_percent": 100,
            "signal": 1.0,
            "overall": 100.0,
            "sensors": {channel: 4.0 for channel in channels},
        }

    def _required_faa_channels(self) -> set[str]:
        return {channel for pair in self.config.faa.channel_pairs for channel in pair}

    def _update_quality(self, quality: dict[str, Any]) -> None:
        signal = quality.get("signal")
        overall = quality.get("overall")
        sensors = {str(k): float(v) for k, v in (quality.get("sensors") or {}).items()}
        reasons: list[str] = []
        if signal is None or float(signal) < self.config.eeg.min_wireless_signal:
            reasons.append("Wireless signal is below the required level")
        if overall is None or float(overall) < self.config.eeg.min_overall_contact_quality:
            reasons.append("Overall contact quality is below the required level")
        poor = sorted(
            channel for channel in self._required_faa_channels()
            if sensors.get(channel, -1) < self.config.eeg.min_sensor_contact_quality
        )
        if poor:
            reasons.append("Adjust required sensors: " + ", ".join(poor))
        now = time.monotonic()
        with self._lock:
            if reasons:
                self._quality_valid_since = None
            elif self._quality_valid_since is None:
                self._quality_valid_since = now
            battery = quality.get("battery_percent")
            warnings: list[str] = []
            if battery is not None and float(battery) < self.config.eeg.low_battery_percent:
                warnings.append("Headset battery is below 10%")
            non_required_poor = sorted(
                channel for channel, value in sensors.items()
                if channel not in self._required_faa_channels()
                and value < self.config.eeg.min_sensor_contact_quality
            )
            if non_required_poor:
                warnings.append("Low contact on non-FAA sensors: " + ", ".join(non_required_poor))
            self._quality = {
                "overall": float(overall) if overall is not None else None,
                "signal": float(signal) if signal is not None else None,
                "sensors": sensors,
                "blocking_reasons": reasons,
                "warnings": warnings,
            }
            if self._device:
                self._device["battery_percent"] = battery
                self._device["signal"] = signal

    def _quality_status_locked(self) -> dict[str, Any]:
        quality = dict(self._quality)
        stable = 0.0
        if self._quality_valid_since is not None:
            stable = max(0.0, time.monotonic() - self._quality_valid_since)
        quality["stable_seconds"] = stable
        quality["required_stable_seconds"] = self.config.eeg.quality_stable_seconds
        return quality

    def _can_calibrate_locked(self) -> bool:
        if not self._device or self._device.get("classification") == "unsupported":
            return False
        if self._device.get("is_virtual") and self._mode != "demo":
            return False
        if self._quality.get("blocking_reasons"):
            return False
        if self._quality_valid_since is None:
            return False
        return time.monotonic() - self._quality_valid_since >= self.config.eeg.quality_stable_seconds

    @staticmethod
    def _empty_quality() -> dict[str, Any]:
        return {
            "overall": None,
            "signal": None,
            "sensors": {},
            "blocking_reasons": ["Waiting for device quality data"],
            "warnings": [],
        }

    @staticmethod
    def _default_source() -> EmotivCortexSource:
        client_id, client_secret = emotiv_credentials()
        return EmotivCortexSource(client_id, client_secret)
