from datetime import datetime
import threading
import time

from src.common.config import Config
from src.server.api.eeg import EEGConnectionManager


class FakeSource:
    def __init__(self, samples=None, fail=False, virtual=False, quality=None):
        self.samples = samples or [(0.0, {"F3": 1.0, "F4": 1.0}), (31.0, {"F3": 1.0, "F4": 1.0})]
        self.fail = fail
        self.connected = False
        self.closed = False
        self.virtual = virtual
        self.quality = quality or {
            "battery_percent": 80,
            "signal": 1.0,
            "overall": 100,
            "sensors": {channel: 4 for channel in ["AF3", "AF4", "F7", "F8", "F3", "F4", "FC5", "FC6"]},
        }

    def connect(self):
        if self.fail:
            raise RuntimeError("no headset")
        self.connected = True

    def stream(self):
        yield from self.samples

    def headset_info(self):
        return {
            "id": "EPOCX-VIRTUAL" if self.virtual else "EPOCX-PHYSICAL",
            "customName": "test headset",
            "isVirtual": self.virtual,
            "headbandPosition": "top",
            "connectedBy": "dongle",
            "sensors": list(self.quality["sensors"]),
        }

    def read_device_quality(self):
        time.sleep(0.01)
        return self.quality

    def is_headset_connected(self):
        return not self.closed

    def close(self):
        self.closed = True


def test_connect_failure_records_retry_state():
    source = FakeSource(fail=True)
    manager = EEGConnectionManager(
        source_factory=lambda: source,
        calibrator=lambda _computer, _stream, _seconds: None,
        retry_interval_s=60.0,
    )

    manager._connect_and_calibrate()
    status = manager.status()

    assert status["state"] == "error"
    assert status["last_error"] == "no headset"
    assert status["next_retry_at"] is not None


def _manager(source, calibrator=lambda _computer, _stream, _seconds: None):
    config = Config()
    config.eeg.quality_stable_seconds = 0
    return EEGConnectionManager(source_factory=lambda: source, calibrator=calibrator, config=config)


def _wait_for_state(manager, expected, timeout=1.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if manager.status()["state"] == expected:
            return
        time.sleep(0.01)
    raise AssertionError(f"manager did not reach {expected}: {manager.status()}")


def test_successful_connect_waits_for_explicit_30_second_calibration():
    calls = []
    source = FakeSource()

    def calibrator(_computer, stream, seconds):
        calls.append(seconds)
        list(stream)

    manager = _manager(source, calibrator)
    thread = threading.Thread(target=manager._connect_and_fit)
    thread.start()
    _wait_for_state(manager, "ready_to_calibrate")
    assert calls == []
    manager.start_calibration()
    thread.join(timeout=1)
    status = manager.status()

    assert calls == [30.0]
    assert status["state"] == "ready"
    assert status["connected"] is True
    assert status["calibrated"] is True
    assert datetime.fromisoformat(status["last_connected_at"])
    assert datetime.fromisoformat(status["last_calibrated_at"])


def test_connection_lifecycle_prints_status_logs(capsys):
    manager = _manager(FakeSource())
    manager.calibration_seconds = 0.0
    thread = threading.Thread(target=manager._connect_and_fit)
    thread.start()
    _wait_for_state(manager, "ready_to_calibrate")
    manager.start_calibration()
    thread.join(timeout=1)
    output = capsys.readouterr().out

    assert "[eeg] scanning Cortex headsets" in output
    assert "[eeg] physical EPOC X validated" in output
    assert "[eeg] calibrating baseline for 0.0s" in output
    assert "[eeg] physical EPOC X ready" in output


def test_retry_now_marks_retry_due():
    manager = EEGConnectionManager(
        source_factory=lambda: FakeSource(fail=True),
        calibrator=lambda _computer, _stream, _seconds: None,
    )

    status = manager.retry_now()

    assert status["state"] == "disconnected"
    assert status["next_retry_at"] is not None


def test_virtual_headset_requires_explicit_demo_mode():
    manager = _manager(FakeSource(virtual=True))
    thread = threading.Thread(target=manager._connect_and_fit)
    thread.start()
    _wait_for_state(manager, "device_validation")
    status = manager.status()
    assert status["device"]["is_virtual"] is True
    assert status["can_calibrate"] is False

    manager.set_demo_mode(True)
    _wait_for_state(manager, "ready_to_calibrate")
    manager.start_calibration()
    thread.join(timeout=1)
    assert manager.status()["state"] == "ready"
    assert manager.status()["mode"] == "demo"


def test_poor_required_sensor_blocks_calibration():
    quality = {
        "battery_percent": 80,
        "signal": 1.0,
        "overall": 90,
        "sensors": {channel: (1 if channel == "F3" else 4) for channel in ["AF3", "AF4", "F7", "F8", "F3", "F4", "FC5", "FC6"]},
    }
    manager = _manager(FakeSource(quality=quality))
    thread = threading.Thread(target=manager._connect_and_fit, daemon=True)
    thread.start()
    _wait_for_state(manager, "fitting")
    deadline = time.monotonic() + 1.0
    while time.monotonic() < deadline:
        if "F3" in " ".join(manager.status()["quality"]["blocking_reasons"]):
            break
        time.sleep(0.01)
    assert manager.status()["can_calibrate"] is False
    assert "F3" in " ".join(manager.status()["quality"]["blocking_reasons"])
    manager._stop.set()
    thread.join(timeout=1)
