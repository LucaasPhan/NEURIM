import io

import numpy as np
from PIL import Image

from src.common.config import Config
from src.session.diffusion_client import DiffusionClient
from src.session.frame_store import FrameStore
from src.session.optimizer_loop import OptimizerRenderLoop


def _png_bytes():
    buffer = io.BytesIO()
    Image.new("RGB", (4, 4), "green").save(buffer, format="PNG")
    return buffer.getvalue()


class FakeResponse:
    def __init__(self, status_code=200, content=b"", payload=None, text=""):
        self.status_code = status_code
        self.content = content
        self._payload = payload
        self.text = text

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(self.status_code)

    def json(self):
        return self._payload


class FakeSession:
    def __init__(self):
        self.calls = []

    def get(self, url, timeout):
        self.calls.append(("GET", url, timeout))
        return FakeResponse(payload={"anchor_count": 7})

    def post(self, url, json, timeout):
        self.calls.append(("POST", url, json, timeout))
        return FakeResponse(content=_png_bytes())


def test_diffusion_client_uses_manifest_and_render_contracts():
    session = FakeSession()
    client = DiffusionClient("http://gpu:8766/", timeout=4, session=session)

    assert client.manifest() == {"anchor_count": 7}
    assert client.render(np.array([0.1, 0.2]), 256).startswith(b"\x89PNG")
    assert session.calls == [
        ("GET", "http://gpu:8766/manifest", 4),
        ("POST", "http://gpu:8766/render", {"z": [0.1, 0.2], "frame_size": 256}, 4),
    ]


def test_frame_store_writes_live_and_snapshots(tmp_path):
    store = FrameStore(tmp_path)
    png = _png_bytes()

    store.save_live(png, capture_start=True)
    store.save_live(png, capture_start=True)
    store.save_end(png)

    assert (tmp_path / "live_frame.png").read_bytes() == png
    assert (tmp_path / "session_start.png").read_bytes() == png
    assert (tmp_path / "session_end.png").read_bytes() == png


def test_frame_store_clear_target_is_idempotent(tmp_path):
    store = FrameStore(tmp_path)
    store.save_target(_png_bytes())

    assert (tmp_path / "target_frame.png").exists()
    store.clear_target()
    assert not (tmp_path / "target_frame.png").exists()
    store.clear_target()  # no error when already gone


class FakeRenderClient:
    def __init__(self, png):
        self.png = png

    def render(self, _z, _frame_size):
        return self.png


class FakeFinalizer:
    def __init__(self, raise_exc=False):
        self.calls = []
        self.raise_exc = raise_exc

    def finalize(self, png, subject):
        self.calls.append((png, subject))
        if self.raise_exc:
            raise RuntimeError("finalize boom")
        return b"FINAL:" + png


def _loop(tmp_path, png, finalizer, prompt="a happy cat"):
    return OptimizerRenderLoop(
        Config(),
        frames_per_step=2,
        client=FakeRenderClient(png),
        frame_store=FrameStore(tmp_path),
        capture_snapshots=True,
        finalizer=finalizer,
        finalize_prompt=prompt,
    )


def test_save_final_frame_writes_raw_end_and_finalized_target(tmp_path):
    png = _png_bytes()
    finalizer = FakeFinalizer()

    _loop(tmp_path, png, finalizer).save_final_frame()

    assert (tmp_path / "session_end.png").read_bytes() == png
    assert (tmp_path / "target_frame.png").read_bytes() == b"FINAL:" + png
    assert finalizer.calls == [(png, "a happy cat")]


def test_save_final_frame_falls_back_to_raw_when_finalize_fails(tmp_path):
    png = _png_bytes()

    _loop(tmp_path, png, FakeFinalizer(raise_exc=True)).save_final_frame()

    assert (tmp_path / "session_end.png").read_bytes() == png
    assert (tmp_path / "target_frame.png").read_bytes() == png


def test_save_final_frame_without_finalizer_serves_raw_frame(tmp_path):
    png = _png_bytes()

    _loop(tmp_path, png, finalizer=None).save_final_frame()

    assert (tmp_path / "target_frame.png").read_bytes() == png


def test_save_final_frame_noop_without_snapshots(tmp_path):
    png = _png_bytes()
    loop = OptimizerRenderLoop(
        Config(),
        frames_per_step=2,
        client=FakeRenderClient(png),
        frame_store=FrameStore(tmp_path),
        capture_snapshots=False,
        finalizer=FakeFinalizer(),
    )

    loop.save_final_frame()

    assert not (tmp_path / "target_frame.png").exists()
    assert not (tmp_path / "session_end.png").exists()
