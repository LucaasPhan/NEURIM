import base64
import io
from types import SimpleNamespace

import pytest
from PIL import Image

from src.generator.image_finalize import ImageFinalizer


def _png(size, color="green"):
    buffer = io.BytesIO()
    Image.new("RGB", size, color).save(buffer, format="PNG")
    return buffer.getvalue()


def _png_b64(size, color="blue"):
    return base64.b64encode(_png(size, color)).decode("ascii")


class FakeImages:
    def __init__(self, b64):
        self.b64 = b64
        self.calls = []

    def edit(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(data=[SimpleNamespace(b64_json=self.b64)])


class FakeOpenAI:
    def __init__(self, b64):
        self.images = FakeImages(b64)


def test_finalize_edits_input_and_returns_frame_sized_png():
    client = FakeOpenAI(_png_b64((1024, 1024)))
    finalizer = ImageFinalizer(
        model="gpt-image-2", size="1024x1024", frame_size=512, client=client
    )

    result = finalizer.finalize(_png((64, 64)), "a happy orange cat")

    assert Image.open(io.BytesIO(result)).size == (512, 512)
    call = client.images.calls[0]
    assert call["model"] == "gpt-image-2"
    assert call["size"] == "1024x1024"
    assert "a happy orange cat" in call["prompt"]
    # The morphed frame itself is fed back in as the image to edit.
    assert call["image"].getvalue() == _png((64, 64))


def test_finalize_defaults_subject_when_prompt_blank():
    client = FakeOpenAI(_png_b64((512, 512)))
    finalizer = ImageFinalizer(size="512x512", frame_size=512, client=client)

    finalizer.finalize(_png((512, 512)), "   ")

    assert "the subject" in client.images.calls[0]["prompt"]


def test_finalize_raises_when_response_has_no_image():
    finalizer = ImageFinalizer(frame_size=512, client=FakeOpenAI(None))

    with pytest.raises(RuntimeError):
        finalizer.finalize(_png((64, 64)), "cat")
