import json

import pytest

from src.generator.anchor_session import load_prompt_session_manifest
from src.generator.prompt_curation import DEFAULT_ANCHOR_COUNT


def _manifest_payload():
    labels = [f"axis_{i}" for i in range(DEFAULT_ANCHOR_COUNT)]
    return {
        "version": 1,
        "user_prompt": "red sneakers",
        "anchor_count": DEFAULT_ANCHOR_COUNT,
        "scaffold": "centered product photograph, neutral backdrop, soft studio light",
        "prompt_template": "centered product photograph of {anchor}, neutral backdrop, soft studio light",
        "anchor_labels": labels,
        "realized_prompts": [
            f"centered product photograph of {label}, neutral backdrop, soft studio light"
            for label in labels
        ],
        "notes": "Keep all anchors compatible with the same subject.",
        "model": {"provider": "openai", "name": "gpt-test"},
    }


def test_load_prompt_session_manifest_accepts_valid_file(tmp_path):
    path = tmp_path / "session.json"
    path.write_text(json.dumps(_manifest_payload()), encoding="utf-8")

    manifest = load_prompt_session_manifest(path)

    assert manifest.user_prompt == "red sneakers"
    assert manifest.anchor_count == DEFAULT_ANCHOR_COUNT
    assert manifest.anchor_labels[0] == "axis_0"


def test_load_prompt_session_manifest_rejects_missing_file(tmp_path):
    with pytest.raises(RuntimeError, match="does not exist"):
        load_prompt_session_manifest(tmp_path / "missing.json")


def test_load_prompt_session_manifest_rejects_malformed_json(tmp_path):
    path = tmp_path / "session.json"
    path.write_text('{"version": 1,', encoding="utf-8")

    with pytest.raises(RuntimeError, match="malformed JSON"):
        load_prompt_session_manifest(path)


def test_load_prompt_session_manifest_rejects_wrong_anchor_count(tmp_path):
    payload = _manifest_payload()
    payload["anchor_count"] = 6
    path = tmp_path / "session.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(RuntimeError, match="anchor_count must be 7"):
        load_prompt_session_manifest(path)


def test_load_prompt_session_manifest_rejects_wrong_anchor_label_count(tmp_path):
    payload = _manifest_payload()
    payload["anchor_labels"] = payload["anchor_labels"][:-1]
    path = tmp_path / "session.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(RuntimeError, match="anchor_labels must contain exactly 7 items"):
        load_prompt_session_manifest(path)


def test_load_prompt_session_manifest_rejects_wrong_realized_prompt_count(tmp_path):
    payload = _manifest_payload()
    payload["realized_prompts"] = payload["realized_prompts"][:-1]
    path = tmp_path / "session.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(RuntimeError, match="realized_prompts must contain exactly 7 items"):
        load_prompt_session_manifest(path)
