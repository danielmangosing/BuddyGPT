"""Unit tests for resilient config loading."""

from __future__ import annotations

import json
import logging

from src import config as config_mod


def test_load_config_skips_invalid_file_and_uses_next_candidate(
    tmp_path, monkeypatch, caplog
):
    bad = tmp_path / "bad_config.json"
    good = tmp_path / "good_config.json"
    bad.write_text("{not-valid-json", encoding="utf-8")
    good.write_text(json.dumps({"model": "custom-model"}), encoding="utf-8")

    monkeypatch.setattr(config_mod, "_load_env_files", lambda: None)
    monkeypatch.setattr(config_mod, "_config_candidates", lambda: [bad, good])

    with caplog.at_level(logging.WARNING, logger="src.config"):
        cfg = config_mod.load_config()

    assert cfg["model"] == "custom-model"
    assert "Ignoring unreadable config file" in caplog.text


def test_load_config_with_only_invalid_candidates_falls_back_to_defaults(
    tmp_path, monkeypatch
):
    bad = tmp_path / "bad_only.json"
    bad.write_text("{bad", encoding="utf-8")

    monkeypatch.setattr(config_mod, "_load_env_files", lambda: None)
    monkeypatch.setattr(config_mod, "_config_candidates", lambda: [bad])

    cfg = config_mod.load_config()
    assert cfg["model"] == config_mod.DEFAULT_CONFIG["model"]
    assert cfg["max_tokens"] == config_mod.DEFAULT_CONFIG["max_tokens"]


def test_load_config_reads_openai_key_from_env_when_empty(monkeypatch):
    monkeypatch.setattr(config_mod, "_load_env_files", lambda: None)
    monkeypatch.setattr(config_mod, "_config_candidates", lambda: [])
    monkeypatch.setenv("OPENAI_API_KEY", "sk-proj-env-test")
    cfg = config_mod.load_config()
    assert cfg["openai_api_key"] == "sk-proj-env-test"
