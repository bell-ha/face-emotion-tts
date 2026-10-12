"""백엔드 선택과 OpenAI instructions 전달(실제 API 호출 없음)."""
import sys
import types

import pytest

import backends


def test_auto_selects_edge_without_key(monkeypatch):
    monkeypatch.setenv("TTS_BACKEND", "auto")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("FACEVOX_DISABLE_E3", "1")
    b = backends.make_backend({"중립": {}})
    assert b.name == "edge" and b.mode == "edge-presets"


def test_forced_openai_without_key_raises(monkeypatch):
    monkeypatch.setenv("TTS_BACKEND", "openai")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        backends.make_backend({})


def test_openai_passes_instructions_and_speed(monkeypatch, appmod):
    captured = {}

    class FakeSpeech:
        def create(self, **kw):
            captured.update(kw)
            return types.SimpleNamespace(read=lambda: b"ID3fake")

    class FakeOpenAI:
        def __init__(self, api_key):
            self.audio = types.SimpleNamespace(speech=FakeSpeech())

    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(OpenAI=FakeOpenAI))
    monkeypatch.setenv("TTS_BACKEND", "auto")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    b = backends.make_backend(appmod.EMOTION_PRESETS, "mp3")
    assert b.name == "openai"
    res = b.synthesize("안녕하세요.", "행복")
    assert res.audio == b"ID3fake"
    assert captured["model"] == "gpt-4o-mini-tts"
    assert captured["input"] == "안녕하세요."
    assert captured["instructions"] == appmod.EMOTION_PRESETS["행복"]["instruction"]
    assert captured["speed"] == pytest.approx(1.22)
