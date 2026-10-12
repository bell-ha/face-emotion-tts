"""Flask 엔드포인트 (edge 백엔드 합성은 mock)."""
import io
import json
import math
import struct
import wave

import pytest


def _wav_bytes(sec=0.3, sr=24000, freq=220.0):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
        frames = b"".join(struct.pack("<h", int(8000 * math.sin(2 * math.pi * freq * i / sr)))
                          for i in range(int(sec * sr)))
        w.writeframes(frames)
    return buf.getvalue()


@pytest.fixture()
def client(fresh, monkeypatch):
    from backends import SynthResult
    calls = []

    def fake_synth(text, emotion):
        calls.append((text, emotion))
        return SynthResult(_wav_bytes(), "wav", {"network_ms": 1.0, "dsp_ms": 0.0, "total_ms": 1.0},
                           fresh.BACKEND.describe(emotion))

    monkeypatch.setattr(fresh.BACKEND, "synthesize", fake_synth)
    c = fresh.app.test_client()
    c.calls = calls
    return c


def test_backend_is_edge_without_key(appmod):
    assert appmod.BACKEND.name == "edge"
    assert appmod.USE_REWRITE is False
    assert appmod.rewrite_with_instruction("그대로.", "지시") == "그대로."


def test_index(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "음성 만들기".encode() in r.data


def test_get_config(client):
    j = client.get("/get_config").get_json()
    assert j["ok"] is True and j["server"] == "openaiTTS" and j["backend"] == "edge"


def test_set_emotion_and_history(client):
    r = client.post("/set_emotion", json={"t": 1000, "label": "Sad", "tag": "", "confidence": 0.8})
    assert r.get_json() == {"status": "ok"}
    client.post("/set_emotion", json={"t": 2000, "label": "Disgust", "confidence": 0.9})
    ev = client.get("/debug/emo_history?limit=10").get_json()["events"]
    assert [e["emo"] for e in ev] == ["슬픔", "분노"]
    assert ev[0] == {"t": 1000, "emo": "슬픔", "label": "Sad", "tag": "", "confidence": 0.8}
    assert len(client.get("/debug/emo_history?limit=1").get_json()["events"]) == 1


def test_resolve_one(client):
    client.post("/set_emotion", json={"t": 0, "label": "Happy", "confidence": 0.9})
    r = client.post("/emotion/resolve_one", json={"since_ms": 500, "until_ms": 2500})
    assert r.get_json() == {"tag": "<행복>"}
    # 잘못된 입력
    assert client.post("/emotion/resolve_one", json={"since_ms": "x"}).status_code == 400
    assert client.post("/emotion/resolve_one", json={"since_ms": 10, "until_ms": 5}).get_json() == {"tag": ""}


def test_debug_segments(client):
    j = client.post("/debug/segments", json={"text": "좋아.<행복> 그래?", "emotion": "슬픔"}).get_json()
    segs = j["segments"]
    assert [s["emotion"] for s in segs] == ["행복", "슬픔"]
    assert segs[0]["tagged"] is True and segs[1]["tagged"] is False
    assert segs[0]["perf_text"] == "좋아."
    for k in ("text", "emotion", "voice", "speed", "perf_text"):
        assert k in segs[0]
    assert client.post("/debug/segments", json={}).get_json()["segments"] == []


def test_tts_multi_sentence(client, appmod):
    r = client.post("/tts", json={"text": "하나.<행복> 둘.<슬픔> 셋", "emotion": "놀람"})
    assert r.status_code == 200
    assert r.mimetype == appmod.MIMETYPE
    assert len(r.data) > 1000
    assert client.calls == [("하나.", "행복"), ("둘.", "슬픔"), ("셋", "놀람")]
    timing = json.loads(r.headers["X-STAGE-Timing"])
    assert len(timing["segments"]) == 3
    # 0.3초 × 3 + 쉼 0.25초 × 3
    assert abs(timing["audio_sec"] - 1.65) < 0.05


def test_tts_single(client):
    r = client.post("/tts", json={"text": "태그 없는 문장", "style": "분노"})
    assert r.status_code == 200
    assert client.calls == [("태그 없는 문장", "분노")]


def test_tts_empty(client):
    r = client.post("/tts", json={"text": "  "})
    assert r.status_code == 400 and "error" in r.get_json()


def test_tts_backend_error_is_502(client, fresh, monkeypatch):
    def boom(text, emotion):
        raise RuntimeError("edge-tts 실패: network unreachable")
    monkeypatch.setattr(fresh.BACKEND, "synthesize", boom)
    r = client.post("/tts", json={"text": "a.<행복>"})
    assert r.status_code == 502


def test_tts_intensity_passed(fresh, monkeypatch):
    from backends import SynthResult
    seen = []

    def fake(text, emotion, intensity=None):
        seen.append(intensity)
        return SynthResult(_wav_bytes(), "wav", {}, {})
    monkeypatch.setattr(fresh.BACKEND, "synthesize", fake)
    c = fresh.app.test_client()
    assert c.post("/tts", json={"text": "a.<행복> b.<슬픔>", "intensity": 0.5}).status_code == 200
    assert c.post("/tts", json={"text": "c", "intensity": 9}).status_code == 200
    assert seen == [0.5, 0.5, 1.5]


def test_describe_intensity_zero_is_neutral(appmod):
    d = appmod.BACKEND.describe("행복", 0.0)
    assert d["rate"] == "+0%" and d["pitch"] == "+0Hz" and d["volume"] == "+0%"
