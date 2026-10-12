"""STAGE TTS 백엔드 추상화.

- ``openai``: gpt-4o-mini-tts. 감정 프리셋 지시문을 ``instructions`` 파라미터로 넘기고 ``speed``를 적용한다.
- ``edge``  : 무료 온라인 TTS + 로컬 운율 후처리. edge-tts(Microsoft 온라인 서비스, ko-KR-SunHiNeural, 키 불필요,
              인터넷 필요)로 합성하고, ``experiments/E3_tts/scripts/prosody.py``가 있으면 그 프리셋과
              parselmouth(PSOLA) 운율 후처리(로컬)를 적용한다. 없으면 rate/pitch/volume 프리셋만 쓴다.

선택 규칙(``TTS_BACKEND`` 환경변수): ``openai`` | ``edge`` | ``auto``(기본).
auto는 ``OPENAI_API_KEY``가 있으면 openai, 없으면 edge.

모든 백엔드는 ``synthesize(text, emotion) -> SynthResult``를 제공한다.
"""
from __future__ import annotations

import asyncio
import importlib.util
import io
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
E3_PROSODY = ROOT / "experiments" / "E3_tts" / "scripts" / "prosody.py"

EDGE_VOICE = os.getenv("EDGE_VOICE", "ko-KR-SunHiNeural")


@dataclass
class SynthResult:
    audio: bytes                 # 인코딩된 오디오 바이트
    fmt: str                     # "mp3" | "wav"
    timings: dict = field(default_factory=dict)  # {"network_ms":..., "dsp_ms":..., "total_ms":...}
    params: dict = field(default_factory=dict)   # 실제 적용된 운율 파라미터(미리보기/로그용)


# ───────────────────────────── OpenAI ─────────────────────────────
class OpenAIBackend:
    name = "openai"
    model = "gpt-4o-mini-tts"

    def __init__(self, api_key: str, presets: dict, response_format: str = "mp3"):
        from openai import OpenAI  # 키가 있을 때만 import
        self.client = OpenAI(api_key=api_key)
        self.presets = presets
        self.response_format = response_format

    def describe(self, emotion: str, intensity: float | None = None) -> dict:
        p = self.presets.get(emotion, self.presets["중립"])
        return {"voice": p["voice"], "speed": p["speed"], "instructions": p["instruction"]}

    def synthesize(self, text: str, emotion: str, intensity: float | None = None) -> SynthResult:
        # intensity(감정 강도 k)는 edge 백엔드의 운율 스케일에만 쓴다. OpenAI는 지시문 그대로(미지원).
        p = self.presets.get(emotion, self.presets["중립"])
        t0 = time.perf_counter()
        resp = self.client.audio.speech.create(
            model=self.model,
            voice=p["voice"],
            input=text,                       # 낭독 텍스트만
            instructions=p["instruction"],    # 감정 지시문은 instructions로 (졸업논문 구현의 누락 수정)
            speed=float(p["speed"]),
            response_format=self.response_format,
        )
        audio = resp.read() if hasattr(resp, "read") else resp.content
        net = (time.perf_counter() - t0) * 1000
        return SynthResult(audio, self.response_format,
                           {"network_ms": net, "dsp_ms": 0.0, "total_ms": net},
                           self.describe(emotion))


# ───────────────────────────── edge-tts ─────────────────────────────
# 폴백 프리셋(E3 prosody.py가 없을 때). OpenAI 프리셋의 speed 방향을 따르고, 피치·음량을 더했다.
EDGE_PRESETS = {
    "행복":   {"rate": "+15%", "pitch": "+25Hz", "volume": "+10%"},
    "슬픔":   {"rate": "-15%", "pitch": "-20Hz", "volume": "-15%"},
    "분노":   {"rate": "+10%", "pitch": "-10Hz", "volume": "+20%"},
    "놀람":   {"rate": "+20%", "pitch": "+40Hz", "volume": "+15%"},
    "두려움": {"rate": "-8%",  "pitch": "+10Hz", "volume": "-20%"},
    "중립":   {"rate": "+0%",  "pitch": "+0Hz",  "volume": "+0%"},
}


async def _edge_stream(text: str, voice: str, rate: str, pitch: str, volume: str) -> bytes:
    import edge_tts
    comm = edge_tts.Communicate(text, voice, rate=rate, pitch=pitch, volume=volume)
    buf = io.BytesIO()
    async for chunk in comm.stream():
        if chunk.get("type") == "audio":
            buf.write(chunk["data"])
    return buf.getvalue()


def edge_tts_bytes(text: str, voice: str = EDGE_VOICE, rate="+0%", pitch="+0Hz", volume="+0%",
                   retries: int = 2) -> bytes:
    last = None
    for _ in range(retries + 1):
        try:
            return asyncio.run(_edge_stream(text, voice, rate, pitch, volume))
        except Exception as e:  # 네트워크 일시 오류 재시도
            last = e
            time.sleep(0.3)
    raise RuntimeError(f"edge-tts 실패: {last}")


def _load_e3_prosody():
    """E3 prosody 모듈을 불러온다. 없거나 import 실패 시 (None, 사유)."""
    if os.getenv("FACEVOX_DISABLE_E3") == "1":
        return None, "disabled by FACEVOX_DISABLE_E3"
    if not E3_PROSODY.exists():
        return None, f"not found: {E3_PROSODY}"
    try:
        scripts_dir = str(E3_PROSODY.parent)
        if scripts_dir not in sys.path:
            sys.path.insert(0, scripts_dir)
        spec = importlib.util.spec_from_file_location("facevox_e3_prosody", E3_PROSODY)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        if not callable(getattr(mod, "synthesize", None)):
            return None, "prosody.py has no synthesize()"
        return mod, "ok"
    except Exception as e:
        return None, f"import error: {e!r}"


def _scale_edge_preset(p: dict, k: float) -> dict:
    """폴백 프리셋 강도 스케일(중립 대비 선형)."""
    def sc(v, unit):
        return f"{int(round(k * float(v.replace(unit, '')))):+d}{unit}"
    return {"rate": sc(p["rate"], "%"), "pitch": sc(p["pitch"], "Hz"), "volume": sc(p["volume"], "%")}


class EdgeBackend:
    name = "edge"

    def __init__(self, voice: str = EDGE_VOICE, use_e3: bool = True):
        self.voice = voice
        self.prosody, self.prosody_status = (_load_e3_prosody() if use_e3 else (None, "off"))
        # C: SSML 프리셋만, D: 프리셋 + PSOLA(논문 주 조건), DP: 중립 edge-tts + PSOLA로 모든 운율 변화(화자 보존형)
        self.prosody_mode = os.getenv("FACEVOX_PROSODY_MODE", "D").upper()
        if self.prosody_mode not in ("C", "D", "DP"):
            self.prosody_mode = "D"
        self.default_intensity = float(os.getenv("FACEVOX_INTENSITY", "1.0"))
        self.mode = f"edge+prosody(E3,{self.prosody_mode})" if self.prosody else "edge-presets"

    # E3 프리셋 해석 ---------------------------------------------------------
    def _e3_params(self, emotion: str, k: float):
        pm = self.prosody
        emo = pm.norm_emotion(emotion)
        if self.prosody_mode in ("C", "D"):
            cp = dict(pm.C_PRESETS[emo])
            if k != 1.0 and hasattr(pm, "scale_c_preset"):
                cp = pm.scale_c_preset(cp, k)
        else:
            cp = dict(pm.C_PRESETS["neutral"])
        dp = None
        if self.prosody_mode in ("D", "DP"):
            table = pm.D_PRESETS if self.prosody_mode == "D" else getattr(pm, "DP_PRESETS", pm.D_PRESETS)
            dp = table[emo]
            if k != 1.0 and hasattr(pm, "scale_dsp_params"):
                dp = pm.scale_dsp_params(dp, k)
        return cp, dp

    def describe(self, emotion: str, intensity: float | None = None) -> dict:
        k = self.default_intensity if intensity is None else float(intensity)
        d = {"voice": self.voice, "mode": self.mode, "intensity": k}
        if self.prosody is not None and hasattr(self.prosody, "C_PRESETS"):
            try:
                cp, dp = self._e3_params(emotion, k)
                d.update(cp)
                if dp is not None:
                    d.update({"f0_range": round(float(dp["f0_range"]), 3), "f0_shift": round(float(dp["f0_shift"]), 2),
                              "tempo": round(float(dp["tempo"]), 3)})
                return d
            except Exception:
                pass
        d.update(_scale_edge_preset(EDGE_PRESETS.get(emotion, EDGE_PRESETS["중립"]), k))
        return d

    # E3 prosody 어댑터 ---------------------------------------------------
    def _synth_e3(self, text: str, emotion: str, k: float) -> SynthResult:
        pm = self.prosody
        staged = all(hasattr(pm, a) for a in ("edge_tts_mp3", "mp3_to_array", "apply_dsp",
                                               "C_PRESETS", "D_PRESETS", "norm_emotion"))
        if staged:
            # 단계별 시간 분리: 네트워크(edge-tts) / 디코딩 / DSP(PSOLA) / WAV 인코딩
            import soundfile as sf
            cp, dp = self._e3_params(emotion, k)
            sr = int(getattr(pm, "SR", 24000))
            t0 = time.perf_counter()
            mp3 = pm.edge_tts_mp3(text, cp["rate"], cp["pitch"], cp["volume"])
            t1 = time.perf_counter()
            y = pm.mp3_to_array(mp3)
            t2 = time.perf_counter()
            if dp is not None:
                y = pm.apply_dsp(y, sr, dp)
            t3 = time.perf_counter()
            buf = io.BytesIO()
            sf.write(buf, y, sr, subtype="PCM_16", format="WAV")
            t4 = time.perf_counter()
            timings = {"network_ms": (t1 - t0) * 1000, "decode_ms": (t2 - t1) * 1000,
                       "dsp_ms": (t3 - t2) * 1000, "encode_ms": (t4 - t3) * 1000,
                       "total_ms": (t4 - t0) * 1000}
            return SynthResult(buf.getvalue(), "wav", timings, self.describe(emotion, k))
        # 일반 경로: synthesize()가 경로/바이트를 돌려주는 경우
        t0 = time.perf_counter()
        try:
            out = pm.synthesize(text, emotion, mode=self.prosody_mode, intensity=k)
        except TypeError:
            out = pm.synthesize(text, emotion)
        total = (time.perf_counter() - t0) * 1000
        if isinstance(out, (str, Path)):
            data = Path(out).read_bytes()
            try:
                os.remove(out)
            except OSError:
                pass
        else:
            data = bytes(out)
        fmt = "wav" if data[:4] == b"RIFF" else "mp3"
        return SynthResult(data, fmt, {"total_ms": total}, self.describe(emotion, k))

    def synthesize(self, text: str, emotion: str, intensity: float | None = None) -> SynthResult:
        k = self.default_intensity if intensity is None else float(intensity)
        if self.prosody is not None:
            return self._synth_e3(text, emotion, k)
        p = _scale_edge_preset(EDGE_PRESETS.get(emotion, EDGE_PRESETS["중립"]), k)
        t0 = time.perf_counter()
        audio = edge_tts_bytes(text, self.voice, p["rate"], p["pitch"], p["volume"])
        net = (time.perf_counter() - t0) * 1000
        return SynthResult(audio, "mp3", {"network_ms": net, "dsp_ms": 0.0, "total_ms": net},
                           self.describe(emotion, k))


# ───────────────────────────── 선택 ─────────────────────────────
def make_backend(presets: dict, response_format: str = "mp3"):
    choice = (os.getenv("TTS_BACKEND", "auto") or "auto").strip().lower()
    key = os.getenv("OPENAI_API_KEY", "").strip()
    if choice == "auto":
        choice = "openai" if key else "edge"
    if choice == "openai":
        if not key:
            raise RuntimeError("TTS_BACKEND=openai 인데 OPENAI_API_KEY가 없습니다.")
        return OpenAIBackend(key, presets, response_format)
    if choice == "edge":
        return EdgeBackend()
    raise ValueError(f"알 수 없는 TTS_BACKEND: {choice}")
