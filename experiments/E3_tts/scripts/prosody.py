"""FaceVox 키 불필요 감정 운율 백엔드 (OpenAI API 키 없이 동작).

주의: edge-tts는 Microsoft의 온라인 서비스이므로 인터넷 연결이 필요하다.
로컬에서 수행되는 것은 parselmouth(Praat) DSP 후처리뿐이다.

조건 C: edge-tts(ko-KR-SunHiNeural) + 감정별 prosody 프리셋(rate/pitch/volume)
조건 D: C + parselmouth(Praat PSOLA) 후처리
        (F0 범위 확대/축소, F0 이동, 윤곽 성형, 떨림(tremor), 템포, 스펙트럼 기울기, 이득)

사용 예:
    from prosody import synthesize, synthesize_bytes
    path = synthesize("안녕하세요", "행복")                 # D, 임시 wav 경로 반환
    wav  = synthesize_bytes("안녕하세요", "행복", mode="Dp", intensity=0.5)  # 화자 보존형, 강도 0.5
    wav  = synthesize_bytes("안녕하세요", "분노", mode="C")  # wav bytes (24 kHz mono PCM16)

감정 인자는 한국어('행복','슬픔','분노','놀람','두려움','중립') 또는
영문 키('happy','sad','angry','surprise','fear','neutral') 모두 허용. 미지의 값은 중립.
의존성: edge-tts(온라인 서비스, 네트워크 필요), praat-parselmouth, numpy, scipy, soundfile, librosa.

프리셋 값은 E3 실험의 dev 문장 4개(S03~S06)로만 조정한 뒤 고정한 값이다
(experiments/E3_tts/results/presets_frozen.json 과 동일해야 함).
"""
from __future__ import annotations

import asyncio
import io
import os
import tempfile
import threading
from pathlib import Path

import numpy as np

VOICE = "ko-KR-SunHiNeural"
SR = 24000

_KO2EN = {"행복": "happy", "슬픔": "sad", "분노": "angry", "놀람": "surprise",
          "두려움": "fear", "중립": "neutral"}

# ───────────── 조건 C: edge-tts SSML prosody 프리셋 ─────────────
# (dev 문장 S03~S06 격자 탐색 후 고정 = results/presets_frozen.json)
C_PRESETS = {
    "happy":    {"rate": "+0%", "pitch": "+65Hz", "volume": "+0%"},
    "sad":      {"rate": "-40%", "pitch": "-45Hz", "volume": "+0%"},
    "angry":    {"rate": "+30%", "pitch": "-5Hz", "volume": "+20%"},
    "surprise": {"rate": "+30%", "pitch": "+90Hz", "volume": "+0%"},
    "fear":     {"rate": "+30%", "pitch": "-35Hz", "volume": "-30%"},
    "neutral":  {"rate": "+0%", "pitch": "+0Hz", "volume": "+0%"},
}

# ───────────── 조건 D: PSOLA 후처리 파라미터 ─────────────
# f0_range : 발화 중앙값 기준 반음(st) 편차 배율 (1=유지, >1 확대, <1 축소)
# f0_shift : 전체 F0 이동(st)
# contour  : ("none"|"onset_rise"|"decline", 진폭 st)
#            onset_rise: 발화 시작 +A st에서 발화 40% 지점까지 선형 감소 → 0
#            decline   : 발화 시작 +A/2 → 끝 -A/2 선형 하강(A>0이면 하강폭 A st)
# tremor   : (주파수 Hz, 깊이 st) 정현파 F0 변조, 깊이 0이면 없음
# tempo    : 말속도 배율(>1 빠르게); PSOLA 길이 조작
# tilt_db  : 1 kHz 이상 대역 이득(dB, 1차 버터워스 고역 성분 가감) — 발성 노력 근사
# gain_db  : 전체 이득(dB); 피크 0.98 초과 시 피크 정규화
# breath_snr: 기식성(whispery) 잡음 혼합 SNR(dB). None이면 없음. 1.5–6 kHz 대역 잡음을
#            음성 진폭 포락선으로 변조해 섞음(시드 42 고정)
D_PRESETS = {
    "happy":    {"f0_range": 1.0, "f0_shift": 0.0, "contour": ("none", 0.0),
                 "tremor": (6.0, 0.0), "tempo": 1.0, "tilt_db": 3.0, "gain_db": 2.0, "breath_snr": None},
    "sad":      {"f0_range": 0.4, "f0_shift": -2.0, "contour": ("decline", 0.0),
                 "tremor": (6.0, 0.0), "tempo": 1.0, "tilt_db": 0.0, "gain_db": -4.0, "breath_snr": None},
    "angry":    {"f0_range": 2.0, "f0_shift": -1.0, "contour": ("none", 0.0),
                 "tremor": (6.0, 0.0), "tempo": 1.1, "tilt_db": 3.0, "gain_db": 4.0, "breath_snr": None},
    "surprise": {"f0_range": 1.4, "f0_shift": 2.0, "contour": ("onset_rise", 6.0),
                 "tremor": (6.0, 0.0), "tempo": 1.0, "tilt_db": 0.0, "gain_db": 2.0, "breath_snr": None},
    "fear":     {"f0_range": 0.6, "f0_shift": 0.0, "contour": ("none", 0.0),
                 "tremor": (6.0, 0.0), "tempo": 1.0, "tilt_db": -3.0, "gain_db": -4.0, "breath_snr": 20.0},
    # 중립: '억양 과장 금지'(졸업논문 서술) 방향으로 범위 축소·약간 낮춤 (dev 튜닝)
    "neutral":  {"f0_range": 0.8, "f0_shift": -1.0, "contour": ("none", 0.0),
                 "tremor": (6.0, 0.0), "tempo": 0.95, "tilt_db": -3.0, "gain_db": 0.0, "breath_snr": None},
}

# ───────────── 조건 Dp (D-psola, E3b): 화자 보존형 ─────────────
# edge-tts 중립 렌더링(SSML 피치·속도 이동 없음) + 모든 운율 변화를 PSOLA로 수행(포먼트 보존).
# f0_shift/tempo/gain_db = dev 문장에서 측정한 C 단계 변화 + D 값 (E3b_intensity/scripts/derive_dpsola.py)
DP_PRESETS = {
    "happy":    {"f0_range": 1.0, "f0_shift": 6.08, "contour": ("none", 0.0),
                 "tremor": (6.0, 0.0), "tempo": 0.998, "tilt_db": 3.0, "gain_db": 2.0, "breath_snr": None},
    "sad":      {"f0_range": 0.4, "f0_shift": -8.12, "contour": ("decline", 0.0),
                 "tremor": (6.0, 0.0), "tempo": 0.609, "tilt_db": 0.0, "gain_db": -3.99, "breath_snr": None},
    "angry":    {"f0_range": 2.0, "f0_shift": -1.59, "contour": ("none", 0.0),
                 "tremor": (6.0, 0.0), "tempo": 1.412, "tilt_db": 3.0, "gain_db": 5.55, "breath_snr": None},
    "surprise": {"f0_range": 1.4, "f0_shift": 8.92, "contour": ("onset_rise", 6.0),
                 "tremor": (6.0, 0.0), "tempo": 1.29, "tilt_db": 0.0, "gain_db": 1.89, "breath_snr": None},
    "fear":     {"f0_range": 0.6, "f0_shift": -4.66, "contour": ("none", 0.0),
                 "tremor": (6.0, 0.0), "tempo": 1.29, "tilt_db": -3.0, "gain_db": -7.18, "breath_snr": 20.0},
    "neutral":  {"f0_range": 0.8, "f0_shift": -1.0, "contour": ("none", 0.0),
                 "tremor": (6.0, 0.0), "tempo": 0.95, "tilt_db": -3.0, "gain_db": 0.0, "breath_snr": None},
}


# ───────────── 감정 강도(intensity) 스케일링 (E3b) ─────────────
# k=0 → 중립(B와 같은 운율), k=1 → 고정 프리셋. 중립 대비 편차를 일관되게 보간:
#   반음·dB·Hz·% 오프셋은 선형(k×), 배율(f0_range, tempo)은 기하(r**k),
#   기식 잡음은 잡음 진폭을 k배(SNR + 20·log10(1/k)), k=0이면 없음.
def scale_c_preset(cp: dict, k: float) -> dict:
    def s(v: str, unit: str) -> str:
        x = float(v.replace(unit, ""))
        return f"{int(round(k * x)):+d}{unit}"
    return {"rate": s(cp["rate"], "%"), "pitch": s(cp["pitch"], "Hz"), "volume": s(cp["volume"], "%")}


def scale_dsp_params(p: dict, k: float) -> dict:
    import math
    kind, amp = p["contour"]
    fr, depth = p["tremor"]
    b = p.get("breath_snr")
    return {"f0_range": p["f0_range"] ** k, "f0_shift": k * p["f0_shift"], "contour": (kind, k * amp),
            "tremor": (fr, k * depth), "tempo": p["tempo"] ** k, "tilt_db": k * p["tilt_db"],
            "gain_db": k * p["gain_db"],
            "breath_snr": None if (b is None or k <= 0) else b - 20 * math.log10(k)}

def norm_emotion(emotion: str | None) -> str:
    e = (emotion or "").strip().strip("<>")
    e = _KO2EN.get(e, e).lower()
    return e if e in C_PRESETS else "neutral"


# ───────────────────────── edge-tts ─────────────────────────
async def _edge_async(text: str, rate: str, pitch: str, volume: str, voice: str) -> bytes:
    import edge_tts
    comm = edge_tts.Communicate(text, voice, rate=rate, pitch=pitch, volume=volume)
    buf = bytearray()
    async for chunk in comm.stream():
        if chunk["type"] == "audio":
            buf.extend(chunk["data"])
    if not buf:
        raise RuntimeError("edge-tts returned no audio")
    return bytes(buf)


def _run(coro):
    """이미 이벤트 루프가 도는 환경(Flask/Jupyter)에서도 동작하도록 별도 스레드에서 실행."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    out = {}

    def _t():
        out["v"] = asyncio.run(coro)
    th = threading.Thread(target=_t)
    th.start()
    th.join()
    return out["v"]


def edge_tts_mp3(text: str, rate="+0%", pitch="+0Hz", volume="+0%", voice=VOICE,
                 retries: int = 3) -> bytes:
    last = None
    for _ in range(retries):
        try:
            return _run(_edge_async(text, rate, pitch, volume, voice))
        except Exception as e:  # 네트워크 일시 오류 재시도
            last = e
    raise last


def mp3_to_array(mp3: bytes, sr: int = SR) -> np.ndarray:
    import librosa
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        f.write(mp3)
        tmp = f.name
    try:
        y, _ = librosa.load(tmp, sr=sr, mono=True)
    finally:
        os.remove(tmp)
    return y.astype(np.float64)


# ───────────────────────── DSP (Praat PSOLA) ─────────────────────────
def _hf_tilt(y: np.ndarray, sr: int, tilt_db: float, fc: float = 1000.0) -> np.ndarray:
    if abs(tilt_db) < 1e-9:
        return y
    from scipy.signal import butter, sosfiltfilt
    sos = butter(1, fc, btype="highpass", fs=sr, output="sos")
    hp = sosfiltfilt(sos, y)
    g = 10 ** (tilt_db / 20.0) - 1.0      # 고역 성분만 g배 가감 (근사 shelving)
    return y + g * hp


def _breath(y: np.ndarray, sr: int, snr_db) -> np.ndarray:
    if snr_db is None:
        return y
    from scipy.signal import butter, hilbert, sosfiltfilt
    rng = np.random.default_rng(42)
    noise = sosfiltfilt(butter(4, [1500, 6000], btype="bandpass", fs=sr, output="sos"),
                        rng.standard_normal(len(y)))
    env = sosfiltfilt(butter(2, 20, btype="lowpass", fs=sr, output="sos"), np.abs(hilbert(y)))
    env = np.clip(env, 0, None)
    n = noise * env
    n *= np.sqrt(np.mean(y ** 2)) / (np.sqrt(np.mean(n ** 2)) + 1e-12) * 10 ** (-snr_db / 20.0)
    return y + n


def apply_dsp(y: np.ndarray, sr: int, p: dict) -> np.ndarray:
    import parselmouth
    from parselmouth.praat import call

    snd = parselmouth.Sound(y, sampling_frequency=sr)
    dur = snd.get_total_duration()
    manip = call(snd, "To Manipulation", 0.01, 75, 600)
    tier = call(manip, "Extract pitch tier")
    n = int(call(tier, "Get number of points"))
    if n > 0:
        ts = np.array([call(tier, "Get time from index", i + 1) for i in range(n)])
        fs = np.array([call(tier, "Get value at index", i + 1) for i in range(n)])
        st = 12 * np.log2(fs / 100.0)
        med = np.median(st)
        st2 = med + p["f0_range"] * (st - med) + p["f0_shift"]
        # 윤곽: 발화 유성 구간 기준 정규화 시간
        t0, t1 = ts.min(), ts.max()
        tn = (ts - t0) / max(t1 - t0, 1e-6)
        kind, amp = p["contour"]
        if kind == "onset_rise" and amp:
            st2 += amp * np.clip(1 - tn / 0.4, 0, 1)
        elif kind == "decline" and amp:
            st2 += amp * (0.5 - tn)
        fr, depth = p["tremor"]
        if depth:
            st2 += depth * np.sin(2 * np.pi * fr * ts)
        f2 = 100.0 * 2 ** (st2 / 12)
        f2 = np.clip(f2, 60, 800)
        new = call("Create PitchTier", "new", 0, dur)
        for t, f in zip(ts, f2):
            call(new, "Add point", float(t), float(f))
        call([manip, new], "Replace pitch tier")
    if abs(p["tempo"] - 1.0) > 1e-9:
        dt = call("Create DurationTier", "dur", 0, dur)
        call(dt, "Add point", 0.0, 1.0 / p["tempo"])
        call([manip, dt], "Replace duration tier")
    out = call(manip, "Get resynthesis (overlap-add)")
    z = out.values.flatten().astype(np.float64)
    z = _hf_tilt(z, sr, p["tilt_db"])
    z = _breath(z, sr, p.get("breath_snr"))
    # 이득: 입력 RMS 기준 상대 이득
    rms_in = np.sqrt(np.mean(y ** 2)) + 1e-12
    rms_out = np.sqrt(np.mean(z ** 2)) + 1e-12
    z = z * (rms_in / rms_out) * 10 ** (p["gain_db"] / 20.0)
    pk = np.max(np.abs(z))
    if pk > 0.98:
        z = z * (0.98 / pk)
    return z


# ───────────────────────── 공개 API ─────────────────────────
def synthesize_array(text: str, emotion: str = "중립", mode: str = "D", intensity: float = 1.0,
                     c_presets: dict | None = None, d_presets: dict | None = None) -> np.ndarray:
    """24 kHz float 배열 반환.
    mode: 'B'(중립 edge-tts), 'C'(SSML 프리셋), 'D'(프리셋+DSP),
          'DP'/'Dp'(D-psola: 중립 edge-tts + PSOLA로 모든 운율 변화, 화자 보존형).
    intensity: 감정 강도 k ∈ [0, 1+] (기본 1.0 = E3 고정 프리셋, 0 = 중립)."""
    emo = norm_emotion(emotion)
    mode = mode.upper()
    k = float(intensity)
    if mode in ("C", "D"):
        cp = (c_presets or C_PRESETS)[emo]
        if k != 1.0:
            cp = scale_c_preset(cp, k)
    else:
        cp = C_PRESETS["neutral"]
    y = mp3_to_array(edge_tts_mp3(text, cp["rate"], cp["pitch"], cp["volume"]))
    if mode in ("D", "DP"):
        base = (d_presets or (D_PRESETS if mode == "D" else DP_PRESETS))[emo]
        y = apply_dsp(y, SR, base if k == 1.0 else scale_dsp_params(base, k))
    return y


def synthesize_bytes(text: str, emotion: str = "중립", mode: str = "D", intensity: float = 1.0) -> bytes:
    """WAV(24 kHz, mono, PCM16) bytes 반환 — Flask send_file(BytesIO(...), 'audio/wav')용."""
    import soundfile as sf
    y = synthesize_array(text, emotion, mode, intensity)
    buf = io.BytesIO()
    sf.write(buf, y, SR, subtype="PCM_16", format="WAV")
    return buf.getvalue()


def synthesize(text: str, emotion: str = "중립", mode: str = "D",
               out_path: str | os.PathLike | None = None, intensity: float = 1.0) -> str:
    """WAV 파일로 저장하고 경로를 반환. out_path 생략 시 임시 파일."""
    data = synthesize_bytes(text, emotion, mode, intensity)
    if out_path is None:
        fd, out_path = tempfile.mkstemp(suffix=".wav", prefix="facevox_")
        os.close(fd)
    Path(out_path).write_bytes(data)
    return str(out_path)


if __name__ == "__main__":
    import sys
    t = sys.argv[1] if len(sys.argv) > 1 else "안녕하세요 만나서 반갑습니다"
    e = sys.argv[2] if len(sys.argv) > 2 else "행복"
    m = sys.argv[3] if len(sys.argv) > 3 else "D"
    print(synthesize(t, e, m))
