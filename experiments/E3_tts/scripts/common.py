"""E3 공통 정의: 문장 세트, 감정, 경로, 음향 특징, SER, 화자 임베딩, SQUIM."""
from __future__ import annotations

import os
import re
import warnings
from functools import lru_cache
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
SEED = 42
np.random.seed(SEED)

ROOT = Path(__file__).resolve().parents[1]          # experiments/E3_tts
PROJ = ROOT.parents[1]                              # face-emotion-tts
AUDIO = ROOT / "audio"
RESULTS = ROOT / "results"
FIGS = ROOT / "figures"
MODELS = PROJ / "data" / "models"
for d in (AUDIO, RESULTS, FIGS, MODELS):
    d.mkdir(parents=True, exist_ok=True)

# 감정 (내부 키 → 한국어)
EMOS = ["happy", "sad", "angry", "surprise", "fear", "neutral"]
EMO_KO = {"happy": "행복", "sad": "슬픔", "angry": "분노", "surprise": "놀람",
          "fear": "두려움", "neutral": "중립"}
KO_EMO = {v: k for k, v in EMO_KO.items()}

# 어휘적으로 감정이 중립/모호한 문장 12개. S01, S02는 졸업논문 샘플 문장.
SENTENCES = {
    "S01": "안녕하세요 만나서 반갑습니다",
    "S02": "깊은 숲속 오두막집에서 작은 토끼가 조심스레 문을 열고 밖으로 나왔어요",
    "S03": "오늘 오후 세 시에 회의실에서 다시 이야기하기로 했어요",
    "S04": "버스는 다음 정류장에서 오른쪽으로 돌아갑니다",
    "S05": "책상 위에 파란색 공책 두 권이 놓여 있어요",
    "S06": "그 사람이 어제 저녁에 우리 집에 왔었어요",
    "S07": "내일은 아침 일찍 기차를 타고 부산에 갑니다",
    "S08": "창문 밖으로 하얀 자동차 한 대가 지나갔어요",
    "S09": "그 상자 안에는 오래된 편지가 들어 있었어요",
    "S10": "지금 바로 문을 열어 보세요",
    "S11": "우리는 다음 주 월요일에 이사를 하게 됐어요",
    "S12": "엄마가 부엌에서 저녁을 준비하고 계셨어요",
}
DEV = ["S03", "S04", "S05", "S06"]                  # 프리셋 조정용
TEST = [k for k in SENTENCES if k not in DEV]        # 보고용(held-out, 8문장)

# (E) 참고 조건: 어휘 자체가 감정적인 문장 (감정당 3문장, 중립 운율로 합성)
LEXICAL = {
    "happy": ["와 정말 너무 기뻐요 오늘 최고의 날이에요",
              "합격했대요 드디어 우리가 해냈어요",
              "선물 고마워요 진짜 행복해요"],
    "sad": ["너무 슬퍼서 아무것도 할 수가 없어요",
            "결국 그 친구를 다시는 볼 수 없게 됐어요",
            "혼자 남겨진 것 같아 마음이 아파요"],
    "angry": ["도대체 왜 약속을 또 어긴 거예요",
              "이건 정말 참을 수가 없어요 당장 그만두세요",
              "몇 번을 말해야 알아듣겠어요"],
    "surprise": ["어머 이게 정말 여기 있었다고요",
                 "세상에 벌써 다 끝났다니 믿을 수가 없어요",
                 "깜짝이야 언제 거기 와 있었어요"],
    "fear": ["무서워요 누가 밖에서 문을 두드리고 있어요",
             "제발 혼자 두지 마세요 너무 겁이 나요",
             "어두운 복도 끝에서 이상한 소리가 들려요"],
    "neutral": ["회의는 내일 오전 열 시에 시작합니다",
                "이 건물은 지하 이층까지 있습니다",
                "서류는 책상 오른쪽 서랍에 있습니다"],
}

HANGUL = re.compile(r"[가-힣]")


def n_syllables(text: str) -> int:
    return len(HANGUL.findall(text))


# ───────────────────────── 음향 특징 ─────────────────────────
def acoustic_features(path: str | Path, text: str) -> dict:
    """F0(Hz, st), 강도(dB), 발화 속도. Praat 기본 알고리즘(parselmouth)."""
    import parselmouth
    from parselmouth.praat import call

    snd = parselmouth.Sound(str(path))
    if snd.n_channels > 1:
        snd = snd.convert_to_mono()
    dur = snd.get_total_duration()
    pitch = snd.to_pitch_ac(time_step=0.01, pitch_floor=75, pitch_ceiling=600)
    f0 = pitch.selected_array["frequency"]
    f0 = f0[f0 > 0]
    st = 12 * np.log2(f0 / 100.0) if len(f0) else np.array([np.nan])
    intensity = snd.to_intensity(minimum_pitch=75, time_step=0.01)
    iv = intensity.values.flatten()
    # 발화 구간(강도 > 최대-35dB) 기준 평균 강도·발화 시간
    thr = np.nanmax(iv) - 35
    active = iv > thr
    speech_time = active.sum() * 0.01
    # 에너지 평균은 dB의 산술평균이 아니라 파워 평균
    i_mean = 10 * np.log10(np.mean(10 ** (iv[active] / 10))) if active.any() else np.nan
    voiced_time = len(f0) * 0.01
    syl = n_syllables(text)
    # 스펙트럼 기울기 근사: 0-1kHz 대비 1-4kHz 에너지 비(dB)
    spec = snd.to_spectrum()
    lo = call(spec, "Get band energy", 0, 1000)
    hi = call(spec, "Get band energy", 1000, 4000)
    return {
        "dur_s": dur,
        "speech_s": speech_time,
        "f0_mean_hz": float(np.mean(f0)) if len(f0) else np.nan,
        "f0_median_hz": float(np.median(f0)) if len(f0) else np.nan,
        "f0_sd_st": float(np.std(st)),
        "f0_range_st": float(np.percentile(st, 95) - np.percentile(st, 5)),
        "int_mean_db": float(i_mean),
        "rate_syl_per_s": syl / speech_time if speech_time > 0 else np.nan,
        "voiced_s": voiced_time,
        "hf_ratio_db": float(10 * np.log10(hi / lo)) if lo > 0 and hi > 0 else np.nan,
        "n_syl": syl,
    }


# ───────────────────────── SER: emotion2vec_plus_large ─────────────────────────
E2V_TO_OURS = {"happy": "happy", "sad": "sad", "angry": "angry", "disgusted": "angry",
               "surprised": "surprise", "fearful": "fear", "neutral": "neutral",
               "other": None, "unknown": None}


@lru_cache(maxsize=1)
def _e2v():
    from funasr import AutoModel
    return AutoModel(model="iic/emotion2vec_plus_large", hub="hf", disable_update=True)


def ser_emotion2vec(path: str | Path) -> dict:
    """반환: 9클래스 원 확률(dict), 6감정 매핑 확률(disgusted는 분노에 합산)."""
    import librosa
    y, _ = librosa.load(str(path), sr=16000, mono=True)
    res = _e2v().generate(y, granularity="utterance", extract_embedding=False,
                          disable_pbar=True)[0]
    raw = {}
    for lab, sc in zip(res["labels"], res["scores"]):
        eng = lab.split("/")[-1].strip()
        if eng == "<unk>":
            eng = "unknown"
        raw[eng] = float(sc)
    ours = {e: 0.0 for e in EMOS}
    for k, v in raw.items():
        m = E2V_TO_OURS.get(k)
        if m:
            ours[m] += v
    top9 = max(raw, key=raw.get)
    return {"raw": raw, "ours": ours,
            "pred9": E2V_TO_OURS.get(top9) or "nomatch",   # 9클래스 argmax → 매핑
            "pred6": max(ours, key=ours.get)}              # 6감정으로 제한한 argmax


# ───────────────────────── 보조 SER: wav2vec2 SUPERB-ER (4클래스) ─────────────────────────
@lru_cache(maxsize=1)
def _w2v_er():
    from transformers import pipeline
    return pipeline("audio-classification", model="superb/wav2vec2-base-superb-er",
                    device="cpu")


W2V_MAP = {"neu": "neutral", "hap": "happy", "ang": "angry", "sad": "sad"}


def ser_w2v(path: str | Path) -> dict:
    import librosa
    y, _ = librosa.load(str(path), sr=16000, mono=True)
    out = _w2v_er()({"raw": y, "sampling_rate": 16000}, top_k=4)
    probs = {W2V_MAP[o["label"]]: float(o["score"]) for o in out}
    return {"probs": probs, "pred": max(probs, key=probs.get)}


# ───────────────────────── 화자 임베딩: ECAPA-TDNN ─────────────────────────
@lru_cache(maxsize=1)
def _ecapa():
    from speechbrain.inference.speaker import EncoderClassifier
    return EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb",
                                          savedir=str(MODELS / "ecapa"),
                                          run_opts={"device": "cpu"})


def spk_embed(path: str | Path) -> np.ndarray:
    import librosa
    import torch
    y, _ = librosa.load(str(path), sr=16000, mono=True)
    with torch.no_grad():
        e = _ecapa().encode_batch(torch.tensor(y)[None]).squeeze().numpy()
    return e / np.linalg.norm(e)


# ───────────────────────── 객관 품질: TorchAudio SQUIM ─────────────────────────
@lru_cache(maxsize=1)
def _squim():
    from torchaudio.pipelines import SQUIM_OBJECTIVE, SQUIM_SUBJECTIVE
    return SQUIM_OBJECTIVE.get_model().eval(), SQUIM_SUBJECTIVE.get_model().eval()


def squim(path: str | Path, nmr_path: str | Path) -> dict:
    """STOI/PESQ/SI-SDR 추정(참조 불필요) + MOS 추정(비매칭 참조 nmr 필요)."""
    import librosa
    import torch
    obj, subj = _squim()
    y, _ = librosa.load(str(path), sr=16000, mono=True)
    r, _ = librosa.load(str(nmr_path), sr=16000, mono=True)
    x = torch.tensor(y)[None]
    with torch.no_grad():
        stoi, pesq, sisdr = obj(x)
        mos = subj(x, torch.tensor(r)[None])
    return {"squim_stoi": float(stoi), "squim_pesq": float(pesq),
            "squim_sisdr": float(sisdr), "squim_mos": float(mos)}
