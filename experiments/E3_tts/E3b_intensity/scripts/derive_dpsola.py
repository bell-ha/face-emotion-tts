"""D-psola 프리셋 도출 (dev 문장 S03~S06만 사용, SER 튜닝 없음).

edge-tts SSML 단계(C)가 만든 운율 변화(평균 F0 반음, 발화 속도 비, 강도 dB)를 dev 문장에서 B 대비로 측정하고,
이를 PSOLA 파라미터로 옮긴 뒤 D 후처리 파라미터와 합성한다.
  f0_shift = ΔF0_C(dev 평균) + D.f0_shift
  tempo    = (1 + Δrate_C/100) × D.tempo
  gain_db  = ΔInt_C + D.gain_db
  f0_range, contour, tremor, tilt_db, breath_snr = D 그대로
C 단계의 스펙트럼 변화(edge-tts 피치 이동에 동반되는 고역비 증가, 포먼트 이동 추정)는 일부러 옮기지 않는다.
"""
import json, sys
from pathlib import Path
import pandas as pd

E3 = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(E3 / "scripts"))
import prosody as P  # noqa: E402

d = pd.read_csv(E3 / "results" / "per_file.csv")
d = d[(d.cond == "C") & (d.split == "dev")]
g = d.groupby("emotion")[["d_f0_st", "d_rate_pct", "d_int_mean_db", "d_hf_ratio_db"]].mean()
out, meas = {}, {}
for emo, p in P.D_PRESETS.items():
    m = g.loc[emo]
    meas[emo] = {k: round(float(v), 3) for k, v in m.items()}
    q = dict(p)
    q["f0_shift"] = round(float(m.d_f0_st) + p["f0_shift"], 2)
    q["tempo"] = round((1 + float(m.d_rate_pct) / 100) * p["tempo"], 3)
    q["gain_db"] = round(float(m.d_int_mean_db) + p["gain_db"], 2)
    out[emo] = q
res = Path(__file__).resolve().parents[1] / "results"
(res / "dpsola_presets.json").write_text(json.dumps({"DP_PRESETS": out, "dev_measured_C_minus_B": meas}, indent=1))
print(json.dumps(out, indent=1)); print(meas)
