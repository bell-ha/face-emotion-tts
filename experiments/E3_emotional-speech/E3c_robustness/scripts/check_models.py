"""측정 모델 재현 확인: 2026-11-19에 다시 받은 모델로 기존 E3(run 0) 음성을 다시 채점해 기존 per_file.csv와 비교한다.

HF 캐시가 지워져 emotion2vec_plus_large, wav2vec2-superb-er, SQUIM을 다시 내려받았기 때문에,
같은 음성에서 같은 판정·값이 나오는지 확인한다(평가 8문장, E3 B·C·D와 E3b DP k=1).
결과: results/model_check.csv, model_check.json
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parents[1]
E3 = HERE.parent
sys.path.insert(0, str(E3 / "scripts"))
import common as C  # noqa: E402

RES = HERE / "results"
NMR = C.AUDIO / "B" / "S05_neutral.wav"


def main():
    old = pd.read_csv(E3 / "results" / "per_file.csv")
    old = old[(old.split == "test") & old.cond.isin(["B", "C", "D"])].drop_duplicates("path")
    old["file"] = [str(C.AUDIO / c / Path(p).name) for c, p in zip(old.cond, old.path)]
    oldb = pd.read_csv(E3 / "E3b_intensity" / "results" / "per_file.csv")
    oldb = oldb[(oldb.split == "test") & (oldb.variant == "DP") & (oldb.k == 1.0)].copy()
    oldb["file"] = [str(C.PROJ / "data" / "E3b_audio" / "DP_k100" / Path(p).name) for p in oldb.path]
    oldb["cond"] = "DP"
    rows = []
    for src, df in (("E3", old), ("E3b", oldb)):
        for r in df.itertuples():
            s = C.ser_emotion2vec(r.file)
            w = C.ser_w2v(r.file)
            q = C.squim(r.file, NMR)
            rows.append(dict(src=src, cond=r.cond, file=r.file, e2v_old=r.e2v_pred9, e2v_new=s["pred9"],
                             w2v_old=r.w2v_pred, w2v_new=w["pred"], mos_old=r.squim_mos, mos_new=q["squim_mos"]))
    d = pd.DataFrame(rows)
    d.to_csv(RES / "model_check.csv", index=False)
    js = {"n_files": len(d), "e2v_pred_agree": float((d.e2v_old == d.e2v_new).mean()),
          "w2v_pred_agree": float((d.w2v_old == d.w2v_new).mean()),
          "mos_max_abs_diff": float((d.mos_old - d.mos_new).abs().max()),
          "e2v_mismatch": d[d.e2v_old != d.e2v_new][["cond", "file", "e2v_old", "e2v_new"]].to_dict("records"),
          "w2v_mismatch_n": int((d.w2v_old != d.w2v_new).sum())}
    (RES / "model_check.json").write_text(json.dumps(js, indent=1, ensure_ascii=False))
    print(json.dumps(js, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
