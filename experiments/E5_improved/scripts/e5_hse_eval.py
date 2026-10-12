"""E5 Part B: HSEmotion (Savchenko, 2022; AffectNet-trained EfficientNet-B0, ONNX, CPU) as a swap-in FER.

  fer2013 : FER2013 PrivateTest (48x48 gray -> replicated to 3 channels -> 224 RGB pipeline).
  ravdess : raw per-frame accuracy / clip accuracy / confusion on RAVDESS (dev 01-04, test 05-24),
            same MediaPipe crops as the thesis CNN (cache from e5_extract.py).
  latency : per-frame latency on this machine (MediaPipe + preprocessing + classifier), 50 warm-up
            + 500 iterations on 60 RAVDESS frames, same protocol as E1 (CNN measured again side by side).
Contempt (AffectNet class 1) has no counterpart -> dropped, softmax over the other 7 logits.
Usage: e5_hse_eval.py [fer2013] [ravdess] [latency]   (default: all)
"""
import json
import os
import platform
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score

HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "E1_fer", "scripts"))
from fer_common import (EMOTION_LABELS, KO6, LABEL_TO_KO, ROOT, detect_largest, load_fer_model,
                        make_detector, preprocess_face, set_seed)
from e5_extract import HSE_DIR, hse_preprocess, hse_session
from e5_analyze import DEV, HSE_TO_OURS, SKIP_S, load_clips, softmax

set_seed()
RES = os.path.join(ROOT, "experiments", "E5_improved", "results")
MODELS = ["enet_b0_8_best_afew", "enet_b0_8_va_mtl"]
SHORT = {"enet_b0_8_best_afew": "afew", "enet_b0_8_va_mtl": "mtl"}


def to6(y7):
    return np.array([KO6.index(LABEL_TO_KO[EMOTION_LABELS[i]]) for i in y7])


def metrics(y, pred):
    return {"n": int(len(y)), "acc7": float((y == pred).mean()),
            "macro_f1_7": float(f1_score(y, pred, labels=list(range(7)), average="macro", zero_division=0)),
            "acc6": float((to6(y) == to6(pred)).mean()),
            "macro_f1_6": float(f1_score(to6(y), to6(pred), labels=list(range(6)), average="macro", zero_division=0)),
            "recall7": {EMOTION_LABELS[k]: float((pred[y == k] == k).mean()) for k in range(7)},
            "confusion7": confusion_matrix(y, pred, labels=list(range(7))).tolist()}


def fer2013():
    sys.path.insert(0, os.path.join(HERE, "..", "..", "E1_fer", "scripts"))
    from e1_eval_fer2013 import load_split
    import cv2
    X, y, _ = load_split("privateTest")
    out = {}
    cnn_p = np.load(os.path.join(ROOT, "experiments", "E1_fer", "results", "probs_privateTest.npy"))
    out["cnn"] = metrics(y, cnn_p.argmax(1))
    for m in MODELS:
        s = hse_session(m)
        L = []
        for i in range(0, len(X), 64):
            xb = np.concatenate([hse_preprocess(cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)) for g in X[i:i + 64]])
            L.append(s.run(None, {"input": xb})[0][:, :8])
        L = np.concatenate(L)[:, HSE_TO_OURS]
        out[SHORT[m]] = metrics(y, L.argmax(1))
        print("FER2013 PrivateTest", m, {k: v for k, v in out[SHORT[m]].items() if k in ("acc7", "acc6", "macro_f1_7")})
    return out


def ravdess():
    out = {}
    for clf in ["cnn", "afew", "mtl"]:
        clips = load_clips(clf)
        for split, sel in [("dev", lambda a: a in DEV), ("test", lambda a: a not in DEV)]:
            y, p, clip_ok, clip_ok6, rows = [], [], [], [], []
            for c in clips:
                if not sel(c["actor"]):
                    continue
                m = c["t"] >= SKIP_S
                lab = np.where(c["face"][m], np.nan_to_num(c["probs"][m], nan=0).argmax(1), 6)
                y += [c["emo_idx"]] * int(m.sum()); p += list(lab)
                vals, cnt = np.unique(lab, return_counts=True)
                clip_ok.append(vals[np.argmax(cnt)] == c["emo_idx"])
                l6 = to6(lab); v6, c6 = np.unique(l6, return_counts=True)
                clip_ok6.append(v6[np.argmax(c6)] == KO6.index(c["ko"]))
                rows.append({"clip": c["clip"], "actor": c["actor"], "emotion": c["emotion"],
                             "frame_acc7": float((lab == c["emo_idx"]).mean()),
                             "frame_acc6": float((to6(lab) == KO6.index(c["ko"])).mean())})
            y, p = np.array(y), np.array(p)
            r = metrics(y, p)
            df = pd.DataFrame(rows)
            r.update({"n_clips": int(len(df)), "frame_acc6_mean_over_clips": float(df.frame_acc6.mean()),
                      "frame_acc7_mean_over_clips": float(df.frame_acc7.mean()),
                      "clip_acc7": float(np.mean(clip_ok)), "clip_acc6": float(np.mean(clip_ok6)),
                      "per_emotion_frame_acc7": df.groupby("emotion").frame_acc7.mean().to_dict()})
            out[f"{clf}_{split}"] = r
            print("RAVDESS", clf, split, "frame acc7 %.3f acc6 %.3f clip acc6 %.3f" %
                  (r["acc7"], r["acc6"], r["clip_acc6"]))
    return out


def latency(n_warm=50, n_iter=500):
    from e1_latency import frames_ravdess
    import cv2  # noqa: F401
    import onnxruntime as ort
    frames = frames_ravdess()
    det = make_detector()
    cnn = load_fer_model()
    sess = {m: hse_session(m) for m in MODELS}

    def bench(kind):
        rec = []
        for it in range(n_warm + n_iter):
            f = frames[it % len(frames)]
            t0 = time.perf_counter()
            roi, _ = detect_largest(det, f)
            t1 = time.perf_counter()
            if roi is None:
                continue
            if kind == "cnn":
                x = preprocess_face(roi); t2 = time.perf_counter()
                cnn(x, training=False).numpy()
            else:
                x = hse_preprocess(roi); t2 = time.perf_counter()
                sess[kind].run(None, {"input": x})
            t3 = time.perf_counter()
            if it >= n_warm:
                rec.append(((t1 - t0) * 1e3, (t2 - t1) * 1e3, (t3 - t2) * 1e3, (t3 - t0) * 1e3))
        return np.array(rec)

    rows = []
    for kind in ["cnn"] + MODELS:
        a = bench(kind)
        for j, st in enumerate(["detect", "preprocess", "classifier", "total"]):
            rows.append({"model": SHORT.get(kind, "cnn (model(x) direct call)"), "stage": st,
                         "mean_ms": float(a[:, j].mean()), "median_ms": float(np.median(a[:, j])),
                         "p95_ms": float(np.percentile(a[:, j], 95)), "n": int(len(a))})
        print("latency", kind, "total mean %.2f p95 %.2f classifier %.2f ms" %
              (a[:, 3].mean(), np.percentile(a[:, 3], 95), a[:, 2].mean()))
    pd.DataFrame(rows).to_csv(os.path.join(RES, "hse_latency.csv"), index=False)
    try:
        load = os.getloadavg()
    except OSError:
        load = None
    return {"rows": rows, "env": {"platform": platform.platform(), "machine": platform.machine(),
                                  "onnxruntime": ort.__version__, "provider": "CPUExecutionProvider",
                                  "loadavg_at_end": load, "frames": "60 RAVDESS frames 1280x720",
                                  "warmup": n_warm, "iterations": n_iter}}


def main():
    parts = sys.argv[1:] or ["fer2013", "ravdess", "latency"]
    path = os.path.join(RES, "hse_eval.json")
    out = json.load(open(path)) if os.path.exists(path) else {}
    out["note"] = ("HSEmotion ONNX models from github.com/HSE-asavchenko/face-emotion-recognition "
                   "(now sb-ai-lab/EmotiEffLib), Apache-2.0; trained on AffectNet. Contempt dropped.")
    for part in parts:
        out[part] = globals()[part]()
        json.dump(out, open(path, "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))


if __name__ == "__main__":
    main()
