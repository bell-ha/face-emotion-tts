"""E1-b: Per-frame latency on Apple M1 Max CPU.

Stages per frame (same as face_emotion.py):
  detect     : BGR->RGB + MediaPipe FaceDetection.process + largest-box pick + crop
  preprocess : gray + resize 48x48 (INTER_AREA) + /255
  cnn        : model.predict(x, verbose=0)  [as in face_emotion.py]  vs  model(x, training=False)

Frame sources:
  user   : 7 user face photos (HUD cropped & green box inpainted), resized to 640x480 (app capture size)
  ravdess: frames sampled from RAVDESS videos (1280x720), if available
"""
import glob
import json
import os
import platform
import sys
import time

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from fer_common import (ROOT, detect_largest, load_fer_model, make_detector, preprocess_face,
                        savefig, set_seed, setup_korean_font)
from e1_user_photos import NAMES, clean

set_seed()
OUT = os.path.join(ROOT, "experiments", "E1_fer")
RES, FIG = os.path.join(OUT, "results"), os.path.join(OUT, "figures")
N_WARM, N_ITER = 50, 500


def frames_user():
    fr = []
    for n in NAMES:
        img = clean(cv2.imread(os.path.join(ROOT, "assets", "figures", f"{n}.png")))[0]
        fr.append(cv2.resize(img, (640, 480), interpolation=cv2.INTER_AREA))
    return fr


def frames_ravdess(k=60):
    vids = sorted(glob.glob(os.path.join(ROOT, "data", "ravdess", "videos", "Actor_*", "01-*.mp4")))
    if not vids:
        return []
    rng = np.random.default_rng(42)
    sel = rng.choice(len(vids), size=min(k, len(vids)), replace=False)
    fr = []
    for i in sel:
        cap = cv2.VideoCapture(vids[i])
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.set(cv2.CAP_PROP_POS_FRAMES, n // 2)
        ok, f = cap.read()
        cap.release()
        if ok:
            fr.append(f)
    return fr


def bench(frames, model, det, mode):
    rec = []
    total = N_WARM + N_ITER
    for it in range(total):
        f = frames[it % len(frames)]
        t0 = time.perf_counter()
        roi, _ = detect_largest(det, f)
        t1 = time.perf_counter()
        if roi is None:
            continue
        x = preprocess_face(roi)
        t2 = time.perf_counter()
        if mode == "predict":
            p = model.predict(x, verbose=0)
        else:
            p = model(x, training=False).numpy()
        t3 = time.perf_counter()
        if it >= N_WARM:
            rec.append(((t1 - t0) * 1e3, (t2 - t1) * 1e3, (t3 - t2) * 1e3, (t3 - t0) * 1e3))
    return np.array(rec)


def stats(a):
    return {"mean": float(a.mean()), "median": float(np.median(a)), "p95": float(np.percentile(a, 95)),
            "std": float(a.std()), "n": int(len(a))}


def main():
    import tensorflow as tf
    model = load_fer_model()
    det = make_detector()
    sources = {"user_640x480": frames_user()}
    rf = frames_ravdess()
    if rf:
        sources["ravdess_1280x720"] = rf
    env = {"machine": platform.machine(), "platform": platform.platform(), "python": platform.python_version(),
           "tensorflow": tf.__version__, "tf_devices": [d.device_type for d in tf.config.list_physical_devices()],
           "warmup": N_WARM, "iterations": N_ITER}
    import mediapipe
    env["mediapipe"] = mediapipe.__version__
    rows, raw = [], {}
    for sname, frames in sources.items():
        for mode in ["predict", "call"]:
            a = bench(frames, model, det, mode)
            raw[f"{sname}_{mode}"] = a
            for j, stage in enumerate(["detect", "preprocess", "cnn", "total"]):
                s = stats(a[:, j])
                rows.append({"source": sname, "cnn_mode": mode, "stage": stage, **s,
                             "fps_from_mean": 1000.0 / s["mean"] if stage == "total" else None})
            print(sname, mode, "total mean %.2f med %.2f p95 %.2f ms, cnn mean %.2f" %
                  (a[:, 3].mean(), np.median(a[:, 3]), np.percentile(a[:, 3], 95), a[:, 2].mean()))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RES, "latency.csv"), index=False)
    with open(os.path.join(RES, "latency_env.json"), "w") as f:
        json.dump(env, f, indent=2)
    np.savez(os.path.join(RES, "latency_raw.npz"), **raw)
    plot(raw)


def plot(raw):
    # Figure: stacked mean bars per (source, mode) + p95 marker
    plt = setup_korean_font()
    fig, ax = plt.subplots(figsize=(6.0, 2.3))
    labels, ypos = [], 0
    hat = {"detect": "", "preprocess": "...", "cnn": "///"}
    col = {"detect": "0.35", "preprocess": "0.95", "cnn": "0.75"}
    name = {"detect": "Face detection", "preprocess": "Preprocessing", "cnn": "CNN inference"}
    keys = list(raw.keys())
    for i, k in enumerate(keys):
        a = raw[k]
        left = 0
        for j, st in enumerate(["detect", "preprocess", "cnn"]):
            m = a[:, j].mean()
            ax.barh(i, m, left=left, color=col[st], edgecolor="black", hatch=hat[st],
                    label=name[st] if i == 0 else None)
            left += m
        ax.plot(np.percentile(a[:, 3], 95), i, marker="|", color="black", markersize=14)
        src, mode = k.rsplit("_", 1)
        labels.append(f"{'User photos 640×480' if src.startswith('user') else 'RAVDESS frames 1280×720'}\n"
                      f"{'model.predict()' if mode == 'predict' else 'model(x) direct call'}")
    ax.set_yticks(range(len(keys))); ax.set_yticklabels(labels, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("Latency per frame (ms; stacked stage means, vertical tick = total p95)")
    ax.legend(frameon=False, fontsize=7, loc="lower right", ncol=3)
    ax.spines[["top", "right"]].set_visible(False)
    savefig(fig, os.path.join(FIG, "fig_e1_latency"))


if __name__ == "__main__":
    if "--plot-only" in sys.argv:  # redraw from saved measurements without re-measuring
        d = np.load(os.path.join(RES, "latency_raw.npz"))
        plot({k: d[k] for k in d.files})
    else:
        main()
