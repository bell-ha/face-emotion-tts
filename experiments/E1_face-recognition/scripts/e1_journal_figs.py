"""Journal figures built from saved E1/E5 results (no model inference).

  fig_e1_user_photos_v2 : enlarged face crops (cleaned screenshots) + 48x48 model input inset + probability bars
  fig_fer_confusion_3   : row-normalized confusion (a) FER2013 PrivateTest, (b) RAVDESS CNN, (c) RAVDESS HSEmotion

실행: .venv/bin/python experiments/E1_face-recognition/scripts/e1_journal_figs.py
"""
import ast
import json
import os
import sys

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from e1_user_photos import INTENDED, NAMES, clean  # noqa: E402
from fer_common import EMOTION_LABELS, ROOT, savefig, setup_korean_font  # noqa: E402

E1 = os.path.join(ROOT, "experiments", "E1_face-recognition")
E5 = os.path.join(ROOT, "experiments", "E5_improvements")
plt = setup_korean_font()


def user_photos():
    df = pd.read_csv(os.path.join(E1, "results", "user_photos_predictions.csv"))
    df = df[df.condition == "cleaned"].set_index("photo")
    fig, axes = plt.subplots(2, 7, figsize=(6.3, 2.75), gridspec_kw={"height_ratios": [1.15, 1]})
    for j, n in enumerate(NAMES):
        r = df.loc[n]
        img, _ = clean(cv2.imread(os.path.join(ROOT, "assets", "figures", f"{n}.png")))
        x, y, w, h = ast.literal_eval(r.bbox)
        # 구 앱이 얼굴 상자 위에 그린 흰 라벨 글자("Angry (0.79)")를 지운다
        band = np.zeros(img.shape[:2], np.uint8)
        band[max(y - int(0.3 * h), 0):y + 4, max(x - int(0.2 * w), 0):x + int(1.6 * w)] = 1
        txt = ((img.min(2) > 225) & (band > 0)).astype(np.uint8) * 255
        img = cv2.inpaint(img, cv2.dilate(txt, np.ones((5, 5), np.uint8)), 5, cv2.INPAINT_TELEA)
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cx, cy, s = x + w / 2, y + h / 2, 1.45 * max(w, h)          # 얼굴 주변을 조금 더 넣어 자름
        x0, y0 = int(max(cx - s / 2, 0)), int(max(cy - s / 2, 0))
        crop = g[y0:int(y0 + s), x0:int(x0 + s)]
        ax = axes[0, j]
        ax.imshow(crop, cmap="gray", vmin=0, vmax=255)
        ax.set_axis_off()
        ax.set_title(INTENDED[n], fontsize=7, pad=2)
        inp = cv2.resize(g[y:y + h, x:x + w], (48, 48), interpolation=cv2.INTER_AREA)
        ins = ax.inset_axes([0.64, 0.0, 0.36, 0.36])                 # 모델 입력(48x48)
        ins.imshow(inp, cmap="gray", vmin=0, vmax=255)
        ins.set_xticks([]); ins.set_yticks([])
        for sp in ins.spines.values():
            sp.set_color("white"); sp.set_linewidth(0.8)
        p = np.array([r[f"p_{c}"] for c in EMOTION_LABELS])
        b = axes[1, j]
        b.barh(range(7), p, color=["0.15" if i == p.argmax() else "0.75" for i in range(7)],
               edgecolor="black", linewidth=0.4)
        b.set_yticks(range(7)); b.set_yticklabels(EMOTION_LABELS if j == 0 else [], fontsize=6)
        b.invert_yaxis(); b.set_xlim(0, 1)
        b.set_xticks([0, 0.5, 1]); b.set_xticklabels(["0", "0.5", "1"]); b.tick_params(axis="x", labelsize=5.5)
        ok = "" if r.correct7 else " ×"
        b.set_xlabel(f"Pred: {EMOTION_LABELS[p.argmax()]}{ok}\n({p.max():.2f})", fontsize=6.5)
    fig.tight_layout(w_pad=0.8, h_pad=0.6)
    savefig(fig, os.path.join(E1, "figures", "fig_e1_user_photos_v2"))
    plt.close(fig)


def confusion_3():
    m = json.load(open(os.path.join(E1, "results", "fer2013_metrics.json")))
    H = json.load(open(os.path.join(E5, "results", "hse_eval.json")))
    hse = json.load(open(os.path.join(E5, "results", "e5_summary.json")))["hse_dev_selection"]["chosen"]
    mats = [(np.array(m["splits"]["privateTest"]["confusion_matrix"], float), "FER2013 PrivateTest\n(CNN)"),
            (np.array(H["ravdess"]["cnn_test"]["confusion7"], float), "RAVDESS test actors\n(CNN)"),
            (np.array(H["ravdess"][f"{hse}_test"]["confusion7"], float), "RAVDESS test actors\n(HSEmotion)")]
    fig, axes = plt.subplots(1, 3, figsize=(6.3, 2.55))
    for k, (ax, (cm, title)) in enumerate(zip(axes, mats)):
        cmn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
        im = ax.imshow(cmn, cmap="Greys", vmin=0, vmax=1)
        for i in range(7):
            for j in range(7):
                ax.text(j, i, f"{cmn[i, j]:.2f}".lstrip("0") if cmn[i, j] < 1 else "1", ha="center",
                        va="center", fontsize=4.6, color="white" if cmn[i, j] > 0.5 else "black")
        ax.set_xticks(range(7)); ax.set_yticks(range(7))
        ax.set_xticklabels(EMOTION_LABELS, fontsize=5.5, rotation=45, ha="right")
        ax.set_yticklabels(EMOTION_LABELS if k == 0 else [], fontsize=5.5)
        ax.set_xlabel("Predicted", fontsize=6.5)
        if k == 0:
            ax.set_ylabel("True", fontsize=6.5)
        ax.set_title(f"({'abc'[k]}) {title}", fontsize=6.5, pad=3)
    cb = fig.colorbar(im, ax=axes, fraction=0.018, pad=0.015)
    cb.ax.tick_params(labelsize=5.5)
    savefig(fig, os.path.join(E1, "figures", "fig_fer_confusion_3"))
    plt.close(fig)


if __name__ == "__main__":
    user_photos()
    confusion_3()
