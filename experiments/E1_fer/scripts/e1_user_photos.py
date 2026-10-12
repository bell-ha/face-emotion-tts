"""E1-c: Detector + classifier on the 7 user face photos (screenshots of the old app).

Two conditions:
  raw     : screenshot as-is (contains HUD bar, green bbox, label text of the old app)
  cleaned : HUD bar (top black band) cropped away and green bbox pixels inpainted
"""
import json
import os
import sys

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from fer_common import (EMOTION_LABELS, KO_LABELS, LABEL_TO_KO, ROOT, detect_largest, load_fer_model,
                        make_detector, preprocess_face, savefig, set_seed, setup_korean_font)

set_seed()
OUT = os.path.join(ROOT, "experiments", "E1_fer")
RES, FIG = os.path.join(OUT, "results"), os.path.join(OUT, "figures")
NAMES = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
INTENDED = dict(zip(NAMES, ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']))
# Values printed on the screenshots by the old live app (read manually from the images)
OVERLAY = {'angry': ('Angry', 0.79), 'disgust': ('Disgust', 1.00), 'fear': ('Fear', 0.30),
           'happy': ('Happy', 1.00), 'sad': ('Sad', 0.60), 'surprise': ('Surprise', 0.78),
           'neutral': ('Neutral', 0.81)}


def clean(img):
    # 1) remove top HUD: window title bar + black HUD band. Find last mostly-black row in the top 25%.
    H = img.shape[0]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    top = int(H * 0.25)
    dark = (gray[:top] < 40).mean(1) > 0.6
    cut = int(np.where(dark)[0].max()) + 2 if dark.any() else 0
    img = img[cut:].copy()
    # 2) inpaint green bbox lines
    b, g, r = [img[..., i].astype(int) for i in range(3)]
    mask = ((g > 150) & (g - r > 60) & (g - b > 60)).astype(np.uint8) * 255
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)
    img = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)
    return img, cut


def main():
    model = load_fer_model()
    det = make_detector()
    rows, crops = [], {}
    for n in NAMES:
        img0 = cv2.imread(os.path.join(ROOT, "assets", "figures", f"{n}.png"))
        for cond in ["raw", "cleaned"]:
            img = img0 if cond == "raw" else clean(img0)[0]
            roi, box = detect_largest(det, img)
            if roi is None:
                rows.append({"photo": n, "condition": cond, "detected": False})
                continue
            x = preprocess_face(roi)
            p = model(x, training=False).numpy()[0]
            k = int(p.argmax())
            row = {"photo": n, "condition": cond, "detected": True, "intended": INTENDED[n],
                   "top1": EMOTION_LABELS[k], "top1_conf": float(p[k]),
                   "top1_ko6": LABEL_TO_KO[EMOTION_LABELS[k]],
                   "correct7": EMOTION_LABELS[k] == INTENDED[n],
                   "correct6": LABEL_TO_KO[EMOTION_LABELS[k]] == LABEL_TO_KO[INTENDED[n]],
                   "old_app_overlay_label": OVERLAY[n][0], "old_app_overlay_conf": OVERLAY[n][1],
                   "bbox": list(map(int, box))}
            row.update({f"p_{c}": float(p[i]) for i, c in enumerate(EMOTION_LABELS)})
            rows.append(row)
            if cond == "cleaned":
                crops[n] = (cv2.resize(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY), (48, 48), interpolation=cv2.INTER_AREA), p)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RES, "user_photos_predictions.csv"), index=False)
    for cond in ["raw", "cleaned"]:
        d = df[df.condition == cond]
        print(cond, "7cls correct", int(d.correct7.sum()), "/", len(d), " 6cls", int(d.correct6.sum()))
        print(d[["photo", "top1", "top1_conf"] + [f"p_{c}" for c in EMOTION_LABELS]].round(3).to_string(index=False))

    # Figure: 48x48 inputs + probability bars (cleaned condition)
    plt = setup_korean_font()
    fig, axes = plt.subplots(2, 7, figsize=(6.3, 2.4), gridspec_kw={"height_ratios": [1, 1.1]})
    for j, n in enumerate(NAMES):
        g, p = crops[n]
        axes[0, j].imshow(g, cmap="gray", vmin=0, vmax=255)
        axes[0, j].set_axis_off()
        ki = EMOTION_LABELS.index(INTENDED[n])
        axes[0, j].text(0.5, 1.03, EMOTION_LABELS[ki], transform=axes[0, j].transAxes,
                        ha="center", va="bottom", fontsize=7)
        if j == 0:
            axes[0, j].text(-0.12, 0.5, "Posed", transform=axes[0, j].transAxes, rotation=90,
                            ha="right", va="center", fontsize=7)
        cols = ["0.15" if i == p.argmax() else "0.75" for i in range(7)]
        axes[1, j].barh(range(7), p, color=cols, edgecolor="black", linewidth=0.5)
        axes[1, j].set_yticks(range(7))
        axes[1, j].set_yticklabels(EMOTION_LABELS if j == 0 else [], fontsize=6)
        axes[1, j].invert_yaxis(); axes[1, j].set_xlim(0, 1)
        axes[1, j].set_xticks([0, 0.5, 1]); axes[1, j].set_xticklabels(["0", "0.5", "1"])
        axes[1, j].tick_params(axis="x", labelsize=5.5)
        axes[1, j].set_xlabel(f"Pred: {EMOTION_LABELS[p.argmax()]}\n({p.max():.2f})", fontsize=6.5)
    fig.tight_layout(w_pad=1.0)
    savefig(fig, os.path.join(FIG, "fig_e1_user_photos"))
    with open(os.path.join(RES, "user_photos_summary.json"), "w") as f:
        json.dump({c: {"n": int((df.condition == c).sum()),
                       "correct7": int(df[df.condition == c].correct7.sum()),
                       "correct6": int(df[df.condition == c].correct6.sum())} for c in ["raw", "cleaned"]},
                  f, indent=2)


if __name__ == "__main__":
    main()
