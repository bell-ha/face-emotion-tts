"""E1-a: FER2013 PublicTest / PrivateTest evaluation of face/face_emotion.h5.
Also evaluates Training split (to probe train/test overlap) and 6-class (Disgust->Anger) accuracy.
"""
import io
import json
import os
import sys

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, f1_score)

sys.path.insert(0, os.path.dirname(__file__))
from fer_common import (EMOTION_LABELS, KO_LABELS, KO6, LABEL_TO_KO, ROOT, load_fer_model,
                        savefig, set_seed, setup_korean_font)

set_seed()
DATA = os.path.join(ROOT, "data", "fer2013")
OUT = os.path.join(ROOT, "experiments", "E1_fer")
RES, FIG = os.path.join(OUT, "results"), os.path.join(OUT, "figures")


def load_split(name):
    df = pd.read_parquet(os.path.join(DATA, f"{name}.parquet"))
    X = np.zeros((len(df), 48, 48), np.uint8)
    n_nongray = 0
    for i, b in enumerate(df.image):
        a = np.array(Image.open(io.BytesIO(b["bytes"])))
        if a.ndim == 3:
            if not (np.array_equal(a[..., 0], a[..., 1]) and np.array_equal(a[..., 1], a[..., 2])):
                n_nongray += 1
            a = a[..., 0]
        X[i] = a
    return X, df.label.values.astype(int), n_nongray


def to6(y7):
    return np.array([KO6.index(LABEL_TO_KO[EMOTION_LABELS[i]]) for i in y7])


def main():
    model = load_fer_model()
    summary = {"dataset": "FER2013 (HF mirror Aaryan333/fer2013_train_publicTest_privateTest)",
               "label_order": EMOTION_LABELS, "splits": {}}
    dist_rows = []
    plt = setup_korean_font()
    cms = {}
    for split in ["publicTest", "privateTest", "train"]:
        X, y, nng = load_split(split)
        x = (X.astype(np.float32) / 255.0)[..., None]
        P = model.predict(x, batch_size=256, verbose=0)
        np.save(os.path.join(RES, f"probs_{split}.npy"), P.astype(np.float32))
        pred = P.argmax(1)
        counts = np.bincount(y, minlength=7)
        for k in range(7):
            dist_rows.append({"split": split, "class": EMOTION_LABELS[k], "class_ko": KO_LABELS[k],
                              "count": int(counts[k]), "ratio": float(counts[k] / len(y))})
        rep = classification_report(y, pred, labels=list(range(7)), target_names=EMOTION_LABELS,
                                    output_dict=True, zero_division=0)
        cm = confusion_matrix(y, pred, labels=list(range(7)))
        y6, p6 = to6(y), to6(pred)
        s = {
            "n": int(len(y)), "non_gray_images": int(nng),
            "prob_sum_mean": float(P.sum(1).mean()),
            "accuracy": float(accuracy_score(y, pred)),
            "macro_f1": float(f1_score(y, pred, average="macro")),
            "weighted_f1": float(f1_score(y, pred, average="weighted")),
            "acc_6class_ko": float(accuracy_score(y6, p6)),
            "macro_f1_6class_ko": float(f1_score(y6, p6, average="macro")),
            "per_class": {c: {"precision": rep[c]["precision"], "recall": rep[c]["recall"],
                              "f1": rep[c]["f1-score"], "support": int(rep[c]["support"])}
                          for c in EMOTION_LABELS},
            "confusion_matrix": cm.tolist(),
            "confusion_matrix_6class_ko": confusion_matrix(y6, p6, labels=list(range(6))).tolist(),
            "mean_max_prob": float(P.max(1).mean()),
        }
        summary["splits"][split] = s
        cms[split] = cm
        rows = [{"class": c, "class_ko": KO_LABELS[i], **s["per_class"][c]} for i, c in enumerate(EMOTION_LABELS)]
        pd.DataFrame(rows).to_csv(os.path.join(RES, f"per_class_{split}.csv"), index=False)
        print(split, {k: v for k, v in s.items() if k in ["accuracy", "macro_f1", "weighted_f1", "acc_6class_ko", "non_gray_images"]})

    pd.DataFrame(dist_rows).to_csv(os.path.join(RES, "class_distribution.csv"), index=False)
    with open(os.path.join(RES, "fer2013_metrics.json"), "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # --- Confusion matrices (row-normalized), Public & Private side by side
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.9))
    fig.subplots_adjust(wspace=0.38)
    for ax, split, title in zip(axes, ["publicTest", "privateTest"], ["(a)", "(b)"]):
        cm = cms[split].astype(float)
        cmn = cm / cm.sum(1, keepdims=True)
        im = ax.imshow(cmn, cmap="Greys", vmin=0, vmax=1)
        for i in range(7):
            for j in range(7):
                ax.text(j, i, f"{cmn[i, j]:.2f}", ha="center", va="center", fontsize=6,
                        color="white" if cmn[i, j] > 0.5 else "black")
        ax.set_xticks(range(7)); ax.set_yticks(range(7))
        ax.set_xticklabels(EMOTION_LABELS, rotation=45, ha="right"); ax.set_yticklabels(EMOTION_LABELS)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        ax.text(-0.32, 1.04, title, transform=ax.transAxes, fontsize=8, va="bottom")
        ax.tick_params(axis="x", labelsize=6.5); ax.tick_params(axis="y", labelsize=6.5)
    fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02, label="Row-normalized proportion")
    savefig(fig, os.path.join(FIG, "fig_e1_confusion_fer2013"))
    plt.close(fig)

    # --- Class distribution (grouped bars with hatching)
    fig, ax = plt.subplots(figsize=(3.0, 2.0))
    dd = pd.DataFrame(dist_rows)
    hatches = ["", "///", "..."]
    colors = ["0.25", "0.6", "0.9"]
    w = 0.27
    for k, split in enumerate(["train", "publicTest", "privateTest"]):
        r = dd[dd.split == split]
        ax.bar(np.arange(7) + (k - 1) * w, r.ratio * 100, w, color=colors[k], edgecolor="black",
               hatch=hatches[k], label={"train": "Training (n=28,709)", "publicTest": "PublicTest (n=3,589)",
                                         "privateTest": "PrivateTest (n=3,589)"}[split])
    ax.set_xticks(range(7)); ax.set_xticklabels(EMOTION_LABELS, rotation=30, ha="right")
    ax.set_ylabel("Proportion (%)"); ax.set_ylim(0, 34); ax.legend(frameon=False, fontsize=6, ncol=2, loc="upper left")
    ax.tick_params(axis="x", labelsize=6.5)
    ax.spines[["top", "right"]].set_visible(False)
    savefig(fig, os.path.join(FIG, "fig_e1_class_distribution"))
    plt.close(fig)

    # --- Per-class F1 bars Public vs Private
    fig, ax = plt.subplots(figsize=(3.0, 2.0))
    for k, split in enumerate(["publicTest", "privateTest"]):
        f1s = [summary["splits"][split]["per_class"][c]["f1"] for c in EMOTION_LABELS]
        ax.bar(np.arange(7) + (k - 0.5) * 0.38, f1s, 0.38, color=["0.35", "0.85"][k], edgecolor="black",
               hatch=["", "///"][k], label=["PublicTest", "PrivateTest"][k])
    ax.set_xticks(range(7)); ax.set_xticklabels(EMOTION_LABELS, rotation=30, ha="right"); ax.set_ylim(0, 1)
    ax.set_ylabel("F1 score"); ax.legend(frameon=False, fontsize=6.5, ncol=2, loc="upper left")
    ax.set_ylim(0, 1.12); ax.tick_params(axis="x", labelsize=6.5)
    ax.spines[["top", "right"]].set_visible(False)
    savefig(fig, os.path.join(FIG, "fig_e1_per_class_f1"))
    plt.close(fig)


if __name__ == "__main__":
    main()
