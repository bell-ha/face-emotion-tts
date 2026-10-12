"""E2-a: Extract RAVDESS video-only clips and cache per-frame FER probabilities.

For each frame (all frames, native ~29.97 fps): MediaPipe face detection (largest face) ->
preprocess (48x48 gray /255) -> CNN probs (7). Frames without a detected face get NaN probs.
Output: data/ravdess/probs/<clip>.npz with t (s), probs (N,7), face (N,) bool.
"""
import glob
import os
import sys
import time
import zipfile

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "E1_fer", "scripts"))
from fer_common import ROOT, detect_largest, load_fer_model, make_detector, preprocess_face, set_seed

set_seed()
RAV = os.path.join(ROOT, "data", "ravdess")
VID, PRB = os.path.join(RAV, "videos"), os.path.join(RAV, "probs")


def unzip_all():
    for z in sorted(glob.glob(os.path.join(RAV, "zips", "Video_Speech_Actor_*.zip"))):
        try:
            with zipfile.ZipFile(z) as zf:
                for n in zf.namelist():
                    b = os.path.basename(n)
                    if b.startswith("02-") and b.endswith(".mp4"):
                        actor = "Actor_" + b.split("-")[-1][:2]
                        dst = os.path.join(VID, actor, b)
                        if not os.path.exists(dst):
                            os.makedirs(os.path.dirname(dst), exist_ok=True)
                            with zf.open(n) as src, open(dst, "wb") as out:
                                out.write(src.read())
        except zipfile.BadZipFile:
            print("incomplete zip, skip:", z)


def main(actors=None):
    # clips are extracted by e2_partial_unzip.py (AV modality 01; video track identical to 02)
    os.makedirs(PRB, exist_ok=True)
    model = load_fer_model()
    det = make_detector()
    vids = sorted(glob.glob(os.path.join(VID, "Actor_*", "01-*.mp4")))
    if actors:
        vids = [v for v in vids if os.path.basename(os.path.dirname(v)) in actors]
    t_start = time.time()
    for i, v in enumerate(vids):
        out = os.path.join(PRB, os.path.basename(v).replace(".mp4", ".npz"))
        if os.path.exists(out):
            continue
        cap = cv2.VideoCapture(v)
        fps = cap.get(cv2.CAP_PROP_FPS)
        xs, face, ts = [], [], []
        k = 0
        while True:
            ok, f = cap.read()
            if not ok:
                break
            roi, _ = detect_largest(det, f)
            ts.append(k / fps)
            if roi is None:
                face.append(False); xs.append(np.zeros((1, 48, 48, 1), np.float32))
            else:
                face.append(True); xs.append(preprocess_face(roi))
            k += 1
        cap.release()
        face = np.array(face)
        P = model.predict(np.concatenate(xs), batch_size=256, verbose=0).astype(np.float32)
        P[~face] = np.nan
        np.savez(out, t=np.array(ts), probs=P, face=face, fps=fps)
        if i % 20 == 0:
            print(f"{i+1}/{len(vids)} {os.path.basename(v)} frames={k} face={face.mean():.2f} "
                  f"elapsed={time.time()-t_start:.0f}s", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or None)
