"""E5: per-frame probabilities of BOTH classifiers on the same MediaPipe face crop.

Pipeline per frame (identical to E2 / face_emotion.py up to the crop):
  MediaPipe FaceDetection(min_conf 0.5) -> largest bbox -> crop (safe_crop)
  (a) thesis CNN: 48x48 gray /255 -> 7 probs              (same as E2 cache)
  (b) HSEmotion ONNX (Savchenko 2022; AffectNet 8 classes): crop BGR->RGB, resize 224 (cv2 bilinear),
      /255, ImageNet mean/std (= hsemotion_onnx.HSEmotionRecognizer.preprocess) -> logits
      models: enet_b0_8_best_afew, enet_b0_8_va_mtl (8 logits + valence/arousal)
Output: data/ravdess/probs_e5/<clip>.npz : t, face, bbox, cnn (N,7), afew (N,8), mtl (N,10)
Usage: e5_extract.py [Actor_01 Actor_02 ...]
"""
import glob, os, sys, time
import cv2
import numpy as np
import onnxruntime as ort

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "E1_fer", "scripts"))
from fer_common import ROOT, load_fer_model, make_detector, preprocess_face, safe_crop, set_seed

set_seed()
VID = os.path.join(ROOT, "data", "ravdess", "videos")
OUT = os.path.join(ROOT, "data", "ravdess", "probs_e5")
HSE_DIR = os.path.join(ROOT, "data", "models", "hsemotion")
HSE_MODELS = ["enet_b0_8_best_afew", "enet_b0_8_va_mtl"]
MEAN = np.array([0.485, 0.456, 0.406], np.float32); STD = np.array([0.229, 0.224, 0.225], np.float32)


def hse_session(name):
    so = ort.SessionOptions()
    return ort.InferenceSession(os.path.join(HSE_DIR, name + ".onnx"), so, providers=["CPUExecutionProvider"])


def hse_preprocess(face_bgr, size=224):
    rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    x = cv2.resize(rgb, (size, size)).astype(np.float32) / 255.0
    x = (x - MEAN) / STD
    return x.transpose(2, 0, 1)[None]


def detect_largest_box(det, frame_bgr):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    r = det.process(rgb)
    if not r.detections:
        return None, None
    h, w = frame_bgr.shape[:2]
    best, ba = None, -1
    for d in r.detections:
        b = d.location_data.relative_bounding_box
        x, y, bw, bh = int(b.xmin * w), int(b.ymin * h), int(b.width * w), int(b.height * h)
        if max(0, bw) * max(0, bh) > ba:
            ba = max(0, bw) * max(0, bh); best = (x, y, bw, bh)
    return safe_crop(frame_bgr, *best), best


def main(actors=None):
    os.makedirs(OUT, exist_ok=True)
    cnn = load_fer_model()
    det = make_detector()
    sess = {m: hse_session(m) for m in HSE_MODELS}
    vids = sorted(glob.glob(os.path.join(VID, "Actor_*", "01-*.mp4")))
    if actors:
        vids = [v for v in vids if os.path.basename(os.path.dirname(v)) in actors]
    t0 = time.time()
    for i, v in enumerate(vids):
        b = os.path.basename(v)[:-4]
        if int(b.split("-")[2]) == 2:
            continue  # calm not used
        out = os.path.join(OUT, b + ".npz")
        if os.path.exists(out):
            continue
        cap = cv2.VideoCapture(v); fps = cap.get(cv2.CAP_PROP_FPS)
        xs, hs, face, ts, boxes = [], [], [], [], []
        k = 0
        while True:
            ok, f = cap.read()
            if not ok:
                break
            roi, box = detect_largest_box(det, f)
            ts.append(k / fps)
            if roi is None:
                face.append(False); boxes.append((0, 0, 0, 0))
                xs.append(np.zeros((1, 48, 48, 1), np.float32)); hs.append(np.zeros((1, 3, 224, 224), np.float32))
            else:
                face.append(True); boxes.append(box)
                xs.append(preprocess_face(roi)); hs.append(hse_preprocess(roi))
            k += 1
        cap.release()
        face = np.array(face)
        P = cnn.predict(np.concatenate(xs), batch_size=256, verbose=0).astype(np.float32)
        H = np.concatenate(hs)
        res = {}
        for m, s in sess.items():
            res[m] = np.concatenate([s.run(None, {"input": H[j:j + 64]})[0] for j in range(0, len(H), 64)]).astype(np.float32)
        P[~face] = np.nan
        for m in res:
            res[m][~face] = np.nan
        np.savez(out, t=np.array(ts), face=face, bbox=np.array(boxes), fps=fps, cnn=P,
                 afew=res["enet_b0_8_best_afew"], mtl=res["enet_b0_8_va_mtl"])
        if i % 10 == 0:
            print(f"{i+1}/{len(vids)} {b} frames={k} face={face.mean():.2f} elapsed={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or None)
