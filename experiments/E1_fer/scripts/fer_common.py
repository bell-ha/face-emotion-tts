"""Shared FER pipeline pieces, copied faithfully from face/face_emotion.py.

(face/ 소스는 수정하지 않고 동일 로직을 복사해 사용)
"""
import os
import random

import cv2
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
MODEL_PATH = os.path.join(ROOT, "face", "face_emotion.h5")

EMOTION_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
KO_LABELS = ['분노', '혐오', '두려움', '행복', '슬픔', '놀람', '중립']
# tts/app.py LABEL_TO_KO (Disgust -> 분노)
LABEL_TO_KO = {
    "Happy": "행복", "Sad": "슬픔", "Angry": "분노", "Disgust": "분노",
    "Surprise": "놀람", "Fear": "두려움", "Neutral": "중립", "no face": "중립",
}
KO6 = ['분노', '두려움', '행복', '슬픔', '놀람', '중립']

SEED = 42


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except Exception:
        pass


def load_fer_model():
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    from tensorflow.keras.models import load_model
    return load_model(MODEL_PATH)


def safe_crop(frame, x, y, w, h):
    H, W = frame.shape[:2]
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(W, x + w), min(H, y + h)
    if x2 - x1 < 10 or y2 - y1 < 10:
        return None
    return frame[y1:y2, x1:x2]


def preprocess_face(face_bgr):
    """Model input: 48x48 grayscale, scaled to [0,1]."""
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (48, 48), interpolation=cv2.INTER_AREA)
    arr = resized.astype(np.float32) / 255.0
    arr = np.expand_dims(arr, axis=(0, -1))  # (1,48,48,1)
    return arr


def make_detector():
    import mediapipe as mp
    return mp.solutions.face_detection.FaceDetection(min_detection_confidence=0.5)


def detect_largest(detector, frame_bgr):
    """Same as face_emotion.py: RGB convert, process, pick largest bbox.
    Returns (roi, (x,y,w,h)) or (None, None)."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    results = detector.process(rgb)
    if not results.detections:
        return None, None
    h, w = frame_bgr.shape[:2]
    best, best_area = None, -1
    for det in results.detections:
        bbox = det.location_data.relative_bounding_box
        x = int(bbox.xmin * w); y = int(bbox.ymin * h)
        bw = int(bbox.width * w); bh = int(bbox.height * h)
        area = max(0, bw) * max(0, bh)
        if area > best_area:
            best_area = area
            best = (x, y, bw, bh)
    x, y, bw, bh = best
    roi = safe_crop(frame_bgr, x, y, bw, bh)
    return roi, best


def setup_korean_font():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # Journal rule (coordinator, 2026-10-12): all in-figure text in English -> Latin sans font
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['figure.dpi'] = 100
    # 2-column layout: column ~76 mm (3.0 in), full width 160 mm (6.3 in); printed text 7-9 pt
    plt.rcParams['font.size'] = 8
    plt.rcParams['axes.labelsize'] = 8
    plt.rcParams['xtick.labelsize'] = 7
    plt.rcParams['ytick.labelsize'] = 7
    plt.rcParams['legend.fontsize'] = 7
    plt.rcParams['axes.titlesize'] = 8
    plt.rcParams['hatch.linewidth'] = 0.6
    plt.rcParams['pdf.fonttype'] = 42
    return plt


def savefig(fig, path_noext):
    fig.savefig(path_noext + ".png", dpi=300, bbox_inches="tight")
    fig.savefig(path_noext + ".pdf", bbox_inches="tight")
