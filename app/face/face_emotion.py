# face_emotion.py (label-only transitions to Orpheus)
# - Sends emotion transitions to Flask server (/set_emotion) using LABEL ONLY
# - Chooses the most stable label via EMA + majority + hold-time
# - 'tag' is always "" (unused); app.py will map label -> Korean emotion
#
# Usage:
#   python face_emotion.py                      # webcam 0 (OS별 기본 백엔드 자동 선택)
#   python face_emotion.py --camera 1           # 다른 카메라
#   python face_emotion.py --video clip.mp4     # 웹캠 대신 영상 파일 (데모/테스트)
#   python face_emotion.py --video clip.mp4 --realtime --headless --server http://127.0.0.1:5000

import argparse
import os
import platform
import sys
import time
import cv2
import numpy as np
from collections import deque
import mediapipe as mp
from tensorflow.keras.models import load_model

# =========================
# CLI
# =========================
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="STAGE real-time facial emotion sender")
    ap.add_argument("--camera", type=int, default=None, help="camera index (default: CAM_INDEX=0)")
    ap.add_argument("--video", type=str, default=None, help="video file path instead of webcam")
    ap.add_argument("--realtime", action="store_true",
                    help="with --video: throttle playback to the file's native FPS")
    ap.add_argument("--headless", action="store_true", help="no preview window (cv2.imshow off)")
    ap.add_argument("--server", type=str, default=None, help="Flask server URL (overrides discovery)")
    ap.add_argument("--no-send", action="store_true", help="do not POST to the server (local test)")
    return ap.parse_args(argv)

ARGS = parse_args()

# =========================
# Server discovery & sender
# =========================
DEFAULT_CANDIDATES = [
    os.getenv("ORPHEUS_URL", ""),              # highest priority if set
    "http://127.0.0.1:5000",
    "http://localhost:5000",
    "http://host.docker.internal:5000",
]

try:
    import requests
    HAVE_REQ = True
except Exception:
    HAVE_REQ = False
    import json as _json
    import urllib.request as _ur

def _try_ping(url: str, timeout=0.3) -> bool:
    try:
        if HAVE_REQ:
            r = requests.get(url + "/get_config", timeout=timeout)
            return r.ok
        else:
            req = _ur.Request(url + "/get_config")
            with _ur.urlopen(req, timeout=timeout) as _:
                return True
    except Exception:
        return False

def choose_server_url() -> str:
    tried = []
    for cand in DEFAULT_CANDIDATES:
        if not cand:
            continue
        ok = _try_ping(cand, timeout=0.3)
        tried.append((cand, ok))
        if ok:
            return cand
    # fallback: first non-empty candidate
    for cand, _ in tried:
        if cand:
            return cand
    return "http://127.0.0.1:5005"

ORPHEUS_URL = ARGS.server.rstrip("/") if ARGS.server else choose_server_url()

def post_json(url, payload, timeout=0.3):
    if ARGS.no_send:
        return
    try:
        if HAVE_REQ:
            requests.post(url, json=payload, timeout=timeout)
        else:
            data = _json.dumps(payload).encode("utf-8")
            req = _ur.Request(url, data=data, headers={"Content-Type": "application/json"})
            _ur.urlopen(req, timeout=timeout).read()
    except Exception:
        # suppress network errors (best-effort sender)
        pass

# ================
# Config
# ================
CAM_INDEX = 0                # 0: 기본/드로이드캠
FRAME_W, FRAME_H = 640, 480
WIN_SEC = 0.35              # majority window(sec)
EMA_ALPHA = 0.8             # prob EMA
CONF_THRESHOLD = 0.3        # 최소 신뢰도(테스트 편의상 살짝 낮춤)
MODEL_PATH = "face_emotion.h5"

# Transition sending
MIN_HOLD_MS = 100           # 같은 label이 이 시간 이상 유지될 때만 전송
MIN_SEND_INTERVAL = 0.2     # 연속 전송 최소 간격(sec)
VERBOSE_LOG = True          # 전송/연결/디버그 로그

# Emotion labels (model order)
EMOTION_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# =========================
# Utils
# =========================
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

# =========================
# Load model
# =========================
_model_file = MODEL_PATH
if not os.path.exists(_model_file):
    # 다른 폴더에서 실행해도 스크립트 옆의 모델을 찾도록
    _model_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), MODEL_PATH)
if not os.path.exists(_model_file):
    raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {MODEL_PATH}")
model = load_model(_model_file)

# =========================
# MediaPipe detect
# =========================
mp_face_detection = mp.solutions.face_detection
detector = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

# =========================
# Video capture
# =========================
def camera_backend():
    """OS별 카메라 백엔드: Windows=MSMF, macOS=AVFoundation, 그 외=OpenCV 기본."""
    system = platform.system()
    if system == "Windows":
        return cv2.CAP_MSMF
    if system == "Darwin":
        return cv2.CAP_AVFOUNDATION
    return cv2.CAP_ANY

if ARGS.video:
    if not os.path.exists(ARGS.video):
        raise FileNotFoundError(f"영상 파일을 찾을 수 없습니다: {ARGS.video}")
    cap = cv2.VideoCapture(ARGS.video)
    VIDEO_FPS = cap.get(cv2.CAP_PROP_FPS) or 30.0
else:
    cam_index = CAM_INDEX if ARGS.camera is None else ARGS.camera
    cap = cv2.VideoCapture(cam_index, camera_backend())
    if not cap.isOpened():  # 지정 백엔드 실패 시 기본 백엔드로 재시도
        cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        sys.exit(f"카메라 {cam_index}를 열 수 없습니다. (macOS: 시스템 설정 > 개인정보 보호 > 카메라 권한 확인)")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    VIDEO_FPS = None

# =========================
# State
# =========================
prob_ema = None
history = deque(maxlen=30)  # 30fps 기준 1s buffer, recent WIN_SEC만 사용
fps_smooth = 0.0
t_prev = time.time()

# label-based transition state (tag 제거)
stable_label = None
stable_conf = 0.0
stable_start_ms = None

last_sent_label = None
last_sent_time = 0.0

if VERBOSE_LOG:
    print(f"[info] Orpheus server: {ORPHEUS_URL}")
    print(f"[info] Source: {'video ' + ARGS.video if ARGS.video else 'camera'}")
    if not ARGS.headless:
        print("[info] Press 'e' to force-send current stable label, 'q' to quit.")

def maybe_send_transition(label: str, conf: float, force=False):
    """Send when stable label transition happens (or force=True)."""
    global last_sent_label, last_sent_time
    now = time.time()
    if not force:
        if (now - last_sent_time) < MIN_SEND_INTERVAL:
            return
        if label == last_sent_label:
            return
    payload = {
        "t": int(now * 1000),   # epoch ms
        "label": label,         # e.g., "Happy"
        "tag": "",              # ← always empty (no <laugh>/<gasp> etc.)
        "confidence": float(conf)
    }
    post_json(f"{ORPHEUS_URL}/set_emotion", payload, timeout=0.3)
    last_sent_label = label
    last_sent_time = now
    if VERBOSE_LOG:
        print(f"[send] {payload}")

while True:
    t_frame_start = time.time()
    ok, frame = cap.read()
    if not ok:
        break

    # FPS
    t_now = time.time()
    fps = 1.0 / max(1e-6, (t_now - t_prev))
    t_prev = t_now
    fps_smooth = fps_smooth * 0.9 + fps * 0.1

    # Detect
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = detector.process(rgb)

    current_label = "no face"
    current_conf = 0.0

    if results.detections:
        # pick largest face
        h, w = frame.shape[:2]
        best = None
        best_area = -1
        for det in results.detections:
            bbox = det.location_data.relative_bounding_box
            x = int(bbox.xmin * w); y = int(bbox.ymin * h)
            bw = int(bbox.width * w); bh = int(bbox.height * h)
            area = max(0, bw) * max(0, bh)
            if area > best_area:
                best_area = area
                best = (x, y, bw, bh)
        x, y, bw, bh = best
        roi = safe_crop(frame, x, y, bw, bh)

        if roi is not None:
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = x1 + roi.shape[1], y1 + roi.shape[0]

            # classify
            inp = preprocess_face(roi)
            probs = model.predict(inp, verbose=0)  # (1, C)

            # smooth (EMA)
            if prob_ema is None:
                prob_ema = probs
            else:
                prob_ema = EMA_ALPHA * probs + (1.0 - EMA_ALPHA) * prob_ema

            # majority(last WIN_SEC)
            history.append(int(np.argmax(probs)))
            window_len = max(1, int(WIN_SEC * max(1.0, fps_smooth)))
            recent = list(history)[-window_len:]
            if recent:
                vals, counts = np.unique(recent, return_counts=True)
                majority_idx = int(vals[np.argmax(counts)])
            else:
                majority_idx = int(np.argmax(prob_ema))

            # final label (prefer agreement)
            ema_label_idx = int(np.argmax(prob_ema))
            final_idx = majority_idx if majority_idx == ema_label_idx else ema_label_idx
            current_label = EMOTION_LABELS[final_idx]
            current_conf = float(prob_ema[0, final_idx])

            # draw bbox & label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            disp = f"{current_label} ({current_conf:.2f})"
            cv2.putText(frame, disp, (x1, max(20, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    else:
        # no face → treat as neutral candidate (won't be sent unless stable & conf)
        current_label = "Neutral"
        current_conf = 0.0

    # =========================
    # Label-based Transition stability
    # =========================
    now_ms = int(time.time() * 1000)
    candidate_label = current_label
    candidate_conf = current_conf

    # update stable
    if candidate_label == stable_label:
        if stable_start_ms is None:
            stable_start_ms = now_ms
        if candidate_conf > stable_conf:
            stable_conf = candidate_conf
    else:
        stable_label = candidate_label
        stable_conf = candidate_conf
        stable_start_ms = now_ms

    held_ms = (now_ms - (stable_start_ms or now_ms))
    if held_ms >= MIN_HOLD_MS and stable_conf >= CONF_THRESHOLD and stable_label not in (None, "no face"):
        # stable enough → maybe send
        maybe_send_transition(stable_label, stable_conf)

    # =========================
    # HUD (debug)
    # =========================
    H, W = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (W, 50), (0, 0, 0), -1)
    cv2.putText(frame, f"FPS: {fps_smooth:.1f}", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 255, 200), 2)
    hud2 = f"Stable: {stable_label or 'None'}  conf:{stable_conf:.2f}  held:{held_ms}ms"
    cv2.putText(frame, hud2, (10, 44),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 255), 2)

    if ARGS.video and ARGS.realtime and VIDEO_FPS:
        # 영상 파일을 원래 속도로 재생 (타임스탬프가 실제 시간과 맞도록)
        wait = (1.0 / VIDEO_FPS) - (time.time() - t_frame_start)
        if wait > 0:
            time.sleep(wait)

    if ARGS.headless:
        continue

    cv2.imshow("Emotion (press e: force send, q: quit)", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('e'):
        # force send current stable state for quick end-to-end testing
        label_to_send = stable_label or current_label
        maybe_send_transition(label_to_send, max(stable_conf, 0.0), force=True)

cap.release()
if not ARGS.headless:
    cv2.destroyAllWindows()
