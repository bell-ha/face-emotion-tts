# app/face/ — 실시간 표정 인식 송신기

웹캠(또는 영상 파일) 프레임에서 얼굴을 찾아 7개 감정을 분류하고, 안정된 감정이 바뀔 때마다 TTS 서버(`app/tts/app.py`)의 `POST /set_emotion`으로 보낸다. 문장 태깅용으로 0.2초마다 원시 확률 평균을 `POST /set_emotion_probs`로 보낸다.

## 처리 과정

1. **얼굴 검출**: MediaPipe Face Detection(`min_detection_confidence=0.5`), 가장 큰 얼굴 하나 선택
2. **분류** (`--classifier`)
   - `cnn`(기본): `face_emotion.h5` CNN — 48×48 흑백, /255 정규화, 클래스 순서 `Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral`. `model(x, training=False)`로 직접 호출(E1: `model.predict`보다 빠르고 출력은 같음)
   - `hsemotion`: HSEmotion `enet_b0_8_va_mtl.onnx`(AffectNet 학습, onnxruntime CPU). 224×224 RGB, ImageNet 정규화. 8개 로짓에서 Contempt를 버리고 7개에 softmax해 위와 같은 순서로 맞춘다. 파일이 `data/models/hsemotion/`에 있어야 한다(`--hse-model`로 경로 지정, `pip install onnxruntime` 필요). 학습 데이터 AffectNet은 연구용 라이선스다.
3. **안정화** (`emotion_core.Stabilizer`)
   - 확률 EMA: `EMA_ALPHA = 0.8`
   - 라벨 규칙: 현재 출력 라벨 c를 EMA argmax k로 바꾸는 조건 (E5에서 개발용 배우로 고른 값)
     - `cnn`: **hyst** — `p_EMA[k] − p_EMA[c] ≥ 0.5`
     - `hsemotion`: **gate_hyst** — 위 조건 + 최근 `WIN_SEC = 0.35`초 argmax 다수결 == k
     - `--rule thesis`는 졸업논문 코드(EMA argmax를 그대로 씀), `--rule`/`--delta`로 바꿀 수 있다.
   - 유지 시간: 같은 라벨이 `MIN_HOLD_MS = 100` ms 이상, 신뢰도 `CONF_THRESHOLD = 0.3` 이상
   - 전송 간격: `MIN_SEND_INTERVAL = 0.2`초, 라벨이 바뀔 때만 전송
4. **전송**
   - `/set_emotion`: `{"t": epoch_ms, "label": "Happy", "tag": "", "confidence": 0.95}` (서버가 라벨을 한국어 6감정으로 매핑, Disgust→분노). 웹 UI 현재 표정·타임라인 표시와, 확률 요약이 없을 때의 문장 태깅에 쓴다.
   - `/set_emotion_probs`: 0.2초마다 `{"t0", "t1", "n", "probs"(7개, 원시 확률 평균), "labels", "classifier"}`. 얼굴 프레임이 없던 구간은 보내지 않는다. 서버가 문장 구간의 확률 질량으로 태그를 정한다(`app/tts/README.md`).

## 설치

```bash
pip install -r requirements.txt     # 저장소 루트의 .venv를 써도 된다
```

## 실행

```bash
python face_emotion.py                         # 기본 카메라(0)
python face_emotion.py --camera 1              # 다른 카메라
python face_emotion.py --video clip.mp4 --realtime   # 영상 파일을 원래 속도로 재생하며 전송
python face_emotion.py --video clip.mp4 --headless --no-send   # 창 없이, 서버 전송 없이 로컬 테스트
python face_emotion.py --server http://127.0.0.1:5000          # 서버 주소 지정
python face_emotion.py --classifier hsemotion                  # HSEmotion ONNX 분류기 (gate_hyst δ=0.5)
```

- 미리보기 창에서 `e`: 현재 안정 라벨 강제 전송, `q`: 종료
- 서버 주소는 `--server` → 환경변수 `ORPHEUS_URL`(옛 이름에서 온 변수명) → `127.0.0.1:5000` 등 후보 순으로 정한다.
- 카메라 백엔드는 OS에 맞게 자동 선택한다: Windows `CAP_MSMF`, macOS `CAP_AVFOUNDATION`, 그 외 OpenCV 기본. 지정 백엔드가 실패하면 기본 백엔드로 다시 연다.
- macOS에서는 터미널(또는 IDE)에 카메라 권한이 필요하다: 시스템 설정 → 개인정보 보호 및 보안 → 카메라.

## 파일

| 파일 | 설명 |
|---|---|
| `face_emotion.py` | 검출·분류·전송 스크립트 (카메라 루프) |
| `emotion_core.py` | 안정화(`Stabilizer`), 0.2초 확률 요약(`ProbSummary`), HSEmotion 전처리·후처리. TensorFlow·카메라 없이 import 가능(테스트용) |
| `face_emotion.h5` | FER2013으로 학습한 CNN (Conv 6층 + Dense 512, 파라미터 약 66.7만 개) |
| `requirements.txt` | 의존 패키지 |

모델 성능과 안정화 효과의 정량 평가는 `experiments/E1_face-recognition`, `experiments/E2_emotion-stabilization`, 개선안과 분류기 교체는 `experiments/E5_improvements`를 참고한다. `emotion_core.py`가 E5 참조 구현(`e5_methods.py`)과 같은 출력을 내는지는 `app/tests/test_e5_equivalence.py`가 확인한다.
