# app/face/ — 실시간 표정 인식 송신기

웹캠(또는 영상 파일) 프레임에서 얼굴을 찾아 7개 감정을 분류하고, 안정된 감정이 바뀔 때마다 TTS 서버(`app/tts/app.py`)의 `POST /set_emotion`으로 보낸다.

## 처리 과정

1. **얼굴 검출**: MediaPipe Face Detection(`min_detection_confidence=0.5`), 가장 큰 얼굴 하나 선택
2. **분류**: `face_emotion.h5` CNN — 48×48 흑백, /255 정규화, 클래스 순서 `Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral`
3. **안정화**
   - 확률 EMA: `EMA_ALPHA = 0.8`
   - 최근 `WIN_SEC = 0.35`초 프레임의 argmax 다수결 (EMA 결과와 일치할 때 채택, 아니면 EMA 결과)
   - 유지 시간: 같은 라벨이 `MIN_HOLD_MS = 100` ms 이상, 신뢰도 `CONF_THRESHOLD = 0.3` 이상
   - 전송 간격: `MIN_SEND_INTERVAL = 0.2`초, 라벨이 바뀔 때만 전송
4. **전송**: `{"t": epoch_ms, "label": "Happy", "tag": "", "confidence": 0.95}` (서버가 라벨을 한국어 6감정으로 매핑, Disgust→분노)

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
```

- 미리보기 창에서 `e`: 현재 안정 라벨 강제 전송, `q`: 종료
- 서버 주소는 `--server` → 환경변수 `ORPHEUS_URL` → `127.0.0.1:5000` 등 후보 순으로 정한다.
- 카메라 백엔드는 OS에 맞게 자동 선택한다: Windows `CAP_MSMF`, macOS `CAP_AVFOUNDATION`, 그 외 OpenCV 기본. 지정 백엔드가 실패하면 기본 백엔드로 다시 연다.
- macOS에서는 터미널(또는 IDE)에 카메라 권한이 필요하다: 시스템 설정 → 개인정보 보호 및 보안 → 카메라.

## 파일

| 파일 | 설명 |
|---|---|
| `face_emotion.py` | 검출·분류·안정화·전송 스크립트 |
| `face_emotion.h5` | FER2013으로 학습한 CNN (Conv 6층 + Dense 512, 파라미터 약 66.7만 개) |
| `requirements.txt` | 의존 패키지 |

모델 성능과 안정화 효과의 정량 평가는 `experiments/E1_face-recognition`, `experiments/E2_emotion-stabilization`을 참고한다.
