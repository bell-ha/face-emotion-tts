# E1. FER 모델 평가 (FER2013 · 지연 시간 · 표정 사진)

## 1. 목적

STAGE의 표정 인식 모듈(`app/face/face_emotion.h5`)을 (1) 공개 벤치마크 FER2013의 PublicTest/PrivateTest로 정량 평가하고, (2) Apple M1 Max CPU에서 프레임당 처리 지연(MediaPipe 얼굴 검출 + 전처리 + CNN)을 측정하며, (3) 제1저자의 표정 사진 7장에 대한 분류 결과를 정성적으로 확인한다.

## 2. 데이터

| 데이터 | 출처 | 라이선스 | 개수 |
|---|---|---|---|
| FER2013 | Goodfellow et al. (2013), ICML 2013 Challenges in Representation Learning. HuggingFace 미러 `Aaryan333/fer2013_train_publicTest_privateTest` (parquet, 48×48 PNG) | 원 Kaggle 대회 데이터(연구용). 미러에 별도 라이선스 표기 없음 | Training 28,709 / PublicTest 3,589 / PrivateTest 3,589 |
| 제1저자의 표정 사진 | `assets/figures/{angry,disgust,fear,happy,sad,surprise,neutral}.png` (구 앱 실행 화면 캡처, 제1저자 본인) | 제1저자 소유 | 7장 |
| RAVDESS 프레임(지연 측정용) | Livingstone & Russo (2018), Zenodo 10.5281/zenodo.1188976 | CC BY-NC-SA 4.0 | 영상 60개에서 중간 프레임 1장씩(1280×720) |

**데이터 검증.** 미러의 라벨 이름은 `0 Angry, 1 Disgust, 2 Fear, 3 Happy, 4 Sad, 5 Surprise, 6 Neutral`로 표준 FER2013 순서와 같고, 모델 출력 순서와도 같다. 분할별 클래스 수가 표준 FER2013 값과 정확히 일치한다(PublicTest 467/56/496/895/653/415/607, PrivateTest 491/55/528/879/594/416/626, Training 3,995/436/4,097/7,215/4,830/3,171/4,965). 이미지는 48×48 무손실 PNG(RGB 3채널)이며, 35,887장 모두 R=G=B(회색조)임을 확인해 첫 채널을 그대로 썼다. 입력은 `/255` 정규화이며 모델 학습 때의 전처리와 같다.

### 클래스 분포 (`results/class_distribution.csv`, 그림 `fig_e1_class_distribution`)

| 감정 | Training | PublicTest | PrivateTest |
|---|---|---|---|
| 분노 | 3,995 (13.9%) | 467 (13.0%) | 491 (13.7%) |
| 혐오 | 436 (1.5%) | 56 (1.6%) | 55 (1.5%) |
| 두려움 | 4,097 (14.3%) | 496 (13.8%) | 528 (14.7%) |
| 행복 | 7,215 (25.1%) | 895 (24.9%) | 879 (24.5%) |
| 슬픔 | 4,830 (16.8%) | 653 (18.2%) | 594 (16.6%) |
| 놀람 | 3,171 (11.0%) | 415 (11.6%) | 416 (11.6%) |
| 중립 | 4,965 (17.3%) | 607 (16.9%) | 626 (17.4%) |

## 3. 절차

1. **FER2013 평가** (`scripts/e1_eval_fer2013.py`): 각 분할 전체를 `model.predict(batch_size=256)`로 분류한다. 정확도, macro-F1, weighted-F1, 클래스별 precision/recall/F1, 혼동행렬을 계산한다. 시스템이 실제로 쓰는 6감정 매핑(`app/tts/app.py`의 `LABEL_TO_KO`: Disgust→분노)을 정답·예측 양쪽에 적용한 6클래스 정확도도 계산한다. 학습 데이터와 겹치는지 보려고 Training 분할도 평가했다.
2. **지연 측정** (`scripts/e1_latency.py`): `face_emotion.py`와 같은 단계(BGR→RGB, MediaPipe FaceDetection `min_detection_confidence=0.5`, 가장 큰 얼굴 선택·크롭 → 회색조, 48×48 `INTER_AREA` 리사이즈, /255 → CNN)를 프레임마다 `time.perf_counter()`로 잰다. 워밍업 50회 후 **500회** 반복한다. CNN 호출은 졸업논문 코드의 `model.predict(x, verbose=0)`과 직접 호출 `model(x, training=False)`를 따로 잰다(현재 앱은 직접 호출). 프레임은 (a) 표정 사진 7장(HUD를 잘라내고 녹색 박스를 지운 뒤, 앱 캡처 크기 640×480으로 리사이즈)과 (b) RAVDESS 영상 프레임 60장(1280×720)을 번갈아 넣는다.
3. **표정 사진** (`scripts/e1_user_photos.py`): 사진은 구 앱 화면 캡처라서 위쪽 HUD(FPS·Stable 표시), 녹색 얼굴 박스, 예측 라벨 글자가 들어 있다. 두 조건으로 돌렸다. (raw) 캡처 그대로, (cleaned) 위쪽 검은 HUD 띠를 잘라내고 녹색 박스 픽셀(G>150, G−R>60, G−B>60, 3×3 팽창)을 `cv2.inpaint`(TELEA)로 지움. 검출부터 분류까지 전 과정을 실행한다.

환경: Apple M1 Max(10코어), macOS 15(Darwin 24.6), Python 3.11, TensorFlow 2.15.1(CPU만 사용, GPU/Metal 플러그인 없음), MediaPipe 0.10.14. 시드 42.

## 4. 결과

### 4.1 FER2013 전체 지표 (`results/fer2013_metrics.json`)

| 분할 | N | 정확도 | macro-F1 | weighted-F1 | 6클래스 정확도(혐오→분노) | 6클래스 macro-F1 |
|---|---|---|---|---|---|---|
| PublicTest | 3,589 | **65.37%** | 0.620 | 0.650 | 65.92% | 0.640 |
| PrivateTest | 3,589 | **67.04%** | 0.640 | 0.665 | 67.37% | 0.652 |
| (참고) Training | 28,709 | 74.83% | 0.734 | 0.746 | 75.14% | 0.737 |

### 4.2 클래스별 지표 (`results/per_class_{publicTest,privateTest}.csv`)

| 감정 | Public P | Public R | Public F1 | Private P | Private R | Private F1 |
|---|---|---|---|---|---|---|
| 분노 | 0.626 | 0.563 | 0.593 | 0.609 | 0.530 | 0.566 |
| 혐오 | 0.767 | 0.411 | 0.535 | 0.813 | 0.473 | 0.598 |
| 두려움 | 0.553 | 0.367 | 0.441 | 0.552 | 0.371 | 0.444 |
| 행복 | 0.880 | 0.847 | 0.863 | 0.899 | 0.869 | 0.884 |
| 슬픔 | 0.541 | 0.565 | 0.553 | 0.523 | 0.588 | 0.554 |
| 놀람 | 0.724 | 0.822 | 0.770 | 0.741 | 0.798 | 0.769 |
| 중립 | 0.515 | 0.675 | 0.584 | 0.591 | 0.765 | 0.667 |

주요 혼동(PrivateTest, 행 정규화): 두려움→슬픔 0.22, 슬픔→중립 0.21, 혐오→분노 0.18, 분노→슬픔 0.17. 두려움 재현율이 가장 낮다(0.37).

### 4.3 학습·평가 데이터 중복 여부

모델이 어떤 분할로 학습됐는지 기록이 없다. 확인한 사실은 다음과 같다.
- Training 정확도 74.8%가 테스트(65–67%)보다 8–9%p 높다. Training 분할로 학습한 모델에서 흔한 차이다.
- PublicTest(65.4%)가 PrivateTest(67.0%)보다 **오히려 낮다**. PublicTest로 학습했다면 나타날 "의심스럽게 높은" 값은 보이지 않는다.
- 다만 졸업논문의 학습 곡선(`assets/figures/accuracycurves.png`)에서 마지막 검증 정확도가 약 0.65로, 이번 PublicTest 정확도 65.4%와 거의 같다. 따라서 **PublicTest를 검증셋(조기 종료·모델 선택)으로 썼을 가능성이 있다.** 그래서 **PrivateTest 67.0%를 대표 수치**로, PublicTest를 참고 수치로 쓴다.

### 4.4 프레임당 지연 (`results/latency.csv`, `results/latency_env.json`; 500회, 단위 ms)

| 프레임 | CNN 호출 | 검출 평균 | 전처리 평균 | CNN 평균 | **전체 평균** | 전체 중앙값 | 전체 p95 | 환산 FPS(평균 기준) |
|---|---|---|---|---|---|---|---|---|
| 표정 사진 640×480 | `model.predict` | 1.98 | 0.31 | 26.54 | **28.82** | 29.43 | 33.13 | 34.7 |
| 표정 사진 640×480 | `model(x)` | 1.98 | 0.30 | 3.87 | **6.15** | 6.18 | 7.31 | 162.7 |
| RAVDESS 1280×720 | `model.predict` | 2.27 | 0.76 | 26.26 | **29.29** | 30.13 | 33.91 | 34.1 |
| RAVDESS 1280×720 | `model(x)` | 2.28 | 0.77 | 3.92 | **6.97** | 7.04 | 8.15 | 143.4 |

- 졸업논문 코드처럼 `model.predict`를 프레임마다 부르면 CNN이 전체 지연의 약 91%를 차지한다(약 26.5 ms). 같은 계산을 `model(x, training=False)`로 직접 부르면 3.9 ms로 줄어(약 6.8배 빠름) 전체 지연이 약 6–7 ms가 된다. 차이는 `predict`가 호출마다 만드는 데이터 어댑터·콜백 오버헤드에서 온다.
- 두 방식 모두 30 fps 카메라 입력(33.3 ms 간격)을 따라갈 수 있지만, `predict` 경로는 p95가 33 ms에 가까워 여유가 거의 없다. 카메라 캡처와 화면 표시 시간은 측정에 들어 있지 않다.
- 측정하는 동안 다른 실험 작업이 함께 돌고 있었다(load average 약 4–5, 10코어). 단독으로 실행하면 수치가 조금 더 낮을 수 있다.
- 현재 앱(`app/face/face_emotion.py`)은 이 결과에 따라 `model(x, training=False)`를 직접 호출한다.

### 4.5 제1저자의 표정 사진 7장 (`results/user_photos_predictions.csv`, 그림 `fig_e1_user_photos`, `fig_e1_user_photos_v2`)

cleaned 조건(HUD·녹색 박스 제거). 7장 모두에서 얼굴이 검출됐다.

| 의도한 표정 | 예측 top-1 (확률) | 분노 | 혐오 | 두려움 | 행복 | 슬픔 | 놀람 | 중립 | 구 앱 화면 표시값 |
|---|---|---|---|---|---|---|---|---|---|
| 분노 | **분노 (0.785)** ✓ | .785 | .000 | .137 | .027 | .033 | .017 | .000 | Angry 0.79 |
| 혐오 | 슬픔 (0.573) ✗ | .214 | .047 | .088 | .000 | .573 | .000 | .076 | Disgust 1.00 |
| 두려움 | 놀람 (0.385) ✗ | .155 | .000 | .263 | .018 | .070 | .385 | .109 | Fear 0.30 |
| 행복 | **행복 (0.994)** ✓ | .000 | .000 | .000 | .994 | .001 | .001 | .004 | Happy 1.00 |
| 슬픔 | **슬픔 (0.638)** ✓ | .111 | .020 | .040 | .000 | .638 | .000 | .191 | Sad 0.60 |
| 놀람 | **놀람 (0.627)** ✓ | .109 | .000 | .243 | .012 | .003 | .627 | .005 | Surprise 0.78 |
| 중립 | **중립 (0.736)** ✓ | .025 | .000 | .028 | .003 | .000 | .208 | .736 | Neutral 0.81 |

- cleaned: 7클래스 기준 5/7, 6감정 기준 5/7이 맞았다. raw(캡처 그대로): 4/7(슬픔 사진이 혐오 0.443으로 분류됨).
- 구 앱 화면에 찍힌 값은 실시간 영상의 다른 프레임에서 EMA를 거친 값이라 단일 정지 화면 결과와 다를 수 있다. 특히 혐오 사진은 화면에 Disgust 1.00으로 표시됐지만 이 정지 화면에서는 슬픔으로 분류됐다. 표정 사진 7장은 자기 연출 표정이며, 정성적 예시일 뿐 정확도 근거가 아니다.

## 5. 그림 목록

학회 그림 규정: 그림 안 글자는 모두 영문(클래스명 Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral), 그림 제목 없음, 흑백 인쇄에서도 구분되게(명도·해칭·마커), 폭 160 mm(전체) 또는 76 mm(단), 인쇄 글자 약 7–9 pt(셀 숫자는 6 pt 내외). 글꼴 Arial, PNG 300 dpi + PDF.

| 파일 | 폭 | 내용 | 생성 스크립트 |
|---|---|---|---|
| `figures/fig_e1_confusion_fer2013` | 155 mm | FER2013 PublicTest·PrivateTest의 행 정규화 혼동행렬 | `e1_eval_fer2013.py` |
| `figures/fig_e1_class_distribution` | 73 mm | FER2013 분할별 감정 클래스 분포 | `e1_eval_fer2013.py` |
| `figures/fig_e1_per_class_f1` | 74 mm | FER2013 두 평가셋의 감정별 F1 | `e1_eval_fer2013.py` |
| `figures/fig_e1_latency` | 157 mm | M1 Max CPU 프레임당 단계별 지연(`model.predict` 대 직접 호출) | `e1_latency.py` |
| `figures/fig_e1_user_photos` | 159 mm | 표정 사진 7장의 48×48 모델 입력과 감정별 출력 확률 | `e1_user_photos.py` |
| `figures/fig_e1_user_photos_v2` | 159 mm | 같은 결과를 얼굴 크롭 확대 + 48×48 입력 삽입 + 확률 막대로 다시 그린 판(학술지 원고용) | `e1_journal_figs.py` |
| `figures/fig_fer_confusion_3` | 147 mm | 행 정규화 혼동행렬 3개: (a) FER2013 PrivateTest(CNN), (b) RAVDESS 시험 배우(CNN), (c) RAVDESS 시험 배우(HSEmotion)(학술지 원고용) | `e1_journal_figs.py` |

## 6. 한계

- 학습에 쓴 분할과 하이퍼파라미터 기록이 없다. PublicTest가 검증셋으로 쓰였을 가능성이 있어 PrivateTest를 대표 수치로 쓴다.
- FER2013은 웹에서 모은 저해상도 얼굴이다. 웹캠 정면 영상과는 분포가 다르다(E2의 RAVDESS 결과 참조).
- 지연은 다른 프로세스가 함께 도는 환경에서 측정했으며 카메라 캡처·화면 표시 시간은 빠져 있다. 표정 사진은 원래 크기가 달라 640×480으로 리사이즈해 넣었다.
- 표정 사진은 1인의 7장뿐이고 구 앱의 오버레이가 남아 있다(cleaned 조건에서도 라벨 글자 일부가 크롭 가장자리에 걸릴 수 있음).
- 환경 메모: 공용 venv의 pyarrow 26이 NumPy 2를 요구해 import가 안 됐다(TF 2.15는 NumPy 1.26 필요). parquet을 읽으려고 `pyarrow==20.0.0`으로 내렸다. `datasets 5.1.0`은 pyarrow>=24를 요구하므로, 이전부터 이 venv에서 `datasets`는 제대로 동작하지 않았을 것이다.

## 7. 재현

```bash
cd face-emotion-tts
.venv/bin/python experiments/E1_face-recognition/scripts/e1_eval_fer2013.py   # data/fer2013/*.parquet 필요
.venv/bin/python experiments/E1_face-recognition/scripts/e1_user_photos.py
.venv/bin/python experiments/E1_face-recognition/scripts/e1_latency.py        # RAVDESS 프레임은 E2 데이터가 있을 때만 포함
.venv/bin/python experiments/E1_face-recognition/scripts/e1_journal_figs.py  # fig_e1_user_photos_v2, fig_fer_confusion_3 (`e1_eval_fer2013.py`·`e1_user_photos.py` 결과와 `experiments/E5_improvements/results/{hse_eval,e5_summary}.json` 필요)
```
FER2013 받기: `https://huggingface.co/datasets/Aaryan333/fer2013_train_publicTest_privateTest/resolve/main/data/{train,publicTest,privateTest}-*.parquet` → `data/fer2013/{train,publicTest,privateTest}.parquet`.
