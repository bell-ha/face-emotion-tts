# E5. 개선된 안정화·문장 태깅(학습 없음)과 공개 사전학습 FER 교체 실험

## 1. 목적

E2에서 확인한 졸업논문 구현의 문제 네 가지에 대응한다. (1) 다수결이 최종 라벨에 영향을 주지 않는 죽은 코드(`final = maj if maj==ema else ema`는 항상 EMA), (2) 깜빡임 감소는 거의 모두 100 ms 유지 시간에서 나옴, (3) 문장 태그는 빈 태그가 57.7%이고 정확도가 43.5%(이벤트 confidence가 전송 시점 값으로 고정되어, conf<0.5 이벤트 뒤 구간이 지속 시간 계산에서 통째로 빠짐), (4) FER2013 CNN이 RAVDESS 영상에서 비중립 표정을 중립으로 많이 봄.

- **Part A (제안 방법):** 학습이 필요 없고 `face_emotion.py`/`app.py`에 바로 넣을 수 있는 (a) 안정화 규칙(합의 게이트·히스테리시스)과 (b) **문장 구간의 확률 질량을 적분하는 문장 태깅**을, 졸업논문 코드(기준선)와 비교한다. 하이퍼파라미터는 개발용 배우(01–04)에서만 고르고, 시험용 배우(05–24)에서 보고한다.
- **Part B (분류기 교체):** 시스템이 분류기에 의존하지 않는다는 점을 보이려고, 공개 사전학습 모델 HSEmotion(AffectNet 학습, CPU ONNX)을 같은 MediaPipe 얼굴 크롭에 붙여 평가하고, Part A의 방법을 그 위에도 적용해 (분류기 2) × (기준선/제안) 2×2 표를 만든다.

## 2. 데이터와 모델

| 항목 | 내용 |
|---|---|
| RAVDESS | Livingstone & Russo (2018), Zenodo 1188976, CC BY-NC-SA 4.0. Video_Speech **배우 01–24 전원**(남 홀수·여 짝수 각 12명). E2가 쓴 01–08에 더해 **09–24를 새로 받음**(`scripts/e5_download.py`: E2의 `e2_partial_unzip.py`를 그대로 import해 크기표만 늘림. zip 앞쪽 절반의 AV(01-*) 클립만 받아 CRC32 검증 후 압축 해제, 부분 zip은 삭제). calm 제외 배우당 52클립 → **1,248클립, 137,445프레임**, 얼굴 검출률 100%. |
| 분할 | **개발(dev): 배우 01–04** (남 2·여 2, 208클립; 중립 16). **시험(test): 배우 05–24** (남 10·여 10, 1,040클립; 중립 80, 나머지 6감정 각 160). 모든 하이퍼파라미터와 HSEmotion 변형 선택은 dev에서만 했다. |
| FER2013 PrivateTest | E1과 같은 HF 미러, 3,589장 (Part B 교차 확인용) |
| 기준 분류기 | 졸업논문 CNN `face/face_emotion.h5` (FER2013, 48×48 흑백, 7클래스) |
| 교체 분류기 | **HSEmotion** (Savchenko, 2022, *IEEE Trans. Affective Computing* 13(4); Savchenko, 2023, ICML). EfficientNet-B0, AffectNet 8클래스 학습. 공식 저장소 `HSE-asavchenko/face-emotion-recognition`(현 `sb-ai-lab/EmotiEffLib`)의 **ONNX 가중치**(`enet_b0_8_best_afew.onnx`, `enet_b0_8_va_mtl.onnx`)를 onnxruntime 1.31 CPU로 실행. 코드 라이선스 **Apache-2.0**. 학습 데이터 AffectNet은 연구용 라이선스이므로 상업 배포 시 별도 확인이 필요하다. |

**HSEmotion 설치 메모.** `pip install hsemotion`(PyTorch판)은 `.pt` 파일을 `torch.load(weights_only=False)`로 읽어야 하는데, 이는 외부 pickle 코드를 실행하는 방식이라 안전 정책상 쓰지 않았다(받은 .pt는 삭제). 대신 같은 저자의 ONNX 가중치를 쓰고, 전처리는 `hsemotion-onnx` 0.3.1의 `preprocess`와 똑같이 구현했다(RGB, `cv2.resize` 224×224 bilinear, /255, ImageNet 평균·표준편차). `onnxruntime`은 `numpy==1.26.4` 제약을 걸어 설치해 NumPy는 그대로다. 함께 설치된 `hsemotion`·`timm 1.0.30`·`torchvision 0.29.1`은 쓰지 않는다.

**클래스 매핑.** HSEmotion 출력 (Anger, Contempt, Disgust, Fear, Happiness, Neutral, Sadness, Surprise) 중 **Contempt는 대응 클래스가 없어 버리고**, 나머지 7개 로짓에 softmax를 다시 씌워 FER2013 순서(Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral)로 맞췄다. 6감정 평가는 시스템과 같이 Disgust→분노다. `va_mtl` 모델의 valence/arousal 두 출력은 쓰지 않는다.

**프레임별 캐시** (`scripts/e5_extract.py` → `data/ravdess/probs_e5/*.npz`): 프레임마다 MediaPipe(가장 큰 얼굴) 크롭 하나에서 CNN 7확률과 HSEmotion 두 모델의 로짓을 함께 저장한다. 배우 01–08의 CNN 확률은 E2 캐시와 **비트 단위로 같음**(최대 절대 차 0)을 확인했다.

## 3. 방법

### 3.1 기준선 (졸업논문 코드, 그대로)
E2의 `stabilizer.py`를 수정 없이 import했다. EMA α=0.8, 다수결 창 0.35 s(죽은 규칙 포함), 유지 100 ms, conf≥0.3, 전송 간격 0.2 s. 문장 태그는 `_pick_dominant_ko(conf_thresh=0.5, min_ratio=0.4)`(전송 이벤트의 지속 시간, 이벤트 confidence 고정).
공정한 비교를 위해 **B1 = 같은 규칙에서 conf_thresh×min_ratio만 dev에서 다시 고른 것**도 보고한다.

### 3.2 제안 안정화 (`scripts/e5_methods.py: ImprovedStabilizer`)
유지 시간·신뢰도·전송 간격 단계는 원 코드와 같고, "현재 후보 라벨"을 정하는 규칙만 바꾼다. 출력 라벨 c를 상태로 두고, 새 EMA argmax k로 바꾸는 조건을
- **gate**(합의 게이트): 다수결 == k 일 때만 (원 코드의 의도로 보이는 규칙을 바로잡은 것)
- **hyst**(히스테리시스): p_EMA[k] − p_EMA[c] ≥ δ 일 때만
- **gate_hyst**: 두 조건 모두
로 둔다. **dev 선택 기준(사전 고정):** dev 전환 반응 지연(E2와 같은 계단형 전환 프로토콜, 배우당 150쌍) 중앙값 ≤ 0.30 s 이고 dev 클립 정확도가 기준선보다 2%p 넘게 떨어지지 않는 후보 중 초당 라벨 전환 수가 가장 적은 것. 후보: gate, hyst/gate_hyst × δ∈{0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7}.

### 3.3 제안 문장 태깅: 확률 질량 적분 (`SummaryClient`, `ProbSummaryServer`)
1. **클라이언트**는 0.2 s(=기존 MIN_SEND_INTERVAL)마다 그 구간 얼굴 프레임들의 **원시 확률 벡터 평균**(7개 값)과 프레임 수를 보낸다.
2. **서버**는 문장 구간 [since, until]과 겹치는 요약들을 (겹친 비율 × 프레임 수)로 가중 합산해 6감정 질량 m을 만든다(Disgust 질량은 분노에 더함).
3. 중립 질량에 **중립 억제 가중치 w_neutral**(≤1)을 곱한 뒤 argmax를 태그로 쓴다. 중립이 이기면(또는 선택적으로 1등 비율이 r_min 미만이면) `""`(기본 감정)을 돌려준다.
4. **dev 선택 기준:** 6감정 **균형 정확도**(클래스별 재현율 평균). RAVDESS는 중립 클립이 7.7%뿐이라 일반 정확도를 최대화하면 "중립을 절대 고르지 않는" 쪽으로 쏠리므로 균형 정확도를 썼다. 동점이면 w=1·r_min=0에 가까운 쪽. 격자: w_neutral 0.10–1.00(0.05 간격) × r_min {0, 0.3, 0.4, 0.5}.
요약 전송은 아직 도착하지 않은 마지막 부분 구간(<0.2 s)을 빼고 계산해 실시간 조건을 그대로 흉내 냈다. 이 태깅은 안정화 규칙과 무관하다(서버가 원시 확률 요약을 받으므로).

### 3.4 평가
배우별로 52클립을 무작위 순서(시드 42+배우 번호)로 이어 붙인 연속 스트림 하나에 안정화기 하나(상태 유지). 문장 = 클립 하나, `""` = 중립으로 채점(E2와 같음). 지표: 초당 라벨 전환 수(전송 라벨, 7클래스), 클립 정확도(0.5 s 이후 전송 라벨 최빈값, 6감정), 전환 반응 지연(E2 프로토콜, 시험 배우당 150쌍=3,000쌍; 원시 출력이 B를 맞힌 클립만), 문장 태그 정확도, 균형 정확도, 빈 태그 비율, 태그 정밀도(빈 태그가 아닌 경우의 정확도).
**95% CI는 배우 단위 클러스터 부트스트랩**(시험 배우 20명 재표집, 2,000회). **짝 검정:** 태그·클립 정확도는 클립 단위 McNemar 정확 검정, 전환 수와 지연은 Wilcoxon 부호순위 검정. 차이의 CI도 같은 클러스터 부트스트랩으로 구했다.

## 4. 결과

### 4.1 Part B: 분류기 비교 (`results/hse_eval.json`, `results/hse_latency.csv`)

dev 선택: 원시 6감정 프레임 정확도 `va_mtl` 65.4% > `best_afew` 64.0% → **`enet_b0_8_va_mtl`**을 HSEmotion 대표로 썼다(이하 "HSEmotion").

| 분류기 | FER2013 PrivateTest 정확도 (7 / 6감정) | macro-F1 (7) | RAVDESS 시험 프레임 정확도 (7 / 6감정) | macro-F1 (7) | RAVDESS 시험 클립 정확도 (6감정, 원시 최빈값) | 프레임당 지연 평균 [p95] (ms) |
|---|---|---|---|---|---|---|
| FER2013-CNN (졸업논문) | **67.0% / 67.4%** | 0.640 | 38.4% / 42.9% | 0.368 | 46.5% | **6.38 [6.91]** (`model(x)` 직접 호출) |
| HSEmotion enet_b0_8_best_afew | 56.3% / 57.8% | 0.509 | 53.1% / 56.0% | 0.512 | 61.6% | 13.78 [15.05] |
| **HSEmotion enet_b0_8_va_mtl** | 57.4% / 58.9% | 0.527 | **54.1% / 56.5%** | **0.517** | **62.6%** | 13.72 [14.91] |

- RAVDESS 시험 배우에서 HSEmotion이 6감정 프레임 정확도 +13.6%p, 클립 정확도 +16.1%p 높다. CNN의 "중립 쏠림"이 거의 사라진다(분노→중립 0.50→0.17, 놀람→중립 0.43→0.08, 혐오 재현율 0.13→0.74; 그림 `fig_e5_confusion_cnn_vs_hse`).
- **대신 HSEmotion은 반대 방향 편향이 있다:** 중립 재현율이 0.89(CNN)→0.41로 떨어지고 중립 얼굴을 슬픔(0.31)·놀람(0.24)으로 본다. 두려움은 두 모델 모두 약하다(0.29 / 0.28).
- FER2013(48×48 흑백)에서는 HSEmotion이 9.6%p 낮다. 학습 영역(컬러 고해상도 AffectNet)과 다른 입력이라 예상한 결과이고, CNN은 FER2013에 맞춰 학습된 모델이다. 즉 "벤치마크 정확도"와 "실제 영상 정확도"의 순위가 뒤집힌다.
- dev(01–04)보다 시험(05–24) 정확도가 두 분류기 모두 낮다(CNN 6감정 50.3→42.9%, HSEmotion 66.1→56.5%). 배우에 따른 차이가 크다.
- 지연: MediaPipe 2.2 ms + 전처리 1.0 ms + HSEmotion 10.5 ms = 13.7 ms(약 73 fps)로, 30 fps 실시간에 충분하다. CNN(직접 호출)보다 약 2.2배 느리다. 측정 중 다른 실험이 함께 돌았다(load average 약 6–7, 10코어).

### 4.2 dev에서 고른 하이퍼파라미터 (`results/dev_grid_*.csv`)

| 분류기 | 안정화 규칙 | 기준선 태그 재조정(B1) | 제안 태깅(T2) |
|---|---|---|---|
| FER2013-CNN | hyst, δ=0.5 | conf_thresh 0.3, min_ratio 0.4 | **w_neutral = 0.25**, r_min = 0 |
| HSEmotion | gate_hyst, δ=0.5 | conf_thresh 0.3, min_ratio 0.3 | **w_neutral = 0.90**, r_min = 0 |

- δ=0.6 이상은 dev 클립 정확도가 2%p 넘게 떨어지거나(CNN hyst 0.6: 0.510 vs 0.524) 지연 예산을 넘어 탈락했다. 즉 선택은 격자 끝이 아니라 제약에서 결정됐다.
- 중립 억제 가중치는 분류기의 편향을 그대로 반영한다. 중립으로 쏠리는 CNN은 0.25(강한 억제), 중립을 덜 고르는 HSEmotion은 0.90(거의 억제 없음).
- r_min은 두 분류기 모두 0이 골라졌다(문턱이 오히려 빈 태그만 늘림).

### 4.3 Part A: 시험 배우 05–24 결과, FER2013-CNN (`results/e5_summary.json` → `cnn.test`)

**문장 태깅** (1,040문장, [ ]는 배우 클러스터 부트스트랩 95% CI)

| 방법 | 태그 정확도 | 균형 정확도 | 빈 태그 비율 | 태그 정밀도 | B0 대비 정확도 차 | McNemar p (vs B0) |
|---|---|---|---|---|---|---|
| B0 졸업논문 (0.5, 0.4) | 37.2% [33.2, 41.3] | 45.2% [41.3, 49.0] | 57.6% [51.6, 63.2] | 69.6% [61.4, 77.1] | – | – |
| B1 졸업논문 규칙, dev 재조정 (0.3, 0.4) | 42.6% [38.3, 47.2] | 50.2% [46.3, 54.1] | 46.5% [41.3, 51.6] | 65.3% [57.7, 72.5] | +5.4%p [3.5, 7.6] | 5.8×10⁻¹² |
| T1 확률 질량 (w=1, 조정 없음) | 46.3% [41.5, 50.9] | 52.9% [48.9, 56.7] | 41.5% [35.6, 47.2] | 66.0% [58.4, 73.4] | +9.0%p [6.2, 12.7] | 3.7×10⁻¹⁹ |
| **T2 확률 질량 + 중립 억제 (w=0.25)** | **56.3% [50.8, 61.6]** | **58.4% [52.2, 65.0]** | **9.8% [6.5, 13.1]** | 57.1% [50.8, 63.9] | **+19.0%p [14.8, 23.3]** | 1.7×10⁻³⁸ |

- T2 vs B1(기준선을 똑같이 dev에서 조정한 경우): 정확도 +13.7%p [10.4, 16.7], 균형 정확도 +8.3%p [4.3, 12.2], McNemar p=7.9×10⁻²⁵. 개선은 파라미터 조정만으로 설명되지 않는다.
- 조정 없는 T1(w=1)만으로도 +9.0%p: "전송 이벤트의 고정 confidence × 지속 시간" 대신 확률 질량을 적분하는 것 자체가 효과가 있다.
- **대가(부정적 결과):** 태그 정밀도가 69.6%→57.1%로 떨어지고, 중립 문장의 재현율이 100%→61%로 떨어진다(중립 80문장 중 26개가 슬픔으로 태그됨). 중립 억제는 분노·놀람 일부도 슬픔으로 보낸다(분노 문장 320개 중 111개가 슬픔). 태그를 비우지 않는 대신 틀린 태그가 늘어나는 교환이다(그림 `fig_e5_tag_tradeoff`, `fig_e5_tag_recall_per_class`).
- dev 균형 정확도(71.6%)보다 시험(58.4%)이 낮다. B0도 dev보다 시험에서 낮으므로 배우 차이가 크다.

**안정화** (1,040클립, 전환 지연은 3,000쌍 중 직전 라벨이 이미 B가 아닌 쌍)

| 방법 | 라벨 전환 (회/s) | 클립 정확도 (6감정) | 클립당 전송 이벤트 | 전환 지연 최초 도달 중앙값 [IQR] (s) | 평균 (s) | 안정 도달 중앙값 (s) | 도달률 |
|---|---|---|---|---|---|---|---|
| 기준선 (코드 그대로) | 1.16 [1.08, 1.24] | 46.4% [42.7, 50.2] | 4.28 | 0.117 [0.100–0.434] | 0.292 | 0.167 | 100% |
| 합의 게이트만 (참고) | 1.01 [0.94, 1.07] | 48.5% [44.6, 52.2] | 3.73 | 0.267 [0.234–0.501] | 0.411 | 0.267 | 100% |
| **제안: hyst δ=0.5** | **0.65 [0.58, 0.73]** | 47.0% [42.3, 51.5] | 2.42 | 0.234 [0.100–0.534] | 0.432 | 0.267 | 98.9% |

- 전환 수 −0.51회/s [−0.56, −0.45] (−44%), Wilcoxon p=2×10⁻¹⁰⁹. 클립 정확도 차 +0.6%p [−1.5, +2.5], McNemar p=0.62(차이 없음).
- 대가: 최초 도달 지연 중앙값 +33 ms(짝 차이 IQR 0–100 ms, 평균 0.29→0.43 s), 1.1%의 전환은 클립(약 3 s) 안에 새 감정에 도달하지 못했다.
- 합의 게이트(죽은 다수결을 바로잡은 것)만으로는 전환 −13%에 그치고 지연이 0.267 s로 늘어난다. 깜빡임 감소에는 확률 차 히스테리시스가 더 효율적이었다.

### 4.4 2×2: 분류기 × 방법 (시험 배우 05–24, 그림 `fig_e5_2x2`)

"기준선" = 졸업논문 안정화 + `_pick_dominant_ko(0.5, 0.4)`, "제안" = dev에서 고른 안정화 규칙 + 확률 질량 태깅(T2).

| 분류기 | 방법 | 태그 정확도 | 균형 정확도 | 빈 태그 | 태그 정밀도 | 전환 (회/s) | 클립 정확도 | 지연 중앙값 (s) |
|---|---|---|---|---|---|---|---|---|
| FER2013-CNN | 기준선 | 37.2% [33.2, 41.3] | 45.2% | 57.6% | 69.6% | 1.16 | 46.4% | 0.117 |
| FER2013-CNN | 제안 | 56.3% [50.8, 61.6] | 58.4% | 9.8% | 57.1% | 0.65 | 47.0% | 0.234 |
| HSEmotion | 기준선 | 53.0% [46.4, 59.0] | 53.3% | 26.9% | 66.6% | 1.07 | 62.0% | 0.100 |
| HSEmotion | **제안** | **64.0% [57.3, 70.3]** | **61.7%** | **7.6%** | 65.3% | **0.50** | 61.4% | 0.267 |

- HSEmotion에서 제안 태깅: 정확도 +11.1%p [7.9, 14.6], McNemar p=3×10⁻²³. 같은 규칙을 dev 재조정한 B1(60.7%) 대비로는 +3.4%p [1.2, 5.6], p=1.6×10⁻⁵로 차이가 작아진다. 중립 억제가 거의 없으므로(w=0.9) 이득은 주로 "질량 적분"에서 온다. 정밀도는 거의 유지된다(66.6→65.3%).
- HSEmotion 안정화(gate_hyst δ=0.5): 전환 1.07→0.50회/s(−53%, 차 −0.57 [−0.64, −0.49]), 클립 정확도 차 −0.6%p [−2.2, +1.1](McNemar p=0.61), 지연 중앙값 +133 ms(IQR 133–167 ms), 도달률 98.3%.
- **분류기 교체와 방법 개선은 서로 더해진다.** 태그 정확도 37.2%(원 시스템) → 56.3%(방법만) / 53.0%(분류기만) → 64.0%(둘 다). 다만 HSEmotion의 두려움 태그 재현율은 26%로 여전히 낮고, 중립 재현율도 48%다.

## 5. 그림 목록과 캡션 초안

그림 규정: 그림 안 글자 영문, 그림 제목 없음, 흑백 구분(명도·해칭·마커), Arial, PNG 300 dpi + PDF. 범례의 "baseline" = 졸업논문 코드, "proposed" = 제안 방법(dev 선택 파라미터).

| 파일 | 폭 | 캡션 초안 (국문 / 영문) |
|---|---|---|
| `figures/fig_e5_2x2` | 160 mm | <그림 X> 시험용 배우 20명(RAVDESS 1,040문장)에서 분류기(FER2013-CNN, HSEmotion)와 방법(기준선, 제안)에 따른 (a) 문장 태그 정확도, (b) 빈 태그 비율, (c) 초당 라벨 전환 수, (d) 클립 정확도. 오차 막대는 배우 단위 부트스트랩 95% 신뢰구간이다. 제안 방법은 클립 정확도를 유지하면서 태그 정확도를 높이고 빈 태그와 라벨 전환을 줄인다.<br>(Fig. X (a) Sentence-tag accuracy, (b) empty-tag rate, (c) label switches per second and (d) clip accuracy for each classifier and method on 20 held-out RAVDESS actors) |
| `figures/fig_e5_tag_tradeoff` | 160 mm | <그림 X> 문장 태그 정확도와 빈 태그 비율의 관계. (a) FER2013-CNN, (b) HSEmotion. 원은 졸업논문 규칙의 conf_thresh×min_ratio 격자, 실선은 확률 질량 방식의 중립 억제 가중치(w_neutral) 변화, 별은 개발용 배우에서 고른 설정이다(시험용 배우 결과).<br>(Fig. X Sentence-tag accuracy versus empty-tag rate for the thesis rule and the probability-mass rule: (a) FER2013-CNN, (b) HSEmotion) |
| `figures/fig_e5_confusion_cnn_vs_hse` | 160 mm | <그림 X> 시험용 배우의 RAVDESS 프레임에 대한 (a) FER2013-CNN과 (b) HSEmotion의 행 정규화 혼동행렬(원시 출력, 7클래스). CNN은 비중립 표정을 중립으로, HSEmotion은 중립 표정을 슬픔·놀람으로 보는 경향이 있다.<br>(Fig. X Row-normalized per-frame confusion matrices on held-out RAVDESS actors: (a) FER2013-CNN, (b) HSEmotion) |
| `figures/fig_e5_tag_recall_per_class` | 76 mm | <그림 X> 감정별 문장 태그 재현율(시험용 배우). \*분노에는 혐오 문장이 포함된다(시스템 매핑 Disgust→분노). 제안 방법은 비중립 감정의 재현율을 높이지만 중립 재현율은 낮춘다.<br>(Fig. X Per-emotion sentence-tag recall on held-out actors; *Angry includes Disgust sentences) |

## 6. 구현 반영 방법 (아직 소스는 고치지 않음)

### 6.1 `face/face_emotion.py`
1. **최종 라벨 규칙 교체** (현재 `final_idx = majority_idx if majority_idx == ema_label_idx else ema_label_idx` 한 줄). 전역 `out_idx = None`, `HYST_DELTA = 0.5`, `USE_GATE = False`(CNN) / `True`(HSEmotion)를 두고:
   ```python
   k = int(np.argmax(prob_ema)); pe = prob_ema[0]
   if out_idx is None:
       out_idx = k
   elif k != out_idx and (not USE_GATE or majority_idx == k) and pe[k] - pe[out_idx] >= HYST_DELTA:
       out_idx = k
   final_idx = out_idx
   ```
   뒤의 유지 시간·신뢰도·전송 간격 코드는 그대로 둔다.
2. **확률 요약 전송 추가.** 루프 밖에 `sum_acc = np.zeros(7); sum_n = 0; sum_t0 = time.time()`. 얼굴이 분류된 프레임마다 `sum_acc += probs[0]; sum_n += 1`(EMA가 아니라 **원시** `probs`). 매 프레임 끝에서 `time.time() - sum_t0 >= 0.2`이면
   `post_json(f"{ORPHEUS_URL}/set_emotion_probs", {"t0": int(sum_t0*1000), "t1": int(now*1000), "n": sum_n, "probs": (sum_acc/max(sum_n,1)).round(4).tolist(), "labels": EMOTION_LABELS})` 후 초기화(`sum_n == 0`이면 보내지 않음). 초당 5회, 요청당 약 200바이트. 기존 `/set_emotion` 이벤트(라벨 전환)는 UI 표시용으로 그대로 둔다(선택: payload에 `"probs": prob_ema[0].tolist()`를 덧붙일 수 있음).
3. (E1 결과) `model.predict(inp, verbose=0)` → `model(inp, training=False).numpy()`로 바꾸면 CNN 지연이 약 26 ms → 4 ms.
4. **HSEmotion으로 교체할 때:** `onnxruntime.InferenceSession("enet_b0_8_va_mtl.onnx", providers=["CPUExecutionProvider"])`, 전처리 `e5_extract.hse_preprocess`(BGR→RGB, 224×224, /255, ImageNet 정규화), 출력 앞 8개 로짓에서 `[0,2,3,4,6,7,5]` 순서로 7개를 골라 softmax(Contempt 제외) → 이후 코드는 같다. `USE_GATE=True`, 서버 `W_NEUTRAL=0.9`.

### 6.2 `tts/app.py`
1. `PROB_HISTORY = deque(maxlen=3000)`(0.2 s × 3000 = 10분) 추가.
2. 새 라우트 `POST /set_emotion_probs`: `{"t0","t1","n","probs"}`를 받아 `(t0, t1, n, np.array(probs))`로 저장.
3. 새 함수(`e5_methods.ProbSummaryServer.pick`과 같음):
   ```python
   KO6 = ['분노','두려움','행복','슬픔','놀람','중립']
   def _pick_dominant_mass(since, until, w_neutral=W_NEUTRAL):
       m = dict.fromkeys(KO6, 0.0)
       for a, b, n, p in PROB_HISTORY:
           ov = min(b, until) - max(a, since)
           if ov > 0 and b > a:
               for lab, v in zip(EMOTION_LABELS, p):
                   m[LABEL_TO_KO[lab]] += (ov / (b - a)) * n * v
       if sum(m.values()) <= 0: return ""
       m['중립'] *= w_neutral
       emo = max(m, key=m.get)
       return "" if emo == '중립' else f"<{emo}>"
   ```
   `W_NEUTRAL = 0.25`(FER2013-CNN) / `0.9`(HSEmotion). 분류기를 바꾸면 이 값만 다시 정한다.
4. `/emotion/resolve_one`에서 `_pick_dominant_ko(...)` 대신 `_pick_dominant_mass(since_ms, until_ms)`를 부른다. `PROB_HISTORY`가 비어 있으면(구버전 클라이언트) 기존 `_pick_dominant_ko`로 폴백한다.
5. 시계: 기존 이벤트와 같이 클라이언트 epoch ms를 쓰므로, 지금처럼 같은 PC에서 돌리는 것을 전제로 한다.

## 7. 한계

- RAVDESS는 북미 배우의 연기 표정이고, 문장 = 클립(약 3.7 s)으로 가정했다. 중립 문장이 80개뿐이라 중립 관련 수치의 불확실성이 크다. 빈 태그를 중립으로 채점한 것도 E2와 같은 가정이다.
- dev가 배우 4명(중립 16문장)뿐이라 w_neutral 선택이 불안정할 수 있다. dev→test에서 수치가 떨어졌다(CNN T2 균형 정확도 71.6→58.4%).
- 중립 억제는 "분류기가 중립을 과대 예측한다"는 사실을 이용한 사후 보정이다. 실제 사용자 영상의 감정 분포가 다르면 다시 정해야 한다. 정밀도·중립 재현율이 떨어지는 대가가 있다.
- 히스테리시스 δ=0.5는 확률 차 기준으로 큰 값이고, 평균 지연이 0.29→0.43 s(CNN)로 늘며 약 1–2%의 전환은 3 s 안에 반영되지 않는다. 문장 태그는 안정화 규칙과 무관하므로 이 지연은 화면 표시·이벤트에만 영향을 준다.
- HSEmotion의 AffectNet 학습 데이터는 연구용 라이선스다. RAVDESS가 HSEmotion 학습에 쓰였는지는 문서에 없다(AffectNet·AFEW·VGAF로 학습·미세조정했다고 적혀 있음).
- 지연은 다른 실험이 함께 도는 환경(load average 6–7)에서 쟀고, 카메라 캡처·화면 표시·네트워크 시간은 빠져 있다.
- RAVDESS 라이선스(CC BY-NC-SA 4.0)에 따라 원 영상·프레임은 싣지 않는다.

## 8. 재현

```bash
cd face-emotion-tts
.venv/bin/pip install onnxruntime -c <(echo numpy==1.26.4)
mkdir -p data/models/hsemotion && for m in enet_b0_8_best_afew enet_b0_8_va_mtl; do \
  curl -sL -o data/models/hsemotion/$m.onnx \
  https://github.com/HSE-asavchenko/face-emotion-recognition/raw/main/models/affectnet_emotions/onnx/$m.onnx; done
.venv/bin/python experiments/E5_improved/scripts/e5_download.py          # 배우 09-24 AV 클립 (약 30분)
.venv/bin/python experiments/E5_improved/scripts/e5_extract.py           # 프레임별 CNN+HSEmotion (약 45분)
.venv/bin/python experiments/E5_improved/scripts/e5_hse_eval.py          # Part B: FER2013, RAVDESS, 지연
.venv/bin/python experiments/E5_improved/scripts/e5_analyze.py           # Part A + 2x2 (약 3분)
.venv/bin/python experiments/E5_improved/scripts/e5_figures.py
```
(`e5_extract_when_ready.sh`는 다운로드와 추출을 겹쳐 돌리려고 쓴 보조 스크립트다.)

## 참고문헌 (확인용)
- Livingstone, S. R., & Russo, F. A. (2018). The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS). *PLoS ONE, 13*(5), e0196391.
- Savchenko, A. V., Savchenko, L. V., & Makarov, I. (2022). Classifying emotions and engagement in online learning based on a single facial expression recognition neural network. *IEEE Transactions on Affective Computing, 13*(4), 2132–2143.
- Savchenko, A. V. (2023). Facial expression recognition with adaptive frame rate based on multiple testing correction. *Proceedings of ICML 2023*, PMLR 202.
- Mollahosseini, A., Hasani, B., & Mahoor, M. H. (2019). AffectNet: A database for facial expression, valence, and arousal computing in the wild. *IEEE Transactions on Affective Computing, 10*(1), 18–31.
