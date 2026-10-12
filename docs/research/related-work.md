# STAGE 관련 연구 목록 (실존 확인 완료)

작성일: 2026-10-12 · 작성: L1(agent-lit)

- 모든 항목은 Crossref DOI 레코드, arXiv API, PMLR·OpenReview·AAAI·ISCA, KCI·DBpia·KoreaScience 중 하나 이상에서 서지 정보(저자·제목·권호·쪽·DOI)를 직접 확인했다. ✅ 옆의 URL이 확인한 곳이다.
- 참고문헌 형식은 KOEN 2026 개정 규정(APA 7판)을 따른다. 국문 저자는 `홍길동, & 김철수.`처럼 쉼표와 `&`를 쓴다(학회 예시 "한혜정, & 한태우."). 원고에서는 국문 → 영문 순으로, 국문은 가나다순, 영문은 알파벳순으로 정렬한다. 이 문서는 주제별로 묶었다.
- 이탤릭(학술지명·권, 단행본·학술대회 Proceedings명)은 `*…*`로 표시했다. 원고에서 서식으로 바꾼다.
- 본문 인용 예는 각 항목 끝에 적었다. 저자가 21명 이상인 Goodfellow et al. (2013), Kahou et al. (2013/2016)은 APA7 규칙대로 앞 19명 … 마지막 저자로 줄였다.
- ★ = 원고에 꼭 넣을 핵심 문헌(29편). ☆ = 필요하면 추가(보조).

---

## A. 표정 인식(FER)·얼굴 검출·데이터셋

### A1 ★ FER2013
Goodfellow, I. J., Erhan, D., Carrier, P. L., Courville, A., Mirza, M., Hamner, B., Cukierski, W., Tang, Y., Thaler, D., Lee, D.-H., Zhou, Y., Ramaiah, C., Feng, F., Li, R., Wang, X., Athanasakis, D., Shawe-Taylor, J., Milakov, M., Park, J., … Bengio, Y. (2013). Challenges in representation learning: A report on three machine learning contests. In *Neural Information Processing: 20th International Conference, ICONIP 2013* (pp. 117–124). Springer. https://doi.org/10.1007/978-3-642-42051-1_16
- ✅ https://api.crossref.org/works/10.1007/978-3-642-42051-1_16 (학술지판: *Neural Networks, 64*, 59–63, 2015, https://doi.org/10.1016/j.neunet.2014.09.005)
- 요약: ICML 2013 표현학습 워크숍의 세 대회를 보고한 논문으로, 48×48 흑백 얼굴 7감정 데이터셋 FER2013을 공개했다.
- STAGE와의 차이: STAGE의 CNN은 FER2013으로 학습했으며, 정지 영상 분류 결과를 실시간 웹캠 스트림의 문장 단위 감정 태그로 확장한다.
- 인용: (Goodfellow et al., 2013)

### A2 ★ BlazeFace
Bazarevsky, V., Kartynnik, Y., Vakunov, A., Raveendran, K., & Grundmann, M. (2019). *BlazeFace: Sub-millisecond neural face detection on mobile GPUs* (arXiv:1907.05047). arXiv. https://doi.org/10.48550/arXiv.1907.05047
- ✅ https://arxiv.org/abs/1907.05047 (CVPR 2019 CV4ARVR 워크숍 발표, 별도 DOI 없음)
- 요약: 모바일 GPU에서 1 ms 미만으로 동작하는 경량 SSD 계열 얼굴 검출기를 제안했다.
- STAGE와의 차이: STAGE는 BlazeFace 기반 MediaPipe 얼굴 검출을 얼굴 영역 추출에만 쓰고, 감정 분류는 별도 CNN이 맡는다.

### A3 ★ MediaPipe
Lugaresi, C., Tang, J., Nash, H., McClanahan, C., Uboweja, E., Hays, M., Zhang, F., Chang, C.-L., Yong, M. G., Lee, J., Chang, W.-T., Hua, W., Georg, M., & Grundmann, M. (2019). *MediaPipe: A framework for building perception pipelines* (arXiv:1906.08172). arXiv. https://doi.org/10.48550/arXiv.1906.08172
- ✅ https://arxiv.org/abs/1906.08172
- 요약: 그래프 기반으로 지각(perception) 파이프라인을 구성하고 여러 플랫폼에 배포하는 프레임워크다.
- STAGE와의 차이: STAGE는 MediaPipe 얼굴 검출 모듈을 웹캠 입력 단계에 그대로 쓴다(인용: (Lugaresi et al., 2019)).

### A4 ★ 딥러닝 FER 서베이
Li, S., & Deng, W. (2022). Deep facial expression recognition: A survey. *IEEE Transactions on Affective Computing, 13*(3), 1195–1215. https://doi.org/10.1109/TAFFC.2020.2981446
- ✅ https://api.crossref.org/works/10.1109/TAFFC.2020.2981446 (온라인 선공개 2020, 권호 2022 → APA7은 2022)
- 요약: FER 데이터셋, 전처리, 정지 영상·영상 시퀀스용 딥러닝 기법을 폭넓게 정리한 서베이다.
- STAGE와의 차이: STAGE는 새 FER 모델을 제안하지 않고, 기존 CNN 출력의 시간적 안정화와 TTS 연결에 초점을 둔다.

### A5 ☆ AffectNet
Mollahosseini, A., Hasani, B., & Mahoor, M. H. (2019). AffectNet: A database for facial expression, valence, and arousal computing in the wild. *IEEE Transactions on Affective Computing, 10*(1), 18–31. https://doi.org/10.1109/TAFFC.2017.2740923
- ✅ https://api.crossref.org/works/10.1109/TAFFC.2017.2740923
- 요약: 약 100만 장의 실제 환경 얼굴 영상에 감정 범주와 valence/arousal을 주석한 대규모 데이터베이스다.
- STAGE와의 차이: STAGE는 작은 FER2013을 쓰므로, AffectNet은 도메인 일반화를 위한 후속 과제로 언급할 수 있다.

### A6 ☆ 한국인 표정 데이터 (AI Hub)
한국지능정보사회진흥원. (2020). *한국인 감정인식을 위한 복합 영상* [Data set]. AI Hub. https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=82
- ✅ 위 URL (구축년도 2020, 갱신 2023-10). 발행 기관명은 AI Hub 운영기관 기준이라 원고 전에 한 번 더 확인할 것.
- 요약: 한국인 얼굴의 감정 표정 영상 데이터셋이다.
- STAGE와의 차이: FER2013(서양인 위주)로 학습한 모델이 한국인 사용자에게 갖는 편향을 논의할 때 대안 데이터로 언급한다.

### A7 ☆ 국내 FER 연구
추송현, & 권범. (2025). 기하학적 특징과 HOG 특징을 사용한 얼굴 감정 인식. *멀티미디어학회논문지, 28*(1), 112–125. https://doi.org/10.9717/kmms.2025.28.1.112
- ✅ KCI ART003169243 https://www.kci.go.kr/kciportal/ci/sereArticleSearch/ciSereArtiView.kci?sereArticleSearchBean.artiId=ART003169243
- 요약: 눈썹·입·눈의 기하학적 특징과 눈가 주름의 HOG 질감 특징을 결합한 얼굴 감정 인식 방법을 제안했다.
- STAGE와의 차이: STAGE는 CNN 기반 인식 결과를 음성 생성까지 연결하는 파이프라인 전체를 다룬다. (졸업논문의 SVM·HOG 인용을 이 국내 문헌으로 대체 가능)

## B. 영상 FER의 시간적 처리·안정화

### B1 ★ EmotiW 2013 우승 시스템
Kahou, S. E., Pal, C., Bouthillier, X., Froumenty, P., Gülçehre, Ç., Memisevic, R., Vincent, P., Courville, A., Bengio, Y., Ferrari, R. C., Mirza, M., Jean, S., Carrier, P.-L., Dauphin, Y., Boulanger-Lewandowski, N., Aggarwal, A., Zumer, J., Lamblin, P., Raymond, J.-P., … Wu, Z. (2013). Combining modality specific deep neural networks for emotion recognition in video. In *Proceedings of the 15th ACM International Conference on Multimodal Interaction* (pp. 543–550). ACM. https://doi.org/10.1145/2522848.2531745
- ✅ https://api.crossref.org/works/10.1145/2522848.2531745
- 요약: 프레임 단위 얼굴 CNN 예측을 영상 단위로 집계하고 음성 등 다른 모달리티와 결합해 EmotiW 2013에서 우승했다.
- STAGE와의 차이: Kahou 등은 클립 전체를 오프라인으로 집계하지만, STAGE는 EMA·다수결·유지시간(hold-time)으로 온라인·저지연 안정화를 한다.

### B2 ☆ EmoNets
Kahou, S. E., Bouthillier, X., Lamblin, P., Gulcehre, C., Michalski, V., Konda, K., Jean, S., Froumenty, P., Dauphin, Y., Boulanger-Lewandowski, N., Chandias Ferrari, R., Mirza, M., Warde-Farley, D., Courville, A., Vincent, P., Memisevic, R., Pal, C., & Bengio, Y. (2016). EmoNets: Multimodal deep learning approaches for emotion recognition in video. *Journal on Multimodal User Interfaces, 10*(2), 99–111. https://doi.org/10.1007/s12193-015-0195-2
- ✅ https://api.crossref.org/works/10.1007/s12193-015-0195-2
- 요약: 프레임별 CNN 특징의 시간적 집계와 다중 모달 융합으로 영상 감정 인식 성능을 높이는 방법을 정리했다.
- STAGE와의 차이: 시간적 모델링을 학습으로 푸는 이 연구와 달리, STAGE는 학습이 필요 없는 후처리 필터로 깜빡임과 지연의 균형을 정량 평가한다.

### B3 ★ 예측 평활화(smooth predicting)
Xue, F., Tan, Z., Zhu, Y., Ma, Z., & Guo, G. (2022). Coarse-to-fine cascaded networks with smooth predicting for video facial expression recognition. In *2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW)* (pp. 2411–2417). IEEE. https://doi.org/10.1109/CVPRW56347.2022.00269
- ✅ https://api.crossref.org/works/10.1109/CVPRW56347.2022.00269 , arXiv:2203.13052
- 요약: 영상 속 표정 변화는 연속적이라는 점에 착안해, 일정 창(window)의 평균 정보를 각 프레임 예측에 더하는 smooth predicting 모듈로 프레임 단위 FER의 요동을 줄였다(ABAW3).
- STAGE와의 차이: 이 연구는 창 평균을 학습 파이프라인 안에서 오프라인으로 쓰지만, STAGE는 인과적(causal) EMA·다수결·유지시간을 실시간으로 적용하고 반응 지연까지 측정한다.
- 참고: EMA·다수결·hold-time을 FER에 직접 비교한 논문은 찾지 못했다. 이 공백 자체를 STAGE E2 실험의 기여로 서술할 수 있다.

## C. 신경망 TTS와 감정 음성 합성

### C1 ★ Tacotron
Wang, Y., Skerry-Ryan, R. J., Stanton, D., Wu, Y., Weiss, R. J., Jaitly, N., Yang, Z., Xiao, Y., Chen, Z., Bengio, S., Le, Q., Agiomyrgiannakis, Y., Clark, R., & Saurous, R. A. (2017). Tacotron: Towards end-to-end speech synthesis. In *Proceedings of Interspeech 2017* (pp. 4006–4010). ISCA. https://doi.org/10.21437/Interspeech.2017-1452
- ✅ https://api.crossref.org/works/10.21437/Interspeech.2017-1452
- 요약: 텍스트에서 스펙트로그램을 직접 생성하는 대표적인 종단간 어텐션 기반 TTS다.
- STAGE와의 차이: 배경 설명용. 졸업논문 그림 1("Tractron" 오타)의 근거 문헌으로 쓴다.

### C2 ☆ Tacotron 2
Shen, J., Pang, R., Weiss, R. J., Schuster, M., Jaitly, N., Yang, Z., Chen, Z., Zhang, Y., Wang, Y., Skerry-Ryan, R. J., Saurous, R. A., Agiomyrgiannakis, Y., & Wu, Y. (2018). Natural TTS synthesis by conditioning WaveNet on mel spectrogram predictions. In *2018 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)* (pp. 4779–4783). IEEE. https://doi.org/10.1109/ICASSP.2018.8461368
- ✅ https://api.crossref.org/works/10.1109/ICASSP.2018.8461368
- 요약: 멜 스펙트로그램 예측과 WaveNet 보코더를 결합해 사람에 가까운 자연성을 얻었다.
- STAGE와의 차이: STAGE는 신경망 TTS를 학습하지 않고 상용·온라인 TTS를 백엔드로 호출한다.

### C3 ★ Global Style Tokens (GST)
Wang, Y., Stanton, D., Zhang, Y., Skerry-Ryan, R. J., Battenberg, E., Shor, J., Xiao, Y., Ren, F., Jia, Y., & Saurous, R. A. (2018). Style tokens: Unsupervised style modeling, control and transfer in end-to-end speech synthesis. In *Proceedings of the 35th International Conference on Machine Learning* (Vol. 80, pp. 5180–5189). PMLR. https://proceedings.mlr.press/v80/wang18h.html
- ✅ https://proceedings.mlr.press/v80/wang18h.html
- 요약: Tacotron에 레이블 없이 학습되는 스타일 토큰 뱅크를 붙여 발화 스타일을 모델링·제어·전이했다.
- STAGE와의 차이: GST는 스타일 임베딩 학습이 필요하지만, STAGE는 TTS를 학습하지 않고 프롬프트와 PSOLA 운율 조작으로 감정을 제어한다.

### C4 ★ FastSpeech 2
Ren, Y., Hu, C., Tan, X., Qin, T., Zhao, S., Zhao, Z., & Liu, T.-Y. (2021). FastSpeech 2: Fast and high-quality end-to-end text to speech. In *International Conference on Learning Representations (ICLR 2021)*. https://openreview.net/forum?id=piLPYqxtWuA
- ✅ https://openreview.net/forum?id=piLPYqxtWuA
- 요약: 피치·에너지·길이를 명시적 분산 어댑터로 예측하는 비자기회귀 TTS다.
- STAGE와의 차이: FastSpeech 2는 운율을 모델 내부에서 예측하지만, STAGE는 합성 뒤 Praat PSOLA로 F0·속도·강도를 규칙 기반으로 조작한다. (졸업논문의 "AITimes, 2023" FastSpeech2 인용을 대체하는 원 출처)

### C5 ★ EmoSpeech
Diatlova, D., & Shutov, V. (2023). EmoSpeech: Guiding FastSpeech2 towards emotional text to speech. In *Proceedings of the 12th ISCA Speech Synthesis Workshop (SSW 2023)* (pp. 106–112). ISCA. https://doi.org/10.21437/SSW.2023-17
- ✅ https://doi.org/10.21437/SSW.2023-17 (Crossref 확인; 학회는 Interspeech가 아니라 SSW)
- 요약: FastSpeech 2에 감정 조건 모듈을 더하고, 감정이 문장 안 토큰에 고르지 않게 분포한다는 점을 반영해 감정 TTS 품질을 높였다.
- STAGE와의 차이: EmoSpeech는 감정 레이블로 TTS를 학습하지만, STAGE는 학습 없이 얼굴에서 추정한 감정 태그를 기존 TTS 제어 신호로 바꾼다.

### C6 ★ 운율 제어 (졸업논문 인용 유지)
Lee, Y., & Kim, T. (2019). Robust and fine-grained prosody control of end-to-end speech synthesis. In *ICASSP 2019 – 2019 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)* (pp. 5911–5915). IEEE. https://doi.org/10.1109/ICASSP.2019.8683501
- ✅ https://api.crossref.org/works/10.1109/ICASSP.2019.8683501 (pp. 5911–5915 확인)
- 요약: 음소 단위의 가변 길이 운율 임베딩으로 종단간 TTS의 운율을 세밀하고 안정적으로 제어했다.
- STAGE와의 차이: 모델 내부 임베딩 제어와 달리 STAGE는 외부 신호처리(PSOLA)와 프롬프트로 운율을 바꾼다.

### C7 ★ 감정 운율 제어 (졸업논문 인용 유지)
Sivaprasad, S., Kosgi, S., & Gandhi, V. (2021). Emotional prosody control for speech generation. In *Proceedings of Interspeech 2021* (pp. 4653–4657). ISCA. https://doi.org/10.21437/Interspeech.2021-307
- ✅ https://api.crossref.org/works/10.21437/Interspeech.2021-307 , https://www.isca-archive.org/interspeech_2021/sivaprasad21_interspeech.html (졸업논문의 4643–4647은 틀림 → 4653–4657)
- 요약: valence·arousal 같은 연속 감정 차원으로 생성 음성의 감정 운율을 제어하는 TTS를 제안했다.
- STAGE와의 차이: STAGE는 연속 차원이 아닌 이산 감정 태그를 쓰며, 감정 입력이 사용자의 표정에서 자동으로 온다.

### C8 ★ MsEmoTTS
Lei, Y., Yang, S., Wang, X., & Xie, L. (2022). MsEmoTTS: Multi-scale emotion transfer, prediction, and control for emotional speech synthesis. *IEEE/ACM Transactions on Audio, Speech, and Language Processing, 30*, 853–864. https://doi.org/10.1109/TASLP.2022.3145293
- ✅ https://api.crossref.org/works/10.1109/TASLP.2022.3145293
- 요약: 전역·발화·음절 수준의 다중 스케일 감정 표현으로 감정 전이, 텍스트 기반 예측, 강도 제어를 통합했다.
- STAGE와의 차이: STAGE는 문장 단위 이산 태그만 다루므로, 세밀한 다중 스케일 제어는 한계이자 후속 과제다.

### C9 ☆ 초기 감정 종단간 TTS
Lee, Y., Rabiee, A., & Lee, S.-Y. (2017). *Emotional end-to-end neural speech synthesizer* (arXiv:1711.05447). arXiv. https://doi.org/10.48550/arXiv.1711.05447
- ✅ https://arxiv.org/abs/1711.05447 (워크숍 게재처는 확인 못 함 → arXiv로 인용)
- 요약: Tacotron에 감정 레이블 임베딩을 넣은 초기 국내 연구진의 종단간 감정 음성합성기다.
- STAGE와의 차이: 감정 레이블을 사용자가 지정하지 않고 표정에서 자동으로 얻는다.

### C10 ★ ESD·감정 음성 변환 개관
Zhou, K., Sisman, B., Liu, R., & Li, H. (2022). Emotional voice conversion: Theory, databases and ESD. *Speech Communication, 137*, 1–18. https://doi.org/10.1016/j.specom.2021.11.006
- ✅ https://api.crossref.org/works/10.1016/j.specom.2021.11.006
- 요약: 감정 음성 변환의 이론과 데이터베이스를 정리하고, 영어·중국어 다화자 감정 음성 데이터셋 ESD를 공개했다.
- STAGE와의 차이: STAGE의 PSOLA 백엔드는 학습 없이 운율만 바꾸는 신호처리 기반 감정 변환에 해당한다.

### C11 ☆ 신경망 음성합성 서베이 (졸업논문 인용 유지)
Tan, X., Qin, T., Soong, F., & Liu, T.-Y. (2021). *A survey on neural speech synthesis* (arXiv:2106.15561). arXiv. https://doi.org/10.48550/arXiv.2106.15561
- ✅ https://arxiv.org/abs/2106.15561
- 요약: 텍스트 분석, 음향 모델, 보코더, 표현력 있는 TTS 등 신경망 음성합성 전반을 정리했다.
- STAGE와의 차이: 배경 설명용(졸업논문 본문의 "Ten et al., 2023"이 이 문헌이다).

## D. 프롬프트 기반 TTS와 사용 도구

### D1 ★ PromptTTS
Guo, Z., Leng, Y., Wu, Y., Zhao, S., & Tan, X. (2023). PromptTTS: Controllable text-to-speech with text descriptions. In *ICASSP 2023 – 2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)* (pp. 1–5). IEEE. https://doi.org/10.1109/ICASSP49357.2023.10096285
- ✅ https://api.crossref.org/works/10.1109/ICASSP49357.2023.10096285
- 요약: 성별·피치·속도·감정 등을 자연어 설명으로 지정해 음성 스타일을 제어하는 TTS다.
- STAGE와의 차이: STAGE는 프롬프트 TTS를 학습하지 않고, 얼굴 감정 태그를 자연어 지시문으로 바꿔 상용 프롬프트 TTS에 넘긴다.

### D2 ★ InstructTTS
Yang, D., Liu, S., Huang, R., Weng, C., & Meng, H. (2024). InstructTTS: Modelling expressive TTS in discrete latent space with natural language style prompt. *IEEE/ACM Transactions on Audio, Speech, and Language Processing, 32*, 2913–2925. https://doi.org/10.1109/TASLP.2024.3402088
- ✅ https://doi.org/10.1109/TASLP.2024.3402088 (Crossref 확인)
- 요약: 자연어 스타일 프롬프트와 이산 잠재공간(코덱) 모델링으로 표현력 있는 음성을 생성한다.
- STAGE와의 차이: InstructTTS의 프롬프트는 사람이 쓰지만, STAGE의 지시문은 실시간 표정 인식 결과에서 자동으로 만들어진다.

### D3 ★ OpenAI TTS 문서 (gpt-4o-mini-tts)
OpenAI. (n.d.). *Text to speech*. OpenAI API documentation. Retrieved October 12, 2026, from https://developers.openai.com/api/docs/guides/text-to-speech
- ✅ 위 URL (구 주소 platform.openai.com/docs/guides/text-to-speech에서 이동됨). gpt-4o-mini-tts와 `instructions` 파라미터가 설명되어 있다. 수시 갱신 페이지이므로 "Retrieved" 형식을 쓴다(학회 규정 허용).
- 요약: gpt-4o-mini-tts에서 instructions로 억양·감정·어조·속도 등을 자연어로 제어할 수 있다.
- STAGE와의 차이: STAGE의 상용 백엔드가 이 API다. 졸업논문 구현은 instructions를 넘기지 않았다는 점을 원고에서 바로잡아야 한다.

### D4 ★ OpenAI 오디오 모델 발표
OpenAI. (2025, March 20). *Introducing next-generation audio models in the API*. https://openai.com/index/introducing-our-next-generation-audio-models/
- ✅ 제목·날짜는 검색 결과로 확인(직접 접속은 403).
- 요약: 말하는 방식을 지시할 수 있는 gpt-4o-mini-tts를 처음 공개했다.
- STAGE와의 차이: 내부 구조가 공개되지 않은 블랙박스이므로, 졸업논문의 "mel-spectrogram 예측기 + LLM 통합 구조" 같은 서술은 근거가 없다 → 삭제하고, 감정 반영 여부는 음향 특징과 SER로 객관 검증한다.

### D5 ★ edge-tts
rany2. (n.d.). *edge-tts* [Computer software]. GitHub. Retrieved October 12, 2026, from https://github.com/rany2/edge-tts
- ✅ 위 URL (GPL-3.0, rate/volume/pitch 지원). 원고에는 실제 사용 버전(`pip show edge-tts`)을 넣는다.
- 요약: API 키 없이 Microsoft Edge 온라인 TTS 서비스를 쓰게 해 주는 Python 모듈이다.
- STAGE와의 차이: 주의 — edge-tts는 Microsoft 온라인 서비스를 호출하므로 "오프라인"이 아니다. 로컬에서 도는 것은 PSOLA 후처리뿐이다. 원고에서 "API 키가 필요 없는 무료 백엔드" 정도로 표현한다.

### D6 ★ Parselmouth
Jadoul, Y., Thompson, B., & de Boer, B. (2018). Introducing Parselmouth: A Python interface to Praat. *Journal of Phonetics, 71*, 1–15. https://doi.org/10.1016/j.wocn.2018.07.001
- ✅ https://api.crossref.org/works/10.1016/j.wocn.2018.07.001
- 요약: Praat의 C/C++ 코드를 직접 감싸 Python에서 Praat 분석·조작 기능을 쓰게 한 라이브러리다.
- STAGE와의 차이: STAGE는 Parselmouth로 F0·강도를 측정하고 PSOLA로 운율을 조작한다.

### D7 ★ Praat
Boersma, P., & Weenink, D. (2026). *Praat: Doing phonetics by computer* (Version 7.0.02) [Computer software]. https://www.praat.org
- ✅ https://www.fon.hum.uva.nl/praat/ — 버전·연도는 Parselmouth에 들어 있는 버전(`parselmouth.PRAAT_VERSION`)으로 바꾼다.
- 요약: 음성 분석·합성·조작용 표준 음성학 소프트웨어로, PSOLA 기반 피치·길이 조작을 제공한다.
- STAGE와의 차이: 운율 제어와 음향 평가가 모두 Praat 알고리즘에 기반한다.

## E. 얼굴 → 음성 (가장 가까운 선행연구)

### E1 ★ FEIM-TTS
Chu, Y., Shim, Y., & Park, U. (2025). Facial expression-enhanced TTS: Combining face representation and emotion intensity for adaptive speech. In *Computer Vision – ECCV 2024 Workshops* (Lecture Notes in Computer Science, Vol. 15637, pp. 117–129). Springer. https://doi.org/10.1007/978-3-031-91581-9_9
- ✅ https://doi.org/10.1007/978-3-031-91581-9_9 (Crossref), arXiv:2409.16203 (ECCV 2024 ABAW7 워크숍). LNCS 권 번호 15637은 Springer 페이지에서 한 번 더 확인 권장.
- 요약: 얼굴 이미지 표현과 감정 강도를 조건으로 감정 음성을 생성하는 제로샷 TTS다(서강대).
- STAGE와의 차이: FEIM-TTS는 정지 얼굴 이미지를 TTS 학습 조건으로 쓰지만, STAGE는 실시간 웹캠 표정을 시간적으로 안정화해 문장 단위 태그로 만들고 기존 TTS를 학습 없이 제어한다.

### E2 ★ DEmoFace (Emotional Face-to-Speech)
Ye, J., Cao, B., & Shan, H. (2025). Emotional face-to-speech. In *Proceedings of the 42nd International Conference on Machine Learning* (Vol. 267, pp. 71938–71958). PMLR. https://proceedings.mlr.press/v267/ye25b.html
- ✅ https://proceedings.mlr.press/v267/ye25b.html , arXiv:2502.01046
- 요약: 표정이 드러난 얼굴에서 감정 음성을 직접 합성하는 과제를 제안하고, 코덱 기반 이산 확산 트랜스포머 DEmoFace를 제시했다.
- STAGE와의 차이: DEmoFace는 대규모 생성 모델을 학습하는 종단간 방식이고, STAGE는 인식과 합성을 나눈 모듈형 실시간 파이프라인이다.

### E3 ★ FaceSpeak
Zhang, T.-H., Zhang, J., Wang, J., Qian, X., & Yin, X.-C. (2025). FaceSpeak: Expressive and high-quality speech synthesis from human portraits of different styles. *Proceedings of the AAAI Conference on Artificial Intelligence, 39*(24), 25922–25930. https://doi.org/10.1609/aaai.v39i24.34786
- ✅ https://api.crossref.org/works/10.1609/aaai.v39i24.34786 (arXiv:2501.03181)
- 요약: 실사부터 애니메이션까지 다양한 화풍의 인물 이미지에서 정체성과 감정을 추출해 표현력 있는 음성을 합성했다.
- STAGE와의 차이: FaceSpeak는 이미지에서 목소리 스타일을 추론하지만, STAGE는 음색이 아닌 사용자의 실시간 감정 상태만 반영한다.

### E4 ★ Face2Speech
Goto, S., Onishi, K., Saito, Y., Tachibana, K., & Mori, K. (2020). Face2Speech: Towards multi-speaker text-to-speech synthesis using an embedding vector predicted from a face image. In *Proceedings of Interspeech 2020* (pp. 1321–1325). ISCA. https://doi.org/10.21437/Interspeech.2020-2136
- ✅ https://doi.org/10.21437/Interspeech.2020-2136 (Crossref)
- 요약: 얼굴 이미지에서 화자 임베딩을 예측해 다화자 TTS에 넣어 얼굴에 어울리는 목소리를 만들었다.
- STAGE와의 차이: Face2Speech는 얼굴에서 정적인 화자 정체성을, STAGE는 동적인 감정을 추론한다.

### E5 ★ Face-TTS (Imaginary Voice)
Lee, J., Chung, J. S., & Chung, S.-W. (2023). Imaginary voice: Face-styled diffusion model for text-to-speech. In *ICASSP 2023 – 2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)* (pp. 1–5). IEEE. https://doi.org/10.1109/ICASSP49357.2023.10094745
- ✅ https://doi.org/10.1109/ICASSP49357.2023.10094745 (Crossref)
- 요약: 얼굴 이미지를 스타일 조건으로 쓰는 확산 기반 TTS로, 처음 보는 얼굴에도 어울리는 음성을 만든다.
- STAGE와의 차이: Face-TTS는 얼굴에서 음색을 생성하고, STAGE는 사용자 표정 변화에 따라 감정 운율을 바꾼다.

## F. 음성 감정의 음향 단서 (운율 프리셋의 이론 근거)

### F1 ★ Scherer (2003)
Scherer, K. R. (2003). Vocal communication of emotion: A review of research paradigms. *Speech Communication, 40*(1–2), 227–256. https://doi.org/10.1016/S0167-6393(02)00084-5
- ✅ https://api.crossref.org/works/10.1016/S0167-6393(02)00084-5
- 요약: 음성 감정 표현·지각 연구의 패러다임(Brunswik 렌즈 모델 등)을 검토하고 감정별 음향 패턴을 정리했다.
- STAGE와의 차이: 감정별 F0·속도·강도 조절 규칙과 음향 평가 가설의 이론 근거로 쓴다.

### F2 ★ Juslin & Laukka (2003)
Juslin, P. N., & Laukka, P. (2003). Communication of emotions in vocal expression and music performance: Different channels, same code? *Psychological Bulletin, 129*(5), 770–814. https://doi.org/10.1037/0033-2909.129.5.770
- ✅ https://api.crossref.org/works/10.1037/0033-2909.129.5.770
- 요약: 음성 연구 104편과 음악 연구 41편을 메타분석해 감정별 속도·강도·F0 수준과 변동 패턴을 정리했다.
- STAGE와의 차이: 운율 프리셋의 방향(기쁨: F0·속도 상승, 슬픔: 하강 등)을 이 결과로 정당화한다. 음악·보컬 맥락도 함께 다뤄 엔터테인먼트 학술지에 잘 맞는다.

### F3 ★ Banse & Scherer (1996)
Banse, R., & Scherer, K. R. (1996). Acoustic profiles in vocal emotion expression. *Journal of Personality and Social Psychology, 70*(3), 614–636. https://doi.org/10.1037/0022-3514.70.3.614
- ✅ https://api.crossref.org/works/10.1037/0022-3514.70.3.614
- 요약: 배우 발화 14개 감정의 음향 프로파일을 분석하고 음향 특징으로 감정을 판별할 수 있음을 보였다.
- STAGE와의 차이: 합성음의 음향 특징 차이가 이 프로파일과 맞는지 확인한다.

### F4 ☆ 기본 정서
Ekman, P. (1992). An argument for basic emotions. *Cognition and Emotion, 6*(3–4), 169–200. https://doi.org/10.1080/02699939208411068
- ✅ https://api.crossref.org/works/10.1080/02699939208411068
- 요약: 고유한 표정과 생리 신호로 구별되는 보편적 기본 정서가 있다고 주장했다.
- STAGE와의 차이: FER2013 7범주와 STAGE 감정 태그 체계의 이론 배경.

### F5 ☆ 정서 원형 모델
Russell, J. A. (1980). A circumplex model of affect. *Journal of Personality and Social Psychology, 39*(6), 1161–1178. https://doi.org/10.1037/h0077714
- ✅ https://api.crossref.org/works/10.1037/h0077714
- 요약: 정서를 쾌-불쾌(valence)와 각성(arousal)의 2차원 원형 구조로 나타냈다.
- STAGE와의 차이: 운율 파라미터 설계와 연속 감정 제어로의 확장 근거.

## G. 평가 도구·데이터

### G1 ★ emotion2vec
Ma, Z., Zheng, Z., Ye, J., Li, J., Gao, Z., Zhang, S., & Chen, X. (2024). emotion2vec: Self-supervised pre-training for speech emotion representation. In *Findings of the Association for Computational Linguistics: ACL 2024* (pp. 15747–15760). Association for Computational Linguistics. https://doi.org/10.18653/v1/2024.findings-acl.931
- ✅ https://api.crossref.org/works/10.18653/v1/2024.findings-acl.931
- 요약: 자기지도 사전학습으로 범용 음성 감정 표현을 학습한 모델로, 여러 언어·과제의 SER에서 성능이 높다.
- STAGE와의 차이: 사람 청취 평가 대신 청취자 대리 지표로 사용한다(한계로 명시).

### G2 ★ RAVDESS
Livingstone, S. R., & Russo, F. A. (2018). The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS): A dynamic, multimodal set of facial and vocal expressions in North American English. *PLOS ONE, 13*(5), Article e0196391. https://doi.org/10.1371/journal.pone.0196391
- ✅ https://api.crossref.org/works/10.1371/journal.pone.0196391
- 요약: 배우 24명이 연기한 감정 발화·노래의 시청각 데이터베이스로, 감정 범주와 강도 레이블을 제공한다.
- STAGE와의 차이: 레이블이 있는 동적 표정 입력으로 시간적 안정화(깜빡임·지연)를 평가하는 데 쓴다.

### G3 ☆ 한국어 감정 음성합성 데이터 (AI Hub)
한국지능정보사회진흥원. (2021). *감성 및 발화 스타일별 음성합성 데이터* [Data set]. AI Hub. https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=466
- ✅ 위 URL (구축 2021, 갱신 2022-07). 발행 기관명 재확인 필요.
- 요약: 감정·발화 스타일별로 녹음한 한국어 음성합성용 데이터다.
- STAGE와의 차이: 한국어 감정 음성 기준 데이터로, 후속 연구에서 SER 미세조정이나 음향 기준값 비교에 쓸 수 있다.

## H. 국내 감정 음성합성·감정 인터랙션 연구

### H1 ★ 박중희·한광희 (2023)
박중희, & 한광희. (2023). 감정별 음성분석을 통한 인공지능 TTS 도구들의 감정발화를 구분 짓는 SSML 문법 요인 설계. *한국콘텐츠학회논문지, 23*(12), 1–22. https://doi.org/10.5392/JKCA.2023.23.12.001
- ✅ https://api.crossref.org/works/10.5392/JKCA.2023.23.12.001 , KCI ART003026881
- 요약: AI Hub 성우 감정음성과 SSML로 만든 TTS 음성을 비교해 감정 구분에 중요한 음향 요인(포먼트·피치·길이·jitter·shimmer·NHR)을 찾고 감정 제어용 SSML 요인을 설계했다.
- STAGE와의 차이: 텍스트·SSML 기반 수동 감정 지정이 아니라, 웹캠 표정에서 감정을 자동 추출해 운율 제어로 연결한다.
- 주의: 졸업논문의 23(2), 148–157은 틀렸다. 23(12), 1–22가 맞다.

### H2 ★ 권새봄·김주섭 (2019)
권새봄, & 김주섭. (2019). 감정 전이와 실시간 얼굴 표정 리타게팅 기술 기반 음악 감상 경험 증진에 관한 연구. *디지털콘텐츠학회논문지, 20*(6), 1117–1124. https://doi.org/10.9728/dcs.2019.20.6.1117
- ✅ https://api.crossref.org/works/10.9728/dcs.2019.20.6.1117 , KCI ART002482558
- 요약: 음악의 감정을 표정으로 연기하는 가상 캐릭터와 함께 음악을 감상하는 시스템을 실시간 표정 리타게팅으로 만들고, 대학생 100명 실험에서 감정 유도 효과를 확인했다.
- STAGE와의 차이: 둘 다 실시간 얼굴 표정을 엔터테인먼트 경험에 쓰지만, 이 연구는 표정을 캐릭터 애니메이션으로, STAGE는 감정 음성으로 옮긴다.
- 주의: 졸업논문의 1097–1106쪽은 틀렸다. 1117–1124가 맞다.

### H3 ★ 유세희 외 (2021) 감정 SSML 처리기
유세희, 조희, 이주현, & 홍기형. (2021). 다음색 감정 음성합성 응용을 위한 감정 SSML 처리기. *한국음향학회지, 40*(5), 523–529. https://doi.org/10.7776/ASK.2021.40.5.523
- ✅ KCI ART002761158 , DOI는 doi.org → KoreaScience로 연결됨
- 요약: SSML을 감정 표현이 가능하게 확장한 감정 SSML과, 편집기·문서 생성기·파서·합성기 연동 시퀀서로 된 처리기를 개발했다.
- STAGE와의 차이: 사용자가 감정을 수동으로 마크업하는 대신, STAGE는 표정 인식으로 문장별 감정 태그를 자동으로 붙인다. 인용: (유세희 외, 2021)

### H4 ☆ 오상신 외 (2019) GST 대표 감정 스타일
오상신, 엄세연, 장인선, 안충현, & 강홍구. (2019). k-평균 알고리즘을 활용한 음성의 대표 감정 스타일 결정 방법. *한국음향학회지, 38*(5), 614–620. https://doi.org/10.7776/ASK.2019.38.5.614
- ✅ KCI ART002509826
- 요약: GST 기반 감정 음성합성에서 k-평균으로 감정별 대표 스타일 벡터를 뽑아 감정 표현도와 구별성을 높였다.
- STAGE와의 차이: 모델 내부 스타일 임베딩 개선이 아니라 학습 없는 응용 계층 제어다.

### H5 ☆ 최연주 외 (2018) 한국어 종단간 TTS
최연주, 정영문, 김영관, 서영주, & 김회린. (2018). 한국어 text-to-speech(TTS) 시스템을 위한 엔드투엔드 합성 방식 연구. *말소리와 음성과학, 10*(1), 39–48. https://doi.org/10.13064/KSSS.2018.10.1.039
- ✅ KCI ART002334302
- 요약: Tacotron 기반 한국어 종단간 TTS를 구현하고 기존 모듈형 합성의 오류 누적 문제를 검토했다.
- STAGE와의 차이: 한국어 신경망 TTS의 기술 배경.

### H6 ★ 류나영·윤수연 (2025) 한국어 감정 TTS 지각
Ryu, N., & Yun, S. (2025). Native Korean listeners' perception of emotional expression in Korean TTS. *응용언어학 [Korean Journal of Applied Linguistics], 41*(4), 101–129. https://doi.org/10.17154/kjal.2025.12.41.4.101
- ✅ KCI ART003284809 (제목이 영문만 있어 영문으로 인용. 저자 로마자 표기는 KCI 영문 저자명으로 최종 확인할 것)
- 요약: 한국어 화자 31명이 ElevenLabs와 Clova Dubbing의 감정 TTS 음성 96개를 자연스러움·억양·감정 표현력 등으로 평가하고, 감정 일치 얼굴 이미지 같은 시각 맥락의 영향을 분석했다.
- STAGE와의 차이: 상용 TTS의 한국어 감정 표현과 얼굴 단서를 다룬 가장 가까운 국내 평가 연구. STAGE는 얼굴을 지각 맥락이 아니라 감정 입력 신호로 쓴다.

### H7 ☆ 김은서 외 (2025) STT의 감정 손실
김은서, 하주혜, & 오창훈. (2025). 음성-텍스트 변환 시스템의 감정 손실과 사용자 보정 메커니즘 이해. *멀티미디어학회논문지, 28*(8), 1026–1039. https://doi.org/10.9717/kmms.2025.28.8.1026
- ✅ KCI ART003240949
- 요약: 음성이 텍스트로 바뀔 때 억양·강조·어조 같은 비언어적 감정이 중립화되고, 사용자가 기호·이모지로 이를 보정한다는 것을 실험과 인터뷰로 보였다.
- STAGE와의 차이: 반대 방향(텍스트 → 음성)에서 표정으로 감정을 복원하려는 시도로 STAGE의 동기를 뒷받침한다.

### H8 ☆ Ge·조동민 (2026) 음성 → 표정 매핑
Ge, Y.-R., & Cho, D. (2026). Design study of FACS-based emotional mapping for voice-driven virtual character facial expressions. *멀티미디어학회논문지 [Journal of Korea Multimedia Society], 29*(2), 283–298. https://doi.org/10.9717/kmms.2026.29.2.283
- ✅ KCI ART003313007 (영문 제목만 있음. 조동민의 로마자 표기는 KCI 확인 필요)
- 요약: 음성의 음향·감정 특징에서 FACS 액션 유닛을 추론해 가상 캐릭터 표정을 만들고 사용자 실험으로 검증했다.
- STAGE와의 차이: 음성 → 표정 방향인 이 연구와 정반대로 STAGE는 표정 → 음성 방향이다.

## I. 한국엔터테인먼트산업학회논문지 게재 논문 (투고 학술지 인용용)

KOEN에서 TTS·음성합성·웹캠 감정 인식을 직접 다룬 논문은 찾지 못했다. → 서론에서 "엔터테인먼트 콘텐츠 제작에서 생성형 AI 활용은 늘고 있으나(최영신, 2024; 양호 & 이인호, 2026), 창작자의 실시간 감정을 음성으로 연결하는 연구는 드물다"는 식으로 공백을 제시할 수 있다.

### I1 ★ 이은혜·김유정 (2020)
이은혜, & 김유정. (2020). 뮤지컬 노래의 극과 음악 분석을 통한 조 에스틸 보컬 기법(EVT)의 적용과 실제. *한국엔터테인먼트산업학회논문지, 14*(8), 91–102. https://doi.org/10.21184/jkeia.2020.12.14.8.91
- ✅ https://api.crossref.org/works/10.21184/jkeia.2020.12.14.8.91 , KCI ART002675254
- 요약: 뮤지컬 〈엘리자벳〉의 '아무것도'를 극 구조·가사·음악 형식으로 분석하고, 인물의 정서를 표현하기 위해 EVT의 음질과 성대·후두 조절을 적용하는 방법을 제시했다.
- STAGE와의 차이: 인물 정서에 맞춰 음색·발성을 바꾸는 인간 보컬의 원리를 보여 주며, 감정 태그별 음성 표현 파라미터 설계의 예술적 근거로 인용한다.
- 주의: 졸업논문의 14(6), 313–324는 틀렸다. 14(8), 91–102가 맞다.

### I2 ★ 최영신 (2024)
최영신. (2024). 생성형 인공지능과 대중음악 산업구조 변화. *한국엔터테인먼트산업학회논문지, 18*(9), 129–138. https://doi.org/10.21184/jkeia.2024.12.18.9.129
- ✅ https://api.crossref.org/works/10.21184/jkeia.2024.12.18.9.129 , KCI ART003159505
- 요약: 생성형 AI가 대중음악 창작·개인 콘텐츠·버추얼 아이돌 제작에 쓰이는 사례를 분석해, 누구나 창작하고 플랫폼으로 직접 유통하는 구조로 산업이 바뀐다고 보았다.
- STAGE와의 차이: 생성형 AI의 콘텐츠 제작 대중화라는 산업 배경을 제공하며, STAGE는 그중 감정 음성 생성 도구의 구체적 구현이다.

### I3 ★ 양호·이인호 (2026)
양호, & 이인호. (2026). 인포테인먼트 환경에서의 생성형 AI 음악 플랫폼 비교 연구: Udio AI와 Suno AI의 출력 특성과 활용 방향을 중심으로. *한국엔터테인먼트산업학회논문지, 20*(3), 537–550. https://doi.org/10.21184/jkeia.2026.6.20.3.537
- ✅ https://api.crossref.org/works/10.21184/jkeia.2026.6.20.3.537 , KCI ART003356939
- 요약: 파형·스펙트럼·RMS 분석과 음악 분석으로 Suno와 Udio의 출력 특성을 비교했다.
- STAGE와의 차이: 상용 생성형 오디오를 음향 분석으로 비교한 KOEN 인포테인먼트 분과 선례 → STAGE의 음향 지표 기반 평가와 분과 선택의 근거가 된다.

### I4 ★ 이연경 외 (2022) 버추얼 가수
이연경, 김현지, & 박건우. (2022). 메타버스를 이용한 버추얼 가수의 유형 제시 및 현황 연구. *한국엔터테인먼트산업학회논문지, 16*(6), 107–121. https://doi.org/10.21184/jkeia.2022.8.16.6.107
- ✅ https://api.crossref.org/works/10.21184/jkeia.2022.8.16.6.107 , KCI ART002873687
- 요약: 버추얼 가수를 캐릭터형·휴먼형·아바타형으로 나누고, 외모와 음성(사람 목소리/음성합성)에 따른 분류 체계를 제시했다.
- STAGE와의 차이: '음성합성 목소리형' 가상 인물에 실시간 감정 표현을 더하는 기술로 STAGE의 활용처를 설명할 수 있다. 인용: (이연경 외, 2022)

### I5 ★ 이서영 (2021) AI 음성 인터랙션
Lee, S.-Y. (2021). Voice interactions with A. I. agent: Analysis of domestic and overseas IT companies. *한국엔터테인먼트산업학회논문지 [Journal of the Korea Entertainment Industry Association], 15*(4), 15–29. https://doi.org/10.21184/jkeia.2021.6.15.4.15
- ✅ KCI ART002735939 (제목이 영문만 있음)
- 요약: 국내외 IT 기업의 AI 음성 비서를 분석하고, 음성 인식·자연어 처리와 함께 감성적 사용자 만족을 높이는 전략을 제언했다.
- STAGE와의 차이: AI 음성 인터랙션에서 감성 차원의 중요성을 지적한 선행연구로, STAGE는 표정에 반응하는 감정 음성으로 이를 구현한다.

### I6 ☆ 허동혁·박재록 (2021) AI 마스터링
허동혁, & 박재록. (2021). 인공지능과 인간 전문가의 오디오 마스터링 비교 연구. *한국엔터테인먼트산업학회논문지, 15*(3), 41–50. https://doi.org/10.21184/jkeia.2021.4.15.3.41
- ✅ KCI ART002714657
- 요약: AI 마스터링과 전문가 결과물을 음향 분석·청취 실험·인터뷰로 비교해, 음향 차이는 작고 소통 방식이 핵심 차이라는 결과를 얻었다.
- STAGE와의 차이: AI 오디오를 음향 분석으로 검증하는 방법론의 KOEN 선례.

### I7 ☆ 김현수·심예림 (2026) 공연예술의 생성형 AI
김현수, & 심예림. (2026). 공연예술 분야 생성형 AI 활용 양상과 쟁점에 관한 탐색적 연구. *한국엔터테인먼트산업학회논문지, 20*(5), 131–148. https://doi.org/10.21184/jkeia.2026.8.20.5.131
- ✅ KCI ART003376007 (Crossref의 발행연도 2023 표기는 오류, 20(5)=2026년 8월)
- 요약: 공연예술 가치사슬 전반의 생성형 AI 활용을 사례와 전문가 인터뷰로 정리했으며, 현장에서는 효율을 높이고 창작을 돕는 보조 도구로 인식됨을 보였다.
- STAGE와의 차이: 생성형 AI를 창작 보조 도구로 보는 현장 인식이 STAGE의 위상(창작자의 표정을 감정 음성으로 확장하는 보조 도구)을 뒷받침한다.

### I8 ☆ 조예신·최재오 (2019) 오디오북 유사언어
조예신, & 최재오. (2019). 오디오북 텍스트 형상화를 위한 공감각적 음성 훈련 연구: 유사언어를 활용하여. *한국엔터테인먼트산업학회논문지, 13*(8), 167–180. https://doi.org/10.21184/jkeia.2019.12.13.8.167
- ✅ KCI ART002549579
- 요약: 음색·톤·포즈·속도·억양·강세 같은 유사언어를 활용한 오디오북 음성 훈련을 적용하고 성과를 질적으로 분석했다.
- STAGE와의 차이: 오디오북 낭독의 운율 요소가 STAGE가 제어하는 rate·pitch·volume에 해당한다. 오디오북·오디오드라마를 활용 시나리오로 제시할 때 인용한다.

### I9 ☆ 이아름·조준희 (2025) 보컬 정서 표현 훈련
이아름, & 조준희. (2025). 감정 구성 이론 기반의 보컬 정서 표현 훈련 연구. *한국엔터테인먼트산업학회논문지, 19*(6), 189–204. https://doi.org/10.21184/jkeia.2025.10.19.6.189
- ✅ KCI ART003261805
- 요약: 보컬 기술과 정서 표현이 따로 훈련되는 문제를 지적하고, 감정 구성 이론에 기반한 보컬 정서 표현 훈련을 제안했다.
- STAGE와의 차이: 감정 범주를 음성 표현과 연결하는 관점이 STAGE의 감정 태그 → TTS 표현 매핑과 같은 맥락이다.

---

## 집계

| 구분 | 편수 |
|---|---|
| 확인 완료 전체 | **58편** |
| └ 국외 문헌·문서 | 38편 (학술지·학술대회 논문 30, arXiv 프리프린트 4, 소프트웨어·웹 문서 4) |
| └ 국내 논문 | 18편 (그중 한국엔터테인먼트산업학회논문지 9편) |
| └ AI Hub 데이터셋 | 2건 |
| ★ 핵심 (원고 권장) | 40편 |
| ☆ 보조 | 18편 |

항목 수: A 7, B 3, C 11, D 7, E 5, F 5, G 3, H 8, I 9 = **58**. C6·C7·C11·H1·H2·I1은 졸업논문에서 이어받은 문헌(서지 오류는 바로잡음)이다. 원고 참고문헌은 ★ 위주로 25~35편을 고르면 학회 최소 기준(15편)을 넉넉히 넘는다.

## 빼거나 보류한 것

- EMA·다수결·hold-time을 FER에 직접 적용해 비교한 논문: 찾지 못해 넣지 않음(B3로 대체).
- Lee, Rabiee & Lee (2017)의 NIPS 워크숍 게재처: 확인 못 해 arXiv로만 인용.
- 확인 중 발견했지만 세부 검증을 하지 않아 넣지 않은 KOEN 논문: 김종걸(2026) 20(2) 165–174 (생성형 AI 음악 가치사슬), 문수지(2025) 19(7) 429–442 (버추얼 아이돌과 준사회적 상호작용), 박지은(2026) 20(3) 177–192 (가상 아이돌). 필요하면 KCI에서 확인 후 추가.
- 2015년 이전 국내 감정 TTS 논문(민소연·나덕수 2013, 이상민·이호준 2014): 오래되어 제외.
- Ge·조동민(2025) 게임 음성 감정 인식(멀티미디어학회논문지 28(10), 1614–1626, https://doi.org/10.9717/kmms.2025.28.10.1614): DBpia로만 확인해 목록에서는 뺌.
