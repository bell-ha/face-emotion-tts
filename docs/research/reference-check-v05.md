# 참고문헌 실존·DOI 검증 (원고 v05)

점검일: 2026-10-12 · 대상: `paper/2_journal-koen/STAGE_KOEN_v05.md` References 53편

방법: DOI 44편은 Crossref API로 제목·발행연도·쪽을 받아 원고와 대조(Crossref에 없는 arXiv·한국음향학회지 DOI는 doi.org 응답 확인). DOI 없는 9편은 URL 응답과 제목 확인.

## 결과 요약

- DOI 44편: 모두 존재하고 제목·쪽이 원고와 일치.
- 김현수·심예림(2026): Crossref 발행연도가 2023으로 나오지만, 한국엔터테인먼트산업학회논문지 20권 5호(2026-08)가 Crossref에 2023-08-31로 잘못 등록된 문제다(DOI 문자열 `2026.8.20.5.131`). 원고의 2026이 맞다.
- Miller(1968): Crossref 쪽은 시작 쪽(267)만 있다. 원고 267–277은 AFIPS 발표집 쪽 범위로 통용되는 값.
- DOI 없는 9편: 8편 확인. **OpenAI(2025, March 20)** 공지 페이지만 자동 접속이 차단(403)돼 확인하지 못함 → 브라우저로 열어 제목·날짜 확인 필요.

## 항목별

| 참고문헌 | DOI | Crossref 연도 | Crossref 제목 · 쪽 / 확인 |
|---|---|---|---|
| 권새봄, & 김주섭. (2019). 감정 전이와 실시간 얼굴 표정 리타게… | 10.9728/dcs.2019.20.6.1117 | 2019 | Enhancing Music Listening Experience Based on Emotional Contagion and  · p.1117-1124 |
| 김현수, & 심예림. (2026). 공연예술 분야 생성형 AI 활용 양상… | 10.21184/jkeia.2026.8.20.5.131 | 2023 | An Exploratory Study on the Applications and Issues of Generative AI i · p.131-148 |
| 박중희, & 한광희. (2023). 감정별 음성분석을 통한 인공지능 TT… | 10.5392/JKCA.2023.23.12.001 | 2023 | Design of SSML Syntax Factors that Distinguish Emotional Utterances of · p.1-22 |
| 양호, & 이인호. (2026). 인포테인먼트 환경에서의 생성형 AI 음… | 10.21184/jkeia.2026.6.20.3.537 | 2026 | A Comparative Study of Generative AI Music Platforms in Infotainment E · p.537-550 |
| 유세희, 조희, 이주현, & 홍기형. (2021). 다음색 감정 음성합성… | 10.7776/ASK.2021.40.5.523 | ? | Crossref 없음, doi.org 응답 200 |
| 이연경, 김현지, & 박건우. (2022). 메타버스를 이용한 버추얼 가… | 10.21184/jkeia.2022.8.16.6.107 | 2022 | Virtual Singer using Metaverse Type and Current Research · p.107-121 |
| 이은혜, & 김유정. (2020). 뮤지컬 노래의 극과 음악 분석을 통한… | 10.21184/jkeia.2020.12.14.8.91 | 2020 | Application and Practice of Estill Vocal Training (EVT) Through Theatr · p.91-102 |
| 이종하. (2025). *실시간 표정 인식을 이용한 감정 기반 TTS 보… | DOI 없음 |  | 학사학위논문(원본 PDF 보유) |
| 조예신, & 최재오. (2019). 오디오북 텍스트 형상화를 위한 공감각… | 10.21184/jkeia.2019.12.13.8.167 | 2019 | Audiobook Text Shaping for Synesthesia Voice Training - Focusing on Pa · p.167-180 |
| 최영신. (2024). 생성형 인공지능과 대중음악 산업구조 변화. *한국… | 10.21184/jkeia.2024.12.18.9.129 | 2024 | Generative Artificial Intelligence and Structural Changes in the Popul · p.129-138 |
| 한국지능정보사회진흥원. (2020). *한국인 감정인식을 위한 복합 영상… | DOI 없음 |  | aihub.or.kr dataSetSn=82 (200) |
| Banse, R., & Scherer, K. R. (1996). Acou… | 10.1037/0022-3514.70.3.614 | 1996 | Acoustic profiles in vocal emotion expression. · p.614-636 |
| Bazarevsky, V., Kartynnik, Y., Vakunov, … | 10.48550/arXiv.1907.05047 | ? | Crossref 없음, doi.org 응답 200 |
| Boersma, P., & Weenink, D. (2021). *Praa… | DOI 없음 |  | praat.org (200) |
| Card, S. K., Robertson, G. G., & Mackinl… | 10.1145/108844.108874 | 1991 | The information visualizer, an information workspace · p.181-186 |
| Chu, Y., Shim, Y., & Park, U. (2025). Fa… | 10.1007/978-3-031-91581-9_9 | 2025 | Facial Expression-Enhanced TTS: Combining Face Representation and Emot · p.117-129 |
| Desplanques, B., Thienpondt, J., & Demuy… | 10.21437/Interspeech.2020-2650 | 2020 | ECAPA-TDNN: Emphasized Channel Attention, Propagation and Aggregation  · p.3830-3834 |
| Diatlova, D., & Shutov, V. (2023). EmoSp… | 10.21437/SSW.2023-17 | 2023 | EmoSpeech: guiding FastSpeech2 towards Emotional Text to Speech · p.106-112 |
| Ekman, P. (1992). An argument for basic … | 10.1080/02699939208411068 | 1992 | An argument for basic emotions · p.169-200 |
| Goodfellow, I. J., Erhan, D., Carrier, P… | 10.1007/978-3-642-42051-1_16 | 2013 | Challenges in Representation Learning: A Report on Three Machine Learn · p.117-124 |
| Goto, S., Onishi, K., Saito, Y., Tachiba… | 10.21437/Interspeech.2020-2136 | 2020 | Face2Speech: Towards Multi-Speaker Text-to-Speech Synthesis Using an E · p.1321-1325 |
| Guo, Z., Leng, Y., Wu, Y., Zhao, S., & T… | 10.1109/ICASSP49357.2023.10096285 | 2023 | Prompttts: Controllable Text-To-Speech With Text Descriptions · p.1-5 |
| Jadoul, Y., Thompson, B., & de Boer, B. … | 10.1016/j.wocn.2018.07.001 | 2018 | Introducing Parselmouth: A Python interface to Praat · p.1-15 |
| Juslin, P. N., & Laukka, P. (2003). Comm… | 10.1037/0033-2909.129.5.770 | 2003 | Communication of emotions in vocal expression and music performance: D · p.770-814 |
| Kahou, S. E., Pal, C., Bouthillier, X., … | 10.1145/2522848.2531745 | 2013 | Combining modality specific deep neural networks for emotion recogniti · p.543-550 |
| Kumar, A., Tan, K., Ni, Z., Manocha, P.,… | 10.1109/ICASSP49357.2023.10096680 | 2023 | Torchaudio-Squim: Reference-Less Speech Quality and Intelligibility Me · p.1-5 |
| Lee, J., Chung, J. S., & Chung, S.-W. (2… | 10.1109/ICASSP49357.2023.10094745 | 2023 | Imaginary Voice: Face-Styled Diffusion Model for Text-to-Speech · p.1-5 |
| Lee, S.-Y. (2021). Voice interactions wi… | 10.21184/jkeia.2021.6.15.4.15 | 2021 | Voice Interactions with A. I. Agent : Analysis of Domestic and Oversea · p.15-29 |
| Lee, Y., & Kim, T. (2019). Robust and fi… | 10.1109/ICASSP.2019.8683501 | 2019 | Robust and Fine-grained Prosody Control of End-to-end Speech Synthesis · p.5911-5915 |
| Lei, Y., Yang, S., Wang, X., & Xie, L. (… | 10.1109/TASLP.2022.3145293 | 2022 | MsEmoTTS: Multi-Scale Emotion Transfer, Prediction, and Control for Em · p.853-864 |
| Li, S., & Deng, W. (2022). Deep facial e… | 10.1109/TAFFC.2020.2981446 | 2022 | Deep Facial Expression Recognition: A Survey · p.1195-1215 |
| Livingstone, S. R., & Russo, F. A. (2018… | 10.1371/journal.pone.0196391 | 2018 | The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDES · p.e0196391 |
| Lugaresi, C., Tang, J., Nash, H., McClan… | 10.48550/arXiv.1906.08172 | ? | Crossref 없음, doi.org 응답 200 |
| Ma, Z., Zheng, Z., Ye, J., Li, J., Gao, … | 10.18653/v1/2024.findings-acl.931 | 2024 | emotion2vec: Self-Supervised Pre-Training for Speech Emotion Represent · p.15747-15760 |
| Miller, R. B. (1968). Response time in m… | 10.1145/1476589.1476628 | 1968 | Response time in man-computer conversational transactions · p.267 |
| OpenAI. (n.d.). *Text to speech*. OpenAI… | DOI 없음 |  | developers.openai.com (200) |
| OpenAI. (2025, March 20). *Introducing n… | DOI 없음 |  | openai.com 403 — 자동 접속 차단. 브라우저로 확인 필요 |
| rany2. (n.d.). *edge-tts* (Version 7.2.8… | DOI 없음 |  | github.com/rany2/edge-tts (200) |
| Ren, Y., Hu, C., Tan, X., Qin, T., Zhao,… | DOI 없음 |  | openreview.net (200) |
| Russell, J. A. (1980). A circumplex mode… | 10.1037/h0077714 | 1980 | A circumplex model of affect. · p.1161-1178 |
| Ryu, N., & Yun, S. (2025). Native Korean… | 10.17154/kjal.2025.12.41.4.101 | 2025 | Native Korean Listeners’ Perception of Emotional Expression in Korean  · p.101-129 |
| Savchenko, A. V. (2022). Video-based fra… | 10.1109/CVPRW56347.2022.00263 | 2022 | Video-based Frame-level Facial Analysis of Affective Behavior on Mobil · p.2358-2365 |
| Scherer, K. R. (2003). Vocal communicati… | 10.1016/S0167-6393(02)00084-5 | 2003 | Vocal communication of emotion: A review of research paradigms · p.227-256 |
| Sivaprasad, S., Kosgi, S., & Gandhi, V. … | 10.21437/Interspeech.2021-307 | 2021 | Emotional Prosody Control for Speech Generation · p.4653-4657 |
| Tan, X., Qin, T., Soong, F., & Liu, T.-Y… | 10.48550/arXiv.2106.15561 | ? | Crossref 없음, doi.org 응답 200 |
| Wang, Y., Skerry-Ryan, R. J., Stanton, D… | 10.21437/Interspeech.2017-1452 | 2017 | Tacotron: Towards End-to-End Speech Synthesis · p.4006-4010 |
| Wang, Y., Stanton, D., Zhang, Y., Skerry… | DOI 없음 |  | PMLR v80 wang18h (200, 제목 일치) |
| Xue, F., Tan, Z., Zhu, Y., Ma, Z., & Guo… | 10.1109/CVPRW56347.2022.00269 | 2022 | Coarse-to-Fine Cascaded Networks with Smooth Predicting for Video Faci · p.2411-2417 |
| Yang, D., Liu, S., Huang, R., Weng, C., … | 10.1109/TASLP.2024.3402088 | 2024 | InstructTTS: Modelling Expressive TTS in Discrete Latent Space With Na · p.2913-2925 |
| Yang, S.-w., Chi, P.-H., Chuang, Y.-S., … | 10.21437/Interspeech.2021-1775 | 2021 | SUPERB: Speech Processing Universal PERformance Benchmark · p.1194-1198 |
| Ye, J., Cao, B., & Shan, H. (2025). Emot… | DOI 없음 |  | PMLR v267 ye25b (200, 제목 일치) |
| Zhang, T.-H., Zhang, J., Wang, J., Qian,… | 10.1609/aaai.v39i24.34786 | 2025 | FaceSpeak: Expressive and High-Quality Speech Synthesis from Human Por · p.25922-25930 |
| Zhou, K., Sisman, B., Liu, R., & Li, H. … | 10.1016/j.specom.2021.11.006 | 2022 | Emotional voice conversion: Theory, databases and ESD · p.1-18 |
