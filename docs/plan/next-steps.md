# 다듬을 것 정리 (2026-10-12, 이름 STAGE·폴더 face-emotion-tts로 변경 후)

## 현재 상태

- 원고: `paper/2_journal-koen/STAGE_KOEN_v05.md` (2026-10-12, 지도교수 검토용: 기여 4가지 명시, 확인 안 된 정보 제거, 규정 체크리스트 `docs/submission/koen-checklist.md` 통과, 15쪽). 제출용 `STAGE_KOEN_v05.docx`, 확인용 `.pdf`. 사람 청취 평가는 이번 투고에서 하지 않기로 함
- 투고 전략: `docs/plan/project-plan.md` "투고 전략". 신청서: `docs/submission/submission-intent.md`.
- 실험 근거: `experiments/E1_face-recognition`, `E2_emotion-stabilization`, `E3_emotional-speech`(+`E3b_intensity`), `E4_system-latency`, `E5_improvements`의 README.
- 가상 심사 의견: `paper/2_journal-koen/review_v01.md` (상세 수정안·후보 참고문헌).

## 1. 사용자가 직접 할 일

- [ ] 10/13 투고예정신청서 메일(office@koen.or.kr). 제목은 "감정 연출 경험" 버전. 인쇄본 컬러 여부를 같이 묻기
- [ ] 공동저자(구자환 교수님) 결정과 동의
- [ ] 원고 1쪽·국문부·저자소개용 정보: 소속, 직위, 교신저자, 이메일, 증명사진, 약력
- [ ] 회원가입, 심사료 8만 원, KCI 유사도 검사, 10/23 홈페이지 업로드

## 2. 원고 내용 (Claude가 다음 세션에서)

- [x] `⟦인용 확인⟧` 3곳 Crossref 확인·추가 완료(Miller 1968, Card et al. 1991, Desplanques 2020, Kumar 2023, Yang 2021, Savchenko 2022). Nielsen 1993·SpeechBrain은 빼고 씀
  - 응답 시간 기준: Miller(1968), Card et al.(1991), Nielsen(1993)
  - 평가 도구: ECAPA-TDNN(Desplanques et al., 2020), SpeechBrain(Ravanelli et al., 2021), TorchAudio-Squim(Kumar et al., 2023), SUPERB(Yang et al., 2021)
  - HSEmotion(Savchenko, 2022)
- [x] Praat 6.1.38(2021)로 채움
- [x] 국문 요약 443자로 정리, 영문 Abstract(224단어)와 수치 일치 확인
- [x] 초록·결론·국문 요약 수치가 결과와 일치함을 대조
- [ ] E5 개선안을 그림 하나로 보일지 결정(`experiments/E5_improvements/figures/fig_e5_2x2`), 넣으면 분량 확인
- [ ] 분량 13–14쪽 확인(현재 PDF 15쪽). 넘치면 2.2절 TTS 계보 축약, 참고문헌 일부 정리(심사 의견 6.3절)

## 3. 서식 (양식 변환기)

- [ ] 글씨가 길쭉해 보이는 문제: 장평 90%·자간이 영문에도 들어간 것, 휴먼명조 WordReady 폭 확인 → 언어별 값 적용
- [ ] 2단 폭 그림·표는 페이지 맨 위/아래에만, 나머지는 1단 폭
- [ ] 그림 크기 상한(2단 폭 약 80 mm, 1단 약 60 mm), 웹 UI 그림 1/3쪽 이하
- [ ] 1쪽(영문 블록)이 1쪽 안에 들어가는지
- [ ] 양쪽 정렬로 줄 간격이 지나치게 벌어진 줄 정리
- HWPX는 만들지 않는다(사용자 결정). docx까지만 만들고, 한글 변환·제출은 사용자가 맥북에서 직접 한다.

## 4. 그림

- [ ] 원고 그림(구조도, 웹 UI, 강도 절충, E5 선택 시 1개)을 같은 팔레트 컬러로 통일, 흑백 인쇄에서도 구분
- [x] 강도 절충 그림 컬러화, 라벨 겹침 수정
- [x] 캡션 형식 통일 `(Fig. N …)` / `(Tab. N …)`

## 5. 선택 (시간이 되면 효과 큼)

- [ ] 창작자 5–8명 소규모 사용성 평가 또는 청취자 10명 감정 판별 → "사람 평가 없음" 약점 완화(심사 의견 2.5절)
- [ ] E5 개선 태깅을 실제 앱(app/face/, app/tts/)에 반영(`experiments/E5_improvements/README.md` "구현 반영 방법")
