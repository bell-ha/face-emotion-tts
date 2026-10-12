"""_pick_dominant_ko: 합성 이벤트 스트림으로 문장 구간 지배 감정 판정 검증."""


def feed(m, events):
    for t, label, conf in events:
        m.EMO_HISTORY.append({"t": t, "emo": m._event_to_ko(label, ""), "label": label,
                              "tag": "", "confidence": conf})


def test_empty_history(fresh):
    assert fresh._pick_dominant_ko(0, 1000) == ""


def test_single_state_before_window(fresh):
    # 구간 시작 전에 들어온 Happy가 구간 전체를 덮는다
    feed(fresh, [(0, "Happy", 0.9)])
    assert fresh._pick_dominant_ko(1000, 3000) == "<행복>"


def test_longest_wins(fresh):
    # 1000~1600 Sad(600ms), 1600~3000 Angry(1400ms)
    feed(fresh, [(0, "Sad", 0.8), (1600, "Angry", 0.8)])
    assert fresh._pick_dominant_ko(1000, 3000) == "<분노>"


def test_disgust_maps_to_anger(fresh):
    feed(fresh, [(0, "Disgust", 0.9)])
    assert fresh._pick_dominant_ko(100, 900) == "<분노>"


def test_neutral_dominant_returns_empty(fresh):
    feed(fresh, [(0, "Neutral", 0.9), (2500, "Happy", 0.9)])
    assert fresh._pick_dominant_ko(1000, 3000) == ""   # 중립 1500 vs 행복 500 → 빈 태그


def test_low_confidence_ignored(fresh):
    # conf < 0.5 인 Happy는 집계되지 않음 → 남는 감정 없음
    feed(fresh, [(0, "Happy", 0.3)])
    assert fresh._pick_dominant_ko(1000, 2000) == ""


def test_min_ratio(fresh):
    # 구간 2000ms 중 Sad 700ms(35%) < 0.4 → 빈 태그. 나머지는 저신뢰.
    feed(fresh, [(0, "Happy", 0.2), (1300, "Sad", 0.9), (2000, "Fear", 0.2)])
    assert fresh._pick_dominant_ko(1000, 3000) == ""
    # 같은 스트림, min_ratio=0.3이면 Sad 채택
    assert fresh._pick_dominant_ko(1000, 3000, min_ratio=0.3) == "<슬픔>"


def test_events_after_window_ignored(fresh):
    feed(fresh, [(0, "Fear", 0.9), (5000, "Happy", 0.9)])
    assert fresh._pick_dominant_ko(1000, 3000) == "<두려움>"


def test_many_flickers(fresh):
    # 100ms 간격으로 Happy/Surprise 깜빡임, Happy가 70%
    ev = []
    t = 1000
    for i in range(20):
        ev.append((t, "Happy", 0.8)); t += 70
        ev.append((t, "Surprise", 0.8)); t += 30
    feed(fresh, ev)
    assert fresh._pick_dominant_ko(1000, 3000) == "<행복>"
