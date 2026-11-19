"""/set_emotion_probs + _pick_dominant_mass (E5 확률 질량 문장 태깅)과 폴백 동작."""
import pytest

LABELS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]


def onehotish(label, p=0.7):
    rest = (1.0 - p) / 6
    return [p if lab == label else rest for lab in LABELS]


def post_summary(c, t0, t1, probs, n=6, **extra):
    return c.post("/set_emotion_probs", json={"t0": t0, "t1": t1, "n": n, "probs": probs,
                                              "labels": LABELS, **extra})


@pytest.fixture()
def client(fresh):
    return fresh.app.test_client()


def test_route_validation(client, fresh):
    assert post_summary(client, 0, 200, onehotish("Happy")).get_json() == {"status": "ok"}
    assert len(fresh.PROB_HISTORY) == 1
    # labels 생략 → 기본 순서
    assert client.post("/set_emotion_probs", json={"t0": 200, "t1": 400, "n": 3,
                                                   "probs": onehotish("Sad")}).status_code == 200
    bad = [{}, {"t0": 0, "t1": 1, "n": 1, "probs": [0.5, 0.5]},
           {"t0": 0, "t1": 1, "n": 0, "probs": onehotish("Sad")},
           {"t0": "x", "t1": 1, "n": 1, "probs": onehotish("Sad")},
           {"t0": 0, "t1": 1, "n": 1, "probs": onehotish("Sad"), "labels": ["X"] * 7}]
    for b in bad:
        assert client.post("/set_emotion_probs", json=b).status_code == 400
    assert len(fresh.PROB_HISTORY) == 2


def test_mass_basic_and_disgust_to_anger(client, fresh):
    for k in range(10):
        post_summary(client, 1000 + 200 * k, 1200 + 200 * k, onehotish("Disgust"))
    assert fresh._pick_dominant_mass(1000, 3000) == "<분노>"
    assert fresh._pick_dominant_mass(5000, 6000) == ""      # 겹치는 요약 없음


def test_neutral_suppression(client, fresh):
    # 중립 0.45 vs 슬픔 0.35: w=1이면 빈 태그, w=0.25면 슬픔
    p = [0.04, 0.04, 0.04, 0.04, 0.35, 0.04, 0.45]
    for k in range(10):
        post_summary(client, 1000 + 200 * k, 1200 + 200 * k, p)
    assert fresh._pick_dominant_mass(1000, 3000, w_neutral=1.0) == ""
    assert fresh._pick_dominant_mass(1000, 3000, w_neutral=0.25) == "<슬픔>"
    assert fresh._pick_dominant_mass(1000, 3000) == "<슬픔>"   # 기본 0.25 (cnn)
    assert fresh._pick_dominant_mass(1000, 3000, w_neutral=0.25, r_min=0.6) == ""


def test_partial_overlap_weighting(client, fresh):
    # 행복 요약 [0,1000) 프레임 30개, 놀람 요약 [1000,2000) 30개. 창 [900,2000]: 행복 10%, 놀람 100%
    post_summary(client, 0, 1000, onehotish("Happy", 0.9), n=30)
    post_summary(client, 1000, 2000, onehotish("Surprise", 0.9), n=30)
    assert fresh._pick_dominant_mass(900, 2000) == "<놀람>"
    assert fresh._pick_dominant_mass(0, 1100) == "<행복>"


def test_w_neutral_by_classifier_and_env(client, fresh, monkeypatch):
    assert fresh._w_neutral() == 0.25
    post_summary(client, 0, 200, onehotish("Happy"), classifier="hsemotion")
    assert fresh._w_neutral() == 0.9
    post_summary(client, 200, 400, onehotish("Happy"), classifier="cnn")
    assert fresh._w_neutral() == 0.25
    monkeypatch.setenv("FACEVOX_W_NEUTRAL", "0.5")
    assert fresh._w_neutral() == 0.5
    j = client.get("/debug/backend").get_json()
    assert j["tagging"] == "mass" and j["classifier"] == "cnn" and j["w_neutral"] == 0.5


def test_resolve_one_uses_mass_when_summaries_exist(client, fresh):
    # 라벨 이벤트는 행복(지속 시간 방식이면 <행복>), 확률 요약은 슬픔 → 질량 방식이 쓰인다
    client.post("/set_emotion", json={"t": 0, "label": "Happy", "confidence": 0.9})
    r = client.post("/emotion/resolve_one", json={"since_ms": 500, "until_ms": 2500})
    assert r.get_json() == {"tag": "<행복>"}          # 요약 없음 → 폴백
    for k in range(10):
        post_summary(client, 500 + 200 * k, 700 + 200 * k, onehotish("Sad"))
    r = client.post("/emotion/resolve_one", json={"since_ms": 500, "until_ms": 2500})
    assert r.get_json() == {"tag": "<슬픔>"}
    assert client.post("/emotion/resolve_one", json={"since_ms": 10, "until_ms": 5}).get_json() == {"tag": ""}
