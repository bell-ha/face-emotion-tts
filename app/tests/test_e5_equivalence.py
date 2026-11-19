"""앱 코드(emotion_core.Stabilizer/ProbSummary, app._pick_dominant_mass)가 E5 참조 구현
(experiments/E5_improvements/scripts/e5_methods.py)과 같은 입력에서 같은 출력을 내는지 확인.

1) 합성 확률 스트림: 안정화 규칙(thesis/gate/hyst/gate_hyst)의 프레임별 라벨과 전송 이벤트, 0.2 s 요약,
   문장 태그(w_neutral, r_min 여러 값)를 비교.
2) RAVDESS 프레임 캐시(data/ravdess/probs_e5)가 있으면: 시험 배우 일부의 스트림을 E5와 같은 순서로 재생해
   앱 함수로 만든 문장 태그·클립당 전송 이벤트 수가 E5 결과(results/test_per_clip_*.csv)와 같은지 확인.
"""
import glob
import os
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
E5 = REPO / "experiments" / "E5_improvements"
sys.path.insert(0, str(E5 / "scripts"))

e5m = pytest.importorskip("e5_methods")
import emotion_core as core  # noqa: E402

LABELS = core.EMOTION_LABELS


# ───────────────────────────── 합성 데이터 ─────────────────────────────
def synth_stream(seed, n=1500):
    """레짐이 바뀌는 확률 스트림(float32), 30 fps ± 지터, 가끔 얼굴 없음(None)."""
    rng = np.random.default_rng(seed)
    t, out = 1_700_000_000.0 + rng.uniform(0, 1), []
    dom = int(rng.integers(7))
    for i in range(n):
        if rng.random() < 0.02:
            dom = int(rng.integers(7))
        t += rng.uniform(0.025, 0.045)
        if rng.random() < 0.03:
            out.append((t, None))
            continue
        a = np.full(7, 0.6); a[dom] = rng.uniform(1.0, 6.0)
        if rng.random() < 0.15:  # 깜빡임
            a[int(rng.integers(7))] += 4.0
        out.append((t, rng.dirichlet(a).astype(np.float32)))
    return out


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("rule,delta", [("thesis", 0.0), ("gate", 0.0), ("hyst", 0.5), ("hyst", 0.1),
                                        ("gate_hyst", 0.5), ("gate_hyst", 0.2)])
def test_stabilizer_matches_reference(seed, rule, delta):
    stream = synth_stream(seed)
    app = core.Stabilizer(rule=rule, delta=delta)
    ref = e5m.Stabilizer() if rule == "thesis" else e5m.ImprovedStabilizer(rule=rule, delta=delta)
    app_events, app_final, ref_final = [], [], []
    for t, p in stream:
        r = app.step(t, p)
        if r["send"] is not None:
            app_events.append(r["send"])
        app_final.append(r["final_idx"])
        ref_final.append(ref.step(t, p)["final"])
    assert app_final == ref_final
    assert app_events == ref.events
    assert len(app_events) > 10


@pytest.mark.parametrize("seed", [0, 3])
def test_summary_and_mass_tag_match_reference(seed, fresh):
    stream = synth_stream(seed, 2500)
    cli, ref_cli, ref_srv = core.ProbSummary(0.2), e5m.SummaryClient(0.2), e5m.ProbSummaryServer(maxlen=3000)
    c = fresh.app.test_client()
    payloads = []
    for t, p in stream:
        ref_cli.step(t, p)
        out = cli.step(t, p)
        if out is not None:
            payloads.append(out)
            assert c.post("/set_emotion_probs", json=out).status_code == 200
    # 요약 자체가 같다
    assert len(payloads) == len(ref_cli.out) > 100
    for pl, (a, b, p, n) in zip(payloads, ref_cli.out):
        assert (pl["t0"], pl["t1"], pl["n"]) == (a, b, n)
        assert np.array_equal(np.array(pl["probs"], np.float32), p)
        ref_srv.add((a, b, p, n))
    # 임의 문장 구간에서 태그가 같다
    rng = np.random.default_rng(seed)
    lo, hi = payloads[0]["t0"], payloads[-1]["t1"]
    n_cmp = 0
    for _ in range(300):
        t0 = int(rng.integers(lo - 500, hi))
        t1 = t0 + int(rng.integers(300, 6000))
        m_app = fresh._mass_ko(t0, t1)
        assert np.allclose([m_app[k] for k in e5m.KO6], ref_srv.mass6(t0, t1), rtol=1e-5, atol=1e-6)
        for w in (1.0, 0.25, 0.9, 0.5):
            for rmin in (0.0, 0.4):
                assert fresh._pick_dominant_mass(t0, t1, w_neutral=w, r_min=rmin) == ref_srv.pick(t0, t1, w, rmin)[0]
                n_cmp += 1
    assert n_cmp == 300 * 8


def test_hse_postprocess_matches_e5_analyze():
    rng = np.random.default_rng(0)
    L = (rng.normal(size=(50, 10)) * 3).astype(np.float32)
    # e5_analyze.load_clips: d[clf][:, :8][:, HSE_TO_OURS] → softmax
    ref = L[:, :8][:, [0, 2, 3, 4, 6, 7, 5]]
    ref = np.exp(ref - ref.max(1, keepdims=True)); ref = (ref / ref.sum(1, keepdims=True)).astype(np.float32)
    assert np.array_equal(core.hse_logits_to_probs(L), ref)


def test_hse_preprocess_matches_e5_extract():
    pytest.importorskip("onnxruntime")
    e5x = pytest.importorskip("e5_extract")
    img = np.random.default_rng(1).integers(0, 256, size=(97, 83, 3), dtype=np.uint8)
    assert np.array_equal(core.hse_preprocess(img), e5x.hse_preprocess(img))


HSE_ONNX = REPO / "data" / "models" / "hsemotion" / "enet_b0_8_va_mtl.onnx"


@pytest.mark.skipif(not HSE_ONNX.exists(), reason="HSEmotion ONNX 파일 없음")
def test_hse_classifier_runs():
    pytest.importorskip("onnxruntime")
    clf = core.HSEmotionClassifier(str(HSE_ONNX))
    img = np.random.default_rng(2).integers(0, 256, size=(120, 100, 3), dtype=np.uint8)
    p = clf(img)
    assert p.shape == (1, 7) and abs(float(p.sum()) - 1.0) < 1e-5


# ───────────────────────────── RAVDESS 캐시 (있으면) ─────────────────────────────
PRB = REPO / "data" / "ravdess" / "probs_e5"
W_NEUTRAL = {"cnn": 0.25, "hsemotion": 0.9}
CSV = {"cnn": "test_per_clip_cnn.csv", "hsemotion": "test_per_clip_mtl.csv"}


def run_actor_stream(appmod, client, actor, clf):
    """e5_analyze.streams/run_sequence와 같은 순서·시간축으로 배우 한 명의 52클립을 앱 함수로 재생.
    반환: {clip: (태그 한국어 또는 '중립', 클립 안 전송 이벤트 수)}"""
    files = sorted(f for f in glob.glob(str(PRB / f"*-{actor:02d}.npz"))
                   if int(os.path.basename(f).split("-")[2]) != 2)
    order = np.random.default_rng(42 + actor).permutation(len(files))
    preset = core.CLASSIFIER_PRESETS[clf]
    stab = core.Stabilizer(rule=preset["rule"], delta=preset["delta"])
    summ = core.ProbSummary(0.2, classifier=clf)
    out, t_off = {}, 0.0
    for i in order:
        d = np.load(files[i])
        t, face = d["t"], d["face"]
        if clf == "cnn":
            P = d["cnn"]
        else:
            P = np.full((len(t), 7), np.nan, np.float32)
            P[face] = core.hse_logits_to_probs(d["mtl"][face])
        n_ev = 0
        for k in range(len(t)):
            tt = t_off + t[k]
            p = P[k] if face[k] else None
            n_ev += stab.step(tt, p)["send"] is not None
            pl = summ.step(tt, p)
            if pl is not None:
                client.post("/set_emotion_probs", json=pl)
        t0, t1 = int((t_off + t[0]) * 1000), int((t_off + t[-1]) * 1000)
        tag = appmod._pick_dominant_mass(t0, t1)   # w_neutral: payload의 classifier로 서버가 고름
        out[os.path.basename(files[i])[:-4]] = (tag.strip("<>") or "중립", n_ev)
        t_off += t[-1] + ((t[1] - t[0]) if len(t) > 1 else 1 / 30)
    return out


@pytest.mark.skipif(not PRB.exists() or not (E5 / "results" / CSV["cnn"]).exists(),
                    reason="RAVDESS 캐시(data/ravdess/probs_e5) 또는 E5 결과 없음")
@pytest.mark.parametrize("clf", ["cnn", "hsemotion"])
@pytest.mark.parametrize("actor", [5, 6])
def test_ravdess_cache_matches_e5_results(clf, actor, fresh):
    import csv
    rows = {r["clip"]: r for r in csv.DictReader(open(E5 / "results" / CSV[clf], encoding="utf-8"))
            if int(r["actor"]) == actor}
    got = run_actor_stream(fresh, fresh.app.test_client(), actor, clf)
    assert fresh._w_neutral() == W_NEUTRAL[clf]
    assert set(got) == set(rows) and len(got) == 52
    tag_mismatch = [c for c in got if got[c][0] != rows[c]["tag_T2_mass_tuned"]]
    ev_mismatch = [c for c in got if got[c][1] != int(rows[c]["events_imp"])]
    assert tag_mismatch == [] and ev_mismatch == []
