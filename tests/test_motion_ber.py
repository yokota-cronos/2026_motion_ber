"""装置 (ADP3250 / Motive) 無しで確かめられる部分のテスト。

    uv run python -m pytest tests     # または uv run python tests/test_motion_ber.py
"""
from __future__ import annotations

import csv
import json
import sys
import tempfile
import threading
import time
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import motion  # noqa: E402
import motion_ber_gui as mb  # noqa: E402


def _mk(model, marker, pos, occluded=False):
    return NS(id_num=(model << 16) | marker, pos=pos, param=1 if occluded else 0)


def test_labeled_marker_mapping():
    mocap = NS(labeled_marker_data=NS(labeled_marker_list=[
        _mk(1, 1, (0.1, 0.2, 0.3)),
        _mk(1, 3, (1.0, 1.0, 1.0), occluded=True),     # occluded ビット
        _mk(1, 4, (0.0, 0.0, 0.0)),                    # 原点 = 隠れ
        _mk(1, 7, (0.7, 0.7, 0.7)),
        _mk(1, 9, (9.0, 9.0, 9.0)),                    # 範囲外の marker_id
    ]))
    xyz = motion.collect_labeled(mocap, 7)
    assert np.allclose(xyz[0], (0.1, 0.2, 0.3))
    assert np.allclose(xyz[6], (0.7, 0.7, 0.7))
    assert np.isnan(xyz[[1, 2, 3, 4, 5]]).all()


def test_natnet_callback_and_window():
    s = motion.MotionStream()
    base = time.perf_counter()
    for k in range(5):
        mocap = NS(labeled_marker_data=NS(labeled_marker_list=[_mk(1, i + 1, (k, i, 1.0)) for i in range(7)]),
                   suffix_data=NS(timestamp=k / 100, stamp_camera_mid_exposure=-1), prefix_data=None)
        s._on_frame({"frame_number": 100 + k, "mocap_data": mocap})
        time.sleep(0.01)
    assert s.n_frames == 5 and s.latest().frame == 104 and s.latest().mid_exp is None
    w = s.window(base, time.perf_counter())
    assert [f.frame for f in w] == [100, 101, 102, 103, 104]
    assert list(s.nearest(np.array([w[2].t + 1e-4]))) == [102]
    assert list(s.nearest(np.array([w[-1].t + 10.0]))) == [-1]          # 離れた古いフレームには結び付けない
    assert "model=1 marker=1" in s.diag


def _session(tmp: Path):
    mb.bg.CAPTURE_ROOT = tmp
    cond = dict(level_dbm=0.0, phase_deg=175.53, div=8, payload=600, memo="test")
    return mb.MotionSessionLog(cond)


def test_session_join(tmp_path=None):
    tmp = Path(tmp_path or tempfile.mkdtemp())
    mb.MOTION.close()
    mb.MOTION.buf.clear()
    mb.MOTION.set_group("upper")
    log = _session(tmp)
    t_cap = time.perf_counter()
    # 収録窓 0.1 秒のあいだに 120 fps ぶんのモーションが届いたことにする
    for k in range(12):
        xyz = np.tile([0.0, 1.0, 0.0], (7, 1)) + [k * 1e-3, 0, 0]
        if k == 5:
            xyz[2] = np.nan
        mb.MOTION._ingest(t_cap + k / 120, 1000 + k, k / 120, 5000 + k, xyz)
    rx_fs = 100e6 / 8
    st = np.arange(0, 1_000_000, 4000, dtype=np.int64)            # 250 フレーム
    status = np.zeros(st.size, int)
    status[3] = 2
    res = dict(starts=st, status=status, frame_errors=np.ones(st.size - 1, int))
    mb.CAPTURE.update(t0=t_cap, n=10_000_000, fs=100e6, rx_fs=rx_fs, res=res)
    p = dict(t=0.1, level=0.0, phase=175.53, frames=249, expected=250, bits=249 * 600, errors=249, q=0.9, eye=0.8)
    log.row(p)
    log.close()
    d = log.dir
    assert mb.MOTION.sink is None
    for name in ("ber_log.csv", "ber_motion.csv", "frames.csv", "motion.csv", "motion_conditions.json"):
        assert (d / name).exists(), name
    rows = list(csv.DictReader(open(d / "ber_motion.csv", encoding="utf-8")))
    assert len(rows) == 1 and rows[0]["motion_frames"] == "12"
    assert abs(float(rows[0]["valid_frac"]) - 11 / 12) < 1e-3
    assert rows[0]["opti_frame_first"] == "1000" and rows[0]["opti_frame_last"] == "1011"
    assert abs(float(rows[0]["chest_X"]) - 5.5e-3) < 1e-9
    assert abs(float(rows[0]["chest_dev"]) - 5.5e-3) < 1e-9
    fr = list(csv.DictReader(open(d / "frames.csv", encoding="utf-8")))
    assert len(fr) == 250 and fr[3]["errors"] == "-1" and fr[4]["errors"] == "1"
    # 先頭のフレームは収録開始の SKIP / rx_fs 後、そこに最も近いモーションは 1000 番
    assert fr[0]["OptiFrame"] == "1000"
    assert fr[-1]["OptiFrame"] == "1010"          # 0.0797 s -> k = 10 (0.0833 s)
    m = list(csv.DictReader(open(d / "motion.csv", encoding="utf-8")))
    assert len(m) == 12 and m[5]["n_valid"] == "6" and m[5]["L_upperarm_X"] == ""
    assert json.loads((d / "motion_conditions.json").read_text(encoding="utf-8"))["marker_names"][0] == "chest"


def test_auto_phase_hook(tmp_path=None):
    tmp = Path(tmp_path or tempfile.mkdtemp())
    mb.MOTION.close()
    mb.MOTION.buf.clear()
    mb.PHASE_MODEL = NS(predict_phase=lambda m: 181.234 if m["valid"] > 0.5 else None)
    mb.OPTS["auto_phase"] = True
    mb.bg.COMMANDS.clear()
    try:
        log = _session(tmp)
        t = time.perf_counter()
        mb.MOTION._ingest(t + 0.005, 1, 0.0, None, np.zeros((7, 3)) + 1)
        mb.CAPTURE.update(t0=t, n=1_000_000, fs=100e6, rx_fs=12.5e6,
                          res=dict(starts=np.zeros(0, np.int64), status=np.zeros(0, int), frame_errors=np.zeros(0, int)))
        log.row(dict(t=0, level=0.0, phase=180.0, frames=0, expected=0, bits=0, errors=0, q=None, eye=None))
        assert mb.bg.STATE.live["phase_deg"] == 181.23
        assert mb.bg.COMMANDS[-1] == dict(phase_deg=181.23)
        # 次の収録はこの位相で撮られ、phase_src = auto で残る
        log.row(dict(t=0, level=0.0, phase=181.23, frames=0, expected=0, bits=0, errors=0, q=None, eye=None))
        log.close()
        rows = list(csv.DictReader(open(log.dir / "ber_motion.csv", encoding="utf-8")))
        assert [r["phase_src"] for r in rows] == ["manual", "auto"]
    finally:
        mb.PHASE_MODEL, mb.OPTS["auto_phase"] = None, False
        mb.bg.STATE.live["phase_deg"] = 180.0
        mb.bg.COMMANDS.clear()


def test_http():
    mb.MOTION.close()
    mb.MOTION.buf.clear()
    mb.Handler.opts = NS(reader="x", tag="y", payload=600)
    srv = ThreadingHTTPServer(("127.0.0.1", 0), mb.Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_address[1]}"

    def post(path, body):
        req = urllib.request.Request(url + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
        return json.loads(urllib.request.urlopen(req).read())

    try:
        html = urllib.request.urlopen(url + "/").read().decode("utf-8")
        assert 'id="motionCard"' in html and "mpoll()" in html and 'id="liveStart"' in html
        # OptiTrack を受けていないと開始を断る (ワーカーは起動しない)
        r = post("/api/live/start", dict(level_dbm=0))
        assert r["ok"] is False and "OptiTrack" in r["message"] and mb.bg.STATE.mode == "idle"
        assert post("/api/motion/connect", dict(source="fake", group="lower"))["ok"]
        time.sleep(0.3)
        s = json.loads(urllib.request.urlopen(url + "/api/motion").read())
        assert s["receiving"] and s["group"] == "lower" and s["markers"][0]["name"] == "waist" and s["fps"] > 50
        assert post("/api/motion/options", dict(auto_phase=True))["ok"] is False      # モデル無し
        p = json.loads(urllib.request.urlopen(url + "/api/motion/pose").read())
        assert p["names"][0] == "waist" and len(p["xyz"]) == 7 and p["frame"] is not None
        assert s["layout"]["bones"][0][:2] == ["waist", "R_thigh"] and s["layout"]["angles"][0][0] == "右膝"
        assert post("/api/motion/close", {})["ok"]
        # 元の API にも届く (本文を読み直せている)
        assert post("/api/view", dict(span_us=20))["ok"] and mb.bg.STATE.span_us == 20
    finally:
        srv.shutdown()
        mb.MOTION.close()
        mb.MOTION.set_group("upper")


class FakeCap:
    """cv2.VideoCapture の代わり: 30 fps で番号入りのコマを返す。"""

    def __init__(self, *a):
        self.k = 0

    def isOpened(self):
        return True

    def set(self, *a):
        return True

    def read(self):
        time.sleep(1 / 30)
        self.k += 1
        f = np.zeros((240, 320, 3), np.uint8)
        f[:, :, 1] = self.k % 256
        return True, f

    def release(self):
        pass


def test_camera_stream_and_recording(tmp_path=None):
    import camera
    if camera.cv2 is None:
        return                                          # OpenCV が無い環境では飛ばす
    tmp = Path(tmp_path or tempfile.mkdtemp())
    real = camera.cv2.VideoCapture
    camera.cv2.VideoCapture = FakeCap
    mb.MOTION.close()
    mb.MOTION.buf.clear()
    mb.Handler.opts = NS(reader="x", tag="y", payload=600)
    srv = ThreadingHTTPServer(("127.0.0.1", 0), mb.Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_address[1]}"
    try:
        req = urllib.request.Request(url + "/api/camera/open", b'{"index": 0}', {"Content-Type": "application/json"})
        assert json.loads(urllib.request.urlopen(req).read())["ok"]
        time.sleep(0.5)
        st = json.loads(urllib.request.urlopen(url + "/api/motion").read())["camera"]
        assert st["running"] and (st["width"], st["height"]) == (320, 240) and st["fps"] > 15
        # MJPEG: 先頭のコマが JPEG で届く
        r = urllib.request.urlopen(url + "/api/camera.mjpg", timeout=5)
        assert "multipart/x-mixed-replace" in r.headers["Content-Type"]
        head = r.read(4096)
        assert b"Content-Type: image/jpeg" in head and b"\xff\xd8" in head
        r.close()
        # 記録の期間だけ録画して、コマごとの時刻が残る
        log = _session(tmp)
        assert log.video is not None and mb.CAMERA.sink is not None
        time.sleep(0.5)
        log.close()
        assert mb.CAMERA.sink is None
        rows = list(csv.DictReader(open(log.dir / "camera_frames.csv", encoding="utf-8")))
        assert 8 <= len(rows) <= 25 and (log.dir / "camera.mp4").stat().st_size > 1000
        t = [float(r["t"]) for r in rows]
        assert all(b > a for a, b in zip(t, t[1:])) and 0 <= t[0] < 0.2
        cond = json.loads((log.dir / "motion_conditions.json").read_text(encoding="utf-8"))
        assert cond["camera"]["width"] == 320
    finally:
        mb.CAMERA.close()
        camera.cv2.VideoCapture = real
        srv.shutdown()


def test_replay(tmp_path=None):
    """記録 (モーション・収録 1 回・カメラ) を作り、時刻で引いて同じものが返ること。"""
    import camera
    import replay
    if camera.cv2 is None:
        return
    tmp = Path(tmp_path or tempfile.mkdtemp())
    real = camera.cv2.VideoCapture
    camera.cv2.VideoCapture = FakeCap
    mb.MOTION.close()
    mb.MOTION.buf.clear()
    try:
        mb.CAMERA.open(0)
        time.sleep(0.3)
        log = _session(tmp)
        t_cap = log.perf0 + 0.2
        for k in range(60):                               # 0.6 秒ぶん、100 fps
            mb.MOTION._ingest(log.perf0 + k / 100, 500 + k, k / 100, None, np.full((7, 3), k * 1e-3))
        st = np.arange(0, 1_000_000, 4000, dtype=np.int64)
        mb.CAPTURE.update(t0=t_cap, n=10_000_000, fs=100e6, rx_fs=12.5e6,
                          res=dict(starts=st, status=np.zeros(st.size, int), frame_errors=np.arange(st.size) % 3))
        log.row(dict(t=0, level=0.0, phase=180.0, frames=250, expected=250, bits=150000, errors=249, q=0.9, eye=0.5),
                "cap_0001.ad3")                           # .ad3 は無いので raw は空扱いになる
        time.sleep(0.5)
        log.close()
    finally:
        mb.CAMERA.close()
        camera.cv2.VideoCapture = real
    r = replay.Replay(log.dir)
    assert r.names[0] == "chest" and len(r.caps) == 1 and r.caps[0]["raw"] == ""
    rows = list(csv.DictReader(open(log.dir / "ber_motion.csv", encoding="utf-8")))
    assert rows[0]["raw_file"] == "cap_0001.ad3"          # どの .ad3 かが残る
    p = r.pose(0.305)
    assert p["frame"] in (530, 531) and p["xyz"][0][0] == round((p["frame"] - 500) * 1e-3, 5)
    assert r.at(0.1)["cap"] is None                       # 収録の外
    a = r.at(0.2 + (8 + SKIP_RX) / 12.5e6 + 0.00032 * 8)  # 収録の中のフレーム 8 付近
    assert a["cap"] == 1 and a["frame"] in (7, 8) and a["frame_errors"] == a["frame"] % 3
    assert r.has_video and r.jpeg(0.3)[:2] == b"\xff\xd8"
    m = r.meta()
    assert m["n_raw"] == 0 and m["duration"] >= 0.59 and m["layout"]["bones"][0][0] == "chest"
    json.dumps(m)
    json.dumps(dict(r.pose(0.3), at=r.at(0.25)), allow_nan=False)   # numpy の型も NaN も混ざっていない
    r.caps[0]["ber"] = float("nan")                       # フレームが 1 枚も取れなかった収録
    json.dumps(dict(at=r.at(0.25), meta=r.meta()), allow_nan=False)
    # HTTP: 開く -> 測定の開始を断る -> 閉じる
    mb.Handler.opts = NS(reader="x", tag="y", payload=600)
    srv = ThreadingHTTPServer(("127.0.0.1", 0), mb.Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_address[1]}"

    def post(path, body):
        req = urllib.request.Request(url + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
        return json.loads(urllib.request.urlopen(req).read())

    try:
        L = json.loads(urllib.request.urlopen(url + "/api/replay/sessions").read())
        assert any(s["name"] == log.dir.name for s in L)
        assert post("/api/replay/open", dict(path=str(log.dir)))["ok"] and mb.bg.STATE.mode == "replay"
        assert json.loads(urllib.request.urlopen(url + "/api/replay/pose?t=0.3").read())["frame"] in (530, 531)
        assert urllib.request.urlopen(url + "/api/replay/frame.jpg?t=0.3").read()[:2] == b"\xff\xd8"
        assert post("/api/replay/view", dict(t=0.25)) == dict(ok=True, view=False, cap=1)
        assert post("/api/live/start", dict(level_dbm=0))["ok"] is False
        assert post("/api/replay/open", dict(path=str(tmp)))["ok"] is False      # 記録フォルダ以外は開かない
        assert post("/api/replay/close", {})["ok"] and mb.bg.STATE.mode == "idle"
    finally:
        mb.replay_close()
        srv.shutdown()


SKIP_RX = mb.SKIP


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
