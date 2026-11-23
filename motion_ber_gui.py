#!/usr/bin/env python3
"""motion_ber_gui.py -- OptiTrack のモーションキャプチャと ADP3250 の BER 測定を同時に回す GUI。

    uv run motion_ber_gui.py                 # http://127.0.0.1:8766 が開く
    uv run motion_ber_gui.py --fake-motion   # Motive 無し (合成モーション) で試す
    uv run motion_ber_gui.py --camera 0      # カメラも開く (画面の「カメラ」欄からも開ける)
    uv run motion_ber_gui.py --phase-model imp_phase_model.py   # 位相の自動調整 (モーション → インピーダンスの先読み)

BER 側は ../2026_nfc_vhbr_ber/instruments/ber_gui.py を**そのまま import して**使い、次の所だけ
差し替える (元のファイルは書き換えない):

  bench.Bridge.record          収録を始めた時刻 (perf_counter) を控える
  fast_demod.FastReceiver.process  直近の復調結果 (各フレームの位置と誤り数) を控える
  ber_gui.SessionLog           記録フォルダに motion.csv / ber_motion.csv / frames.csv / phase_pred.csv を足す
  ber_gui.live_worker          **置き換え** (このファイルの live_worker)。収録と処理を別スレッドにし、
                               位相の自動調整は「収録の直前に予測 → 位相を変えて 5 ms 待つ → 95 ms 撮る」
  ber_gui.Handler              画面にモーションの欄を足す

live_worker は ber_gui.live_worker の手順 (電源 → リーダ・タグ → Bridge → retune → 収録 → fast_ddc →
rx.process → make_view → STATE.push → SessionLog.row) を写している。**あちらの手順や名前を変えたら、ここも直すこと**
(あちらの改良は自動では効かない)。

同期のしかた: ADP3250 も Motive も外部トリガで結ばれていないので、どちらも PC の
time.perf_counter() で打った時刻で突き合わせる。
  収録窓   = [Configure を呼ぶ直前, + 点数 / fs]  (USB で開始を伝える遅れ ~1 ms は未較正)
  フレーム = 収録窓の先頭 + (starts + SKIP) / 受信部の fs   (DDC の遅れ < 1 µs は無視)
  モーション = NatNet のフレームを受けた時刻 (Motive の処理・配信の遅れ数 ms は未較正)
"""
from __future__ import annotations

import argparse
import collections
import ctypes
import io
import json
import os
import queue
import sys
import threading
import time
import traceback
import warnings
import webbrowser
from datetime import datetime
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np

HERE = Path(__file__).resolve().parent
BER_DIR = Path(os.environ.get("NFC_BER_DIR", HERE.parent / "2026_nfc_vhbr_ber"))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(BER_DIR / "instruments"))
sys.path.insert(0, str(BER_DIR))
import ber_gui as bg  # noqa: E402
import bench  # noqa: E402
import fast_demod as fd  # noqa: E402
from gui_view import SKIP, build_view  # noqa: E402

import camera  # noqa: E402
import motion  # noqa: E402
import overlay  # noqa: E402
import replay  # noqa: E402
import phase_control  # noqa: E402

MOTION = motion.MotionStream()
REPLAY: replay.Replay | None = None      # 開いている記録 (記録の再生)
REPLAY_LOCK = threading.Lock()
VIEWER = dict(on=False, calib=None, status="")    # --replay で開く再生用の画面と、動画への重ね合わせの較正
_RAW_CACHE: dict = {}                     # .ad3 のパス -> 復調の材料 (直近 2 本)
CAMERA = camera.Camera()
CAPTURE: dict = {}            # 直近の収録: t0 [perf], n, fs, rx_fs, res。収録スレッドが t0 を、処理スレッドが res を書く
OPTS = dict(require_motion=True, auto_phase=False, record_video=True, horizon=phase_control.HORIZONS[0],
            swap_lr=False,       # swap_lr: Motive のラベルが左右逆。位相のモデルに渡すマーカー名だけ R_* ↔ L_* にする
            offset_test=False,   # 補正量の試験: 自動補正に phase_control.TEST_OFFSETS を収録ごとに足す
            gain=phase_control.DEFAULT_GAIN)   # 補正のゲイン: 送る位相 = 初期位相 + gain × (モデルの位相 − 基準)
PHASE_MODEL = None
_TEST_SEQ: list = []          # 補正量の試験で、この巡にまだ使っていないオフセット
_TEST_RNG = np.random.default_rng()


def next_test_offset() -> float:
    """TEST_OFFSETS を 1 巡ずつ並べ替えて 1 つずつ返す (どのオフセットも同じ回数になる)。"""
    if not _TEST_SEQ:
        _TEST_SEQ.extend(_TEST_RNG.permutation(phase_control.TEST_OFFSETS).tolist())
    return float(_TEST_SEQ.pop())
PHASE_MODEL_NAME = ""
LAST: dict = {}               # 画面に出す直近の収録窓のモーション
# 画面の「補正の状況」: 収録 1 回 = 1 件 (BER とその収録で使った位相・補正量・先読みの実測)。記録ごとに空にする
CORR: collections.deque = collections.deque(maxlen=600)
CORR_LOCK = threading.Lock()  # 処理スレッドが足し、HTTP のスレッドが読む
CORR_SEND = 300               # /api/motion で返す件数 (直近)
# 位相の自動調整の基準: 初期位相・基準のモデルの位相 (warm の中央値)・最後に送った位相・基準を取る間のモデルの位相
REF = dict(base=None, ref=None, last=None, warm=[])


# ライブの 3D 表示の「予測を重ねる」: 収録の直前に位相を決めるのに**実際に使った**予測した姿勢と、その収録窓の実際の姿勢 (平均)。
# 収録スレッドが足し (live_pred_add / live_pred_window)、HTTP のスレッドが読む (live_pred_view)。記録ごとに空にする
LIVE_PRED: collections.deque = collections.deque(maxlen=8)
LIVE_PRED_LOCK = threading.Lock()


def live_pred_add(entry: dict) -> None:
    with LIVE_PRED_LOCK:
        LIVE_PRED.append(entry)


def live_pred_window(cap: int, t0: float, t1: float) -> None:
    """収録 cap の窓 [t0, t1] (perf)。窓が済んだら、その間の実際の姿勢と比べられる。"""
    with LIVE_PRED_LOCK:
        for e in LIVE_PRED:
            if e["cap"] == cap:
                e.update(t0=t0, t1=t1)


def live_pred_view() -> dict:
    """窓まで済んだいちばん新しい予測と、その窓の実際の姿勢 (平均)・誤差・位相 (初回だけ計算して覚える)。"""
    now = time.perf_counter()
    with LIVE_PRED_LOCK:
        done = [e for e in LIVE_PRED if e.get("t1") is not None and e["t1"] <= now]
        e = done[-1] if done else None
    if e is None:
        return dict(ready=False, message="自動補正した収録がまだありません（位相の自動調整を入れ、基準を取り終えると出ます）")
    if "actual" not in e:
        win = MOTION.window(e["t0"], e["t1"])
        act = None
        if win:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                act = np.nanmean(np.stack([f.xyz for f in win]), axis=0)
        err = None if act is None else np.linalg.norm(e["xyz"] - act, axis=1) * 1e3
        ph_act = None
        f = getattr(PHASE_MODEL, "phase_for_pose", None)
        if f is not None and act is not None:
            try:
                ph_act = f(e["names_model"], act)
            except Exception:                                # noqa: BLE001 (表示だけ。測定は止めない)
                ph_act = None
        e.update(actual=act, err=err, phase_act=ph_act)
    rnd = lambda a: None if a is None else [None if np.isnan(p).any() else [round(float(v), 5) for v in p] for p in a]
    num = lambda v, d=1: None if v is None or v != v else round(float(v), d)
    return dict(ready=True, live=True, cap=e["cap"], h=e["h"], xyz=rnd(e["xyz"]), ref=rnd(e["actual"]),
                err_mm=None if e["err"] is None else [num(v) for v in e["err"]], phase_pred=num(e["phase_pred"], 2),
                phase_act=num(e["phase_act"], 2), age_s=round(now - e["t1"], 2),
                lead_ms=round(((e["t0"] + e["t1"]) / 2 - e["t_motion"]) * 1e3, 1))


def reset_phase_ref() -> None:
    """次の収録で、そのときの画面の位相を初期位相として基準を取り直す。"""
    REF.update(base=None, ref=None, last=None, warm=[])


# --------------------------------------------------------------------------- 差し替え
_orig_record = bench.Bridge.record
_orig_process = fd.FastReceiver.process


def _record(self):
    t0 = time.perf_counter()                  # record() の最初が FDwfAnalogInConfigure (= 収録開始)
    codes, t = _orig_record(self)
    CAPTURE.update(t0=t0, n=int(codes.size), fs=float(self.fs))
    return codes, t


def _process(self, x, detail=False):
    r = _orig_process(self, x, detail=detail)
    CAPTURE.update(res=r, rx_fs=float(self.fs))
    return r


bench.Bridge.record = _record
fd.FastReceiver.process = _process


def _wall(epoch: float) -> str:
    return datetime.fromtimestamp(epoch).isoformat(sep=" ", timespec="milliseconds")


def _f(v, fmt="{:.6f}") -> str:
    return "" if v is None or (isinstance(v, float) and v != v) else fmt.format(v)


class MotionSessionLog(bg.SessionLog):
    """ber_gui の記録フォルダに、モーションと BER を突き合わせた CSV を足す。

      motion.csv      Motive の全フレーム (受信順)。t = 記録開始からの秒 (perf_counter)
      ber_motion.csv  収録 1 回 = 1 行: BER と、その収録窓のモーション (平均座標・動いた量・見えていた割合)
      frames.csv      テストフレーム 1 枚 = 1 行: 時刻・誤り数・いちばん近い Motive フレーム番号
      camera.mp4 / camera_frames.csv  カメラを繋いでいて「カメラも録画」のとき。コマごとの時刻つき
      motion_conditions.json  部位・マーカー名・時計の対応 (perf_counter ↔ 壁時計)

    t はどの CSV でも同じ時計なので、motion.csv と frames.csv は t か OptiFrame で結合できる。
    """

    def __init__(self, cond: dict):
        super().__init__(cond)
        reset_phase_ref()
        with LIVE_PRED_LOCK:
            LIVE_PRED.clear()
        self.perf0, self.wall0 = time.perf_counter(), time.time()
        self.names = MOTION.names
        self.n_cap = 0
        self.pred_by_cap: dict[int, dict] = {}      # phase_before_capture が決めた位相の情報 (収録が済んだら CORR へ)
        with CORR_LOCK:
            CORR.clear()
        self.mlock = threading.Lock()
        self.video = None
        if CAMERA.running and OPTS["record_video"]:
            try:
                self.video = camera.VideoLog(CAMERA, self.dir / "camera.mp4", self.dir / "camera_frames.csv",
                                             self.perf0)
            except Exception as e:                           # noqa: BLE001 (BER の測定は止めない)
                CAMERA.error = str(e)
        (self.dir / "motion_conditions.json").write_text(json.dumps(dict(
            camera=None if self.video is None else dict(
                index=CAMERA.index, width=CAMERA.width, height=CAMERA.height, nominal_fps=self.video.fps,
                note="camera.mp4 の再生速度は名目 fps。各コマの本当の時刻は camera_frames.csv の t"),
            started=_wall(self.wall0), source=MOTION.source, group=MOTION.group,
            marker_names=list(self.names), marker_model_id=motion.MARKER_MODEL_ID,
            units="m (Motive の座標系そのまま)",
            clock="t = time.perf_counter() - perf0。wall = wall0 + t",
            perf0=self.perf0, wall0=self.wall0,
            capture_window="[FDwfAnalogInConfigure の直前, + samples / fs]。開始の USB 遅れ (~1 ms) は未較正",
            motion_time="NatNet のフレームを受けた時刻。Motive の処理・配信の遅れ (数 ms) は未較正",
            phase_model=PHASE_MODEL_NAME or None,
            phase_model_swap_lr=OPTS["swap_lr"] if PHASE_MODEL_NAME else None,   # 開始時。途中の切り替えは phase_pred.csv の swap_lr
            # 補正量の試験 (開始時)。各収録で足した値は phase_pred.csv の test_offset (phase = base + gain × delta + test_offset)
            offset_test=OPTS["offset_test"] if PHASE_MODEL_NAME else None,
            test_offsets=list(phase_control.TEST_OFFSETS),
            gain=OPTS["gain"] if PHASE_MODEL_NAME else None,      # 開始時。各収録の値は phase_pred.csv の gain
        ), ensure_ascii=False, indent=2), encoding="utf-8")

        xyz = [f"{n}_{a}" for n in self.names for a in "XYZ"]
        self.mcsv = open(self.dir / "motion.csv", "w", encoding="utf-8", newline="", buffering=1 << 16)
        self.mcsv.write(",".join(["t", "wall", "OptiFrame", "OptiTime", "OptiMidExp", "n_valid"] + xyz) + "\n")
        self.bcsv = open(self.dir / "ber_motion.csv", "w", encoding="utf-8", newline="", buffering=1)
        self.bcsv.write(",".join(
            ["cap", "t_start", "t_end", "wall_start", "level", "phase", "phase_src", "frames", "expected",
             "bits", "errors", "ber", "q", "eye", "motion_frames", "valid_frac", "opti_frame_first",
             "opti_frame_last"] + xyz + [f"{n}_dev" for n in self.names] + ["raw_file"]) + "\n")
        self.fcsv = open(self.dir / "frames.csv", "w", encoding="utf-8", newline="", buffering=1 << 16)
        self.fcsv.write("cap,frame,t,errors,OptiFrame\n")
        self.pcsv = None            # phase_pred.csv: 位相のモデルが dict を返したときだけ、最初の 1 回で開く
        self.pcols: list[str] = []
        with MOTION.lock:
            MOTION.sink = self._motion_row

    # NatNet の受信スレッドから呼ばれる
    def _motion_row(self, f: motion.Frame) -> None:
        t = f.t - self.perf0
        n_valid = int((~np.isnan(f.xyz).any(axis=1)).sum())
        cells = ["" if v != v else f"{v:.6f}" for v in f.xyz.ravel()]
        line = ",".join([f"{t:.6f}", _wall(self.wall0 + t), _f(f.frame, "{}"), _f(f.motive_t),
                         _f(f.mid_exp, "{}"), str(n_valid)] + cells) + "\n"
        with self.mlock:
            if not self.mcsv.closed:
                self.mcsv.write(line)

    def row(self, p: dict, raw_file: str = "", cap: dict | None = None):
        super().row(p, raw_file)
        try:
            self._joined_row(p, raw_file, cap)
        except Exception as e:                               # noqa: BLE001 (BER の測定は止めない)
            traceback.print_exc()
            MOTION.error = f"突き合わせの記録に失敗: {type(e).__name__}: {e}"

    def _joined_row(self, p: dict, raw_file: str = "", cap: dict | None = None) -> None:
        """cap: この収録の t0 [perf]・n・fs・rx_fs・res・phase_src (処理を別スレッドで回すので、収録ごとに渡す)。
        無ければ CAPTURE (直近の収録) を使う。"""
        c = dict(cap) if cap is not None else dict(CAPTURE)
        r = c.get("res")
        if r is None or "t0" not in c:
            return
        self.n_cap = int(c["cap"]) if "cap" in c else self.n_cap + 1      # 収録スレッドの番号 (phase_pred.csv と同じ)
        t0, t1 = c["t0"], c["t0"] + c["n"] / c["fs"]
        win = MOTION.window(t0, t1)
        n = len(self.names)
        if win:
            xyz = np.stack([f.xyz for f in win])              # (k, 7, 3)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                mean = np.nanmean(xyz, axis=0)
                dev = np.nanmax(np.linalg.norm(xyz - mean, axis=2), axis=0)
            valid = float((~np.isnan(xyz).any(axis=2)).all(axis=1).mean())
        else:
            mean, dev, valid = np.full((n, 3), np.nan), np.full(n, np.nan), 0.0
        auto = c.get("phase_src") == "auto"
        ber =p["errors"] / p["bits"] if p["bits"] else float("nan")
        cells = [str(self.n_cap), f"{t0 - self.perf0:.6f}", f"{t1 - self.perf0:.6f}",
                 _wall(self.wall0 + t0 - self.perf0), _f(p.get("level"), "{:g}"), _f(p.get("phase"), "{:g}"),
                 "auto" if auto else "manual", str(p["frames"]), str(p["expected"]), str(p["bits"]),
                 str(p["errors"]), f"{ber:.4e}", _f(p.get("q"), "{:.4f}"), _f(p.get("eye"), "{:.4f}"),
                 str(len(win)), f"{valid:.3f}",
                 _f(win[0].frame if win else None, "{}"), _f(win[-1].frame if win else None, "{}")]
        cells += ["" if v != v else f"{v:.6f}" for v in mean.ravel()]
        cells += ["" if v != v else f"{v:.6f}" for v in dev]
        cells.append(raw_file)                               # この収録の生データ (.ad3)。無ければ空
        self.bcsv.write(",".join(cells) + "\n")

        # フレームごと: 時刻 = 収録窓の先頭 + 窓の位置 / 受信部の fs
        st = np.asarray(r["starts"], np.int64)
        fe = np.full(st.size, -1, np.int64)
        fe[r["status"] == 0] = r["frame_errors"]
        tf = t0 + (st + SKIP) / c["rx_fs"]
        opti = MOTION.nearest(tf)
        rel = tf - self.perf0
        self.fcsv.write("".join(f"{self.n_cap},{k},{rel[k]:.6f},{fe[k]},{opti[k]}\n" for k in range(st.size)))

        LAST.update(cap=self.n_cap, motion_frames=len(win), valid_frac=round(valid, 3),
                    dev_mm=[None if v != v else round(float(v) * 1e3, 1) for v in dev], ber=ber)
        pr = self.pred_by_cap.pop(self.n_cap, {})
        entry = dict(
            cap=self.n_cap, t=round(t0 - self.perf0, 3), errors=int(p["errors"]), bits=int(p["bits"]),
            frames=int(p["frames"]), expected=int(p["expected"]), phase=p.get("phase"), src="auto" if auto else "manual",
            # delta = 実際に入れた補正 (ゲインを掛けた後、試験のオフセットは含まない)。delta_model = モデルの補正量
            valid=round(valid, 3), delta=None if pr.get("delta") is None else round(pr["delta"] * pr.get("gain", 1.0), 3),
            delta_model=pr.get("delta"), gain=pr.get("gain"), base=pr.get("base"), ref=pr.get("ref"),
            warmup=pr.get("warmup"), n_warm=pr.get("n_warm"), horizon=pr.get("horizon"), test_offset=pr.get("test_offset"),
            # 先読みの実測: 最後のモーションのフレーム → この収録窓の中央 [ms]
            lead_ms=None if pr.get("t_motion") is None else round(((t0 + t1) / 2 - self.perf0 - pr["t_motion"]) * 1e3, 1))
        with CORR_LOCK:
            CORR.append(entry)

    def phase_before_capture(self, cap: int) -> float | None:
        """位相の自動調整。**収録の直前に**呼ぶ (収録ワーカーのスレッド)。

        いま届いているモーションで、固定の先読み OPTS["horizon"] 秒先のインピーダンスを推定し、
        送る位相 [°] を返す (None = 変えない)。呼んだ側が位相を変え (phase_control.PHASE_SETTLE_S 待つ)、
        そのまま収録 cap を撮る。決め方は phase_pred.csv に cap = **この位相で撮った収録** として残す。
        """
        if PHASE_MODEL is None or not OPTS["auto_phase"]:
            return None
        live = dict(bg.STATE.live)
        now = time.perf_counter()
        hist = MOTION.window(now - phase_control.HISTORY_SEC, now)
        history = (np.array([f.t - self.perf0 for f in hist]),
                   np.stack([f.xyz for f in hist]) if hist else np.empty((0, len(self.names), 3)))
        names = tuple(phase_control.swap_lr(n) for n in self.names) if OPTS["swap_lr"] else self.names
        last = hist[-1].xyz if hist else np.full((len(self.names), 3), np.nan)
        h = float(OPTS["horizon"])
        m = dict(names=names, xyz=last, valid=float(not np.isnan(last).any()),
                 t=(hist[-1].t if hist else now) - self.perf0, history=history, horizon=h,
                 phase_deg=live["phase_deg"], level_dbm=live["level_dbm"], div=live["div"])
        try:
            r = PHASE_MODEL.predict_phase(m)
        except Exception as e:                               # noqa: BLE001
            OPTS["auto_phase"] = False
            MOTION.error = f"位相モデルが失敗したので自動調整を切りました: {type(e).__name__}: {e}"
            return None
        info = dict(r) if isinstance(r, dict) else None
        private = {k: info.pop(k) for k in [k for k in info if k.startswith("_")]} if info else {}   # 画面だけに使う値
        p = info.get("phase") if info is not None else r
        if p is None:                                        # モデルが「分からない」→ 位相は変えない
            return None
        p, cur = float(p), live["phase_deg"]
        # 初期位相 = 基準を取ったときの画面の位相。そこから「モデルの位相が基準から動いたぶん」だけずらす。
        # 基準を取り直すのは、記録の開始・自動調整を入れたとき・先読みを変えたとき (reset_phase_ref) と、
        # 手で位相を変えたとき (画面の位相が最後に送った値と違う = それを新しい初期位相にする)
        if REF["base"] is None or abs(cur - REF["last"]) >= 0.01:
            REF.update(base=cur, ref=None, last=cur, warm=[])
        # 基準のモデルの位相は、最初の REF_WARMUP 回の中央値 (1 回だけだと、マーカーが隠れかけた回などの外れ値で
        # 記録全体がずれる。2026-11-18 18:26 の記録で、最初の 1 回が 144° (ふだん約 172°) → 全体が +28° ずれた)。
        # その間は位相を初期位相のまま変えない
        warming = len(REF["warm"]) < phase_control.REF_WARMUP
        off = 0.0                                            # 補正量の試験で足したオフセット [°]
        if warming:
            REF["warm"].append(p)
            REF["ref"] = float(np.median(REF["warm"]))
            v = REF["base"]
        else:
            if OPTS["offset_test"]:
                off = next_test_offset()
            v = phase_control.clamp(REF["base"] + OPTS["gain"] * (p - REF["ref"]) + off)
        sent = v is not None and abs(v - cur) >= 0.01
        if v is not None:
            with bg.STATE.lock:
                bg.STATE.live["phase_deg"] = v              # 画面の位相も合わせる (手で変えたら次から初期位相になる)
            REF["last"] = v
            LAST["auto_phase"] = v
        row = dict(phase_model=p, base=REF["base"], ref=round(REF["ref"], 3), warmup=int(warming),
                   delta=round(p - REF["ref"], 3), gain=OPTS["gain"], test_offset=off, swap_lr=int(OPTS["swap_lr"]),
                   t_motion=round(hist[-1].t - self.perf0, 6) if hist else None,
                   t_decided=round(time.perf_counter() - self.perf0, 6))
        LAST["pred"] = dict(info or {}, **row, horizon=h, phase=v)
        self.pred_by_cap[cap] = dict(row, horizon=h, n_warm=len(REF["warm"]))
        if v is not None and not warming and private.get("_pose") is not None and hist:
            # 3D 表示に重ねる: この収録の位相を決めた予測 (m["names"] の並び = 記録のマーカーの並び)
            live_pred_add(dict(cap=cap, xyz=np.asarray(private["_pose"], float), names_model=names, h=h,
                               t_motion=hist[-1].t, phase_pred=(info or {}).get("phase_actual")))
        pose_cols = {}
        if private.get("_pose") is not None:          # 予測した姿勢 (記録のマーカー名で。再生の「補正に使った予測」が読む)
            for n, xyz in zip(self.names, np.asarray(private["_pose"], float)):
                pose_cols.update({f"pred_{n}_{a}": (None if v != v else round(float(v), 6)) for a, v in zip("XYZ", xyz)})
        self._pred_row(cap, now - self.perf0, h, dict(row, **{k: x for k, x in (info or {}).items() if k != "phase"}, **pose_cols), v, sent)
        return v

    def _pred_row(self, cap: int, t, h: float, info: dict, v, sent: bool) -> None:
        """phase_pred.csv: 1 回ごとの位相の決め方 (先読み・初期位相・補正量) と、モデルが出した値 (推定したインピーダンスなど)。"""
        if self.pcsv is None:
            self.pcols = list(info)
            self.pcsv = open(self.dir / "phase_pred.csv", "w", encoding="utf-8", newline="", buffering=1)
            self.pcsv.write(",".join(["cap", "t", "horizon", "phase", "sent"] + self.pcols) + "\n")
        cells = [str(cap), f"{t:.6f}", f"{h:g}", _f(v, "{:g}"), "1" if sent else "0"]
        cells += [_f(info.get(k), "{}") for k in self.pcols]
        with self.mlock:
            if not self.pcsv.closed:
                self.pcsv.write(",".join(cells) + "\n")

    def close(self):
        with MOTION.lock:
            if MOTION.sink == self._motion_row:
                MOTION.sink = None
        if self.video is not None:
            self.video.close()
        with self.mlock:
            for f in (self.mcsv, self.bcsv, self.fcsv, self.pcsv):
                if f is not None:
                    f.close()
        super().close()


bg.SessionLog = MotionSessionLog


# --------------------------------------------------------------------------- 収録ワーカー
def set_phase_fast(b, deg: float) -> None:
    """出したまま W2 の位相だけ変え、PHASE_SETTLE_S だけ待つ (bench.Bridge.set_phase は 50 ms 待つ)。"""
    b.d.FDwfAnalogOutNodePhaseSet(b.h, 1, bench.AnalogOutNodeCarrier, ctypes.c_double(deg))
    b.d.FDwfAnalogOutConfigure(b.h, 1, 3)
    time.sleep(phase_control.PHASE_SETTLE_S)


def _process_capture(it: dict, payload: int) -> None:
    """処理スレッド: 収録 1 回ぶんを DDC → 復調 → 画面・記録へ (ber_gui.live_worker の後半と同じ手順)。"""
    S = bg.STATE
    codes, rec, rx = it["codes"], it["rec"], it["rx"]
    pk = int(np.abs(codes.astype(np.int32)).max())
    t1 = time.monotonic()
    x = fd.fast_ddc(codes, it["fs"], rec["volts_per_lsb"], rec["offset_v"], div=it["div"], decim=it["decim"])[SKIP:]
    t_ddc = time.monotonic() - t1
    bg.apply_rx_options(rx)
    res = rx.process(x, detail=True)
    view = bg.make_view(codes, rec["volts_per_lsb"], rec["offset_v"], it["fs"], x, rx, res, it["ddc_delay"])
    pt = S.push(res, view, payload, src="live", level=it["level"], phase=it["phase"],
                t_cap=round(it["t_cap"] * 1e3, 1), t_ddc=round(t_ddc * 1e3, 1),
                sig_ms=round(it["seconds"] * 1e3, 1), peak=round(pk / 32767 * 100, 1))
    bg.LAST_RAW.update(codes=codes, fs=it["fs"], rec=rec, frame=it["frame"], level=it["level"],
                       phase=it["phase"], info=it["info"], seconds=it["t_cap"])
    s = bg.SESSION
    raw_name = s.save_raw(bg.LAST_RAW).name if it["record_raw"] else ""
    s.row(pt, raw_name, cap=dict(cap=it["cap"], t0=it["t0"], n=int(codes.size), fs=it["fs"], rx_fs=float(rx.fs), res=res,
                                 phase_src=it["phase_src"]))
    clip = "  ※入力が振り切れています" if pk >= 32700 else ""
    bg.set_status(f"ライブ測定中（{it['level']:+.1f} dBm、位相差 {it['phase']:.2f}°"
                  f"{'・自動' if it['phase_src'] == 'auto' else ''}）{clip}　記録先: log/{s.dir.name}")


def live_worker(reader: str, tag: str, payload: int):
    """ber_gui.live_worker の置き換え。収録と処理を別のスレッドで回し、位相は収録の直前に決める。

    収録スレッド (これ): [操作の反映] → [位相の自動調整: 予測 → 位相を変える → PHASE_SETTLE_S 待つ]
                         → 収録 (窓 + USB 転送) → 処理スレッドへ渡す → すぐ次の周へ
    処理スレッド:        DDC → 復調 → 画面 → ber_log.csv / ber_motion.csv / frames.csv (+ 生データ)

    ADP3250 に触るのは収録スレッドだけ。処理が追いつかないときは、収録が 1 本ぶん待つ (キューの長さ 1)。
    窓の転送 (100 ms 分で約 0.87 s) の間は次を撮れないので、周期は「窓 + 転送 + 予測と位相 (約 35 ms)」で決まる。
    """
    from ad3 import Ad3
    from bench import Bridge, frame_meta, tag_play
    S, STOP = bg.STATE, bg.STOP
    q: queue.Queue = queue.Queue(maxsize=1)
    failed: list[BaseException] = []

    started = dict(n=0)                                     # 収録を始めた回数 (収録スレッドが record の直前に増やす)
    started_cv = threading.Condition()

    def processor():
        while True:
            it = q.get()
            if it is None:
                return
            # 次の収録が始まるまで待ってから処理する。転送の直後は次の予測 (収録の直前) と重なり、DDC (~150 ms) と
            # CPU を取り合って予測が 26 → 49 ms に延びていた (2026-11-18 18:27 の記録)。処理 (~0.3 s) は次の収録
            # (窓 + 転送 ~0.6〜0.9 s) の間に終わるので、周期は変わらない
            with started_cv:
                started_cv.wait_for(lambda: started["n"] > it["cap"] or STOP.is_set(), timeout=5.0)
            try:
                _process_capture(it, payload)
            except BaseException as e:                       # noqa: BLE001
                traceback.print_exc()
                failed.append(e)
                STOP.set()
                return

    def put(item) -> bool:
        while True:
            try:
                q.put(item, timeout=0.2)
                return True
            except queue.Full:
                if failed or (item is not None and STOP.is_set()):
                    return False

    proc = threading.Thread(target=processor, name="ber-process", daemon=True)
    try:
        bg.set_status("ブリッジに電源を入れています…", "")
        st = bg.power_set(True)
        if not st["ok"]:
            raise RuntimeError("電源を入れられません: " + st["message"])
        bg.set_status("リーダを開いています…")
        with Ad3(serial=reader, keep_running=True) as r:
            bg.set_status("タグを開いています…")
            with Ad3(serial=tag, keep_running=True) as t:
                cfg = dict(S.live)
                b = Bridge(r, bench.FC, bg.dbm_to_vset(cfg["level_dbm"]))
                try:
                    div = info = frame = ddc_delay = None
                    decim = 8
                    rx, rec, seconds = None, {}, None

                    def retune(d):
                        nonlocal div, info, frame, ddc_delay, seconds, decim
                        bg.set_status(f"タグの波形を FC/{d} で作り直しています…")
                        i = tag_play(t, "frame", d, payload, 0.0, 3.3)
                        div, info, frame = d, i, frame_meta(i, d, payload)
                        decim = bg.DECIM_FOR_DIV[d]
                        ddc_delay = (fd.ddc_taps(b.fs, d, decim).size - 1) / 2
                        seconds = None          # rec と rx を作り直させる

                    def drive(c):
                        if c["drive"] == "w1":
                            b.one(0)
                        elif c["drive"] == "w2":
                            b.one(1)
                        else:
                            b.both(c["phase_deg"], c["w2_scale"])

                    retune(cfg["div"])
                    drive(cfg)
                    bg.SESSION = bg.SessionLog(dict(cfg, payload=payload, reader=reader, tag=tag,
                                                    frame_s=info["frame_s"], sample_rate_hz=b.fs,
                                                    tag_gate="W1/W2 complementary 0 V / 3.3 V",
                                                    phase_settle_s=phase_control.PHASE_SETTLE_S,
                                                    pipeline="capture / process in separate threads"))
                    proc.start()
                    n_cap = 0
                    while not STOP.is_set():
                        # -- 操作の反映 (同じ項目は最後の値だけ) -----------------------------------
                        with S.lock:
                            cmds, bg.COMMANDS[:] = list(bg.COMMANDS), []
                            want = dict(S.live)
                        # データレート・駆動を変えたときは、ber_gui が測定ごと入れ直す (このワーカーを開き直す)
                        if any("level_dbm" in c or "w2_scale" in c for c in cmds):
                            b.level = bg.dbm_to_vset(want["level_dbm"])
                            drive(want)                              # 出し直し (W1/W2 同時起動)
                            time.sleep(0.2)
                        elif any("phase_deg" in c for c in cmds) and want["drive"] == "both":
                            b.set_phase(want["phase_deg"])          # 手で変えた位相 (元どおり 50 ms 待つ)
                        if seconds != want["seconds"]:
                            seconds = want["seconds"]
                            rec = b.setup_record(seconds)
                            rx = fd.FastReceiver(b.fs / decim, div=div, payload=payload,
                                                 frame_period_s=info["frame_s"])
                        # -- 位相の自動調整: 収録の直前に予測して、この収録の位相にする -------------
                        n_cap += 1
                        phase, src = want["phase_deg"], "manual"
                        if want["drive"] == "both" and isinstance(bg.SESSION, MotionSessionLog):
                            v = bg.SESSION.phase_before_capture(n_cap)
                            if v is not None:
                                if abs(v - phase) >= 0.01:
                                    set_phase_fast(b, v)
                                phase, src = v, "auto"
                        # -- 収録 (窓 + 転送)。前の収録の処理は、ここから別スレッドで始まる ---------------
                        with started_cv:
                            started["n"] = n_cap
                            started_cv.notify_all()
                        codes, t_cap = b.record()
                        t0 = CAPTURE["t0"]                           # _record が控えた収録開始 [perf]
                        live_pred_window(n_cap, t0, t0 + codes.size / b.fs)
                        if not put(dict(cap=n_cap, codes=codes.copy(), t0=t0, fs=float(b.fs), t_cap=t_cap, rec=rec, rx=rx,
                                        div=div, decim=decim, ddc_delay=ddc_delay, frame=frame,
                                        info=dict(r.info), level=want["level_dbm"], phase=phase, phase_src=src,
                                        seconds=seconds, record_raw=bool(want.get("record_raw")))):
                            break
                finally:
                    try:
                        with started_cv:                             # 次の収録はもう無い。残りはすぐ処理させる
                            started["n"] = 1 << 60
                            started_cv.notify_all()
                        put(None)
                        if proc.is_alive():
                            proc.join(timeout=30)
                    finally:
                        try:
                            tag_play(t, "off", 16, payload, 0.0, 3.3)
                        finally:
                            b.close()
        if failed:
            raise failed[0]
        bg.set_status("停止しました（出力オフ）" + (f"　記録: {bg.SESSION.dir}" if bg.SESSION else ""))
    except Exception as e:                                   # noqa: BLE001
        traceback.print_exc()
        bg.set_status("エラーで停止しました（出力はオフにしてあります）", f"{type(e).__name__}: {e}")
    finally:
        s = bg.SESSION
        if isinstance(s, MotionSessionLog):
            s.close()
        with S.lock:
            S.mode = "idle"


bg.live_worker = live_worker


# --------------------------------------------------------------------------- 画面
PANEL = """
  <section class="card controls" id="motionCard">
    <div class="grp">
      <div class="t">モーション</div>
      <label title="Motive のアセットに合わせる。記録中は変えられない">計測部位<select id="mGroup"></select></label>
      <label>取得元<select id="mSrc"><option value="natnet" selected>OptiTrack (NatNet)</option><option value="fake">合成（試験用）</option></select></label>
      <label title="Motive が動いている PC。同じ PC なら 127.0.0.1">Motive の IP<input id="mServer" type="text" value="127.0.0.1" style="width:110px"></label>
      <label title="この PC の、Motive と同じネットワークの IP">この PC の IP<input id="mLocal" type="text" value="127.0.0.1" style="width:110px"></label>
      <label class="chk" title="Motive の Data Streaming の Transmission Type に合わせる"><input id="mMulti" type="checkbox">Multicast</label>
      <button class="go" id="mConnect">接続</button>
      <button id="mClose">切断</button>
    </div>
    <div class="grp">
      <div class="t">カメラ</div>
      <label title="この PC では 1 = 外付けの C920、0 = 内蔵 (ふたを閉じていると真っ暗)">番号<input id="cIndex" type="number" min="0" max="9" step="1" value="1" style="width:52px"></label>
      <button class="go" id="cOpen">接続</button>
      <button id="cClose">切断</button>
      <label class="chk" title="BER の記録中、記録フォルダに camera.mp4 とコマごとの時刻 camera_frames.csv を残す"><input id="cRec" type="checkbox" checked>カメラも録画</label>
    </div>
    <div class="grp">
      <div class="t">記録の再生</div>
      <label title="motion.csv がある記録フォルダ。「生」= 波形まで見られる収録の数（ライブ欄の「生データも記録」を入れて撮ったもの）">記録<select id="rSess" style="max-width:340px"></select></label>
      <button id="rList" title="一覧を読み直す">↻</button>
      <button class="go" id="rOpen">開く</button>
      <button id="rClose">閉じる</button>
    </div>
    <div class="grp">
      <label class="chk" title="OFF: OptiTrack を受けていないときは BER の「開始」を断る"><input id="mNoReq" type="checkbox">OptiTrack なしでも開始できる</label>
      <label class="chk" id="mAutoL" title="入れたときの位相を初期位相にして、--phase-model のモデルが推定した変化ぶんだけ収録ごとにずらす。手で位相を変えると、それが新しい初期位相になる"><input id="mAuto" type="checkbox">位相の自動調整</label>
      <label title="モデルに渡す先読み (固定)。収録の直前に予測して位相を変えるので、最後のモーションから収録窓の中央までの時間に合わせる (95 ms の窓で約 100 ms)">先読み<select id="mHor">""" + "".join(
    f'<option value="{h:g}">{h * 1000:.0f} ms 先</option>' for h in phase_control.HORIZONS) + """</select></label>
      <label class="chk" title="Motive のラベルが左右逆 (R_* が本人の左腕) のとき入れる。位相のモデルに渡す名前だけ入れ替える。記録の CSV はラベルのまま (後で fix_lr.py で直す)"><input id="mSwap" type="checkbox">マーカーの左右が逆</label>
      <label class="chk" id="mOffTL" title="補正量の試験。自動補正した位相に、収録ごとに """ + " / ".join(f"{o:+g}" for o in phase_control.TEST_OFFSETS) + """° のどれかを順に (巡ごとに並べ替えて) 足す。姿勢ごとに BER が最小になるオフセット = モデルの補正量のずれ。ふだんの測定では切っておく"><input id="mOffT" type="checkbox">試験: 補正にオフセットを足す</label>
      <label id="mGainL" title="送る位相 = 初期位相 + ゲイン × モデルの補正量。1 = モデルのまま。2026-11-20 の補正量の試験では約 0.6 で打ち消しが最も深かった (学習データとコネクタが違うため)">補正のゲイン<input id="mGain" type="number" min="0" max="2" step="0.05" style="width:4.5em"></label>
    </div>
    <div style="flex-basis:100%;display:flex;flex-wrap:wrap;gap:4px 18px;align-items:center;font-size:13px">
      <span><span id="mDot" style="width:10px;height:10px;border-radius:50%;background:#888;display:inline-block;margin-right:6px"></span><b id="mText">未接続</b></span>
      <span id="mMarkers" style="display:flex;flex-wrap:wrap;gap:4px 12px;color:var(--sub)"></span>
      <span id="mLast" style="color:var(--sub)"></span>
      <span id="mErr" style="color:var(--red)"></span>
      <details style="font-size:12px;color:var(--sub)"><summary>受信フレームの構造（marker_id と名前の対応を確かめる）</summary><pre id="mDiag" style="margin:4px 0;white-space:pre-wrap"></pre></details>
    </div>
  </section>
  <style>
    #corrCard { position:relative; }
    #corrCard h2 { display:flex; flex-wrap:wrap; align-items:center; gap:2px 0; }
    .ctiles { display:grid; grid-template-columns:repeat(6, minmax(0, 1fr)); gap:8px; margin:2px 0 6px; }
    .ctile { border:1px solid var(--line); border-radius:8px; padding:5px 10px; min-width:0; }
    .ctile .k { font-size:12px; color:var(--sub); }
    .ctile .v { font-size:22px; font-weight:700; line-height:1.25; font-variant-numeric:tabular-nums; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
    .ctile .s { font-size:12px; color:var(--sub); line-height:1.4; }
    .ctile .dot { display:inline-block; width:9px; height:9px; border-radius:50%; margin-right:5px; vertical-align:middle; }
    .clegend { display:inline-flex; gap:14px; font-size:12px; color:var(--sub); font-weight:400; margin-left:10px; }
    .clegend i { display:inline-block; width:9px; height:9px; border-radius:50%; margin-right:4px; vertical-align:-1px; font-style:normal; }
    #corrTip { position:absolute; pointer-events:none; background:#1d2330; color:#fff; font-size:12px; line-height:1.5;
               padding:6px 9px; border-radius:6px; white-space:nowrap; z-index:5; display:none; }
    @media (max-width:1100px) { .ctiles { grid-template-columns:repeat(3, minmax(0, 1fr)); } }
  </style>
  <section class="card" id="corrCard">
    <h2>補正の状況<small>収録 1 回ごとの BER と、その収録で使った位相（記録中だけ）</small>
      <span class="clegend"><span><i style="background:#1f77b4"></i>自動補正</span><span><i style="border:1.5px solid #98a2b3;width:8px;height:8px"></i>補正なし（手動・基準取り・モデルが変えない）</span><span>▽ 誤り 0（上限 3 ÷ ビット数）</span><span><i style="background:rgba(152,162,179,.35);border-radius:2px"></i>基準を取っている間</span></span>
      <span style="margin-left:auto;display:flex;gap:16px;font-weight:400">
        <label class="chk">表示<select id="corrSpan"><option value="60">直近 1 分</option><option value="180" selected>直近 3 分</option><option value="0">全部（直近 300 回）</option></select></label>
        <label class="chk" title="下の BER の図を上に寄せる。ブラウザに覚える"><input type="checkbox" id="corrHideViews">3D とカメラを隠す</label>
      </span>
    </h2>
    <div class="ctiles">
      <div class="ctile"><div class="k">直近の収録の BER</div><div class="v" id="ctBer">—</div><div class="s" id="ctBerS"></div></div>
      <div class="ctile"><div class="k">直近 10 回の BER</div><div class="v" id="ctBer10">—</div><div class="s" id="ctBer10S"></div></div>
      <div class="ctile"><div class="k">この収録の位相差（指定値）</div><div class="v" id="ctPh">—</div><div class="s" id="ctPhS"></div></div>
      <div class="ctile"><div class="k">補正量（初期位相から）</div><div class="v" id="ctDel">—</div><div class="s" id="ctDelS"></div></div>
      <div class="ctile"><div class="k">先読み</div><div class="v" id="ctHor">—</div><div class="s" id="ctHorS"></div></div>
      <div class="ctile"><div class="k">自動補正した収録の BER（表示中）</div><div class="v" id="ctCmp">—</div><div class="s" id="ctCmpS"></div></div>
    </div>
    <canvas id="corrBer" data-h="150" data-fixed="1" style="height:150px"></canvas>
    <canvas id="corrPh" data-h="150" data-fixed="1" style="height:150px"></canvas>
    <div id="corrTip"></div>
  </section>
  <section class="card" id="rCard" hidden>
    <h2>記録の再生<small id="rInfo"></small></h2>
    <div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap">
      <button class="go" id="rPlay" style="min-width:80px">▶ 再生</button>
      <label class="chk">速さ<select id="rSpeed"><option value="0.02">×0.02</option><option value="0.1">×0.1</option><option value="0.25">×0.25</option><option value="1" selected>×1</option><option value="2">×2</option></select></label>
      <label class="chk" title="BER の収録 (0.1 秒) と収録の間 (転送待ち、約 0.5 秒) を飛ばして、収録の中だけを再生する"><input id="rSkip" type="checkbox">収録の中だけ</label>
      <button id="rPrev" title="前の収録の先頭へ">◀ 収録</button><button id="rNext" title="次の収録の先頭へ">収録 ▶</button>
      <button id="rFrPrev" title="テストフレーム 1 枚ぶん戻る">◀ フレーム</button><button id="rFrNext" title="テストフレーム 1 枚ぶん進む">フレーム ▶</button>
      <span id="rTime" style="font-variant-numeric:tabular-nums;font-weight:600"></span>
    </div>
    <input id="rSeek" type="range" min="0" max="1" step="any" value="0" style="width:100%;margin:6px 0 0">
    <canvas id="rTl" data-h="110" data-fixed="1" style="height:110px;cursor:pointer"></canvas>
    <div class="note" id="rAt"></div>
  </section>
  <div class="pair" id="mViews">
    <section class="card">
      <h2>マーカーの 3D 表示<small>ドラッグで回転、ホイールで拡大縮小、ダブルクリックで戻す</small><span style="float:right;display:inline-flex;flex-wrap:wrap;justify-content:flex-end;max-width:100%;gap:4px 14px;font-weight:400"><label class="chk" title="ライブ: 直近の収録の位相を決めるのに使った予測 (破線) と、その収録窓の実際の姿勢 (灰色、平均)。記録の再生: その時刻の、先読みの時間だけ前までのモーションから Ridge で予測した姿勢 (破線) を実際の姿勢に重ねる"><input type="checkbox" id="mPred">予測を重ねる</label><label class="chk" title="記録の再生のときだけ。連続 = その時刻の、先読みの時間だけ前までのモーションから予測した姿勢。補正に使った予測 = 収録窓の中だけ、その収録の位相を決めた予測 (と、その窓の実際の姿勢)">表示<select id="mPredMode" disabled><option value="both" selected>両方（収録中は補正に使った予測）</option><option value="cont">連続の予測だけ</option><option value="used">補正に使った予測だけ</option></select></label><label class="chk" title="記録の再生のときだけ選べる (ライブは位相の自動調整の先読みのまま)">先読み<select id="mPredH" disabled>""" + "".join(f'<option value="{h:g}"{" selected" if h == phase_control.HORIZONS[0] else ""}>{h * 1000:.0f} ms</option>' for h in (0.05, 0.1, 0.15, 0.2, 0.3)) + """</select></label><button id="mCamView" style="padding:2px 10px;font-weight:400" title="動画のカメラと同じ向き (本人の正面) から見る。ダブルクリックでも戻る">カメラの向き</button><label class="chk" style="font-weight:400" title="Motive の設定 (Up Axis) に合わせる。この台では Z が上">上の軸<select id="mUp"><option value="z" selected>Z</option><option value="y">Y</option></select></label></span></h2>
      <canvas id="m3d" data-h="360" data-fixed="1" style="height:360px;cursor:grab;touch-action:none"></canvas>
      <div id="mPredInfo" class="note" style="min-height:0;line-height:1.6"></div>
    </section>
    <section class="card">
      <h2>カメラ<small id="cInfo"></small></h2>
      <div style="height:360px;display:flex;align-items:center;justify-content:center;background:#11151c;border-radius:6px;overflow:hidden">
        <img id="cImg" alt="" style="max-width:100%;max-height:100%;display:none">
        <span id="cNone" style="color:#98a2b3;font-size:13px">カメラは繋がっていません</span>
      </div>
    </section>
  </div>
"""

PANEL_JS = """
<script>
"use strict";
(() => {
  let groupsFilled = false, layout = null, pose = null, camSeq = null;
  // 記録の再生の状態。t = 記録開始からの秒 (どのファイルも同じ時計)
  const R = { open: false, meta: null, t: 0, playing: false, last: 0, drag: false,
              viewKey: null, viewBusy: false, viewAt: 0, imgBusy: false, imgT: -1 };
  const val = id => document.getElementById(id);
  async function mpost(path, body) {
    const r = await fetch(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body || {}) });
    const j = await r.json(); if (j.ok === false && j.message) alert(j.message); return j;
  }
  async function mpoll() {
    try {
      const s = await (await fetch("/api/motion")).json();
      if (!R.open) layout = s.layout;
      if (!groupsFilled) {
        val("mGroup").innerHTML = Object.entries(s.groups).map(([k, v]) => `<option value="${k}">${v}</option>`).join("");
        groupsFilled = true;
      }
      if (document.activeElement !== val("mGroup")) val("mGroup").value = s.group;
      val("mGroup").disabled = s.logging;
      val("mNoReq").checked = !s.require_motion;
      val("mAuto").checked = s.auto_phase; val("mAuto").disabled = !s.model;
      val("mAutoL").lastChild.textContent = s.model ? `位相の自動調整（${s.model}）` : "位相の自動調整（--phase-model 未指定）";
      if (document.activeElement !== val("mHor")) val("mHor").value = String(s.horizon);
      val("mHor").disabled = !s.model;
      val("mSwap").checked = s.swap_lr; val("mSwap").disabled = !s.model;
      val("mOffT").checked = s.offset_test; val("mOffT").disabled = !s.model;
      val("mOffTL").style.color = s.offset_test ? "#d62728" : "";
      if (document.activeElement !== val("mGain")) val("mGain").value = String(s.gain);
      val("mGain").disabled = !s.model;
      val("mGainL").style.color = s.gain !== 1 ? "#1f77b4" : "";
      const src = { natnet: "OptiTrack", fake: "合成", "": "" }[s.source];
      val("mDot").style.background = s.receiving ? "#2ca02c" : (s.source ? "#f28e2b" : "#888");
      val("mText").textContent = !s.source ? "未接続" : s.receiving
        ? `${src} 受信中 ${s.fps} fps（フレーム #${s.frame}、${s.age_ms} ms 前）${s.logging ? "・記録中" : ""}`
        : `${src} 待機中（フレームが来ていません）`;
      val("mMarkers").innerHTML = s.markers.map(m =>
        `<span title="${m.pos ? m.pos.join(", ") + " m" : "見えていない"}"><span style="color:${m.pos ? "#2ca02c" : "#d62728"}">●</span> ${m.name}</span>`).join("");
      K.corr = s.corr || []; K.refWarm = s.ref_warmup || K.refWarm;
      renderTiles(); if (K.hover == null) drawCorr();
      const L = s.last;
      val("mLast").textContent = L && L.cap ? `直近の収録 #${L.cap}: モーション ${L.motion_frames} フレーム、全点見えていた ${(L.valid_frac * 100).toFixed(0)}%` +
        (L.auto_phase != null ? `、自動位相 ${L.auto_phase}°` : "") : "";
      const P = L && L.pred, z = (r, x) => `${r.toFixed(1)}${x < 0 ? "−" : "+"}j${Math.abs(x).toFixed(1)}`;
      if (P && P.L_Z_R != null) val("mLast").textContent += `　推定 Z（${Math.round(P.horizon * 1000)} ms 先）: 左 ${z(P.L_Z_R, P.L_Z_X)} Ω・右 ${z(P.R_Z_R, P.R_Z_X)} Ω`;
      if (P && P.base != null && P.warmup) val("mLast").textContent += `　基準を取っています（初期 ${P.base}° のまま）`;
      else if (P && P.base != null) val("mLast").textContent += `　初期 ${P.base}° ${P.delta < 0 ? "−" : "+"} 補正 ${Math.abs(P.delta * (P.gain ?? 1)).toFixed(2)}°` +
        (P.gain != null && P.gain !== 1 ? `（モデル ${P.delta.toFixed(2)}° × ゲイン ${P.gain}）` : "") +
        (P.test_offset ? ` ${P.test_offset < 0 ? "−" : "+"} 試験 ${Math.abs(P.test_offset)}°` : "") + ` → ` +
        (P.phase != null ? `指定 ${P.phase}°` : "90〜270° の外なので変えない");
      const c = s.camera;
      val("mErr").textContent = [s.error, c.error].filter(Boolean).join("　");
      val("mDiag").textContent = s.diag || "（まだフレームを受けていません）";
      // カメラ: 繋ぎ直したら配信を開き直す
      val("cRec").checked = s.record_video;
      val("cOpen").disabled = !c.available;
      if (!R.open) val("cInfo").textContent = !c.available ? "OpenCV がありません" : c.running
        ? `カメラ ${c.index}、${c.width}×${c.height}、${c.fps} fps${s.video_recording ? "・録画中" : ""}` : "";
      const want = R.open ? "replay" : c.running ? c.index + ":" + c.width : null;
      if (want !== camSeq && !R.open) {
        camSeq = want;
        val("cImg").src = want ? "/api/camera.mjpg?" + Date.now() : "";
        val("cImg").style.display = want ? "" : "none";
        val("cNone").style.display = want ? "none" : "";
      }
    } catch (e) { val("mText").textContent = "状態を読めません"; }
    setTimeout(mpoll, 500);
  }

  // ------------------------------------------------------------ 補正の状況 (収録ごとの BER と位相)
  // 描画は ber_gui.html の axes / line / empty / berOf / sci と同じ道具を使う (C も同じ色)
  const K = { corr: [], refWarm: 5, hover: null, ax: null };
  const GRAY = "#98a2b3", WARM = "rgba(152,162,179,.22)";
  const isAuto = c => c.src === "auto" && !c.warmup;
  const berTxt = (e, b) => { const r = berOf(e, b); return !r ? "—" : (r.zero ? "< " : "") + sci(r.v); };
  const berStat = v => v < 1e-5 ? ["良い", C.green] : v < 1e-3 ? ["注意", "#e0a100"] : ["悪い", C.red];
  const sumK = (cs, k) => cs.reduce((s, c) => s + (c[k] || 0), 0);
  const med = xs => { const a = [...xs].sort((p, q) => p - q); return a.length ? (a.length % 2 ? a[(a.length - 1) / 2] : (a[a.length / 2 - 1] + a[a.length / 2]) / 2) : null; };
  const dotHtml = v => { if (v == null) return ""; const [lab, col] = berStat(v); return `<span class="dot" style="background:${col}"></span>${lab}・`; };
  function stateOf(c) {
    if (c.src !== "auto") return "補正なし（手動、またはマーカーが欠けて変えない）";
    return c.warmup ? `基準を取っています ${c.n_warm} / ${K.refWarm}（初期位相のまま）` : "自動補正";
  }
  function corrVisible() {
    const cs = K.corr; if (!cs.length) return null;
    const span = +val("corrSpan").value, t1 = cs[cs.length - 1].t;
    const vis = span ? cs.filter(c => c.t >= t1 - span) : cs;
    return { vis, t0: span ? Math.max(t1 - span, vis[0].t) : vis[0].t, t1 };
  }
  function renderTiles() {
    const cs = K.corr, L = cs[cs.length - 1], set = (id, v, s) => { val(id).textContent = v; if (s !== undefined) val(id + "S").innerHTML = s; };
    if (!L) { ["ctBer", "ctBer10", "ctPh", "ctDel", "ctHor", "ctCmp"].forEach(id => set(id, "—", "")); return; }
    const r = berOf(L.errors, L.bits);
    set("ctBer", berTxt(L.errors, L.bits), dotHtml(r && r.v) + `#${L.cap}・誤り ${L.errors.toLocaleString()}・取得 ${L.frames}/${L.expected}`);
    const l10 = cs.slice(-10), e10 = sumK(l10, "errors"), b10 = sumK(l10, "bits"), r10 = berOf(e10, b10);
    set("ctBer10", berTxt(e10, b10), dotHtml(r10 && r10.v) + `誤り ${e10.toLocaleString()} / ${(b10 / 1e6).toFixed(2)} M ビット`);
    set("ctPh", L.phase != null ? `${L.phase.toFixed(2)}°` : "—", stateOf(L));
    set("ctDel", L.delta == null || L.warmup || L.src !== "auto" ? "—" : `${L.delta < 0 ? "−" : "+"}${Math.abs(L.delta).toFixed(2)}°`,
        (L.base != null ? `初期位相 ${L.base}°` : "") + (L.gain != null && L.gain !== 1 ? `・ゲイン ${L.gain}` : "") + (L.test_offset ? `・試験 ${L.test_offset > 0 ? "+" : ""}${L.test_offset}°` : ""));
    const leads = cs.slice(-10).map(c => c.lead_ms).filter(x => x != null);
    set("ctHor", L.horizon != null ? `${Math.round(L.horizon * 1000)} ms` : "—",
        leads.length ? `実測 ${Math.round(med(leads))} ms（モーション → 窓の中央）` : "");
    const R = corrVisible(), au = R.vis.filter(isAuto), ma = R.vis.filter(c => !isAuto(c));
    set("ctCmp", berTxt(sumK(au, "errors"), sumK(au, "bits")),
        `${au.length} 回・補正なし ${ma.length} 回は ${berTxt(sumK(ma, "errors"), sumK(ma, "bits"))}`);
  }
  function shadeWarm(a, vis) {
    const g = a.g; g.fillStyle = WARM;
    for (let i = 0; i < vis.length; i++) {
      if (vis[i].src !== "auto" || !vis[i].warmup) continue;
      let j = i; while (j + 1 < vis.length && vis[j + 1].src === "auto" && vis[j + 1].warmup) j++;
      const x0 = a.sx(vis[i].t - 0.4), x1 = a.sx(vis[j].t + 0.4);
      g.fillRect(x0, a.Y1, x1 - x0, a.Y0 - a.Y1); i = j;
    }
  }
  function mark(g, x, y, c, zero) {
    const auto = isAuto(c), col = auto ? C.blue : GRAY;
    g.lineWidth = 1.6;
    if (zero) {                                    // 誤り 0: 上限 (3 ÷ ビット数) に ▽
      g.strokeStyle = col; g.beginPath(); g.moveTo(x - 5, y - 4); g.lineTo(x + 5, y - 4); g.lineTo(x, y + 4); g.closePath(); g.stroke(); return;
    }
    g.beginPath(); g.arc(x, y, 4.5, 0, 6.3);
    if (auto) { g.fillStyle = col; g.fill(); g.strokeStyle = "#fff"; g.stroke(); } else { g.fillStyle = "#fff"; g.fill(); g.strokeStyle = col; g.stroke(); }
  }
  function hoverLine(a) {
    if (K.hover == null) return;
    const x = a.sx(K.hover.t), g = a.g; g.strokeStyle = "#475467"; g.lineWidth = 1; g.setLineDash([3, 3]);
    g.beginPath(); g.moveTo(x, a.Y1); g.lineTo(x, a.Y0); g.stroke(); g.setLineDash([]);
  }
  function drawCorr() {
    const R = corrVisible();
    if (!R || !R.vis.length) { empty(val("corrBer"), "BER の記録を始めると、収録ごとの BER がここに出ます"); empty(val("corrPh"), "その収録で使った位相差"); K.ax = null; return; }
    const vis = R.vis, xmin = R.t0 - 0.5, xmax = Math.max(R.t1 + 0.5, xmin + 10), M = { l: 62, r: 14 };
    const a = axes(val("corrBer"), { xmin, xmax, ymin: 1e-7, ymax: 1, logy: true, noXLabels: true, margin: Object.assign({ t: 8, b: 6 }, M), ylabel: "BER" });
    shadeWarm(a, vis);
    for (const c of vis) { const b = berOf(c.errors, c.bits); if (b) mark(a.g, a.sx(c.t), a.sy(b.v), c, b.zero); }
    hoverLine(a); a.done();
    const ph = vis.filter(c => c.phase != null), ys = ph.map(c => c.phase).concat(ph.filter(c => c.base != null).map(c => c.base));
    if (!ph.length) { empty(val("corrPh"), "—"); K.ax = null; return; }
    const lo = Math.min(...ys), hi = Math.max(...ys), pad = Math.max(2, (hi - lo) * 0.08);
    const b2 = axes(val("corrPh"), { xmin, xmax, ymin: lo - pad, ymax: hi + pad, ny: 4, margin: Object.assign({ t: 6, b: 34 }, M),
                                     xlabel: "記録開始からの時間 [秒]", ylabel: "位相差 [°]" });
    shadeWarm(b2, vis);
    const g = b2.g, bs = ph.filter(c => c.base != null);          // 初期位相 (手で変えると段になる)
    if (bs.length) { g.setLineDash([5, 4]); line(b2, bs.map(c => c.t), bs.map(c => c.base), GRAY, 1.5); g.setLineDash([]); }
    line(b2, ph.map(c => c.t), ph.map(c => c.phase), "rgba(31,119,180,.55)", 2);
    for (const c of ph) mark(g, b2.sx(c.t), b2.sy(c.phase), c, false);
    if (bs.length) {
      const yb = b2.sy(bs[bs.length - 1].base) - 3, tw = g.measureText("初期位相").width;
      g.fillStyle = "rgba(255,255,255,.85)"; g.fillRect(b2.X1 - 8 - tw, yb - 14, tw + 6, 14);
      g.fillStyle = C.sub; g.textAlign = "right"; g.textBaseline = "bottom"; g.fillText("初期位相", b2.X1 - 5, yb);
    }
    hoverLine(b2); b2.done();
    K.ax = { sx: a.sx, vis };
  }
  function corrHover(ev) {
    if (!K.ax) return;
    const cv = ev.currentTarget, rc = cv.getBoundingClientRect(), x = ev.clientX - rc.left;
    let best = null, bd = 24;
    for (const c of K.ax.vis) { const d = Math.abs(K.ax.sx(c.t) - x); if (d < bd) { bd = d; best = c; } }
    const tip = val("corrTip");
    if (!best) { K.hover = null; tip.style.display = "none"; drawCorr(); return; }
    K.hover = best; drawCorr();
    const c = best, card = val("corrCard").getBoundingClientRect();
    tip.innerHTML = `<b>収録 #${c.cap}</b>　${c.t.toFixed(1)} 秒<br>BER ${berTxt(c.errors, c.bits)}（${c.errors.toLocaleString()} / ${c.bits.toLocaleString()} ビット、取得 ${c.frames}/${c.expected}）<br>` +
      `位相差 ${c.phase != null ? c.phase.toFixed(2) + "°" : "—"}` + (c.delta != null && isAuto(c) ? `（初期 ${c.base}° ${c.delta < 0 ? "−" : "+"} ${Math.abs(c.delta).toFixed(2)}°` + (c.gain != null && c.gain !== 1 ? `（モデル ${c.delta_model.toFixed(2)}° × ${c.gain}）` : "") +
        (c.test_offset ? ` ${c.test_offset < 0 ? "−" : "+"} 試験 ${Math.abs(c.test_offset)}°` : "") + `）` : "") + `<br>` +
      `${stateOf(c)}` + (c.lead_ms != null ? `<br>先読み ${Math.round(c.horizon * 1000)} ms・実測 ${Math.round(c.lead_ms)} ms` : "") +
      `<br>窓の中でマーカーが全部見えていた ${Math.round(c.valid * 100)} %`;
    tip.style.display = "block";
    const left = Math.min(ev.clientX - card.left + 14, card.width - tip.offsetWidth - 8);
    tip.style.left = Math.max(8, left) + "px"; tip.style.top = (ev.clientY - card.top + 14) + "px";
  }
  for (const id of ["corrBer", "corrPh"]) {
    val(id).addEventListener("mousemove", corrHover);
    val(id).addEventListener("mouseleave", () => { K.hover = null; val("corrTip").style.display = "none"; drawCorr(); });
  }
  val("corrSpan").onchange = () => { renderTiles(); drawCorr(); };
  const hideViews = on => { val("mViews").hidden = on; if (typeof fit === "function") fit(); drawCorr(); };
  try { val("corrHideViews").checked = localStorage.getItem("corrHideViews") === "1"; } catch (e) { /* 保存できない環境 */ }
  hideViews(val("corrHideViews").checked);
  val("corrHideViews").onchange = () => {
    try { localStorage.setItem("corrHideViews", val("corrHideViews").checked ? "1" : "0"); } catch (e) { /* 保存できない環境 */ }
    hideViews(val("corrHideViews").checked);
  };
  window.addEventListener("resize", drawCorr);

  // ------------------------------------------------------------ 3D 表示
  // 描くときは Y を上にする。Motive が Z 上なら (x, y, z) -> (x, z, -y) に回してから描く (右手系のまま)
  // 既定の向き = 動画のカメラと同じ側 (本人の正面) から。2026-11-12・11-18 の較正でカメラは胸から見て yaw 2.99 rad・仰角 0.25 rad。
  // 記録の再生で overlay_calib.json があれば、その記録のカメラの位置から見る (meta.camera_view)
  const VIEW0 = { yaw: 2.99, pitch: 0.25 };
  let view = VIEW0;
  const V = { yaw: VIEW0.yaw, pitch: VIEW0.pitch, zoom: 1, c: null, r: null };
  function viewFromCamera(cam) {                 // Motive の座標のカメラ位置 → 描画の yaw / pitch (上の軸の選び方に合わせる)
    if (!cam) return VIEW0;
    const v = [0, 1, 2].map(k => cam.pos[k] - cam.body[k]);
    const d = val("mUp").value === "z" ? [v[0], v[2], -v[1]] : v;
    return { yaw: Math.atan2(d[0], d[2]), pitch: Math.atan2(d[1], Math.hypot(d[0], d[2])) };
  }
  function resetView() { V.yaw = view.yaw; V.pitch = view.pitch; V.zoom = 1; V.c = V.r = null; }
  const cv = val("m3d");
  let drag = null;
  cv.addEventListener("pointerdown", e => { drag = [e.clientX, e.clientY]; cv.setPointerCapture(e.pointerId); cv.style.cursor = "grabbing"; });
  cv.addEventListener("pointermove", e => {
    if (!drag) return;
    V.yaw += (e.clientX - drag[0]) * 0.01;
    V.pitch = Math.max(-1.5, Math.min(1.5, V.pitch + (e.clientY - drag[1]) * 0.01));
    drag = [e.clientX, e.clientY];
  });
  cv.addEventListener("pointerup", () => { drag = null; cv.style.cursor = "grab"; });
  cv.addEventListener("wheel", e => { e.preventDefault(); V.zoom = Math.max(0.2, Math.min(8, V.zoom * Math.exp(-e.deltaY * 0.001))); }, { passive: false });
  cv.addEventListener("dblclick", resetView);
  val("mCamView").onclick = resetView;
  val("mUp").onchange = () => { view = R.open && R.meta && R.meta.camera_view ? viewFromCamera(R.meta.camera_view) : VIEW0; resetView(); };

  function angle(a, b, c) {
    if (!a || !b || !c) return null;
    const u = [0, 1, 2].map(i => a[i] - b[i]), v = [0, 1, 2].map(i => c[i] - b[i]);
    const nu = Math.hypot(...u), nv = Math.hypot(...v);
    if (nu < 1e-9 || nv < 1e-9) return null;
    return Math.acos(Math.max(-1, Math.min(1, (u[0] * v[0] + u[1] * v[1] + u[2] * v[2]) / (nu * nv)))) * 180 / Math.PI;
  }

  function draw() {
    const dpr = window.devicePixelRatio || 1, W = cv.clientWidth, H = cv.clientHeight;
    if (cv.width !== Math.round(W * dpr) || cv.height !== Math.round(H * dpr)) { cv.width = Math.round(W * dpr); cv.height = Math.round(H * dpr); }
    const g = cv.getContext("2d");
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, W, H);
    g.font = "12px system-ui, sans-serif";
    if (!pose || !pose.xyz || !layout) {
      g.fillStyle = "#667085"; g.textAlign = "center"; g.fillText("モーションを受けていません", W / 2, H / 2); return;
    }
    const P = {}, seen = [];
    const zUp = val("mUp").value === "z";
    pose.names.forEach((n, i) => { const q = pose.xyz[i]; P[n] = q && (zUp ? [q[0], q[2], -q[1]] : q); if (P[n]) seen.push(P[n]); });
    if (seen.length) {                 // 中心はなめらかに追い、範囲は広がるときは速く・縮むときはゆっくり
      const m = [0, 1, 2].map(k => seen.reduce((s, p) => s + p[k], 0) / seen.length);
      V.c = V.c ? V.c.map((v, k) => v * 0.9 + m[k] * 0.1) : m;
      const r = Math.max(0.3, ...seen.map(p => Math.hypot(p[0] - V.c[0], p[1] - V.c[1], p[2] - V.c[2])));
      V.r = V.r ? Math.max(r, V.r * 0.995) : r;
    }
    if (!V.c) return;
    if (!seen.length) { g.fillStyle = "#d62728"; g.textAlign = "center"; g.fillText("マーカーが 1 つも見えていません", W / 2, H / 2); return; }
    const cy = Math.cos(V.yaw), sy = Math.sin(V.yaw), cp = Math.cos(V.pitch), sp = Math.sin(V.pitch);
    const s = Math.min(W, H) * 0.42 / V.r * V.zoom;
    const proj = p => {
      const x = p[0] - V.c[0], y = p[1] - V.c[1], z = p[2] - V.c[2];
      const x1 = x * cy - z * sy, z1 = x * sy + z * cy;
      const y2 = y * cp - z1 * sp, z2 = y * sp + z1 * cp;
      return [W / 2 + x1 * s, H / 2 - y2 * s, z2];
    };
    // 床のグリッド (いちばん低いマーカーの少し下、0.25 m 間隔)
    const yf = Math.min(...seen.map(p => p[1])) - 0.05, half = Math.ceil(V.r * 2 / 0.25) * 0.25;
    g.strokeStyle = "#e3e6ec"; g.lineWidth = 1;
    for (let d = -half; d <= half + 1e-9; d += 0.25) {
      for (const [a, b] of [[[V.c[0] + d, yf, V.c[2] - half], [V.c[0] + d, yf, V.c[2] + half]],
                            [[V.c[0] - half, yf, V.c[2] + d], [V.c[0] + half, yf, V.c[2] + d]]]) {
        const A = proj(a), B = proj(b); g.beginPath(); g.moveTo(A[0], A[1]); g.lineTo(B[0], B[1]); g.stroke();
      }
    }
    // ボーン
    g.lineWidth = 3; g.lineCap = "round";
    for (const [a, b, col] of layout.bones) {
      if (!P[a] || !P[b]) continue;
      const A = proj(P[a]), B = proj(P[b]);
      g.strokeStyle = col; g.beginPath(); g.moveTo(A[0], A[1]); g.lineTo(B[0], B[1]); g.stroke();
    }
    // マーカー (奥から)
    const pts = pose.names.filter(n => P[n]).map(n => [n, proj(P[n])]).sort((a, b) => a[1][2] - b[1][2]);
    g.textAlign = "left";
    for (const [n, q] of pts) {
      g.fillStyle = layout.colors[n] || "#000"; g.beginPath(); g.arc(q[0], q[1], 6, 0, 2 * Math.PI); g.fill();
      g.strokeStyle = "#fff"; g.lineWidth = 1.5; g.stroke();
      g.fillStyle = "#344054"; g.fillText(n, q[0] + 8, q[1] - 6);
    }
    // 予測した姿勢 (記録の再生で「予測を重ねる」のとき): 破線のボーンと白抜きの点。実際の点との間を細い線で結ぶ
    // 記録の再生: 収録窓の中で「補正に使った予測」があればそちら (表示が「連続の予測だけ」のときを除く)
    const U = pose.used, useU = R.open && U && U.ready && U.inside;
    const Q = useU ? U : (R.open && val("mPredMode").value === "used" ? (U && !U.ready ? U : { ready: true, none: true }) : pose.pred);
    if (Q && Q.ready && Q.xyz) {
      const PP = {}, RR = {}, tr = q => q && (zUp ? [q[0], q[2], -q[1]] : q);
      // 窓の平均の姿勢 (灰色) はライブだけ (いまの姿勢はもう窓から離れている)。記録の再生ではその時刻の実際の姿勢と結ぶ
      const showRef = Q.ref && Q.kind !== "used";
      pose.names.forEach((n, i) => { PP[n] = tr(Q.xyz[i]); RR[n] = showRef ? tr(Q.ref[i]) : P[n]; });
      if (showRef) {                       // ライブ: その収録窓の実際の姿勢 (平均) を灰色で
        g.lineWidth = 2; g.strokeStyle = "rgba(102,112,133,.55)";
        for (const [a, b] of layout.bones) {
          if (!RR[a] || !RR[b]) continue;
          const A = proj(RR[a]), B = proj(RR[b]); g.beginPath(); g.moveTo(A[0], A[1]); g.lineTo(B[0], B[1]); g.stroke();
        }
        g.fillStyle = "rgba(102,112,133,.75)";
        for (const n of pose.names) { if (!RR[n]) continue; const q = proj(RR[n]); g.beginPath(); g.arc(q[0], q[1], 4, 0, 2 * Math.PI); g.fill(); }
      }
      g.lineWidth = 1; g.strokeStyle = "rgba(71,84,103,.7)";
      for (const n of pose.names) {
        if (!RR[n] || !PP[n]) continue;
        const A = proj(RR[n]), B = proj(PP[n]); g.beginPath(); g.moveTo(A[0], A[1]); g.lineTo(B[0], B[1]); g.stroke();
      }
      g.lineWidth = 2.5; g.setLineDash([7, 5]);
      for (const [a, b, col] of layout.bones) {
        if (!PP[a] || !PP[b]) continue;
        const A = proj(PP[a]), B = proj(PP[b]);
        g.strokeStyle = col; g.beginPath(); g.moveTo(A[0], A[1]); g.lineTo(B[0], B[1]); g.stroke();
      }
      g.setLineDash([]);
      for (const n of pose.names) {
        if (!PP[n]) continue;
        const q = proj(PP[n]); g.strokeStyle = layout.colors[n] || "#000"; g.lineWidth = 2;
        g.beginPath(); g.arc(q[0], q[1], 6, 0, 2 * Math.PI); g.stroke();
      }
    }
    if (!val("mPred").checked) val("mPredInfo").textContent = "";
    if (val("mPred").checked) {
      g.font = "12px system-ui, sans-serif"; g.textAlign = "right"; g.fillStyle = "#344054";
      const lines = [];
      if (!Q || !Q.ready) lines.push(Q && Q.error ? "予測できません: " + Q.error : Q && Q.message ? Q.message : "予測を計算しています…");
      else if (Q.none) lines.push("補正に使った予測は、収録窓の中だけ出ます（基準を取っていた収録・位相を変えなかった収録は出ません）");
      else if (Q.kind === "used") {
        const ix = n => pose.names.indexOf(n), mm = v => v == null ? "—" : v.toFixed(0) + " mm";
        lines.push(`実線 = 実際の姿勢、破線 = 収録 #${Q.cap} の補正に使った予測（${Math.round(Q.h * 1000)} ms 先 = 窓の中央）`);
        lines.push(Q.recorded ? "予測は記録時の値" : `予測は${Q.legacy ? "記録時のモデル" : "今のモデル"}で計算し直した値（記録時の位相との差 ${Q.phase_diff_rec == null ? "—" : Q.phase_diff_rec.toFixed(2) + "°"}）`);
        if (Q.err_mm) lines.push(`誤差（この時刻の実際の姿勢と比べて）: 右前腕 ${mm(Q.err_mm[ix("R_forearm")])}・左前腕 ${mm(Q.err_mm[ix("L_forearm")])}・右肘 ${mm(Q.err_mm[ix("R_joint")])}・左肘 ${mm(Q.err_mm[ix("L_joint")])}`);
        if (Q.phase_pred != null && Q.phase_act != null)
          lines.push(`位相差: 予測 ${Q.phase_pred.toFixed(1)}° / この時刻の実際の姿勢 ${Q.phase_act.toFixed(1)}°（差 ${(Q.phase_pred - Q.phase_act).toFixed(1)}°）・指定した位相 ${Q.phase_applied}°`);
        if (Q.lead_ms != null) lines.push(`最後のモーション → 窓の中央 ${Math.round(Q.lead_ms)} ms`);
      }
      else if (Q.live) {
        const ix = n => pose.names.indexOf(n), mm = v => v == null ? "—" : v.toFixed(0) + " mm";
        lines.push(`破線 = 収録 #${Q.cap} の位相を決めた予測（${Math.round(Q.h * 1000)} ms 先）、灰色 = その収録窓の実際の姿勢（平均）`);
        if (Q.err_mm) lines.push(`誤差: 右前腕 ${mm(Q.err_mm[ix("R_forearm")])}・左前腕 ${mm(Q.err_mm[ix("L_forearm")])}・右肘 ${mm(Q.err_mm[ix("R_joint")])}・左肘 ${mm(Q.err_mm[ix("L_joint")])}`);
        if (Q.phase_pred != null && Q.phase_act != null)
          lines.push(`位相差: 予測 ${Q.phase_pred.toFixed(1)}° / 窓の実際の姿勢 ${Q.phase_act.toFixed(1)}°（差 ${(Q.phase_pred - Q.phase_act).toFixed(1)}°）`);
        lines.push(`最後のモーション → 窓の中央 ${Math.round(Q.lead_ms)} ms・窓から ${Q.age_s.toFixed(1)} 秒前`);
      } else {
        const ix = n => pose.names.indexOf(n), mm = v => v == null ? "—" : v.toFixed(0) + " mm";
        lines.push(`実線 = 実際の姿勢、破線 = ${Math.round(Q.h * 1000)} ms 前までのモーションから予測した姿勢`);
        if (!Q.ok) lines.push("この時刻は予測できません（直前のモーションが欠けている）");
        else {
          lines.push(`誤差: 右前腕 ${mm(Q.err_mm[ix("R_forearm")])}・左前腕 ${mm(Q.err_mm[ix("L_forearm")])}・右肘 ${mm(Q.err_mm[ix("R_joint")])}・左肘 ${mm(Q.err_mm[ix("L_joint")])}`);
          if (Q.phase_pred != null && Q.phase_act != null)
            lines.push(`位相差: 予測 ${Q.phase_pred.toFixed(1)}° / 実際の姿勢 ${Q.phase_act.toFixed(1)}°（差 ${(Q.phase_pred - Q.phase_act).toFixed(1)}°）`);
        }
        lines.push(`記録全体: 位置 RMSE ${Q.rmse_mm == null ? "—" : Q.rmse_mm.toFixed(1) + " mm"}・位相 RMSE ${Q.phase_rmse == null ? "—" : Q.phase_rmse.toFixed(2) + "°"}`);
      }
      val("mPredInfo").innerHTML = lines.join("<br>");      // 図の下に出す (図の中だと骨格に重なる)
      g.textAlign = "left";
    }
    // 座標軸 (左下、1 本 = 方向だけ)
    const o = [40, H - 34], ax = zUp
      ? [[1, 0, 0, "#d62728", "X"], [0, 0, -1, "#2ca02c", "Y"], [0, 1, 0, "#1f77b4", "Z"]]
      : [[1, 0, 0, "#d62728", "X"], [0, 1, 0, "#2ca02c", "Y"], [0, 0, 1, "#1f77b4", "Z"]];
    for (const [x, y, z, col, lab] of ax) {
      const x1 = x * cy - z * sy, z1 = x * sy + z * cy, y2 = y * cp - z1 * sp;
      g.strokeStyle = col; g.lineWidth = 2; g.beginPath(); g.moveTo(o[0], o[1]); g.lineTo(o[0] + x1 * 22, o[1] - y2 * 22); g.stroke();
      g.fillStyle = col; g.fillText(lab, o[0] + x1 * 28 - 3, o[1] - y2 * 28 + 4);
    }
    // 関節角度と見えていないマーカー
    g.font = "600 14px system-ui, sans-serif";
    layout.angles.forEach(([lab, [a, b, c], col], i) => {
      const v = angle(P[a], P[b], P[c]);
      g.fillStyle = col; g.fillText(`${lab}: ${v == null ? "—" : v.toFixed(1) + "°"}`, 10, 20 + i * 20);
    });
    g.font = "12px system-ui, sans-serif";
    const miss = pose.names.filter(n => !P[n]);
    if (miss.length) { g.fillStyle = "#d62728"; g.fillText("見えていない: " + miss.join(", "), 10, 64); }
    g.fillStyle = "#667085"; g.textAlign = "right";
    g.fillText(`フレーム #${pose.frame}${pose.receiving ? "" : "（途絶）"}`, W - 10, 18);
    // 向き: ドラッグで回していると、動画と左右が合わなくなる (背中側から見ると逆に見える)
    const dyaw = Math.abs(Math.atan2(Math.sin(V.yaw - view.yaw), Math.cos(V.yaw - view.yaw)));
    if (dyaw > 0.35) {
      g.fillStyle = dyaw > Math.PI / 2 ? "#d62728" : "#667085";
      g.fillText(dyaw > Math.PI / 2 ? "背中側から見ています（動画と左右が逆）。「カメラの向き」で戻す" : "回しています。「カメラの向き」で動画と同じ向きに戻す", W - 10, 36);
    }
  }

  async function ploop() {
    try {
      const md = val("mPredMode").value;
      const pq = R.open && val("mPred").checked ? (md !== "used" ? `&pred=${val("mPredH").value}` : "") + (md !== "cont" ? "&used=1" : "") : "";
      pose = await (await fetch(R.open ? `/api/replay/pose?t=${R.t}${pq}` : "/api/motion/pose" + (val("mPred").checked ? "?pred=1" : ""))).json();
      if (R.open && pose.at) rShowAt(pose.at);
    } catch (e) { pose = null; }
    draw();
    setTimeout(ploop, 33);
  }

  val("mConnect").onclick = () => mpost("/api/motion/connect", { source: val("mSrc").value, group: val("mGroup").value,
    server_ip: val("mServer").value, local_ip: val("mLocal").value, multicast: val("mMulti").checked });
  val("mClose").onclick = () => mpost("/api/motion/close");
  val("mGroup").onchange = () => mpost("/api/motion/options", { group: val("mGroup").value });
  val("mNoReq").onchange = () => mpost("/api/motion/options", { require_motion: !val("mNoReq").checked });
  val("mAuto").onchange = () => mpost("/api/motion/options", { auto_phase: val("mAuto").checked });
  val("mHor").onchange = () => mpost("/api/motion/options", { horizon: +val("mHor").value });
  val("mSwap").onchange = () => mpost("/api/motion/options", { swap_lr: val("mSwap").checked });
  val("mOffT").onchange = () => mpost("/api/motion/options", { offset_test: val("mOffT").checked });
  val("mGain").onchange = () => mpost("/api/motion/options", { gain: +val("mGain").value });
  val("cOpen").onclick = () => mpost("/api/camera/open", { index: +val("cIndex").value });
  val("cClose").onclick = () => mpost("/api/camera/close");
  val("cRec").onchange = () => mpost("/api/motion/options", { record_video: val("cRec").checked });
  // ------------------------------------------------------------ 記録の再生
  const fmtT = t => `${Math.floor(t / 60)}:${(t % 60).toFixed(3).padStart(6, "0")}`;
  const capAt = t => R.meta && R.meta.caps.find(c => c.t0 <= t && t <= c.t1);
  async function rList() {
    try {
      const L = await (await fetch("/api/replay/sessions")).json();
      val("rSess").innerHTML = L.length ? L.map(s => `<option value="${s.path}">${s.name}（収録 ${s.caps}・生 ${s.raw}${s.video ? "・動画" : ""}）</option>`).join("")
        : '<option value="">記録がありません</option>';
    } catch (e) { }
  }
  function rSetOpen(meta) {
    R.open = !!meta; R.meta = meta; R.t = meta && meta.caps.length ? meta.caps[0].t0 : 0; R.playing = false;
    R.viewKey = null; R.imgT = -1;
    val("rCard").hidden = !meta;
    V.c = V.r = null;
    val("cImg").src = "";
    val("mPredH").disabled = val("mPredMode").disabled = !meta;
    // 表示・先読みを選んだら重ねる (「予測を重ねる」が入っていないと何も出ないため)
    val("mPredMode").onchange = val("mPredH").onchange = () => { val("mPred").checked = true; };
    view = meta && meta.camera_view ? viewFromCamera(meta.camera_view) : VIEW0;
    resetView();
    if (meta) {
      if (meta.horizon) {                  // 記録で使った先読みに合わせる (選択肢に無ければいちばん近いもの)
        const hs = [...val("mPredH").options].map(o => +o.value);
        val("mPredH").value = String(hs.reduce((a, b) => Math.abs(b - meta.horizon) < Math.abs(a - meta.horizon) ? b : a));
      }
      layout = meta.layout;
      val("rSeek").max = meta.duration;
      val("rInfo").textContent = `${meta.name}　${meta.duration.toFixed(1)} 秒・収録 ${meta.caps.length}（生データ ${meta.n_raw}）・モーション ${meta.motion_frames} フレーム` +
        `${meta.video ? "・動画あり" : "・動画なし"}　FC/${meta.div}・${meta.level_dbm} dBm・${meta.phase_deg}°`;
      val("cImg").style.display = meta.video ? "" : "none";
      val("cNone").style.display = meta.video ? "none" : "";
      val("cNone").textContent = "この記録には動画がありません";
      val("cInfo").textContent = meta.video ? "記録の再生" : "";
    } else {
      camSeq = null;                     // mpoll がライブの配信に戻す
      val("cNone").textContent = "カメラは繋がっていません";
    }
  }
  function rShowAt(a) {
    val("rAt").textContent = a.cap == null ? "収録の外（この時刻には BER がありません。モーションとカメラだけ）"
      : `収録 #${a.cap}: BER ${a.ber == null ? "—" : a.ber.toExponential(2)}（${a.frames}/${a.expected} フレーム）` +
        (a.frame != null ? `　フレーム ${a.frame}: ${a.frame_errors < 0 ? "取得失敗" : `誤り ${a.frame_errors} ビット`}` : "") +
        (a.raw ? "　波形 = ①〜③" : "　（生データなし: 波形は出ません）");
  }
  function rSeek(t, pause) {
    if (!R.meta) return;
    R.t = Math.max(0, Math.min(R.meta.duration, t));
    if (pause) R.playing = false;
  }
  function rDrawTimeline() {
    const cv = val("rTl"), dpr = window.devicePixelRatio || 1, W = cv.clientWidth, H = cv.clientHeight;
    if (cv.width !== Math.round(W * dpr) || cv.height !== Math.round(H * dpr)) { cv.width = Math.round(W * dpr); cv.height = Math.round(H * dpr); }
    const g = cv.getContext("2d"); g.setTransform(dpr, 0, 0, dpr, 0, 0); g.clearRect(0, 0, W, H);
    const m = R.meta; if (!m) return;
    const L = 44, Rr = 10, T = 8, B = 22, d = Math.max(m.duration, 1e-6);
    const x = t => L + (W - L - Rr) * t / d, y = b => T + (H - T - B) * (1 - (Math.log10(Math.max(b, 1e-6)) + 6) / 6);
    g.font = "11px system-ui, sans-serif"; g.fillStyle = "#667085"; g.strokeStyle = "#e8ebf0"; g.lineWidth = 1;
    for (let e = -6; e <= 0; e += 2) { const yy = y(10 ** e); g.beginPath(); g.moveTo(L, yy); g.lineTo(W - Rr, yy); g.stroke(); g.textAlign = "right"; g.fillText(`1e${e}`, L - 4, yy + 4); }
    const step = [1, 2, 5, 10, 20, 30, 60, 120, 300].find(s => d / s <= 12) || 600;
    g.textAlign = "center";
    for (let t = 0; t <= d + 1e-9; t += step) g.fillText(`${t}s`, x(t), H - 6);
    for (const c of m.caps) {
      const x0 = x(c.t0), w = Math.max(2, x(c.t1) - x0);
      g.fillStyle = c.raw ? "#1f77b4" : "#98a2b3";
      if (c.ber == null || c.errors === 0) { g.fillRect(x0, H - B - 4, w, 4); continue; }
      const yy = y(c.ber); g.fillRect(x0, yy, w, H - B - yy);
    }
    g.strokeStyle = "#d62728"; g.lineWidth = 2; g.beginPath(); g.moveTo(x(R.t), T - 4); g.lineTo(x(R.t), H - B + 2); g.stroke();
    g.fillStyle = "#667085"; g.textAlign = "left"; g.fillText("BER（青 = 生データあり、灰 = なし）", L + 4, T + 10);
  }
  function rUpdate(now) {
    if (!R.open) return;
    if (!R.drag) val("rSeek").value = R.t;
    val("rTime").textContent = `${fmtT(R.t)} / ${fmtT(R.meta.duration)}`;
    val("rPlay").textContent = R.playing ? "❚❚ 停止" : "▶ 再生";
    rDrawTimeline();
    // カメラ: 前のコマを読み終えてから次を頼む (読み込みが自然な間引きになる)
    if (R.meta.video && !R.imgBusy && Math.abs(R.imgT - R.t) > 1e-4) {
      R.imgBusy = true; R.imgT = R.t;
      const img = val("cImg");
      img.onload = img.onerror = () => { R.imgBusy = false; };
      img.src = `/api/replay/frame.jpg?t=${R.t}`;
    }
    // 波形: 生データのある収録の中で、テストフレームが変わったら復調し直す (再生中は 0.3 秒に 1 回まで)
    const c = capAt(R.t);
    if (c && c.raw && !R.viewBusy) {
      const key = `${c.cap}:${Math.floor((R.t - c.t0) / (R.meta.frame_s || 1e-3))}`;
      if (key !== R.viewKey && (!R.playing || now - R.viewAt > 300)) {
        R.viewBusy = true; R.viewAt = now;
        fetch("/api/replay/view", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ t: R.t }) })
          .then(r => r.json()).then(j => { R.viewKey = key; if (j.ok === false) val("rAt").textContent = j.message; })
          .catch(() => { }).finally(() => { R.viewBusy = false; });
      }
    }
  }
  function rTick(now) {
    if (R.open && R.playing) {
      R.t += (now - R.last) / 1000 * +val("rSpeed").value;
      if (val("rSkip").checked && !capAt(R.t)) {
        const nx = R.meta.caps.find(c => c.t0 > R.t);
        if (nx) R.t = nx.t0; else { R.t = R.meta.duration; }
      }
      if (R.t >= R.meta.duration) { R.t = R.meta.duration; R.playing = false; }
    }
    R.last = now;
    rUpdate(now);
    requestAnimationFrame(rTick);
  }
  val("rList").onclick = rList;
  val("rOpen").onclick = async () => { if (!val("rSess").value) return; const j = await mpost("/api/replay/open", { path: val("rSess").value }); if (j.ok) rSetOpen(j.meta); };
  val("rClose").onclick = async () => { await mpost("/api/replay/close"); rSetOpen(null); };
  val("rPlay").onclick = () => { if (R.t >= R.meta.duration) R.t = 0; R.playing = !R.playing; };
  val("rSeek").addEventListener("input", () => { R.drag = true; rSeek(+val("rSeek").value); });
  val("rSeek").addEventListener("change", () => { R.drag = false; });
  val("rTl").addEventListener("click", e => {
    const b = val("rTl").getBoundingClientRect(), L = 44, Rr = 10;
    rSeek((e.clientX - b.left - L) / (b.width - L - Rr) * R.meta.duration);
  });
  val("rPrev").onclick = () => { const c = [...R.meta.caps].reverse().find(c => c.t0 < R.t - 1e-3); if (c) rSeek(c.t0 + 1e-4, true); };
  val("rNext").onclick = () => { const c = R.meta.caps.find(c => c.t0 > R.t + 1e-6); if (c) rSeek(c.t0 + 1e-4, true); };
  val("rFrPrev").onclick = () => rSeek(R.t - (R.meta.frame_s || 1e-3), true);
  val("rFrNext").onclick = () => rSeek(R.t + (R.meta.frame_s || 1e-3), true);
  rList();
  fetch("/api/replay/meta").then(r => r.json()).then(m => { if (m) rSetOpen(m); }).catch(() => { });
  requestAnimationFrame(rTick);
  mpoll(); ploop();
})();
</script>
"""


# --------------------------------------------------------------------------- 記録の再生
# ---------------------------------------------------------------- 先読みの答え合わせ (再生の 3D 表示)
# 記録の全フレームについて「その時刻の h 秒前までの履歴から Ridge で予測した姿勢」を、記録を開いたあと裏で 1 回だけ
# 計算して覚えておく (imp_phase_model.forecast_series。2 万フレームで 1 秒弱)。3D 表示が実際の姿勢に重ねて描く
FC = dict(key=None, res=None, busy=None, error="")
FC_LOCK = threading.Lock()


def _model_names(r: replay.Replay) -> tuple:
    """モデルに渡すマーカー名 (本人の左右)。fix_lr で直した記録はラベルどおり、記録中に「左右が逆」を入れていたら入れ替える。"""
    if r.mc.get("lr_fixed") or not r.mc.get("phase_model_swap_lr"):
        return r.names
    return tuple(phase_control.swap_lr(n) for n in r.names)


def replay_forecast(r: replay.Replay, h: float) -> dict:
    """再生中の記録の先読みの計算結果。まだなら裏で計算を始めて dict(ready=False) を返す。"""
    key = (str(r.dir), round(float(h), 3))
    with FC_LOCK:
        if FC["key"] == key and FC["res"] is not None:
            return dict(ready=True, res=FC["res"])
        if FC["key"] == key and FC["error"]:
            return dict(ready=False, error=FC["error"])
        if FC["busy"] == key:
            return dict(ready=False)
        FC.update(key=key, res=None, busy=key, error="")

    def work():
        try:
            # ライブと同じモデルを使う (--phase-model で読み込んだもの。無ければ imp_phase_model を読む)
            if hasattr(PHASE_MODEL, "forecast_series"):
                M = PHASE_MODEL
            else:
                import imp_phase_model as M
            res = M.forecast_series(r.tm, r.xyz, _model_names(r), key[1])
            err = np.linalg.norm(res["xyz_pred"] - r.xyz, axis=2)            # (N, 7) [m]。どちらかが無ければ NaN
            res["err"] = err
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                res["rmse_mm"] = float(np.sqrt(np.nanmean(err ** 2)) * 1e3)
                res["phase_rmse"] = float(np.sqrt(np.nanmean((res["phase_pred"] - res["phase_act"]) ** 2)))
            with FC_LOCK:
                if FC["key"] == key:
                    FC.update(res=res, busy=None)
        except Exception as e:                               # noqa: BLE001 (モデルが無いなど。画面に出す)
            traceback.print_exc()
            with FC_LOCK:
                if FC["key"] == key:
                    FC.update(busy=None, error=f"{type(e).__name__}: {e}")

    threading.Thread(target=work, name="replay-forecast", daemon=True).start()
    return dict(ready=False)


# ---------------------------------------------------------------- 補正に実際に使った予測 (再生の 3D 表示)
# 収録ごとに「その収録の位相を決めた予測した姿勢」と「その収録窓の実際の姿勢 (平均)」。2026-11-19 以降の記録は
# phase_pred.csv の pred_<マーカー>_X/Y/Z (記録時の予測そのもの)、それより前は同じ履歴から今のモデルで計算し直す
# (Ridge を作り直したので記録時とは少し違う。記録時の phase_actual との差を出す)
USED = dict(key=None, res=None, busy=False, error="")
USED_LOCK = threading.Lock()
_LEGACY = {}                  # 作り直す前のモデル (古い記録の「補正に使った予測」を記録時と同じモデルで出し直す用)。1 回だけ読む


def _legacy_bundle(M, r: replay.Replay):
    """記録が Ridge を作り直す前 (LEGACY_UNTIL より前) なら、そのときのモデル。無ければ None (= 今のモデルを使う)。"""
    until, path = getattr(M, "LEGACY_UNTIL", None), getattr(M, "LEGACY_MODEL_PATH", None)
    if not until or path is None or not Path(path).exists() or str(r.cond.get("started", ""))[:10] >= until:
        return None
    if "b" not in _LEGACY:
        _LEGACY["b"] = M.load_bundle(path)
    return _LEGACY["b"]


def _used_compute(r: replay.Replay) -> dict:
    import csv as _csv
    if hasattr(PHASE_MODEL, "forecast_impedance") and hasattr(PHASE_MODEL, "phase_for_pose"):
        M = PHASE_MODEL
    else:
        import imp_phase_model as M
    f = r.dir / "phase_pred.csv"
    if not f.exists():
        return {}
    rows = list(_csv.DictReader(open(f, encoding="utf-8")))
    new_cap = "pipeline" in r.cond                 # 2026-11-18 から cap = その位相で撮った収録 (前は直前の収録)
    caps = {c["cap"]: c for c in r.caps}
    vis = ~np.isnan(r.xyz).any(axis=(1, 2))
    fl = lambda v: float(v) if v not in (None, "") else float("nan")
    legacy = _legacy_bundle(M, r)
    out = {}
    for row in rows:
        if row.get("warmup") == "1" or row.get("phase", "") == "":
            continue                                    # 基準を取っていた回・位相を変えなかった回は「補正に使った」ではない
        cap = int(row["cap"]) + (0 if new_cap else 1)
        c = caps.get(cap)
        if c is None:
            continue
        names_m = tuple(phase_control.swap_lr(n) for n in r.names) if row.get("swap_lr") == "1" else r.names
        t_call, h = fl(row["t"]), fl(row["horizon"])
        cols = [f"pred_{n}_{a}" for n in r.names for a in "XYZ"]
        recorded = all(k in row for k in cols)
        ph_diff = None
        if recorded:
            pose = np.array([fl(row[k]) for k in cols]).reshape(len(r.names), 3)
            ph_pred = fl(row.get("phase_actual"))
        else:
            sel = (r.tm > t_call - phase_control.HISTORY_SEC) & (r.tm <= t_call)
            z, pose = M.forecast_impedance(r.tm[sel], r.xyz[sel], names_m, h, with_pose=True, bundle=legacy)
            if z is None:
                continue
            ph_pred = M.phase_for_impedance(z)
            if row.get("phase_actual", "") != "":
                ph_diff = ph_pred - fl(row["phase_actual"])
        w = (r.tm >= c["t0"]) & (r.tm <= c["t1"]) & vis
        act = r.xyz[w].mean(axis=0) if w.any() else None
        err = None if act is None else np.linalg.norm(pose - act, axis=1) * 1e3
        ph_act = None if act is None else M.phase_for_pose(names_m, act, bundle=None if recorded else legacy)
        tm_last = r.tm[r.tm <= t_call]
        t_motion = fl(row["t_motion"]) if row.get("t_motion", "") != "" else (tm_last[-1] if tm_last.size else float("nan"))
        out[cap] = dict(cap=cap, t0=c["t0"], t1=c["t1"], xyz=pose, ref=act, err=err, h=h, phase_pred=ph_pred,
                        phase_act=ph_act, recorded=recorded, legacy=legacy is not None and not recorded, phase_diff_rec=ph_diff, phase_applied=fl(row["phase"]),
                        lead_ms=((c["t0"] + c["t1"]) / 2 - t_motion) * 1e3,
                        names_m=names_m, model=M, bundle=None if recorded else legacy)
    return out


def replay_used(r: replay.Replay, t: float) -> dict:
    """時刻 t を含む収録の「補正に使った予測」。収録の外・補正していない収録なら None。"""
    key = str(r.dir)
    with USED_LOCK:
        if USED["key"] != key:
            USED.update(key=key, res=None, busy=False, error="")
        if USED["res"] is None and not USED["busy"] and not USED["error"]:
            USED["busy"] = True

            def work():
                try:
                    res = _used_compute(r)
                    with USED_LOCK:
                        if USED["key"] == key:
                            USED.update(res=res, busy=False)
                except Exception as e:                       # noqa: BLE001
                    traceback.print_exc()
                    with USED_LOCK:
                        if USED["key"] == key:
                            USED.update(busy=False, error=f"{type(e).__name__}: {e}")

            threading.Thread(target=work, name="replay-used", daemon=True).start()
        res, err = USED["res"], USED["error"]
    if res is None:
        return dict(ready=False, error=err)
    c = r.cap_at(t)
    e = None if c is None else res.get(c["cap"])
    if e is None:
        return dict(ready=True, inside=False, n=len(res))
    # 誤差と位相差は、その時刻の実際の姿勢と比べる (窓の平均ではない)。同じフレームは覚えておく
    i = r.index(t)
    if e.get("_now_i") != i:
        cur = r.xyz[i]
        err = np.linalg.norm(e["xyz"] - cur, axis=1) * 1e3                 # 隠れていたマーカーは NaN
        try:
            ph = e["model"].phase_for_pose(e["names_m"], cur, bundle=e["bundle"])
        except Exception:                                    # noqa: BLE001 (表示だけ)
            ph = None
        e.update(_now_i=i, _now_err=err, _now_ph=ph)
    num = lambda v, d=1: None if v is None or v != v else round(float(v), d)
    rnd = lambda a: [None if np.isnan(p).any() else [round(float(v), 5) for v in p] for p in a]
    return dict(ready=True, inside=True, kind="used", n=len(res), cap=e["cap"], h=e["h"], xyz=rnd(e["xyz"]),
                err_mm=[num(v) for v in e["_now_err"]], phase_pred=num(e["phase_pred"], 2), phase_act=num(e["_now_ph"], 2),
                recorded=e["recorded"], legacy=e["legacy"], phase_diff_rec=num(e["phase_diff_rec"], 2),
                phase_applied=num(e["phase_applied"], 2), lead_ms=num(e["lead_ms"]))


def replay_pred(r: replay.Replay, t: float, h: float) -> dict:
    """時刻 t の「h 秒前に予測した姿勢」と、実際の姿勢との差 (3D 表示に重ねる)。"""
    f = replay_forecast(r, h)
    if not f["ready"]:
        return dict(ready=False, h=h, error=f.get("error", ""))
    res, i = f["res"], r.index(t)
    xp, e = res["xyz_pred"][i], res["err"][i]
    num = lambda v, d=1: None if v != v else round(float(v), d)
    return dict(ready=True, h=h, h_model=res["h_model"], ok=bool(res["ok"][i]),
                xyz=[None if np.isnan(p).any() else [round(float(v), 5) for v in p] for p in xp],
                err_mm=[num(v * 1e3) for v in e], phase_pred=num(res["phase_pred"][i], 2),
                phase_act=num(res["phase_act"][i], 2), rmse_mm=num(res["rmse_mm"]), phase_rmse=num(res["phase_rmse"], 2))


def replay_open(path: str) -> dict:
    global REPLAY
    root = (bg.CAPTURE_ROOT / "log").resolve()     # SessionLog が書く場所と同じ名前を見る
    d = Path(path).resolve()
    if d.parent != root or not (d / "motion.csv").exists():
        raise ValueError(f"再生できるのは {root} の中の記録フォルダだけです: {path}")
    if bg.STATE.mode not in ("idle", "replay"):
        raise RuntimeError("測定・再生中は開けません。先に停止してください")
    r = replay.Replay(d)
    with REPLAY_LOCK:
        if REPLAY is not None:
            REPLAY.close()
        REPLAY = r
        _RAW_CACHE.clear()
    with bg.STATE.lock:
        bg.STATE.reset()
        bg.STATE.mode = "replay"
        # ber_gui.html の poll() は next (= base + 点の数) が 0 だと届いた波形を捨てる。再生では
        # BER の点を足さないので、base を 1 にして「点が 1 つ流れた後」の扱いにする
        bg.STATE.base = 1
    bg.set_status(f"記録の再生: {d.name}", "")
    return dict(ok=True, meta=r.meta())


def replay_close() -> dict:
    global REPLAY
    with REPLAY_LOCK:
        if REPLAY is not None:
            REPLAY.close()
        REPLAY = None
        _RAW_CACHE.clear()
    with bg.STATE.lock:
        if bg.STATE.mode == "replay":
            bg.STATE.mode = "idle"
    bg.set_status("待機中", "")
    return dict(ok=True)


def _raw_material(path: Path) -> dict:
    """.ad3 を、ライブと**同じ手順**で復調できる形にする (2 本まで覚えておく)。

    ライブの live_worker と同じく fast_ddc (C) と decim_for(div) を使う。ber_gui の play_worker の
    手順 (ddc() + 間引き 8 固定) だと、FC/8 では 1 枚もフレームが取れなかった (2026-10-27)。"""
    key = str(path)
    if key not in _RAW_CACHE:
        J = json.loads(Path(str(path) + ".json").read_text())
        div = int(J["frame"]["div"])
        decim = fd.decim_for(div)
        fs_raw = float(J["sample_rate_hz"])
        vpl, off = float(J["volts_per_lsb"]), float(J.get("offset_v", 0.0))
        raw = np.fromfile(path, dtype="<i2")
        x = np.ascontiguousarray(fd.fast_ddc(raw, fs_raw, vpl, off, div=div, decim=decim)[SKIP:])
        while len(_RAW_CACHE) >= 2:
            _RAW_CACHE.pop(next(iter(_RAW_CACHE)))
        _RAW_CACHE[key] = dict(x=x, J=J, fs_raw=fs_raw, rx=fd.receiver_for(J, fs_raw / decim), decim=decim,
                               raw=raw, vpl=vpl, off=off, ddc_delay=(fd.ddc_taps(fs_raw, div, decim).size - 1) / 2)
    return _RAW_CACHE[key]


def _demod_all(M: dict) -> dict:
    """収録を丸ごと、ライブの rx.process() と同じ順 (設定 -> 窓の位置 -> まとめて復調) で復調する。

    1 フレームだけ復調すると、前のフレームから引き継ぐ受信部の状態が無いので誤り数がライブと
    違う (2026-10-27、フレーム 10 がライブ 12 ビット / 単独 5 ビット)。窓の位置も受信部の設定で
    変わるので、設定を変えたときだけやり直す。こうするとフレームごとの誤り数がライブと一致する。"""
    rx = M["rx"]
    opts = (bg.STATE.rx_filter, bg.STATE.agc_div, bg.STATE.agc_auto)
    if M.get("opts") != opts:
        bg.apply_rx_options(rx)
        st, _ = rx.timing(M["x"])
        res = rx.demod(M["x"], st, detail=True)
        fe = np.full(st.size, -1, np.int64)
        fe[res["status"] == 0] = res["frame_errors"]
        M.update(opts=opts, st=st, res=res, fe=fe)
    return M


def _one_frame(res: dict, fe: np.ndarray, k: int) -> dict:
    """まとめて復調した結果から、フレーム k だけの結果を作る (build_view は渡された先頭のフレームを描く)。"""
    n = fe.size
    out = {key: (v[k:k + 1] if isinstance(v, np.ndarray) and v.shape[:1] == (n,) else v)
           for key, v in res.items() if key != "frame_errors"}
    out["frame_errors"] = fe[k:k + 1][fe[k:k + 1] >= 0]
    return out


def _held(r: replay.Replay, t: float) -> tuple[dict | None, float, bool]:
    """時刻 t に出す収録と、そのための時刻。収録と収録の間 (転送待ち) では、直前の収録の最後の
    フレームを出し続ける (次の BER の結果が出るまで前の結果を見せる)。

    返り値: (収録 or None, 引く時刻, 前の収録を出し続けているか)"""
    c = r.cap_at(t)
    if c is not None:
        return c, t, False
    i = int(np.searchsorted(r.cap_t0, t, side="right")) - 1
    if i < 0:
        return None, t, False                            # 最初の収録より前
    c = r.caps[i]
    sel = np.nonzero(r.f_cap == c["cap"])[0]
    te = float(r.f_t[sel[-1]]) + 1e-6 if sel.size else c["t1"]
    return c, min(te, c["t1"]), True


def replay_frame_view(t: float, span_us: float | None = None) -> tuple[dict | None, dict]:
    """時刻 t のテストフレームを .ad3 から復調して、gui_view.build_view の図の材料を返す。

    返り値: (図の材料 or None, 情報)。収録の外・生データの無い収録では図は None。"""
    with REPLAY_LOCK:
        r = REPLAY
        if r is None:
            raise RuntimeError("記録を開いていません")
        c, t, held = _held(r, t)
        if c is None or not c["raw"]:
            return None, dict(ok=True, view=False, cap=None if c is None else c["cap"], held=held)
        M = _demod_all(_raw_material(r.dir / c["raw"]))
        rx, st, fe = M["rx"], M["st"], M["fe"]
        if st.size == 0:
            return None, dict(ok=True, view=False, cap=c["cap"], message="この収録ではフレームが見つかりません")
        # 窓の先頭の時刻 (収録の先頭から) で、t を含むフレームを選ぶ。frames.csv と同じ式
        ts = (st + SKIP) / rx.fs
        k = int(np.clip(np.searchsorted(ts, t - c["t0"], side="right") - 1, 0, st.size - 1))
        res = _one_frame(M["res"], fe, k)
        r0 = max(0, int((st[k] + SKIP) * M["decim"] - M["ddc_delay"]))
        S = bg.STATE
        view = build_view(M["raw"], M["vpl"], M["off"], M["fs_raw"], M["x"], rx, res,
                          span_us or S.span_us, M["ddc_delay"], spec_from=r0, err_focus=S.view_mode == "error",
                          start_bit=S.next_start_bit(advance=False), spec_span_mhz=S.spec_span_mhz)
        return view, dict(ok=True, view=True, cap=c["cap"], frame=k, frames=int(st.size), errors=int(fe[k]),
                          npay=rx.npay, held=held)


def replay_view(t: float) -> dict:
    """時刻 t のテストフレームの図で、元の画面の ①〜③ (STATE.latest) を差し替える。"""
    view, info = replay_frame_view(t)
    if view is not None:
        with bg.STATE.lock:
            bg.STATE.latest.update(view, frame_errors=[info["errors"]], npay=info["npay"])
            bg.STATE.view_id += 1
    info.pop("npay", None)
    info.pop("held", None)
    return info


def replay_numbers(r: replay.Replay, t: float, n_recent: int = 100) -> dict:
    """再生用の画面の ④: その時刻までの数値だけ。"""
    nc, nf = r.ber_upto(t)
    ber = lambda e, b: None if b == 0 else e / b
    i = nf - 1
    tot_b = int(r.cum_bits[i]) if nf else 0
    tot_e = int(r.cum_errors[i]) if nf else 0
    tail = r.f_err[max(0, nf - n_recent):nf]
    got = tail[tail >= 0]
    rec_b, rec_e = int(got.size) * r.npay, int(got.sum())
    c, te, held = _held(r, t)
    return dict(recent=dict(ber=ber(rec_e, rec_b), errors=rec_e, bits=rec_b, frames=int(got.size), expected=int(tail.size)),
                total=dict(ber=ber(tot_e, tot_b), errors=tot_e, bits=tot_b, frames=int(r.cum_frames[i]) if nf else 0,
                           expected=nf),
                cap=None if c is None else dict(cap=c["cap"], ber=None if c["ber"] != c["ber"] else c["ber"],
                                                errors=c["errors"], bits=c["bits"], frames=c["frames"],
                                                expected=c["expected"], level=c["level"], phase=c["phase"]),
                at=r.at(te), npay=r.npay, held=held)


def replay_ber(r: replay.Replay, t: float) -> None:
    """④ 現在の BER・BER の推移・レベルごとの BER を、再生している時刻までの記録で作り直す。

    ber_gui の画面はライブと同じ STATE (points / total / fe_hist / curve) を読むので、そこへ入れる。
    時刻が進むときは点を足すだけ。戻るときは base を進めて、画面に点を全部読み直させる
    (snapshot() は since < base のとき reset を立てて先頭から返す)。"""
    nc, nf = r.ber_upto(t)
    if r.ber_key == (nc, nf):
        return
    with bg.STATE.lock:
        S = bg.STATE
        if nc < len(S.points):
            S.base += len(S.points) + 1
        S.points = list(r.points[:nc])
        i = nf - 1
        S.total = dict(bits=int(r.cum_bits[i]) if nf else 0, errors=int(r.cum_errors[i]) if nf else 0,
                       frames=int(r.cum_frames[i]) if nf else 0, expected=nf)
        S.fe_hist = r.f_err[:nf][-bg.FRAME_HISTORY:].astype(np.int16)
        S.npay = r.npay
        S.curve = {}
        for p in S.points:
            key = f"{p['level'] or 0.0:+.1f}"
            c = S.curve.setdefault(key, dict(level=p["level"] or 0.0, bits=0, errors=0, frames=0, expected=0))
            for k in ("bits", "errors", "frames", "expected"):
                c[k] += p[k]
    r.ber_key = (nc, nf)


def viewer_calibrate(force: bool = False) -> None:
    """開いている記録で、動画にマーカーを重ねるための較正をする (別スレッド)。"""
    r = REPLAY
    if r is None:
        return

    def run():
        VIEWER.update(calib=None, status="較正の準備中")
        try:
            c = overlay.calibrate(r, lambda m: VIEWER.update(status=m), force=force)
            VIEWER.update(calib=c if c.get("ok") else None,
                          status="" if c.get("ok") else c.get("message", "較正できません"))
        except Exception as e:                             # noqa: BLE001 (画面は使える)
            traceback.print_exc()
            VIEWER.update(calib=None, status=f"較正できません: {type(e).__name__}: {e}")

    threading.Thread(target=run, daemon=True).start()


def viewer_overlay(r: replay.Replay, t: float, with_mp: bool) -> dict:
    """時刻 t に出す動画のコマに重ねる点 [px]: マーカー (較正で投影) と、必要なら MediaPipe の骨格。"""
    c = VIEWER["calib"]
    out = dict(w=None, h=None, markers=None, mp=None, status=VIEWER["status"],
               cut=dict(done=len(CUT["masks"]), total=CUT["total"]) if CUT["dir"] == r.dir else None,
               calib=None if c is None else {k: c[k] for k in ("rms_px", "median_px", "lag_s", "focal_px", "lr_swapped")})
    if not r.has_video:
        return out
    k = int(np.clip(np.searchsorted(r.tc, t, side="right") - 1, 0, r.tc.size - 1))
    if c is not None:
        out.update(w=c["width"], h=c["height"])
        tq = r.tc[k] - c["lag_s"]                          # このコマを撮った時刻のモーション
        i = int(np.clip(np.searchsorted(r.tm, tq), 0, r.tm.size - 1))
        if i > 0 and abs(r.tm[i - 1] - tq) < abs(r.tm[i] - tq):
            i -= 1
        P = overlay.project(c, r.xyz[i])
        out["markers"] = {n: (None if np.isnan(P[j]).any() else [round(float(v), 1) for v in P[j]])
                          for j, n in enumerate(r.names)}
    if with_mp:
        out["mp"] = viewer_mediapipe(r, k)
    return out


_MP = dict(det=None, cache={})

# ---------------------------------------------------------------- 人物の切り出し (背景を薄く)
# 記録を開くと、動画の全コマの人物のマスクを裏で作り (1 コマ 約 50 ms)、記録フォルダの person_mask.npz に
# 置く (次からは読むだけ)。まだ作っていないコマは、頼まれたときにその場で作る。マスクは幅 320 px の PNG。
CUT = dict(dir=None, masks={}, tmp={}, total=0, stop=threading.Event(), thread=None, det=None, lock=threading.Lock(),
           last=(None, None))
CUT_FILE = "person_mask.npz"
CUT_WIDTH = 320
CUT_VERSION = 2          # 2: 動画モードで追跡 + 手の帯 + 抜けたコマを隣で埋める (1 は画像モードで、207/3592 コマ人ごと消えた)
CUT_FILL = 10            # 人が消えたコマは、前後この数のコマまでで見つかったいちばん近いマスクで埋める


def _cut_encode(mask) -> bytes:
    h, w = mask.shape
    small = cv2_resize(mask, CUT_WIDTH, int(h * CUT_WIDTH / w))
    return overlay.cv2.imencode(".png", small)[1].tobytes()


def cv2_resize(img, w, h):
    return overlay.cv2.resize(img, (w, h), interpolation=overlay.cv2.INTER_AREA)


def cut_start(r: replay.Replay) -> None:
    """開いた記録の人物のマスクを、保存したものがあれば読み、無ければ裏で作り始める。"""
    CUT["stop"].set()
    if CUT["thread"] is not None:
        CUT["thread"].join(timeout=5)
    stop = threading.Event()
    CUT.update(dir=r.dir, masks={}, tmp={}, total=int(r.tc.size) if r.has_video else 0, stop=stop, thread=None,
               last=(None, None))
    if not r.has_video:
        return
    f = r.dir / CUT_FILE
    if f.exists():
        try:
            z = np.load(f)
            ver = int(z["version"]) if "version" in z.files else 1
            if ver == CUT_VERSION:                         # 古いやり方で作ったものは作り直す
                blob, off = z["blob"].tobytes(), z["offsets"]
                CUT["masks"] = {int(k): blob[off[i]:off[i + 1]] for i, k in enumerate(z["frames"])}
                if len(CUT["masks"]) >= CUT["total"]:
                    return
        except Exception:                                  # noqa: BLE001 (壊れていたら作り直す)
            traceback.print_exc()
            CUT["masks"] = {}

    def run():
        try:
            # 動画モード: 前のコマの結果を手がかりにするので、頭から順に全部通す (途中から始めない)
            seg = overlay.PersonSegmenter(video=True)
            cap = overlay.cv2.VideoCapture(str(r.video_path))
            empty = []
            for k in range(CUT["total"]):
                if stop.is_set():
                    return
                ok, frame = cap.read()
                if not ok:
                    break
                m = seg.mask(frame, int(round(r.tc[k] * 1000)))
                if not m.any():
                    empty.append(k)
                CUT["masks"][k] = _cut_encode(m)
                CUT["tmp"].pop(k, None)
            cap.release()
            for k in empty:                                # それでも人が消えたコマは、いちばん近いコマのマスクで埋める
                for dk in range(1, CUT_FILL + 1):
                    src = next((j for j in (k - dk, k + dk) if j in CUT["masks"] and j not in empty), None)
                    if src is not None:
                        CUT["masks"][k] = CUT["masks"][src]
                        break
            if not stop.is_set() and CUT["dir"] == r.dir:
                ks = sorted(CUT["masks"])
                blobs = [CUT["masks"][k] for k in ks]
                off = np.concatenate([[0], np.cumsum([len(b) for b in blobs])]).astype(np.int64)
                np.savez(r.dir / CUT_FILE, frames=np.array(ks), offsets=off, version=np.array(CUT_VERSION),
                         blob=np.frombuffer(b"".join(blobs), np.uint8), empty_filled=np.array(empty, np.int64))
        except Exception:                                  # noqa: BLE001 (切り出し無しでも再生はできる)
            traceback.print_exc()

    CUT["thread"] = threading.Thread(target=run, daemon=True)
    CUT["thread"].start()


def cut_jpeg(r: replay.Replay, t: float) -> bytes | None:
    """時刻 t に出すコマで、人物はそのまま・背景は薄くした JPEG (幅 640 px)。"""
    if not r.has_video:
        return None
    k = int(np.clip(np.searchsorted(r.tc, t, side="right") - 1, 0, r.tc.size - 1))
    if CUT["last"][0] == (str(r.dir), k):
        return CUT["last"][1]
    frame = r.frame_bgr(k)
    if frame is None:
        return None
    png = CUT["masks"].get(k) or CUT["tmp"].get(k)
    if png is None:                                        # 裏の処理がまだ来ていないコマ: その場で作る (仮。裏が来たら置き換わる)
        with CUT["lock"]:
            if CUT["det"] is None:                         # 1 コマだけなので画像モード。しきい値を下げて見失いにくく
                CUT["det"] = overlay.PersonSegmenter(video=False, confidence=0.2)
            png = CUT["tmp"][k] = _cut_encode(CUT["det"].mask(frame))
    mask = overlay.cv2.imdecode(np.frombuffer(png, np.uint8), overlay.cv2.IMREAD_GRAYSCALE)
    out = overlay.fade_background(frame, mask)
    h, w = out.shape[:2]
    if w > replay.STREAM_WIDTH:
        out = cv2_resize(out, replay.STREAM_WIDTH, int(h * replay.STREAM_WIDTH / w))
    jpg = overlay.cv2.imencode(".jpg", out, [overlay.cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes()
    CUT["last"] = ((str(r.dir), k), jpg)
    return jpg


def viewer_mediapipe(r: replay.Replay, k: int):
    """動画のコマ k の MediaPipe の骨格 (見えている点だけ)。コマごとに覚えておく。"""
    key = (str(r.dir), k)
    if key not in _MP["cache"]:
        if _MP["det"] is None:
            _MP["det"] = overlay.PoseDetector()
        frame = r.frame_bgr(k)
        lm = None if frame is None else _MP["det"].detect(frame)
        _MP["cache"][key] = None if lm is None else [None if p[2] < overlay.VIS_MIN else [round(float(p[0]), 1),
                                                                                          round(float(p[1]), 1)]
                                                      for p in lm]
        if len(_MP["cache"]) > 3000:
            _MP["cache"].pop(next(iter(_MP["cache"])))
    return dict(points=_MP["cache"][key], edges=overlay.MP_EDGES)


def page() -> bytes:
    html = (BER_DIR / "instruments" / "ber_gui.html").read_text(encoding="utf-8")
    for anchor in ('<div class="cols">', "</body>"):
        if anchor not in html:
            raise RuntimeError(f"ber_gui.html に {anchor!r} がありません。motion_ber_gui.py の差し込み位置を直してください")
    html = html.replace("<title>VHBR BER モニタ</title>", "<title>VHBR BER × モーション</title>", 1)
    html = html.replace('<div class="cols">', PANEL + '  <div class="cols">', 1)
    html = html.replace("</body>", PANEL_JS + "</body>", 1)
    # fit() は「ページの先頭から画面の高さ」に図を収める。上にモーションの欄を足したので、
    # そのままだと ① の図が最小 (0.5 倍) まで縮む。図の欄 (.cols) の先頭から 1 画面ぶんに収める
    fit_old = "box.getBoundingClientRect().bottom + window.scrollY - (window.innerHeight - 10)"
    if fit_old not in html:
        raise RuntimeError("ber_gui.html の fit() が変わっています。motion_ber_gui.py の page() を直してください")
    html = html.replace(fit_old, "box.getBoundingClientRect().bottom - document.querySelector('.cols')"
                                 ".getBoundingClientRect().top - (window.innerHeight - 10)", 1)
    # 上の状態表示に「記録の再生」を足す (元は待機 / ライブ / 再生 だけ)
    html = html.replace('{ idle: "待機", live: "ライブ", play: "再生" }',
                        '{ idle: "待機", live: "ライブ", play: "再生", replay: "記録の再生" }', 1)
    # 「1 回の収録」に 0.095 秒 (位相の 5 ms と合わせて 100 ms) を足す。既定はこれ (main で STATE.live に入れる)
    secs_old = '<option value="0.05">0.05 秒</option>'
    if secs_old not in html:
        raise RuntimeError("ber_gui.html の「1 回の収録」の選択肢が変わっています。motion_ber_gui.py の page() を直してください")
    s = phase_control.CAPTURE_SECONDS
    html = html.replace(secs_old, secs_old + f'<option value="{s:g}">{s:g} 秒（+ 位相 5 ms）</option>', 1)
    return html.encode("utf-8")


def corr_recent() -> list[dict]:
    with CORR_LOCK:
        return list(CORR)[-CORR_SEND:]


def motion_status() -> dict:
    s = bg.SESSION
    logging = isinstance(s, MotionSessionLog) and MOTION.sink is not None
    return dict(MOTION.status(), require_motion=OPTS["require_motion"], auto_phase=OPTS["auto_phase"],
                horizon=OPTS["horizon"], swap_lr=OPTS["swap_lr"], offset_test=OPTS["offset_test"], gain=OPTS["gain"],
                test_offsets=list(phase_control.TEST_OFFSETS), model=PHASE_MODEL_NAME, logging=logging, log_dir=s.dir.name if logging else "", last=LAST,
                corr=corr_recent(), ref_warmup=phase_control.REF_WARMUP,
                camera=CAMERA.status(), record_video=OPTS["record_video"],
                video_recording=logging and getattr(s, "video", None) is not None,
                replay=None if REPLAY is None else REPLAY.dir.name)


class Handler(bg.Handler):
    server_version = "motion_ber_gui"

    def do_GET(self):
        u = urlparse(self.path)
        if u.path.startswith("/assets/"):                  # 再生用の画面の図 (回路図など)。assets/ の中だけ
            f = HERE / "assets" / Path(u.path).name
            if f.suffix.lower() == ".png" and f.is_file():
                return self._send(f.read_bytes(), "image/png")
            return self._send(dict(error="not found"), code=404)
        if u.path == "/":
            if VIEWER["on"]:
                return self._send((HERE / "replay_viewer.html").read_bytes(), "text/html")
            return self._send(page(), "text/html")
        if u.path == "/api/motion":
            return self._send(motion_status())
        if u.path == "/api/motion/pose":
            out = MOTION.pose()
            if "pred" in parse_qs(u.query):               # 3D 表示の「予測を重ねる」: 直近の収録に使った予測
                out = dict(out, pred=live_pred_view())
            return self._send(out)
        if u.path == "/api/camera.mjpg":
            return self._mjpeg()
        if u.path.startswith("/api/replay/"):
            return self._replay_get(u)
        return super().do_GET()

    def _replay_get(self, u):
        q = parse_qs(u.query)
        if u.path == "/api/replay/sessions":
            return self._send(replay.list_sessions(bg.CAPTURE_ROOT / "log"))
        r = REPLAY
        if u.path == "/api/replay/meta":
            return self._send(None if r is None else r.meta())
        if r is None:
            return self._send(dict(ok=False, message="記録を開いていません"), code=409)
        t = float(q.get("t", ["0"])[0])
        if u.path == "/api/replay/pose":
            replay_ber(r, t)                               # ④ も再生している時刻に合わせる
            out = dict(r.pose(t), at=r.at(t))
            if "pred" in q:                                # 3D 表示の「予測を重ねる」: h 秒前に予測した姿勢
                out["pred"] = replay_pred(r, t, float(q["pred"][0]))
            if "used" in q:                                # 同: その収録の補正に実際に使った予測 (収録窓の中だけ)
                out["used"] = replay_used(r, t)
            return self._send(out)
        if u.path == "/api/replay/panel":
            span = float(q["span"][0]) if "span" in q else None
            view, info = replay_frame_view(t, span)
            return self._send(dict(info, view=None if view is None else
                                   {k: view[k] for k in ("wave", "stages", "const") if k in view},
                                   numbers=replay_numbers(r, t, int(q.get("n", ["100"])[0]))))
        if u.path == "/api/replay/overlay":
            return self._send(viewer_overlay(r, t, q.get("mp", ["0"])[0] == "1"))
        if u.path == "/api/replay/frame.jpg":
            jpg = cut_jpeg(r, t) if VIEWER["on"] and q.get("cut", ["0"])[0] == "1" else r.jpeg(t)
            return self._send(jpg, "image/jpeg") if jpg else self._send(b"", "image/jpeg", code=404)
        return self._send(dict(error="not found"), code=404)

    def _mjpeg(self):
        """カメラを MJPEG で流し続ける (<img src> でそのまま見える)。カメラを閉じるか、画面を閉じるまで。"""
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        seq = -1
        try:
            while CAMERA.running:
                jpg, seq = CAMERA.jpeg(seq)
                if jpg is None:
                    continue
                self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                 + str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n")
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass

    def do_POST(self):
        u = urlparse(self.path)
        n = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(n) if n else b""
        self.rfile = io.BytesIO(body)                      # 元の do_POST がもう一度読めるように
        try:
            if u.path.startswith(("/api/motion/", "/api/camera/")):
                return self._send(self._motion(u.path, json.loads(body or b"{}")))
            if u.path == "/api/replay/open":
                out = replay_open(str(json.loads(body or b"{}").get("path", "")))
                if VIEWER["on"]:
                    viewer_calibrate()
                    cut_start(REPLAY)
                return self._send(out)
            if u.path == "/api/replay/calibrate":
                viewer_calibrate(force=True)
                return self._send(dict(ok=True))
            if u.path == "/api/replay/close":
                return self._send(replay_close())
            if u.path == "/api/replay/view":
                return self._send(replay_view(float(json.loads(body or b"{}").get("t", 0.0))))
            if u.path in ("/api/live/start", "/api/play/start", "/api/power") and VIEWER["on"]:
                return self._send(dict(ok=False, message="再生用の画面 (--replay) では装置を使いません"))
            if u.path in ("/api/live/start", "/api/play/start") and REPLAY is not None:
                return self._send(dict(ok=False, message="記録の再生を閉じてから始めてください"))
            if u.path == "/api/live/start" and OPTS["require_motion"] and not MOTION.receiving():
                return self._send(dict(ok=False, message="OptiTrack のフレームが来ていません。モーションの欄で接続するか、"
                                                         "「OptiTrack なしでも開始できる」を入れてください"))
            if u.path == "/api/shutdown":
                replay_close()
                MOTION.close()
                CAMERA.close()
        except Exception as e:                             # noqa: BLE001
            traceback.print_exc()
            return self._send(dict(ok=False, message=f"{type(e).__name__}: {e}"))
        return super().do_POST()

    @staticmethod
    def _motion(path: str, q: dict) -> dict:
        if path == "/api/motion/connect":
            if bg.STATE.mode == "live":
                raise RuntimeError("BER の測定中は繋ぎ直せません (記録の途中でモーションが途切れるため)。先に停止してください")
            if "group" in q:
                MOTION.set_group(str(q["group"]))
            if q.get("source") == "fake":
                MOTION.start_fake()
            else:
                MOTION.connect(str(q.get("server_ip", "127.0.0.1")), str(q.get("local_ip", "127.0.0.1")),
                               bool(q.get("multicast", False)))
            return dict(ok=True)
        if path == "/api/motion/close":
            if bg.STATE.mode == "live":
                raise RuntimeError("BER の測定中は切断できません。先に停止してください")
            MOTION.close()
            return dict(ok=True)
        if path == "/api/motion/options":
            if "group" in q:
                MOTION.set_group(str(q["group"]))
            if "require_motion" in q:
                OPTS["require_motion"] = bool(q["require_motion"])
            if "auto_phase" in q:
                if q["auto_phase"] and PHASE_MODEL is None:
                    raise RuntimeError("位相のモデルがありません (--phase-model で渡す)")
                OPTS["auto_phase"] = bool(q["auto_phase"])
                reset_phase_ref()
                MOTION.error = ""
            if "horizon" in q:
                h = phase_control.parse_horizon(q["horizon"])     # HORIZONS のどれか
                if h != OPTS["horizon"]:
                    OPTS["horizon"] = h                            # 次の収録の予測から効く
                    reset_phase_ref()                              # モデルが変わるので、いまの位相を初期位相に取り直す
            if "swap_lr" in q and bool(q["swap_lr"]) != OPTS["swap_lr"]:
                OPTS["swap_lr"] = bool(q["swap_lr"])
                reset_phase_ref()                                  # 推定が変わるので、いまの位相を初期位相に取り直す
            if "gain" in q:
                OPTS["gain"] = phase_control.parse_gain(q["gain"])  # 次の収録から。基準は取り直さない (基準からの変化に掛ける)
            if "offset_test" in q:
                OPTS["offset_test"] = bool(q["offset_test"])       # 次の収録から。基準は取り直さない (足すのは基準の後)
                _TEST_SEQ.clear()
            if "record_video" in q:
                OPTS["record_video"] = bool(q["record_video"])     # 次の記録から効く
            return dict(ok=True)
        if path in ("/api/camera/open", "/api/camera/close"):
            if bg.STATE.mode == "live" and CAMERA.running:
                raise RuntimeError("BER の測定中はカメラを繋ぎ直せません (録画の途中で切れるため)。先に停止してください")
            if path.endswith("open"):
                CAMERA.open(int(q.get("index", 0)))
            else:
                CAMERA.close()
            return dict(ok=True)
        raise ValueError(f"not found: {path}")


# --------------------------------------------------------------------------- main
class Server(ThreadingHTTPServer):
    # http.server の既定 (True) だと、Windows では同じポートに 2 つ目の GUI が黙って bind でき、
    # 要求が 2 つのプロセスに振り分けられる (2026-10-27、測定の指示が別の GUI に届いた)。
    allow_reuse_address = False


def main() -> int:
    global PHASE_MODEL, PHASE_MODEL_NAME
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8766)
    ap.add_argument("--reader", default=bench.READER_SN)
    ap.add_argument("--tag", default=bench.TAG_SN)
    ap.add_argument("--payload", type=int, default=600)
    ap.add_argument("--div", type=int, choices=(16, 8, 7, 6, 5, 4, 3, 2), default=8,
                    help="起動時のデータレート fc/div (既定 8)")
    ap.add_argument("--group", choices=tuple(motion.MARKER_GROUPS), default="upper", help="計測部位 (既定 upper)")
    ap.add_argument("--natnet-server", default="127.0.0.1", help="起動時に繋ぐ Motive の IP")
    ap.add_argument("--natnet-local", default="127.0.0.1", help="この PC の IP")
    ap.add_argument("--multicast", action="store_true")
    ap.add_argument("--connect", action="store_true", help="起動時に OptiTrack へ繋ぐ")
    ap.add_argument("--fake-motion", action="store_true", help="起動時に合成モーションを流す (Motive 無しの試験用)")
    ap.add_argument("--camera", type=int, metavar="N", help="起動時にカメラ N を開く (0 = 内蔵のことが多い)")
    ap.add_argument("--phase-model", help="predict_phase(motion) を持つ .py (phase_control.py 参照)。"
                    "モーションからインピーダンスを先読みして決めるなら imp_phase_model.py")
    ap.add_argument("--horizon", type=phase_control.parse_horizon, default=phase_control.HORIZONS[0],
                    help=f"位相のモデルに渡す固定の先読み [s]: {' / '.join(f'{h:g}' for h in phase_control.HORIZONS)}"
                         f" (既定 {phase_control.HORIZONS[0]:g} = 収録窓 95 ms の中央の見込み。画面でも変えられる)")
    ap.add_argument("--swap-lr", action="store_true",
                    help="Motive のラベルが左右逆。位相のモデルに渡すマーカー名だけ入れ替える (画面でも変えられる)")
    ap.add_argument("--offset-test", action="store_true",
                    help="補正量の試験: 自動補正に phase_control.TEST_OFFSETS を収録ごとに足す (画面でも変えられる)")
    ap.add_argument("--gain", type=phase_control.parse_gain, default=phase_control.DEFAULT_GAIN,
                    help="補正のゲイン: 送る位相 = 初期位相 + gain × モデルの補正量 (既定 1。画面でも変えられる)")
    ap.add_argument("--replay", nargs="?", const="latest", metavar="FOLDER",
                    help="記録の再生用の画面で開く (装置には触らない)。FOLDER を省くといちばん新しい記録")
    ap.add_argument("--video-only", action="store_true",
                    help="--replay の画面を動画のみ (①③④ を隠す) で開く。画面の「動画のみ」/ キー V でも切り替えられる")
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args()
    if a.video_only and not a.replay:
        a.replay = "latest"
    if a.replay:
        return main_viewer(a)

    bg.DEFAULT_DIV = a.div
    bg.STATE.live["div"] = a.div
    bg.STATE.live["seconds"] = phase_control.CAPTURE_SECONDS     # 収録窓 95 ms + 位相の 5 ms (画面で変えられる)
    MOTION.set_group(a.group)
    OPTS["horizon"] = a.horizon
    OPTS["swap_lr"] = a.swap_lr
    OPTS["offset_test"] = a.offset_test
    OPTS["gain"] = a.gain
    if a.phase_model:
        PHASE_MODEL = phase_control.load(a.phase_model)
        PHASE_MODEL_NAME = Path(a.phase_model).name
    if a.fake_motion:
        MOTION.start_fake()
    elif a.connect:
        try:
            MOTION.connect(a.natnet_server, a.natnet_local, a.multicast)
        except Exception as e:                             # noqa: BLE001 (画面から繋ぎ直せる)
            print(f"OptiTrack に繋げません: {e}", flush=True)
            MOTION.error = str(e)
    if a.camera is not None:
        try:
            CAMERA.open(a.camera)
        except Exception as e:                             # noqa: BLE001 (画面から繋ぎ直せる)
            print(f"カメラを開けません: {e}", flush=True)
            CAMERA.error = str(e)
    fd.load_kernel()
    Handler.opts = a
    try:
        srv = Server(("127.0.0.1", a.port), Handler)
    except OSError as e:
        print(f"ポート {a.port} を開けません。もう 1 つ GUI が動いていないか確かめてください "
              f"(画面の終了ボタンで止める。強制終了はしない): {e}", flush=True)
        MOTION.close()
        CAMERA.close()
        return 1
    url = f"http://127.0.0.1:{a.port}/"
    print(f"BER × モーション GUI: {url}   (Ctrl-C で終了。ライブ中なら出力を止めてから終わる)", flush=True)
    if not a.no_browser:
        threading.Timer(0.5, webbrowser.open, args=(url,)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        bg.stop()
        MOTION.close()
        CAMERA.close()
    return 0


def main_viewer(a) -> int:
    """--replay: 記録の再生専用の画面。ADP3250・電源・OptiTrack・カメラには触らない。

    左に動画とマーカーの骨格 (MediaPipe で較正して重ねる)、右に ① と ③・④ を出す。"""
    VIEWER["on"] = True
    bg.power_read = lambda max_age=1.0: dict(ok=False, on=None, rails=[], message="再生用の画面では電源を見ない")
    root = bg.CAPTURE_ROOT / "log"
    if a.replay == "latest":
        L = replay.list_sessions(root)
        if not L:
            print(f"記録がありません: {root}", flush=True)
            return 1
        folder = L[0]["path"]
    else:
        folder = a.replay if Path(a.replay).is_absolute() else str(root / a.replay)
    fd.load_kernel()
    replay_open(folder)
    viewer_calibrate()
    cut_start(REPLAY)

    def warm():                                          # 受信部の立ち上げ (初回 1 秒強) を先に済ませる
        try:
            if REPLAY is not None and REPLAY.caps:
                replay_frame_view(REPLAY.caps[0]["t0"])
        except Exception:                                # noqa: BLE001
            traceback.print_exc()

    threading.Thread(target=warm, daemon=True).start()
    Handler.opts = a
    port = a.port if a.port != 8766 else 8767           # 測定用の GUI と同時に開けるように
    try:
        srv = Server(("127.0.0.1", port), Handler)
    except OSError as e:
        print(f"ポート {port} を開けません (再生用の画面がもう開いていませんか): {e}", flush=True)
        return 1
    url = f"http://127.0.0.1:{port}/" + ("?mode=video" if a.video_only else "")
    print(f"記録の再生: {Path(folder).name}\n  {url}   (Ctrl-C で終了)", flush=True)
    if not a.no_browser:
        threading.Timer(0.5, webbrowser.open, args=(url,)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        replay_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
