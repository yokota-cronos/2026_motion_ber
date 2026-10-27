#!/usr/bin/env python3
"""motion_ber_gui.py -- OptiTrack のモーションキャプチャと ADP3250 の BER 測定を同時に回す GUI。

    uv run motion_ber_gui.py                 # http://127.0.0.1:8766 が開く
    uv run motion_ber_gui.py --fake-motion   # Motive 無し (合成モーション) で試す
    uv run motion_ber_gui.py --camera 0      # カメラも開く (画面の「カメラ」欄からも開ける)
    uv run motion_ber_gui.py --phase-model phase_model_example.py   # 位相の自動調整 (先々用)

BER 側は ../2026_nfc_vhbr_ber/instruments/ber_gui.py を**そのまま import して**使い、次の 4 か所だけ
差し替える (元のファイルは書き換えない。BER 側の改良はそのままこちらにも効く):

  bench.Bridge.record          収録を始めた時刻 (perf_counter) を控える
  fast_demod.FastReceiver.process  直近の復調結果 (各フレームの位置と誤り数) を控える
  ber_gui.SessionLog           記録フォルダに motion.csv / ber_motion.csv / frames.csv を足す
  ber_gui.Handler / live_worker  画面にモーションの欄を足す、終了時に記録を閉じる

ber_gui.py の中の名前 (live_worker の中の SessionLog、b.record()、rx.process()) に依存しているので、
**あちらでそれらの名前や呼び方を変えたら、ここも直すこと。**

同期のしかた: ADP3250 も Motive も外部トリガで結ばれていないので、どちらも PC の
time.perf_counter() で打った時刻で突き合わせる。
  収録窓   = [Configure を呼ぶ直前, + 点数 / fs]  (USB で開始を伝える遅れ ~1 ms は未較正)
  フレーム = 収録窓の先頭 + (starts + SKIP) / 受信部の fs   (DDC の遅れ < 1 µs は無視)
  モーション = NatNet のフレームを受けた時刻 (Motive の処理・配信の遅れ数 ms は未較正)
"""
from __future__ import annotations

import argparse
import io
import json
import os
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
from gui_view import SKIP  # noqa: E402

import camera  # noqa: E402
import motion  # noqa: E402
import replay  # noqa: E402
import phase_control  # noqa: E402

MOTION = motion.MotionStream()
REPLAY: replay.Replay | None = None      # 開いている記録 (記録の再生)
REPLAY_LOCK = threading.Lock()
_RAW_CACHE: dict = {}                     # .ad3 のパス -> 復調の材料 (直近 2 本)
CAMERA = camera.Camera()
CAPTURE: dict = {}            # 直近の収録: t0 [perf], n, fs, rx_fs, res。ワーカースレッドだけが書く
OPTS = dict(require_motion=True, auto_phase=False, record_video=True)
PHASE_MODEL = None
PHASE_MODEL_NAME = ""
LAST: dict = {}               # 画面に出す直近の収録窓のモーション


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
        self.perf0, self.wall0 = time.perf_counter(), time.time()
        self.names = MOTION.names
        self.n_cap = 0
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

    def row(self, p: dict, raw_file: str = ""):
        super().row(p, raw_file)
        try:
            self._joined_row(p, raw_file)
        except Exception as e:                               # noqa: BLE001 (BER の測定は止めない)
            traceback.print_exc()
            MOTION.error = f"突き合わせの記録に失敗: {type(e).__name__}: {e}"

    def _joined_row(self, p: dict, raw_file: str = "") -> None:
        c = dict(CAPTURE)
        r = c.get("res")
        if r is None or "t0" not in c:
            return
        self.n_cap += 1
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
        auto = OPTS["auto_phase"] and LAST.get("auto_phase") == p.get("phase")
        ber = p["errors"] / p["bits"] if p["bits"] else float("nan")
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
        self._auto_phase(mean, valid, (t0 + t1) / 2 - self.perf0)

    def _auto_phase(self, mean, valid, t_mid) -> None:
        """位相の自動調整 (先々用)。収録 1 回に 1 回、GUI で位相を入れたのと同じ経路で反映する。"""
        if PHASE_MODEL is None or not OPTS["auto_phase"]:
            return
        live = dict(bg.STATE.live)
        m = dict(names=self.names, xyz=mean, valid=valid, t=t_mid, phase_deg=live["phase_deg"],
                 level_dbm=live["level_dbm"], div=live["div"])
        try:
            v = phase_control.clamp(PHASE_MODEL.predict_phase(m))
        except Exception as e:                               # noqa: BLE001
            OPTS["auto_phase"] = False
            MOTION.error = f"位相モデルが失敗したので自動調整を切りました: {type(e).__name__}: {e}"
            return
        if v is None or abs(v - live["phase_deg"]) < 0.01:
            return
        with bg.STATE.lock:
            bg.STATE.live["phase_deg"] = v
            bg.COMMANDS.append(dict(phase_deg=v))
        LAST["auto_phase"] = v

    def close(self):
        with MOTION.lock:
            if MOTION.sink == self._motion_row:
                MOTION.sink = None
        if self.video is not None:
            self.video.close()
        with self.mlock:
            for f in (self.mcsv, self.bcsv, self.fcsv):
                f.close()
        super().close()


bg.SessionLog = MotionSessionLog
_orig_live_worker = bg.live_worker


def live_worker(*args):
    try:
        _orig_live_worker(*args)
    finally:
        s = bg.SESSION
        if isinstance(s, MotionSessionLog):
            s.close()


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
      <label class="chk" id="mAutoL" title="--phase-model で渡したモデルが、収録ごとに位相差を決める"><input id="mAuto" type="checkbox">位相の自動調整</label>
    </div>
    <div style="flex-basis:100%;display:flex;flex-wrap:wrap;gap:4px 18px;align-items:center;font-size:13px">
      <span><span id="mDot" style="width:10px;height:10px;border-radius:50%;background:#888;display:inline-block;margin-right:6px"></span><b id="mText">未接続</b></span>
      <span id="mMarkers" style="display:flex;flex-wrap:wrap;gap:4px 12px;color:var(--sub)"></span>
      <span id="mLast" style="color:var(--sub)"></span>
      <span id="mErr" style="color:var(--red)"></span>
      <details style="font-size:12px;color:var(--sub)"><summary>受信フレームの構造（marker_id と名前の対応を確かめる）</summary><pre id="mDiag" style="margin:4px 0;white-space:pre-wrap"></pre></details>
    </div>
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
      <h2>マーカーの 3D 表示<small>ドラッグで回転、ホイールで拡大縮小、ダブルクリックで戻す</small><label class="chk" style="float:right;font-weight:400" title="Motive の設定 (Up Axis) に合わせる。この台では Z が上">上の軸<select id="mUp"><option value="z" selected>Z</option><option value="y">Y</option></select></label></h2>
      <canvas id="m3d" data-h="360" data-fixed="1" style="height:360px;cursor:grab;touch-action:none"></canvas>
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
      const src = { natnet: "OptiTrack", fake: "合成", "": "" }[s.source];
      val("mDot").style.background = s.receiving ? "#2ca02c" : (s.source ? "#f28e2b" : "#888");
      val("mText").textContent = !s.source ? "未接続" : s.receiving
        ? `${src} 受信中 ${s.fps} fps（フレーム #${s.frame}、${s.age_ms} ms 前）${s.logging ? "・記録中" : ""}`
        : `${src} 待機中（フレームが来ていません）`;
      val("mMarkers").innerHTML = s.markers.map(m =>
        `<span title="${m.pos ? m.pos.join(", ") + " m" : "見えていない"}"><span style="color:${m.pos ? "#2ca02c" : "#d62728"}">●</span> ${m.name}</span>`).join("");
      const L = s.last;
      val("mLast").textContent = L && L.cap ? `直近の収録 #${L.cap}: モーション ${L.motion_frames} フレーム、全点見えていた ${(L.valid_frac * 100).toFixed(0)}%` +
        (L.auto_phase != null ? `、自動位相 ${L.auto_phase}°` : "") : "";
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

  // ------------------------------------------------------------ 3D 表示
  // 描くときは Y を上にする。Motive が Z 上なら (x, y, z) -> (x, z, -y) に回してから描く (右手系のまま)
  const V = { yaw: -0.6, pitch: 0.3, zoom: 1, c: null, r: null };
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
  cv.addEventListener("dblclick", () => { V.yaw = -0.6; V.pitch = 0.3; V.zoom = 1; V.c = V.r = null; });
  val("mUp").onchange = () => { V.c = V.r = null; };

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
  }

  async function ploop() {
    try {
      pose = await (await fetch(R.open ? `/api/replay/pose?t=${R.t}` : "/api/motion/pose")).json();
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
    if (meta) {
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


def replay_view(t: float) -> dict:
    """時刻 t のテストフレームを .ad3 から復調して、①〜③ の図 (STATE.latest) を差し替える。"""
    with REPLAY_LOCK:
        r = REPLAY
        if r is None:
            raise RuntimeError("記録を開いていません")
        c = r.cap_at(t)
        if c is None or not c["raw"]:
            return dict(ok=True, view=False, cap=None if c is None else c["cap"])
        M = _demod_all(_raw_material(r.dir / c["raw"]))
        rx, st, fe = M["rx"], M["st"], M["fe"]
        if st.size == 0:
            return dict(ok=True, view=False, cap=c["cap"], message="この収録ではフレームが見つかりません")
        # 窓の先頭の時刻 (収録の先頭から) で、t を含むフレームを選ぶ。frames.csv と同じ式
        ts = (st + SKIP) / rx.fs
        k = int(np.clip(np.searchsorted(ts, t - c["t0"], side="right") - 1, 0, st.size - 1))
        res = _one_frame(M["res"], fe, k)
        r0 = max(0, int((st[k] + SKIP) * M["decim"] - M["ddc_delay"]))
        view = bg.make_view(M["raw"], M["vpl"], M["off"], M["fs_raw"], M["x"], rx, res, M["ddc_delay"],
                            spec_from=r0)
        with bg.STATE.lock:
            bg.STATE.latest.update(view, frame_errors=[int(fe[k])], npay=rx.npay)
            bg.STATE.view_id += 1
        return dict(ok=True, view=True, cap=c["cap"], frame=k, frames=int(st.size), errors=int(fe[k]))


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
    return html.encode("utf-8")


def motion_status() -> dict:
    s = bg.SESSION
    logging = isinstance(s, MotionSessionLog) and MOTION.sink is not None
    return dict(MOTION.status(), require_motion=OPTS["require_motion"], auto_phase=OPTS["auto_phase"],
                model=PHASE_MODEL_NAME, logging=logging, log_dir=s.dir.name if logging else "", last=LAST,
                camera=CAMERA.status(), record_video=OPTS["record_video"],
                video_recording=logging and getattr(s, "video", None) is not None,
                replay=None if REPLAY is None else REPLAY.dir.name)


class Handler(bg.Handler):
    server_version = "motion_ber_gui"

    def do_GET(self):
        u = urlparse(self.path)
        if u.path == "/":
            return self._send(page(), "text/html")
        if u.path == "/api/motion":
            return self._send(motion_status())
        if u.path == "/api/motion/pose":
            return self._send(MOTION.pose())
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
            return self._send(dict(r.pose(t), at=r.at(t)))
        if u.path == "/api/replay/frame.jpg":
            jpg = r.jpeg(t)
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
                return self._send(replay_open(str(json.loads(body or b"{}").get("path", ""))))
            if u.path == "/api/replay/close":
                return self._send(replay_close())
            if u.path == "/api/replay/view":
                return self._send(replay_view(float(json.loads(body or b"{}").get("t", 0.0))))
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
                MOTION.error = ""
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
    ap.add_argument("--phase-model", help="predict_phase(motion) を持つ .py (phase_control.py 参照)")
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args()

    bg.DEFAULT_DIV = a.div
    bg.STATE.live["div"] = a.div
    MOTION.set_group(a.group)
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


if __name__ == "__main__":
    sys.exit(main())
