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
from urllib.parse import urlparse

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
import phase_control  # noqa: E402

MOTION = motion.MotionStream()
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
             "opti_frame_last"] + xyz + [f"{n}_dev" for n in self.names]) + "\n")
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
            self._joined_row(p)
        except Exception as e:                               # noqa: BLE001 (BER の測定は止めない)
            traceback.print_exc()
            MOTION.error = f"突き合わせの記録に失敗: {type(e).__name__}: {e}"

    def _joined_row(self, p: dict) -> None:
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
  <div class="pair" id="mViews">
    <section class="card">
      <h2>マーカーの 3D 表示<small>ドラッグで回転、ホイールで拡大縮小、ダブルクリックで戻す</small><label class="chk" style="float:right;font-weight:400" title="Motive の設定 (Up Axis) に合わせる。この台では Z が上">上の軸<select id="mUp"><option value="z" selected>Z</option><option value="y">Y</option></select></label></h2>
      <canvas id="m3d" style="height:360px;cursor:grab;touch-action:none"></canvas>
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
  const val = id => document.getElementById(id);
  async function mpost(path, body) {
    const r = await fetch(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body || {}) });
    const j = await r.json(); if (j.ok === false && j.message) alert(j.message); return j;
  }
  async function mpoll() {
    try {
      const s = await (await fetch("/api/motion")).json();
      layout = s.layout;
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
      val("cInfo").textContent = !c.available ? "OpenCV がありません" : c.running
        ? `カメラ ${c.index}、${c.width}×${c.height}、${c.fps} fps${s.video_recording ? "・録画中" : ""}` : "";
      const want = c.running ? c.index + ":" + c.width : null;
      if (want !== camSeq) {
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
    try { pose = await (await fetch("/api/motion/pose")).json(); } catch (e) { pose = null; }
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
  mpoll(); ploop();
})();
</script>
"""


def page() -> bytes:
    html = (BER_DIR / "instruments" / "ber_gui.html").read_text(encoding="utf-8")
    for anchor in ('<div class="cols">', "</body>"):
        if anchor not in html:
            raise RuntimeError(f"ber_gui.html に {anchor!r} がありません。motion_ber_gui.py の差し込み位置を直してください")
    html = html.replace("<title>VHBR BER モニタ</title>", "<title>VHBR BER × モーション</title>", 1)
    html = html.replace('<div class="cols">', PANEL + '  <div class="cols">', 1)
    html = html.replace("</body>", PANEL_JS + "</body>", 1)
    return html.encode("utf-8")


def motion_status() -> dict:
    s = bg.SESSION
    logging = isinstance(s, MotionSessionLog) and MOTION.sink is not None
    return dict(MOTION.status(), require_motion=OPTS["require_motion"], auto_phase=OPTS["auto_phase"],
                model=PHASE_MODEL_NAME, logging=logging, log_dir=s.dir.name if logging else "", last=LAST,
                camera=CAMERA.status(), record_video=OPTS["record_video"],
                video_recording=logging and getattr(s, "video", None) is not None)


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
        return super().do_GET()

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
            if u.path == "/api/live/start" and OPTS["require_motion"] and not MOTION.receiving():
                return self._send(dict(ok=False, message="OptiTrack のフレームが来ていません。モーションの欄で接続するか、"
                                                         "「OptiTrack なしでも開始できる」を入れてください"))
            if u.path == "/api/shutdown":
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
