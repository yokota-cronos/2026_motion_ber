#!/usr/bin/env python3
"""motion.py -- OptiTrack (Motive / NatNet) からラベル付きマーカー 7 点の座標を受け続ける。

nanoVNA_OptiTrack_get_dataset/sync_optitrack_nanovna.py の NatNet 受信とラベル認識だけを
抜き出したもの。あちらは tkinter・pyserial・scikit-rf を import するので、BER 側の環境
(uv、numpy/scipy だけ) からはそのまま読めない。**対応付けの規則は同じにしてある**:

  * ラベル付きマーカーの id_num = 上位 16 bit model_id / 下位 16 bit marker_id
  * marker_id = i のマーカーを names[i-1] に割り当てる (names は MARKER_GROUPS の配信順)
  * occluded ビット付き・原点近傍・NaN は「このフレームでは見えていない」

Motive 側でラベルの並びを変えたら、あちらの MARKER_GROUPS とここを両方直すこと。

時刻は **time.perf_counter() で受信時刻を打つ**。BER 側の収録開始もこの時計で打つので、
2 つを直接比べられる。Motive の処理・配信の遅延 (数 ms、未較正) はこの時刻に乗ったまま。
物理的な撮影時刻に近いのは OptiMidExp (カメラ露光中央) だが、Motive の時計なので
perf_counter とは直接比べられない。
"""
from __future__ import annotations

import collections
import math
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
NATNET_DIR = Path(os.environ.get(
    "NATNET_DIR", HERE.parent / "nanoVNA_OptiTrack_get_dataset" / "NatNetSDK" / "Samples" / "PythonClient"))

# sync_optitrack_nanovna.py の MARKER_GROUPS と同じ。names は Motive の配信順 (= marker_id 1..7)。
MARKER_GROUPS = {
    "upper": {"label": "上半身",
              "names": ["chest", "R_upperarm", "L_upperarm", "R_joint", "R_forearm", "L_forearm", "L_joint"]},
    "lower": {"label": "下半身",
              "names": ["waist", "L_shin", "L_thigh", "R_thigh", "L_knee", "R_knee", "R_shin"]},
}
# 3D 表示: live_dashboard.py の MARKER_STYLE / BODY_LAYOUTS と同じ配色・ボーン・角度 (右 = 赤系、左 = 青系)
MARKER_COLORS = {
    "R_upperarm": "red", "R_joint": "darkorange", "R_forearm": "firebrick", "chest": "dimgray",
    "L_upperarm": "blue", "L_joint": "deepskyblue", "L_forearm": "navy",
    "R_thigh": "red", "R_knee": "darkorange", "R_shin": "firebrick", "waist": "dimgray",
    "L_thigh": "blue", "L_knee": "deepskyblue", "L_shin": "navy",
}
BODY_LAYOUTS = {
    "upper": dict(bones=[("chest", "R_upperarm", "gray"), ("R_upperarm", "R_joint", "red"),
                         ("R_joint", "R_forearm", "red"), ("chest", "L_upperarm", "gray"),
                         ("L_upperarm", "L_joint", "blue"), ("L_joint", "L_forearm", "blue")],
                  angles=[("右肘", ("R_upperarm", "R_joint", "R_forearm"), "crimson"),
                          ("左肘", ("L_upperarm", "L_joint", "L_forearm"), "royalblue")]),
    "lower": dict(bones=[("waist", "R_thigh", "gray"), ("R_thigh", "R_knee", "red"),
                         ("R_knee", "R_shin", "red"), ("waist", "L_thigh", "gray"),
                         ("L_thigh", "L_knee", "blue"), ("L_knee", "L_shin", "blue")],
                  angles=[("右膝", ("R_thigh", "R_knee", "R_shin"), "crimson"),
                          ("左膝", ("L_thigh", "L_knee", "L_shin"), "royalblue")]),
}
# 対象アセットの model_id。None なら model_id を問わない。上半身・下半身のアセットを同時に
# 配信していると相手側のマーカーを拾うので、そのときは診断に出る model= の値を入れる。
MARKER_MODEL_ID: int | None = None
OCCLUSION_ORIGIN_EPS = 1e-6
BUFFER_SEC = 60.0            # 覚えておくフレーム (360 fps でも 60 秒ぶん)
STALE_SEC = 1.0              # これだけフレームが来なければ「途絶」
NEAREST_MAX_GAP = 0.05       # テストフレームとモーションを結び付ける時刻差の上限 [s] (120 fps の 6 フレーム)


def _is_occluded(pos) -> bool:
    try:
        x, y, z = float(pos[0]), float(pos[1]), float(pos[2])
    except (TypeError, ValueError, IndexError):
        return True
    if x != x or y != y or z != z:
        return True
    e = OCCLUSION_ORIGIN_EPS
    return abs(x) <= e and abs(y) <= e and abs(z) <= e


def collect_labeled(mocap, n: int) -> np.ndarray:
    """フレームから (n, 3) の座標を返す。見えていないマーカーは NaN。"""
    xyz = np.full((n, 3), np.nan)
    lm = getattr(mocap, "labeled_marker_data", None)
    for mk in getattr(lm, "labeled_marker_list", None) or []:
        if getattr(mk, "param", 0) & 0x01:            # occluded ビット
            continue
        if MARKER_MODEL_ID is not None and (mk.id_num >> 16) != MARKER_MODEL_ID:
            continue
        i = (mk.id_num & 0xFFFF) - 1                  # marker_id は 1 始まり
        if 0 <= i < n and not _is_occluded(mk.pos):
            xyz[i] = (float(mk.pos[0]), float(mk.pos[1]), float(mk.pos[2]))
    return xyz


def describe_frame(mocap) -> str:
    """受信フレームの構造 (ラベル付きマーカーの id と座標)。names の並びを Motive に合わせるのに使う。"""
    lm = getattr(mocap, "labeled_marker_data", None)
    rows = getattr(lm, "labeled_marker_list", None) or []
    out = [f"LabeledMarker {len(rows)} 個"]
    for mk in rows:
        occ = bool(getattr(mk, "param", 0) & 0x01)
        out.append(f"  id={mk.id_num} model={mk.id_num >> 16} marker={mk.id_num & 0xFFFF} occ={occ} "
                   f"pos=({mk.pos[0]:.3f}, {mk.pos[1]:.3f}, {mk.pos[2]:.3f})")
    return "\n".join(out)


class Frame:
    __slots__ = ("t", "frame", "motive_t", "mid_exp", "xyz")

    def __init__(self, t, frame, motive_t, mid_exp, xyz):
        self.t, self.frame, self.motive_t, self.mid_exp, self.xyz = t, frame, motive_t, mid_exp, xyz


class MotionStream:
    """NatNet の受信スレッドからフレームを受け、リングバッファと (あれば) CSV に流す。

    読み出しは別スレッド (BER のワーカー、HTTP) から。全部 self.lock の下で触る。
    """

    def __init__(self):
        self.lock = threading.Lock()
        self.group = "upper"
        self.names = tuple(MARKER_GROUPS["upper"]["names"])
        self.buf: collections.deque[Frame] = collections.deque(maxlen=int(360 * BUFFER_SEC))
        self.client = None
        self.source = ""             # "natnet" / "fake" / ""
        self.n_frames = 0
        self.diag = ""
        self.error = ""
        self.sink = None             # フレームごとに呼ぶ (Frame) -> None。SessionLog が付け外しする
        self._fake_stop = threading.Event()
        self._fake_thread: threading.Thread | None = None

    # ------------------------------------------------------------------ 開く / 閉じる
    def set_group(self, key: str) -> None:
        if key not in MARKER_GROUPS:
            raise ValueError(f"計測部位は {list(MARKER_GROUPS)} のどれか")
        with self.lock:
            if self.sink is not None:
                raise RuntimeError("記録中は計測部位を変えられません (CSV の列が変わるため)")
            self.group, self.names = key, tuple(MARKER_GROUPS[key]["names"])
            self.buf.clear()

    def connect(self, server_ip="127.0.0.1", local_ip="127.0.0.1", multicast=False) -> None:
        self.close()
        if str(NATNET_DIR) not in sys.path:
            sys.path.insert(0, str(NATNET_DIR))
        try:
            from NatNetClient import NatNetClient
        except ImportError as e:
            raise RuntimeError(f"NatNetClient.py を読めません ({NATNET_DIR}): {e}") from e
        c = NatNetClient()
        c.set_client_address(local_ip)
        c.set_server_address(server_ip)
        c.set_use_multicast(bool(multicast))
        try:
            c.set_print_level(0)                # 毎フレームのダンプを止める
        except Exception:                        # noqa: BLE001 (古い SDK には無い)
            pass
        c.new_frame_with_data_listener = self._on_frame
        with self.lock:
            self.diag, self.error, self.n_frames = "", "", 0
        if not c.run("d"):
            try:
                c.shutdown()
            except Exception:                    # noqa: BLE001
                pass
            raise RuntimeError("NatNetClient を起動できません。Motive の Data Streaming "
                               "(IP / Unicast・Multicast / Labeled Markers) を確かめてください")
        self.client, self.source = c, "natnet"

    def start_fake(self, fps: float = 120.0) -> None:
        """Motive 無しで GUI と記録を試すための合成データ (腕を振るような動き)。"""
        self.close()
        self._fake_stop.clear()
        self.source = "fake"

        def run():
            k, t0 = 0, time.perf_counter()
            base = np.array([[0, 1.3, 0], [0.2, 1.4, 0], [-0.2, 1.4, 0], [0.35, 1.2, 0],
                             [0.4, 1.0, 0], [-0.4, 1.0, 0], [-0.35, 1.2, 0]], float)
            while not self._fake_stop.is_set():
                k += 1
                due = t0 + k / fps
                time.sleep(max(0.0, due - time.perf_counter()))
                w = 2 * math.pi * 0.5 * (due - t0)
                xyz = base + 0.1 * np.sin(w + np.arange(7))[:, None] * np.array([1, 0.5, 1])
                if k % 400 < 20:
                    xyz[4] = np.nan                  # 時々 1 点隠す
                self._ingest(time.perf_counter(), k, k / fps, None, xyz)

        self._fake_thread = threading.Thread(target=run, daemon=True)
        self._fake_thread.start()

    def close(self) -> None:
        self._fake_stop.set()
        if self._fake_thread is not None:
            self._fake_thread.join(timeout=2)
            self._fake_thread = None
        c, self.client = self.client, None
        if c is not None:
            try:
                c.shutdown()
            except Exception:                    # noqa: BLE001
                pass
            for attr in ("data_thread", "command_thread"):
                th = getattr(c, attr, None)
                if th is not None and th.is_alive():
                    th.join(timeout=2.0)
        self.source = ""
        with self.lock:
            self.buf.clear()                       # 切断前のフレームを次の記録に混ぜない
            self.n_frames = 0

    # ------------------------------------------------------------------ 受信
    def _on_frame(self, data: dict) -> None:
        t = time.perf_counter()
        mocap = data.get("mocap_data")
        frame = data.get("frame_number")
        ts = mid = None
        if mocap is not None:
            if frame is None:
                frame = getattr(getattr(mocap, "prefix_data", None), "frame_number", None)
            suffix = getattr(mocap, "suffix_data", None)
            if suffix is not None:
                ts = getattr(suffix, "timestamp", None)
                mid = getattr(suffix, "stamp_camera_mid_exposure", None)
                ts = None if ts == -1 else ts                     # SDK は未設定を -1 で表す
                mid = None if mid == -1 else mid
        n = len(self.names)
        xyz = collect_labeled(mocap, n) if mocap is not None else np.full((n, 3), np.nan)
        if mocap is not None and self.n_frames < 3:
            self.diag = describe_frame(mocap)
        self._ingest(t, frame, ts, mid, xyz)

    def _ingest(self, t, frame, motive_t, mid_exp, xyz) -> None:
        f = Frame(t, frame, motive_t, mid_exp, xyz)
        with self.lock:
            self.buf.append(f)
            self.n_frames += 1
            sink = self.sink
        if sink is not None:
            try:
                sink(f)
            except Exception as e:               # noqa: BLE001 (受信スレッドを落とさない)
                self.error = f"記録に失敗: {type(e).__name__}: {e}"

    # ------------------------------------------------------------------ 読み出し
    def window(self, t0: float, t1: float) -> list[Frame]:
        """受信時刻が [t0, t1] のフレーム (古い順)。"""
        out = []
        with self.lock:
            for f in reversed(self.buf):
                if f.t < t0:
                    break
                if f.t <= t1:
                    out.append(f)
        out.reverse()
        return out

    def nearest(self, ts: np.ndarray, max_gap: float = NEAREST_MAX_GAP) -> np.ndarray:
        """各時刻にいちばん近いフレームの Motive フレーム番号。max_gap [s] より離れていれば -1。

        離れたフレームを返すと、切断前の古いフレームに結び付いてしまう (2026-10-27、
        3 時間前の合成モーションのフレーム番号が frames.csv の全行に入った)。"""
        with self.lock:
            if not self.buf:
                return np.full(len(ts), -1, np.int64)
            t = np.fromiter((f.t for f in self.buf), float, len(self.buf))
            fr = np.fromiter((-1 if f.frame is None else f.frame for f in self.buf), np.int64, len(self.buf))
        i = np.clip(np.searchsorted(t, ts), 1, len(t) - 1) if len(t) > 1 else np.zeros(len(ts), int)
        if len(t) > 1:
            i = np.where(np.abs(t[i - 1] - ts) <= np.abs(t[i] - ts), i - 1, i)
        return np.where(np.abs(t[i] - ts) <= max_gap, fr[i], -1)

    def latest(self) -> Frame | None:
        with self.lock:
            return self.buf[-1] if self.buf else None

    def fps(self, span: float = 1.0) -> float:
        with self.lock:
            if len(self.buf) < 2:
                return 0.0
            t_end = t_first = self.buf[-1].t
            n = 0
            for f in reversed(self.buf):
                if f.t < t_end - span:
                    break
                n, t_first = n + 1, f.t
        return (n - 1) / (t_end - t_first) if n > 1 and t_end > t_first else 0.0

    def receiving(self) -> bool:
        f = self.latest()
        return f is not None and time.perf_counter() - f.t < STALE_SEC

    def pose(self) -> dict:
        """3D 表示用の最新の姿勢 (軽いので 30 Hz で呼ばれてよい)。"""
        f = self.latest()
        xyz = None if f is None else [None if np.isnan(p).any() else [round(float(v), 5) for v in p]
                                      for p in f.xyz]
        return dict(group=self.group, names=self.names, frame=None if f is None else f.frame, xyz=xyz,
                    receiving=self.receiving())

    def layout(self) -> dict:
        L = BODY_LAYOUTS[self.group]
        return dict(colors={n: MARKER_COLORS.get(n, "black") for n in self.names},
                    bones=L["bones"], angles=L["angles"])

    def status(self) -> dict:
        f = self.latest()
        age = None if f is None else time.perf_counter() - f.t
        markers = []
        for i, name in enumerate(self.names):
            p = None if f is None or np.isnan(f.xyz[i]).any() else [round(float(v), 4) for v in f.xyz[i]]
            markers.append(dict(name=name, pos=p))
        return dict(source=self.source, receiving=self.receiving(), group=self.group,
                    group_label=MARKER_GROUPS[self.group]["label"], fps=round(self.fps(), 1),
                    frames=self.n_frames, frame=None if f is None else f.frame,
                    age_ms=None if age is None else round(age * 1e3, 1),
                    markers=markers, diag=self.diag, error=self.error, layout=self.layout(),
                    groups={k: v["label"] for k, v in MARKER_GROUPS.items()})
