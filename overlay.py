#!/usr/bin/env python3
"""overlay.py -- 記録した動画に OptiTrack のマーカーの骨格を重ねるための、カメラの較正。

カメラ (C920) と OptiTrack は較正されていないので、記録そのものから求める:

  1. 動画のコマを間引いて MediaPipe Pose で骨格 (肩・肘・手首…) を取る
  2. マーカーを MediaPipe の点に対応させる (MARKER_TO_MP。chest = 両肩の中点、R_joint = 右肘、…)
  3. 同じ時刻の「マーカーの 3D 座標」と「動画上の 2D 位置」の組をたくさん集め、ピンホールカメラ
     (焦点距離 f は C920 の公称値に固定、主点は画面中央、歪みなし) の位置と向きを solvePnP で求める
  4. 動画とモーションの時刻のずれ (カメラの遅れ) も、再投影誤差が最小になるように探す

求めたあとは、どの時刻のマーカーも動画の上に投影して描ける (MediaPipe の骨格の代わりに、
実測したマーカーの骨格を出す)。

マーカーは体の正面のほぼ 1 枚の面に並ぶので、一般の射影行列 (DLT) は退化する。焦点距離も
決まらない (2026-10-27 の 30 秒の記録: f = 500 px で 8.0 px、2000 px で 6.4 px と差が無く、
探すと上限に張り付いて「6 m 先から望遠」の解になった)。だから f は公称値に固定して solvePnP。対応は「肘 = R_joint」以外は
体の寸法からの目安 (MARKER_TO_MP の比) なので、数 px〜十数 px の誤差は残る。
結果は記録フォルダの overlay_calib.json と pose_mediapipe.npz に置き、次からは読むだけにする。
"""
from __future__ import annotations

import json
import threading
import time
import urllib.request
from pathlib import Path

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

HERE = Path(__file__).resolve().parent
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
             "pose_landmarker_full/float16/1/pose_landmarker_full.task")
MODEL_PATH = HERE / "models" / "pose_landmarker_full.task"

# マーカー -> MediaPipe の点: (a, b, s) は「点 a から点 b へ s の位置」。
# MediaPipe の番号は本人から見た左右 (11 左肩 12 右肩 13 左肘 14 右肘 15 左手首 16 右手首 ...)。
# 比は 2026-10-27 の記録のマーカー間距離から決めた目安: 上腕のマーカーは肘から約 25 cm (= 肩寄り)、
# 前腕のマーカーは肘から約 24 cm (= 手首寄り)、chest は両上腕の中点より少し上。
MARKER_TO_MP = {
    "chest": (11, 12, 0.5),
    "R_upperarm": (12, 14, 0.2), "R_joint": (14, 14, 0.0), "R_forearm": (14, 16, 0.9),
    "L_upperarm": (11, 13, 0.2), "L_joint": (13, 13, 0.0), "L_forearm": (13, 15, 0.9),
    "waist": (23, 24, 0.5),
    "R_thigh": (24, 26, 0.4), "R_knee": (26, 26, 0.0), "R_shin": (26, 28, 0.5),
    "L_thigh": (23, 25, 0.4), "L_knee": (25, 25, 0.0), "L_shin": (25, 27, 0.5),
}
LR_SWAP = {11: 12, 12: 11, 13: 14, 14: 13, 15: 16, 16: 15, 23: 24, 24: 23, 25: 26, 26: 25, 27: 28, 28: 27}
# MediaPipe の骨格を薄く描くときのつなぎ方 (上半身 + 脚)
MP_EDGES = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24), (23, 24),
            (23, 25), (25, 27), (24, 26), (26, 28)]
VIS_MIN = 0.5                 # MediaPipe の visibility がこれ未満の点は使わない
N_CALIB_FRAMES = 150          # 較正に使うコマ数 (動画全体から等間隔)
LAGS = np.arange(-0.20, 0.401, 0.02)       # 探すカメラの遅れ [s]
FOCAL_PX_640 = 500.0          # C920 の 640x480 (対角画角 約 78°) の焦点距離 [px]。幅に比例させる


def ensure_model() -> Path:
    """MediaPipe の姿勢モデルが無ければ取ってくる (約 9 MB、初回だけ)。"""
    if not MODEL_PATH.exists():
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = MODEL_PATH.with_suffix(".part")
        urllib.request.urlretrieve(MODEL_URL, tmp)
        tmp.replace(MODEL_PATH)
    return MODEL_PATH


class PoseDetector:
    """MediaPipe PoseLandmarker (画像 1 枚ずつ)。detect() は (33, 3) = (x [px], y [px], visibility)。"""

    def __init__(self):
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision
        self.mp = mp
        opt = vision.PoseLandmarkerOptions(base_options=BaseOptions(model_asset_path=str(ensure_model())),
                                           running_mode=vision.RunningMode.IMAGE, num_poses=1)
        self.lm = vision.PoseLandmarker.create_from_options(opt)
        self.lock = threading.Lock()

    def detect(self, bgr: np.ndarray) -> np.ndarray | None:
        h, w = bgr.shape[:2]
        img = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        with self.lock:
            r = self.lm.detect(img)
        if not r.pose_landmarks:
            return None
        return np.array([(p.x * w, p.y * h, p.visibility) for p in r.pose_landmarks[0]])


def detect_frames(video: Path, frames: list[int], det: PoseDetector, progress=None) -> np.ndarray:
    """指定のコマで骨格を取る。(n, 33, 3)、見つからなかったコマは NaN。"""
    cap = cv2.VideoCapture(str(video))
    out = np.full((len(frames), 33, 3), np.nan)
    try:
        for i, k in enumerate(frames):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(k))
            ok, f = cap.read()
            if ok:
                p = det.detect(f)
                if p is not None:
                    out[i] = p
            if progress:
                progress(f"MediaPipe で骨格を取っています {i + 1}/{len(frames)}")
    finally:
        cap.release()
    return out


def _mp_point(lm: np.ndarray, spec, swap: bool):
    a, b, s = spec
    if swap:
        a, b = LR_SWAP.get(a, a), LR_SWAP.get(b, b)
    if lm[a, 2] < VIS_MIN or lm[b, 2] < VIS_MIN:
        return None
    return lm[a, :2] + s * (lm[b, :2] - lm[a, :2])


def _pairs(rep, frames, lms, lag: float, swap: bool):
    """(3D マーカー, 2D の MediaPipe 点) の組を集める。"""
    X, U = [], []
    names = rep.names
    for k, lm in zip(frames, lms):
        if np.isnan(lm).all():
            continue
        t = rep.tc[k] - lag
        i = int(np.clip(np.searchsorted(rep.tm, t), 0, rep.tm.size - 1))
        if i > 0 and abs(rep.tm[i - 1] - t) < abs(rep.tm[i] - t):
            i -= 1
        if abs(rep.tm[i] - t) > 0.015:
            continue
        for j, n in enumerate(names):
            spec = MARKER_TO_MP.get(n)
            if spec is None or np.isnan(rep.xyz[i, j]).any():
                continue
            u = _mp_point(lm, spec, swap)
            if u is not None:
                X.append(rep.xyz[i, j])
                U.append(u)
    return np.array(X, np.float64).reshape(-1, 3), np.array(U, np.float64).reshape(-1, 2)


def _fit(X, U, w, h, f):
    """焦点距離 f を決めたときのカメラの位置と向き、外れ値を除いた再投影誤差 [px]。"""
    K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]], np.float64)
    ok, rvec, tvec, inl = cv2.solvePnPRansac(X, U, K, None, reprojectionError=20.0, iterationsCount=200,
                                              flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok or inl is None or len(inl) < 8:
        return None
    inl = inl.ravel()
    rvec, tvec = cv2.solvePnPRefineLM(X[inl], U[inl], K, None, rvec, tvec)
    P, _ = cv2.projectPoints(X, rvec, tvec, K, None)
    e = np.linalg.norm(P.reshape(-1, 2) - U, axis=1)
    # 比べるのは「全部の組の誤差の中央値」。外れ値を除いた後の rms で比べると、多く捨てた
    # 方が良く見えてしまう (遅れの探索が上限 0.4 s に張り付いた)
    return dict(K=K, rvec=rvec, tvec=tvec, rms=float(np.sqrt(np.mean(e[inl] ** 2))), med=float(np.median(e)),
                n_in=int(inl.size), n=int(len(X)))


def calibrate(rep, progress=None, force: bool = False) -> dict:
    """記録 rep (replay.Replay) の動画とモーションから、カメラを較正する。結果は記録フォルダに置く。"""
    out_json = rep.dir / "overlay_calib.json"
    if out_json.exists() and not force:
        c = json.loads(out_json.read_text(encoding="utf-8"))
        if c.get("ok"):
            return c
    if not rep.has_video or rep.tm.size == 0:
        return dict(ok=False, message="動画かモーションがありません")
    say = progress or (lambda m: None)
    cap = cv2.VideoCapture(str(rep.video_path))
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    n_video = min(n_video, rep.tc.size)
    frames = sorted(set(np.linspace(0, n_video - 1, min(N_CALIB_FRAMES, n_video)).astype(int).tolist()))
    cache = rep.dir / "pose_mediapipe.npz"
    lms = None
    if cache.exists() and not force:
        z = np.load(cache)
        if list(z["frames"]) == frames:
            lms = z["lm"]
    if lms is None:
        say("MediaPipe を準備しています (初回はモデルを取ってくる)")
        lms = detect_frames(rep.video_path, frames, PoseDetector(), say)
        np.savez_compressed(cache, frames=np.array(frames), lm=lms)
    n_found = int((~np.isnan(lms[:, 0, 0])).sum())
    if n_found < 10:
        return dict(ok=False, message=f"MediaPipe が人を見つけたコマが少なすぎます ({n_found}/{len(frames)})")

    say("カメラの位置と向きを探しています")
    t0 = time.time()
    f = FOCAL_PX_640 * w / 640
    # 1) 遅れ 0 で、マーカーの左右と MediaPipe の左右の対応を決める。マーカーがほぼ 1 枚の面に
    #    並ぶと「正面から」と「左右を入れ替えて背中から」が見分けられない (2026-10-27: 8.47 対
    #    8.40 px)。だからラベルどおり (R_* = 本人の右 = MediaPipe の右) を基本にし、入れ替えは
    #    10 % 以上良いときだけ採る
    fits = {}
    for swap in (False, True):
        X, U = _pairs(rep, frames, lms, 0.0, swap)
        r = _fit(X, U, w, h, f) if len(X) >= 12 else None
        if r:
            fits[swap] = dict(r, f=f, swap=swap, lag=0.0)
    best = fits.get(False)
    if True in fits and (best is None or fits[True]["med"] < 0.9 * best["med"]):
        best = fits[True]
    if best is None:
        return dict(ok=False, message="マーカーと MediaPipe の点の組が足りません")
    # 2) カメラの遅れを探す。動きの少ない記録では遅れがほとんど決まらないので、
    #    遅れ 0 より 5 % 以上良くなるときだけ採る
    lag0, cand = best, None
    for lag in LAGS:
        X, U = _pairs(rep, frames, lms, float(lag), best["swap"])
        r = _fit(X, U, w, h, f) if len(X) >= 12 else None
        if r and (cand is None or r["med"] < cand["med"]):
            cand = dict(r, f=f, swap=best["swap"], lag=float(lag))
    if cand is not None and cand["med"] < 0.95 * lag0["med"]:
        best = cand
    c = dict(ok=True, width=w, height=h, focal_px=round(best["f"], 1), rvec=best["rvec"].ravel().tolist(),
             tvec=best["tvec"].ravel().tolist(), lag_s=round(best["lag"], 3), lr_swapped=bool(best["swap"]),
             rms_px=round(best["rms"], 2), median_px=round(best["med"], 2), inliers=best["n_in"], pairs=best["n"], frames_used=n_found,
             frames_sampled=len(frames), seconds=round(time.time() - t0, 1),
             note="ピンホール (主点 = 画面中央、歪みなし)。lag_s = モーションの時刻 = 動画の時刻 - lag_s")
    out_json.write_text(json.dumps(c, ensure_ascii=False, indent=2), encoding="utf-8")
    return c


def project(calib: dict, xyz: np.ndarray) -> np.ndarray:
    """(n, 3) [m] -> (n, 2) [px]。NaN の行は NaN のまま。"""
    out = np.full((len(xyz), 2), np.nan)
    ok = ~np.isnan(xyz).any(axis=1)
    if ok.any():
        f, w, h = calib["focal_px"], calib["width"], calib["height"]
        K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]], np.float64)
        P, _ = cv2.projectPoints(xyz[ok].astype(np.float64), np.array(calib["rvec"]), np.array(calib["tvec"]), K, None)
        out[ok] = P.reshape(-1, 2)
    return out
