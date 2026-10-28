#!/usr/bin/env python3
"""replay.py -- motion_ber_gui.py が残した記録フォルダを読み、時刻を指定して中身を引く。

    姿勢       motion.csv            pose(t)  いちばん近い Motive フレーム
    カメラ     camera.mp4 + camera_frames.csv  jpeg(t)  t 以前で最後に撮れたコマ
    BER        ber_motion.csv / frames.csv     at(t)    t を含む収録と、そのフレームの誤り数
    生データ   cap_*.ad3 (ber_motion.csv の raw_file)   波形は motion_ber_gui.py が ber_gui の
               再生と同じ手順 (fd.ddc -> receiver_for -> make_view) で描く

時刻 t はどのファイルも同じ「記録開始からの秒」(perf_counter) なので、そのまま突き合わせられる。
"""
from __future__ import annotations

import csv
import json
import threading
from pathlib import Path

import numpy as np

import motion

try:
    import cv2
except ImportError:
    cv2 = None

STREAM_WIDTH = 640


def _rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _float(v, default=np.nan) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def list_sessions(root: Path) -> list[dict]:
    """motion.csv がある記録フォルダ (新しい順)。"""
    out = []
    if not root.is_dir():
        return out
    for d in sorted((p for p in root.iterdir() if (p / "motion.csv").exists()), key=lambda p: p.name, reverse=True):
        n_cap = max(0, sum(1 for _ in open(d / "ber_motion.csv", encoding="utf-8")) - 1) if (d / "ber_motion.csv").exists() else 0
        out.append(dict(path=str(d), name=d.name, caps=n_cap, raw=len(list(d.glob("*.ad3"))),
                        video=(d / "camera.mp4").exists() and (d / "camera_frames.csv").exists()))
    return out


class Replay:
    def __init__(self, folder: Path):
        self.dir = Path(folder)
        mc = json.loads((self.dir / "motion_conditions.json").read_text(encoding="utf-8"))
        self.cond = json.loads((self.dir / "conditions.json").read_text(encoding="utf-8"))
        self.names = tuple(mc["marker_names"])
        self.group = mc.get("group", "upper")

        # 姿勢: (n,) 時刻と (n, 7, 3) 座標 (見えていなければ NaN)
        m = _rows(self.dir / "motion.csv")
        self.tm = np.array([_float(r["t"]) for r in m])
        self.fr = np.array([int(r["OptiFrame"]) if r["OptiFrame"] else -1 for r in m], np.int64)
        self.xyz = np.array([[[_float(r[f"{n}_{a}"]) for a in "XYZ"] for n in self.names] for r in m]).reshape(-1, len(self.names), 3)

        # 収録 (BER の 1 回ぶん)
        self.caps = []
        for r in _rows(self.dir / "ber_motion.csv"):
            raw = r.get("raw_file", "") or ""
            self.caps.append(dict(cap=int(r["cap"]), t0=_float(r["t_start"]), t1=_float(r["t_end"]),
                                  ber=_float(r["ber"]), errors=int(r["errors"]), bits=int(r["bits"]),
                                  frames=int(r["frames"]), expected=int(r["expected"]),
                                  level=_float(r["level"]), phase=_float(r["phase"]),
                                  raw=raw if raw and (self.dir / raw).exists() else ""))
        self.cap_t0 = np.array([c["t0"] for c in self.caps])

        # テストフレーム
        f = _rows(self.dir / "frames.csv")
        self.f_cap = np.array([int(r["cap"]) for r in f], np.int64)
        self.f_k = np.array([int(r["frame"]) for r in f], np.int64)
        self.f_t = np.array([_float(r["t"]) for r in f])
        self.f_err = np.array([int(r["errors"]) for r in f], np.int64)

        # ④ 現在の BER 用: その時刻までの累積を、フレーム数から O(1) で引けるようにしておく
        self.npay = int(self.cond.get("payload", 600))
        ok = self.f_err >= 0
        self.cum_frames = np.cumsum(ok)
        self.cum_bits = self.cum_frames * self.npay
        self.cum_errors = np.cumsum(np.where(ok, self.f_err, 0))
        # BER の推移の点: ber_log.csv (ライブの画面に出していた値そのもの) を収録の順に
        self.points = []
        for c, r in zip(self.caps, _rows(self.dir / "ber_log.csv")):
            p = {k: _float(r.get(k), None) for k in ("t", "level", "phase", "q", "eye", "peak", "sig_ms",
                                                    "t_cap", "t_ddc", "t_demod")}
            p = {k: (None if v is not None and v != v else v) for k, v in p.items()}   # NaN は JSON に書けない
            p.update(frames=c["frames"], expected=c["expected"], bits=c["bits"], errors=c["errors"], src="replay")
            if p["t_demod"] is None:
                p["t_demod"] = 0.0
            self.points.append(p)
        self.ber_key = None          # 直近に ④ へ反映した (収録数, フレーム数)

        # カメラ
        c = _rows(self.dir / "camera_frames.csv")
        self.tc = np.array([_float(r["t"]) for r in c])
        self.video_path = self.dir / "camera.mp4"
        self.has_video = cv2 is not None and self.tc.size > 0 and self.video_path.exists()
        self.vlock = threading.Lock()
        self.vcap = None
        self.vpos = -1               # 直近に読んだコマの番号
        self.vframe = None
        self.vjpeg = (-1, None)

        ends = [a[-1] for a in (self.tm, self.f_t, self.tc) if a.size] + [c["t1"] for c in self.caps[-1:]]
        self.duration = float(max(ends)) if ends else 0.0

    # ------------------------------------------------------------------ 時刻で引く
    def pose(self, t: float) -> dict:
        if self.tm.size == 0:
            return dict(names=self.names, xyz=None, frame=None, receiving=False)
        i = int(np.clip(np.searchsorted(self.tm, t), 1, self.tm.size - 1)) if self.tm.size > 1 else 0
        if self.tm.size > 1 and abs(self.tm[i - 1] - t) <= abs(self.tm[i] - t):
            i -= 1
        xyz = [None if np.isnan(p).any() else [round(float(v), 5) for v in p] for p in self.xyz[i]]
        return dict(names=self.names, xyz=xyz, frame=int(self.fr[i]), receiving=bool(abs(self.tm[i] - t) < 0.1))

    def cap_at(self, t: float) -> dict | None:
        """t を含む収録。どれにも入っていなければ None。"""
        i = int(np.searchsorted(self.cap_t0, t, side="right")) - 1
        if 0 <= i < len(self.caps) and t <= self.caps[i]["t1"]:
            return self.caps[i]
        return None

    def at(self, t: float) -> dict:
        """画面の文字表示用: t の収録・フレームと誤り数。"""
        c = self.cap_at(t)
        if c is None:
            return dict(cap=None)
        sel = np.nonzero(self.f_cap == c["cap"])[0]
        # フレームが 1 枚も取れなかった収録の BER は NaN。JSON に NaN は書けない (ブラウザが読めない) ので None にする
        out = dict(cap=c["cap"], ber=None if c["ber"] != c["ber"] else c["ber"], raw=bool(c["raw"]),
                   frames=c["frames"], expected=c["expected"])
        if sel.size:
            j = sel[int(np.clip(np.searchsorted(self.f_t[sel], t, side="right") - 1, 0, sel.size - 1))]
            out.update(frame=int(self.f_k[j]), frame_errors=int(self.f_err[j]))
        return out

    def ber_upto(self, t: float) -> tuple[int, int]:
        """t までに始まった収録の数と、t までに始まったテストフレームの数。

        収録の先頭と最初のフレームの間 (約 0.4 ms) では、at() と波形は最初のフレームを出すので、
        数もそのフレームまで入れて揃える。"""
        nc = int(np.searchsorted(self.cap_t0, t, side="right"))
        nf = int(np.searchsorted(self.f_t, t, side="right"))
        c = self.cap_at(t)
        if c is not None:
            sel = np.nonzero(self.f_cap == c["cap"])[0]
            if sel.size and nf <= sel[0]:
                nf = int(sel[0]) + 1
        return nc, nf

    def _read(self, k: int):
        """動画のコマ k (BGR)。self.vlock の中から呼ぶ。続けて読むと順に読むだけなので速い。"""
        if self.vcap is None:
            self.vcap = cv2.VideoCapture(str(self.video_path))
            self.vpos, self.vframe = -1, None
        if k == self.vpos and self.vframe is not None:
            return self.vframe
        if not (self.vpos < k <= self.vpos + 30):          # 戻る / 大きく飛ぶときだけシーク
            self.vcap.set(cv2.CAP_PROP_POS_FRAMES, k)
            self.vpos = k - 1
        f = None
        while self.vpos < k:
            ok, f = self.vcap.read()
            if not ok:
                return None
            self.vpos += 1
        self.vframe = f
        return f

    def frame_bgr(self, k: int):
        if not self.has_video:
            return None
        with self.vlock:
            f = self._read(int(np.clip(k, 0, self.tc.size - 1)))
            return None if f is None else f.copy()

    def jpeg(self, t: float) -> bytes | None:
        """t 以前で最後に撮れたコマ (縮小した JPEG)。"""
        if not self.has_video:
            return None
        k = int(np.clip(np.searchsorted(self.tc, t, side="right") - 1, 0, self.tc.size - 1))
        with self.vlock:
            if self.vjpeg[0] == k:
                return self.vjpeg[1]
            f = self._read(k)
            if f is None:
                return self.vjpeg[1]
            h, w = f.shape[:2]
            if w > STREAM_WIDTH:
                f = cv2.resize(f, (STREAM_WIDTH, int(h * STREAM_WIDTH / w)), interpolation=cv2.INTER_AREA)
            ok, buf = cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 80])
            self.vjpeg = (k, buf.tobytes() if ok else None)
            return self.vjpeg[1]

    def meta(self) -> dict:
        L = motion.BODY_LAYOUTS.get(self.group, motion.BODY_LAYOUTS["upper"])
        return dict(name=self.dir.name, duration=round(self.duration, 4), names=self.names, group=self.group,
                    layout=dict(colors={n: motion.MARKER_COLORS.get(n, "black") for n in self.names},
                                bones=L["bones"], angles=L["angles"]),
                    caps=[dict(c, ber=None if c["ber"] != c["ber"] else c["ber"], raw=bool(c["raw"])) for c in self.caps],
                    video=self.has_video, n_raw=sum(1 for c in self.caps if c["raw"]),
                    motion_frames=int(self.tm.size), div=self.cond.get("div"), frame_s=self.cond.get("frame_s"),
                    level_dbm=self.cond.get("level_dbm"), phase_deg=self.cond.get("phase_deg"))

    def close(self) -> None:
        with self.vlock:
            if self.vcap is not None:
                self.vcap.release()
                self.vcap = None
