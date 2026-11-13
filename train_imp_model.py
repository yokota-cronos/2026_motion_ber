#!/usr/bin/env python3
"""train_imp_model.py -- imp_phase_model.py が使うモデル (Ridge AR + ExtraTrees) を学習して保存する。

    uv run train_imp_model.py              # 1111 で学習 → models/imp_1111.joblib
    uv run train_imp_model.py --eval       # 先に 20 ブロック分割で精度を出す (レポートの 4 節と比べる用)
    uv run train_imp_model.py --trees 500 --leaf 1   # レポートと同じ木 (ファイルが数 GB になる)
    uv run train_imp_model.py --dataset 1013ver1     # 別の学習データ → models/imp_1013ver1.joblib

データと前処理は ../2026_motion_to_meander_imp の report_20261111_dataset1111 (1013ver1 は report_20261013_dataset1013ver1。手法は
同じ) と同じものを、あちらのコード (src/report0923_common.py) をそのまま呼んで作る:
  1111_1547_kajihara_@13.56MHz.csv、左右チャネルの入れ替え補正 (1013ver1・1111 とも逆だった)、despike、
  急変フレーム (|dZ/dt| の p99 超) の除外。
違うのは 3 点:
  * 運用のモデルなので、ブロック分割で test に取っておかず**全フレームで学習する** (--eval のときだけ分割する)
  * 先読みの「過去 8 点」「h 秒先」を時刻で数える (imp_phase_model.py の docstring)
  * ExtraTrees を 100 本・葉の最小サンプル数 5 にした (レポートは 500 本・1)。1013ver1 では、レポートの分割で比べて
    test の RMSE は 4 ターゲットとも同じか少し良く (L_Z_X 1.487 → 1.459 Ω)、ファイルは 6.5 GB → 約 0.2 GB、
    1 点の推定は 71 → 13 ms だった
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

import imp_phase_model as M

HERE = Path(__file__).resolve().parent
IMP_DIR = Path(os.environ.get("MEANDER_IMP_DIR", HERE.parent / "2026_motion_to_meander_imp"))
DATASET = M.DATASET     # 既定の学習データ。imp_phase_model.MODEL_PATH もこの名前
SEED = 0
# --eval で並べるレポート 4 節の値 (20 ブロック分割の test、ET 500 本・葉 1)
REPORT_RMSE = {
    "1013ver1": "L_Z_X 現在 1.49 / 0.1 s 先 1.74 / 0.2 s 先 2.36、R_Z_X 2.44 / 2.62 / 3.10",
    "1111": "L_Z_X 現在 0.95 / 0.1 s 先 1.23 / 0.2 s 先 2.03、R_Z_X 1.22 / 1.40 / 2.02",
}


def load(dataset: str):
    sys.path.insert(0, str(IMP_DIR / "src"))
    import report0923_common as C
    from dataset import MARKERS, _xyz, build_features, steady_mask

    assert tuple(MARKERS) == M.MARKERS, "dataset.MARKERS と imp_phase_model.MARKERS の並びが違う"
    sessions = C.load_sessions(dataset=dataset)
    thresh = C.steady_threshold(sessions)
    out = []
    for s in sessions:
        P = np.hstack([_xyz(s.df, m) for m in MARKERS])
        for side in M.SIDES:                       # 特徴量がレポートのコードと同じになっているか
            ref = build_features(s.df, side=side)[0]
            got = M.side_features(P, side)
            assert np.allclose(got, ref, atol=1e-5, equal_nan=True), f"{side} の 6 特徴量がレポートと違う"
        vis = ~np.isnan(P).any(axis=1)
        ok = steady_mask(s.y, s.t, thresh=thresh) & vis
        out.append(dict(name=s.name, t=s.t, P=P, y=s.y, vis=vis, ok=ok))
        print(f"  [{s.name}] {s.n} フレーム、学習に使う {ok.sum()} (急変 {thresh:.0f} Ω/s 超・隠れを除外)")
    return out, thresh


def ar_samples(sess: dict, h: float, idx: np.ndarray):
    """フレーム idx の時刻を「いま」とした Ridge の入力と、h 秒先の変位。履歴か先が欠けるものは落とす。"""
    t, P = sess["t"][sess["vis"]], sess["P"][sess["vis"]]   # 実行時と同じく、隠れたフレームは補間に使わない
    now = sess["t"][idx]
    keep = M.spans_ok(t, now - (M.TAPS - 1) * M.STEP, now + h)
    now = now[keep]
    tq = M.history_times(now)                                           # (n, TAPS)
    Pq = M.resample(t, P, tq.ravel()).reshape(len(now), M.TAPS, -1)
    X, cur = M.ar_design(Pq)
    return X, M.resample(t, P, now + h) - cur, cur, idx[keep]


def fit(sessions, idx_by_sess, trees: int, leaf: int):
    from sklearn.ensemble import ExtraTreesRegressor
    from sklearn.linear_model import Ridge

    et = {}
    for side, cols in M.SIDES.items():
        X = np.vstack([M.side_features(s["P"][i], side) for s, i in zip(sessions, idx_by_sess)])
        y = np.vstack([s["y"][i][:, list(cols)] for s, i in zip(sessions, idx_by_sess)])
        t0 = time.perf_counter()
        et[side] = ExtraTreesRegressor(n_estimators=trees, min_samples_leaf=leaf, random_state=SEED,
                                         n_jobs=-1).fit(X, y)
        print(f"  ExtraTrees {side}: {len(X)} フレーム、{time.perf_counter() - t0:.1f} s")
    ridge = {}
    for h in M.HORIZON_GRID:                                           # 収録窓に合わせるため 0.02〜0.30 s の 0.01 刻み
        parts = [ar_samples(s, h, i) for s, i in zip(sessions, idx_by_sess)]
        X = np.vstack([p[0] for p in parts])
        Y = np.vstack([p[1] for p in parts])
        ridge[h] = Ridge(alpha=1.0).fit(X, Y)                           # レポートと同じ alpha
    print(f"  Ridge AR {M.HORIZON_GRID[0]:.2f}〜{M.HORIZON_GRID[-1]:.2f} s 先 ({len(ridge)} 本): "
          f"{len(X)} サンプル ({M.HORIZON_GRID[-1]:.2f} s 先のとき)")
    return et, ridge


def evaluate(sessions, trees: int, leaf: int, dataset: str) -> None:
    """レポートと同じ 20 ブロックのインターリーブ分割 (位相 1、境界 0.5 s) で学習 → test の精度。"""
    from dataset import block_split

    tr, te = [], []
    for s in sessions:
        fs = 1.0 / float(np.median(np.diff(s["t"])))
        a, b = block_split(len(s["t"]), n_blocks=20, gap=int(round(0.5 * fs)), phase=1)
        tr.append(a[s["ok"][a]])
        te.append(b[s["ok"][b]])
    et, ridge = fit(sessions, tr, trees, leaf)
    ref = REPORT_RMSE.get(dataset)
    print("\n  test の RMSE [Ω]" + (f" (レポート 4 節の Ridge AR → ET: {ref}。あちらはフレーム数で数えている)"
                                  if ref else ""))
    print("  " + "".join(f"{n:>9}" for n in ("", *M.TARGETS, "位相°")))

    def z_of(P):
        z = np.empty((len(P), 4))
        for side, cols in M.SIDES.items():
            z[:, list(cols)] = et[side].predict(M.side_features(P, side))
        return z

    for h in (0.0, *M.HORIZONS):
        Ps, ys, ph_err = [], [], []
        for s, i in zip(sessions, te):
            if h == 0.0:
                Ps.append(s["P"][i])
                ys.append(s["y"][i])
                continue
            X, _, cur, k = ar_samples(s, h, i)
            Ps.append(cur + ridge[h].predict(X))
            ys.append(np.vstack([np.interp(s["t"][k] + h, s["t"], s["y"][:, j]) for j in range(4)]).T)
        P, y = np.vstack(Ps), np.vstack(ys)
        z = z_of(P)
        rmse = np.sqrt(np.mean((z - y) ** 2, axis=0))
        ph = np.array([M.phase_for_impedance(v) for v in z]) - np.array([M.phase_for_impedance(v) for v in y])
        label = "現在" if h == 0.0 else f"{h:.1f} s 先"
        print("  " + f"{label:>8} " + "".join(f"{v:9.2f}" for v in rmse) + f"{np.sqrt(np.mean(ph ** 2)):9.2f}")
    print("  (位相° = 推定した Z から出した位相と、実測の Z から出した位相の差の RMSE)\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default=DATASET, help=f"report0923_common.DATASETS のキー (既定 {DATASET})")
    ap.add_argument("--trees", type=int, default=100, help="ExtraTrees の本数 (既定 100。レポートは 500)")
    ap.add_argument("--leaf", type=int, default=5, help="ExtraTrees の葉の最小サンプル数 (既定 5。レポートは 1)")
    ap.add_argument("--eval", action="store_true", help="保存の前に 20 ブロック分割で精度を出す")
    ap.add_argument("--out", type=Path, help="既定は models/imp_<dataset>.joblib (環境変数 IMP_MODEL があればそれ)")
    a = ap.parse_args()
    if a.out is None:
        a.out = M.MODEL_PATH if a.dataset == DATASET else HERE / "models" / f"imp_{a.dataset}.joblib"

    print(f"データセット {a.dataset} を読む ({IMP_DIR})")
    sessions, thresh = load(a.dataset)
    if a.eval:
        print("\n評価 (20 ブロック分割)")
        evaluate(sessions, a.trees, a.leaf, a.dataset)
    print("全フレームで学習")
    et, ridge = fit(sessions, [np.where(s["ok"])[0] for s in sessions], a.trees, a.leaf)

    import joblib
    a.out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(dict(
        dataset=a.dataset, sessions=[s["name"] for s in sessions], trained=datetime.now().isoformat(timespec="seconds"),
        steady_thresh=thresh, trees=a.trees, leaf=a.leaf, markers=M.MARKERS, targets=M.TARGETS, taps=M.TAPS, step=M.STEP,
        et=et, ridge=ridge,
    ), a.out)
    print(f"  -> {a.out} ({a.out.stat().st_size / 1e6:.0f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
