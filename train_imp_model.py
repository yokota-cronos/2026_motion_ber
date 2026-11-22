#!/usr/bin/env python3
"""train_imp_model.py -- imp_phase_model.py が使うモデル (Ridge AR + ExtraTrees) を学習して保存する。

    uv run train_imp_model.py              # 1121 (8 マーカー) で学習 → models/imp_1121.joblib (既定、2026-11-22 から)
    uv run train_imp_model.py --eval       # 先に精度を出す (8 マーカー: 2123 で学習 → 2101 でテスト。7 マーカー: 20 ブロック分割)
    uv run train_imp_model.py --dataset 1118   # 7 マーカー (chest あり) → models/imp_1118.joblib (7 マーカーの記録・配信で自動で使う)

8 マーカー (1120・1121。../2026_motion_to_meander_imp/src/report1120_run.py の DATASETS) は rpy_same4 のモデル:
コイルごとに独立で、左コイルは左腕・右コイルは右腕の上腕・前腕のピッチ・ヨー 4 個だけを使う
(report_20261122_imu_features.pdf の 7 章。前処理は report1120_run と同じ。本編と別収録の両方の全フレームで学習する)。
以下は 7 マーカー (side6) の説明:
    uv run train_imp_model.py --trees 500 --leaf 1   # レポートと同じ木 (ファイルが数 GB になる)
    uv run train_imp_model.py --dataset 1111         # 別の学習データ → models/imp_1111.joblib

データと前処理は ../2026_motion_to_meander_imp の report_20261118_dataset1118 (1111 は report_20261111_dataset1111、
1013ver1 は report_20261013_dataset1013ver1。手法は同じ) と同じものを、あちらのコード (src/report0923_common.py) をそのまま呼んで作る:
  1118_1449_kajihara_@13.56MHz.csv、左右チャネルの入れ替え補正 (1013ver1・1111 は逆だった。1118 は入れ替え不要)、despike、
  急変フレーム (|dZ/dt| の p99 超) の除外、マーカー欠落・取り違えのフレームの除外 (marker_qc。1118 だけ有効。
  時刻の飛び・骨格長の伸び・速度の跳びとその前後 0.5 s。ET の学習から外し、Ridge の窓 (過去 8 点〜h 秒先) にもかからないようにする)。
違うのは 3 点:
  * 運用のモデルなので、ブロック分割で test に取っておかず**全フレームで学習する** (--eval のときだけ分割する)
  * 先読みの「過去 8 点」「h 秒先」を時刻で数える (imp_phase_model.py の docstring)
  * ExtraTrees を 100 本・葉の最小サンプル数 5 にした (レポートは 500 本・1)。1013ver1 では、レポートの分割で比べて
    test の RMSE は 4 ターゲットとも同じか少し良く (L_Z_X 1.487 → 1.459 Ω)、ファイルは 6.5 GB → 約 0.2 GB、
    1 点の推定は 71 → 13 ms だった
  * Ridge の入力と出力は chest に対する相対座標 (imp_phase_model.RIDGE_INPUT。2026-11-19 から。前かがみ・腕を大きく
    上げる動きで chest が動いても、腕と胸の位置関係だけを予測する)
  * Ridge は入力と出力を標準化し、α を学習フレームの後ろ 1 割 (時間で切る) で選ぶ (レポートは標準化なし・α = 1)。
    1111 の 20 ブロック分割で 0.1 s 先の位相の RMSE 3.15 → 3.07°・0.2 s 先 4.34 → 4.01°。1118_1449 では 4.49 → 4.30°
    (2026-11-19、A 座標そのまま / B chest 相対 / C 特徴量 を比べた結果。3 つの差より標準化の効果が大きかった)
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
    "1118": "L_Z_X 現在 1.74 / 0.1 s 先 1.99 / 0.2 s 先 2.67、R_Z_X 1.11 / 1.47 / 2.31",
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
        # marker_qc (1118 から) で外したフレームは、隠れたフレームと同じ扱い: ET の学習に使わず、
        # Ridge の補間にも使わない (除外は前後 0.5 s ずつ広げてあるので、窓にかかると spans_ok で落ちる)
        bad = getattr(s, "bad", np.zeros(s.n, bool))
        vis = ~np.isnan(P).any(axis=1) & ~bad
        ok = steady_mask(s.y, s.t, thresh=thresh) & vis
        out.append(dict(name=s.name, t=s.t, P=P, y=s.y, vis=vis, ok=ok))
        print(f"  [{s.name}] {s.n} フレーム、学習に使う {ok.sum()} (急変 {thresh:.0f} Ω/s 超・隠れ"
              + (f"・marker_qc {bad.sum()}" if bad.any() else "") + " を除外)")
    return out, thresh


def ar_samples(sess: dict, h: float, idx: np.ndarray, mode: str = M.RIDGE_INPUT):
    """フレーム idx の時刻を「いま」とした Ridge の入力と、h 秒先の変位 (どちらも表現 mode で)。履歴か先が欠けるものは落とす。

    返す: X、先の変位、いまの値 (表現 mode)、使ったフレーム、いまの絶対座標 (表現から姿勢に戻す用)。
    """
    t, P = sess["t"][sess["vis"]], sess["P"][sess["vis"]]   # 実行時と同じく、隠れたフレームは補間に使わない
    now = sess["t"][idx]
    keep = M.spans_ok(t, now - (M.TAPS - 1) * M.STEP, now + h)
    now = now[keep]
    tq = M.history_times(now)                                           # (n, TAPS)
    Pq = M.resample(t, P, tq.ravel()).reshape(len(now), M.TAPS, -1)
    Rq = M.ridge_repr(Pq.reshape(-1, Pq.shape[-1]), mode).reshape(len(now), M.TAPS, -1)
    X, cur = M.ar_design(Rq)
    fut = M.ridge_repr(M.resample(t, P, now + h), mode)
    return X, fut - cur, cur, idx[keep], Pq[:, -1]


RIDGE_ALPHAS = (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1e3, 1e4)   # 標準化した入力・出力に対する正則化の強さの候補
RIDGE_VAL = 0.1                                                   # α を選ぶ検証に使う、学習フレームの後ろの割合 (時間で切る)


def make_ridge(alpha: float):
    """入力 (過去 7 点の変位 147 列) と出力 (21 列) をそれぞれ平均 0・標準偏差 1 にそろえてから解く Ridge。

    そろえないと、正則化 α × Σ係数² が列の大きさ (0.02 s 前の変位は数 mm、0.14 s 前は数 cm) によって効き方が変わる。
    predict は元の単位 (m) で返すので、呼ぶ側 (imp_phase_model) はそのまま使える。
    """
    from sklearn.compose import TransformedTargetRegressor
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return TransformedTargetRegressor(regressor=make_pipeline(StandardScaler(), Ridge(alpha=alpha)),
                                      transformer=StandardScaler())


def fit_ridge(sessions, idx_by_sess, h: float):
    """先読み h の Ridge。α は学習フレームの後ろ RIDGE_VAL (時間で切る) で選び、全部で学習し直す。"""
    def stack(idxs):
        parts = [ar_samples(s, h, i) for s, i in zip(sessions, idxs)]   # 表現は M.RIDGE_INPUT (chest 相対)
        return np.vstack([p[0] for p in parts]), np.vstack([p[1] for p in parts])

    cut = [i[:int(len(i) * (1 - RIDGE_VAL))] for i in idx_by_sess]
    val = [i[int(len(i) * (1 - RIDGE_VAL)):] for i in idx_by_sess]
    Xf, Yf = stack(cut)
    Xv, Yv = stack(val)
    score = {a: float(np.mean((make_ridge(a).fit(Xf, Yf).predict(Xv) - Yv) ** 2 / Yv.var(axis=0)))
             for a in RIDGE_ALPHAS}
    alpha = min(score, key=score.get)
    X, Y = stack(idx_by_sess)
    return make_ridge(alpha).fit(X, Y), alpha, len(X)


def fit(sessions, idx_by_sess, trees: int, leaf: int):
    from sklearn.ensemble import ExtraTreesRegressor

    et = {}
    for side, cols in M.SIDES.items():
        X = np.vstack([M.side_features(s["P"][i], side) for s, i in zip(sessions, idx_by_sess)])
        y = np.vstack([s["y"][i][:, list(cols)] for s, i in zip(sessions, idx_by_sess)])
        t0 = time.perf_counter()
        et[side] = ExtraTreesRegressor(n_estimators=trees, min_samples_leaf=leaf, random_state=SEED,
                                         n_jobs=-1).fit(X, y)
        print(f"  ExtraTrees {side}: {len(X)} フレーム、{time.perf_counter() - t0:.1f} s")
    ridge, alphas = {}, {}
    for h in M.HORIZON_GRID:                                           # 0.02〜0.30 s の 0.01 刻み
        ridge[h], alphas[h], n = fit_ridge(sessions, idx_by_sess, h)
    print(f"  Ridge AR {M.HORIZON_GRID[0]:.2f}〜{M.HORIZON_GRID[-1]:.2f} s 先 ({len(ridge)} 本、標準化 + α 選択): "
          f"{n} サンプル ({M.HORIZON_GRID[-1]:.2f} s 先のとき)")
    print("    選んだ α: " + "  ".join(f"{h:g} s→{alphas[h]:g}" for h in M.HORIZONS))
    return et, ridge, alphas


def evaluate(sessions, trees: int, leaf: int, dataset: str) -> None:
    """レポートと同じ 20 ブロックのインターリーブ分割 (位相 1、境界 0.5 s) で学習 → test の精度。"""
    from dataset import block_split

    tr, te = [], []
    for s in sessions:
        fs = 1.0 / float(np.median(np.diff(s["t"])))
        a, b = block_split(len(s["t"]), n_blocks=20, gap=int(round(0.5 * fs)), phase=1)
        tr.append(a[s["ok"][a]])
        te.append(b[s["ok"][b]])
    et, ridge, _ = fit(sessions, tr, trees, leaf)
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
            X, _, cur, k, Pnow = ar_samples(s, h, i)
            Ps.append(M.from_repr(cur + ridge[h].predict(X), Pnow, M.RIDGE_INPUT))
            ys.append(np.vstack([np.interp(s["t"][k] + h, s["t"], s["y"][:, j]) for j in range(4)]).T)
        P, y = np.vstack(Ps), np.vstack(ys)
        z = z_of(P)
        rmse = np.sqrt(np.mean((z - y) ** 2, axis=0))
        ph = np.array([M.phase_for_impedance(v) for v in z]) - np.array([M.phase_for_impedance(v) for v in y])
        label = "現在" if h == 0.0 else f"{h:g} s 先"
        print("  " + f"{label:>8} " + "".join(f"{v:9.2f}" for v in rmse) + f"{np.sqrt(np.mean(ph ** 2)):9.2f}")
    print("  (位相° = 推定した Z から出した位相と、実測の Z から出した位相の差の RMSE)\n")


# --------------------------------------------------------------------------- 8 マーカー (rpy_same4)
def is_8marker(dataset: str) -> bool:
    """report1120_run.DATASETS にある学習データ (1120・1121) は 8 マーカー → rpy_same4 のモデル。"""
    sys.path.insert(0, str(IMP_DIR / "src"))
    import report1120_run as R
    return dataset in R.DATASETS


def load8(dataset: str):
    """8 マーカーの学習データを report1120_run と同じ前処理で読む (despike、marker_qc、急変ゲート)。

    返す: sessions (dict のリスト。role = "main" (本編) か "external" (別収録))、急変のしきい値。
    しきい値は report1120_run と同じく本編の |dZ/dt| の p99 (marker_qc で外したフレームを除く)。
    """
    sys.path.insert(0, str(IMP_DIR / "src"))
    import report1120_run as R
    import report1121_rpy as RP
    from dataset import steady_mask

    R.set_dataset(dataset)
    assert tuple(R.MARKERS8) == M.MARKERS8, "report1120_run.MARKERS8 と imp_phase_model.MARKERS8 の並びが違う"
    raw = [(R.Session(n, p), "main") for n, p in R.SESSIONS] + [(R.Session(n, p), "external") for n, p in R.EXTERNAL]
    r = np.concatenate([s.rate()[~(s.bad[1:] | s.bad[:-1])] for s, role in raw if role == "main"])
    thresh = float(np.percentile(r, 99))
    out = []
    for s, role in raw:
        P = np.hstack([s.P[m] for m in M.MARKERS8])
        ref, names, _ = RP.features(s.P)                       # レポートの特徴量と同じになっているか
        for side in M.SIDES:
            got = M.rpy_side(P, side)
            want = ref[:, [names.index(f"{side}_{n}") for n in M.RPY_NAMES]]
            fin = np.isfinite(want).all(axis=1)
            assert np.allclose(got[fin], want[fin], atol=1e-6), f"{side} の 4 個がレポートと違う"
        vis = ~np.isnan(P).any(axis=1) & ~s.bad
        ok = steady_mask(s.y, s.t, thresh=thresh) & vis
        out.append(dict(name=s.name, role=role, t=s.t, P=P, y=s.y, vis=vis, ok=ok))
        print(f"  [{s.name}] ({role}) {s.n} フレーム、学習に使う {ok.sum()} (急変 {thresh:.0f} Ω/s 超・隠れ・marker_qc {s.bad.sum()} を除外)")
    return out, thresh


def ar_samples_rpy(sess: dict, h: float, idx: np.ndarray, side: str):
    """rpy_same4 の Ridge の入力 (その側の 4 個の角度の過去 7 点の変位) と、h 秒先の角度の変化。

    姿勢 (座標) を時刻で 0.02 s 刻みに補間してから角度にする (実行時の forecast_impedance と同じ順)。
    返す: X、先の変化、いまの角度、使ったフレーム。
    """
    t, P = sess["t"][sess["vis"]], sess["P"][sess["vis"]]
    now = sess["t"][idx]
    keep = M.spans_ok(t, now - (M.TAPS - 1) * M.STEP, now + h)
    now = now[keep]
    Pq = M.resample(t, P, M.history_times(now).ravel()).reshape(len(now), M.TAPS, -1)
    X, cur = M.rpy_ar_design(M.rpy_side(Pq, side))
    fut = M.rpy_side(M.resample(t, P, now + h), side)
    return X, M.rpy_target(fut, cur), cur, idx[keep]


def fit_ridge_rpy(sessions, idx_by_sess, h: float, side: str):
    """fit_ridge と同じ (標準化 + α を学習フレームの後ろ 1 割で選ぶ) を、その側の 4 個の角度で。"""
    def stack(idxs):
        parts = [ar_samples_rpy(s, h, i, side) for s, i in zip(sessions, idxs)]
        return np.vstack([p[0] for p in parts]), np.vstack([p[1] for p in parts])

    cut = [i[:int(len(i) * (1 - RIDGE_VAL))] for i in idx_by_sess]
    val = [i[int(len(i) * (1 - RIDGE_VAL)):] for i in idx_by_sess]
    Xf, Yf = stack(cut)
    Xv, Yv = stack(val)
    score = {a: float(np.mean((make_ridge(a).fit(Xf, Yf).predict(Xv) - Yv) ** 2 / Yv.var(axis=0)))
             for a in RIDGE_ALPHAS}
    alpha = min(score, key=score.get)
    X, Y = stack(idx_by_sess)
    return make_ridge(alpha).fit(X, Y), alpha, len(X)


def fit_rpy(sessions, idx_by_sess, trees: int, leaf: int):
    """左右のコイルそれぞれ、同じ側の腕の 4 個だけで ExtraTrees と Ridge AR を学習する。"""
    from sklearn.ensemble import ExtraTreesRegressor

    et = {}
    for side, cols in M.SIDES.items():
        X = np.vstack([M.rpy_side(s["P"][i], side) for s, i in zip(sessions, idx_by_sess)])
        y = np.vstack([s["y"][i][:, list(cols)] for s, i in zip(sessions, idx_by_sess)])
        t0 = time.perf_counter()
        et[side] = ExtraTreesRegressor(n_estimators=trees, min_samples_leaf=leaf, random_state=SEED,
                                         n_jobs=-1).fit(X, y)
        print(f"  ExtraTrees {side} ({side} の腕の {', '.join(M.RPY_NAMES)}): {len(X)} フレーム、"
              f"{time.perf_counter() - t0:.1f} s")
    ridge, alphas = {}, {}
    for h in M.HORIZON_GRID:
        ridge[h], alphas[h] = {}, {}
        for side in M.SIDES:
            ridge[h][side], alphas[h][side], n = fit_ridge_rpy(sessions, idx_by_sess, h, side)
    print(f"  Ridge AR {M.HORIZON_GRID[0]:.2f}〜{M.HORIZON_GRID[-1]:.2f} s 先 ({len(ridge)} × 2 本、角度、標準化 + α 選択): "
          f"{n} サンプル ({M.HORIZON_GRID[-1]:.2f} s 先のとき)")
    print("    選んだ α: " + "  ".join(f"{h:g} s→L {alphas[h]['L']:g} / R {alphas[h]['R']:g}" for h in M.HORIZONS))
    return et, ridge, alphas


def evaluate_rpy(sessions, trees: int, leaf: int) -> None:
    """本編 (2123) の全フレームで学習 → 別収録 (2101) で評価 (report_20261122_imu_features の 7 章と同じ組み方)。"""
    train = [s for s in sessions if s["role"] == "main"]
    test = [s for s in sessions if s["role"] == "external"]
    if not test:
        print("  別収録が無いので評価しない")
        return
    et, ridge, _ = fit_rpy(train, [np.where(s["ok"])[0] for s in train], trees, leaf)
    B = dict(kind="rpy_same4", markers=M.MARKERS8, et=et, ridge=ridge)
    print(f"\n  {', '.join(s['name'] for s in test)} での精度 (レポート 7 章: ET 500 本・葉 1 で 100 ms 先の R² "
          f"0.159 / 0.883 / 0.405 / 0.899、位相 5.32°)")
    print("  " + "".join(f"{n:>16}" for n in ("", *M.TARGETS, "位相°")))
    for h in (0.0, *M.HORIZONS):
        zs, ys = [], []
        for s in test:
            i = np.where(s["ok"])[0]
            if h == 0.0:
                zs.append(np.hstack([et[sd].predict(M.rpy_side(s["P"][i], sd)) for sd in M.SIDES]))
                ys.append(s["y"][i])
                continue
            z = np.empty((0, 4))
            parts = {}
            for sd in M.SIDES:
                X, _, cur, k = ar_samples_rpy(s, h, i, sd)
                a = cur + ridge[h][sd].predict(X)
                a[:, M.RPY_CIRC] = M.wrap180(a[:, M.RPY_CIRC])
                parts[sd] = (et[sd].predict(a), k)
            k = parts["L"][1]
            assert np.array_equal(k, parts["R"][1])
            zs.append(np.hstack([parts["L"][0], parts["R"][0]]))
            ys.append(np.vstack([np.interp(s["t"][k] + h, s["t"], s["y"][:, j]) for j in range(4)]).T)
        z, y = np.vstack(zs), np.vstack(ys)
        rmse = np.sqrt(np.mean((z - y) ** 2, axis=0))
        r2 = 1 - np.sum((z - y) ** 2, axis=0) / np.sum((y - y.mean(0)) ** 2, axis=0)
        ph = (np.array([M.phase_for_impedance(v) for v in z]) - np.array([M.phase_for_impedance(v) for v in y]) + 180) % 360 - 180
        label = "現在" if h == 0.0 else f"{h:g} s 先"
        print("  " + f"{label:>15} " + "".join(f"{f'{e:.2f}Ω R²{q:.3f}':>16}" for e, q in zip(rmse, r2))
              + f"{np.sqrt(np.mean(ph ** 2)):16.2f}")
    print()


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
    if is_8marker(a.dataset):
        sessions, thresh = load8(a.dataset)
        if a.eval:
            print("\n評価 (本編で学習 → 別収録でテスト)")
            evaluate_rpy(sessions, a.trees, a.leaf)
        print("全フレーム (本編 + 別収録) で学習")
        et, ridge, alphas = fit_rpy(sessions, [np.where(s["ok"])[0] for s in sessions], a.trees, a.leaf)
        import joblib
        a.out.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(dict(
            dataset=a.dataset, sessions=[s["name"] for s in sessions], trained=datetime.now().isoformat(timespec="seconds"),
            steady_thresh=thresh, trees=a.trees, leaf=a.leaf, kind="rpy_same4", markers=M.MARKERS8,
            features=M.RPY_NAMES, targets=M.TARGETS, taps=M.TAPS, step=M.STEP,
            et=et, ridge=ridge, ridge_alpha=alphas, ridge_scaled=True, ridge_input="rpy_same4",
        ), a.out)
        print(f"  -> {a.out} ({a.out.stat().st_size / 1e6:.0f} MB)")
        return 0
    sessions, thresh = load(a.dataset)
    if a.eval:
        print("\n評価 (20 ブロック分割)")
        evaluate(sessions, a.trees, a.leaf, a.dataset)
    print("全フレームで学習")
    et, ridge, alphas = fit(sessions, [np.where(s["ok"])[0] for s in sessions], a.trees, a.leaf)

    import joblib
    a.out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(dict(
        dataset=a.dataset, sessions=[s["name"] for s in sessions], trained=datetime.now().isoformat(timespec="seconds"),
        steady_thresh=thresh, trees=a.trees, leaf=a.leaf, markers=M.MARKERS, targets=M.TARGETS, taps=M.TAPS, step=M.STEP,
        et=et, ridge=ridge, ridge_alpha=alphas, ridge_scaled=True, ridge_input=M.RIDGE_INPUT,
    ), a.out)
    print(f"  -> {a.out} ({a.out.stat().st_size / 1e6:.0f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
