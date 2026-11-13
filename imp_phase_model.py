"""imp_phase_model.py -- モーション → 先のインピーダンス → W2−W1 位相差 (`--phase-model` に渡すモデル)。

    uv run train_imp_model.py                                  # 最初に 1 回。models/imp_1111.joblib を作る
    uv run motion_ber_gui.py --phase-model imp_phase_model.py  # 画面の「位相の自動調整」で入り切り

手法は ../2026_motion_to_meander_imp/report_20261013_dataset1013ver1.pdf の 4 節 (モーションの先読み) と同じ
(学習データは 1111。report_20261111_dataset1111.pdf も 4 節が同じ手順):

  1. 過去 8 点 (0.02 s 間隔 = 0.14 s 分) のマーカー位置から、h 秒先のマーカー位置を Ridge で予測 (Ridge AR)
  2. 予測した姿勢から片側 6 特徴量を作り、左右それぞれの ExtraTrees で h 秒先のインピーダンスを推定
  3. 左右のインピーダンスから、搬送波の打ち消しが最も深くなる位相差を計算する (phase_for_impedance)

GUI はこの位相の**変化ぶんだけ**を、手で合わせた初期位相に足す (phase_control.py)。なので下の式の
定数項 (PHASE_OFFSET_DEG、180°) は効かず、効くのは左右の arg Z の差が姿勢でどれだけ動くかだけ。

h は motion["horizon"]。既定 (画面の「先読み」が自動) では、GUI が実測した「最後のモーション → 次の収録窓の中央」
(2026-11-12 の記録で約 0.14 s)。Ridge は 0.02〜0.30 s を 0.01 s 刻みで学習しておき、いちばん近いものを使う。
固定の 0.1 s / 0.2 s も選べる。レポートの L_Z_X の RMSE は、現在推定 / 0.1 s 先 /
0.2 s 先で 1013ver1 が 1.49 / 1.74 / 2.36 Ω、1111 が 0.95 / 1.23 / 2.03 Ω。

**レポートとの違い:** レポートは「過去 8 点」「h 秒先」をフレーム数で数えている (VNA の掃引落ちで 1111 は 14.8 %、
1013ver1 は 16.8 % の刻みが欠けているので、0.2 s 指定で実効 0.23〜0.24 s 程度)。ここではライブで使う都合上、**時刻で数える**
(学習時も実行時も、マーカーを時刻で線形補間して 0.02 s 刻みに並べ直す)。

位相の決め方 (phase_for_impedance):
  リーダは W1 → コイル A、W2 → コイル B の電流を OPA836 の加算点で足す。搬送波が消える条件は
      V1 / Z_A + V2 / Z_B = 0    →    V2 / V1 = −Z_B / Z_A
  なので **実測の位相差 = 180° + arg Z_B − arg Z_A**。AWG の指定値には phase_control.PHASE_OFFSET_DEG を足す
  (Z_A = Z_B なら 175.49° で、README の「175.53° 指定で実測 180.04°」と合う)。
  * W1 (コイル A、タグ側) は左の体のコイル (2026-10-28 に確認)。加算点までの直列抵抗は **未確認** (SERIES_OHM。
    相対で使っても、抵抗があると補正の大きさが変わる)。
  * 振幅比 |Z_B| / |Z_A| もずれるが、ここでは位相だけ直す (振幅比は GUI の値のまま)。
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from phase_control import HORIZON_GRID, HORIZONS, PHASE_OFFSET_DEG  # 学習する先読み = HORIZON_GRID (0.02〜0.30 s、0.01 刻み)

HERE = Path(__file__).resolve().parent
DATASET = "1111"        # 学習データ (../2026_motion_to_meander_imp の report0923_common.DATASETS のキー)
MODEL_PATH = Path(os.environ.get("IMP_MODEL", HERE / "models" / f"imp_{DATASET}.joblib"))

W1_COIL = "L"           # リーダ W1 が駆動するコイル (= コイル A、タグ側)。左の体のコイル (2026-10-28 に確認)
SERIES_OHM = 0.0        # コイルから加算点までの直列抵抗 [Ω]。Z に足してから位相を出す。**要確認**

# 学習データ (dataset.MARKERS) の列順。motion["names"] (Motive の配信順) からこの順に並べ直す
MARKERS = ("chest", "R_upperarm", "R_joint", "R_forearm", "L_upperarm", "L_joint", "L_forearm")
TARGETS = ("L_Z_R", "L_Z_X", "R_Z_R", "R_Z_X")
SIDES = {"L": (0, 1), "R": (2, 3)}
TAPS = 8                # レポートと同じ「過去 8 点」
STEP = 0.02             # その間隔 [s]。8 点で 0.14 s
MAX_GAP = 0.05          # 使う区間の中にこれより長い抜け (マーカーが隠れた・VNA の掃引落ち) があれば使わない [s]


# --------------------------------------------------------------------------- 特徴量・先読み (学習と共用)
def _angle(a, b, c):
    v1, v2 = a - b, c - b
    cos = np.sum(v1 * v2, axis=1) / (np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1) + 1e-9)
    return np.arccos(np.clip(cos, -1.0, 1.0))


def side_features(P: np.ndarray, side: str) -> np.ndarray:
    """(N, 21) のマーカー座標 (MARKERS 順、Z が上) → 片側 6 特徴量 (N, 6)。

    ../2026_motion_to_meander_imp/src/dataset.py の _build_features_side と同じ量・同じ順
    (肘角度、肩角度、肘の胸基準の高さ・水平距離、前腕の同じ 2 つ)。train_imp_model.py が一致を確かめる。
    """
    def m(name):
        k = MARKERS.index(name) * 3
        return P[:, k:k + 3]

    chest, upper, joint, fore = m("chest"), m(f"{side}_upperarm"), m(f"{side}_joint"), m(f"{side}_forearm")
    cols = [_angle(upper, joint, fore), _angle(chest, upper, joint)]
    for arr in (joint, fore):
        rel = arr - chest
        cols += [rel[:, 2], np.linalg.norm(rel[:, :2], axis=1)]
    return np.stack(cols, axis=1).astype(np.float32)


def resample(t: np.ndarray, P: np.ndarray, tq: np.ndarray) -> np.ndarray:
    """時刻 t の (N, D) を時刻 tq に線形補間する。"""
    return np.stack([np.interp(tq, t, P[:, j]) for j in range(P.shape[1])], axis=-1)


def history_times(now) -> np.ndarray:
    """過去 8 点の時刻 (古い順、最後が now)。now は配列でもよい → (..., TAPS)。"""
    return np.asarray(now, float)[..., None] - STEP * np.arange(TAPS - 1, -1, -1)


def ar_design(Pq: np.ndarray):
    """(M, TAPS, 21) の履歴 → Ridge の入力 (過去 7 点の「いまからの変位」) と、いまの位置。"""
    cur = Pq[:, -1]
    return (Pq[:, :-1] - cur[:, None]).reshape(len(Pq), -1), cur


def spans_ok(t: np.ndarray, a, b) -> np.ndarray:
    """区間 [a, b] がサンプル t の範囲に収まり、中に MAX_GAP より長い抜けが無いか。"""
    a, b = np.atleast_1d(a), np.atleast_1d(b)
    g = np.where(np.diff(t) > MAX_GAP)[0]
    n_before = np.searchsorted(t[g + 1], a, side="right")      # a より前に終わっている抜け
    n_until = np.searchsorted(t[g], b, side="left")             # b より前に始まっている抜け
    return (a >= t[0]) & (b <= t[-1]) & (n_until - n_before == 0)


def phase_for_impedance(z) -> float:
    """[L_Z_R, L_Z_X, R_Z_R, R_Z_X] → 搬送波が消える **実測の** W2−W1 位相差 [°]。"""
    zl = complex(z[0] + SERIES_OHM, z[1])
    zr = complex(z[2] + SERIES_OHM, z[3])
    za, zb = (zl, zr) if W1_COIL == "L" else (zr, zl)
    return 180.0 + float(np.degrees(np.angle(zb / za)))


# --------------------------------------------------------------------------- 実行時
BUNDLE = None


def setup() -> None:
    """学習済みモデルを読む。phase_control.load が起動時に呼ぶ (初回の収録を遅らせない)。"""
    global BUNDLE
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"{MODEL_PATH} がありません。先に `uv run train_imp_model.py` で作ってください")
    import joblib
    b = joblib.load(MODEL_PATH)
    for m in b["et"].values():
        m.n_jobs = 1                # 1 点ずつしか推定しないので、スレッドを立てる方が遅い
    missing = [h for h in HORIZON_GRID if h not in b["ridge"]]
    if missing:
        raise RuntimeError(f"{MODEL_PATH} は先読み {sorted(b['ridge'])} s しか学習していません "
                           f"(収録窓に合わせるには 0.02〜0.30 s の 0.01 刻みが要る)。"
                           f"`uv run train_imp_model.py` で作り直してください")
    BUNDLE = b


def ridge_horizon(h: float) -> float:
    """先読み h [s] に使う Ridge のキー (学習してある先読みのうち、いちばん近いもの)。

    0.01 s 刻みより細かくは合わせない (その間の姿勢の差は 0.01 s × 腕の速さ 1 m/s ≈ 5 mm 以下)。
    刻みの外 (0.3 s より先など) は端の値にする。0.1 / 0.2 s しか無い古いモデルでは誤差が大きいので、
    setup() の時点で学習し直しを促す。
    """
    if BUNDLE is None:
        setup()
    keys = np.array(sorted(BUNDLE["ridge"]))
    return float(keys[np.argmin(np.abs(keys - h))])


def forecast_impedance(t: np.ndarray, xyz: np.ndarray, names, h: float):
    """直近のモーション → h 秒先の [L_Z_R, L_Z_X, R_Z_R, R_Z_X] [Ω]。履歴が足りなければ None。

    t: (k,) [s]、xyz: (k, マーカー数, 3) [m]、names: xyz の並び (Motive の配信順)。
    h は学習してある先読みのうちいちばん近いものに丸める (ridge_horizon)。
    """
    if BUNDLE is None:
        setup()
    h = ridge_horizon(h)
    missing = [m for m in MARKERS if m not in names]
    if missing:
        raise ValueError(f"マーカー {missing} がありません (計測部位を上半身にする)")
    P = np.asarray(xyz, float)[:, [list(names).index(m) for m in MARKERS], :].reshape(len(t), -1)
    ok = ~np.isnan(P).any(axis=1)                 # 1 点でも隠れていたフレームは使わない
    t = np.asarray(t, float)
    if not ok.any() or t[-1] - t[ok][-1] > MAX_GAP:
        return None                               # 直近が隠れている (古い姿勢を「いま」として先読みしない)
    t, P = t[ok], P[ok]
    tq = history_times(t[-1])
    if not spans_ok(t, tq[0], tq[-1])[0]:
        return None
    X, cur = ar_design(resample(t, P, tq)[None])
    Pf = cur + BUNDLE["ridge"][h].predict(X)
    z = np.empty(4)
    for side, cols in SIDES.items():
        z[list(cols)] = BUNDLE["et"][side].predict(side_features(Pf, side))[0]
    return z


def predict_phase(motion):
    """phase_control.py の約束どおり。位相に加えて、記録・表示用に推定したインピーダンスも返す。"""
    h = float(motion.get("horizon", HORIZONS[0]))
    t, xyz = motion["history"]
    z = forecast_impedance(t, xyz, motion["names"], h)
    if z is None:
        return None                               # マーカーが隠れていた → 位相は変えない
    actual = phase_for_impedance(z)
    return dict(phase=actual + PHASE_OFFSET_DEG, phase_actual=round(actual, 2), horizon_model=ridge_horizon(h),
                **{k: round(float(v), 3) for k, v in zip(TARGETS, z)})
