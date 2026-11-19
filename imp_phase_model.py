"""imp_phase_model.py -- モーション → 先のインピーダンス → W2−W1 位相差 (`--phase-model` に渡すモデル)。

    uv run train_imp_model.py                                  # 最初に 1 回。models/imp_1118.joblib を作る
    uv run motion_ber_gui.py --phase-model imp_phase_model.py  # 画面の「位相の自動調整」で入り切り

手法は ../2026_motion_to_meander_imp/report_20261013_dataset1013ver1.pdf の 4 節 (モーションの先読み) と同じ
(学習データは 1118。report_20261111_dataset1111.pdf・report_20261118_dataset1118.pdf も 4 節が同じ手順。
 1118 はマーカー欠落・取り違えのフレーム (marker_qc) を学習から外している):

  1. 過去 8 点 (0.02 s 間隔 = 0.14 s 分) のマーカー位置から、h 秒先のマーカー位置を Ridge で予測 (Ridge AR)。
     Ridge に渡すのは chest に対する相対座標 (ridge_repr。2026-11-19 から。前かがみなどで chest が動いても腕の動きだけを見る)。
     入力と出力は標準化してあり、α は学習時に選んである (train_imp_model.make_ridge。predict は元の単位 [m] で返す)
  2. 予測した姿勢から片側 6 特徴量を作り、左右それぞれの ExtraTrees で h 秒先のインピーダンスを推定
  3. 左右のインピーダンスから、搬送波の打ち消しが最も深くなる位相差を計算する (phase_for_impedance)

GUI はこの位相の**変化ぶんだけ**を、手で合わせた初期位相に足す (phase_control.py)。なので下の式の
定数項 (PHASE_OFFSET_DEG、180°) は効かず、効くのは左右の arg Z の差が姿勢でどれだけ動くかだけ。

h は motion["horizon"]。既定 (画面の「先読み」が自動) では、GUI が実測した「最後のモーション → 次の収録窓の中央」
(2026-11-12 の記録で約 0.14 s)。Ridge は 0.02〜0.30 s を 0.01 s 刻みで学習しておき、いちばん近いものを使う。
固定の 0.1 s / 0.2 s も選べる。レポートの L_Z_X の RMSE は、現在推定 / 0.1 s 先 /
0.2 s 先で 1013ver1 が 1.49 / 1.74 / 2.36 Ω、1111 が 0.95 / 1.23 / 2.03 Ω、1118 が 1.74 / 1.99 / 2.67 Ω。

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
DATASET = "1118"        # 学習データ (../2026_motion_to_meander_imp の report0923_common.DATASETS のキー)
MODEL_PATH = Path(os.environ.get("IMP_MODEL", HERE / "models" / f"imp_{DATASET}.joblib"))
# 2026-11-19 に Ridge を標準化 + α 選択に作り直す前のモデル (標準化なし・α = 1)。それより前の記録を再生するとき、
# 「補正に実際に使った予測」を記録時と同じモデルで出し直すのに使う (無ければ今のモデルで出し直す)
# (そのころの学習データは 1111 なので、DATASET を替えてもこのファイルのまま)
LEGACY_MODEL_PATH = HERE / "models" / "imp_1111_alpha1.joblib"
LEGACY_UNTIL = "2026-11-19"


def load_bundle(path) -> dict:
    """モデルのファイルを読む (setup と違い、BUNDLE には入れない)。"""
    import joblib
    b = joblib.load(path)
    for m in b["et"].values():
        m.n_jobs = 1
    return b

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
    """(M, TAPS, d) の履歴 → Ridge の入力 (過去 7 点の「いまからの変位」) と、いまの値。d は表現の次元 (21 か 18)。"""
    cur = Pq[:, -1]
    return (Pq[:, :-1] - cur[:, None]).reshape(len(Pq), -1), cur


# --------------------------------------------------------------------------- Ridge に渡す表現
# "chest_rel" (2026-11-19 から): chest 以外の 6 マーカーの、chest に対する相対座標 (18 次元)。前かがみ・歩くなどで
#   chest そのものが動いても、腕と胸の位置関係だけを予測する。ExtraTrees の特徴量 (角度・胸基準の高さと水平距離) は
#   平行移動で変わらないので、相対座標の予測だけで足りる。予測した姿勢 (画面・記録用) は「いまの chest の位置」に足して出す
# "abs" (それまで): 7 マーカーの絶対座標 (21 次元)。古いモデル (ridge_input が無い) はこちら
RIDGE_INPUT = "chest_rel"
CHEST = MARKERS.index("chest")
OTHERS = [i for i in range(len(MARKERS)) if i != CHEST]


def ridge_repr(P: np.ndarray, mode: str) -> np.ndarray:
    """(N, 21) の絶対座標 (MARKERS 順) → Ridge の表現 (N, 21 か 18)。"""
    if mode == "abs":
        return P
    Q = P.reshape(len(P), len(MARKERS), 3)
    return (Q[:, OTHERS] - Q[:, [CHEST]]).reshape(len(P), -1)


def from_repr(R: np.ndarray, P_now: np.ndarray, mode: str) -> np.ndarray:
    """Ridge の表現の予測 (N, d) → 絶対座標の姿勢 (N, 21)。chest_rel は、いまの chest の位置 (P_now から) に足す。"""
    if mode == "abs":
        return R
    c = P_now.reshape(len(P_now), len(MARKERS), 3)[:, CHEST]
    Q = np.empty((len(R), len(MARKERS), 3))
    Q[:, CHEST] = c
    Q[:, OTHERS] = R.reshape(len(R), len(OTHERS), 3) + c[:, None]
    return Q.reshape(len(R), -1)


def ridge_forecast(Pq: np.ndarray, h: float, B: dict) -> np.ndarray:
    """(M, TAPS, 21) の絶対座標の履歴 (0.02 s 刻み、最後がいま) → h 秒先の姿勢 (M, 21)。モデルの表現 (ridge_input) に合わせる。"""
    mode = B.get("ridge_input", "abs")
    Rq = ridge_repr(Pq.reshape(-1, Pq.shape[-1]), mode).reshape(len(Pq), Pq.shape[1], -1)
    X, cur = ar_design(Rq)
    return from_repr(cur + B["ridge"][h].predict(X), Pq[:, -1], mode)


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
    if not b.get("ridge_scaled"):
        raise RuntimeError(f"{MODEL_PATH} は Ridge を標準化していない古いモデルです (2026-11-19 から標準化 + α 選択)。"
                           f"`uv run train_imp_model.py` で作り直してください")
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


def forecast_impedance(t: np.ndarray, xyz: np.ndarray, names, h: float, with_pose: bool = False, bundle=None):
    """直近のモーション → h 秒先の [L_Z_R, L_Z_X, R_Z_R, R_Z_X] [Ω]。履歴が足りなければ None。

    t: (k,) [s]、xyz: (k, マーカー数, 3) [m]、names: xyz の並び (Motive の配信順)。
    h は学習してある先読みのうちいちばん近いものに丸める (ridge_horizon)。
    with_pose: True なら (Z, 予測した姿勢 (マーカー数, 3) を names の並びで) を返す (画面に重ねる用)。
    bundle: 使うモデル (省略時は読み込んだモデル)。古い記録で、記録時のモデルの予測を出し直す用
    """
    if BUNDLE is None and bundle is None:
        setup()
    B = BUNDLE if bundle is None else bundle
    keys = np.array(sorted(B["ridge"]))
    h = float(keys[np.argmin(np.abs(keys - h))])
    missing = [m for m in MARKERS if m not in names]
    if missing:
        raise ValueError(f"マーカー {missing} がありません (計測部位を上半身にする)")
    P = np.asarray(xyz, float)[:, [list(names).index(m) for m in MARKERS], :].reshape(len(t), -1)
    ok = ~np.isnan(P).any(axis=1)                 # 1 点でも隠れていたフレームは使わない
    t = np.asarray(t, float)
    if not ok.any() or t[-1] - t[ok][-1] > MAX_GAP:
        return (None, None) if with_pose else None  # 直近が隠れている (古い姿勢を「いま」として先読みしない)
    t, P = t[ok], P[ok]
    tq = history_times(t[-1])
    if not spans_ok(t, tq[0], tq[-1])[0]:
        return (None, None) if with_pose else None
    Pf = ridge_forecast(resample(t, P, tq)[None], h, B)
    z = np.empty(4)
    for side, cols in SIDES.items():
        z[list(cols)] = B["et"][side].predict(side_features(Pf, side))[0]
    if not with_pose:
        return z
    pose = np.full((len(names), 3), np.nan)
    pose[[list(names).index(m) for m in MARKERS]] = Pf.reshape(len(MARKERS), 3)
    return z, pose


def phase_for_pose(names, xyz, bundle=None) -> float | None:
    """1 つの姿勢 (マーカー数, 3) [m] (names の並び) → その姿勢のインピーダンスから出した実測の位相差 [°]。隠れていれば None。"""
    if BUNDLE is None and bundle is None:
        setup()
    B = BUNDLE if bundle is None else bundle
    P = np.asarray(xyz, float)[[list(names).index(m) for m in MARKERS]].reshape(1, -1)
    if np.isnan(P).any():
        return None
    z = np.empty(4)
    for side, cols in SIDES.items():
        z[list(cols)] = B["et"][side].predict(side_features(P, side))[0]
    return phase_for_impedance(z)


def predict_phase(motion):
    """phase_control.py の約束どおり。位相に加えて、記録・表示用に推定したインピーダンスも返す。"""
    h = float(motion.get("horizon", HORIZONS[0]))
    t, xyz = motion["history"]
    z, pose = forecast_impedance(t, xyz, motion["names"], h, with_pose=True)
    if z is None:
        return None                               # マーカーが隠れていた → 位相は変えない
    actual = phase_for_impedance(z)
    # "_" で始まる値は記録 (phase_pred.csv) に書かず、画面だけに使う (予測した姿勢を 3D 表示に重ねる)
    return dict(phase=actual + PHASE_OFFSET_DEG, phase_actual=round(actual, 2), horizon_model=ridge_horizon(h),
                **{k: round(float(v), 3) for k, v in zip(TARGETS, z)}, _pose=pose)


# --------------------------------------------------------------------------- 記録の答え合わせ (再生の画面)
def forecast_series(t: np.ndarray, xyz: np.ndarray, names, h: float) -> dict:
    """記録の全フレームについて「その時刻の h 秒前までの履歴から予測した姿勢」をまとめて出す。

    ライブの forecast_impedance と同じ手順 (過去 8 点を 0.02 s 刻みに補間 → Ridge で h 秒先の変位) を、
    起点を各フレームの時刻 − h にして全フレームぶん回す。実際の姿勢と比べれば、Ridge の先読みの当たり外れが見える。

    t: (N,) [s]、xyz: (N, マーカー数, 3) [m] (names の並び、隠れは NaN)、names: モデルに渡す名前 (本人の左右)。
    返す: xyz_pred (N, マーカー数, 3) (予測できないフレームは NaN)、ok (N,)、
          phase_pred / phase_act (N,) = 予測した姿勢 / 実際の姿勢から出した「実測の」位相差 [°] (出せなければ NaN)。
    """
    if BUNDLE is None:
        setup()
    t = np.asarray(t, float)
    idx = [list(names).index(m) for m in MARKERS]
    P = np.asarray(xyz, float)[:, idx, :].reshape(len(t), -1)
    vis = ~np.isnan(P).any(axis=1)
    out = np.full(P.shape, np.nan)
    ph_pred = np.full(len(t), np.nan)
    ph_act = np.full(len(t), np.nan)
    if vis.sum() >= TAPS:
        tv, Pv = t[vis], P[vis]
        now = t - h
        ok = spans_ok(tv, now - (TAPS - 1) * STEP, now)
        k = np.nonzero(ok)[0]
        if k.size:
            Pq = resample(tv, Pv, history_times(now[k]).ravel()).reshape(k.size, TAPS, -1)
            out[k] = ridge_forecast(Pq, ridge_horizon(h), BUNDLE)

        def phases(Q, sel):
            z = np.empty((sel.size, 4))
            for side, cols in SIDES.items():
                z[:, list(cols)] = BUNDLE["et"][side].predict(side_features(Q[sel], side))
            zl = (z[:, 0] + SERIES_OHM) + 1j * z[:, 1]               # phase_for_impedance と同じ式をまとめて
            zr = (z[:, 2] + SERIES_OHM) + 1j * z[:, 3]
            za, zb = (zl, zr) if W1_COIL == "L" else (zr, zl)
            return 180.0 + np.degrees(np.angle(zb / za))

        if k.size:
            ph_pred[k] = phases(out, k)
        a = np.nonzero(vis)[0]
        if a.size:
            ph_act[a] = phases(P, a)
    else:
        ok = np.zeros(len(t), bool)
    back = np.full(np.asarray(xyz).shape, np.nan)                   # names の並びに戻す
    back[:, idx, :] = out.reshape(len(t), len(MARKERS), 3)
    return dict(xyz_pred=back, ok=ok, phase_pred=ph_pred, phase_act=ph_act, h=float(h), h_model=ridge_horizon(h))
