#!/usr/bin/env python3
"""phase_control.py -- モーションから W2−W1 の位相差を決める差し込み口。

学習済みモデルでモーション (7 マーカーの座標) からコイルのインピーダンスを推定し、
ブリッジの打ち消しが最も深くなる位相差をリーダの AWG に入れる (実装は imp_phase_model.py)。

  motion_ber_gui.py --phase-model my_model.py

で、そのファイルの `predict_phase(motion) -> float | None` を **収録 1 回ごとに 1 回**呼ぶ。
返すのは「この姿勢で打ち消しが最も深くなる位相」[°] だが、**GUI はその絶対値ではなく変化ぶんだけを使う**:

    送る位相 = 初期位相 + (いま返した値 − 基準を取ったときに返した値)

初期位相は、基準を取ったときの画面の位相 (手で合わせた値)。基準を取り直すのは、記録の開始・
自動調整を入れたとき・先読みを変えたとき・手で位相を変えたとき。モデルの絶対値の誤差
(直列抵抗、回路の位相ずれ、学習データと台のコイルの Z の違い) は初期位相の中で打ち消される。
送る位相は GUI で位相を入れたのと同じ経路 (STATE.live + COMMANDS) で、次の収録の前に
`Bridge.set_phase()` として反映される。None を返せば位相は変えない。

motion (dict):
    names     マーカー名 (配信順) のタプル。画面の「マーカーの左右が逆」(--swap-lr) が入っていれば
              R_* と L_* を入れ替えた名前 (= 本人の左右)。記録の CSV は Motive のラベルのまま
    xyz       (7, 3) ndarray [m]。直近の収録窓の平均。見えていなかったマーカーは NaN
    valid     収録窓のうち全マーカーが見えていたフレームの割合 (0..1)
    t         収録窓の中央 [s、記録開始から]
    history   (t (k,), xyz (k, 7, 3)): 呼んだ時点までの直近 HISTORY_SEC 秒の全フレーム (古い順、t は記録開始からの秒)。
              先読みするモデル用。次の収録は、このあと位相を送る待ち (set_phase の 50 ms) などを挟んで始まる
    horizon   先読み [s] = 「history の最後のフレーム → 次の収録窓の中央」の時間。
              画面の「先読み」が「自動 (収録窓に合わせる)」なら、直近の収録で実測した時間から決める (毎回少し変わる)。
              100 / 200 ms を選んでいればその値。使わないモデルは無視してよい
    phase_deg いまの位相の指定値 [°]
    level_dbm いまのレベル [dBm]
    div       データレート FC/div

戻り値は位相 [°] か None。dict(phase=..., ほかの値...) を返すと、ほかの値も記録フォルダの
phase_pred.csv と画面に出る (推定したインピーダンスなど)。
モジュールに setup() があれば、読み込んだときに 1 回呼ぶ (重いモデルの読み込みなど)。

守ること:
  * 範囲は 90〜270° (GUI と同じ)。初期位相 + 補正がこれを外れたら、その回は位相を変えない。
  * 呼ばれる頻度は収録の頻度 (1 秒に 1〜2 回) まで。それ以上速く位相を叩くと
    リーダが固まる (2026-10-09、無間隔 68 回で電源の入れ直しが要った)。
  * 位相の「指定値」と「実測」には +3.9° のずれがある (175.53 指定で実測 180.04°)。
    モデルが実測の位相を出すなら、ここで指定値に直すこと。
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

PHASE_MIN, PHASE_MAX = 90.0, 270.0
HORIZONS = (0.1, 0.2)                     # 画面で選べる固定の先読み [s]
HORIZON_AUTO = "auto"                     # 先読みを次の収録窓の中央に合わせる (既定)
# 自動のときにモデルが受け付ける先読みの刻み [s]。imp_phase_model の Ridge はこの全部を学習しておく
HORIZON_GRID = tuple(round(0.02 + 0.01 * i, 2) for i in range(29))      # 0.02〜0.30 s
# 自動のとき: 「位相を決めた (predict_phase を呼んだ) 時刻 → 次の収録窓の中央」の初期値 [s]。
# 記録の最初だけ使い、位相を送った収録が 1 回あれば実測 (直近 LEAD_KEEP 回の中央値) に替わる。
# 2026-11-12 19:31 の記録では、送ってから収録開始まで中央値 103 ms (p10 94・p90 133) + 窓 50 ms の半分
LEAD_DEFAULT = 0.13
LEAD_KEEP = 10


def parse_horizon(v):
    """画面・コマンドラインの先読み ("auto" / 0.1 / 0.2) → HORIZON_AUTO か float。"""
    if str(v).strip().lower() == HORIZON_AUTO:
        return HORIZON_AUTO
    h = float(v)
    if h not in HORIZONS:
        raise ValueError(f"先読みは {HORIZON_AUTO} か {HORIZONS} s のどれか")
    return h
HISTORY_SEC = 0.3                         # motion["history"] に渡す長さ [s]
PHASE_OFFSET_DEG = 175.53 - 180.04        # 実測 → 指定値 (README「実測で決めた値」)


def load(path: str | None):
    """predict_phase を持つモジュールを読む。path が None なら None。"""
    if not path:
        return None
    p = Path(path).resolve()
    spec = importlib.util.spec_from_file_location("phase_model", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not callable(getattr(mod, "predict_phase", None)):
        raise RuntimeError(f"{p} に predict_phase(motion) がありません")
    if callable(getattr(mod, "setup", None)):
        mod.setup()                     # 重いモデルの読み込みなど (任意)
    return mod


def swap_lr(name: str) -> str:
    """R_* ↔ L_* (Motive のラベルが左右逆のとき、モデルに渡す名前を本人の左右に直す)。"""
    if name.startswith(("R_", "L_")):
        return ("L_" if name[0] == "R" else "R_") + name[2:]
    return name


def clamp(v) -> float | None:
    if v is None:
        return None
    v = float(v)
    if not PHASE_MIN <= v <= PHASE_MAX or v != v:
        return None
    return round(v, 2)
