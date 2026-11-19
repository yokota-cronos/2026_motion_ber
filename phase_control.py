#!/usr/bin/env python3
"""phase_control.py -- モーションから W2−W1 の位相差を決める差し込み口。

学習済みモデルでモーション (7 マーカーの座標) からコイルのインピーダンスを推定し、
ブリッジの打ち消しが最も深くなる位相差をリーダの AWG に入れる (実装は imp_phase_model.py)。

  motion_ber_gui.py --phase-model my_model.py

で、そのファイルの `predict_phase(motion) -> float | None` を **収録 1 回ごとに 1 回、その収録の直前に**呼ぶ。
返すのは「この姿勢で打ち消しが最も深くなる位相」[°] だが、**GUI はその絶対値ではなく変化ぶんだけを使う**:

    送る位相 = 初期位相 + (いま返した値 − 基準を取ったときに返した値)

初期位相は、基準を取ったときの画面の位相 (手で合わせた値)。基準を取り直すのは、記録の開始・
自動調整を入れたとき・先読みを変えたとき・手で位相を変えたとき。モデルの絶対値の誤差
(直列抵抗、回路の位相ずれ、学習データと台のコイルの Z の違い) は初期位相の中で打ち消される。

1 回の流れ (motion_ber_gui.live_worker の収録スレッド):

    predict_phase (約 26 ms) → 位相を変える (API 約 10 ms) → PHASE_SETTLE_S (5 ms) 待つ → 収録窓 (既定 95 ms)
    → USB 転送 (95 ms 分で約 0.8 s。この間は次を撮れない) → 次の周へ。DDC・復調・記録は別スレッドで並行

None を返せば位相は変えない (その収録は直前の位相のまま)。

motion (dict):
    names     マーカー名 (配信順) のタプル。画面の「マーカーの左右が逆」(--swap-lr) が入っていれば
              R_* と L_* を入れ替えた名前 (= 本人の左右)。記録の CSV は Motive のラベルのまま
    xyz       (7, 3) ndarray [m]。いちばん新しいモーションのフレーム。見えていなかったマーカーは NaN
    valid     そのフレームで全マーカーが見えていれば 1.0、そうでなければ 0.0
    t         そのフレームの時刻 [s、記録開始から]
    history   (t (k,), xyz (k, 7, 3)): 呼んだ時点までの直近 HISTORY_SEC 秒の全フレーム (古い順、t は記録開始からの秒)。
              先読みするモデル用
    horizon   固定の先読み [s] (画面の「先読み」、--horizon。既定 HORIZONS[0])。history の最後のフレームから、
              この位相で撮る収録窓の中央までの時間に合わせてある。使わないモデルは無視してよい
    phase_deg いまの位相の指定値 [°]
    level_dbm いまのレベル [dBm]
    div       データレート FC/div

戻り値は位相 [°] か None。dict(phase=..., ほかの値...) を返すと、ほかの値も記録フォルダの
phase_pred.csv と画面に出る (推定したインピーダンスなど)。"_" で始まる値は記録に書かず、画面だけに使う
(_pose = 予測した姿勢 (マーカー数, 3) を names の並びで。3D 表示の「予測を重ねる」が描く)。
モジュールに setup() があれば、読み込んだときに 1 回呼ぶ (重いモデルの読み込みなど)。

守ること:
  * 範囲は 90〜270° (GUI と同じ)。初期位相 + 補正がこれを外れたら、その回は位相を変えない。
  * 呼ばれる頻度は収録の頻度 (1 秒に 1 回ほど) まで。それ以上速く位相を叩くと
    リーダが固まる (2026-10-09、無間隔 68 回で電源の入れ直しが要った)。
  * 位相の「指定値」と「実測」には +3.9° のずれがある (175.53 指定で実測 180.04°)。
    モデルが実測の位相を出すなら、ここで指定値に直すこと。
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

PHASE_MIN, PHASE_MAX = 90.0, 270.0
# 画面で選べる固定の先読み [s]。先頭が既定。0.1 s = 「最後のモーションのフレーム → 収録窓 (95 ms) の中央」:
#   フレームの古さ (Motive で ~10 ms) + predict_phase ~26 ms + 位相の API と PHASE_SETTLE_S ~16 ms + 窓の半分 47.5 ms
#   2026-11-18 に実機で測ると、合成モーション (古さ 4 ms) で中央値 95 ms (p10 88・p90 101)
# 実際の値は phase_pred.csv の t_motion と ber_motion.csv の t_start / t_end で確かめられる
HORIZONS = (0.1, 0.15, 0.2)
# imp_phase_model の Ridge が学習しておく先読みの刻み [s] (0.02〜0.30 s)。HORIZONS はこの中から選ぶ
HORIZON_GRID = tuple(round(0.02 + 0.01 * i, 2) for i in range(29))
PHASE_SETTLE_S = 0.005                    # 自動調整で位相を変えてから収録を始めるまで待つ時間 [s] (README「約 5 ms で移る」)
REF_WARMUP = 5                            # 基準のモデルの位相 = 最初のこの回数の中央値。その間は位相を変えない
CAPTURE_SECONDS = 0.095                  # 既定の収録窓 [s]。位相の 5 ms と合わせて 100 ms


def parse_horizon(v) -> float:
    """画面・コマンドラインの先読み → float (HORIZONS のどれか)。"""
    h = round(float(v), 3)
    if h not in HORIZONS:
        raise ValueError(f"先読みは {HORIZONS} s のどれか")
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
