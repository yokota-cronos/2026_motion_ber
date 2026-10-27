#!/usr/bin/env python3
"""phase_control.py -- モーションから W2−W1 の位相差を決める差し込み口 (**いまは何もしない**)。

先々の構想: 学習済みモデルでモーション (7 マーカーの座標) からコイルのインピーダンスを
推定し、ブリッジの打ち消しが最も深くなる位相差をリーダの AWG に入れる。

  motion_ber_gui.py --phase-model my_model.py

で、そのファイルの `predict_phase(motion) -> float | None` を **収録 1 回ごとに 1 回**呼ぶ。
返した値は GUI で位相を入れたのと同じ経路 (STATE.live + COMMANDS) で、次の収録の前に
`Bridge.set_phase()` として反映される。None を返せば位相は変えない。

motion (dict):
    names     マーカー名 (配信順) のタプル
    xyz       (7, 3) ndarray [m]。直近の収録窓の平均。見えていなかったマーカーは NaN
    valid     収録窓のうち全マーカーが見えていたフレームの割合 (0..1)
    t         収録窓の中央 [s、記録開始から]
    phase_deg いまの位相の指定値 [°]
    level_dbm いまのレベル [dBm]
    div       データレート FC/div

守ること:
  * 範囲は 90〜270° (GUI と同じ)。外れた値は捨てる。
  * 呼ばれる頻度は収録の頻度 (1 秒に 1〜2 回) まで。それ以上速く位相を叩くと
    リーダが固まる (2026-10-09、無間隔 68 回で電源の入れ直しが要った)。
  * 位相の「指定値」と「実測」には +3.9° のずれがある (175.53 指定で実測 180.04°)。
    モデルが実測の位相を出すなら、ここで指定値に直すこと。
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

PHASE_MIN, PHASE_MAX = 90.0, 270.0
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
    return mod


def clamp(v) -> float | None:
    if v is None:
        return None
    v = float(v)
    if not PHASE_MIN <= v <= PHASE_MAX or v != v:
        return None
    return round(v, 2)
