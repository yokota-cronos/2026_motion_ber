"""phase_model_example.py -- `--phase-model` に渡すファイルのひな形。

    uv run motion_ber_gui.py --phase-model phase_model_example.py

いまは位相を変えない (None を返す)。学習済みモデルができたら、たとえば:

    import joblib
    from phase_control import PHASE_OFFSET_DEG
    MODEL = joblib.load("impedance_model.joblib")       # 座標 21 次元 → Z (R, X)

    def predict_phase(motion):
        if motion["valid"] < 0.9:                         # 隠れたマーカーが多い窓は使わない
            return None
        z = MODEL.predict(motion["xyz"].reshape(1, -1))[0]
        best = phase_for_impedance(z)                     # Z → 打ち消しが最も深い位相 (実測)
        return best + PHASE_OFFSET_DEG                    # 実測 → AWG の指定値

引数 motion の中身は phase_control.py の docstring を参照。
"""


def predict_phase(motion):
    return None
