"""../2026_nfc_vhbr_ber/results の JSON から、ber_plots.ipynb が使う CSV を作り直す。

    uv run --no-project python data/make_csv.py     # 2026_motion_ber で実行

1 行 = 1 つの (系列, データレート, 位相, AWG レベル) の BER。列は
    series        整合系列 (e05, e20, ...)
    div           サブキャリア FC/div
    rate_mbps     データレート [Mbps]
    phase_deg     W2-W1 位相の指定値 [度]
    role          default = 既定位相 183°（補正前）、corrected = 補正後、sweep = 位相のずれ、
                  original = 差し替える前の値（e40 202° の 0〜+10 dBm、点線で描く参考）、
                  175.53 = 旧既定値（e20 の灰色の破線）
    date          測った日
    level_dbm     AWG の設定レベル [dBm]（+10 dBm = 1 V peak）
    bits, errors  比べたビット数と誤りビット数。ber = errors / bits
    frames, expected  取れたフレーム数と送ったフレーム数（frames < 100 は参考値）
    note          差し替えなどの注記
"""
import csv
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
RES = HERE.parent.parent / "2026_nfc_vhbr_ber" / "results"
RATE = {32: 0.42, 16: 0.85, 8: 1.70, 4: 3.39}
COLS = ["series", "div", "rate_mbps", "phase_deg", "role", "date", "level_dbm",
        "bits", "errors", "ber", "frames", "expected", "note"]


def load(path, key):
    return sorted(json.load(open(RES / path))[key], key=lambda r: r["level"])


def rows(series, div, phase, role, date, data, note=""):
    """date / note は 1 つの値か、レベルを受けて値を返す関数。"""
    out = []
    for r in data:
        if r["bits"] <= 0:
            continue                      # フレームが 1 枚も取れなかったレベルは載せない
        lv = r["level"]
        out.append({"series": series, "div": div, "rate_mbps": RATE[div], "phase_deg": phase,
                    "role": role, "date": date(lv) if callable(date) else date, "level_dbm": lv,
                    "bits": r["bits"], "errors": r["errors"], "ber": r["errors"] / r["bits"],
                    "frames": r["frames"], "expected": r["expected"],
                    "note": note(lv) if callable(note) else note})
    return out


def patch(base, redo, lo, hi):
    m = {r["level"]: r for r in redo if lo <= r["level"] <= hi}
    return [m.get(r["level"], r) for r in base]


def write(name, data):
    with open(HERE / name, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        w.writerows(data)
    print(f"{name}: {len(data)} rows")


# 既定位相 183° の各系列。e50 / e60 は −20〜+14 dBm を再測定で差し替えたもの。
e50_183 = rows("e50", 8, 183, "default", lambda lv: "2026-10-26" if lv >= -20 else "2026-10-25",
               load("e50_20261025/e50_183_ber_redo_m20to14.json", "8@183"),
               lambda lv: "-20〜+14 dBm は 10/26 の再測定" if lv >= -20 else "")
e60_183 = rows("e60", 8, 183, "default", lambda lv: "2026-10-27" if lv >= -20 else "2026-10-25",
               load("e60_20261025/e60_183_ber_redo_m20to14d.json", "8@183"),
               lambda lv: "-20〜+14 dBm は 10/27（タグ貼り直し後）の再測定" if lv >= -20 else "")
DEFAULT = {
    "e05": rows("e05", 8, 183, "default", "2026-10-22", load("e05_20261022/e05_183_ber.json", "8@183")),
    "e20": rows("e20", 8, 183, "default", "2026-10-22", load("e20_20261022/e20_183_ber.json", "8@183")),
    "e30": rows("e30", 8, 183, "default", "2026-10-22", load("e30_20261022/e30_183_ber.json", "8@183")),
    "e40": rows("e40", 8, 183, "default", "2026-10-23", load("e40_20261023/e40_183_ber.json", "8@183")),
    "e50": e50_183,
    "e60": e60_183,
    "e70": rows("e70", 8, 183, "default", "2026-10-25", load("e70_20261025/e70_183_ber.json", "8@183")),
}

# 図 1: e05、FC/32〜FC/4、183°
write("e05_fcN_183.csv", [r for d in (32, 16, 8, 4)
                          for r in rows("e05", d, 183, "default", "2026-10-22",
                                        load("e05_20261022/e05_183_ber.json", f"{d}@183"))])

# 図 2: FC/8・183° で e05〜e70
write("fc8_183_series.csv", [r for s in DEFAULT.values() for r in s])


def sweep(series, corrected, date, path, phases, extra=()):
    out = list(DEFAULT[series])
    for ph in phases:
        out += rows(series, 8, ph, "corrected" if ph == corrected else "sweep", date, load(path, f"8@{ph}"))
    out += list(extra)
    return out


# 図 3〜8: 系列ごとの位相補正
write("phase_e20.csv", sweep("e20", 187, "2026-10-22", "e20_20261022/e20_fc8_phases.json", (185, 187, 189, 191),
                             rows("e20", 8, 175.53, "175.53", "2026-10-20", load("e20_20261020/e20_ber.json", "8"),
                                  "旧既定値 175.53°")))
write("phase_e30.csv", sweep("e30", 196, "2026-10-23", "e30_20261023/e30_fc8_phases2.json", range(188, 205, 2)))

e40_202 = load("e40_20261023/e40_fc8_phases.json", "8@202")
e40_rows = list(DEFAULT["e40"])
for ph in range(192, 213, 2):
    if ph == 202:
        e40_rows += rows("e40", 8, 202, "corrected", "2026-10-23",
                         patch(e40_202, load("e40_20261023/e40_fc8_202_redo.json", "8@202"), 0, 10),
                         lambda lv: "0〜+10 dBm は 12:37〜の測り直し" if 0 <= lv <= 10 else "")
    else:
        e40_rows += rows("e40", 8, ph, "sweep", "2026-10-23", load("e40_20261023/e40_fc8_phases.json", f"8@{ph}"))
e40_rows += rows("e40", 8, 202, "original", "2026-10-23", [r for r in e40_202 if 0 <= r["level"] <= 10],
                 "1 回目の 0〜+10 dBm（差し替え前）")
write("phase_e40.csv", e40_rows)

e50_rows = list(DEFAULT["e50"])
for ph in range(196, 217, 2):
    data = load("e50_20261025/e50_fc8_phases.json", f"8@{ph}")
    if ph == 208:
        data = patch(data, load("e50_20261025/e50_fc8_208_redo6to12.json", "8@208"), 6, 12)
        e50_rows += rows("e50", 8, 208, "corrected", "2026-10-25", data,
                         lambda lv: "+6〜+12 dBm は再測定" if 6 <= lv <= 12 else "")
    else:
        e50_rows += rows("e50", 8, ph, "sweep", "2026-10-25", data)
write("phase_e50.csv", e50_rows)

write("phase_e60.csv", sweep("e60", 212, "2026-10-27", "e60_20261025/e60_fc8_phases_1027_202redo.json", range(200, 219, 2)))
write("phase_e70.csv", sweep("e70", 216, "2026-10-25", "e70_20261025/e70_fc8_phases.json", range(200, 217, 2)))
