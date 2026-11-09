"""video/ の CSV と図を作る: 位相の自動補正なし (20261102_160447) とあり (20261101_203535) の BER、
自動補正ありの記録のインピーダンスの推定。

    uv run python video/make_figs.py              # CSV から図を描く (このリポジトリだけで動く)
    uv run python video/make_figs.py --from-logs  # 記録フォルダ (~/nfc_vhbr_captures/log) から CSV を作り直してから描く

CSV:
  ber_captures.csv  収録 1 回 (0.05 秒、約 99 フレーム) = 1 行。2 つの記録をまとめたもの
      recording  記録フォルダ名
      correction off = 位相 183° 固定 / on = インピーダンスの推定で位相を自動補正
      cap        収録の番号 (1 から)
      t_start / t_end  記録の始まりからの時刻 [s]
      phase_deg  その収録で使った W2−W1 位相の指定値 [度]
      phase_src  manual = 手で入れた値、auto = 自動補正で送った値
      frames / expected  取れたフレーム数 / 送ったフレーム数
      bits / errors / ber  比べたビット数 / 誤りビット数 / errors ÷ bits
  imp_20261101.csv  自動補正ありの記録の、収録 1 回ごとの推定 (phase_pred.csv を整理したもの)
      cap        推定に使った収録の番号。送った位相は次の収録 (cap + 1) で使われる
      t          推定した時刻 [s]
      L_Z_R / L_Z_X / R_Z_R / R_Z_X  推定した左右のコイルのインピーダンス (実部・虚部) [Ω]。左右は本人の左右
      phase_model  推定から求めた、打ち消しが最も深くなる位相 (指定値に換算) [度]
      base       初期位相 (手で合わせた値) [度]
      delta      補正量 = phase_model − 記録の始まりの phase_model [度]
      phase_sent 送った位相 = base + delta [度]、sent = 1 なら実際に送った (前回と同じなら 0)
      ber_next   その位相を使った次の収録の BER
"""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"
LOG = Path.home() / "nfc_vhbr_captures" / "log"
REC = {"off": "20261102_160447_FC8_600bit_+10dBm_183deg",     # 位相 183° 固定
       "on": "20261101_203535_FC8_600bit_+10dBm_183deg"}      # インピーダンスの推定で自動補正
LABEL = {"off": "補正なし（183° 固定）", "on": "自動補正あり"}
FLOOR = 1e-5                      # これより下 (誤り 0 個) は 1e-5 の線上に白抜きの下向き三角で描く
# 色: 既定パレットの順 (1 青、2 橙)。系列は記録 (補正なし / あり) と、左右のコイル (左 / 右)
C1, C2 = "#2a78d6", "#eb6834"
INK, SUB, GRID = "#0b0b0b", "#52514e", "#e6e6e3"

_have = {f.name for f in font_manager.fontManager.ttflist}
_jp = [n for n in ("Yu Gothic", "Meiryo", "Hiragino Sans", "Noto Sans CJK JP", "IPAexGothic") if n in _have]
plt.rcParams.update({"font.family": _jp[:1] or ["sans-serif"], "axes.unicode_minus": False,
                     "axes.edgecolor": SUB, "axes.labelcolor": INK, "xtick.color": SUB, "ytick.color": SUB,
                     "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 150})


# ---------------------------------------------------------------- 記録から CSV を作る
def build_csv() -> None:
    rows = []
    for key, name in REC.items():
        for r in csv.DictReader(open(LOG / name / "ber_motion.csv", encoding="utf-8")):
            rows.append(dict(recording=name, correction=key, cap=int(r["cap"]), t_start=float(r["t_start"]),
                             t_end=float(r["t_end"]), phase_deg=float(r["phase"]), phase_src=r["phase_src"],
                             frames=int(r["frames"]), expected=int(r["expected"]), bits=int(r["bits"]),
                             errors=int(r["errors"]), ber=float(r["ber"]) if r["bits"] != "0" else ""))
    with open(HERE / "ber_captures.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    on = {r["cap"]: r for r in rows if r["correction"] == "on"}
    out = []
    for r in csv.DictReader(open(LOG / REC["on"] / "phase_pred.csv", encoding="utf-8")):
        k = int(r["cap"])
        nxt = on.get(k + 1)
        out.append(dict(cap=k, t=float(r["t"]), L_Z_R=float(r["L_Z_R"]), L_Z_X=float(r["L_Z_X"]),
                        R_Z_R=float(r["R_Z_R"]), R_Z_X=float(r["R_Z_X"]), phase_model=round(float(r["phase_model"]), 3),
                        base=float(r["base"]), delta=float(r["delta"]), phase_sent=float(r["phase"]), sent=int(r["sent"]),
                        ber_next=nxt["ber"] if nxt else ""))
    with open(HERE / "imp_20261101.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    cond = {k: json.load(open(LOG / n / "conditions.json", encoding="utf-8")) for k, n in REC.items()}
    print("CSV を作り直しました:", {k: (c["level_dbm"], c["div"], c["seconds"]) for k, c in cond.items()})


def load(name):
    rows = list(csv.DictReader(open(HERE / name, encoding="utf-8")))
    for r in rows:
        for k, v in r.items():
            try:
                r[k] = float(v)
            except ValueError:
                pass
    return rows


def ber_marks(ax, t, ber, color, label):
    """BER の点。誤り 0 個は 1e-5 の線上に白抜きの下向き三角。"""
    t, ber = np.asarray(t), np.asarray(ber, float)
    pos = ber > 0
    ax.plot(t[pos], ber[pos], "o", ms=5, color=color, label=label, zorder=3)
    ax.plot(t[~pos], np.full((~pos).sum(), FLOOR), "v", ms=7, mfc="white", mec=color, mew=1.4, zorder=3)


def style_ber_axis(ax):
    ax.set_yscale("log")
    ax.set_ylim(FLOOR / 2, 1)
    ax.axhline(FLOOR, color=SUB, lw=0.8, ls=":", zorder=1)
    ax.grid(True, which="major", color=GRID, lw=0.8)
    ax.set_ylabel("BER（収録 1 回ごと）")


# ---------------------------------------------------------------- 図
def fig_timeline(rows):
    fig, axes = plt.subplots(2, 1, figsize=(13, 7.2), sharey=True)
    for ax, key, color in zip(axes, ("off", "on"), (C1, C2)):
        rr = [r for r in rows if r["correction"] == key]
        t = [(r["t_start"] + r["t_end"]) / 2 for r in rr]
        ber_marks(ax, t, [r["ber"] for r in rr], color, LABEL[key])
        tot = sum(r["errors"] for r in rr) / sum(r["bits"] for r in rr)
        ax.axhline(tot, color=color, lw=1.6, ls="--", zorder=2)
        ax.text(1.0, tot, f" 全体 {tot:.1e}", transform=ax.get_yaxis_transform(), va="center", ha="left",
                color=INK, fontsize=10)
        style_ber_axis(ax)
        ax.set_title(f"{LABEL[key]}　{rr[0]['recording'][:15]}（{len(rr)} 収録）", loc="left", fontsize=12, color=INK)
        ax.set_xlim(0, max(r["t_end"] for r in rr) * 1.01)
    axes[-1].set_xlabel("記録の始まりからの時刻 [s]")
    fig.text(0.01, 0.005, "白抜きの下向き三角 = その収録で誤り 0 個（1e-5 の線上に描く）。破線 = 記録全体の BER。",
             color=SUB, fontsize=9)
    fig.tight_layout(rect=(0, 0.02, 0.97, 1))
    fig.savefig(FIG / "ber_timeline.png")
    plt.close(fig)


def fig_distribution(rows):
    """収録ごとの BER の分布 (累積)。誤り 0 個の収録は左端 (1e-5 以下) に積む。"""
    fig, ax = plt.subplots(figsize=(10, 6))
    for key, color in (("off", C1), ("on", C2)):
        ber = np.sort(np.array([r["ber"] for r in rows if r["correction"] == key], float))
        x = np.maximum(ber, FLOOR)
        y = np.arange(1, x.size + 1) / x.size
        ax.step(np.concatenate([[FLOOR / 2], x]), np.concatenate([[0], y]), where="post", color=color, lw=2,
                label=f"{LABEL[key]}（{x.size} 収録、誤り 0 個 {int((ber == 0).sum())}）")
        med = np.median(ber)
        ax.plot(max(med, FLOOR), 0.5, "o", ms=8, color=color, mec="white", mew=2, zorder=4)
        ax.annotate(f"中央値 {med:.1e}" if med > 0 else "中央値 0", (max(med, FLOOR), 0.5), xytext=(10, -16) if key == "on" else (10, -20),
                    textcoords="offset points", color=INK, fontsize=10)
    ax.set_xscale("log")
    ax.set_xlim(FLOOR / 2, 1)
    ax.set_ylim(0, 1.02)
    ax.axvline(FLOOR, color=SUB, lw=0.8, ls=":")
    ax.text(FLOOR, 1.02, " ← 誤り 0 個", color=SUB, fontsize=9, va="bottom")
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_xlabel("BER（収録 1 回ごと）")
    ax.set_ylabel("この BER 以下だった収録の割合")
    ax.legend(loc="upper left", frameon=False, bbox_to_anchor=(0, 0.95))
    fig.tight_layout()
    fig.savefig(FIG / "ber_distribution.png")
    plt.close(fig)


def fig_impedance(imp, rows):
    fig, axes = plt.subplots(4, 1, figsize=(13, 11), sharex=True,
                             gridspec_kw=dict(height_ratios=[1, 1, 1, 1.1]))
    t = np.array([r["t"] for r in imp])
    for ax, part, unit in ((axes[0], "R", "実部 R [Ω]"), (axes[1], "X", "虚部 X [Ω]")):
        for side, color, name in (("L", C1, "左"), ("R", C2, "右")):
            y = np.array([r[f"{side}_Z_{part}"] for r in imp])
            ax.plot(t, y, color=color, lw=1.8, label=f"{name}のコイル")
            ax.text(t[-1], y[-1], f" {name}", color=INK, fontsize=10, va="center")
        ax.set_ylabel(unit)
        ax.grid(True, color=GRID, lw=0.8)
    axes[0].legend(loc="upper left", frameon=False, ncol=2)
    axes[0].set_title("インピーダンスの推定（モーションから、0.1 s 先）", loc="left", fontsize=12, color=INK)
    ax = axes[2]
    ps = np.array([r["phase_sent"] for r in imp])
    ax.step(t, ps, where="post", color=INK, lw=1.8, label="送った位相（指定値）")
    ax.axhline(imp[0]["base"], color=SUB, lw=1, ls="--")
    ax.text(t[0], imp[0]["base"], f"初期位相 {imp[0]['base']:g}°", color=SUB, fontsize=9, va="bottom")
    ax.set_ylabel("W2−W1 位相 [度]")
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_title("送った位相（初期位相 + 推定の変化ぶん。次の収録で使う）", loc="left", fontsize=12, color=INK)
    ax = axes[3]
    rr = [r for r in rows if r["correction"] == "on"]
    ber_marks(ax, [(r["t_start"] + r["t_end"]) / 2 for r in rr], [r["ber"] for r in rr], C2, None)
    style_ber_axis(ax)
    ax.set_title("BER（収録 1 回ごと。白抜きの下向き三角 = 誤り 0 個）", loc="left", fontsize=12, color=INK)
    ax.set_xlabel("記録の始まりからの時刻 [s]")
    axes[0].set_xlim(0, max(t.max(), max(r["t_end"] for r in rr)) * 1.04)
    fig.tight_layout()
    fig.savefig(FIG / "imp_20261101.png")
    plt.close(fig)


def summary(rows, imp):
    out = {}
    for key in ("off", "on"):
        rr = [r for r in rows if r["correction"] == key]
        ber = np.array([r["ber"] for r in rr], float)
        bits = sum(r["bits"] for r in rr)
        out[key] = dict(caps=len(rr), seconds=round(max(r["t_end"] for r in rr), 1), bits=int(bits),
                        errors=int(sum(r["errors"] for r in rr)), ber=sum(r["errors"] for r in rr) / bits,
                        zero=int((ber == 0).sum()), median=float(np.median(ber)), p90=float(np.percentile(ber, 90)),
                        phase=(min(r["phase_deg"] for r in rr), max(r["phase_deg"] for r in rr)))
    z = {k: np.array([r[k] for r in imp]) for k in ("L_Z_R", "L_Z_X", "R_Z_R", "R_Z_X", "delta", "phase_sent")}
    out["imp"] = {k: (float(v.mean()), float(v.min()), float(v.max()), float(v.std())) for k, v in z.items()}
    out["imp"]["sent"] = int(sum(r["sent"] for r in imp))
    # 補正の大きさと次の収録の BER
    d = np.array([abs(r["delta"]) for r in imp if r["ber_next"] != ""])
    b = np.array([r["ber_next"] for r in imp if r["ber_next"] != ""], float)
    big = d >= np.median(d)
    out["delta_split"] = dict(median_abs_delta=float(np.median(d)), ber_small=float(b[~big].mean()), ber_big=float(b[big].mean()))
    json.dump(out, open(HERE / "summary.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-logs", action="store_true", help="記録フォルダから CSV を作り直す")
    a = ap.parse_args()
    if a.from_logs:
        build_csv()
    FIG.mkdir(exist_ok=True)
    rows, imp = load("ber_captures.csv"), load("imp_20261101.csv")
    fig_timeline(rows)
    fig_distribution(rows)
    fig_impedance(imp, rows)
    print(json.dumps(summary(rows, imp), ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
