#!/usr/bin/env python3
"""補正量の試験 (画面の「試験: 補正にオフセットを足す」、--offset-test) の記録を集計する。

    python analyze_offset_test.py                 # いちばん新しい試験の記録
    python analyze_offset_test.py 20261120_1530   # 記録フォルダ名の先頭 (複数可。まとめて集計)

自動補正した収録ごとに、モデルの補正量 delta にゲイン gain を掛け、試験のオフセット test_offset を足して撮っている
(phase = 初期位相 + gain × delta + test_offset。gain の列が無い記録は 1)。モデルの補正量の大きさ・向きで分け、オフセットごとに
  BER (その組の誤り / ビット) と peak (入力の振れ [% FS]。小さいほど搬送波の打ち消しが深い) を並べる。
  モデルの補正量が正しければ、どの組でも BER・peak はオフセット 0 で最小。
  最小がオフセット x° にずれていれば、その組では入れた補正が x° ずれている (本当に要る補正 ≈ gain × delta + x)。
  いまのゲインのままで良ければ、どの組でも最小は 0 付近。
  最小が 0 付近なのに peak が「補正量が小さい組」まで下がらなければ、位相だけでは打ち消せない (振幅の比のずれ)。
最小の位置は、peak (と log10 BER) をオフセットの 2 次式で近似した頂点。
最後に、全収録の peak を「peak = c + a (入れた位相の補正 − k × delta)²」で当てはめた、最も打ち消しが深くなるゲイン k を出す。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

LOG = Path.home() / "nfc_vhbr_captures" / "log"
# モデルの補正量 delta [°] の組。負 = 右腕を上げる向き、正 = 左腕を上げる向き (2026-11 の配置で)
DELTA_BINS = [(-90, -20, "補正 −20° より負"), (-20, -8, "補正 −20〜−8°"), (-8, 8, "補正 ±8° 以内"),
              (8, 20, "補正 +8〜+20°"), (20, 90, "補正 +20° より正")]


def find(prefixes: list[str]) -> list[Path]:
    if prefixes:
        return [next(LOG.glob(p + "*")) for p in prefixes]
    for d in sorted(LOG.iterdir(), reverse=True):
        mc = d / "motion_conditions.json"
        if (d / "phase_pred.csv").exists() and mc.exists():
            pp = pd.read_csv(d / "phase_pred.csv", nrows=2000)
            if "test_offset" in pp and (pp.test_offset.fillna(0) != 0).any():
                return [d]
    raise SystemExit(f"{LOG} に試験の記録 (phase_pred.csv の test_offset が 0 でない) がありません")


def load(d: Path) -> pd.DataFrame:
    b = pd.read_csv(d / "ber_motion.csv")
    lg = pd.read_csv(d / "ber_log.csv")
    pp = pd.read_csv(d / "phase_pred.csv")
    if "test_offset" not in pp:
        raise SystemExit(f"{d.name}: phase_pred.csv に test_offset がありません (試験の機能より前の記録)")
    if len(lg) == len(b):
        b["peak"] = lg.peak.to_numpy()
    else:
        b["peak"] = np.nan
    if "gain" not in pp:
        pp["gain"] = 1.0
    x = b.merge(pp[["cap", "delta", "gain", "test_offset", "warmup", "base"]], on="cap")
    x["applied"] = x.gain * x.delta                      # 入れた補正 (試験のオフセットは含まない)
    x = x[(x.phase_src == "auto") & (x.warmup == 0) & (x.bits > 0)].copy()
    x["rec"] = d.name[:15]
    return x


def vertex(off, y, w):
    """y ≈ a off² + b off + c の頂点 (a > 0 のときだけ)。点が 4 つ未満・範囲の外なら None。"""
    if len(off) < 4:
        return None
    a, b_, _ = np.polyfit(off, y, 2, w=np.sqrt(w))
    if a <= 0:
        return None
    v = -b_ / (2 * a)
    return v if off.min() - 4 <= v <= off.max() + 4 else None


def main():
    dirs = find(sys.argv[1:])
    d = pd.concat([load(p) for p in dirs], ignore_index=True)
    offs = sorted(float(o) for o in d.test_offset.unique())
    print("記録:", ", ".join(p.name for p in dirs))
    for p in dirs:
        mc = json.loads((p / "motion_conditions.json").read_text(encoding="utf-8"))
        print(f"  {p.name[:15]}: モデル {mc.get('phase_model')}、試験のオフセット {mc.get('test_offsets')}")
    print(f"自動補正した収録 {len(d)} 回 (基準を取る間を除く)。オフセット {offs}\n")
    head = "".join(f"{o:>+11g}°" for o in offs)
    for lo, hi, lab in DELTA_BINS:
        g = d[(d.delta >= lo) & (d.delta < hi)]
        if len(g) < len(offs):
            continue
        s = g.groupby("test_offset").agg(n=("cap", "size"), err=("errors", "sum"), bits=("bits", "sum"),
                                         peak=("peak", "median")).reindex(offs)
        s["ber"] = s.err / s.bits
        ok = s[s.n >= 3]
        vb = vertex(ok.index.to_numpy(float), np.log10(ok.ber.to_numpy() + 1e-6), ok.n.to_numpy())
        vp = vertex(ok.index.to_numpy(float), ok.peak.to_numpy(), ok.n.to_numpy()) if ok.peak.notna().all() else None
        best = s.ber.idxmin()
        gains = sorted(g.gain.unique())
        print(f"== {lab}（{len(g)} 回、モデルの補正量 中央値 {g.delta.median():+.1f}°・ゲイン {', '.join(f'{v:g}' for v in gains)}"
              f"・入れた補正 中央値 {g.applied.median():+.1f}°）")
        print(f"   オフセット {head}")
        print("   回数       " + "".join(f"{int(v) if v == v else 0:>12d}" for v in s.n))
        print("   BER        " + "".join(f"{v:>12.1e}" if v == v else f"{'-':>12s}" for v in s.ber))
        print("   peak       " + "".join(f"{v:>12.2f}" if v == v else f"{'-':>12s}" for v in s.peak))
        fmt = lambda v: "—（決まらない）" if v is None else f"{v:+.1f}°"
        print(f"   → BER 最小のオフセット {best:+g}°・近似の頂点 BER {fmt(vb)}・peak {fmt(vp)}"
              + ("" if vp is None else f"  ⇒ 本当に要る補正 ≈ {g.applied.median() + vp:+.1f}°"
                 + (f"（モデル {g.delta.median():+.1f}° の {(g.applied.median() + vp) / g.delta.median():.2f} 倍）"
                    if abs(g.delta.median()) >= 8 else "")))
        print()
    x = d[d.peak.notna() & (d.delta.abs() >= 8)]
    if len(x) >= 20:
        tot = x.applied + x.test_offset
        r = least_squares(lambda q: q[0] + q[1] * (tot - q[2] * x.delta) ** 2 - x.peak, [4, 0.01, 1], loss="soft_l1")
        sd = np.sqrt(np.diag(np.linalg.pinv(r.jac.T @ r.jac) * np.median(r.fun ** 2)))
        print(f"全体 (|モデルの補正| ≥ 8°、{len(x)} 回): 打ち消しが最も深くなるゲイン k ≈ {r.x[2]:.2f} ± {sd[2]:.2f}"
              f"（いまのゲイン {', '.join(f'{v:g}' for v in sorted(d.gain.unique()))}）")
        for sgn, lab in ((-1, "負の補正"), (1, "正の補正")):
            y = x[np.sign(x.delta) == sgn]
            if len(y) >= 20:
                t = y.applied + y.test_offset
                r = least_squares(lambda q: q[0] + q[1] * (t - q[2] * y.delta) ** 2 - y.peak, [4, 0.01, 1], loss="soft_l1")
                print(f"  {lab} ({len(y)} 回): k ≈ {r.x[2]:.2f}")
        print()
    out = Path.cwd() / f"offset_test_{dirs[-1].name[:15]}.csv"
    d.to_csv(out, index=False)
    print("収録ごとの表:", out)


if __name__ == "__main__":
    main()
