#!/usr/bin/env python3
"""初期位相のスイープ: 動いている motion_ber_gui に位相を順に送り、位相ごとの BER・SNR・打ち消しの残りを比べる。

    uv run phase_sweep.py run                         # 170〜200° を 2° 刻み、各 8 回、上り → 下りの 2 往路
    uv run phase_sweep.py run --start 160 --stop 206 --step 3 --caps 10 --passes 1
    uv run phase_sweep.py analyze                     # いちばん新しい記録を集計 (記録フォルダ名の先頭も渡せる)

run の前に GUI で: 測定を開始して記録中にする・「位相の自動調整」を切る・(SNR も見るなら)「生データも記録」を入れる。
位相は収録の直前に入るので、変えた直後の収録から新しい位相 (念のため最初の --settle 回は数えない)。
2 往路 (上り・下り) にするのは、時間による変化 (2026-11-01 は記録の途中で SNR が 22 → 14 dB に下がった) と
位相の効きを分けるため。終わったら元の位相に戻す。

analyze は位相ごとに: BER (誤り / ビット)、収録ごとの BER の中央値、eye、q、peak。生データがあれば
  SNR    = 3 値化する前の信号を同じ局発・フィルタに通した判定点の値 (理想 ±1) のばらつきから [dB]
  carrier = 無変調区間の搬送波の大きさ [mV] (打ち消しの残り。小さいほど深い)
を位相ごとに数本ずつ測る (1 本 数秒)。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
LOG = Path.home() / "nfc_vhbr_captures" / "log"


# --------------------------------------------------------------------------- run
def _get(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=5) as r:
        return json.loads(r.read())


def _post(url: str, body: dict) -> dict:
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=5) as r:
        return json.loads(r.read())


def run(a) -> None:
    base = a.url.rstrip("/")
    s = _get(base + "/api/motion")
    if s.get("auto_phase"):
        raise SystemExit("「位相の自動調整」が入っています。切ってから実行してください (入っていると手の位相が初期位相になるだけ)")
    if not s.get("logging"):
        print("注意: 記録していません (記録フォルダに残りません)。測定を開始して記録中にしてください")
    st = _get(base + "/api/state")
    live = st.get("live", {})
    orig = float(live.get("phase_deg", 183.0))
    if not live.get("record_raw"):
        print("注意: 生データを記録していません (analyze の SNR・打ち消しの残りは出ません。BER・eye・peak は出ます)")
    n = int(round((a.stop - a.start) / a.step)) + 1
    up = [round(a.start + a.step * i, 2) for i in range(n)]
    order = []
    for p in range(a.passes):
        seq = up if p % 2 == 0 else up[::-1]
        order += seq[1:] if order else seq            # 折り返しの点は 1 回だけ (続けて同じ位相だと集計で 1 つの塊になる)
    per = 0.9                                         # 収録 1 回 ≈ 0.9 s
    print(f"記録先: {s.get('log_dir') or '—'}  元の位相 {orig}°")
    print(f"{len(order)} 点 × {a.caps} 回 (+ 捨てる {a.settle} 回)。目安 {len(order) * (a.caps + a.settle) * per / 60:.1f} 分。"
          f"直立のまま動かないでください。Ctrl+C で止める (元の位相に戻す)")

    def last_cap():
        c = _get(base + "/api/motion").get("corr") or []
        return (c[-1]["cap"], c[-1].get("phase")) if c else (0, None)

    try:
        for i, ph in enumerate(order):
            _post(base + "/api/live/set", {"phase_deg": ph})
            c0, _ = last_cap()
            t0 = time.time()
            while True:
                time.sleep(0.3)
                c, p = last_cap()
                done = c - c0 - a.settle
                if done >= a.caps and p is not None and abs(p - ph) < 0.01:
                    break
                if time.time() - t0 > 60 + a.caps * 3:
                    raise SystemExit(f"{ph}° で収録が進みません (測定が止まっていませんか)")
            print(f"  {i + 1:3d}/{len(order)}  {ph:6.1f}°  収録 #{c0 + 1}〜#{c}")
    except KeyboardInterrupt:
        print("止めました")
    finally:
        _post(base + "/api/live/set", {"phase_deg": orig})
        print(f"位相を {orig}° に戻しました。集計: uv run phase_sweep.py analyze {s.get('log_dir', '')[:15]}")


# --------------------------------------------------------------------------- analyze
def _ber_modules():
    ber = Path(os.environ.get("NFC_BER_DIR", HERE.parent / "2026_nfc_vhbr_ber"))
    sys.path[:0] = [str(ber / "instruments"), str(ber)]
    import fast_demod as fd
    from gui_view import SKIP
    return fd, SKIP


def _raw_metrics(path: Path, lg: dict, fd, SKIP, max_frames: int = 80) -> dict:
    """生データ 1 本: 判定点の SNR (3 値化の前) と、無変調区間の搬送波の大きさ。"""
    import numpy as np
    from scipy.signal import lfilter
    J = json.loads(Path(str(path) + ".json").read_text())
    div = int(J["frame"]["div"])
    decim = fd.decim_for(div)
    fs = float(J["sample_rate_hz"])
    x = fd.fast_ddc(np.fromfile(path, dtype="<i2"), fs, float(J["volts_per_lsb"]), float(J.get("offset_v", 0.0)),
                    div=div, decim=decim, fc=float(J.get("carrier_hz", fd.FC)))[SKIP:]
    x = np.ascontiguousarray(x, np.complex64)
    rx = fd.FastReceiver(fs / decim, div=div, payload=int(J["frame"]["payload"]), frame_period_s=float(J["frame"]["frame_s"]))
    rx.P.rxfilter = fd.rx_filter_mask(lg.get("rx_filter") or "ic")
    rx.P.agc_div = float(lg.get("agc_div") or 3.5)
    rx.P.agc_mode = 1 if int(lg.get("agc_auto") or 1) else 0
    r = rx.process(x, detail=True)
    ok = np.nonzero(r["status"] == 0)[0][:max_frames]
    lo = rx.lo[:, 0].astype(np.float64) + 1j * rx.lo[:, 1]
    err, car = [], []
    for k in ok:
        s0 = int(r["starts"][k])
        w = x[s0:s0 + rx.L]
        if w.size < rx.L:
            continue
        car.append(abs(w[:rx.P.nb].mean()))
        g = rx.stages(w, int(r["sidx"][k]), int(r["phase"][k]))
        sl = lfilter(rx.h.astype(np.float64), 1.0, g["d"] * lo[:g["d"].size])

        def at(pos):
            i0 = np.clip(np.floor(pos).astype(int), 0, sl.size - 2)
            f = pos - i0
            return sl[i0] * (1 - f) + sl[i0 + 1] * f
        sp = (int(r["phase"][k]) + rx.nsps * (int(r["sidx"][k]) + np.arange(fd.SYNC.size))) * rx.P.step
        pk = np.sum(fd.SYNC * at(sp))
        y = (at(g["pay_pos"]) * np.conj(pk) * fd.SYNC.size / abs(pk) ** 2).real
        err.append(y - (2.0 * rx.ref - 1))
    e = np.concatenate(err) if err else np.array([np.nan])
    return dict(snr_db=float(10 * np.log10(1 / np.var(e))), carrier_mv=float(np.median(car) * 1e3) if car else np.nan)


def analyze(a) -> None:
    import numpy as np
    import pandas as pd
    if a.rec:
        D = next(LOG.glob(a.rec + "*"))
    else:
        D = max((d for d in LOG.iterdir() if (d / "ber_motion.csv").exists()), key=lambda d: d.name)
    b = pd.read_csv(D / "ber_motion.csv")
    lg = pd.read_csv(D / "ber_log.csv")
    if len(lg) != len(b):
        raise SystemExit(f"{D.name}: ber_log と ber_motion の行数が違います")
    for c in ("peak", "raw_file", "rx_filter", "agc_div", "agc_auto"):
        b[c] = lg[c].values
    b = b[(b.bits > 0) & (b.phase_src == "manual")].copy()
    b["ber"] = b.errors / b.bits
    b["ph"] = b.phase.round(2)
    # 往路: 位相が変わった順の塊ごとに番号。最初の a.settle 回は捨てる
    b["blk"] = (b.ph != b.ph.shift()).cumsum()
    b["k"] = b.groupby("blk").cumcount()
    b = b[b.k >= a.settle]
    bl = b.groupby("blk").agg(ph=("ph", "first"), t=("t_start", "first")).sort_values("t")
    # 上り / 下り: その塊の位相が前の塊より大きいか
    bl["dir"] = np.where(bl.ph.diff().fillna(1) > 0, "上り", "下り")
    b = b.merge(bl[["dir"]], left_on="blk", right_index=True)
    fd = SKIP = None
    if a.raw and b.raw_file.notna().any():
        fd, SKIP = _ber_modules()
    rows = []
    for (ph, dr), x in b.groupby(["ph", "dir"]):
        row = dict(phase=ph, dir=dr, n=len(x), ber=x.errors.sum() / x.bits.sum(), ber_med=x.ber.median(),
                   gt1e3=(x.ber > 1e-3).mean() * 100, eye=x.eye.median(), q=x.q.median(), peak=x.peak.median(),
                   t=x.t_start.min())
        if fd is not None:
            xs = x[x.raw_file.notna()]
            pick = xs.iloc[np.linspace(0, len(xs) - 1, min(a.raw, len(xs))).astype(int)] if len(xs) else xs
            m = [_raw_metrics(D / r.raw_file, r.to_dict(), fd, SKIP) for _, r in pick.iterrows()]
            if m:
                row.update(snr_db=float(np.median([v["snr_db"] for v in m])), carrier_mv=float(np.median([v["carrier_mv"] for v in m])))
        rows.append(row)
        print(f"  {ph:6.1f}° {dr}  {len(x):3d} 回  BER {row['ber']:.1e}  中央値 {row['ber_med']:.1e}  eye {row['eye']:.3f}  peak {row['peak']:.1f}"
              + (f"  SNR {row['snr_db']:5.1f} dB  搬送波 {row['carrier_mv']:5.1f} mV" if "snr_db" in row else ""), flush=True)
    r = pd.DataFrame(rows).sort_values(["dir", "phase"])
    out = HERE / f"phase_sweep_{D.name[:15]}"
    r.to_csv(out.with_suffix(".csv"), index=False)
    print(f"\n記録: {D.name}")
    for dr, x in r.groupby("dir"):
        best = x.loc[x.ber.idxmin()]
        line = f"  {dr}: BER 最小 {best.phase:g}° ({best.ber:.1e})・peak 最小 {x.loc[x.peak.idxmin()].phase:g}°"
        if "snr_db" in x and x.snr_db.notna().any():
            line += f"・SNR 最大 {x.loc[x.snr_db.idxmax()].phase:g}° ({x.snr_db.max():.1f} dB)・搬送波 最小 {x.loc[x.carrier_mv.idxmin()].phase:g}°"
        print(line)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = "Meiryo"
    panels = [("ber", "BER（誤り / ビット）", True), ("eye", "eye", False), ("peak", "peak [% FS]", False)]
    if "snr_db" in r and r.snr_db.notna().any():
        panels += [("snr_db", "SNR（3 値化の前）[dB]", False), ("carrier_mv", "打ち消しの残りの搬送波 [mV]", False)]
    fig, axs = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 3.6))
    style = {"上り": dict(color="#2f64c0", marker="o"), "下り": dict(color="#d9822b", marker="s")}
    for ax, (c, lab, log) in zip(axs, panels):
        for dr, x in r.groupby("dir"):
            ax.plot(x.phase, x[c].clip(lower=1e-6) if log else x[c], lw=2, ms=5, label=dr, **style[dr])
        ax.set_xlabel("位相差の指定値 [°]")
        ax.set_title(lab, fontsize=11)
        if log:
            ax.set_yscale("log")
        ax.grid(alpha=.3)
    axs[0].legend(frameon=False)
    fig.suptitle(f"初期位相のスイープ（{D.name[:15]}、補正なし）", fontsize=12)
    fig.tight_layout()
    fig.savefig(out.with_suffix(".png"), dpi=120)
    print("表:", out.with_suffix(".csv"), "\n図:", out.with_suffix(".png"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="動いている GUI に位相を順に送る")
    r.add_argument("--url", default="http://127.0.0.1:8766")
    r.add_argument("--start", type=float, default=170.0)
    r.add_argument("--stop", type=float, default=200.0)
    r.add_argument("--step", type=float, default=2.0)
    r.add_argument("--caps", type=int, default=8, help="1 点あたりの収録回数")
    r.add_argument("--settle", type=int, default=1, help="位相を変えた直後に数えない回数")
    r.add_argument("--passes", type=int, default=2, help="往路の数 (上り・下りを交互)")
    z = sub.add_parser("analyze", help="記録を位相ごとに集計")
    z.add_argument("rec", nargs="?", help="記録フォルダ名の先頭 (省くといちばん新しい記録)")
    z.add_argument("--settle", type=int, default=1)
    z.add_argument("--raw", type=int, default=2, help="位相・往路ごとに SNR を測る生データの本数 (0 で測らない)")
    a = ap.parse_args()
    run(a) if a.cmd == "run" else analyze(a)


if __name__ == "__main__":
    main()
