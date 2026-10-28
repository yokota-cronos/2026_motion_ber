#!/usr/bin/env python3
"""fix_lr.py -- Motive のラベルを左右逆に付けたまま撮った記録の、マーカー名の左右を入れ替える。

    uv run fix_lr.py <記録フォルダ>          # 記録全体で R_* <-> L_* の名前を入れ替える
    uv run fix_lr.py <記録フォルダ> --flips  # 途中で左右が入れ替わった所を見つけて、その後ろの値を戻す
    uv run fix_lr.py <記録フォルダ> --undo   # 元に戻す

--flips: Motive は記録の途中で左右のラベルを入れ替えることがある (2026-10-28 14:31 の記録で、
43.2 s から右と左が入れ替わった)。入れ替わった瞬間は、右のマーカーが前のフレームの左の位置に、
左が右の位置に飛ぶ (約 40 cm) ので、対ごと (上腕・肘・前腕) にそれを見つけ、それより後ろの値を
右と左で入れ替えて、記録の始めの割り当てに揃える。motion.csv の値と、それから計算している
ber_motion.csv の収録窓の平均・動いた量を書き換える。始めの割り当て自体が逆なら、そのあと
名前の入れ替え (--flips なし) もかける。

数値は 1 つも変えない。列の名前 (R_joint_X <-> L_joint_X など) と、motion_conditions.json の
marker_names (marker_id の順の名前) だけを入れ替える。marker_id = 2 に「R_upperarm」と付いていたが
実は左腕だった、という間違いをそのまま直す形になる。

  motion.csv / ber_motion.csv   1 行目 (列名) だけ書き換える
  motion_conditions.json        marker_names を入れ替え、lr_fixed に日時を残す
  overlay_calib.json            消す (左右が変わるとカメラの解き直しが要る。次に開いたとき作り直す)

書き換える前のファイルは <記録フォルダ>/_before_fix_lr/ に残す (--undo はそこから戻す)。
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

FILES = ("motion.csv", "ber_motion.csv", "motion_conditions.json")
PAIRS = (("R_upperarm", "L_upperarm"), ("R_joint", "L_joint"), ("R_forearm", "L_forearm"),
         ("R_thigh", "L_thigh"), ("R_knee", "L_knee"), ("R_shin", "L_shin"))
JUMP_M = 0.05                 # これより動いていない対は入れ替わりとみなさない [m]
BACKUP = "_before_fix_lr"


def swap_name(n: str) -> str:
    if n.startswith("R_"):
        return "L_" + n[2:]
    if n.startswith("L_"):
        return "R_" + n[2:]
    return n


def _rewrite_header(path: Path) -> None:
    """CSV の 1 行目の列名だけ左右を入れ替える (2 行目以降はそのまま写す)。"""
    tmp = path.with_suffix(".tmp")
    with open(path, encoding="utf-8", newline="") as src, open(tmp, "w", encoding="utf-8", newline="") as dst:
        head = src.readline()
        end = "\r\n" if head.endswith("\r\n") else "\n"
        dst.write(",".join(swap_name(c) for c in head.rstrip("\r\n").split(",")) + end)
        shutil.copyfileobj(src, dst, 1 << 20)
    tmp.replace(path)


def fix(folder: Path) -> str:
    cond = json.loads((folder / "motion_conditions.json").read_text(encoding="utf-8"))
    if cond.get("lr_fixed"):
        raise SystemExit(f"もう入れ替えてあります ({cond['lr_fixed']})。戻すときは --undo")
    bk = folder / BACKUP
    bk.mkdir(exist_ok=True)
    for f in FILES:                                   # 先に --flips をかけていても、いちばん元のものを残す
        if (folder / f).exists() and not (bk / f).exists():
            shutil.copy2(folder / f, bk / f)
    for f in ("motion.csv", "ber_motion.csv"):
        if (folder / f).exists():
            _rewrite_header(folder / f)
    cond["marker_names"] = [swap_name(n) for n in cond["marker_names"]]
    cond["lr_fixed"] = datetime.now().isoformat(timespec="seconds")
    cond["lr_fixed_note"] = "撮ったときの Motive のラベルが左右逆だったので、R_* と L_* の名前を入れ替えた (数値はそのまま)"
    (folder / "motion_conditions.json").write_text(json.dumps(cond, ensure_ascii=False, indent=2), encoding="utf-8")
    (folder / "overlay_calib.json").unlink(missing_ok=True)
    return "左右を入れ替えました: " + ", ".join(cond["marker_names"])


def _read_motion(folder: Path):
    import csv
    with open(folder / "motion.csv", encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f))
    return rows[0], rows[1:]


def detect_flips(folder: Path) -> dict[str, list[float]]:
    """対ごとに、右と左が入れ替わった時刻 [s] (motion.csv の t)。"""
    import numpy as np
    head, rows = _read_motion(folder)
    t = np.array([float(r[0]) for r in rows])
    out = {}
    for a, b in PAIRS:
        if f"{a}_X" not in head:
            continue
        ia, ib = head.index(f"{a}_X"), head.index(f"{b}_X")
        val = lambda r, i: None if r[i] == "" else np.array([float(r[i]), float(r[i + 1]), float(r[i + 2])])
        la = lb = None
        flips = []
        for k, r in enumerate(rows):
            A, B = val(r, ia), val(r, ib)
            if A is None or B is None:
                continue
            if la is not None:
                keep = np.linalg.norm(A - la) + np.linalg.norm(B - lb)
                swap = np.linalg.norm(A - lb) + np.linalg.norm(B - la)
                if swap < 0.5 * keep and keep > JUMP_M:
                    flips.append(float(t[k]))
            la, lb = A, B
        out[a[2:]] = flips
    return out


def repair_flips(folder: Path) -> str:
    """途中で入れ替わった所より後ろの値を、対ごとに右と左で入れ替えて、記録の始めの割り当てに揃える。"""
    import csv
    import numpy as np
    flips = detect_flips(folder)
    if not any(flips.values()):
        return "途中での入れ替わりは見つかりませんでした"
    cond = json.loads((folder / "motion_conditions.json").read_text(encoding="utf-8"))
    if cond.get("lr_flips_repaired"):
        raise SystemExit(f"もう直してあります ({cond['lr_flips_repaired']})。戻すときは --undo")
    bk = folder / BACKUP
    bk.mkdir(exist_ok=True)
    for f in FILES:
        if (folder / f).exists() and not (bk / f).exists():
            shutil.copy2(folder / f, bk / f)
    head, rows = _read_motion(folder)
    t = np.array([float(r[0]) for r in rows])
    for (a, b) in PAIRS:
        ts = flips.get(a[2:]) or []
        if not ts:
            continue
        ia, ib = head.index(f"{a}_X"), head.index(f"{b}_X")
        # 入れ替わりの回数が奇数の区間 (= 始めと逆の割り当て) の行で、右と左の値を入れ替える
        odd = (np.searchsorted(np.array(ts), t, side="right") % 2) == 1
        for k in np.nonzero(odd)[0]:
            r = rows[k]
            r[ia:ia + 3], r[ib:ib + 3] = r[ib:ib + 3], r[ia:ia + 3]
    with open(folder / "motion.csv", "w", encoding="utf-8", newline="") as f:
        f.write(",".join(head) + "\n")
        f.writelines(",".join(r) + "\n" for r in rows)
    _recompute_ber_motion(folder, head, rows)
    cond["lr_flips_repaired"] = dict(at=datetime.now().isoformat(timespec="seconds"),
                                     flips_s={k: [round(x, 3) for x in v] for k, v in flips.items() if v})
    cond["lr_flips_note"] = "Motive が記録の途中で左右のラベルを入れ替えた。入れ替わった後ろの値を、記録の始めの割り当てに戻した"
    (folder / "motion_conditions.json").write_text(json.dumps(cond, ensure_ascii=False, indent=2), encoding="utf-8")
    (folder / "overlay_calib.json").unlink(missing_ok=True)
    return "途中の入れ替わりを直しました: " + ", ".join(f"{k} {v}" for k, v in flips.items() if v)


def _recompute_ber_motion(folder: Path, mhead, mrows) -> None:
    """ber_motion.csv の収録窓のマーカーの平均 (_X/_Y/_Z) と動いた量 (_dev) を、直した motion.csv から計算し直す。"""
    import csv
    import warnings
    import numpy as np
    path = folder / "ber_motion.csv"
    if not path.exists():
        return
    with open(path, encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f))
    head, body = rows[0], rows[1:]
    names = [h[:-4] for h in head if h.endswith("_dev")]
    t = np.array([float(r[0]) for r in mrows])
    cols = {n: mhead.index(f"{n}_X") for n in names}
    xyz = np.array([[[float(r[cols[n] + a]) if r[cols[n] + a] != "" else np.nan for a in range(3)] for n in names]
                    for r in mrows]).reshape(len(mrows), len(names), 3)
    i0, i1 = head.index("t_start"), head.index("t_end")
    for r in body:
        sel = (t >= float(r[i0])) & (t <= float(r[i1]))
        if not sel.any():
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            mean = np.nanmean(xyz[sel], axis=0)
            dev = np.nanmax(np.linalg.norm(xyz[sel] - mean, axis=2), axis=0)
        for j, n in enumerate(names):
            for a, ax in enumerate("XYZ"):
                v = mean[j, a]
                r[head.index(f"{n}_{ax}")] = "" if v != v else f"{v:.6f}"
            r[head.index(f"{n}_dev")] = "" if dev[j] != dev[j] else f"{dev[j]:.6f}"
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(",".join(head) + "\n")
        f.writelines(",".join(r) + "\n" for r in body)


def undo(folder: Path) -> str:
    bk = folder / BACKUP
    if not bk.is_dir():
        raise SystemExit("戻すためのファイルがありません (_before_fix_lr/)")
    for f in FILES:
        if (bk / f).exists():
            shutil.copy2(bk / f, folder / f)
    shutil.rmtree(bk)
    (folder / "overlay_calib.json").unlink(missing_ok=True)
    return "元に戻しました"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", help="記録フォルダ (フォルダ名だけなら ~/nfc_vhbr_captures/log/ の中)")
    ap.add_argument("--flips", action="store_true", help="途中で左右が入れ替わった所を直す")
    ap.add_argument("--undo", action="store_true")
    a = ap.parse_args()
    d = Path(a.folder)
    if not d.is_absolute() and not d.exists():
        d = Path.home() / "nfc_vhbr_captures" / "log" / a.folder
    if not (d / "motion_conditions.json").exists():
        print(f"記録フォルダではありません: {d}")
        return 1
    print(undo(d) if a.undo else repair_flips(d) if a.flips else fix(d))
    return 0


if __name__ == "__main__":
    sys.exit(main())
