#!/usr/bin/env python3
"""fix_lr.py -- Motive のラベルを左右逆に付けたまま撮った記録の、マーカー名の左右を入れ替える。

    uv run fix_lr.py <記録フォルダ>          # R_* <-> L_* を入れ替える
    uv run fix_lr.py <記録フォルダ> --undo   # 元に戻す

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
    for f in FILES:
        if (folder / f).exists():
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
    ap.add_argument("--undo", action="store_true")
    a = ap.parse_args()
    d = Path(a.folder)
    if not d.is_absolute() and not d.exists():
        d = Path.home() / "nfc_vhbr_captures" / "log" / a.folder
    if not (d / "motion_conditions.json").exists():
        print(f"記録フォルダではありません: {d}")
        return 1
    print(undo(d) if a.undo else fix(d))
    return 0


if __name__ == "__main__":
    sys.exit(main())
