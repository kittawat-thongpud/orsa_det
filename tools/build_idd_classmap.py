"""Scan IDD VOC annotations, count class frequencies, write class map JSON.

Usage:
  python tools/build_idd_classmap.py --root <IDD_Detection> --split train.txt --out configs/idd_classes.json
"""
from __future__ import annotations
import argparse, json, os, xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path


def iter_ann_paths(root: Path, split: str | None):
    ann_dir = root / "Annotations"
    if split:
        for line in (root / split).read_text().splitlines():
            line = line.strip()
            if line:
                yield ann_dir / f"{line}.xml"
    else:
        for p in ann_dir.rglob("*.xml"):
            yield p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--split", default="train.txt")
    ap.add_argument("--out", default="configs/idd_classes.json")
    args = ap.parse_args()

    root = Path(args.root)
    counter = Counter()
    missing = 0
    for p in iter_ann_paths(root, args.split):
        if not p.exists():
            missing += 1
            continue
        try:
            for obj in ET.parse(p).getroot().findall("object"):
                name = obj.findtext("name")
                if name:
                    counter[name.strip()] += 1
        except ET.ParseError:
            continue

    classes = sorted(counter, key=lambda c: (-counter[c], c))
    out = {
        "classes": classes,
        "name_to_id": {c: i for i, c in enumerate(classes)},
        "counts": dict(counter.most_common()),
        "num_classes": len(classes),
        "missing_xml": missing,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"classes={len(classes)} missing_xml={missing}")
    for c in classes:
        print(f"  {counter[c]:>8d}  {c}")
    print(f"written -> {out_path}")


if __name__ == "__main__":
    main()
