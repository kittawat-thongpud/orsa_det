"""Scan IDD VOC annotations, count class frequencies, write class map JSON.

Robust + observable: per-file errors are caught and tallied (never abort the
scan), progress is logged on an interval (tqdm if available, else periodic line),
and a run log is written next to the output.

Usage:
  python tools/build_idd_classmap.py --root <IDD_Detection> --split train.txt --out configs/idd_classes.json
Flags:
  --quiet            only warnings/errors to console (progress + summary still logged to file)
  --log-every N      progress line every N files when tqdm absent (default 2000)
  --log FILE         run log path (default: <out>.scan.log)
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

log = logging.getLogger("idd_classmap")


def setup_logging(log_path: Path, quiet: bool) -> None:
    log.setLevel(logging.DEBUG)
    log.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(message)s", "%H:%M:%S")

    ch = logging.StreamHandler(sys.stderr)
    ch.setLevel(logging.WARNING if quiet else logging.INFO)
    ch.setFormatter(fmt)
    log.addHandler(ch)

    log_path.parent.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    log.addHandler(fh)
    log.debug("log file -> %s", log_path)


def _cache_num_classes(out_path: Path) -> int:
    """Return num_classes if out_path is an existing valid classmap, else 0."""
    if not out_path.is_file():
        return 0
    try:
        d = json.loads(out_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return 0
    classes = d.get("classes")
    if isinstance(classes, list) and classes and d.get("name_to_id") and d.get("counts"):
        return len(classes)
    return 0


def list_ann_paths(root: Path, split: str | None) -> list[Path]:
    """Resolve annotation paths up front so we know the total for progress."""
    ann_dir = root / "Annotations"
    if not ann_dir.is_dir():
        raise FileNotFoundError(f"Annotations dir not found: {ann_dir}")
    if split:
        split_file = root / split
        if not split_file.is_file():
            raise FileNotFoundError(f"split file not found: {split_file}")
        stems = [ln.strip() for ln in split_file.read_text(encoding="utf-8").splitlines()
                 if ln.strip()]
        log.info("split %s -> %d entries", split, len(stems))
        return [ann_dir / f"{s}.xml" for s in stems]
    paths = sorted(ann_dir.rglob("*.xml"))
    log.info("globbed %d xml under %s", len(paths), ann_dir)
    return paths


def _progress(iterable, total: int, log_every: int):
    """Always log progress to the logger (so file logs track it, tty or not).
    Overlay a tqdm bar on an interactive console as a bonus."""
    bar = None
    if sys.stderr.isatty():
        try:
            from tqdm import tqdm
            bar = tqdm(total=total, unit="xml", desc="scan")
        except Exception:  # noqa: BLE001 -- tqdm optional
            bar = None
    t0 = time.time()
    for i, item in enumerate(iterable, 1):
        if bar is not None:
            bar.update(1)
        if i % log_every == 0 or i == total:
            rate = i / max(time.time() - t0, 1e-9)
            log.info("scanned %d/%d (%.0f xml/s)", i, total, rate)
        yield item
    if bar is not None:
        bar.close()


def scan(paths: list[Path], log_every: int) -> tuple[Counter, dict]:
    counter: Counter = Counter()
    stats = {"total": len(paths), "ok": 0, "missing": 0,
             "parse_error": 0, "read_error": 0, "empty": 0, "objects": 0}
    for p in _progress(paths, len(paths), log_every):
        if not p.exists():
            stats["missing"] += 1
            log.debug("missing: %s", p)
            continue
        try:
            root = ET.parse(p).getroot()
        except ET.ParseError as e:
            stats["parse_error"] += 1
            log.warning("parse error %s: %s", p.name, e)
            continue
        except (OSError, ValueError) as e:
            stats["read_error"] += 1
            log.warning("read error %s: %s", p.name, e)
            continue
        objs = root.findall("object")
        if not objs:
            stats["empty"] += 1
        n = 0
        for obj in objs:
            name = obj.findtext("name")
            if name and name.strip():
                counter[name.strip()] += 1
                n += 1
        stats["objects"] += n
        stats["ok"] += 1
    return counter, stats


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--split", default="train.txt")
    ap.add_argument("--out", default="configs/idd_classes.json")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--log-every", type=int, default=2000)
    ap.add_argument("--log", default=None)
    ap.add_argument("--force", action="store_true",
                    help="rescan even if a valid output JSON already exists")
    args = ap.parse_args()

    out_path = Path(args.out)
    log_path = Path(args.log) if args.log else out_path.with_suffix(".scan.log")
    setup_logging(log_path, args.quiet)

    # cache: the output JSON is deterministic for a fixed dataset+split.
    # skip the multi-minute rescan unless --force or it's missing/invalid.
    if not args.force:
        n = _cache_num_classes(out_path)
        if n:
            log.info("cache hit: %s already valid (num_classes=%d) -- skipping scan "
                     "(use --force to rebuild)", out_path, n)
            return 0

    root = Path(args.root)
    t0 = time.time()
    try:
        paths = list_ann_paths(root, args.split or None)
    except FileNotFoundError as e:
        log.error("%s", e)
        return 2
    if not paths:
        log.error("no annotation paths resolved -- nothing to scan")
        return 2

    counter, stats = scan(paths, args.log_every)
    dt = time.time() - t0

    if not counter:
        log.error("no class labels found across %d files (parse_error=%d missing=%d) "
                  "-- output NOT written", stats["total"], stats["parse_error"],
                  stats["missing"])
        return 1

    classes = sorted(counter, key=lambda c: (-counter[c], c))
    out = {
        "classes": classes,
        "name_to_id": {c: i for i, c in enumerate(classes)},
        "counts": dict(counter.most_common()),
        "num_classes": len(classes),
        "scan_stats": stats,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    log.info("=== scan summary (%.1fs) ===", dt)
    log.info("files: total=%d ok=%d missing=%d parse_error=%d read_error=%d empty=%d",
             stats["total"], stats["ok"], stats["missing"],
             stats["parse_error"], stats["read_error"], stats["empty"])
    log.info("objects=%d classes=%d", stats["objects"], len(classes))
    for c in classes:
        log.info("  %8d  %s", counter[c], c)
    log.info("written -> %s", out_path)
    if stats["parse_error"] or stats["read_error"] or stats["missing"]:
        log.warning("completed with %d missing / %d parse / %d read issues "
                    "(details in %s)", stats["missing"], stats["parse_error"],
                    stats["read_error"], log_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
