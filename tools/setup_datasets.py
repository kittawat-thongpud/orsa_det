"""Dataset installer / verifier for ORSA-Det.

Handles the THREE datasets the loaders expect, each with a different layout:

  IDD  (VOC, orsa/data/idd_dataset.py)  -> <repo>/../IDD_Detection/
            JPEGImages/<sub.../stem>.jpg   Annotations/<sub.../stem>.xml
            train.txt val.txt test.txt   (one stem per line, may contain '/')
  coco128 (YOLO, orsa/data/yolo_dataset.py) -> datasets/coco128/
            images/train2017/*.jpg   labels/train2017/*.txt
  coco2017 (YOLO) -> datasets/coco/
            images/{train2017,val2017}/*.jpg   labels/{train2017,val2017}/*.txt

DESIGN: gated / huge sources are NOT silently auto-fetched.
  - IDD requires registration (idd.insaan.iiit.ac.in). You DOWNLOAD the
    archive manually, drop it in datasets/_archives/, this script extracts it.
  - coco128 ships as a small zip (already in repo) -> auto-extract.
  - coco2017 (~19GB images) -> opt-in download via --download (ultralytics URLs).

Usage:
  PYTHONPATH=. ./.venv/Scripts/python.exe tools/setup_datasets.py --dataset all
  ...                                       tools/setup_datasets.py --dataset idd
  ...                                       tools/setup_datasets.py --dataset coco2017 --download
Drop archives in: <repo>/datasets/_archives/
  IDD_Detection.tar.gz | IDD_Detection.zip   (or pass --archive PATH)
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]                  # .../orsa_det
DETROOT = REPO.parent                                        # .../detection_models
DATASETS = REPO / "datasets"
ARCHIVES = DATASETS / "_archives"
PY = sys.executable

IDD_ROOT = DETROOT / "IDD_Detection"                         # matches orsa_small_idd.yaml
COCO128_ROOT = DATASETS / "coco128"
COCO2017_ROOT = DATASETS / "coco"

GREEN, RED, DIM, END = "\033[32m", "\033[31m", "\033[2m", "\033[0m"


def ok(msg: str) -> None:
    print(f"{GREEN}[ok]{END} {msg}")


def warn(msg: str) -> None:
    print(f"{RED}[!!]{END} {msg}")


def info(msg: str) -> None:
    print(f"{DIM}     {msg}{END}")


# ---------------------------------------------------------------- extract utils
def _find_archive(names: list[str], explicit: Path | None) -> Path | None:
    if explicit is not None:
        return explicit if explicit.exists() else None
    for n in names:
        p = ARCHIVES / n
        if p.exists():
            return p
    return None


def _extract(archive: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    info(f"extracting {archive.name} -> {dest} (may take a while)")
    if archive.suffix == ".zip" or archive.name.endswith(".zip"):
        with zipfile.ZipFile(archive) as z:
            z.extractall(dest)
    elif archive.name.endswith((".tar.gz", ".tgz", ".tar")):
        mode = "r:gz" if archive.name.endswith((".tar.gz", ".tgz")) else "r:"
        with tarfile.open(archive, mode) as t:
            t.extractall(dest)
    else:
        raise ValueError(f"unknown archive type: {archive.name}")


# ---------------------------------------------------------------- IDD (VOC)
def setup_idd(archive: Path | None) -> bool:
    """Extract (if needed) + verify IDD VOC layout + build classmap."""
    if not _idd_layout_ok():
        arc = _find_archive(
            ["IDD_Detection.tar.gz", "IDD_Detection.tgz", "IDD_Detection.zip"], archive
        )
        if arc is None:
            warn(f"IDD not found at {IDD_ROOT} and no archive in {ARCHIVES}")
            info("register + download IDD Detection from idd.insaan.iiit.ac.in,")
            info(f"then drop IDD_Detection.tar.gz in {ARCHIVES} and rerun.")
            return False
        # archive root folder IS 'IDD_Detection' -> extract into DETROOT
        _extract(arc, DETROOT)
    if not _idd_layout_ok():
        warn("IDD layout still invalid after extraction (see required tree below)")
        info("need: IDD_Detection/{JPEGImages,Annotations,train.txt,val.txt}")
        return False
    ok(f"IDD layout valid @ {IDD_ROOT}")
    return _build_idd_classmap()


def _idd_layout_ok() -> bool:
    need = [IDD_ROOT / "JPEGImages", IDD_ROOT / "Annotations",
            IDD_ROOT / "train.txt", IDD_ROOT / "val.txt"]
    miss = [p.name for p in need if not p.exists()]
    if miss:
        return False
    # spot-check first train stem resolves to image + xml
    stem = (IDD_ROOT / "train.txt").read_text().splitlines()[0].strip()
    img = any((IDD_ROOT / "JPEGImages" / f"{stem}{e}").exists()
              for e in (".jpg", ".png"))
    xml = (IDD_ROOT / "Annotations" / f"{stem}.xml").exists()
    return img and xml


def _build_idd_classmap() -> bool:
    out = REPO / "configs" / "idd_classes.json"
    if out.exists():
        ok(f"classmap exists @ {out.relative_to(REPO)} (delete to rebuild)")
        return True
    info("building configs/idd_classes.json from Annotations ...")
    r = subprocess.run(
        [PY, str(REPO / "tools" / "build_idd_classmap.py"),
         "--root", str(IDD_ROOT), "--split", "train.txt", "--out", str(out)],
        cwd=str(REPO),
    )
    if r.returncode == 0 and out.exists():
        ok(f"classmap built @ {out.relative_to(REPO)}")
        return True
    warn("classmap build failed")
    return False


# ---------------------------------------------------------------- coco128 (YOLO)
def setup_coco128() -> bool:
    if not _yolo_layout_ok(COCO128_ROOT, ["train2017"]):
        zip_ = _find_archive(["coco128.zip"], None) or (DATASETS / "coco128.zip")
        if not zip_.exists():
            warn("coco128 missing and no coco128.zip found")
            info("get it: https://ultralytics.com/assets/coco128.zip -> datasets/")
            return False
        _extract(zip_, DATASETS)        # zip contains top-level coco128/
    if _yolo_layout_ok(COCO128_ROOT, ["train2017"]):
        n = len(list((COCO128_ROOT / "images" / "train2017").glob("*")))
        ok(f"coco128 valid @ {COCO128_ROOT.relative_to(REPO)} ({n} imgs)")
        return True
    warn("coco128 layout invalid after extraction")
    return False


# ---------------------------------------------------------------- coco2017 (YOLO)
def setup_coco2017(download: bool) -> bool:
    splits = ["train2017", "val2017"]
    if _yolo_layout_ok(COCO2017_ROOT, splits):
        ok(f"coco2017 valid @ {COCO2017_ROOT.relative_to(REPO)}")
        return True
    if not download:
        warn(f"coco2017 not found at {COCO2017_ROOT}")
        info("rerun with --download to fetch (~19GB images + YOLO labels), or place")
        info("manually: datasets/coco/{images,labels}/{train2017,val2017}/")
        return False
    return _download_coco2017(splits)


def _download_coco2017(splits: list[str]) -> bool:
    """Fetch via Ultralytics URLs: pre-converted YOLO labels + COCO images."""
    try:
        from ultralytics.utils.downloads import download
    except Exception as e:                                   # noqa: BLE001
        warn(f"ultralytics not available for download: {e}")
        return False
    COCO2017_ROOT.mkdir(parents=True, exist_ok=True)
    # labels zip extracts to coco/labels/{train2017,val2017} + train/val .txt lists
    info("downloading YOLO labels ...")
    download(
        ["https://github.com/ultralytics/assets/releases/download/v0.0.0/coco2017labels.zip"],
        dir=DATASETS, unzip=True, delete=False,
    )
    info("downloading images (train2017 ~18GB, val2017 ~1GB) ...")
    base = "http://images.cocodataset.org/zips/"
    download([base + "train2017.zip", base + "val2017.zip"],
             dir=COCO2017_ROOT / "images", unzip=True, delete=False, threads=2)
    if _yolo_layout_ok(COCO2017_ROOT, splits):
        ok(f"coco2017 downloaded @ {COCO2017_ROOT.relative_to(REPO)}")
        info("set configs to: root=datasets/coco train_split=train2017 val_split=val2017")
        return True
    warn("coco2017 layout invalid after download")
    return False


# ---------------------------------------------------------------- shared verify
def _yolo_layout_ok(root: Path, splits: list[str]) -> bool:
    if not root.exists():
        return False
    for s in splits:
        imgs = root / "images" / s
        lbls = root / "labels" / s
        if not imgs.is_dir() or not lbls.is_dir():
            return False
        if not any(imgs.iterdir()):
            return False
    return True


# ---------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description="ORSA-Det dataset installer/verifier")
    ap.add_argument("--dataset", choices=["idd", "coco128", "coco2017", "all"],
                    default="all")
    ap.add_argument("--archive", type=Path, default=None,
                    help="explicit path to IDD archive (overrides _archives lookup)")
    ap.add_argument("--download", action="store_true",
                    help="allow fetching coco2017 (~19GB) from the internet")
    args = ap.parse_args()

    ARCHIVES.mkdir(parents=True, exist_ok=True)
    sel = args.dataset
    results: dict[str, bool] = {}

    if sel in ("idd", "all"):
        print("\n== IDD (VOC) ==")
        results["idd"] = setup_idd(args.archive)
    if sel in ("coco128", "all"):
        print("\n== COCO128 (YOLO) ==")
        results["coco128"] = setup_coco128()
    if sel in ("coco2017", "all"):
        print("\n== COCO2017 (YOLO) ==")
        results["coco2017"] = setup_coco2017(args.download)

    print("\n== summary ==")
    for k, v in results.items():
        (ok if v else warn)(k)
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
