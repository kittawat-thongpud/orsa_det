# Dataset Setup

Three datasets, two layouts. Installer: `tools/setup_datasets.py`
(verifies layout, extracts dropped archives, builds the IDD classmap, can fetch COCO2017).

```bash
# from orsa_det/, venv active
PYTHONPATH=. ./.venv/Scripts/python.exe tools/setup_datasets.py --dataset all
```

`datasets/` is **gitignored** — nothing here is versioned. You install it locally.

---

## Quick reference

| Dataset  | Loader              | Format | Location                          | Source                         |
|----------|---------------------|--------|-----------------------------------|--------------------------------|
| IDD      | `IDDDetection`      | VOC    | `../IDD_Detection/`               | idd.insaan.iiit.ac.in (gated)  |
| COCO128  | `YOLODetection`     | YOLO   | `datasets/coco128/`               | ultralytics.com (small zip)    |
| COCO2017 | `YOLODetection`     | YOLO   | `datasets/coco/`                  | cocodataset.org + ultralytics  |

Drop archives in `datasets/_archives/` (created on first run).

---

## IDD (VOC) — gated, manual download

1. Register + download **IDD Detection** from <https://idd.insaan.iiit.ac.in/>.
2. Drop the archive in `datasets/_archives/` (named `IDD_Detection.tar.gz` / `.zip`)
   — or pass `--archive PATH`.
3. Run:
   ```bash
   PYTHONPATH=. ./.venv/Scripts/python.exe tools/setup_datasets.py --dataset idd
   ```
   Extracts into `detection_models/IDD_Detection/`, verifies, then builds
   `configs/idd_classes.json` (15 classes). Delete that json to force a rebuild.

Expected layout (the archive already ships this; `train.txt` stems may contain `/`):
```
IDD_Detection/
  JPEGImages/<sub.../stem>.jpg        # .png fallback supported
  Annotations/<sub.../stem>.xml
  train.txt  val.txt  test.txt        # one stem per line
```
Config: `configs/orsa_small_idd.yaml` → `root: .../IDD_Detection`, `format: voc`.

---

## COCO128 (YOLO) — auto

Small 128-image sanity set. `datasets/coco128.zip` already in repo → auto-extract:
```bash
PYTHONPATH=. ./.venv/Scripts/python.exe tools/setup_datasets.py --dataset coco128
```
Layout:
```
datasets/coco128/
  images/train2017/*.jpg
  labels/train2017/*.txt              # "cls cx cy w h" normalized
```
Config: `configs/orsa_small_coco128.yaml` (train==val overfit demo).

---

## COCO2017 (YOLO) — opt-in, ~19GB

Not fetched unless you ask. Pre-converted YOLO labels + COCO images via Ultralytics:
```bash
PYTHONPATH=. ./.venv/Scripts/python.exe tools/setup_datasets.py --dataset coco2017 --download
```
Or place manually:
```
datasets/coco/
  images/{train2017,val2017}/*.jpg
  labels/{train2017,val2017}/*.txt
```
Then point a config at `root: datasets/coco`, `train_split: train2017`,
`val_split: val2017`, `num_classes: 80`, `format: yolo`.

---

## Verify only

Re-running any target with the data already in place just validates the layout
(image+label dirs non-empty, IDD spot-checks the first train stem resolves to
both `.jpg` and `.xml`). Exit code `0` = all selected datasets valid.
