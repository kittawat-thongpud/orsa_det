# ORSA-Det 2027

**O**cclusion-**R**obust **S**parse-**A**daptive **Det**ector — an edge-friendly object detector
targeting Jetson AGX Orin, evaluated on the India Driving Dataset (IDD).

> Pure-PyTorch core. Ultralytics is used **only** to reproduce the YOLO26-N baseline.

## Architecture

```
image
  └─ CSPBackbone (CNN)            P3 /8, P4 /16, P5 /32
       └─ PAN-lite neck           top-down + bottom-up, all scales -> embed_dim
            └─ LEM (P3/P4)         Local Evidence Module: DWConv + multi-dilation, gated residual
                 └─ Sparse Token Bank   per-scale top-K (256/160/96), sigmoid gate + STE
                      └─ Query-lite decoder   cross-attn to tokens, DAB ref-points, iterative box refine
                           ├─ main head        one-to-one (Hungarian)            [inference]
                           ├─ aux query group  one-to-many (Group-DETR)          [train-only]
                           └─ aux dense head   one-to-many (TAL)                 [train-only]
```

- **Token Survival Loss** keeps gate-scores alive at GT centers (larger `tau` for small objects).
- **Static top-K** → fixed shapes → TensorRT FP16/INT8 friendly.
- Train-only aux branches are cut by `fuse_for_inference()` before export.

## Baseline

YOLO26-N, 300 epochs on IDD: **mAP50 = 0.6131**, **mAP50-95 = 0.3996** (muSGD).

## Setup (venv)

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# install CUDA torch FIRST, then the rest
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

Verify CUDA: `python -c "import torch; print(torch.__version__, torch.cuda.is_available())"`

## Dataset (IDD)

Pascal-VOC layout at `dataset.root`:

```
IDD_Detection/
  JPEGImages/<stem>.jpg
  Annotations/<stem>.xml
  train.txt  val.txt  test.txt      # split files: one <stem> per line
```

Build the class map (15 classes, frequency-sorted):

```bash
python tools/build_idd_classmap.py \
  --root D:/.../IDD_Detection --split train.txt --out configs/idd_classes.json
```

## Train

```bash
python scripts/train.py --cfg configs/orsa_small_idd.yaml
# overrides
python scripts/train.py --cfg configs/orsa_small_idd.yaml --epochs 100 --phase A
```

- **Phase A** = AdamW bring-up/debug. **Phase B** = MuSGD-hybrid (Muon on 2D weights +
  SGD-Nesterov on 1D + AdamW on embeddings/ref-points), for baseline-fair comparison.
- 8 GB profile: AMP bf16, batch 4, grad-accum ×8 (eff. 32), EMA, grad-clip 1.0, cosine LR.
- Artifacts → `runs/orsa_small_idd/`: TensorBoard, `metrics.jsonl`, checkpoints.

## Evaluate

```bash
python scripts/eval.py --cfg configs/orsa_small_idd.yaml --ckpt runs/orsa_small_idd/best.pth --use-ema
```

Metrics via pycocotools: mAP50, mAP50-95, AP-S/M/L, AR@100.

## Export (ONNX → TensorRT)

```bash
python scripts/export_onnx.py --cfg configs/orsa_small_idd.yaml \
  --ckpt runs/orsa_small_idd/best.pth --out orsa_small.onnx --imgsz 512 --use-ema
```

Static-K graph; on Orin build the engine with `trtexec --fp16` (or INT8 + calibration).

## Smoke test

```bash
python tools/smoke_test.py   # forward + loss.backward + postprocess on a dummy batch
```

## Layout

```
orsa/
  models/   blocks backbone neck lem sparse_token deformable head orsa_det
  losses/   box_ops matcher survival criterion
  data/     transforms idd_dataset
  engine/   optim trainer evaluator
  utils/    logger metrics
configs/    orsa_small_idd.yaml  idd_classes.json
scripts/    train.py eval.py export_onnx.py
tools/      build_idd_classmap.py smoke_test.py
```
