# Fine masking

This document adds a high-resolution BiRefNet_HR-matting workflow to
`disc3d-mask-helper`.

The script is derived from a single-specimen prototype but follows the standard
DISC3D export folder layout:

```text
SPECIMEN_ID/
├── images/
├── masks_fine/
├── transforms.json
├── dataset.json
└── metadata.json
```

`masks_fine/` is the default output to avoid overwriting the existing `masks/`
folder. Use `--masks-subdir masks` only when the refined masks should become the
primary masks.

## Environment

Use the separated mask environment:

```bash
conda activate disc3d-mask
python -m pip install -r requirements/fine-masks-birefnet.txt
```

This must stay separate from the Nerfstudio/Splatfacto environment when
`nerfstudio==1.1.5` pins older `timm` versions.

## Single specimen

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/fine_mask_generation.py \
  --dataset-dir /path/to/disc3d_exports/SPECIMEN_ID \
  --masks-subdir masks_fine \
  --update-transforms \
  --update-dataset \
  --update-metadata
```

To replace the primary mask folder:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/fine_mask_generation.py \
  --dataset-dir /path/to/disc3d_exports/SPECIMEN_ID \
  --masks-subdir masks \
  --overwrite \
  --update-transforms \
  --update-dataset \
  --update-metadata
```

## Batch

```bash
python scripts/batch_fine_mask_exports.py \
  --exports-root /path/to/disc3d_exports \
  --python /path/to/miniconda3/envs/disc3d-mask/bin/python \
  --workers 2 \
  --gpus 0,1 \
  --masks-subdir masks_fine \
  --update-transforms \
  --update-dataset \
  --update-metadata
```

## Important parameters

Defaults follow the fine-mask prototype:

```text
model: ZhengPeng7/BiRefNet_HR-matting
inference-size: 2048
contrast: 2.0
brightness: 1.0
final-min-probability: 0.03
guided-filter-radius: 8
guided-filter-eps: 1e-3
core-confidence-thresh: 0.75
edge-band-px: 30
```

The pin/mount suppression is heuristic. Disable it when it selects the wrong
component:

```bash
--no-pin-mount-removal
```

For COLMAP feature extraction, downscale fine masks deterministically into
`masks_2/` and normally threshold them before resizing.
