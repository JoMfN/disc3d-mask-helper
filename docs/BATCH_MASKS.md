# Batch mask generation for DISC3D exports

This helper belongs in the ETL/preprocessing repository, but it should be run with
the mask-generation environment, not the Nerfstudio training environment.

The intended environment split is:

```text
disc3d-mask  -> BiRefNet mask generation, timm>=0.9.12
disc3d-splat -> Nerfstudio/Splatfacto, keep Nerfstudio's timm pin
```

## Install mask environment

```bash
conda create -n disc3d-mask python=3.10 -y
conda activate disc3d-mask

python -m pip install --upgrade pip
python -m pip install -r requirements/masks-birefnet.txt
```

## One specimen

```bash
conda activate disc3d-mask

CUDA_VISIBLE_DEVICES=0 python scripts/mask_generation.py   --dataset-dir /mnt/sda/DISC3Dscans/disc3d_exports/SPECIMEN_ID
```

This reads:

```text
SPECIMEN_ID/images/
```

and writes:

```text
SPECIMEN_ID/masks/
```

## Batch over export folders

```bash
python scripts/batch_mask_exports.py   --exports-root /mnt/sda/DISC3Dscans/disc3d_exports   --mask-python /home/eliolc/miniconda3/envs/disc3d-mask/bin/python   --workers 2   --gpus 0,1   --limit 4
```

For all specimens:

```bash
python scripts/batch_mask_exports.py   --exports-root /mnt/sda/DISC3Dscans/disc3d_exports   --mask-python /home/eliolc/miniconda3/envs/disc3d-mask/bin/python   --workers 2   --gpus 0,1
```

## Only update JSON files after masks already exist

```bash
python scripts/batch_mask_exports.py   --exports-root /mnt/sda/DISC3Dscans/disc3d_exports   --update-only   --workers 8
```

## What is patched

For each frame in `transforms.json`, the helper adds:

```json
"mask_path": "masks/image_0001.png"
```

It also adds light metadata to:

```text
dataset.json
metadata.json
```

Backups are created once:

```text
transforms.json.before_masks
dataset.json.before_masks
metadata.json.before_masks
```

## Training with masks

Use the Nerfstudio environment after masks are generated and `transforms.json` has
been patched:

```bash
conda activate disc3d-splat
cd /mnt/sda/DISC3Dscans/disc3d_exports/SPECIMEN_ID

CUDA_VISIBLE_DEVICES=0 ns-train splatfacto   --output-dir ../outputs   --max-num-iterations 50000   --machine.num-devices 1   --pipeline.model.random-init True   --pipeline.model.rasterize-mode antialiased   nerfstudio-data   --data .   --load-3D-points False
```
