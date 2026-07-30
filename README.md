# DISC3D mask batch helper

Overlay these files into the ETL repository.

Main files:

```text
scripts/mask_generation.py
scripts/batch_mask_exports.py
requirements/masks-birefnet.txt
docs/BATCH_MASKS.md
```

The batch helper calls `mask_generation.py` for each specimen export folder and
then patches `transforms.json`, `dataset.json`, and `metadata.json` so that
Nerfstudio can consume the generated masks.

## Mask-generation of original image (GPU recommended)

```
CUDA_VISIBLE_DEVICES=0 python scripts/mask_generation.py \
  --dataset-dir /path/to/disc3d_exports/SPECIMEN_ID
```

### Batch mask processing 

```
CUDA_VISIBLE_DEVICES=0 python scripts/batch_mask_exports.py \
  --exports-root /path/to/disc3d_exports \
  --gpus 0 \
  --workers 4
```

## Downscaling both image/ and mask/ folders 

Make it executable:

```bash
chmod +x scripts/batch_downscale_export_assets.py
```

### Dry run

```bash
python scripts/batch_downscale_export_assets.py \
  --exports-root /path/to/disc3d_exports \
  --workers 4 \
  --factor 2 \
  --dry-run
```

### Run for two specimens first

```bash
python scripts/batch_downscale_export_assets.py \
  --exports-root /path/to/disc3d_exports \
  --workers 2 \
  --factor 2 \
  --limit 2
```

### Full batch

```bash
python scripts/batch_downscale_export_assets.py \
  --exports-root /path/to/disc3d_exports \
  --workers 4 \
  --factor 2
```

## Splatfacto training should use GPU (other repo)

```
CUDA_VISIBLE_DEVICES=0 ns-train splatfacto \
  --data . \
  --output-dir ./outputs \
  --max-num-iterations 50000 \
  --machine.num-devices 1 \
  --pipeline.model.random-init False \
  --pipeline.model.rasterize-mode antialiased \
  colmap \
  --colmap-path colmap_known_pose_2/sparse_triangulated/0 \
  --masks-path masks \
  --downscale-factor 2
```

### More specific for RTX 3090 HW conditions and small specimen single-camera pole-dome aqcuisition conditions and COLMAP route

**colmap**
```
docker run --rm --gpus all \
  -e QT_QPA_PLATFORM=offscreen \
  -v /path/to/disc3d_exports:/data \
  -w /data/SPECIMEN_ID \
  colmap/colmap colmap feature_extractor \
    --database_path colmap_known_pose_2/database.db \
    --image_path images_2 \
    --ImageReader.mask_path masks_2 \
    --SiftExtraction.use_gpu 1 \
    --SiftExtraction.gpu_index 0
```

And

```
docker run --rm --gpus all \
  -e QT_QPA_PLATFORM=offscreen \
  -v /path/to/disc3d_exports:/data \
  -w /data/SPECIMEN_ID \
  colmap/colmap colmap sequential_matcher \
    --database_path colmap_known_pose_2/database.db \
    --SiftMatching.use_gpu 1 \
    --SiftMatching.gpu_index 0
```

Then Splatfacto training

```
export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CONDA_PREFIX/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
 
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
 
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_CUDA_ARCH_LIST="8.6"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True


CUDA_VISIBLE_DEVICES=0 ns-train splatfacto \
  --data . \
  --output-dir ../outputs \
  --max-num-iterations 50000 \
  --machine.num-devices 1 \
  --viewer.quit-on-train-completion True \
  --pipeline.model.stop-split-at 3000 \
  --pipeline.model.refine-every 30 \
  --pipeline.model.densify-grad-thresh 0.00003 \
  --pipeline.model.densify-size-thresh 0.0001 \
  --pipeline.model.cull-scale-thresh 0.002 \
  --pipeline.model.cull-alpha-thresh 0.08 \
  --pipeline.model.warmup-length 1000 \
  --pipeline.model.use-scale-regularization True \
  --pipeline.model.random-init False \
  --pipeline.model.rasterize-mode antialiased \
  colmap \
  --colmap-path colmap/sparse/0 \
  --masks-path masks \
  --downscale-factor 2
  ```

# Acknowledgement

We gratefully acknowledge Martin Wettig (@Species521) for generously sharing his Gaussian Splatting / Nerfstudio reconstruction pipeline and practical parameter insights for DISC3D specimen data. His experience with mask generation, Splatfacto configuration, and specimen-level training workflows provided valuable guidance for refining our own batch-processing strategy and helped inform the computational setup used in this work.
