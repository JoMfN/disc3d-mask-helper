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

# Acknowledgement

We gratefully acknowledge Martin Wettig (@Species521) for generously sharing his Gaussian Splatting / Nerfstudio reconstruction pipeline and practical parameter insights for DISC3D specimen data. His experience with mask generation, Splatfacto configuration, and specimen-level training workflows provided valuable guidance for refining our own batch-processing strategy and helped inform the computational setup used in this work.
