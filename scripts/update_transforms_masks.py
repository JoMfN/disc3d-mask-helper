#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def backup_once(path: Path, suffix: str = ".before_masks") -> None:
    backup = path.with_name(path.name + suffix)
    if path.exists() and not backup.exists():
        shutil.copy2(path, backup)


def update_one(dataset_dir: Path) -> dict:
    transforms = dataset_dir / "transforms.json"
    dataset_json = dataset_dir / "dataset.json"
    metadata_json = dataset_dir / "metadata.json"

    if not transforms.exists():
        return {"dataset": str(dataset_dir), "status": "missing_transforms"}

    data = json.loads(transforms.read_text(encoding="utf-8"))
    frames = data.get("frames", [])
    added = 0
    missing = 0

    for frame in frames:
        image_name = Path(frame["file_path"]).name
        mask_path = Path("masks") / image_name
        if (dataset_dir / mask_path).exists():
            frame["mask_path"] = str(mask_path).replace("\\", "/")
            added += 1
        else:
            missing += 1

    backup_once(transforms)
    transforms.write_text(json.dumps(data, indent=2), encoding="utf-8")

    if dataset_json.exists():
        d = json.loads(dataset_json.read_text(encoding="utf-8"))
        d["has_masks"] = added > 0
        d["mask_dir"] = "masks"
        backup_once(dataset_json)
        dataset_json.write_text(json.dumps(d, indent=2), encoding="utf-8")

    if metadata_json.exists():
        m = json.loads(metadata_json.read_text(encoding="utf-8"))
        m["masks"] = {"mask_dir": "masks", "frames_with_masks": added, "frames_missing_masks": missing}
        backup_once(metadata_json)
        metadata_json.write_text(json.dumps(m, indent=2), encoding="utf-8")

    return {"dataset": str(dataset_dir), "status": "ok", "added": added, "missing": missing}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default=None)
    parser.add_argument("--exports-root", default=None)
    args = parser.parse_args()

    if args.dataset_dir:
        print(update_one(Path(args.dataset_dir).resolve()))
    elif args.exports_root:
        for d in sorted(Path(args.exports_root).resolve().iterdir()):
            if d.is_dir() and (d / "dataset.json").exists():
                print(update_one(d))
    else:
        raise SystemExit("Use --dataset-dir or --exports-root")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
