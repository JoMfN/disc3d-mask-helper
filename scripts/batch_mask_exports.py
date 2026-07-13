#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def iter_export_dirs(exports_root: Path) -> list[Path]:
    dirs: list[Path] = []
    for child in sorted(exports_root.iterdir()):
        if not child.is_dir():
            continue
        if child.name.startswith("_"):
            continue
        if (child / "dataset.json").exists() and (child / "transforms.json").exists() and (child / "images").is_dir():
            dirs.append(child)
    return dirs


def backup_once(path: Path, suffix: str) -> None:
    backup = path.with_name(path.name + suffix)
    if path.exists() and not backup.exists():
        backup.write_bytes(path.read_bytes())


def update_dataset_json(export_dir: Path, masks_subdir: str, added: int) -> None:
    path = export_dir / "dataset.json"
    if not path.exists():
        return

    data = json.loads(path.read_text(encoding="utf-8"))
    changed = False

    if added > 0 and data.get("mask_dir") != masks_subdir:
        data["mask_dir"] = masks_subdir
        changed = True

    if data.get("has_masks") is not (added > 0):
        data["has_masks"] = added > 0
        changed = True

    if changed:
        backup_once(path, ".before_masks")
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def update_metadata_json(export_dir: Path, masks_subdir: str, total_frames: int, added: int, missing: int) -> None:
    path = export_dir / "metadata.json"
    data = {}
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))

    data["masks"] = {
        "mask_dir": masks_subdir,
        "n_frames": total_frames,
        "n_masks_linked": added,
        "n_masks_missing": missing,
        "mask_path_field": "mask_path",
        "notes": "mask_path entries were added to transforms.json by batch_mask_exports.py",
    }

    backup_once(path, ".before_masks")
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def update_transforms_with_masks(
    export_dir: Path,
    masks_subdir: str = "masks",
    backup: bool = True,
    require_all: bool = False,
) -> dict:
    transforms_path = export_dir / "transforms.json"
    if not transforms_path.exists():
        raise FileNotFoundError(f"Missing transforms.json: {transforms_path}")

    data = json.loads(transforms_path.read_text(encoding="utf-8"))
    frames = data.get("frames", [])

    if not frames:
        raise ValueError(f"No frames in transforms.json: {transforms_path}")

    added = 0
    missing = 0
    changed = False

    for frame in frames:
        file_path = frame.get("file_path")
        if not file_path:
            missing += 1
            continue

        image_name = Path(file_path).name
        mask_rel = Path(masks_subdir) / image_name
        mask_abs = export_dir / mask_rel

        if mask_abs.exists():
            mask_rel_str = str(mask_rel).replace("\\", "/")
            if frame.get("mask_path") != mask_rel_str:
                frame["mask_path"] = mask_rel_str
                changed = True
            added += 1
        else:
            if "mask_path" in frame:
                frame.pop("mask_path")
                changed = True
            missing += 1

    if require_all and missing:
        raise RuntimeError(f"{export_dir.name}: missing {missing} mask(s)")

    if changed:
        if backup:
            backup_once(transforms_path, ".before_masks")
        transforms_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    update_dataset_json(export_dir, masks_subdir=masks_subdir, added=added)
    update_metadata_json(export_dir, masks_subdir=masks_subdir, total_frames=len(frames), added=added, missing=missing)

    return {
        "frames": len(frames),
        "masks_linked": added,
        "masks_missing": missing,
        "transforms_changed": changed,
    }


def run_mask_generation(
    export_dir: Path,
    mask_script: Path,
    mask_python: str,
    gpu: str | None,
    overwrite: bool,
    input_size: int,
    pattern: str,
    threshold: float | None,
    model: str,
    log_file: Path,
) -> int:
    cmd = [
        mask_python,
        str(mask_script),
        "--dataset-dir",
        str(export_dir),
        "--input-size",
        str(input_size),
        "--pattern",
        pattern,
        "--model",
        model,
        "--device",
        "cuda" if gpu is not None else "cuda",
    ]

    if overwrite:
        cmd.append("--overwrite")

    if threshold is not None:
        cmd.extend(["--threshold", str(threshold)])

    env = os.environ.copy()
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)

    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("w", encoding="utf-8") as log:
        log.write("[COMMAND] " + " ".join(cmd) + "\n")
        if gpu is not None:
            log.write(f"[CUDA_VISIBLE_DEVICES] {gpu}\n")
        log.flush()

        proc = subprocess.run(
            cmd,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
            text=True,
        )

    return proc.returncode


def process_one(payload: dict) -> dict:
    export_dir = Path(payload["export_dir"])
    mask_script = Path(payload["mask_script"])
    mask_python = payload["mask_python"]
    gpu = payload["gpu"]
    overwrite = payload["overwrite"]
    input_size = payload["input_size"]
    pattern = payload["pattern"]
    threshold = payload["threshold"]
    model = payload["model"]
    skip_generation = payload["skip_generation"]
    update_only = payload["update_only"]
    require_all = payload["require_all"]
    logs_dir = Path(payload["logs_dir"])

    started = time.time()
    log_file = logs_dir / f"{export_dir.name}.mask_generation.log"

    status = "ok"
    message = ""

    try:
        if not skip_generation and not update_only:
            rc = run_mask_generation(
                export_dir=export_dir,
                mask_script=mask_script,
                mask_python=mask_python,
                gpu=gpu,
                overwrite=overwrite,
                input_size=input_size,
                pattern=pattern,
                threshold=threshold,
                model=model,
                log_file=log_file,
            )
            if rc != 0:
                raise RuntimeError(f"mask_generation.py failed with exit code {rc}; see {log_file}")

        update = update_transforms_with_masks(
            export_dir=export_dir,
            masks_subdir="masks",
            backup=True,
            require_all=require_all,
        )

    except Exception as exc:
        status = "failed"
        message = repr(exc)
        update = {"frames": 0, "masks_linked": 0, "masks_missing": 0, "transforms_changed": False}

    return {
        "specimen_id": export_dir.name,
        "status": status,
        "gpu": "" if gpu is None else gpu,
        "seconds": round(time.time() - started, 3),
        "frames": update["frames"],
        "masks_linked": update["masks_linked"],
        "masks_missing": update["masks_missing"],
        "transforms_changed": update["transforms_changed"],
        "message": message,
    }


def write_report(rows: list[dict], report_path: Path) -> None:
    fieldnames = [
        "specimen_id",
        "status",
        "gpu",
        "seconds",
        "frames",
        "masks_linked",
        "masks_missing",
        "transforms_changed",
        "message",
    ]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Batch-generate masks for exported DISC3D specimens and patch transforms.json."
    )
    parser.add_argument("--exports-root", required=True, help="Folder containing one export folder per specimen.")
    parser.add_argument(
        "--mask-script",
        default=None,
        help="Path to mask_generation.py. Defaults to sibling script next to this file.",
    )
    parser.add_argument(
        "--mask-python",
        default=sys.executable,
        help="Python executable from the mask environment, e.g. /home/.../envs/disc3d-mask/bin/python.",
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--gpus", default="", help="Comma-separated GPU ids, e.g. 0,1,2,3.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-generation", action="store_true", help="Do not call mask_generation.py; only patch JSON files.")
    parser.add_argument("--update-only", action="store_true", help="Alias for --skip-generation.")
    parser.add_argument("--require-all", action="store_true", help="Fail a specimen if any frame lacks a mask.")
    parser.add_argument("--input-size", type=int, default=1024)
    parser.add_argument("--pattern", default="*.png")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--model", default="ZhengPeng7/BiRefNet")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    exports_root = Path(args.exports_root).expanduser().resolve()
    if not exports_root.exists():
        raise FileNotFoundError(exports_root)

    mask_script = Path(args.mask_script).expanduser().resolve() if args.mask_script else Path(__file__).with_name("mask_generation.py").resolve()
    if not mask_script.exists() and not args.skip_generation and not args.update_only:
        raise FileNotFoundError(f"Mask script not found: {mask_script}")

    export_dirs = iter_export_dirs(exports_root)
    if args.limit is not None:
        export_dirs = export_dirs[: args.limit]

    gpu_list = [gpu.strip() for gpu in args.gpus.split(",") if gpu.strip()]
    workers = max(1, int(args.workers))
    logs_dir = exports_root / "_logs" / "masks"

    print(f"[INFO] exports_root: {exports_root}")
    print(f"[INFO] specimens:    {len(export_dirs)}")
    print(f"[INFO] workers:      {workers}")
    print(f"[INFO] gpus:         {gpu_list if gpu_list else 'not pinned'}")
    print(f"[INFO] mask_python:  {args.mask_python}")
    print(f"[INFO] mask_script:  {mask_script}")
    print(f"[INFO] logs:         {logs_dir}")

    if args.dry_run:
        for i, export_dir in enumerate(export_dirs):
            gpu = gpu_list[i % len(gpu_list)] if gpu_list else None
            print(f"[DRY-RUN] {export_dir.name} gpu={gpu}")
        return 0

    payloads = []
    for i, export_dir in enumerate(export_dirs):
        gpu = gpu_list[i % len(gpu_list)] if gpu_list else None
        payloads.append(
            {
                "export_dir": str(export_dir),
                "mask_script": str(mask_script),
                "mask_python": args.mask_python,
                "gpu": gpu,
                "overwrite": args.overwrite,
                "input_size": args.input_size,
                "pattern": args.pattern,
                "threshold": args.threshold,
                "model": args.model,
                "skip_generation": args.skip_generation,
                "update_only": args.update_only,
                "require_all": args.require_all,
                "logs_dir": str(logs_dir),
            }
        )

    rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(process_one, payload) for payload in payloads]
        for fut in as_completed(futures):
            row = fut.result()
            rows.append(row)
            print(
                f"[{row['status'].upper()}] {row['specimen_id']} "
                f"gpu={row['gpu']} masks={row['masks_linked']}/{row['frames']} "
                f"missing={row['masks_missing']} seconds={row['seconds']}"
            )

    rows.sort(key=lambda r: r["specimen_id"])
    report_path = logs_dir / "batch_masks_report.csv"
    write_report(rows, report_path)

    failed = sum(1 for r in rows if r["status"] != "ok")
    print()
    print("============================================================")
    print(f"[SUMMARY] specimens={len(rows)} failed={failed}")
    print(f"[SUMMARY] report={report_path}")
    print("============================================================")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
