#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path


def discover_specimens(
    exports_root: Path,
    image_subdir: str,
    mask_subdir: str,
    require_dataset_json: bool,
) -> list[Path]:
    specimens: list[Path] = []

    for child in sorted(exports_root.iterdir()):
        if not child.is_dir():
            continue

        if child.name.startswith("_"):
            continue

        if require_dataset_json and not (child / "dataset.json").exists():
            continue

        if not (child / image_subdir).is_dir():
            continue

        if not (child / mask_subdir).is_dir():
            continue

        specimens.append(child)

    return specimens


def count_files(path: Path, pattern: str) -> int:
    if not path.exists():
        return 0
    return sum(1 for _ in path.glob(pattern))


def run_one(payload: dict) -> dict:
    specimen_dir = Path(payload["specimen_dir"])
    script = Path(payload["script"])
    factor = payload["factor"]
    pattern = payload["pattern"]
    overwrite = payload["overwrite"]
    image_src = payload["image_src"]
    image_dst = payload["image_dst"]
    mask_src = payload["mask_src"]
    mask_dst = payload["mask_dst"]
    image_resample = payload["image_resample"]
    mask_resample = payload["mask_resample"]
    threshold_mask = payload["threshold_mask"]
    python_exe = payload["python_exe"]
    log_dir = Path(payload["log_dir"])

    started = time.time()

    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{specimen_dir.name}.log"

    cmd = [
        python_exe,
        str(script),
        "--dataset-dir",
        str(specimen_dir),
        "--factor",
        str(factor),
        "--image-src",
        image_src,
        "--image-dst",
        image_dst,
        "--mask-src",
        mask_src,
        "--mask-dst",
        mask_dst,
        "--pattern",
        pattern,
        "--image-resample",
        image_resample,
        "--mask-resample",
        mask_resample,
    ]

    if threshold_mask is not None:
        cmd.extend(["--threshold-mask", str(threshold_mask)])

    if overwrite:
        cmd.append("--overwrite")

    before_images = count_files(specimen_dir / image_dst, pattern)
    before_masks = count_files(specimen_dir / mask_dst, pattern)

    with log_file.open("w", encoding="utf-8") as log:
        log.write("[COMMAND]\n")
        log.write(" ".join(cmd) + "\n\n")
        log.flush()

        proc = subprocess.run(
            cmd,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )

    after_images = count_files(specimen_dir / image_dst, pattern)
    after_masks = count_files(specimen_dir / mask_dst, pattern)

    elapsed = time.time() - started

    status = "ok" if proc.returncode == 0 else "failed"

    return {
        "specimen_id": specimen_dir.name,
        "status": status,
        "returncode": proc.returncode,
        "seconds": f"{elapsed:.2f}",
        "images_before": before_images,
        "images_after": after_images,
        "masks_before": before_masks,
        "masks_after": after_masks,
        "log_file": str(log_file),
    }


def write_report(rows: list[dict], report_path: Path) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "specimen_id",
        "status",
        "returncode",
        "seconds",
        "images_before",
        "images_after",
        "masks_before",
        "masks_after",
        "log_file",
    ]

    with report_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            writer.writerow(row)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Batch-create images_2/masks_2 style downscaled folders for "
            "prepared DISC3D export folders."
        )
    )

    parser.add_argument(
        "--exports-root",
        required=True,
        help="Root folder containing one prepared export folder per specimen.",
    )

    parser.add_argument(
        "--script",
        default=None,
        help=(
            "Path to downscale_export_assets.py. "
            "Defaults to the script in the same directory as this batch helper."
        ),
    )

    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable used to run downscale_export_assets.py.",
    )

    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--factor", type=int, default=2)
    parser.add_argument("--pattern", default="*.png")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")

    parser.add_argument("--image-src", default="images")
    parser.add_argument("--mask-src", default="masks")

    parser.add_argument(
        "--image-dst",
        default=None,
        help="Default: images_<factor>",
    )

    parser.add_argument(
        "--mask-dst",
        default=None,
        help="Default: masks_<factor>",
    )

    parser.add_argument(
        "--image-resample",
        default="lanczos",
        choices=["lanczos", "bilinear", "nearest"],
    )

    parser.add_argument(
        "--mask-resample",
        default="nearest",
        choices=["nearest", "lanczos", "bilinear"],
    )

    parser.add_argument(
        "--threshold-mask",
        type=int,
        default=128,
        help=(
            "Threshold masks before downscaling. "
            "Use --threshold-mask -1 to keep soft masks."
        ),
    )

    parser.add_argument(
        "--require-dataset-json",
        action="store_true",
        help="Only process folders that also contain dataset.json.",
    )

    args = parser.parse_args()

    exports_root = Path(args.exports_root).expanduser().resolve()

    if args.script is None:
        script = Path(__file__).resolve().parent / "downscale_export_assets.py"
    else:
        script = Path(args.script).expanduser().resolve()

    if not exports_root.exists():
        raise FileNotFoundError(exports_root)

    if not script.exists():
        raise FileNotFoundError(script)

    image_dst = args.image_dst or f"{args.image_src}_{args.factor}"
    mask_dst = args.mask_dst or f"{args.mask_src}_{args.factor}"

    threshold_mask = None if args.threshold_mask == -1 else args.threshold_mask

    specimens = discover_specimens(
        exports_root=exports_root,
        image_subdir=args.image_src,
        mask_subdir=args.mask_src,
        require_dataset_json=args.require_dataset_json,
    )

    if args.limit is not None:
        specimens = specimens[: args.limit]

    log_dir = exports_root / "_logs" / f"downscale_factor_{args.factor}"

    print(f"[INFO] exports root: {exports_root}")
    print(f"[INFO] script:       {script}")
    print(f"[INFO] specimens:    {len(specimens)}")
    print(f"[INFO] workers:      {args.workers}")
    print(f"[INFO] factor:       {args.factor}")
    print(f"[INFO] images:       {args.image_src} -> {image_dst}")
    print(f"[INFO] masks:        {args.mask_src} -> {mask_dst}")
    print(f"[INFO] dry run:      {args.dry_run}")
    print(f"[INFO] logs:         {log_dir}")

    if not specimens:
        print("[WARN] No specimen folders found.")
        return 1

    if args.dry_run:
        print()
        print("[DRY-RUN] First commands:")
        for specimen in specimens[:10]:
            print(
                f"{args.python} {script} "
                f"--dataset-dir {specimen} "
                f"--factor {args.factor} "
                f"--image-src {args.image_src} "
                f"--image-dst {image_dst} "
                f"--mask-src {args.mask_src} "
                f"--mask-dst {mask_dst} "
                f"--pattern {args.pattern} "
                f"--image-resample {args.image_resample} "
                f"--mask-resample {args.mask_resample}"
            )
        return 0

    payloads = [
        {
            "specimen_dir": str(specimen),
            "script": str(script),
            "factor": args.factor,
            "pattern": args.pattern,
            "overwrite": args.overwrite,
            "image_src": args.image_src,
            "image_dst": image_dst,
            "mask_src": args.mask_src,
            "mask_dst": mask_dst,
            "image_resample": args.image_resample,
            "mask_resample": args.mask_resample,
            "threshold_mask": threshold_mask,
            "python_exe": args.python,
            "log_dir": str(log_dir),
        }
        for specimen in specimens
    ]

    rows: list[dict] = []

    workers = max(1, args.workers)

    if workers == 1:
        for payload in payloads:
            row = run_one(payload)
            rows.append(row)
            print(
                f"[{row['status'].upper()}] {row['specimen_id']} "
                f"images={row['images_after']} masks={row['masks_after']}"
            )
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(run_one, payload) for payload in payloads]

            for fut in as_completed(futures):
                row = fut.result()
                rows.append(row)
                print(
                    f"[{row['status'].upper()}] {row['specimen_id']} "
                    f"images={row['images_after']} masks={row['masks_after']}"
                )

    rows.sort(key=lambda r: r["specimen_id"])

    report_path = log_dir / "batch_downscale_report.csv"
    write_report(rows, report_path)

    ok = sum(1 for r in rows if r["status"] == "ok")
    failed = sum(1 for r in rows if r["status"] != "ok")

    print()
    print("============================================================")
    print(f"[SUMMARY] ok={ok}, failed={failed}")
    print(f"[SUMMARY] report={report_path}")
    print("============================================================")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
