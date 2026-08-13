#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def discover(exports_root: Path, images_subdir: str, require_dataset_json: bool) -> list[Path]:
    out = []
    for child in sorted(exports_root.iterdir()):
        if not child.is_dir() or child.name.startswith("_"):
            continue
        if require_dataset_json and not (child / "dataset.json").exists():
            continue
        if (child / images_subdir).is_dir():
            out.append(child)
    return out


def run_one(payload: dict) -> dict:
    specimen = Path(payload["specimen"])
    script = Path(payload["script"])
    log_dir = Path(payload["log_dir"])
    args = payload["args"]
    gpu = payload["gpu"]

    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{specimen.name}.log"

    cmd = [
        args["python"], str(script),
        "--dataset-dir", str(specimen),
        "--images-subdir", args["images_subdir"],
        "--masks-subdir", args["masks_subdir"],
        "--model", args["model"],
        "--device", "cuda" if gpu is not None else "cpu",
        "--inference-size", str(args["inference_size"]),
        "--contrast", str(args["contrast"]),
        "--brightness", str(args["brightness"]),
        "--final-min-probability", str(args["final_min_probability"]),
        "--guided-filter-radius", str(args["guided_filter_radius"]),
        "--guided-filter-eps", str(args["guided_filter_eps"]),
        "--core-confidence-thresh", str(args["core_confidence_thresh"]),
        "--edge-band-px", str(args["edge_band_px"]),
        "--component-min-area", str(args["component_min_area"]),
        "--component-dilation-kernel", str(args["component_dilation_kernel"]),
        "--patterns", *args["patterns"],
    ]

    if args["overwrite"]:
        cmd.append("--overwrite")
    if args["no_half"]:
        cmd.append("--no-half")
    if args["no_pin_mount_removal"]:
        cmd.append("--no-pin-mount-removal")
    if args["binary_output"]:
        cmd.extend(["--binary-output", "--binary-threshold", str(args["binary_threshold"])])
    if args["update_transforms"]:
        cmd.append("--update-transforms")
    if args["update_dataset"]:
        cmd.append("--update-dataset")
    if args["update_metadata"]:
        cmd.append("--update-metadata")
    if args["limit_per_specimen"]:
        cmd.extend(["--limit", str(args["limit_per_specimen"])])

    env = os.environ.copy()
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)

    started = time.time()
    with log_file.open("w", encoding="utf-8") as log:
        log.write("[COMMAND]\n")
        if gpu is not None:
            log.write(f"CUDA_VISIBLE_DEVICES={gpu} ")
        log.write(" ".join(cmd) + "\n\n")
        log.flush()
        proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, text=True, env=env)

    mask_dir = specimen / args["masks_subdir"]
    mask_count = 0
    if mask_dir.exists():
        for pattern in args["patterns"]:
            mask_count += sum(1 for _ in mask_dir.glob(pattern))

    return {
        "specimen_id": specimen.name,
        "status": "ok" if proc.returncode == 0 else "failed",
        "returncode": proc.returncode,
        "gpu": "" if gpu is None else gpu,
        "seconds": f"{time.time() - started:.2f}",
        "mask_count": mask_count,
        "log_file": str(log_file),
    }


def parse_gpus(text: str) -> list[str | None]:
    if text.lower() in {"cpu", "none", ""}:
        return [None]
    return [x.strip() for x in text.split(",") if x.strip()]


def write_report(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["specimen_id", "status", "returncode", "gpu", "seconds", "mask_count", "log_file"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    p = argparse.ArgumentParser(description="Batch fine-mask generation for DISC3D exports.")
    p.add_argument("--exports-root", required=True)
    p.add_argument("--script", default=None)
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--gpus", default="0")
    p.add_argument("--limit", type=int)
    p.add_argument("--limit-per-specimen", type=int)
    p.add_argument("--require-dataset-json", action="store_true")

    p.add_argument("--images-subdir", default="images")
    p.add_argument("--masks-subdir", default="masks_fine")
    p.add_argument("--patterns", nargs="+", default=["*.png"])

    p.add_argument("--model", default="ZhengPeng7/BiRefNet_HR-matting")
    p.add_argument("--inference-size", type=int, default=2048)
    p.add_argument("--no-half", action="store_true")
    p.add_argument("--overwrite", action="store_true")

    p.add_argument("--contrast", type=float, default=2.0)
    p.add_argument("--brightness", type=float, default=1.0)
    p.add_argument("--final-min-probability", type=float, default=0.03)
    p.add_argument("--guided-filter-radius", type=int, default=8)
    p.add_argument("--guided-filter-eps", type=float, default=1e-3)
    p.add_argument("--core-confidence-thresh", type=float, default=0.75)
    p.add_argument("--edge-band-px", type=int, default=30)
    p.add_argument("--component-min-area", type=int, default=30)
    p.add_argument("--component-dilation-kernel", type=int, default=15)
    p.add_argument("--no-pin-mount-removal", action="store_true")

    p.add_argument("--binary-output", action="store_true")
    p.add_argument("--binary-threshold", type=float, default=0.5)

    p.add_argument("--update-transforms", action="store_true")
    p.add_argument("--update-dataset", action="store_true")
    p.add_argument("--update-metadata", action="store_true")
    p.add_argument("--dry-run", action="store_true")

    args = p.parse_args()
    exports_root = Path(args.exports_root).expanduser().resolve()
    script = Path(args.script).expanduser().resolve() if args.script else Path(__file__).resolve().parent / "fine_mask_generation.py"

    if not exports_root.exists():
        raise FileNotFoundError(exports_root)
    if not script.exists():
        raise FileNotFoundError(script)

    specimens = discover(exports_root, args.images_subdir, args.require_dataset_json)
    if args.limit:
        specimens = specimens[: args.limit]

    gpus = parse_gpus(args.gpus)
    log_dir = exports_root / "_logs" / "fine_masks"

    print(f"[INFO] exports_root={exports_root}")
    print(f"[INFO] script={script}")
    print(f"[INFO] specimens={len(specimens)}")
    print(f"[INFO] workers={args.workers}")
    print(f"[INFO] gpus={gpus}")
    print(f"[INFO] masks_subdir={args.masks_subdir}")

    payloads = []
    args_dict = vars(args)
    for i, specimen in enumerate(specimens):
        payloads.append({
            "specimen": str(specimen),
            "script": str(script),
            "gpu": gpus[i % len(gpus)],
            "log_dir": str(log_dir),
            "args": args_dict,
        })

    if args.dry_run:
        for payload in payloads[:10]:
            print("[DRY-RUN]", payload["gpu"], payload["specimen"])
        return 0

    rows = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = [pool.submit(run_one, payload) for payload in payloads]
        for fut in as_completed(futures):
            row = fut.result()
            rows.append(row)
            print(f"[{row['status'].upper()}] {row['specimen_id']} gpu={row['gpu']} masks={row['mask_count']}")

    rows.sort(key=lambda r: r["specimen_id"])
    report = log_dir / "batch_fine_masks_report.csv"
    write_report(rows, report)

    ok = sum(r["status"] == "ok" for r in rows)
    failed = len(rows) - ok
    print(f"[SUMMARY] ok={ok}, failed={failed}, report={report}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
