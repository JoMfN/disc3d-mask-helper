#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageEnhance
from scipy import ndimage
from tqdm import tqdm
from transformers import AutoModelForImageSegmentation
from torchvision.transforms.functional import normalize, to_tensor


IMAGE_NUMBER_RE = re.compile(r"image[_-]?0*(\d+)", re.IGNORECASE)


def natural_key(path: Path) -> tuple[int, str]:
    m = IMAGE_NUMBER_RE.search(path.name)
    if m:
        return int(m.group(1)), path.name
    nums = re.findall(r"\d+", path.name)
    return (int(nums[0]) if nums else 10**12, path.name)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA requested but unavailable; using CPU.")
        return torch.device("cpu")
    return torch.device(name)


def prepare_grayscale_image(orig: Image.Image, contrast: float, brightness: float) -> Image.Image:
    gray = orig.convert("L")
    if contrast != 1.0:
        gray = ImageEnhance.Contrast(gray).enhance(contrast)
    if brightness != 1.0:
        gray = ImageEnhance.Brightness(gray).enhance(brightness)
    return Image.merge("RGB", (gray, gray, gray))


def unpack_mask(output: object) -> np.ndarray:
    if isinstance(output, (list, tuple)):
        output = output[-1]
    if isinstance(output, (list, tuple)):
        output = output[-1]
    if not torch.is_tensor(output):
        raise TypeError(f"Unsupported model output type: {type(output)!r}")
    if output.ndim == 4:
        output = output[0, 0]
    elif output.ndim == 3:
        output = output[0]
    elif output.ndim != 2:
        raise ValueError(f"Unsupported model output shape: {tuple(output.shape)}")
    return output.sigmoid().float().detach().cpu().numpy().astype(np.float32)


def guided_filter(guide: np.ndarray, src: np.ndarray, radius: int, eps: float) -> np.ndarray:
    guide = guide.astype(np.float32)
    src = src.astype(np.float32)
    ksize = (radius * 2 + 1, radius * 2 + 1)

    mean_i = cv2.boxFilter(guide, cv2.CV_32F, ksize)
    mean_p = cv2.boxFilter(src, cv2.CV_32F, ksize)
    mean_ip = cv2.boxFilter(guide * src, cv2.CV_32F, ksize)
    cov_ip = mean_ip - mean_i * mean_p

    mean_ii = cv2.boxFilter(guide * guide, cv2.CV_32F, ksize)
    var_i = mean_ii - mean_i * mean_i

    a = cov_ip / (var_i + eps)
    b = mean_p - a * mean_i

    return cv2.boxFilter(a, cv2.CV_32F, ksize) * guide + cv2.boxFilter(b, cv2.CV_32F, ksize)


def keep_upper_component(alpha: np.ndarray, min_area: int, dilation_kernel: int) -> np.ndarray:
    fallback = alpha.copy()
    labels = stats = None
    num_labels = 0

    for thresh in (0.5, 0.4, 0.3, 0.2):
        bin_mask = (alpha > thresh).astype(np.uint8) * 255
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bin_mask)
        if num_labels > 1:
            break

    if labels is None or stats is None or num_labels <= 1:
        return fallback

    best_label = -1
    min_top = float("inf")

    for i in range(1, num_labels):
        top = stats[i, cv2.CC_STAT_TOP]
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area and top < min_top:
            min_top = top
            best_label = i

    if best_label < 0:
        return fallback

    component = (labels == best_label).astype(np.uint8)
    k = max(1, int(dilation_kernel))
    if k % 2 == 0:
        k += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    cleaned = alpha * cv2.dilate(component, kernel, iterations=1)

    return fallback if float(np.max(cleaned)) < 0.05 else cleaned


def refine_alpha(prob: np.ndarray, gray: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    guide = gray.astype(np.float32) / 255.0
    refined = guided_filter(guide, prob, args.guided_filter_radius, args.guided_filter_eps)
    refined = np.clip(refined, 0.0, 1.0)

    if not args.no_pin_mount_removal:
        refined = keep_upper_component(refined, args.component_min_area, args.component_dilation_kernel)

    core = (refined >= args.core_confidence_thresh).astype(np.uint8)
    core_filled = ndimage.binary_fill_holes(core).astype(np.uint8)
    dist_inside = cv2.distanceTransform(core_filled, cv2.DIST_L2, 5)

    alpha = refined.copy()
    alpha[dist_inside > args.edge_band_px] = 1.0
    alpha[alpha < args.final_min_probability] = 0.0

    if args.binary_output:
        alpha = (alpha >= args.binary_threshold).astype(np.float32)

    return np.clip(alpha, 0.0, 1.0)


def infer_one(img_path: Path, out_path: Path, model, device: torch.device, use_half: bool, args: argparse.Namespace) -> None:
    with Image.open(img_path) as img:
        orig = img.convert("RGB")
    w, h = orig.size

    prepared = prepare_grayscale_image(orig, args.contrast, args.brightness)
    inp = prepared.resize((args.inference_size, args.inference_size), Image.Resampling.BILINEAR)

    tensor = normalize(
        to_tensor(inp),
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225],
    ).unsqueeze(0).to(device)

    if use_half:
        tensor = tensor.half()

    with torch.inference_mode():
        prob = unpack_mask(model(tensor))

    prob = cv2.resize(prob, (w, h), interpolation=cv2.INTER_LINEAR)
    gray = cv2.cvtColor(np.array(orig), cv2.COLOR_RGB2GRAY)
    alpha = refine_alpha(prob, gray, args)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((alpha * 255).clip(0, 255).astype(np.uint8), mode="L").save(out_path)


def backup_json_once(path: Path, suffix: str) -> None:
    backup = path.with_name(path.name + suffix)
    if path.exists() and not backup.exists():
        backup.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")


def update_json_sidecars(dataset_dir: Path, mask_dir: Path, args: argparse.Namespace) -> None:
    try:
        mask_dir_rel = mask_dir.relative_to(dataset_dir)
    except ValueError:
        mask_dir_rel = mask_dir

    transforms_path = dataset_dir / "transforms.json"
    if args.update_transforms and transforms_path.exists():
        data = json.loads(transforms_path.read_text(encoding="utf-8"))
        added = missing = 0
        for frame in data.get("frames", []):
            name = Path(frame["file_path"]).name
            if (mask_dir / name).exists():
                frame["mask_path"] = str(mask_dir_rel / name).replace("\\", "/")
                added += 1
            else:
                missing += 1
        backup_json_once(transforms_path, ".before_fine_masks")
        transforms_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        print(f"[INFO] transforms.json updated: added={added}, missing={missing}")

    dataset_path = dataset_dir / "dataset.json"
    if args.update_dataset and dataset_path.exists():
        data = json.loads(dataset_path.read_text(encoding="utf-8"))
        data["has_masks"] = True
        data["mask_dir"] = str(mask_dir_rel).replace("\\", "/")
        data["mask_generation"] = "BiRefNet_HR-matting fine mask"
        backup_json_once(dataset_path, ".before_fine_masks")
        dataset_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        print("[INFO] dataset.json updated")

    metadata_path = dataset_dir / "metadata.json"
    if args.update_metadata and metadata_path.exists():
        data = json.loads(metadata_path.read_text(encoding="utf-8"))
        data["fine_masks"] = {
            "mask_dir": str(mask_dir_rel).replace("\\", "/"),
            "model": args.model,
            "inference_size": args.inference_size,
            "contrast": args.contrast,
            "brightness": args.brightness,
            "final_min_probability": args.final_min_probability,
            "guided_filter_radius": args.guided_filter_radius,
            "guided_filter_eps": args.guided_filter_eps,
            "core_confidence_thresh": args.core_confidence_thresh,
            "edge_band_px": args.edge_band_px,
            "pin_mount_removal": not args.no_pin_mount_removal,
            "binary_output": args.binary_output,
        }
        backup_json_once(metadata_path, ".before_fine_masks")
        metadata_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        print("[INFO] metadata.json updated")


def main() -> int:
    p = argparse.ArgumentParser(description="Fine BiRefNet_HR-matting masks for DISC3D exports.")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--dataset-dir")
    g.add_argument("--images-dir")

    p.add_argument("--masks-dir")
    p.add_argument("--images-subdir", default="images")
    p.add_argument("--masks-subdir", default="masks_fine")
    p.add_argument("--patterns", nargs="+", default=["*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff"])

    p.add_argument("--model", default="ZhengPeng7/BiRefNet_HR-matting")
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--inference-size", type=int, default=2048)
    p.add_argument("--no-half", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--limit", type=int)

    p.add_argument("--contrast", type=float, default=2.0)
    p.add_argument("--brightness", type=float, default=1.0)
    p.add_argument("--final-min-probability", type=float, default=0.03)
    p.add_argument("--guided-filter-radius", type=int, default=8)
    p.add_argument("--guided-filter-eps", type=float, default=1e-3)
    p.add_argument("--core-confidence-thresh", type=float, default=0.75)
    p.add_argument("--edge-band-px", type=int, default=30)

    p.add_argument("--no-pin-mount-removal", action="store_true")
    p.add_argument("--component-min-area", type=int, default=30)
    p.add_argument("--component-dilation-kernel", type=int, default=15)

    p.add_argument("--binary-output", action="store_true")
    p.add_argument("--binary-threshold", type=float, default=0.5)

    p.add_argument("--update-transforms", action="store_true")
    p.add_argument("--update-dataset", action="store_true")
    p.add_argument("--update-metadata", action="store_true")

    args = p.parse_args()

    if args.dataset_dir:
        dataset_dir = Path(args.dataset_dir).expanduser().resolve()
        image_dir = dataset_dir / args.images_subdir
        mask_dir = dataset_dir / args.masks_subdir if not args.masks_dir else Path(args.masks_dir).expanduser().resolve()
    else:
        dataset_dir = None
        image_dir = Path(args.images_dir).expanduser().resolve()
        mask_dir = Path(args.masks_dir).expanduser().resolve() if args.masks_dir else image_dir.parent / args.masks_subdir

    if not image_dir.exists():
        raise FileNotFoundError(image_dir)
    mask_dir.mkdir(parents=True, exist_ok=True)

    files: list[Path] = []
    for pattern in args.patterns:
        files.extend(image_dir.glob(pattern))
    files = sorted(set(files), key=natural_key)
    if args.limit:
        files = files[: args.limit]
    if not files:
        raise RuntimeError(f"No images found in {image_dir}")

    device = resolve_device(args.device)
    use_half = device.type == "cuda" and not args.no_half

    print(f"[INFO] image_dir={image_dir}")
    print(f"[INFO] mask_dir={mask_dir}")
    print(f"[INFO] images={len(files)}")
    print(f"[INFO] model={args.model}")
    print(f"[INFO] device={device}, fp16={use_half}")

    model = AutoModelForImageSegmentation.from_pretrained(args.model, trust_remote_code=True)
    model = model.to(device).eval()
    if use_half:
        model = model.half()

    wrote = 0
    for img_path in tqdm(files, desc="fine masks"):
        out_path = mask_dir / img_path.name
        if out_path.exists() and not args.overwrite:
            continue
        infer_one(img_path, out_path, model, device, use_half, args)
        wrote += 1

    if dataset_dir:
        update_json_sidecars(dataset_dir, mask_dir, args)

    print(f"[DONE] wrote={wrote}, mask_dir={mask_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
