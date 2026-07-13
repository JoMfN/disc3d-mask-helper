#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForImageSegmentation
from torchvision.transforms.functional import normalize, to_tensor


IMAGE_NUMBER_RE = re.compile(r"image[_-]?0*(\d+)", re.IGNORECASE)


def natural_image_key(path: Path) -> tuple[int, str]:
    match = IMAGE_NUMBER_RE.search(path.name)
    if match:
        return int(match.group(1)), path.name
    numbers = re.findall(r"\d+", path.name)
    if numbers:
        return int(numbers[0]), path.name
    return 10**12, path.name


def resolve_paths(
    dataset_dir: str | Path | None = None,
    images_dir: str | Path | None = None,
    masks_dir: str | Path | None = None,
    images_subdir: str = "images",
    masks_subdir: str = "masks",
) -> tuple[Path, Path]:
    if dataset_dir is not None:
        dataset_dir = Path(dataset_dir).expanduser().resolve()
        image_dir = dataset_dir / images_subdir
        mask_dir = dataset_dir / masks_subdir
    else:
        if images_dir is None:
            raise ValueError("Use either dataset_dir or images_dir.")
        image_dir = Path(images_dir).expanduser().resolve()
        if masks_dir is None:
            mask_dir = image_dir.parent / masks_subdir
        else:
            mask_dir = Path(masks_dir).expanduser().resolve()

    return image_dir, mask_dir


def get_model_output_mask(output) -> np.ndarray:
    """Normalize BiRefNet-style output to a single [H, W] probability mask."""
    if isinstance(output, (list, tuple)):
        output = output[-1]
    if isinstance(output, (list, tuple)):
        output = output[-1]

    if output.ndim == 4:
        output = output[0, 0]
    elif output.ndim == 3:
        output = output[0]

    return output.sigmoid().detach().cpu().float().numpy()


def process_image(
    img_path: Path,
    mask_path: Path,
    model,
    device: torch.device,
    input_size: int,
    threshold: float | None,
) -> None:
    orig = Image.open(img_path).convert("RGB")
    w, h = orig.size

    inp = orig.resize((input_size, input_size), Image.BILINEAR)
    tensor = normalize(
        to_tensor(inp),
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225],
    ).unsqueeze(0).to(device)

    with torch.inference_mode():
        pred = get_model_output_mask(model(tensor))

    if threshold is not None:
        mask = (pred >= threshold).astype(np.uint8) * 255
    else:
        mask = (pred * 255).clip(0, 255).astype(np.uint8)

    mask_img = Image.fromarray(mask, mode="L").resize((w, h), Image.BILINEAR)
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    mask_img.save(mask_path)


def generate_masks(
    dataset_dir: str | Path | None = None,
    images_dir: str | Path | None = None,
    masks_dir: str | Path | None = None,
    images_subdir: str = "images",
    masks_subdir: str = "masks",
    model_name: str = "ZhengPeng7/BiRefNet",
    device_name: str = "cuda",
    input_size: int = 1024,
    pattern: str = "*.png",
    overwrite: bool = False,
    threshold: float | None = None,
) -> dict:
    image_dir, mask_dir = resolve_paths(
        dataset_dir=dataset_dir,
        images_dir=images_dir,
        masks_dir=masks_dir,
        images_subdir=images_subdir,
        masks_subdir=masks_subdir,
    )

    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory does not exist: {image_dir}")

    mask_dir.mkdir(parents=True, exist_ok=True)

    if device_name.startswith("cuda") and not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(device_name)

    image_paths = sorted(image_dir.glob(pattern), key=natural_image_key)
    if not image_paths:
        raise FileNotFoundError(f"No images found in {image_dir} with pattern {pattern}")

    print(f"[INFO] Image dir: {image_dir}", flush=True)
    print(f"[INFO] Mask dir:  {mask_dir}", flush=True)
    print(f"[INFO] Model:     {model_name}", flush=True)
    print(f"[INFO] Device:    {device}", flush=True)
    print(f"[INFO] Images:    {len(image_paths)}", flush=True)

    model = AutoModelForImageSegmentation.from_pretrained(
        model_name,
        trust_remote_code=True,
    )
    model = model.to(device).eval().float()

    generated = 0
    skipped = 0

    for img_path in tqdm(image_paths, desc="Generating masks"):
        mask_path = mask_dir / img_path.name
        if mask_path.exists() and not overwrite:
            skipped += 1
            continue

        process_image(
            img_path=img_path,
            mask_path=mask_path,
            model=model,
            device=device,
            input_size=input_size,
            threshold=threshold,
        )
        generated += 1

    summary = {
        "image_dir": str(image_dir),
        "mask_dir": str(mask_dir),
        "images": len(image_paths),
        "generated": generated,
        "skipped": skipped,
        "device": str(device),
        "model": model_name,
    }
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate DISC3D specimen masks with BiRefNet.")
    parser.add_argument("--dataset-dir", default=None, help="Specimen export folder containing images/.")
    parser.add_argument("--images-dir", default=None, help="Explicit image directory. Alternative to --dataset-dir.")
    parser.add_argument("--masks-dir", default=None, help="Explicit output mask directory.")
    parser.add_argument("--images-subdir", default="images")
    parser.add_argument("--masks-subdir", default="masks")
    parser.add_argument("--model", default="ZhengPeng7/BiRefNet")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--input-size", type=int, default=1024)
    parser.add_argument("--pattern", default="*.png")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--threshold", type=float, default=None)
    args = parser.parse_args()

    generate_masks(
        dataset_dir=args.dataset_dir,
        images_dir=args.images_dir,
        masks_dir=args.masks_dir,
        images_subdir=args.images_subdir,
        masks_subdir=args.masks_subdir,
        model_name=args.model,
        device_name=args.device,
        input_size=args.input_size,
        pattern=args.pattern,
        overwrite=args.overwrite,
        threshold=args.threshold,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
