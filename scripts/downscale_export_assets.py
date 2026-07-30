#!/usr/bin/env python3

from __future__ import annotations

import argparse
import re
from pathlib import Path

from PIL import Image
from tqdm import tqdm


IMAGE_NUMBER_RE = re.compile(r"image[_-]?0*(\d+)", re.IGNORECASE)


def natural_key(path: Path) -> tuple[int, str]:
    match = IMAGE_NUMBER_RE.search(path.name)
    if match:
        return int(match.group(1)), path.name
    return 10**12, path.name


def resampling_filter(name: str):
    name = name.lower()
    if name == "nearest":
        return Image.Resampling.NEAREST
    if name == "lanczos":
        return Image.Resampling.LANCZOS
    if name == "bilinear":
        return Image.Resampling.BILINEAR
    raise ValueError(f"Unknown resampling filter: {name}")


def downscale_one(
    src: Path,
    dst: Path,
    factor: int,
    resample,
    overwrite: bool,
    threshold: int | None = None,
) -> None:
    if dst.exists() and not overwrite:
        return

    dst.parent.mkdir(parents=True, exist_ok=True)

    with Image.open(src) as img:
        if threshold is not None:
            img = img.convert("L")
            img = img.point(lambda p: 255 if p >= threshold else 0)

        new_size = (img.width // factor, img.height // factor)
        out = img.resize(new_size, resample=resample)

        save_kwargs = {}
        if "exif" in img.info:
            save_kwargs["exif"] = img.info["exif"]

        if dst.suffix.lower() in {".jpg", ".jpeg"}:
            out.save(dst, quality=95, subsampling=0, **save_kwargs)
        else:
            out.save(dst, **save_kwargs)


def downscale_folder(
    src_dir: Path,
    dst_dir: Path,
    factor: int,
    pattern: str,
    resample_name: str,
    overwrite: bool,
    threshold: int | None = None,
) -> int:
    if not src_dir.exists():
        print(f"[SKIP] Missing source folder: {src_dir}")
        return 0

    paths = sorted(src_dir.glob(pattern), key=natural_key)
    if not paths:
        print(f"[SKIP] No files found in {src_dir} with pattern {pattern}")
        return 0

    resample = resampling_filter(resample_name)

    for src in tqdm(paths, desc=f"{src_dir.name} -> {dst_dir.name}"):
        dst = dst_dir / src.name
        downscale_one(
            src=src,
            dst=dst,
            factor=factor,
            resample=resample,
            overwrite=overwrite,
            threshold=threshold,
        )

    return len(paths)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create images_2/masks_2 style folders for DISC3D exports."
    )
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--factor", type=int, default=2)
    parser.add_argument("--image-src", default="images")
    parser.add_argument("--image-dst", default=None)
    parser.add_argument("--mask-src", default="masks")
    parser.add_argument("--mask-dst", default=None)
    parser.add_argument("--pattern", default="*.png")
    parser.add_argument("--overwrite", action="store_true")

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
        default=None,
        help="Optional threshold for masks before downscaling, e.g. 128.",
    )

    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir).expanduser().resolve()

    image_src = dataset_dir / args.image_src
    image_dst = dataset_dir / (args.image_dst or f"{args.image_src}_{args.factor}")

    mask_src = dataset_dir / args.mask_src
    mask_dst = dataset_dir / (args.mask_dst or f"{args.mask_src}_{args.factor}")

    print(f"[INFO] dataset: {dataset_dir}")
    print(f"[INFO] factor:  {args.factor}")
    print(f"[INFO] images:  {image_src} -> {image_dst}")
    print(f"[INFO] masks:   {mask_src} -> {mask_dst}")

    n_images = downscale_folder(
        src_dir=image_src,
        dst_dir=image_dst,
        factor=args.factor,
        pattern=args.pattern,
        resample_name=args.image_resample,
        overwrite=args.overwrite,
        threshold=None,
    )

    n_masks = downscale_folder(
        src_dir=mask_src,
        dst_dir=mask_dst,
        factor=args.factor,
        pattern=args.pattern,
        resample_name=args.mask_resample,
        overwrite=args.overwrite,
        threshold=args.threshold_mask,
    )

    print(f"[DONE] images={n_images}, masks={n_masks}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
