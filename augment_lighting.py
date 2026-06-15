#!/usr/bin/env python3
"""
Augment training images with lighting variations to simulate different environments.

For each image, generates augmented copies using named lighting presets and/or
random parameter sampling. YOLO label files are copied unchanged — bounding box
coordinates are unaffected by lighting-only augmentations.

Presets: overcast, sunny, sunset, dim, foggy  (use --preset, repeatable)
Random:  --count N generates N randomly parameterized augmentations per image

Usage:
    python augment_lighting.py
    python augment_lighting.py -i images -l labels -n 4
    python augment_lighting.py --preset overcast --preset sunset --count 0
    python augment_lighting.py --preset foggy -n 3 --seed 42
"""

import cv2
import numpy as np
import argparse
import shutil
from pathlib import Path


# ── Augmentation primitives ───────────────────────────────────────────────────

def adjust_brightness(img: np.ndarray, factor: float) -> np.ndarray:
    """Multiply every pixel by factor. factor > 1 brightens, < 1 darkens."""
    return np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)


def adjust_gamma(img: np.ndarray, gamma: float) -> np.ndarray:
    """Gamma correction via LUT. gamma < 1 lifts midtones, > 1 crushes them."""
    table = np.array([(i / 255.0) ** gamma * 255 for i in range(256)], dtype=np.uint8)
    return cv2.LUT(img, table)


def adjust_contrast(img: np.ndarray, factor: float) -> np.ndarray:
    """Stretch or compress pixel values around the image mean."""
    mean = img.astype(np.float32).mean()
    return np.clip((img.astype(np.float32) - mean) * factor + mean, 0, 255).astype(np.uint8)


def adjust_saturation(img: np.ndarray, factor: float) -> np.ndarray:
    """Scale the HSV saturation channel. factor > 1 more vivid, < 1 more grey."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * factor, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def color_temperature(img: np.ndarray, warmth: float) -> np.ndarray:
    """
    Shift color balance toward warm (positive) or cool (negative) tones.
    warmth range: roughly -60 (icy blue) to +60 (warm orange).
    """
    out = img.astype(np.float32).copy()
    if warmth > 0:
        out[:, :, 2] = np.clip(out[:, :, 2] + warmth,       0, 255)  # R up
        out[:, :, 1] = np.clip(out[:, :, 1] + warmth * 0.4, 0, 255)  # G slightly
        out[:, :, 0] = np.clip(out[:, :, 0] - warmth * 0.4, 0, 255)  # B down
    else:
        w = abs(warmth)
        out[:, :, 0] = np.clip(out[:, :, 0] + w,       0, 255)  # B up
        out[:, :, 1] = np.clip(out[:, :, 1] + w * 0.2, 0, 255)  # G slightly
        out[:, :, 2] = np.clip(out[:, :, 2] - w * 0.4, 0, 255)  # R down
    return out.astype(np.uint8)


def add_fog(img: np.ndarray, intensity: float) -> np.ndarray:
    """Blend toward a light grey to simulate haze or fog. intensity in [0, 1]."""
    haze = np.full_like(img, 200)
    return cv2.addWeighted(img, 1.0 - intensity, haze, intensity, 0)


def add_shadow(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Darken a random convex polygon region to simulate a cast shadow."""
    h, w = img.shape[:2]
    n_pts = int(rng.integers(3, 7))
    # Limit shadow to at most half the image in each dimension, at a random offset
    max_w, max_h = w // 2, h // 2
    ox = int(rng.integers(0, w - max_w))
    oy = int(rng.integers(0, h - max_h))
    pts = rng.integers(0, [max_w, max_h], size=(n_pts, 2))
    pts[:, 0] += ox
    pts[:, 1] += oy
    pts = cv2.convexHull(pts.reshape(-1, 1, 2).astype(np.int32))
    mask = np.ones(img.shape, dtype=np.float32)
    shadow_strength = float(rng.uniform(0.3, 0.6))
    cv2.fillPoly(mask, [pts], (shadow_strength,) * 3)
    return np.clip(img.astype(np.float32) * mask, 0, 255).astype(np.uint8)


# ── Named presets ─────────────────────────────────────────────────────────────

PRESETS: dict[str, dict] = {
    "overcast": {
        "brightness": 0.85,
        "contrast":   0.85,
        "saturation": 0.70,
        "temperature": -15,
    },
    "sunny": {
        "brightness": 1.25,
        "contrast":   1.15,
        "saturation": 1.20,
        "temperature": 20,
    },
    "sunset": {
        "brightness":  0.90,
        "saturation":  1.30,
        "temperature": 45,
        "gamma":       1.10,
    },
    "dim": {
        "brightness": 0.50,
        "contrast":   0.80,
        "saturation": 0.75,
        "gamma":      1.30,
    },
    "foggy": {
        "fog":        0.38,
        "contrast":   0.75,
        "saturation": 0.55,
        "brightness": 0.95,
    },
}

AUGMENTATION_ORDER = [
    "brightness", "gamma", "contrast", "saturation", "temperature", "fog", "shadow"
]


def apply_preset(img: np.ndarray, preset_name: str,
                 rng: np.random.Generator) -> np.ndarray:
    p = PRESETS[preset_name]
    result = img.copy()
    for key in AUGMENTATION_ORDER:
        if key == "shadow" and p.get("shadow", False):
            result = add_shadow(result, rng)
        elif key == "brightness"  and "brightness"  in p:
            result = adjust_brightness(result, p["brightness"])
        elif key == "gamma"       and "gamma"       in p:
            result = adjust_gamma(result, p["gamma"])
        elif key == "contrast"    and "contrast"    in p:
            result = adjust_contrast(result, p["contrast"])
        elif key == "saturation"  and "saturation"  in p:
            result = adjust_saturation(result, p["saturation"])
        elif key == "temperature" and "temperature" in p:
            result = color_temperature(result, p["temperature"])
        elif key == "fog"         and "fog"         in p:
            result = add_fog(result, p["fog"])
    return result


def apply_random(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Random combination of augmentations sampled from realistic ranges."""
    result = img.copy()
    result = adjust_brightness(result, float(rng.uniform(0.55, 1.45)))
    if rng.random() > 0.4:
        result = adjust_gamma(result, float(rng.uniform(0.65, 1.50)))
    result = adjust_contrast(result, float(rng.uniform(0.70, 1.35)))
    result = adjust_saturation(result, float(rng.uniform(0.55, 1.45)))
    result = color_temperature(result, float(rng.uniform(-40, 40)))
    if rng.random() > 0.55:
        result = add_fog(result, float(rng.uniform(0.05, 0.40)))
    if rng.random() > 0.65:
        result = add_shadow(result, rng)
    return result


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Augment training images with lighting variations"
    )
    parser.add_argument('-i', '--images', default='trainImagesZed/images',
                        help="Directory of training images (default: trainImagesZed/images)")
    parser.add_argument('-l', '--labels', default='trainImagesZed/labels',
                        help="Directory of YOLO label files (default: trainImagesZed/labels)")
    parser.add_argument('--out-images', default=None,
                        help="Output directory for augmented images (default: same as --images)")
    parser.add_argument('--out-labels', default=None,
                        help="Output directory for augmented labels (default: same as --labels)")
    parser.add_argument('--preset', action='append', dest='presets', default=[],
                        choices=list(PRESETS.keys()), metavar='PRESET',
                        help=f"Named lighting preset to apply; repeatable. "
                             f"Choices: {', '.join(PRESETS)}")
    parser.add_argument('-n', '--count', type=int, default=None,
                        help="Number of random augmentations per image "
                             "(default: 3 when no preset is given, 0 when a preset is given); "
                             "set to 0 to use presets only")
    parser.add_argument('--seed', type=int, default=None,
                        help="Random seed for reproducibility")
    parser.add_argument('--dry-run', action='store_true',
                        help="Print what would be created without writing any files")
    parser.add_argument('--root', '-r', default=None,
                        help="Root input directory containing images/ and labels/ subdirs; overrides -i and -l")
    parser.add_argument('--out-root', '-or', default=None,
                        help="Root output directory; saves to <out-root>/images and <out-root>/labels, overrides --out-images and --out-labels")
    args = parser.parse_args()

    if args.count is None:
        args.count = 0 if args.presets else 3

    if args.root:
        img_dir = Path(args.root) / 'images'
        lbl_dir = Path(args.root) / 'labels'
    else:
        img_dir = Path(args.images)
        lbl_dir = Path(args.labels)

    if args.out_root:
        out_img_dir = Path(args.out_root) / 'images'
        out_lbl_dir = Path(args.out_root) / 'labels'
    elif args.out_images or args.out_labels:
        out_img_dir = Path(args.out_images) if args.out_images else img_dir
        out_lbl_dir = Path(args.out_labels) if args.out_labels else lbl_dir
    else:
        out_img_dir = img_dir
        out_lbl_dir = lbl_dir

    if not img_dir.exists():
        print(f"Error: images directory not found: {img_dir}")
        return

    exts = {'.jpg', '.jpeg', '.png', '.bmp'}
    image_paths = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in exts)

    if not image_paths:
        print(f"No images found in {img_dir}")
        return

    if not args.presets and args.count == 0:
        print("Nothing to do: specify at least one --preset or --count > 0.")
        return

    rng = np.random.default_rng(args.seed)

    if not args.dry_run:
        out_img_dir.mkdir(parents=True, exist_ok=True)
        out_lbl_dir.mkdir(parents=True, exist_ok=True)

    print(f"Images : {img_dir}  ({len(image_paths)} files)")
    print(f"Labels : {lbl_dir}")
    print(f"Output : {out_img_dir}  /  {out_lbl_dir}")
    if args.presets:
        print(f"Presets: {', '.join(args.presets)}")
    if args.count > 0:
        print(f"Random : {args.count} per image")
    if args.dry_run:
        print("DRY RUN — no files will be written\n")
    else:
        print()

    total_written = 0

    for img_path in image_paths:
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  Warning: could not read {img_path.name}, skipping")
            continue

        src_lbl = lbl_dir / (img_path.stem + '.txt')
        has_label = src_lbl.exists()

        augmentations: list[tuple[str, np.ndarray]] = []

        for preset_name in args.presets:
            augmentations.append((f"aug_{preset_name}", apply_preset(img, preset_name, rng)))

        for i in range(args.count):
            augmentations.append((f"aug_rand{i + 1}", apply_random(img, rng)))

        for suffix, aug_img in augmentations:
            out_img = out_img_dir / f"{img_path.stem}_{suffix}{img_path.suffix}"
            out_lbl = out_lbl_dir / f"{img_path.stem}_{suffix}.txt"
            print(f"  {out_img.name}" + (f"  +  {out_lbl.name}" if has_label else ""))

            if not args.dry_run:
                cv2.imwrite(str(out_img), aug_img)
                if has_label:
                    shutil.copy(src_lbl, out_lbl)
                total_written += 1

    if args.dry_run:
        print(f"\nDry run complete — would create "
              f"{len(image_paths) * (len(args.presets) + args.count)} image(s).")
    else:
        print(f"\nDone — wrote {total_written} augmented image(s).")


if __name__ == '__main__':
    main()
