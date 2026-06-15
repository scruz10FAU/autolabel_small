#!/usr/bin/env python3
"""
Remove near-duplicate images (and their YOLO labels) from a training dataset.

Two images are considered duplicates only when BOTH conditions hold:
  1. Structurally similar — perceptual hash (pHash) distance <= --hash-thresh
  2. Same color distribution — hue-histogram correlation >= --color-thresh

Images that are structurally identical but differ in color (e.g., same scene
with a red buoy vs a green buoy) satisfy condition 1 but NOT condition 2,
so they are kept.

Usage:
    python dedup_dataset.py
    python dedup_dataset.py -i trainImagesZed/images -l trainImagesZed/labels
    python dedup_dataset.py --dry-run
"""

import os
import cv2
import numpy as np
import argparse
from pathlib import Path


def compute_phash(img, hash_size=8):
    """
    Perceptual hash over the grayscale DCT of the image.
    Captures scene structure; color differences do NOT affect the hash.
    Returns a boolean array of length hash_size².
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (hash_size * 4, hash_size * 4))
    dct = cv2.dct(np.float32(resized))
    dct_low = dct[:hash_size, :hash_size]
    median = np.median(dct_low)
    return (dct_low > median).flatten()


def hash_distance(h1, h2):
    return int(np.sum(h1 != h2))


def hue_histogram_correlation(img1, img2):
    """
    Pearson correlation of the hue-channel histograms.
    Returns 1.0 for identical hue distributions, lower for different colors.
    Red vs green buoys typically score < 0.5; same-color near-duplicates > 0.9.
    """
    hsv1 = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
    hsv2 = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)
    h1 = cv2.calcHist([hsv1], [0], None, [36], [0, 180])
    h2 = cv2.calcHist([hsv2], [0], None, [36], [0, 180])
    cv2.normalize(h1, h1)
    cv2.normalize(h2, h2)
    return float(cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL))


def find_duplicates(image_paths, hash_thresh, color_thresh):
    """
    Returns a set of image Paths to remove.

    Steps:
      1. Compute pHash for every image.
      2. Union-Find: group images whose pHash distance <= hash_thresh (same structure).
      3. Within each group, sub-group by hue histogram correlation >= color_thresh.
         Images in the same color sub-group are true duplicates — keep the first
         (alphabetically), mark the rest for removal.
    """
    print(f"Loading and hashing {len(image_paths)} images...")

    images = {}
    hashes = {}
    for p in image_paths:
        img = cv2.imread(str(p))
        if img is None:
            print(f"  Warning: could not read {p.name}, skipping")
            continue
        images[p] = img
        hashes[p] = compute_phash(img)

    valid = list(images.keys())

    # --- Union-Find ---
    parent = {p: p for p in valid}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        parent[find(x)] = find(y)

    print(f"Comparing pairs (hash_thresh={hash_thresh}, color_thresh={color_thresh})...")
    for i in range(len(valid)):
        for j in range(i + 1, len(valid)):
            if hash_distance(hashes[valid[i]], hashes[valid[j]]) <= hash_thresh:
                union(valid[i], valid[j])

    # Collect groups with more than one image
    groups: dict = {}
    for p in valid:
        groups.setdefault(find(p), []).append(p)
    groups = [g for g in groups.values() if len(g) > 1]
    print(f"  {len(groups)} structurally similar group(s) found")

    to_remove = set()

    for group in groups:
        # Sub-group by color: greedy — compare each image to the representative
        # of existing color buckets
        color_buckets: list[list] = []

        for p in sorted(group, key=lambda x: x.name):
            placed = False
            for bucket in color_buckets:
                rep = bucket[0]
                if hue_histogram_correlation(images[p], images[rep]) >= color_thresh:
                    bucket.append(p)
                    placed = True
                    break
            if not placed:
                color_buckets.append([p])

        # Within each color bucket: keep the first, remove the rest
        for bucket in color_buckets:
            if len(bucket) > 1:
                to_remove.update(bucket[1:])

    return to_remove


def main():
    parser = argparse.ArgumentParser(
        description="Remove near-duplicate images and labels from a training dataset"
    )
    parser.add_argument('-i', '--images', default='trainImagesZed/images',
                        help="Directory of training images (default: trainImagesZed/images)")
    parser.add_argument('-l', '--labels', default='trainImagesZed/labels',
                        help="Directory of YOLO label files (default: trainImagesZed/labels)")
    parser.add_argument('--hash-thresh', type=int, default=8,
                        help="Max pHash hamming distance to consider two images structurally "
                             "similar; 0–64 range, lower = stricter (default: 8)")
    parser.add_argument('--color-thresh', type=float, default=0.90,
                        help="Min hue-histogram correlation to consider two images the same "
                             "color; 0–1 range, higher = stricter (default: 0.90)")
    parser.add_argument('--dry-run', action='store_true',
                        help="Print what would be deleted without deleting anything")
    parser.add_argument('--root', '-r', default=None,
                        help="Root dataset directory containing images/ and labels/ subdirs")
    args = parser.parse_args()

    if args.root:
        img_dir = Path(args.root) / 'images'
        lbl_dir = Path(args.root) / 'labels'
    else:
        img_dir = Path(args.images)
        lbl_dir = Path(args.labels)

    if not img_dir.exists():
        print(f"Error: images directory not found: {img_dir}")
        return

    exts = {'.jpg', '.jpeg', '.png', '.bmp'}
    image_paths = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in exts)

    if not image_paths:
        print(f"No images found in {img_dir}")
        return

    print(f"Images : {img_dir}  ({len(image_paths)} files)")
    print(f"Labels : {lbl_dir}")
    if args.dry_run:
        print("DRY RUN — no files will be deleted\n")
    else:
        print()

    to_remove = find_duplicates(image_paths, args.hash_thresh, args.color_thresh)

    if not to_remove:
        print("\nNo duplicates found.")
        return

    removed_imgs = 0
    removed_lbls = 0

    print(f"\n{len(to_remove)} duplicate(s) to remove:")
    for img_path in sorted(to_remove, key=lambda p: p.name):
        lbl_path = lbl_dir / (img_path.stem + '.txt')
        has_label = lbl_path.exists()
        suffix = f"  +  {lbl_path.name}" if has_label else ""
        print(f"  {img_path.name}{suffix}")

        if not args.dry_run:
            img_path.unlink()
            removed_imgs += 1
            if has_label:
                lbl_path.unlink()
                removed_lbls += 1

    if args.dry_run:
        print(f"\nDry run complete — would remove {len(to_remove)} image(s) and their labels.")
    else:
        print(f"\nDone — removed {removed_imgs} image(s) and {removed_lbls} label file(s).")


if __name__ == '__main__':
    main()
