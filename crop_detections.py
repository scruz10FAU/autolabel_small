"""
Crop bounding boxes from labeled training images and save each crop as its own image.

For each image/label pair, every bounding box is extracted and saved as a separate
cropped image. Crops can optionally be organized into per-class subdirectories.

Usage:
    python crop_detections.py --root trainImagesZed
    python crop_detections.py -i images -l labels -o crops --by-class -c classes.json
"""

import argparse
import json
import cv2
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="Crop YOLO bounding boxes into individual images")
    p.add_argument("--root", "-r", default=None,
                   help="Root dataset directory containing images/ and labels/ subdirs")
    p.add_argument("--images", "-i", default="trainImagesZed/images",
                   help="Directory of training images (default: trainImagesZed/images)")
    p.add_argument("--labels", "-l", default="trainImagesZed/labels",
                   help="Directory of YOLO label files (default: trainImagesZed/labels)")
    p.add_argument("--output", "-o", default="crops",
                   help="Output directory for cropped images (default: crops)")
    p.add_argument("--classes", "-c", default=None,
                   help="JSON file mapping class IDs to names; used for subfolder names with --by-class")
    p.add_argument("--by-class", action="store_true", dest="by_class",
                   help="Organize crops into subdirectories by class name")
    p.add_argument("--padding", type=float, default=0.0,
                   help="Fractional padding to add around each crop (e.g. 0.1 adds 10%% on each side, default: 0)")
    p.add_argument("--dry-run", action="store_true",
                   help="Print what would be saved without writing any files")
    return p.parse_args()


def load_classes(path):
    with open(path) as f:
        raw = json.load(f)
    return {int(k): v["name"] for k, v in raw.items()}


def main():
    args = parse_args()

    if args.root:
        img_dir = Path(args.root) / "images"
        lbl_dir = Path(args.root) / "labels"
    else:
        img_dir = Path(args.images)
        lbl_dir = Path(args.labels)

    out_dir = Path(args.output)

    class_names = {}
    if args.classes:
        class_names = load_classes(args.classes)

    if not img_dir.exists():
        print(f"Error: images directory not found: {img_dir}")
        return

    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    image_paths = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in exts)

    if not image_paths:
        print(f"No images found in {img_dir}")
        return

    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Images : {img_dir}  ({len(image_paths)} files)")
    print(f"Labels : {lbl_dir}")
    print(f"Output : {out_dir}")
    if args.dry_run:
        print("DRY RUN — no files will be written\n")
    else:
        print()

    total_crops = 0
    skipped = 0

    for img_path in image_paths:
        lbl_path = lbl_dir / (img_path.stem + ".txt")
        if not lbl_path.exists() or not lbl_path.read_text().strip():
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  Warning: could not read {img_path.name}, skipping")
            continue

        h, w = img.shape[:2]

        lines = [l.strip() for l in lbl_path.read_text().splitlines() if l.strip()]
        for crop_idx, line in enumerate(lines):
            parts = line.split()
            if len(parts) != 5:
                continue
            cls, xc, yc, bw, bh = int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])

            x1 = (xc - bw / 2) * w
            y1 = (yc - bh / 2) * h
            x2 = (xc + bw / 2) * w
            y2 = (yc + bh / 2) * h

            if args.padding > 0:
                pad_x = (x2 - x1) * args.padding
                pad_y = (y2 - y1) * args.padding
                x1 -= pad_x
                y1 -= pad_y
                x2 += pad_x
                y2 += pad_y

            x1, y1 = max(0, int(x1)), max(0, int(y1))
            x2, y2 = min(w, int(x2)), min(h, int(y2))

            if x2 <= x1 or y2 <= y1:
                skipped += 1
                continue

            crop = img[y1:y2, x1:x2]
            cls_name = class_names.get(cls, str(cls))
            crop_name = f"{img_path.stem}_crop{crop_idx}{img_path.suffix}"

            if args.by_class:
                dest = out_dir / cls_name / crop_name
            else:
                dest = out_dir / crop_name

            print(f"  {dest}")
            if not args.dry_run:
                dest.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(dest), crop)
            total_crops += 1

    if args.dry_run:
        print(f"\nDry run — would save {total_crops} crop(s).")
    else:
        print(f"\nDone — saved {total_crops} crop(s) to '{out_dir}'.")
    if skipped:
        print(f"Skipped {skipped} degenerate box(es).")


if __name__ == "__main__":
    main()
