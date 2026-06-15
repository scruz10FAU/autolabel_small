import argparse
import json
import os
import math
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from matplotlib.widgets import Button

parser = argparse.ArgumentParser(description='View and select training images with YOLO labels')
parser.add_argument('--img_dir', '-i', type=str, default='training_data/images',
                    help='Directory of training images')
parser.add_argument('--label_dir', '-l', type=str, default='training_data/labels',
                    help='Directory of YOLO label files')
parser.add_argument('--output_file', '-o', type=str, default='to_update_zed.txt',
                    help='Output file to append selected image paths')
parser.add_argument('--classes', '-c', type=str, default='classes.json',
                    help='JSON file mapping class IDs to name and color')
parser.add_argument('--start-batch', '-b', type=int, default=1,
                    help='Batch number to start on (1-indexed, default: 1)')
parser.add_argument('--root', '-r', type=str, default=None,
                    help='Root dataset directory containing images/ and labels/ subdirs')
args = parser.parse_args()

if args.root:
    img_dir   = Path(args.root) / 'images'
    label_dir = Path(args.root) / 'labels'
else:
    img_dir   = Path(args.img_dir)
    label_dir = Path(args.label_dir)
output_file = Path(args.output_file)
with open(args.classes) as f:
    raw = json.load(f)
class_names = {int(k): v['name'] for k, v in raw.items()}

cols = 4
rows = 4
batch_size = cols * rows

def get_font(image_height):
    size = max(16, image_height // 15)
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()

# Get all image files
image_files = sorted([f for f in img_dir.glob("*.*") if f.suffix.lower() in ['.jpg', '.png']])
total_images = len(image_files)
total_batches = math.ceil(total_images / batch_size)

print(f"Total images: {total_images} | Batches: {total_batches}")
print(f"Click an image to select/deselect it, then press 'Save & Continue'.")

all_selected = set()
user_quit = [False]

start_batch = max(0, min(args.start_batch - 1, total_batches - 1))
if start_batch > 0:
    print(f"Starting at batch {start_batch + 1}/{total_batches}.")

for batch_idx in range(start_batch, total_batches):
    if user_quit[0]:
        break
    start = batch_idx * batch_size
    end = min(start + batch_size, total_images)
    batch_files = image_files[start:end]
    n = len(batch_files)

    fig = plt.figure(figsize=(15, 13))
    fig.canvas.manager.set_window_title(
        f"Batch {batch_idx + 1}/{total_batches} — Click to select | Save & Continue when done"
    )
    try:
        fig.canvas.manager.window.wm_geometry("+0+0")  # Tk backend
    except AttributeError:
        try:
            fig.canvas.manager.window.move(0, 0)  # Qt backend
        except AttributeError:
            pass
    gs = GridSpec(rows + 1, cols, figure=fig,
                  height_ratios=[1] * rows + [0.12], hspace=0.35, wspace=0.1)
    axes_flat = [fig.add_subplot(gs[r, c]) for r in range(rows) for c in range(cols)]
    selected = set()   # paths selected in this batch
    overlays = {}      # ax -> (overlay_patch, img_path)

    for ax, img_path in zip(axes_flat, batch_files):
        label_path = label_dir / img_path.with_suffix('.txt').name
        img = Image.open(img_path).convert("RGB")
        draw = ImageDraw.Draw(img)
        w, h = img.size

        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        cls, x, y, bw, bh = map(float, parts)
                        x1 = (x - bw / 2) * w
                        y1 = (y - bh / 2) * h
                        x2 = (x + bw / 2) * w
                        y2 = (y + bh / 2) * h
                        draw.rectangle([x1, y1, x2, y2], outline='blue', width=4)
                        names = list(class_names.values())
                        cls_idx = int(cls)
                        label = names[cls_idx] if cls_idx < len(names) else str(cls_idx)
                        font = get_font(h)
                        bbox = draw.textbbox((0, 0), label, font=font)
                        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
                        pad = 3
                        tx = x1
                        ty = y1 - th - pad * 2
                        if ty < 0:
                            ty = y2 + pad  # fall back to below the box
                        draw.rectangle([tx, ty, tx + tw + pad * 2, ty + th + pad * 2],
                                       fill='blue')
                        draw.text((tx + pad, ty + pad), label, fill='white', font=font)

        ax.imshow(img)
        ax.set_title(img_path.name, fontsize=7)
        ax.axis('off')

        overlay = mpatches.Rectangle(
            (0, 0), 1, 1, transform=ax.transAxes,
            color='red', alpha=0.35, zorder=5, visible=False
        )
        ax.add_patch(overlay)
        overlays[ax] = (overlay, img_path)

    for ax in axes_flat[n:]:
        ax.axis('off')

    def on_click(event, _selected=selected, _overlays=overlays, _fig=fig):
        ax_clicked = event.inaxes
        if ax_clicked is None or ax_clicked not in _overlays:
            return
        overlay, img_path = _overlays[ax_clicked]
        path_str = str(img_path)
        if path_str in _selected:
            _selected.discard(path_str)
            overlay.set_visible(False)
            ax_clicked.set_title(img_path.name, fontsize=7, color='black')
        else:
            _selected.add(path_str)
            overlay.set_visible(True)
            ax_clicked.set_title(f"[SELECTED] {img_path.name}", fontsize=7, color='red')
        _fig.canvas.draw_idle()

    fig.canvas.mpl_connect('button_press_event', on_click)

    ax_btn = fig.add_subplot(gs[rows, 1:3])
    btn = Button(ax_btn, f"Save & Continue  ({batch_idx + 1}/{total_batches})")

    btn_clicked = [False]

    def on_save(_event, _selected=selected, _all=all_selected, _fig=fig, _flag=btn_clicked):
        _flag[0] = True
        _all.update(_selected)
        plt.close(_fig)

    def on_close(_event, _selected=selected, _all=all_selected, _flag=btn_clicked, _quit=user_quit):
        if not _flag[0]:
            _all.update(_selected)
            _quit[0] = True

    btn.on_clicked(on_save)
    fig.canvas.mpl_connect('close_event', on_close)

    plt.tight_layout()
    plt.show()

if all_selected:
    with open(output_file, 'a') as f:
        for path in sorted(all_selected):
            f.write(path + '\n')
    print(f"Saved {len(all_selected)} path(s) to '{output_file}'")
else:
    print("No images selected.")
