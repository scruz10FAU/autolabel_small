import argparse
import json
import tkinter as tk
from PIL import Image, ImageTk
import os

parser = argparse.ArgumentParser(description='YOLO Label Editor')
parser.add_argument('--image_dir', '-i', type=str, default='training_data/images',
                    help='Directory of training images')
parser.add_argument('--label_dir', '-l', type=str, default='training_data/labels',
                    help='Directory of YOLO label files')
parser.add_argument('--to_update_file', '-u', type=str, default='to_update_zed.txt',
                    help='File listing image paths to review')
parser.add_argument('--classes', '-c', type=str, default='classes.json',
                    help='JSON file mapping class IDs to name and color')
parser.add_argument('--root', '-r', type=str, default=None,
                    help='Root dataset directory containing images/ and labels/ subdirs')
args = parser.parse_args()

if args.root:
    image_dir = os.path.join(args.root, 'images')
    label_dir = os.path.join(args.root, 'labels')
else:
    image_dir = args.image_dir
    label_dir = args.label_dir
to_update_file = args.to_update_file

with open(args.classes) as f:
    raw = json.load(f)
class_names  = {int(k): v['name']  for k, v in raw.items()}
class_colors = {int(k): v['color'] for k, v in raw.items()}
if os.path.exists(to_update_file):
    with open(to_update_file) as f:
        image_files = [os.path.basename(line.strip()) for line in f if line.strip()]
    print(f"Loaded {len(image_files)} image(s) from {to_update_file}")
else:
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))])
    print(f"'{to_update_file}' not found — loading all {len(image_files)} images")

index = 0
boxes = []  # each entry: (cls, xc, yc, w, h, rect_id, text_id)

# === GUI SETUP ===
root = tk.Tk()
root.title("YOLO Label Editor (Mouse Drawing)")

current_class = tk.IntVar(value=0)

canvas_width, canvas_height = 640, 480
canvas = tk.Canvas(root, width=canvas_width, height=canvas_height, cursor="tcross")
canvas.pack()

img_tk = None
start_x = start_y = 0
rect = None

# === FUNCTIONS ===

def box_color(cls):
    return class_colors.get(int(cls), 'red')

def box_label(cls):
    return class_names.get(int(cls), str(int(cls)))

def draw_box_on_canvas(cls, x1, y1, x2, y2):
    color = box_color(cls)
    rect_id = canvas.create_rectangle(x1, y1, x2, y2, outline=color, width=2)
    text_id = canvas.create_text(x1 + 4, y1 - 2, text=box_label(cls),
                                  anchor='sw', fill=color,
                                  font=('Arial', 11, 'bold'))
    return rect_id, text_id

def load_image(idx):
    global img_tk, boxes
    boxes = []
    canvas.delete("all")

    img_path = os.path.join(image_dir, image_files[idx])
    image = Image.open(img_path).resize((canvas_width, canvas_height))
    img_tk = ImageTk.PhotoImage(image)
    canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)

    root.title(f"YOLO Label Editor — {image_files[idx]}  ({idx + 1}/{len(image_files)})")

    name = os.path.splitext(image_files[idx])[0]
    label_path = os.path.join(label_dir, f"{name}.txt")
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    cls, xc, yc, w, h = map(float, parts)
                    x1 = (xc - w / 2) * canvas_width
                    y1 = (yc - h / 2) * canvas_height
                    x2 = (xc + w / 2) * canvas_width
                    y2 = (yc + h / 2) * canvas_height
                    rect_id, text_id = draw_box_on_canvas(int(cls), x1, y1, x2, y2)
                    boxes.append((int(cls), xc, yc, w, h, rect_id, text_id))

def on_mouse_down(event):
    global start_x, start_y, rect
    start_x, start_y = event.x, event.y
    color = box_color(current_class.get())
    rect = canvas.create_rectangle(start_x, start_y, start_x, start_y, outline=color, width=2)

def on_mouse_drag(event):
    canvas.coords(rect, start_x, start_y, event.x, event.y)

def find_box_at(x, y):
    for i, box in reversed(list(enumerate(boxes))):
        cx = box[1] * canvas_width
        cy = box[2] * canvas_height
        bw = box[3] * canvas_width / 2
        bh = box[4] * canvas_height / 2
        if cx - bw < x < cx + bw and cy - bh < y < cy + bh:
            return i, box
    return None, None

def delete_box_at(x, y):
    i, box = find_box_at(x, y)
    if box is not None:
        canvas.delete(box[5])
        canvas.delete(box[6])
        boxes.pop(i)
        return True
    return False

def on_right_click(event):
    delete_box_at(event.x, event.y)

def relabel_box(box_idx, new_cls):
    box = boxes[box_idx]
    _, xc, yc, w, h, rect_id, text_id = box
    color = box_color(new_cls)
    canvas.itemconfig(rect_id, outline=color)
    canvas.itemconfig(text_id, text=box_label(new_cls), fill=color)
    boxes[box_idx] = (new_cls, xc, yc, w, h, rect_id, text_id)

def on_middle_click(event):
    box_idx, _ = find_box_at(event.x, event.y)
    if box_idx is None:
        return
    menu = tk.Menu(root, tearoff=0)
    for cls_id, cls_name in class_names.items():
        menu.add_command(
            label=cls_name,
            foreground=class_colors[cls_id],
            command=lambda cid=cls_id, bi=box_idx: relabel_box(bi, cid)
        )
    menu.tk_popup(event.x_root, event.y_root)

def on_mouse_up(event):
    global boxes
    x1, y1 = start_x, start_y
    x2, y2 = event.x, event.y
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))

    if abs(x2 - x1) < 5 or abs(y2 - y1) < 5:
        canvas.delete(rect)
        return

    # Finalize the drag rectangle with correct color, then add label text
    cls = current_class.get()
    canvas.delete(rect)
    rect_id, text_id = draw_box_on_canvas(cls, x1, y1, x2, y2)

    w = (x2 - x1) / canvas_width
    h = (y2 - y1) / canvas_height
    xc = (x1 + x2) / 2 / canvas_width
    yc = (y1 + y2) / 2 / canvas_height
    boxes.append((cls, xc, yc, w, h, rect_id, text_id))

def save_label():
    name = os.path.splitext(image_files[index])[0]
    label_path = os.path.join(label_dir, f"{name}.txt")
    with open(label_path, 'w') as f:
        for box in boxes:
            f.write(f"{box[0]} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f} {box[4]:.6f}\n")

def next_image():
    global index
    save_label()
    if index < len(image_files) - 1:
        index += 1
        load_image(index)

def prev_image():
    global index
    save_label()
    if index > 0:
        index -= 1
        load_image(index)

def clear_boxes():
    global boxes
    for box in boxes:
        canvas.delete(box[5])
        canvas.delete(box[6])
    boxes = []
    name = os.path.splitext(image_files[index])[0]
    label_path = os.path.join(label_dir, f"{name}.txt")
    if os.path.exists(label_path):
        os.remove(label_path)


# === CLASS SELECTOR ===
class_frame = tk.LabelFrame(root, text="Active Class", padx=6, pady=4)
for cls_id, cls_name in class_names.items():
    color = class_colors[cls_id]
    tk.Radiobutton(
        class_frame, text=cls_name, variable=current_class, value=cls_id,
        fg=color, selectcolor='black', activeforeground=color,
        font=('Arial', 11, 'bold'), indicatoron=True
    ).pack(side=tk.LEFT, padx=8)
class_frame.pack(pady=4)

# === NAV BUTTONS ===
btn_frame = tk.Frame(root)
tk.Button(btn_frame, text="◀ Prev", command=prev_image).pack(side=tk.LEFT, padx=4)
tk.Button(btn_frame, text="Clear Boxes", command=clear_boxes).pack(side=tk.LEFT, padx=4)
tk.Button(btn_frame, text="Next ▶", command=next_image).pack(side=tk.LEFT, padx=4)
btn_frame.pack(pady=5)

canvas.bind("<ButtonPress-1>", on_mouse_down)
canvas.bind("<B1-Motion>", on_mouse_drag)
canvas.bind("<ButtonRelease-1>", on_mouse_up)
canvas.bind("<ButtonPress-3>", on_right_click)
canvas.bind("<ButtonPress-2>", on_middle_click)

tk.Label(root, text="Left-drag: draw  |  Middle-click: relabel  |  Right-click: delete",
         font=('Arial', 9), fg='gray').pack(pady=(0, 4))

# === START ===
load_image(index)
root.mainloop()
