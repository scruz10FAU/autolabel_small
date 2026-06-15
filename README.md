# Autolabeling Data
This is designed to automatically label data and finetune based on a specified class list. The default class labels buoys like those used in the robotx competition. 

Existing labels can be verified and updated in most environments.

## Installation

Clone the repository

```bash
git clone https://github.com/scruz10FAU/autolabel.git
```

Install all dependencies

```bash
cd autolabel
pip install -r requirements.txt
```

## Usage

### Detect buoys and label images

This runs YOLO auto-labeling on a folder of images (or other input source) and saves the resulting images and labels for training in Yolov8 

```bash
python autoLabelGen.py -m models/best_alex.pt -s buoy_images -t folder
```

To label from a live ZED camera ROS topic and visualize the detection stream in real time:
The initial code was set up using ZED cameras in an IsaacROS docker container, as shown in this link: [IsaacROS with ZED Cameras](https://www.stereolabs.com/docs/isaac-ros/setting_up_isaac_ros). 

```bash
python autoLabelGen.py -m models/best_alex.pt -t ros -s /zed/zed_node/rgb/color/rect/image -v
```

The `-v` flag opens a window showing every incoming frame with bounding boxes and class labels overlaid. Frames are still filtered by confidence and blur threshold before being saved. Press `q` to stop.

To collect hard negative (background) images for reducing false positives, use `--background-only`. Only frames where the model fires no detections are saved, each with an empty label file. Frames with any detection are skipped entirely:

```bash
python autoLabelGen.py -m models/best_real.pt -t video -s background_footage.mp4 --background-only
```

Use `--no-detect-interval` alongside it to avoid saving near-duplicate frames from static footage:

```bash
python autoLabelGen.py -m models/best_real.pt -t video -s background_footage.mp4 --background-only --no-detect-interval 5
```

Output is saved in the following format

```
training-directory/
├── images/
│   └── img0.jpg
├── labels/
    └── img0.txt
```

You can add arguments as follows when running the autoLabelGen program
| Flags | Data type | Function | options |
| -------------------------------- | -------- | ------------------------------------| ---------------------------- |
| `-m`, `--model` | str | Path to YOLO .pt model | model path |
| `-s`, `--source` | str | Camera index, video path, image folder, or ROS topic | path or index |
| `-t`, `--source-type` | str | Input source type (default: `ros`) | `camera`, `video`, `folder`, `ros` |
| `-r`, `--root` | str | Root output directory; saves to `<root>/images` and `<root>/labels`, overrides `-i` and `-l` | directory path |
| `-i` , `--out-images` | str | Output folder for labeled images | directory path |
| `-l`, `--out-labels` | str | Output folder for YOLO label files | directory path |
| `-c`, `--conf` | float | Detection confidence threshold | float between 0 and 1 |
| `--blur` | float | Blur threshold (Laplacian variance); 0 to disable | float |
| `-v`, `--show` | bool | Display a live window with detection boxes and class labels overlaid. Only active for `camera`, `video`, and `ros` source types. Press `q` to stop. | flag sets to True |
| `--no-detect-interval` | int | Save a frame with an empty label file after this many consecutive frames with no detection; `0` to disable (default: `0`) | integer |
| `--no-save-no-detect` | bool | Explicitly disable saving frames with no detection, overriding `--no-detect-interval` | flag sets to True |
| `--background-only` | bool | Save only frames where the model fires no detections, with empty label files; skips all frames with detections. Useful for collecting hard negative training data to reduce false positives | flag sets to True |
| `--preview-only`, `-p` | bool | Run detection without saving any images or labels; combine with `--show` to visually preview detections | flag sets to True |


### Crop bounding boxes into individual images

This extracts each labeled bounding box from a set of training images and saves each crop as its own file. Useful for building classification datasets or inspecting individual detections. Crops can optionally be organized into per-class subdirectories and expanded with padding.

```bash
python crop_detections.py --root trainImagesZed
```

Organize crops into subfolders by class name:

```bash
python crop_detections.py --root trainImagesZed --by-class -c classes.json -o crops
```

Add 10% padding around each crop and preview without writing:

```bash
python crop_detections.py --root trainImagesZed --padding 0.1 --dry-run
```

Each crop is saved as `<original_stem>_crop<idx>.jpg`. With `--by-class`, output is organized as `<output>/<class_name>/<crop_file>`.

You can add arguments as follows when running the crop_detections program
| Flags | Data type | Function | options |
| -------------------------------- | -------- | ------------------------------------| ---------------------------- |
| `-r`, `--root` | str | Root dataset directory containing `images/` and `labels/` subdirs; overrides `-i` and `-l` | directory path |
| `-i`, `--images` | str | Directory of training images (default: `trainImagesZed/images`) | directory path |
| `-l`, `--labels` | str | Directory of YOLO label files (default: `trainImagesZed/labels`) | directory path |
| `-o`, `--output` | str | Output directory for cropped images (default: `crops`) | directory path |
| `-c`, `--classes` | str | JSON file mapping class IDs to names; used for subfolder names with `--by-class` | json path |
| `--by-class` | bool | Organize crops into subdirectories by class name | flag sets to True |
| `--padding` | float | Fractional padding to add around each crop on all sides (e.g. `0.1` adds 10%, default: `0`) | float |
| `--dry-run` | bool | Print what would be saved without writing any files | flag sets to True |

### Remove duplicate images

This scans an images folder and removes near-duplicate images along with their YOLO label files. Two images are only removed as duplicates if they are both structurally similar (same scene layout) AND have a similar color distribution. Images that look structurally the same but contain differently colored buoys are kept.

```bash
python dedup_dataset.py
```

Run with `--dry-run` first to preview what would be deleted without removing anything:

```bash
python dedup_dataset.py --dry-run
```

You can add arguments as follows when running the dedup_dataset program
| Flags | Data type | Function | options |
| -------------------------------- | -------- | ------------------------------------| ---------------------------- |
| `-r`, `--root` | str | Root dataset directory containing `images/` and `labels/` subdirs; overrides `-i` and `-l` | directory path |
| `-i`, `--images` | str | Directory of training images (default: `trainImagesZed/images`) | directory path |
| `-l`, `--labels` | str | Directory of YOLO label files (default: `trainImagesZed/labels`) | directory path |
| `--hash-thresh` | int | Max perceptual hash distance to consider two images structurally similar; lower is stricter (default: `8`, range: 0–64) | integer |
| `--color-thresh` | float | Min hue-histogram correlation to consider two images the same color; higher is stricter (default: `0.90`, range: 0–1) | float |
| `--dry-run` | bool | Print what would be deleted without deleting anything | flag sets to True |

### View and verify image labels

This displays saved training images with their YOLO bounding box labels overlaid. Images can be clicked to select them, and selected paths are appended to an output file for further review or correction.

```bash
python viewImages.py 
```

Images are viewed 16 at a time with the labels added. Click on images if their label is incorrect so they can be added to a queue for labels to be fixed. Images with incorrect labels will be highlighted in red, like shown below:

![Grid image view](labelViewer.png)

Selected images will be added to a list for use in the "Edit image labels" section.

You can add arguments as follows when running the viewImages program
| Flags | Data type | Function | options |
| -------------------------------- | -------- | ------------------------------------| ---------------------------- |
| `-r`, `--root` | str | Root dataset directory containing `images/` and `labels/` subdirs; overrides `-i` and `-l` | directory path |
| `-i`, `--img_dir` | str | Directory of training images | directory path |
| `-l`, `--label_dir` | str | Directory of YOLO label files | directory path |
| `-o`, `--output_file` | str | File to append selected image paths | file path |
| `-c`, `--classes` | str | JSON file mapping class IDs to name and color | json path |
| `-b`, `--start-batch` | int | Batch number to start viewing from (1-indexed, default: `1`) | integer |

### Edit image labels

This opens a tkinter GUI for manually drawing, editing, and deleting YOLO bounding boxes on training images. If a `to_update_file` exists, only those images are loaded; otherwise all images in the image directory are shown. 

```bash
python labeleditor.py -i training_data/images -l training_data/labels -c classes.json
```

Use the mouse controls to manage bounding boxes:
- **Left-drag** — draw a new box using the active class
- **Middle-click** a box — open a class picker menu to relabel it without redrawing
- **Right-click** a box — delete it

Use the Prev/Next buttons to navigate and auto-save.

![Incorrectly labeled image](labelEdit.png)

Select the correct class and redraw the box to fix the label
![Corrected label](labelFix.png)

You can add arguments as follows when running the labeleditor program
| Flags | Data type | Function | options |
| -------------------------------- | -------- | ------------------------------------| ---------------------------- |
| `-r`, `--root` | str | Root dataset directory containing `images/` and `labels/` subdirs; overrides `-i` and `-l` | directory path |
| `-i`, `--image_dir` | str | Directory of training images | directory path |
| `-l`, `--label_dir` | str | Directory of YOLO label files | directory path |
| `-u`, `--to_update_file` | str | File listing image paths to review | file path |
| `-c`, `--classes` | str | JSON file mapping class IDs to name and color | json path |

The classes JSON file should follow this format:
```json
{
    "0": {"name": "red_buoy",   "color": "red"},
    "1": {"name": "green_buoy", "color": "green"}
}
```


### Augment images with lighting variations

This generates new training images by applying lighting augmentations to existing ones, simulating different environments such as overcast sky, bright sun, fog, or sunset. Each augmented image is saved alongside a copy of the original label file — bounding box coordinates are unaffected by lighting changes so the labels remain valid.

There are five named presets and a random mode that samples brightness, contrast, color temperature, fog, and shadow parameters independently.

```bash
python augment_lighting.py
```

Apply specific presets only (no random augmentations):

```bash
python augment_lighting.py --preset overcast --preset foggy --preset sunset --count 0
```

Generate 5 random augmentations per image with a fixed seed:

```bash
python augment_lighting.py -n 5 --seed 42
```

Use `--dry-run` to preview what would be created without writing any files:

```bash
python augment_lighting.py --preset sunny -n 3 --dry-run
```

You can add arguments as follows when running the augment_lighting program
| Flags | Data type | Function | options |
| -------------------------------- | -------- | ------------------------------------| ---------------------------- |
| `-r`, `--root` | str | Root input directory containing `images/` and `labels/` subdirs; overrides `-i` and `-l` | directory path |
| `-i`, `--images` | str | Directory of source training images (default: `trainImagesZed/images`) | directory path |
| `-l`, `--labels` | str | Directory of source YOLO label files (default: `trainImagesZed/labels`) | directory path |
| `-or`, `--out-root` | str | Root output directory; saves to `<out-root>/images` and `<out-root>/labels`, overrides `--out-images` and `--out-labels` | directory path |
| `--out-images` | str | Output directory for augmented images (default: same as `--images`) | directory path |
| `--out-labels` | str | Output directory for augmented labels (default: same as `--labels`) | directory path |
| `--preset` | str | Named lighting preset to apply; can be repeated (default: none) | `overcast`, `sunny`, `sunset`, `dim`, `foggy` |
| `-n`, `--count` | int | Number of randomly parameterized augmentations per image (default: `3`); set to `0` to use presets only | integer |
| `--seed` | int | Random seed for reproducible random augmentations | integer |
| `--dry-run` | bool | Print what would be created without writing any files | flag sets to True |

### Finetune the YOLOv8 model

This finetunes an existing YOLOv8 `.pt` checkpoint on the local buoy dataset. It automatically merges the available dataset roots, shuffles the images, and splits them into train and validation sets before running training. Class names are loaded from a JSON file, so they can be renamed for the new model without modifying the label files. The best checkpoint is saved to `models/finetuned.pt` when training completes.

```bash
python finetune.py
```

Use a specific GPU, more epochs, and a larger batch size:

```bash
python finetune.py --device 0 --epochs 100 --batch 32
```

Unfreeze all layers (trains the full network, recommended when you have a lot of new data):

```bash
python finetune.py --freeze 0
```

Reuse the existing train/val split instead of rebuilding it:

```bash
python finetune.py --keep-split
```

Use a custom classes file to rename classes in the finetuned model:

```bash
python finetune.py --classes my_classes.json
```

The classes JSON file should follow this format:
```json
{
    "0": {"name": "port_marker",      "color": "red"},
    "1": {"name": "starboard_marker", "color": "green"},
    "2": {"name": "blue_marker",      "color": "blue"},
    "3": {"name": "other_marker",     "color": "black"}
}
```

You can add arguments as follows when running the finetune program
| Flags | Data type | Function | options |
| -------------------------------- | -------- | ------------------------------------| ---------------------------- |
| `-m`, `--model` | str | Path to the base `.pt` checkpoint to finetune (default: `models/best_alex.pt`) | file path |
| `-r`, `--data-roots` | str (multiple) | Dataset root folders, each must contain `images/` and `labels/` subdirectories (default: `trainImagesZed`) | directory paths |
| `--val-split` | float | Fraction of images held out for validation (default: `0.15`) | float between 0 and 1 |
| `--epochs` | int | Number of training epochs (default: `50`) | integer |
| `--imgsz` | int | Training image size in pixels (default: `640`) | integer |
| `--batch` | int | Batch size; use `-1` for AutoBatch (default: `16`) | integer |
| `--lr0` | float | Initial learning rate (default: `0.001`) | float |
| `--freeze` | int | Number of backbone layers to freeze; set `0` to train the full network (default: `10`) | integer |
| `--device` | str | Device to train on (default: auto-detect) | `0`, `1`, `cpu`, etc. |
| `--project` | str | Output directory for training runs (default: `runs/finetune`) | directory path |
| `-n`, `--name` | str | Run name inside `--project` (default: `buoy`) | string |
| `--seed` | int | Random seed for reproducible train/val split (default: `42`) | integer |
| `--keep-split` | bool | Reuse existing `autosplit_train/` and `autosplit_val/` instead of rebuilding (by default the split is always rebuilt) | flag sets to True |
| `-c`, `--classes` | str | JSON file mapping class IDs to names used in the finetuned model (default: `classes.json`) | file path |
| `--loss-threshold` | float | Stop training early when total loss drops below this value; best weights are still saved normally; `0` to disable (default: `0`) | float |
