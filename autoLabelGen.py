import os
import cv2
import time
import argparse
import logging
import threading
import queue as queue_module
from pathlib import Path
from ultralytics import YOLO

# Optional ROS2 support
try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import Image as RosImage
    from cv_bridge import CvBridge
    ROS_AVAILABLE = True
except ImportError:
    ROS_AVAILABLE = False


def blur_score(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def to_yolo_label(frame, box):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2 / w
    cy = (y1 + y2) / 2 / h
    bw = (x2 - x1) / w
    bh = (y2 - y1) / h
    return cx, cy, bw, bh


# === Input sources ===

def iter_camera(source):
    src = int(source) if str(source).isdigit() else source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera: {source}")
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            yield frame, None
    finally:
        cap.release()


def iter_video(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            yield frame, None
    finally:
        cap.release()


def iter_folder(folder):
    exts = {'.jpg', '.jpeg', '.png', '.bmp'}
    paths = sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in exts)
    if not paths:
        raise RuntimeError(f"No images found in folder: {folder}")
    for p in paths:
        frame = cv2.imread(str(p))
        if frame is not None:
            yield frame, p


def iter_ros_topic(topic):
    if not ROS_AVAILABLE:
        raise RuntimeError("ROS2 / cv_bridge not available. Install ros-humble-cv-bridge.")

    frame_queue = queue_module.Queue(maxsize=5)
    bridge = CvBridge()

    class FrameNode(Node):
        def __init__(self):
            super().__init__('auto_label_subscriber')
            self.create_subscription(RosImage, topic, self._cb, 10)

        def _cb(self, msg):
            try:
                frame = bridge.imgmsg_to_cv2(msg, 'bgr8')
                if not frame_queue.full():
                    frame_queue.put_nowait(frame)
            except Exception as e:
                self.get_logger().error(f"cv_bridge error: {e}")

    rclpy.init()
    node = FrameNode()
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    print(f"Subscribed to ROS topic: {topic}")
    try:
        while rclpy.ok():
            try:
                yield frame_queue.get(timeout=1.0), None
            except queue_module.Empty:
                continue
    finally:
        node.destroy_node()
        rclpy.shutdown()


def get_frame_iter(source_type, source):
    if source_type == 'camera':
        return iter_camera(source)
    elif source_type == 'video':
        return iter_video(source)
    elif source_type == 'folder':
        return iter_folder(source)
    elif source_type == 'ros':
        return iter_ros_topic(source)
    raise ValueError(f"Unknown source type: {source_type}")


def next_img_index(folder):
    """Return the next available img index based on existing files."""
    indices = []
    for f in Path(folder).glob('img*.jpg'):
        try:
            indices.append(int(f.stem[3:]))
        except ValueError:
            pass
    return max(indices) + 1 if indices else 0


# === Main loop ===

def run(model_path, source_type, source, out_images, out_labels,
        conf=0.5, blur_thresh=25.0, show=False, no_detect_interval=0,
        background_only=False, preview_only=False):

    logging.getLogger('ultralytics').setLevel(logging.ERROR)
    model = YOLO(model_path)

    live_source = source_type in ('camera', 'video', 'ros')
    if show and not live_source:
        print("Note: --show is only supported for camera, video, and ros sources. Disabling.")
        show = False

    if not preview_only:
        os.makedirs(out_images, exist_ok=True)
        os.makedirs(out_labels, exist_ok=True)

    same_folder = (source_type == 'folder' and
                   Path(source).resolve() == Path(out_images).resolve())

    img_idx = next_img_index(out_images) if not (preview_only or same_folder) else 0
    saved = 0
    no_detect_count = 0
    print(f"Model: {model_path}")
    print(f"Source: {source_type} — {source}")
    if preview_only:
        print("Mode: preview only — detections will not be saved.")
    elif same_folder:
        print("Mode: in-place labeling — labels written alongside existing images, no duplicates created.")
    else:
        print(f"Output: {out_images} / {out_labels}")
    if background_only:
        print("Mode: background-only — saving frames with no detections (empty labels).")
    elif no_detect_interval > 0:
        print(f"Auto-save empty labels every {no_detect_interval} consecutive frame(s) with no detection.")
    print(f"Press Q to stop.\n")

    try:
        for frame, src_path in get_frame_iter(source_type, source):
            results = model(frame, conf=conf, verbose=False)
            boxes = results[0].boxes

            # Show every frame on live sources, with detection boxes and class names overlaid
            if show:
                display = frame.copy()
                if boxes is not None and len(boxes) > 0:
                    for box in boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        cls = int(box.cls[0])
                        conf_val = float(box.conf[0])
                        name = model.names.get(cls, str(cls))
                        cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(display, f"{name} {conf_val:.2f}", (x1, y1 - 6),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.imshow("Auto Label", display)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            # Determine output paths for this frame
            if same_folder and src_path is not None:
                img_path = str(src_path)
                lbl_path = os.path.join(out_labels, src_path.stem + ".txt")
                frame_name = src_path.name
            else:
                img_path = os.path.join(out_images, f"img{img_idx}.jpg")
                lbl_path = os.path.join(out_labels, f"img{img_idx}.txt")
                frame_name = f"img{img_idx}.jpg"

            if boxes is None or len(boxes) == 0:
                no_detect_count += 1
                if not preview_only:
                    interval = 1 if background_only else no_detect_interval
                    if interval > 0 and no_detect_count >= interval:
                        if not same_folder:
                            cv2.imwrite(img_path, frame)
                        open(lbl_path, 'w').close()
                        print(f"Saved {frame_name} — no detection (empty label)")
                        saved += 1
                        if not same_folder:
                            img_idx += 1
                        no_detect_count = 0
                continue

            if background_only:
                continue

            no_detect_count = 0
            label_lines = []
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cls = int(box.cls[0])

                if blur_thresh > 0:
                    crop = frame[int(y1):int(y2), int(x1):int(x2)]
                    if crop.size == 0:
                        continue
                    score = blur_score(crop)
                    if score < blur_thresh:
                        print(f"{frame_name}: blur {score:.1f} below threshold, skipping crop")
                        continue

                cx, cy, bw, bh = to_yolo_label(frame, (x1, y1, x2, y2))
                label_lines.append(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

            if not label_lines:
                continue

            if preview_only:
                print(f"Detected {len(label_lines)} object(s) in {frame_name} — not saving")
                continue

            if not same_folder:
                cv2.imwrite(img_path, frame)
            with open(lbl_path, 'w') as f:
                f.write('\n'.join(label_lines) + '\n')

            print(f"Labeled {frame_name} — {len(label_lines)} detection(s)")
            saved += 1
            if not same_folder:
                img_idx += 1

    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        if show:
            cv2.destroyAllWindows()
        print(f"\nDone. Saved {saved} labeled image(s) to '{out_images}'.")


if __name__ == '__main__':
    def_model_path   = "models/best_alex.pt"
    def_source_type  = "ros"
    def_source       = "/zed/zed_node/rgb/color/rect/image"
    def_out_images   = "trainImagesZed/images"
    def_out_labels   = "trainImagesZed/labels"
    def_conf         = 0.5
    def_blur         = 25.0

    parser = argparse.ArgumentParser(description="Generic YOLO auto-labeling tool")
    parser.add_argument('-m', '--model', default=def_model_path,
                        help=f"Path to YOLO .pt model (default: {def_model_path})")
    parser.add_argument('-s', '--source', default=def_source,
                        help=f"Camera index, video path, image folder, or ROS topic (default: {def_source})")
    parser.add_argument('-t', '--source-type', dest='source_type',
                        choices=['camera', 'video', 'folder', 'ros'], default=def_source_type,
                        help=f"Input source type (default: {def_source_type})")
    parser.add_argument('--root', '-r', default=None,
                        help="Root output directory; saves to <root>/images and <root>/labels, overrides --out-images and --out-labels")
    parser.add_argument('--out-images', '-i', default=def_out_images,
                        help=f"Output folder for images (default: {def_out_images})")
    parser.add_argument('--out-labels', '-l', default=def_out_labels,
                        help=f"Output folder for labels (default: {def_out_labels})")
    parser.add_argument('-c', '--conf', type=float, default=def_conf,
                        help=f"Detection confidence threshold (default: {def_conf})")
    parser.add_argument('--blur', type=float, default=def_blur,
                        help=f"Blur threshold (Laplacian variance); 0 to disable (default: {def_blur})")
    parser.add_argument('-v', '--show', action='store_true',
                        help="Show detections in a live window")
    parser.add_argument('--no-detect-interval', type=int, default=0, dest='no_detect_interval',
                        help="Save image with empty label after this many consecutive frames with no detection; 0 to disable (default: 0)")
    parser.add_argument('--no-save-no-detect', action='store_true', dest='no_save_no_detect',
                        help="Explicitly disable saving frames with no detection, overriding --no-detect-interval")
    parser.add_argument('--background-only', action='store_true', dest='background_only',
                        help="Save only frames where the model fires no detections (empty labels); skips all frames with detections")
    parser.add_argument('--preview-only', '-p', action='store_true', dest='preview_only',
                        help="Run detection without saving any images or labels (use with --show to preview)")
    args = parser.parse_args()

    if args.root:
        out_images = os.path.join(args.root, 'images')
        out_labels = os.path.join(args.root, 'labels')
    else:
        out_images = args.out_images
        out_labels = args.out_labels

    no_detect_interval = 0 if args.no_save_no_detect else args.no_detect_interval
    run(args.model, args.source_type, args.source,
        out_images, out_labels,
        args.conf, args.blur, args.show, no_detect_interval, args.background_only,
        args.preview_only)
