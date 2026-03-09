from __future__ import annotations

"""
YOLO ground-truth annotation visualizer.

Purpose
-------
This script visualizes a YOLO-format object-detection dataset by drawing the
ground-truth bounding boxes stored in YOLO label files on top of the original
images. It is intended as a quick dataset-inspection tool for checking whether
annotations are present, correctly aligned, and visually reasonable.

Expected input structure
------------------------
The script expects:
- an images directory containing image files
- a labels directory containing YOLO .txt label files

Each image is matched to a label file using the same filename stem.

Example:
    images/doc001.jpg
    labels/doc001.txt

Label format
------------
YOLO labels must use the standard detection format:

    <class_id> <x_center> <y_center> <width> <height>

where:
- coordinates are normalized to [0, 1]
- values are relative to image width and height
- each line describes one object

If a label file is missing or empty, the image is treated as having no objects.

Optional class names
--------------------
If a Ultralytics-style data.yaml file is provided with --data, the script tries
to load class names from the "names" field. It supports both:

    names: ['signature']

and

    names:
      0: signature

If no YAML file is provided, class IDs are shown as numbers.

Main features
-------------
- Draws YOLO ground-truth boxes on images
- Supports browsing with keyboard controls
- Can recursively search for images
- Can show all images or a random sample
- Can optionally skip empty-label images
- Prints simple per-image box statistics in the terminal

Command-line API
----------------
Required arguments
~~~~~~~~~~~~~~~~~~
--images <path>
    Path to the directory containing the image files.

--labels <path>
    Path to the directory containing the YOLO label files (.txt).

Optional arguments
~~~~~~~~~~~~~~~~~~
--data <path>
    Path to an optional Ultralytics-style data.yaml file.
    If provided, class names are loaded from the "names" field and shown
    instead of numeric class IDs.

--recursive
    If set, search for image files recursively inside the images directory.
    Useful when the dataset is organized with nested subfolders.

--sample <int>
    If greater than 0, randomly select only that many images from the dataset
    for inspection instead of browsing the full set.

--start <filename>
    Start the viewer at a specific image filename, for example:
        --start doc001.jpg
    This is useful when you want to jump directly to a known problematic file.

--show_empty
    If set, include images whose label files are missing or empty.
    By default, such images are filtered out.

Behavior notes
~~~~~~~~~~~~~~
- Images are matched to label files by filename stem.
  Example:
      doc001.jpg  ->  doc001.txt

- If --data is omitted, class IDs are displayed as numbers.

- If --show_empty is not used, only images with non-empty label files are shown.

- Keyboard navigation is available after the viewer opens:
  - n / Right Arrow : next image
  - p / Left Arrow  : previous image
  - q / Escape      : quit

Example commands
~~~~~~~~~~~~~~~~
Visualize all annotated images:
    python visualize_yolo_annotations.py --images ./images --labels ./labels

Visualize recursively and show class names:
    python visualize_yolo_annotations.py \
        --images ./images \
        --labels ./labels \
        --data ./data.yaml \
        --recursive

Inspect a random sample of 25 images:
    python visualize_yolo_annotations.py \
        --images ./images \
        --labels ./labels \
        --sample 25

Start from a specific image and include empty-label images:
    python visualize_yolo_annotations.py \
        --images ./images \
        --labels ./labels \
        --start doc001.jpg \
        --show_empty
        
Controls
--------
- n or Right Arrow : next image
- p or Left Arrow  : previous image
- q or Escape      : quit

Typical use case
----------------
Use this script to inspect whether a YOLO-format dataset is correctly annotated
before training or converting it to another annotation format.
"""

import argparse
import random
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image

try:
    import yaml  # optional: for reading class names from data.yaml
except Exception:
    yaml = None


# Supported image extensions
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def load_class_names(data_yaml_path: Path) -> dict[int, str] | None:
    """
    Load class names from a Ultralytics-style data.yaml file.

    Supported formats:
        names: ['signature']
    or
        names:
          0: signature

    Returns:
        A dictionary mapping class IDs to class names, or None if loading fails.
    """
    if yaml is None:
        print("[Warn] PyYAML is not installed, so data.yaml cannot be parsed.")
        return None

    if not data_yaml_path.exists():
        print(f"[Warn] data.yaml not found: {data_yaml_path}")
        return None

    config = yaml.safe_load(data_yaml_path.read_text())
    names = config.get("names")

    if names is None:
        return None

    if isinstance(names, list):
        return {i: str(name) for i, name in enumerate(names)}

    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}

    return None


def list_image_files(images_dir: Path, recursive: bool) -> list[Path]:
    """
    Return all supported image files in the given directory.

    If recursive is True, also search subdirectories.
    """
    if recursive:
        image_paths = [
            path
            for path in images_dir.rglob("*")
            if path.suffix.lower() in IMAGE_EXTENSIONS
        ]
    else:
        image_paths = [
            path
            for path in images_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ]

    return sorted(image_paths, key=lambda path: path.name)


def read_yolo_label_file(label_path: Path) -> np.ndarray:
    """
    Read a YOLO label file.

    Returns:
        A NumPy array of shape (N, 5) where each row is:
            [class_id, x_center, y_center, width, height]

        All values are returned as floats.
        If the file is missing, empty, or invalid, an empty array is returned.
    """
    if not label_path.exists():
        return np.zeros((0, 5), dtype=np.float32)

    text = label_path.read_text().strip()
    if not text:
        return np.zeros((0, 5), dtype=np.float32)

    rows = []

    for line in text.splitlines():
        parts = line.strip().split()

        # Standard YOLO detection format requires exactly 5 values
        if len(parts) < 5:
            continue

        try:
            class_id = float(parts[0])
            x_center = float(parts[1])
            y_center = float(parts[2])
            box_width = float(parts[3])
            box_height = float(parts[4])
            rows.append([class_id, x_center, y_center, box_width, box_height])
        except ValueError:
            continue

    if not rows:
        return np.zeros((0, 5), dtype=np.float32)

    return np.array(rows, dtype=np.float32)


def yolo_box_to_pixel_corners(
    x_center: float,
    y_center: float,
    box_width: float,
    box_height: float,
    image_width: int,
    image_height: int,
) -> tuple[float, float, float, float]:
    """
    Convert a YOLO normalized box into pixel corner format.

    Input:
        x_center, y_center, box_width, box_height in normalized YOLO format

    Returns:
        (x1, y1, x2, y2) in pixel coordinates
    """
    x1 = (x_center - box_width / 2.0) * image_width
    y1 = (y_center - box_height / 2.0) * image_height
    x2 = (x_center + box_width / 2.0) * image_width
    y2 = (y_center + box_height / 2.0) * image_height

    return x1, y1, x2, y2


def clamp_box_to_image(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    image_width: int,
    image_height: int,
) -> tuple[float, float, float, float]:
    """
    Clamp box corners so they remain within the image boundaries.
    """
    x1 = max(0, min(image_width - 1, x1))
    y1 = max(0, min(image_height - 1, y1))
    x2 = max(0, min(image_width - 1, x2))
    y2 = max(0, min(image_height - 1, y2))

    return x1, y1, x2, y2


def get_label_path_for_image(image_path: Path, labels_dir: Path) -> Path:
    """
    Return the expected YOLO label path for a given image.
    """
    return labels_dir / f"{image_path.stem}.txt"


def filter_out_empty_images(image_paths: list[Path], labels_dir: Path) -> list[Path]:
    """
    Keep only images whose label file exists and is non-empty.
    """
    kept = []

    for image_path in image_paths:
        label_path = get_label_path_for_image(image_path, labels_dir)
        if label_path.exists() and label_path.stat().st_size > 0:
            kept.append(image_path)

    return kept


def print_box_statistics(
    labels: np.ndarray, image_width: int, image_height: int, image_name: str
) -> None:
    """
    Print simple box statistics for the current image.
    """
    box_count = len(labels)

    if box_count == 0:
        print(f"\n{image_name} | {image_width}x{image_height} | boxes=0")
        return

    box_widths_px = labels[:, 3] * image_width
    box_heights_px = labels[:, 4] * image_height

    print(f"\n{image_name} | {image_width}x{image_height} | boxes={box_count}")
    print(
        f"  box_w px: min={box_widths_px.min():.1f} "
        f"med={np.median(box_widths_px):.1f} "
        f"max={box_widths_px.max():.1f}"
    )
    print(
        f"  box_h px: min={box_heights_px.min():.1f} "
        f"med={np.median(box_heights_px):.1f} "
        f"max={box_heights_px.max():.1f}"
    )


def main() -> None:
    """
    Command-line entry point.

    Example:
        python visualize_yolo_dataset.py \
            --images /path/to/images \
            --labels /path/to/labels \
            --data /path/to/data.yaml \
            --recursive
    """
    parser = argparse.ArgumentParser(
        description="Visualize YOLO ground-truth annotations on images."
    )
    parser.add_argument(
        "--images",
        required=True,
        help="Path to the images directory.",
    )
    parser.add_argument(
        "--labels",
        required=True,
        help="Path to the YOLO labels directory.",
    )
    parser.add_argument(
        "--data",
        default=None,
        help="Optional path to data.yaml for class names.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search images recursively inside the images directory.",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=0,
        help="If greater than 0, view a random sample of this many images.",
    )
    parser.add_argument(
        "--start",
        type=str,
        default=None,
        help="Start from a specific image filename, e.g. 001.jpg",
    )
    parser.add_argument(
        "--show_empty",
        action="store_true",
        help="Include images with missing or empty label files.",
    )

    args = parser.parse_args()

    images_dir = Path(args.images)
    labels_dir = Path(args.labels)

    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    if not labels_dir.exists():
        raise FileNotFoundError(f"Labels directory not found: {labels_dir}")

    class_names = None
    if args.data is not None:
        class_names = load_class_names(Path(args.data))

    image_paths = list_image_files(images_dir, recursive=args.recursive)
    if not image_paths:
        raise RuntimeError(f"No supported image files found in: {images_dir}")

    # If --show_empty is NOT used, filter them out
    if not args.show_empty:
        image_paths = filter_out_empty_images(image_paths, labels_dir)
        if not image_paths:
            raise RuntimeError(
                "No images remain after filtering out empty-label images."
            )

    if args.sample > 0:
        image_paths = random.sample(image_paths, min(args.sample, len(image_paths)))

    current_index = 0
    if args.start is not None:
        for i, image_path in enumerate(image_paths):
            if image_path.name == args.start:
                current_index = i
                break

    print(f"[Info] Loaded {len(image_paths)} images.")
    print("[Controls] n / right = next | p / left = previous | q / esc = quit")

    fig, ax = plt.subplots(figsize=(10, 10))
    plt.tight_layout()

    def show_image(index: int) -> None:
        """
        Display one image with its YOLO ground-truth boxes.
        """
        ax.clear()

        image_path = image_paths[index]
        label_path = get_label_path_for_image(image_path, labels_dir)

        image = Image.open(image_path).convert("RGB")
        image_array = np.array(image)

        image_height, image_width = image_array.shape[0], image_array.shape[1]

        labels = read_yolo_label_file(label_path)

        ax.imshow(image_array)

        for row in labels:
            class_id, x_center, y_center, box_width, box_height = row.tolist()

            x1, y1, x2, y2 = yolo_box_to_pixel_corners(
                x_center=x_center,
                y_center=y_center,
                box_width=box_width,
                box_height=box_height,
                image_width=image_width,
                image_height=image_height,
            )

            x1, y1, x2, y2 = clamp_box_to_image(
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                image_width=image_width,
                image_height=image_height,
            )

            rect = patches.Rectangle(
                (x1, y1),
                max(0, x2 - x1),
                max(0, y2 - y1),
                linewidth=2,
                edgecolor="lime",
                facecolor="none",
            )
            ax.add_patch(rect)

            class_index = int(round(class_id))
            class_label = (
                class_names.get(class_index, str(class_index))
                if class_names is not None
                else str(class_index)
            )

            ax.text(
                x1,
                max(0, y1 - 5),
                class_label,
                fontsize=10,
                color="lime",
                bbox=dict(facecolor="black", alpha=0.4, pad=1, edgecolor="none"),
            )

        if len(labels) == 0:
            label_state = "missing" if not label_path.exists() else "empty"
            print(
                f"\n{image_path.name} | {image_width}x{image_height} "
                f"| boxes=0 (label: {label_state})"
            )
        else:
            print_box_statistics(
                labels=labels,
                image_width=image_width,
                image_height=image_height,
                image_name=image_path.name,
            )

        ax.set_title(f"{image_path.name} ({index + 1}/{len(image_paths)})")
        ax.axis("off")
        fig.canvas.draw_idle()

    def on_key_press(event) -> None:
        """
        Keyboard navigation callback.
        """
        nonlocal current_index

        if event.key in ["q", "escape"]:
            plt.close(fig)
            return

        if event.key in ["n", "right"]:
            current_index = (current_index + 1) % len(image_paths)
            show_image(current_index)
            return

        if event.key in ["p", "left"]:
            current_index = (current_index - 1) % len(image_paths)
            show_image(current_index)
            return

    fig.canvas.mpl_connect("key_press_event", on_key_press)
    show_image(current_index)
    plt.show()


if __name__ == "__main__":
    main()
