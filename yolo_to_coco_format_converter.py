"""
YOLO-to-COCO dataset conversion utility for model comparison experiments.

Purpose
-------
This script converts an object-detection dataset stored in YOLO format into a
COCO-style dataset layout compatible with RF-DETR. It is intended to simplify
comparative experiments where the same dataset must be reused across detectors
that expect different annotation formats.

In this workflow, the YOLO dataset is treated as the canonical source format.
The script reads YOLO images and label files, converts normalized YOLO bounding
boxes into pixel-based COCO bounding boxes, and exports one COCO annotation
JSON file per split.

Expected input structure
------------------------
The input dataset root must follow this directory layout:

    <yolo_root>/
        images/
            train/
            valid/
            test/
        labels/
            train/
            valid/
            test/

For each image file in a split directory, the script looks for a matching YOLO
label file with the same stem inside the corresponding labels/<split>/ folder.

Example:

    images/train/doc001.jpg
    labels/train/doc001.txt

Label format
------------
The script expects standard YOLO detection annotations, one object per line:

    <class_id> <x_center> <y_center> <width> <height>

where:
- x_center, y_center, width, height are normalized to [0, 1]
- coordinates are relative to image width/height
- each label file may contain zero or more object lines

If a label file is missing, the image is treated as containing no objects.

Expected output structure
-------------------------
The output directory will be created in the following form:

    <out_root>/
        train/
            _annotations.coco.json
            <image files...>
        valid/
            _annotations.coco.json
            <image files...>
        test/
            _annotations.coco.json
            <image files...>

Each split receives:
- a folder of exported images
- a COCO annotation file named exactly:
      _annotations.coco.json

This filename is used because the downstream RF-DETR pipeline expects that
exact naming convention.

Current scope and constraints
-----------------------------
1. Single-class only
   This script currently supports only one object class.
   Although it reads the YOLO class_id value from each annotation line, it does
   not preserve multiple classes in the COCO export. Every annotation is written
   with category_id = 0, and the COCO file contains exactly one category whose
   name is controlled by --category_name.

2. Lowercase split names only
   The script expects lowercase split folder names:
       train, valid, test
   It does not automatically map alternatives such as Train/Val/Test.

3. One label file per image stem
   The script assumes that each image and label pair share the same filename
   stem, for example:
       image123.jpg  <->  image123.txt

4. Malformed annotation lines are skipped
   YOLO annotation lines that do not contain exactly 5 whitespace-separated
   values are ignored.

5. Images with no objects are supported
   Missing label files are treated as empty annotations, which is valid for
   detection datasets.

Main features
-------------
- Converts YOLO normalized boxes to COCO pixel-based boxes
- Processes train / valid / test splits automatically
- Supports common image formats including TIFF
- Optionally converts TIFF images to PNG
- Can either symlink or copy image files into the output dataset
- Prevents filename collisions in the output directory by appending suffixes
- Clamps bounding boxes to image boundaries for safer COCO export

Command-line interface
----------------------
Required arguments:
    --yolo_root   Path to the input YOLO dataset root
    --out_root    Path to the output COCO dataset root

Optional arguments:
    --category_name        Name of the single exported object category
    --image_mode           "symlink" or "copy" (default: symlink)
    --convert_tif_to_png   If set, TIFF images are exported as PNG

Example usage
-------------
    python yolo_to_coco_format_converter.py \
        --yolo_root /path/to/yolo_dataset \
        --out_root /path/to/coco_dataset \
        --image_mode symlink \
        --convert_tif_to_png

Typical use case
----------------
This script is useful when you want to maintain only one canonical dataset in
YOLO format and automatically generate the equivalent COCO-format dataset needed
for RF-DETR training. This helps keep comparative experiments reproducible and
reduces manual dataset preparation work.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image

# Optional: better TIFF handling. If unavailable, PIL is used instead.
try:
    import tifffile
except Exception:
    tifffile = None


# Supported image formats
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# We assume one canonical lowercase convention everywhere
SPLITS = ["train", "valid", "test"]


def read_image_size(path: Path) -> Tuple[int, int]:
    """
    Return image size as (width, height).

    TIFF files can sometimes be handled more robustly with tifffile,
    but PIL is used as a fallback for all formats.
    """
    suffix = path.suffix.lower()

    if suffix in {".tif", ".tiff"} and tifffile is not None:
        arr = tifffile.imread(str(path))
        if arr.ndim >= 2:
            height, width = arr.shape[0], arr.shape[1]
            return width, height

    with Image.open(path) as img:
        width, height = img.size
        return width, height


def yolo_to_coco_bbox(
    x_center: float,
    y_center: float,
    width_norm: float,
    height_norm: float,
    img_width: int,
    img_height: int,
) -> List[float]:
    """
    Convert a YOLO bounding box:
        [x_center, y_center, width, height] (normalized)
    into a COCO bounding box:
        [x_min, y_min, width, height] (pixels)

    Also clamps the box so it stays within image boundaries.
    """
    box_width = width_norm * img_width
    box_height = height_norm * img_height

    x_min = (x_center * img_width) - (box_width / 2.0)
    y_min = (y_center * img_height) - (box_height / 2.0)

    # Clamp top-left corner to image bounds
    x_min = max(0.0, min(x_min, img_width - 1.0))
    y_min = max(0.0, min(y_min, img_height - 1.0))

    # Clamp width and height so the box stays inside the image
    box_width = max(1.0, min(box_width, img_width - x_min))
    box_height = max(1.0, min(box_height, img_height - y_min))

    return [float(x_min), float(y_min), float(box_width), float(box_height)]


def make_unique_filename(destination_dir: Path, desired_name: str) -> str:
    """
    Avoid filename collisions in the output directory.

    Example:
        image.png
        image__1.png
        image__2.png
    """
    stem = Path(desired_name).stem
    suffix = Path(desired_name).suffix

    candidate = desired_name
    counter = 1

    while (destination_dir / candidate).exists():
        candidate = f"{stem}__{counter}{suffix}"
        counter += 1

    return candidate


def place_image(src: Path, dst: Path, mode: str) -> None:
    """
    Place an image into the output dataset by either:
    - symlinking it
    - copying it
    """
    dst.parent.mkdir(parents=True, exist_ok=True)

    if dst.exists():
        return

    if mode == "symlink":
        os.symlink(src, dst)
    elif mode == "copy":
        dst.write_bytes(src.read_bytes())
    else:
        raise ValueError(f"Unknown image mode: {mode}")


def convert_tif_to_png(src: Path, dst_png: Path) -> None:
    """
    Convert a TIFF image to PNG for better compatibility with some pipelines.
    """
    dst_png.parent.mkdir(parents=True, exist_ok=True)

    if dst_png.exists():
        return

    with Image.open(src) as img:
        img = img.convert("RGB")
        img.save(dst_png, format="PNG", optimize=False)


def create_empty_coco_dict(category_name: str, split_name: str) -> Dict[str, object]:
    """
    Create an empty COCO annotation structure for one split.
    """
    return {
        "info": {"description": f"Signature COCO export ({split_name})"},
        "licenses": [],
        "categories": [
            {
                "id": 0,
                "name": category_name,
                "supercategory": "object",
            }
        ],
        "images": [],
        "annotations": [],
    }


def build_coco_split(
    yolo_root: Path,
    out_root: Path,
    split_name: str,
    category_name: str,
    image_mode: str,
    convert_tif: bool,
) -> None:
    """
    Convert one dataset split from YOLO format to COCO format.

    Expected YOLO structure:
        yolo_root/
            images/train, valid, test
            labels/train, valid, test

    Output COCO structure:
        out_root/
            train/_annotations.coco.json
            valid/_annotations.coco.json
            test/_annotations.coco.json
    """
    images_dir = yolo_root / "images" / split_name
    labels_dir = yolo_root / "labels" / split_name
    output_split_dir = out_root / split_name

    if not images_dir.exists():
        raise FileNotFoundError(f"Missing images directory: {images_dir}")

    if not labels_dir.exists():
        raise FileNotFoundError(f"Missing labels directory: {labels_dir}")

    output_split_dir.mkdir(parents=True, exist_ok=True)

    coco = create_empty_coco_dict(category_name, split_name)

    image_id = 1
    annotation_id = 1

    image_paths = sorted(
        path for path in images_dir.rglob("*") if path.suffix.lower() in IMG_EXTS
    )

    for src_image_path in image_paths:
        # Match label file by filename stem
        label_path = labels_dir / f"{src_image_path.stem}.txt"

        # Missing label file is allowed: means no objects in the image
        if label_path.exists():
            yolo_lines = [
                line.strip()
                for line in label_path.read_text().splitlines()
                if line.strip()
            ]
        else:
            yolo_lines = []

        # Read original image size
        img_width, img_height = read_image_size(src_image_path)

        # Decide how the image should appear in the COCO export
        suffix = src_image_path.suffix.lower()

        if convert_tif and suffix in {".tif", ".tiff"}:
            output_name = make_unique_filename(output_split_dir, f"{src_image_path.stem}.png")
            output_image_path = output_split_dir / output_name
            convert_tif_to_png(src_image_path, output_image_path)
        else:
            output_name = make_unique_filename(output_split_dir, src_image_path.name)
            output_image_path = output_split_dir / output_name
            place_image(src_image_path, output_image_path, image_mode)

        # Add image entry to COCO
        coco["images"].append(
            {
                "id": image_id,
                "file_name": output_name,
                "width": img_width,
                "height": img_height,
            }
        )

        # Convert each YOLO annotation line into COCO format
        for line in yolo_lines:
            parts = line.split()

            # Standard YOLO detection format has exactly 5 values:
            # class_id x_center y_center width height
            if len(parts) != 5:
                continue

            class_id, x_center, y_center, width_norm, height_norm = parts

            # This project assumes a single class: signature
            # So we ignore the original class_id and always write category_id = 0
            x_center = float(x_center)
            y_center = float(y_center)
            width_norm = float(width_norm)
            height_norm = float(height_norm)

            coco_bbox = yolo_to_coco_bbox(
                x_center=x_center,
                y_center=y_center,
                width_norm=width_norm,
                height_norm=height_norm,
                img_width=img_width,
                img_height=img_height,
            )

            area = coco_bbox[2] * coco_bbox[3]

            coco["annotations"].append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": 0,
                    "bbox": coco_bbox,
                    "area": float(area),
                    "iscrowd": 0,
                    "segmentation": [],
                }
            )

            annotation_id += 1

        image_id += 1

    # RF-DETR expects exactly this JSON filename in each split folder
    output_json_path = output_split_dir / "_annotations.coco.json"
    output_json_path.write_text(json.dumps(coco, indent=2))

    print(
        f"[OK] Wrote {output_json_path} "
        f"with {len(coco['images'])} images and {len(coco['annotations'])} annotations"
    )


def main() -> None:
    """
    Command-line entry point.

    Example:
        python yolo_to_coco.py \
            --yolo_root /path/to/yolo_dataset \
            --out_root /path/to/coco_dataset \
            --image_mode symlink \
            --convert_tif_to_png
    """
    parser = argparse.ArgumentParser(
        description="Convert a YOLO-format dataset into COCO format for RF-DETR."
    )
    parser.add_argument(
        "--yolo_root",
        type=str,
        required=True,
        help="Path to the input YOLO dataset root.",
    )
    parser.add_argument(
        "--out_root",
        type=str,
        required=True,
        help="Path to the output COCO dataset root.",
    )
    parser.add_argument(
        "--category_name",
        type=str,
        default="signature",
        help="Name of the single object class.",
    )
    parser.add_argument(
        "--image_mode",
        type=str,
        choices=["symlink", "copy"],
        default="symlink",
        help="Whether to symlink or copy images into the COCO dataset.",
    )
    parser.add_argument(
        "--convert_tif_to_png",
        action="store_true",
        help="Convert TIFF/TIF images to PNG during export.",
    )

    args = parser.parse_args()

    yolo_root = Path(args.yolo_root).expanduser().resolve()
    out_root = Path(args.out_root).expanduser().resolve()

    for split_name in SPLITS:
        build_coco_split(
            yolo_root=yolo_root,
            out_root=out_root,
            split_name=split_name,
            category_name=args.category_name,
            image_mode=args.image_mode,
            convert_tif=args.convert_tif_to_png,
        )


if __name__ == "__main__":
    main()