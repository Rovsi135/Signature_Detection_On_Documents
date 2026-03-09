"""
RF-DETR inference and visualization tool.

Purpose
-------
This script runs inference with a trained RF-DETR model on one image or a
folder of images. It can show predictions interactively, save annotated output
images, or simply print how many detections were found per image.

Why this script exists
----------------------
In practice, RF-DETR did not provide a simple enough out-of-the-box inference
workflow for the kind of quick inspection used in this project. In particular,
I wanted a lightweight tool that could:

- load a trained checkpoint directly
- run inference on either one image or a whole folder
- draw predicted bounding boxes and labels
- browse results interactively with keyboard controls
- batch-save annotated prediction images

To support this workflow, I implemented this custom inference utility.

Typical use case
----------------
Use this script after training RF-DETR to inspect predictions on
document images, compare results visually, or export annotated outputs for
reporting and qualitative analysis.

Expected inputs
---------------
This script expects:

1. A trained RF-DETR checkpoint file
   Example:
       best.pth

2. A source path
   This can be either:
   - a single image file
   - a directory containing images

Supported image formats:
- .jpg
- .jpeg
- .png
- .bmp
- .tif
- .tiff
- .webp

Command-line API
----------------

Required arguments
~~~~~~~~~~~~~~~~~~
--weights <path>
    Path to the trained RF-DETR checkpoint file.

--source <path>
    Path to either:
    - a single image
    - a directory of images

Optional arguments
~~~~~~~~~~~~~~~~~~
--model <name>
    RF-DETR model size.

    Choices:
    - nano
    - small
    - base
    - medium
    - large

    This must match the model size used during training.

    Default:
        small

--conf <float>
    Confidence threshold for predictions.

    Lower values keep more detections.
    Higher values remove weaker detections.

    Default:
        0.0

--show
    Open an interactive viewer to browse annotated predictions.

    Controls:
    - Right Arrow / Down / n / Space : next image
    - Left Arrow / Up / p / Backspace : previous image
    - q / Escape : quit

--save
    Save annotated prediction images for all input images.

--out_dir <path>
    Output directory where annotated images will be saved.

    Default:
        runs_rfdetr/predictions

--save_ext <ext>
    File extension for saved output images.

    Recommended:
    - .jpg
    - .png

    Default:
        .jpg

--label_mode <mode>
    Controls what text is shown on prediction boxes.

    Choices:
    - name : show class name if available
    - id   : show numeric class ID
    - none : show no label text

    Default:
        name


Behavior notes
--------------
- If --save is used, the script first saves annotated outputs for all images.
- If --show is also used, the interactive viewer opens afterwards.
- If neither --save nor --show is used, the script only prints detection counts.
- If class names are available from the model, they are used when
  --label_mode name is selected.
- Folder structure is preserved when saving outputs from a source directory.

Interactive viewer controls
---------------------------
When using --show:

- Right Arrow / Down / n / Space
    Go to the next image

- Left Arrow / Up / p / Backspace
    Go to the previous image

- q / Escape
    Quit the viewer

Closing the window also exits the viewer cleanly.

Example commands
----------------
Run inference on one image and show the result with confidence level > 0.5:

    python predict_rfdetr.py \
        --weights /path/to/best.pth \
        --source /path/to/image.jpg \
        --conf 0.5
        --show


Run inference on a folder and save all annotated outputs:

    python predict_rfdetr.py \
        --weights /path/to/best.pth \
        --source /path/to/images \
        --conf 0.5
        --save \
        --out_dir /path/to/output

Run inference on a folder, save outputs, and also inspect them interactively:

    python predict_rfdetr.py \
        --weights /path/to/best.pth \
        --source /path/to/images \
        --conf 0.5
        --save \
        --show \
        --out_dir /path/to/output

Print detection counts only:

    python predict_rfdetr.py \
        --weights /path/to/best.pth \
        --source /path/to/images
        --conf 0.5

Important notes
---------------
- The chosen model size must match the checkpoint that was trained.
- This script is for inference only; it does not train the model.
- The script visualizes predicted boxes, not ground-truth annotations.
- Prediction quality depends on the chosen confidence threshold.
- Saving is disabled inside the interactive viewer in the current workflow,
  because batch export is handled before the viewer opens.

Dependencies
------------
This script requires:
- rfdetr
- supervision
- pillow
- numpy
- matplotlib
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import supervision as sv
from PIL import Image
import matplotlib.pyplot as plt

from rfdetr import RFDETRNano, RFDETRSmall, RFDETRBase, RFDETRMedium, RFDETRLarge

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def iter_images(source: Path) -> list[Path]:
    if source.is_file():
        return [source]
    return sorted([p for p in source.rglob("*") if p.suffix.lower() in IMG_EXTS])


def get_model(model_size: str, weights: str):
    model_size = model_size.lower()
    cls = {
        "nano": RFDETRNano,
        "small": RFDETRSmall,
        "base": RFDETRBase,
        "medium": RFDETRMedium,
        "large": RFDETRLarge,
    }.get(model_size)
    if cls is None:
        raise ValueError(
            f"Unknown --model {model_size}. Choose from nano/small/base/medium/large."
        )
    return cls(pretrain_weights=weights)


def load_image_rgb(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def normalize_class_names(class_names: Any) -> dict[int, str]:
    if class_names is None:
        return {}
    if isinstance(class_names, list):
        return {i: str(n) for i, n in enumerate(class_names)}
    if isinstance(class_names, dict):
        # name -> id
        if all(isinstance(v, int) for v in class_names.values()):
            return {int(v): str(k) for k, v in class_names.items()}
        # id -> name
        if all(isinstance(k, int) for k in class_names.keys()):
            return {int(k): str(v) for k, v in class_names.items()}
    return {}


def build_labels(
    dets: sv.Detections, label_mode: str, id2name: dict[int, str]
) -> list[str]:
    if label_mode == "none" or len(dets) == 0:
        return []
    labels: list[str] = []
    for cid, conf in zip(dets.class_id, dets.confidence):
        cid_int = int(cid) if cid is not None else -1
        if label_mode == "id":
            cls_txt = str(cid_int)
        else:  # "name"
            cls_txt = id2name.get(cid_int, str(cid_int))
        labels.append(f"{cls_txt} {float(conf):.2f}")
    return labels


def annotate_one(
    img_path: Path,
    model,
    conf: float,
    label_mode: str,
    id2name: dict[int, str],
    box_annotator: sv.BoxAnnotator,
    label_annotator: sv.LabelAnnotator(text_scale=1.0, text_thickness=2),
) -> tuple[np.ndarray, int]:
    image = load_image_rgb(img_path)
    dets: sv.Detections = model.predict(image, threshold=conf)

    scene = np.array(image)
    annotated = box_annotator.annotate(scene=scene, detections=dets)

    labels = build_labels(dets, label_mode, id2name)
    if labels:
        annotated = label_annotator.annotate(
            scene=annotated, detections=dets, labels=labels
        )

    return annotated, len(dets)


def save_annotated(
    out_dir: Path, base: Path, img_path: Path, save_ext: str, annotated: np.ndarray
) -> Path:
    rel = img_path.relative_to(base)
    out_path = (out_dir / rel).with_suffix(save_ext)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(annotated).save(out_path)
    return out_path


def viewer(
    paths: list[Path],
    source: Path,
    model,
    conf: float,
    label_mode: str,
    id2name: dict[int, str],
    save: bool,
    out_dir: Path,
    save_ext: str,
) -> None:
    """
    Arrow-key navigable viewer:
      - Left / Right arrows: prev / next
      - q or Esc: quit
      - s: save current annotated image (only if --save)
    Closing the window exits and does NOT proceed to the next image.
    """
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    base = source if source.is_dir() else source.parent
    idx = 0
    should_exit = {"v": False}

    fig, ax = plt.subplots(figsize=(10, 10), dpi=100)
    ax.axis("off")

    # initial render
    annotated, n = annotate_one(
        paths[idx], model, conf, label_mode, id2name, box_annotator, label_annotator
    )
    im = ax.imshow(annotated)

    def set_title():
        ax.set_title(f"[{idx+1}/{len(paths)}] {paths[idx]} | detections: {n}")

    set_title()
    fig.canvas.draw_idle()

    def on_close(_event):
        should_exit["v"] = True

    def on_key(event):
        nonlocal idx, annotated, n
        if event.key in ("q", "escape"):
            should_exit["v"] = True
            plt.close(fig)
            return

        if event.key in ("right", "down", "n", " "):
            if idx < len(paths) - 1:
                idx += 1
            else:
                return
        elif event.key in ("left", "up", "p", "backspace"):
            if idx > 0:
                idx -= 1
            else:
                return
        elif event.key == "s":
            if save:
                out_path = save_annotated(
                    out_dir, base, paths[idx], save_ext, annotated
                )
                print(f"[saved] {paths[idx]} -> {out_path}")
            return
        else:
            return

        annotated, n = annotate_one(
            paths[idx], model, conf, label_mode, id2name, box_annotator, label_annotator
        )
        im.set_data(annotated)
        set_title()
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("close_event", on_close)
    fig.canvas.mpl_connect("key_press_event", on_key)

    plt.show()

    # Window closed -> exit cleanly (no next-image processing)
    if should_exit["v"]:
        return


def main() -> None:
    ap = argparse.ArgumentParser(
        description="YOLO-like RF-DETR inference on an image or folder of images."
    )
    ap.add_argument(
        "--weights", required=True, help="Path to RF-DETR checkpoint (.pth)"
    )
    ap.add_argument(
        "--source",
        required=True,
        help="Path to an image or a directory containing images",
    )
    ap.add_argument(
        "--model",
        default="small",
        help="nano|small|base|medium|large (must match your trained model)",
    )
    ap.add_argument(
        "--conf", type=float, default=0.0, help="Confidence threshold for predictions"
    )

    ap.add_argument(
        "--show",
        action="store_true",
        help="Arrow-key viewer (Left/Right). Close = exit.",
    )
    ap.add_argument(
        "--save",
        action="store_true",
        help="Save annotated outputs for ALL images to --out_dir",
    )
    ap.add_argument(
        "--out_dir",
        default="runs_rfdetr/predictions",
        help="Directory to save annotated images",
    )
    ap.add_argument(
        "--save_ext",
        type=str,
        default=".jpg",
        help="Output image extension (.jpg or .png recommended)",
    )

    ap.add_argument(
        "--label_mode",
        choices=["name", "id", "none"],
        default="name",
        help="Label text: name (if available) | id | none",
    )
    args = ap.parse_args()

    source = Path(args.source)
    paths = iter_images(source)
    if not paths:
        raise SystemExit(f"No images found in: {source}")

    model = get_model(args.model, args.weights)
    id2name = normalize_class_names(getattr(model, "class_names", None))

    out_dir = Path(args.out_dir)
    base = source if source.is_dir() else source.parent

    # --- 1) If --save: ALWAYS batch-export everything first ---
    if args.save:
        out_dir.mkdir(parents=True, exist_ok=True)
        box_annotator = sv.BoxAnnotator()
        label_annotator = sv.LabelAnnotator(text_scale=1.0, text_thickness=2)

        for img_path in paths:
            annotated, n = annotate_one(
                img_path,
                model,
                args.conf,
                args.label_mode,
                id2name,
                box_annotator,
                label_annotator,
            )
            out_path = save_annotated(out_dir, base, img_path, args.save_ext, annotated)
            print(f"{img_path} -> {out_path} | detections: {n}")

        print(f"Saved {len(paths)} images to: {out_dir}")

    # --- 2) If --show: open viewer ---
    if args.show:
        
        viewer(
            paths=paths,
            source=source,
            model=model,
            conf=args.conf,
            label_mode=args.label_mode,
            id2name=id2name,
            save=False,  # disable saving in viewer
            out_dir=out_dir, 
            save_ext=args.save_ext,
        )
        return

    # --- 3) If neither save nor show: just print detections count ---
    if not args.save and not args.show:
        box_annotator = sv.BoxAnnotator()
        label_annotator = sv.LabelAnnotator(text_scale=1.0, text_thickness=2)
        for img_path in paths:
            _, n = annotate_one(
                img_path,
                model,
                args.conf,
                args.label_mode,
                id2name,
                box_annotator,
                label_annotator,
            )
            print(f"{img_path} | detections: {n}")
        print("Done.")


if __name__ == "__main__":
    main()
