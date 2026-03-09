"""
YOLO training launcher.

Purpose
-------
This script provides a reproducible command-line interface for training a YOLO
object detection model using the Ultralytics framework. It wraps the standard
Ultralytics training pipeline with explicit configuration parameters so that
experiments can be easily repeated, modified, and documented.

The script loads a pretrained YOLO model, trains it on a dataset described by a
YOLO-style `data.yaml` configuration file, and performs validation after
training.

Typical use case
----------------
This script is intended for training and evaluating a YOLO model on a dataset
prepared in YOLO format. It is particularly useful when running controlled
experiments where training parameters (image size, batch size, number of
epochs, etc.) must be recorded and reproduced later.

Example usage
-------------
Basic training:

    python train_yolo.py \
        --data /path/to/data.yaml

Custom experiment:

    python train_yolo.py \
        --data /path/to/data.yaml \
        --model yolo11s.pt \
        --imgsz 896 \
        --epochs 120 \
        --batch -1 \
        --device 0 \
        --name experiment_sig_detection

Expected dataset format
-----------------------
The dataset must follow the standard YOLO dataset structure and be referenced
by a `data.yaml` configuration file.

Example `data.yaml`:

    path: /dataset/root
    train: images/train
    val: images/valid
    test: images/test
    names:
      0: signature

Images and labels should follow YOLO conventions:

    dataset/
        images/
            train/
            valid/
            test/
        labels/
            train/
            valid/
            test/

Each label file must contain YOLO-format bounding boxes:

    <class_id> <x_center> <y_center> <width> <height>

where coordinates are normalized to [0,1].

Command-line API
----------------

Required arguments
~~~~~~~~~~~~~~~~~~

--data <path>
    Path to the YOLO dataset configuration file (`data.yaml`).
    This file defines dataset splits and class names.

Optional arguments
~~~~~~~~~~~~~~~~~~

--model <string>
    Pretrained YOLO model to load before training.
    Default: "yolo11s.pt"

    Examples:
        yolo11n.pt
        yolo11s.pt
        yolo11m.pt
        yolo11l.pt
        yolo11x.pt

--imgsz <int>
    Input image size used during training and validation.
    Images are resized to this resolution.
    Default: 896

--epochs <int>
    Total number of training epochs.
    Default: 120

--batch <int>
    Training batch size.
    -1 lets Ultralytics automatically choose the batch size based on GPU memory.
    Default: -1

--patience <int>
    Early stopping patience.
    Training stops if validation metrics do not improve for this many epochs.
    Default: 30

--seed <int>
    Random seed for reproducibility.
    Default: 42

--device <string>
    Device used for training.

    Examples:
        "0"    → GPU 0
        "0,1"  → multiple GPUs
        "cpu"  → CPU only

    Default: "0"

--workers <int>
    Number of dataloader worker processes.
    Higher values may improve data loading speed.
    Default: 4

--project <string>
    Root directory where training results will be saved.
    Default: runs/signature_detect

--name <string>
    Name of the training run.
    Output directory becomes:

        <project>/<name>/

    Default: sig_yolo11s_img896

--lr0 <float>
    Initial learning rate.
    If not specified, Ultralytics uses its default training schedule.

--lrf <float>
    Final learning rate fraction used in cosine learning rate decay.
    If not specified, Ultralytics uses its default.


Training options used
---------------------
The script enables several useful defaults:

pretrained=True
    Initializes the model with pretrained weights.

cache=False
    Images are loaded from disk each epoch rather than cached in memory.

amp=True
    Enables automatic mixed precision for faster training on GPUs.

verbose=True
    Enables detailed training logs.

Outputs
-------
Training outputs are saved under:

    <project>/<name>/

This directory typically contains:

- model checkpoints (best.pt, last.pt)
- training logs
- training curves
- evaluation plots
- tensorboard logs

Notes
-----
- Validation is automatically performed during training by Ultralytics.
- A final validation pass is run after training for completeness.
- This script assumes the dataset is already prepared in YOLO format.
- Dataset preparation or annotation conversion should be performed separately.

Dependencies
------------
Required Python packages:

    ultralytics
    torch
    torchvision

Install via:

    pip install ultralytics
"""
from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, required=True, help="Path to data.yaml")
    p.add_argument("--model", type=str, default="yolo11s.pt", help="e.g., yolo11s.pt or yolo11n.pt")
    p.add_argument("--imgsz", type=int, default=896)
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--batch", type=int, default=-1, help="-1 lets Ultralytics auto-pick; else set manually")
    p.add_argument("--patience", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="0", help="GPU id like '0' or 'cpu'")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--name", type=str, default="sig_yolo11s_img896")
    p.add_argument("--project", type=str, default="runs/signature_detect")
    p.add_argument("--lr0", type=float, default=None, help="Initial learning rate (None = Ultralytics default)")
    p.add_argument("--lrf", type=float, default=None, help="Final LR fraction (None = Ultralytics default)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    data_path = Path(args.data)
    
    if not data_path.exists():
        raise FileNotFoundError(f"data.yaml not found: {data_path}")

    # Load pretrained model 
    model = YOLO(args.model)

    # Train
    model.train(
        data=str(data_path),
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        patience=args.patience,
        seed=args.seed,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=args.name,
        # Good defaults:
        pretrained=True,
        cache=False,   
        amp=True,      
        verbose=True,
        # lr0=args.lr0, #learning rate
        # lrf=args.lrf,
    )

    #run validation at the end (Ultralytics also validates during training)
    model.val(data=str(data_path), imgsz=args.imgsz, device=args.device)


if __name__ == "__main__":
    main()
