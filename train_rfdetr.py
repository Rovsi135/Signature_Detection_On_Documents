"""
Train RF-DETR on a COCO-format dataset.

Purpose
-------
This script starts RF-DETR training from the command line using a dataset in
COCO format. It is meant to make experiments easier to run, repeat, and
document.

Typical use case
----------------
Use this script when your dataset is already prepared in COCO format and you
want to train RF-DETR in a clean, reproducible way.

Expected dataset structure
--------------------------
The dataset directory must look like this:

    <dataset_dir>/
        train/
            _annotations.coco.json
            ...
        valid/
            _annotations.coco.json
            ...
        test/          # optional for training, useful for later evaluation
            _annotations.coco.json
            ...

At minimum, this script requires:
- train/_annotations.coco.json
- valid/_annotations.coco.json

The image files referenced inside those COCO JSON files must also be present in
the corresponding split folders.

Command-line API
----------------

Required arguments
~~~~~~~~~~~~~~~~~~
--dataset_dir <path>
    Path to the COCO dataset root.
    This folder must contain at least:
    - train/_annotations.coco.json
    - valid/_annotations.coco.json

--output_dir <path>
    Path to the folder where checkpoints, logs, and other training outputs will
    be saved.

Optional arguments
~~~~~~~~~~~~~~~~~~
--model <name>
    Which RF-DETR model size to train.

    Choices:
    - nano
    - small
    - base
    - medium
    - large

    Default:
        small

--epochs <int>
    Number of training epochs.

    Default:
        20

--batch_size <int>
    Number of images processed in one training step.

    Default:
        2

    Note:
        Lower this if you run out of GPU memory.

--grad_accum_steps <int>
    Number of gradient accumulation steps.

    This is used to simulate a larger effective batch size when GPU memory is
    limited.

    Approximate effective batch size:
        batch_size * grad_accum_steps

    Default:
        8

--resolution <int>
    Input image size used during training.

    Recommended values are multiples of 224, for example:
    - 448
    - 672
    - 896

--seed <int>
    Random seed for reproducibility.

    Default:
        0

--gradient_checkpointing
    Enable gradient checkpointing.

    This reduces GPU memory usage but usually makes training slower.

--no_gradient_checkpointing
    Force gradient checkpointing to be disabled, even if
    --gradient_checkpointing was also passed.

--wandb
    Enable Weights & Biases logging.

--wandb_project <name>
    Name of the Weights & Biases project.

    Default:
        sigdet-rfdetr

--wandb_run <name>
    Optional run name for Weights & Biases.
    If omitted, the run name is auto-generated.

--tensorboard
    Enable TensorBoard logging.

--no_tensorboard
    Disable TensorBoard logging, even if --tensorboard was also passed.

Logging behavior
----------------
Weights & Biases:
- disabled by default
- enabled only if you pass --wandb

TensorBoard:
- disabled by default
- enabled only if you pass --tensorboard
- can be force-disabled with --no_tensorboard

Gradient checkpointing:
- disabled by default
- enabled only if you pass --gradient_checkpointing
- can be force-disabled with --no_gradient_checkpointing

Example commands
----------------
Basic training:

    python train_rfdetr.py \
        --dataset_dir /path/to/coco_dataset \
        --output_dir /path/to/output

Train a larger model with W&B logging:

    python train_rfdetr.py \
        --dataset_dir /path/to/coco_dataset \
        --output_dir /path/to/output \
        --model small \
        --epochs 50 \
        --batch_size 2 \
        --grad_accum_steps 8 \
        --wandb \
        --wandb_project sigdet-rfdetr

Train with TensorBoard and gradient checkpointing:

    python train_rfdetr.py \
        --dataset_dir /path/to/coco_dataset \
        --output_dir /path/to/output \
        --tensorboard \
        --gradient_checkpointing

Important notes
---------------
- This script expects the dataset to already be in COCO format.
- It does not convert annotations.
- It does not evaluate on the test set by itself.
- It only checks that the train and valid annotation files exist.
- The chosen batch size and resolution must fit in available GPU memory.

Dependencies
------------
This script requires the RF-DETR package to be installed.

It imports the RF-DETR classes inside main(), so that:
- `python train_rfdetr.py --help` still works even if RF-DETR is not installed
- import errors appear only when training is actually started
"""
from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train RF-DETR on a COCO-style dataset directory.")

    # Required paths
    p.add_argument("--dataset_dir", type=str, required=True,
                   help="COCO dataset root containing train/valid/test folders (each with _annotations.coco.json).")
    p.add_argument("--output_dir", type=str, required=True,
                   help="Where to write checkpoints/logs.")

    # Model / training
    p.add_argument("--model", type=str, default="small", choices=["nano", "small", "base", "medium", "large"],
                   help="Which RF-DETR size to use.")
    p.add_argument("--epochs", type=int, default=20,
                   help="Number of training epochs.")
    p.add_argument("--batch_size", type=int, default=2,
                   help="Per-step batch size (reduce if OOM).")
    p.add_argument("--grad_accum_steps", type=int, default=8,
                   help="Gradient accumulation steps. Effective batch ~= batch_size * grad_accum_steps.")
    p.add_argument("--resolution", type=int, default=672,
                   help="Training resolution. Prefer multiples of 224 (e.g., 448/672/896).")
    p.add_argument("--seed", type=int, default=0,
                   help="Random seed.")

    # Memory / speed
    p.add_argument("--gradient_checkpointing", action="store_true",
                   help="Enable gradient checkpointing to reduce VRAM (slower).")
    p.add_argument("--no_gradient_checkpointing", action="store_true",
                   help="Force-disable gradient checkpointing (overrides --gradient_checkpointing).")

    # Logging
    p.add_argument("--wandb", action="store_true",
                   help="Enable Weights & Biases logging.")
    p.add_argument("--wandb_project", type=str, default="sigdet-rfdetr",
                   help="W&B project name (groups runs together).")
    p.add_argument("--wandb_run", type=str, default=None,
                   help="Optional W&B run name (otherwise auto-generated).")

    p.add_argument("--tensorboard", action="store_true",
                   help="Enable TensorBoard logging (requires tensorboard installed).")
    p.add_argument("--no_tensorboard", action="store_true",
                   help="Disable TensorBoard logging (overrides --tensorboard).")

    return p.parse_args()


def validate_dataset_dir(dataset_dir: Path) -> None:
    required = [
        dataset_dir / "train" / "_annotations.coco.json",
        dataset_dir / "valid" / "_annotations.coco.json",
    ]
    missing = [str(x) for x in required if not x.exists()]
    if missing:
        raise FileNotFoundError("COCO dataset structure not found. Missing: " + ", ".join(missing))


def main() -> None:
    args = parse_args()

    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    validate_dataset_dir(dataset_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Import here so `--help` works even if rfdetr isn't installed
    from rfdetr import RFDETRNano, RFDETRSmall, RFDETRBase, RFDETRMedium, RFDETRLarge

    model_map = {
        "nano": RFDETRNano,
        "small": RFDETRSmall,
        "base": RFDETRBase,
        "medium": RFDETRMedium,
        "large": RFDETRLarge,
    }

    use_gc = bool(args.gradient_checkpointing and not args.no_gradient_checkpointing)

    # Default behavior: tensorboard OFF unless explicitly enabled (or not overridden)
    tb_enabled = bool(args.tensorboard and not args.no_tensorboard)

    # W&B only if you pass --wandb
    wandb_enabled = bool(args.wandb)

    model = model_map[args.model]()

    model.train(
        dataset_dir=str(dataset_dir),
        output_dir=str(output_dir),
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        resolution=args.resolution,
        gradient_checkpointing=use_gc,
        seed=args.seed,
        wandb=wandb_enabled,
        project=(args.wandb_project if wandb_enabled else None),
        run=(args.wandb_run if wandb_enabled else None),
        tensorboard=tb_enabled,
    )


if __name__ == "__main__":
    main()
