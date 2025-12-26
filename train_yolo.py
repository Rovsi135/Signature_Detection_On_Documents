# train_yolo.py
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
        lr0=args.lr0, #learning rate
        lrf=args.lrf,
    )

    #run validation at the end (Ultralytics also validates during training)
    model.val(data=str(data_path), imgsz=args.imgsz, device=args.device)


if __name__ == "__main__":
    main()
