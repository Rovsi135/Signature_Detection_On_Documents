import argparse
import random
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from PIL import Image
import tifffile

"""python viz_labels_csv_tif.py \
  --csv /home/rovsi/Projects/Vakifbank_Internship/Signature_Detection/KaggleDataSet/images/Train_data.csv \
  --img_dir /home/rovsi/Projects/Vakifbank_Internship/Signature_Detection/KaggleDataSet/images \
  --no_header
 """

"""
"""

def read_image_any_tif(path: Path) -> np.ndarray:
    """
    - uses tifffile for .tif
    - returns an array shaped (H,W) or (H,W,3)
    - if multi-page, uses first page
    """
    if path.suffix.lower() in [".tif", ".tiff"]:
        arr = tifffile.imread(str(path))
        # Multi-page TIFF often: (pages, H, W) or (pages, H, W, C)
        if arr.ndim == 3 and arr.shape[0] < 10 and arr.shape[-1] not in (3, 4):
            # likely (pages,H,W)
            arr = arr[0]
        elif arr.ndim == 4:
            # likely (pages,H,W,C)
            arr = arr[0]
        return arr
    else:
        img = Image.open(path)
        return np.array(img)


def to_uint8_for_display(arr: np.ndarray) -> np.ndarray:
    """
    Converts image arrays (possibly 16-bit) into uint8 for display.
    Preserves grayscale or RGB-ish shapes.
    """
    if arr.dtype == np.uint8:
        return arr

    # Handle 16-bit grayscale or other integer types
    arr_f = arr.astype(np.float32)
    mn, mx = float(np.min(arr_f)), float(np.max(arr_f))
    if mx <= mn + 1e-6:
        return np.zeros_like(arr, dtype=np.uint8)

    arr_norm = (arr_f - mn) / (mx - mn)
    arr_u8 = (arr_norm * 255.0).clip(0, 255).astype(np.uint8)
    return arr_u8


def guess_yolo_xywh(df: pd.DataFrame) -> bool:
    """
    Detect if the CSV is YOLO-style normalized xywh:
    values are typically in [0,1].
    """
    cols = ["x_min", "y_min", "x_max", "y_max"]
    sub = df[cols].astype(float)
    mx = sub.max().max()
    mn = sub.min().min()
    return (mx <= 1.5) and (mn >= -0.25)


def draw_boxes(ax, rows: pd.DataFrame, w: int, h: int, yolo_xywh: bool):
    for _, r in rows.iterrows():
        a = float(r.x_min)
        b = float(r.y_min)
        c = float(r.x_max)
        d = float(r.y_max)

        if yolo_xywh:
            # Interpret as normalized YOLO: (x_center, y_center, box_w, box_h)
            x_center = a * w
            y_center = b * h
            bw = c * w
            bh = d * h
            x1 = x_center - bw / 2
            y1 = y_center - bh / 2
            x2 = x_center + bw / 2
            y2 = y_center + bh / 2
        else:
            # Interpret as pixel corners: (x_min, y_min, x_max, y_max)
            x1, y1, x2, y2 = a, b, c, d

        # clamp for display
        x1c, y1c = max(0, x1), max(0, y1)
        x2c, y2c = min(w - 1, x2), min(h - 1, y2)

        rect = patches.Rectangle(
            (x1c, y1c),
            x2c - x1c,
            y2c - y1c,
            linewidth=2,
            edgecolor="lime",
            facecolor="none",
        )
        ax.add_patch(rect)

        ax.text(
            x1c,
            max(0, y1c - 5),
            str(getattr(r, "label", "")),
            fontsize=10,
            color="lime",
            bbox=dict(facecolor="black", alpha=0.4, pad=1, edgecolor="none"),
        )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to Train_data.csv or Test_data.csv")
    ap.add_argument("--img_dir", required=True, help="Directory containing .tif images")
    ap.add_argument("--no_header", action="store_true", help="Use if your CSV has NO header row")
    ap.add_argument("--sample", type=int, default=0, help="If >0, view a random sample of this many unique images")
    ap.add_argument("--start", type=str, default=None, help="Start from a specific filename like '680.tif'")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    img_dir = Path(args.img_dir)

    # Header handling: your current code uses names=[...] which implies NO header.
    # If your CSV actually HAS a header row, don't use --no_header.
    if args.no_header:
        df = pd.read_csv(csv_path, names=["name", "label", "x_min", "y_min", "x_max", "y_max"])
    else:
        df = pd.read_csv(csv_path)
        # If columns are unnamed, fall back to expected schema
        if df.shape[1] >= 6 and "name" not in df.columns:
            df.columns = ["name", "label", "x_min", "y_min", "x_max", "y_max"]

    df["name"] = df["name"].astype(str)
    df = df.sort_values("name").reset_index(drop=True)

    # Auto-detect normalized vs pixel coords
    yolo_xywh = guess_yolo_xywh(df)
    print(f"[Info] CSV box format guess: {'YOLO xywh normalized' if yolo_xywh else 'pixel x1y1x2y2'}")      

    # Build list of unique image names referenced in CSV
    names = df["name"].dropna().unique().tolist()
    print(names)
    print(len(names), "unique images in CSV")

    if args.sample and args.sample > 0:
        names = random.sample(names, min(args.sample, len(names)))

    # If start is provided, jump to it
    idx = 0
    if args.start is not None and args.start in names:
        idx = names.index(args.start)

    print("[Controls] n/→ = next, p/← = prev, q = quit")

    fig, ax = plt.subplots(figsize=(10, 10))
    plt.tight_layout()

    def show_at(i: int):
        ax.clear()
        name = names[i]
        img_path = img_dir / name
        if not img_path.exists():
            ax.set_title(f"Missing file: {name}")
            ax.axis("off")
            fig.canvas.draw_idle()
            return

        arr = read_image_any_tif(img_path)
        arr_disp = to_uint8_for_display(arr)

        # Determine image shape
        if arr_disp.ndim == 2:
            h, w = arr_disp.shape
            ax.imshow(arr_disp, cmap="gray")
        else:
            h, w = arr_disp.shape[0], arr_disp.shape[1]
            # If channels are last and >3, keep first 3
            if arr_disp.shape[-1] > 3:
                arr_disp = arr_disp[..., :3]
            ax.imshow(arr_disp)

        rows = df[df["name"] == name]
        draw_boxes(ax, rows, w, h, yolo_xywh)

        # Print quick stats
        print(f"\n{name} | size={w}x{h} | boxes={len(rows)}")
        if len(rows) > 0:
            # Compute box sizes in pixels (for yolo_xywh, scale first)
            xs1 = rows["x_min"].astype(float).to_numpy()
            ys1 = rows["y_min"].astype(float).to_numpy()
            xs2 = rows["x_max"].astype(float).to_numpy()
            ys2 = rows["y_max"].astype(float).to_numpy()
            if yolo_xywh:
                xs1, xs2 = xs1 * w, xs2 * w
                ys1, ys2 = ys1 * h, ys2 * h
            bw = np.maximum(0, xs2 - xs1)
            bh = np.maximum(0, ys2 - ys1)
            print(f"  box_w px: min={bw.min():.1f} med={np.median(bw):.1f} max={bw.max():.1f}")
            print(f"  box_h px: min={bh.min():.1f} med={np.median(bh):.1f} max={bh.max():.1f}")

        ax.set_title(f"{name}  ({i+1}/{len(names)})")
        ax.axis("off")
        fig.canvas.draw_idle()

    def on_key(event):
        nonlocal idx
        if event.key in ["q", "escape"]:
            plt.close(fig)
            return
        if event.key in ["n", "right"]:
            idx = (idx + 1) % len(names)
            show_at(idx)
        if event.key in ["p", "left"]:
            idx = (idx - 1) % len(names)
            show_at(idx)

    fig.canvas.mpl_connect("key_press_event", on_key)
    show_at(idx)
    plt.show()



if __name__ == "__main__":
    main()
