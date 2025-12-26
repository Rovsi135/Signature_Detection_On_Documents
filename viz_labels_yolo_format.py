from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

try:
    import yaml  # optional (for reading data.yaml)
except Exception:
    yaml = None


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def load_names_from_data_yaml(data_yaml: Path) -> dict[int, str] | None:
    """
    Reads Ultralytics-style data.yaml and returns {class_id: class_name}.
    Supports both:
      names: ['signature']
    and
      names:
        0: signature
    """
    if yaml is None:
        print("[Warn] PyYAML not installed, cannot parse data.yaml for class names.")
        return None
    if not data_yaml.exists():
        print(f"[Warn] data.yaml not found: {data_yaml}")
        return None

    d = yaml.safe_load(data_yaml.read_text())
    names = d.get("names", None)
    if names is None:
        return None

    if isinstance(names, list):
        return {i: str(n) for i, n in enumerate(names)}
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    return None


def list_images(images_dir: Path, recursive: bool) -> list[Path]:
    if recursive:
        imgs = [p for p in images_dir.rglob("*") if p.suffix.lower() in IMG_EXTS]
    else:
        imgs = [p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS]
    return sorted(imgs, key=lambda p: p.name)


def read_yolo_labels(label_path: Path) -> np.ndarray:
    """
    Returns Nx5 array: [cls, xc, yc, w, h] (all float), normalized.
    """
    rows = []
    if not label_path.exists():
        return np.zeros((0, 5), dtype=np.float32)

    txt = label_path.read_text().strip()
    if not txt:
        return np.zeros((0, 5), dtype=np.float32)

    for ln in txt.splitlines():
        parts = ln.strip().split()
        if len(parts) < 5:
            continue
        try:
            cls = float(parts[0])
            xc = float(parts[1])
            yc = float(parts[2])
            w = float(parts[3])
            h = float(parts[4])
            rows.append([cls, xc, yc, w, h])
        except ValueError:
            continue

    if not rows:
        return np.zeros((0, 5), dtype=np.float32)
    return np.array(rows, dtype=np.float32)


def yolo_to_xyxy_px(xc, yc, bw, bh, W, H):
    x1 = (xc - bw / 2.0) * W
    y1 = (yc - bh / 2.0) * H
    x2 = (xc + bw / 2.0) * W
    y2 = (yc + bh / 2.0) * H
    return x1, y1, x2, y2


def clamp_xyxy(x1, y1, x2, y2, W, H):
    x1c = max(0, min(W - 1, x1))
    y1c = max(0, min(H - 1, y1))
    x2c = max(0, min(W - 1, x2))
    y2c = max(0, min(H - 1, y2))
    return x1c, y1c, x2c, y2c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, help="Images directory (jpg/png/...)")
    ap.add_argument("--labels", required=True, help="Labels directory (YOLO txt files)")
    ap.add_argument("--data", default=None, help="Optional data.yaml (for class names)")
    ap.add_argument("--recursive", action="store_true", help="Search images recursively")
    ap.add_argument("--sample", type=int, default=0, help="If >0, view random sample of this many images")
    ap.add_argument("--start", type=str, default=None, help="Start from a specific image filename (e.g. 001.jpg)")
    ap.add_argument("--show_empty", action="store_true", help="Include images with no labels (default: yes)")
    args = ap.parse_args()

    images_dir = Path(args.images)
    labels_dir = Path(args.labels)

    if not images_dir.exists():
        raise FileNotFoundError(f"Images dir not found: {images_dir}")
    if not labels_dir.exists():
        raise FileNotFoundError(f"Labels dir not found: {labels_dir}")

    names_map = None
    if args.data is not None:
        names_map = load_names_from_data_yaml(Path(args.data))

    imgs = list_images(images_dir, recursive=args.recursive)
    if not imgs:
        raise RuntimeError(f"No images found in {images_dir}")

    # Optionally filter out empty-label images (if user wants that)
    if not args.show_empty:
        keep = []
        for img_path in imgs:
            lab = labels_dir / (img_path.stem + ".txt")
            if lab.exists() and lab.stat().st_size > 0:
                keep.append(img_path)
        imgs = keep
        if not imgs:
            raise RuntimeError("After filtering empties, no images remain.")

    if args.sample and args.sample > 0:
        imgs = random.sample(imgs, min(args.sample, len(imgs)))

    # Start index
    idx = 0
    if args.start is not None:
        for i, p in enumerate(imgs):
            if p.name == args.start:
                idx = i
                break

    print(f"[Info] {len(imgs)} images loaded.")
    print("[Controls] n/→ = next, p/← = prev, q/esc = quit")

    fig, ax = plt.subplots(figsize=(10, 10))
    plt.tight_layout()

    def show_at(i: int):
        ax.clear()
        img_path = imgs[i]

        img = Image.open(img_path).convert("RGB")
        arr = np.array(img)
        H, W = arr.shape[0], arr.shape[1]
        ax.imshow(arr)

        label_path = labels_dir / (img_path.stem + ".txt")
        y = read_yolo_labels(label_path)

        # Draw boxes
        for row in y:
            cls, xc, yc, bw, bh = row.tolist()
            x1, y1, x2, y2 = yolo_to_xyxy_px(xc, yc, bw, bh, W, H)
            x1, y1, x2, y2 = clamp_xyxy(x1, y1, x2, y2, W, H)

            rect = patches.Rectangle(
                (x1, y1),
                max(0, x2 - x1),
                max(0, y2 - y1),
                linewidth=2,
                edgecolor="lime",
                facecolor="none",
            )
            ax.add_patch(rect)

            cls_i = int(round(cls))
            cls_name = names_map.get(cls_i, str(cls_i)) if names_map else str(cls_i)
            ax.text(
                x1,
                max(0, y1 - 5),
                cls_name,
                fontsize=10,
                color="lime",
                bbox=dict(facecolor="black", alpha=0.4, pad=1, edgecolor="none"),
            )

        # Stats
        n = len(y)
        if n > 0:
            bw_px = y[:, 3] * W
            bh_px = y[:, 4] * H
            print(f"\n{img_path.name} | {W}x{H} | boxes={n}")
            print(f"  box_w px: min={bw_px.min():.1f} med={np.median(bw_px):.1f} max={bw_px.max():.1f}")
            print(f"  box_h px: min={bh_px.min():.1f} med={np.median(bh_px):.1f} max={bh_px.max():.1f}")
        else:
            print(f"\n{img_path.name} | {W}x{H} | boxes=0 (label: {'missing' if not label_path.exists() else 'empty'})")

        ax.set_title(f"{img_path.name}  ({i+1}/{len(imgs)})")
        ax.axis("off")
        fig.canvas.draw_idle()

    def on_key(event):
        nonlocal idx
        if event.key in ["q", "escape"]:
            plt.close(fig)
            return
        if event.key in ["n", "right"]:
            idx = (idx + 1) % len(imgs)
            show_at(idx)
        if event.key in ["p", "left"]:
            idx = (idx - 1) % len(imgs)
            show_at(idx)

    fig.canvas.mpl_connect("key_press_event", on_key)
    show_at(idx)
    plt.show()


if __name__ == "__main__":
    main()
