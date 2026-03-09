from __future__ import annotations
import json
from pathlib import Path
from PIL import Image
import math

def check_split(split_dir: Path) -> None:
    ann_path = split_dir / "_annotations.coco.json"
    data = json.loads(ann_path.read_text())

    images = {im["id"]: im for im in data["images"]}
    anns_by_image = {}
    for ann in data["annotations"]:
        anns_by_image.setdefault(ann["image_id"], []).append(ann)

    for image_id, im in images.items():
        fn = im["file_name"]
        w_json, h_json = im["width"], im["height"]
        img_path = split_dir / fn

        # File exists + open
        if not img_path.exists():
            raise FileNotFoundError(f"[{split_dir.name}] Missing image file: {img_path}")

        with Image.open(img_path) as img:
            w_real, h_real = img.size

        # Dimensions match
        if (w_real, h_real) != (w_json, h_json):
            raise ValueError(
                f"[{split_dir.name}] SIZE MISMATCH for {fn}: "
                f"json=({w_json},{h_json}) real=({w_real},{h_real})"
            )

        # Check all bboxes
        for ann in anns_by_image.get(image_id, []):
            x, y, bw, bh = ann["bbox"]

            vals = [x, y, bw, bh]
            if any((not isinstance(v, (int, float))) for v in vals) or any((not math.isfinite(v)) for v in vals):
                raise ValueError(f"[{split_dir.name}] Non-finite bbox in {fn}: {ann['bbox']}")

            if bw <= 0 or bh <= 0:
                raise ValueError(f"[{split_dir.name}] Non-positive bbox in {fn}: {ann['bbox']}")

            if x < 0 or y < 0:
                raise ValueError(f"[{split_dir.name}] Negative bbox coord in {fn}: {ann['bbox']}")

            if x + bw > w_json + 1e-3 or y + bh > h_json + 1e-3:
                raise ValueError(
                    f"[{split_dir.name}] BBOX OOB in {fn}: {ann['bbox']} vs image=({w_json},{h_json})"
                )

    print(f"[OK] {split_dir} passed. images={len(data['images'])} anns={len(data['annotations'])}")

def main():
    root = Path("/home/rovsi/Projects/Vakifbank_Internship/Signature_Detection/")
    for split in ["train", "valid", "test"]:
        check_split(root / split)

if __name__ == "__main__":
    main()
