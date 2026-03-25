#!/usr/bin/env python3
"""Convert a COCO detection dataset to YOLOv8 detection format.

Example:
    python convert_coco_to_yolov8.py \
        --coco-json yolov8_001.coco/train/_annotations.coco.json \
        --output-dir train_yolo/data/min001
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path
from typing import Dict, List, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert COCO labels to YOLOv8 format")
    parser.add_argument(
        "--coco-json",
        type=Path,
        required=True,
        help="Path to _annotations.coco.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output dataset root for YOLOv8 (contains images/, labels/, data.yaml)",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Validation split ratio in [0, 1). Use 0 to disable split.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for train/val split",
    )
    return parser.parse_args()


def clamp(value: float, min_value: float, max_value: float) -> float:
    return max(min_value, min(value, max_value))


def coco_bbox_to_yolo(
    bbox: List[float], image_w: int, image_h: int
) -> Tuple[float, float, float, float] | None:
    if len(bbox) != 4:
        return None

    x, y, w, h = [float(v) for v in bbox]
    if w <= 0 or h <= 0 or image_w <= 0 or image_h <= 0:
        return None

    # Clip box to image border to avoid invalid normalized values.
    x1 = clamp(x, 0.0, float(image_w))
    y1 = clamp(y, 0.0, float(image_h))
    x2 = clamp(x + w, 0.0, float(image_w))
    y2 = clamp(y + h, 0.0, float(image_h))

    clipped_w = x2 - x1
    clipped_h = y2 - y1
    if clipped_w <= 0 or clipped_h <= 0:
        return None

    cx = (x1 + x2) / 2.0 / image_w
    cy = (y1 + y2) / 2.0 / image_h
    nw = clipped_w / image_w
    nh = clipped_h / image_h
    return (cx, cy, nw, nh)


def write_data_yaml(output_dir: Path, class_names: List[str], has_val: bool) -> None:
    yaml_path = output_dir / "data.yaml"
    train_rel = "images/train"
    val_rel = "images/val" if has_val else "images/train"

    lines = [
        f"path: {output_dir.resolve().as_posix()}",
        f"train: {train_rel}",
        f"val: {val_rel}",
        f"nc: {len(class_names)}",
        "names:",
    ]
    for i, name in enumerate(class_names):
        safe_name = str(name).replace("\n", " ").strip() or f"class_{i}"
        lines.append(f"  {i}: {safe_name}")

    yaml_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()

    if not (0.0 <= args.val_ratio < 1.0):
        raise ValueError("--val-ratio must be in [0, 1)")

    coco_json = args.coco_json.resolve()
    if not coco_json.exists():
        raise FileNotFoundError(f"COCO json not found: {coco_json}")

    dataset_dir = coco_json.parent
    output_dir = args.output_dir.resolve()

    with coco_json.open("r", encoding="utf-8") as f:
        coco = json.load(f)

    images = coco.get("images", [])
    annotations = coco.get("annotations", [])
    categories = coco.get("categories", [])

    if not images:
        raise ValueError("No images found in COCO json")

    image_info: Dict[int, dict] = {img["id"]: img for img in images}

    anns_by_image: Dict[int, List[dict]] = {img_id: [] for img_id in image_info}
    used_category_ids = set()
    for ann in annotations:
        img_id = ann.get("image_id")
        cat_id = ann.get("category_id")
        if img_id not in image_info:
            continue
        if cat_id is None:
            continue
        anns_by_image[img_id].append(ann)
        used_category_ids.add(int(cat_id))

    if not used_category_ids:
        raise ValueError("No usable annotations found in COCO json")

    category_name_by_id = {int(c["id"]): str(c.get("name", "")) for c in categories}
    sorted_used_categories = sorted(used_category_ids)
    class_id_map = {cat_id: i for i, cat_id in enumerate(sorted_used_categories)}
    class_names = [
        category_name_by_id.get(cat_id, f"class_{i}")
        for i, cat_id in enumerate(sorted_used_categories)
    ]

    all_image_ids = sorted(image_info.keys())
    random.seed(args.seed)
    random.shuffle(all_image_ids)

    val_count = int(len(all_image_ids) * args.val_ratio)
    val_ids = set(all_image_ids[:val_count])

    for split in ("train", "val"):
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    converted_boxes = 0
    skipped_boxes = 0

    for img_id in all_image_ids:
        img = image_info[img_id]
        file_name = img.get("file_name")
        if not file_name:
            continue

        src_img_path = dataset_dir / file_name
        if not src_img_path.exists():
            print(f"[WARN] Missing image file, skipping: {src_img_path}")
            continue

        split = "val" if img_id in val_ids and val_count > 0 else "train"
        dst_img_path = output_dir / "images" / split / file_name
        dst_label_path = output_dir / "labels" / split / f"{Path(file_name).stem}.txt"

        shutil.copy2(src_img_path, dst_img_path)

        iw = int(img.get("width", 0))
        ih = int(img.get("height", 0))
        lines: List[str] = []

        for ann in anns_by_image.get(img_id, []):
            cat_id = int(ann.get("category_id", -1))
            if cat_id not in class_id_map:
                skipped_boxes += 1
                continue

            yolo_box = coco_bbox_to_yolo(ann.get("bbox", []), iw, ih)
            if yolo_box is None:
                skipped_boxes += 1
                continue

            cls = class_id_map[cat_id]
            cx, cy, bw, bh = yolo_box
            lines.append(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            converted_boxes += 1

        dst_label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    write_data_yaml(output_dir, class_names, has_val=val_count > 0)

    print("=" * 60)
    print("COCO -> YOLOv8 conversion finished")
    print("=" * 60)
    print(f"Input json : {coco_json}")
    print(f"Output dir : {output_dir}")
    print(f"Images     : {len(all_image_ids)}")
    print(f"Classes    : {len(class_names)} -> {class_names}")
    print(f"Train/Val  : {len(all_image_ids) - val_count}/{val_count}")
    print(f"Boxes      : converted={converted_boxes}, skipped={skipped_boxes}")
    print(f"data.yaml  : {output_dir / 'data.yaml'}")


if __name__ == "__main__":
    main()
