import os
import cv2
import glob
import argparse
import torch
import numpy as np
import xml.etree.ElementTree as ET


from .tracking_pipeline import TrackingPipeline


CONFIG = {
    "IMG_ROOT": "./datasets/raw/DETRAC-Images",
    "SEQMAP_DIR": "./datasets/trackeval/data/gt/seqmaps",
    "TRACKER_OUT_BASE": "./datasets/trackeval/data/trackers",
    "XML_TRAIN_ANNOTATIONS_DIR": "./datasets/raw/DETRAC-Train-Annotations-XML",
    "XML_TEST_ANNOTATIONS_DIR": "./datasets/raw/DETRAC-Test-Annotations-XML",
    "BENCHMARK": "ua-detrac",
    "TRACKER_NAME": "my_tracker",
    "MODEL_PATH": "models/weights/best.pt",
}


def get_sequences(split):
    seqmap_filename = f"{CONFIG['BENCHMARK']}-{split}.txt"
    seqmap_path = os.path.join(CONFIG["SEQMAP_DIR"], seqmap_filename)

    if not os.path.exists(seqmap_path):
        return []

    with open(seqmap_path, "r") as f:
        lines = f.read().strip().splitlines()

    if lines and lines[0] == "name":
        return lines[1:]

    return lines


def parse_ignored_regions(seq_name):
    ignored_boxes = []

    xml_path = os.path.join(
        CONFIG["XML_TEST_ANNOTATIONS_DIR"],
        f"{seq_name}.xml"
    )

    if not os.path.exists(xml_path):
        xml_path = os.path.join(
            CONFIG["XML_TRAIN_ANNOTATIONS_DIR"],
            f"{seq_name}.xml"
        )

    if not os.path.exists(xml_path):
        return ignored_boxes

    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        ignored_region = root.find("ignored_region")
        if ignored_region is not None:
            for box in ignored_region.findall("box"):
                left = float(box.get("left"))
                top = float(box.get("top"))
                width = float(box.get("width"))
                height = float(box.get("height"))

                ignored_boxes.append([left, top, left + width, top + height])
    except Exception as e:
        print(f"[ERROR] Cannot parse XML for {seq_name}: {e}")

    return ignored_boxes


def is_box_ignored(pred_box, sequence_ignore_boxes_tensor, ioa_threshold=0.55):
    if sequence_ignore_boxes_tensor is None \
        or sequence_ignore_boxes_tensor.numel() == 0:
        return False

    if not isinstance(pred_box, torch.Tensor):
        p_box = torch.tensor(pred_box, dtype=torch.float32)
    else:
        p_box = pred_box

    x1 = torch.max(p_box[0], sequence_ignore_boxes_tensor[:, 0])
    y1 = torch.max(p_box[1], sequence_ignore_boxes_tensor[:, 1])
    x2 = torch.min(p_box[2], sequence_ignore_boxes_tensor[:, 2])
    y2 = torch.min(p_box[3], sequence_ignore_boxes_tensor[:, 3])

    intersection_area = torch.clamp(x2 - x1, min=0) \
        * torch.clamp(y2 - y1, min=0)
    pred_box_area = (p_box[2] - p_box[0]) * (p_box[3] - p_box[1])

    ioas = intersection_area / pred_box_area

    if torch.any(ioas >= ioa_threshold):
        return True
    return False


def run_sequence(seq_name, split_name, pipeline):
    image_dir = os.path.join(CONFIG["IMG_ROOT"], seq_name)
    images = sorted(glob.glob(os.path.join(image_dir, "img*.jpg")))

    if len(images) == 0:
        print(f"[WARNING] No images found for sequence: {seq_name}")
        return

    sequence_ignore_boxes = parse_ignored_regions(seq_name)

    if sequence_ignore_boxes:
        sequence_ignore_boxes_tensor = torch.tensor(sequence_ignore_boxes,
                                                    dtype=torch.float32)
    else:
        sequence_ignore_boxes_tensor = None

    results = []

    for frame_id, img_path in enumerate(images, start=1):
        frame = cv2.imread(img_path)
        if frame is None:
            continue

        det = pipeline.detect(frame)
        tracked = pipeline.track(det)

        if tracked.tracker_id is None:
            continue

        for box, tid in zip(tracked.xyxy, tracked.tracker_id):
            if int(tid) < 0:
                continue

            x1, y1, x2, y2 = box

            if is_box_ignored(box, sequence_ignore_boxes_tensor):
                continue

            results.append(
                f"{frame_id},"
                f"{int(tid)},"
                f"{x1:.2f},{y1:.2f},{x2-x1:.2f},{y2-y1:.2f},"
                f"1,1,-1"
            )

    out_dir = os.path.join(
        CONFIG["TRACKER_OUT_BASE"],
        f"{CONFIG['BENCHMARK']}-{split_name}",
        CONFIG["TRACKER_NAME"],
        "data"
    )
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, f"{seq_name}.txt")

    with open(out_file, "w") as f:
        f.write("\n".join(results))

    print(f"[OK] {split_name} | {seq_name} -> {len(results)} lines")


def run(split_choice):
    pipeline = TrackingPipeline(
        model_path=CONFIG["MODEL_PATH"]
    )

    if split_choice == "all":
        splits = ["val", "test"]
    else:
        splits = [split_choice]

    for split in splits:
        seqs = get_sequences(split)

        if not seqs:
            print(f"\n[SKIP] No data for {split} (Check seqmaps).")
            continue

        print(f"\n=== RUNNING {split.upper()} SET ({len(seqs)} SEQS) ===")

        for seq in seqs:
            run_sequence(seq, split, pipeline)

    print("\n[DONE] Finished running tracking.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run tracking pipeline.")
    parser.add_argument(
        "--split",
        type=str,
        choices=["val", "test", "all"],
        default="all",
        help="Select split to run: 'val', 'test', or 'all' (default)."
    )

    args = parser.parse_args()
    run(split_choice=args.split)
