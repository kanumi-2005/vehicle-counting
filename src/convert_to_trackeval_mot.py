#!/usr/bin/env python3
import argparse
import json
import sys
import xml.etree.ElementTree as ET

from config_utils import (
    add_common_config_args,
    load_config_from_args,
    project_path,
)


def safe_float(value, default=0.0):
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def safe_int(value, default=0):
    try:
        return int(value)
    except (ValueError, TypeError):
        return default


def load_split_manifest(config):
    manifest_path = project_path(config["paths"]["split_manifest_path"])
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Split manifest not found: {manifest_path}. "
            "Run src/preprocessing.py first."
        )
    with open(manifest_path, "r", encoding="utf-8") as file_in:
        return json.load(file_in)


def list_xml_files(annotation_dir):
    if not annotation_dir.exists():
        return []
    return sorted(annotation_dir.glob("*.xml"), key=lambda path: path.name)


def get_sequence_name(xml_path):
    root = ET.parse(xml_path).getroot()
    return root.attrib.get("name", xml_path.stem)


def write_gt(frames, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = []

    for frame in sorted(frames, key=lambda item: safe_int(item.attrib.get("num"))):
        frame_id = safe_int(frame.attrib.get("num"))
        if frame_id <= 0:
            continue

        for target in frame.findall(".//target"):
            box = target.find("box")
            if box is None:
                continue

            width = safe_float(box.attrib.get("width"))
            height = safe_float(box.attrib.get("height"))
            if width <= 0 or height <= 0:
                continue

            object_id = safe_int(target.attrib.get("id"), default=-1)
            attribute = target.find("attribute")
            truncation_ratio = 0.0
            if attribute is not None:
                truncation_ratio = safe_float(
                    attribute.attrib.get("truncation_ratio"),
                    default=0.0,
                )
            visibility = max(0.0, min(1.0, 1.0 - truncation_ratio))

            lines.append(
                f"{frame_id},{object_id},"
                f"{safe_float(box.attrib.get('left')):.2f},"
                f"{safe_float(box.attrib.get('top')):.2f},"
                f"{width:.2f},{height:.2f},1,1,{visibility:.2f}"
            )

    with open(output_path, "w", encoding="utf-8") as file_out:
        file_out.write("\n".join(lines))
        if lines:
            file_out.write("\n")


def get_sequence_length(frames):
    frame_ids = [safe_int(frame.attrib.get("num")) for frame in frames]
    return max(frame_ids) if frame_ids else 0


def write_seqinfo(sequence_dir, sequence_name, frames, config):
    sequence_dir.mkdir(parents=True, exist_ok=True)
    dataset_config = config["dataset"]
    content = (
        "[Sequence]\n"
        f"name={sequence_name}\n"
        "imDir=img1\n"
        f"frameRate={dataset_config['frame_rate']}\n"
        f"seqLength={get_sequence_length(frames)}\n"
        f"imWidth={dataset_config['image_width']}\n"
        f"imHeight={dataset_config['image_height']}\n"
        f"imExt={dataset_config['image_ext']}\n"
    )
    with open(sequence_dir / "seqinfo.ini", "w", encoding="utf-8") as file_out:
        file_out.write(content)


def process_sequence(xml_path, split_name, config):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    sequence_name = root.attrib.get("name", xml_path.stem)
    frames = root.findall(".//frame")

    if not frames:
        print(f"[SKIP] Empty: {sequence_name}")
        return None

    trackeval_data_dir = project_path(config["paths"]["trackeval_data_dir"])
    benchmark_name = config["dataset"]["benchmark_name"]
    sequence_dir = trackeval_data_dir / "gt" / f"{benchmark_name}-{split_name}" / sequence_name

    write_gt(frames, sequence_dir / "gt" / "gt.txt")
    write_seqinfo(sequence_dir, sequence_name, frames, config)
    print(f"[OK] TrackEval {split_name}: {sequence_name} ({len(frames)} frames)")
    return sequence_name


def write_seqmap(sequence_names, split_name, config):
    trackeval_data_dir = project_path(config["paths"]["trackeval_data_dir"])
    benchmark_name = config["dataset"]["benchmark_name"]
    seqmap_dir = trackeval_data_dir / "gt" / "seqmaps"
    seqmap_dir.mkdir(parents=True, exist_ok=True)
    seqmap_path = seqmap_dir / f"{benchmark_name}-{split_name}.txt"

    with open(seqmap_path, "w", encoding="utf-8") as file_out:
        file_out.write("name\n")
        for sequence_name in sequence_names:
            file_out.write(sequence_name + "\n")

    print(f"[OK] Seqmap saved: {seqmap_path}")


def run(config, split_names, sequence_filter=None):
    manifest = load_split_manifest(config)
    selected_splits = ["train", "val", "test"] if "all" in split_names else split_names
    requested_sequences = set(sequence_filter or [])

    train_annotation_dir = project_path(config["paths"]["raw_train_annotations_dir"])
    test_annotation_dir = project_path(config["paths"]["raw_test_annotations_dir"])
    split_to_annotation_dir = {
        "train": train_annotation_dir,
        "val": train_annotation_dir,
        "test": test_annotation_dir,
    }

    for split_name in selected_splits:
        annotation_dir = split_to_annotation_dir[split_name]
        sequence_lookup = {
            get_sequence_name(path): path for path in list_xml_files(annotation_dir)
        }
        processed_sequences = []
        sequence_names = manifest["splits"].get(split_name, [])
        if requested_sequences:
            sequence_names = [
                name for name in sequence_names if name in requested_sequences
            ]

        print(f"\n=== CONVERT TRACKEVAL {split_name.upper()} ({len(sequence_names)} SEQS) ===")
        for sequence_name in sequence_names:
            xml_path = sequence_lookup.get(sequence_name)
            if xml_path is None:
                print(f"[WARNING] XML not found for sequence: {sequence_name}")
                continue
            processed_sequence = process_sequence(xml_path, split_name, config)
            if processed_sequence:
                processed_sequences.append(processed_sequence)

        write_seqmap(processed_sequences, split_name, config)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert UA-DETRAC annotations to TrackEval MOT format."
    )
    add_common_config_args(parser)
    parser.add_argument(
        "--split",
        action="append",
        choices=["train", "val", "test", "all"],
        default=None,
        help="Split to convert. Can be passed multiple times."
    )
    parser.add_argument(
        "--sequence",
        action="append",
        default=[],
        help="Only convert this sequence. Can be passed multiple times."
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config_from_args(args)
    if config is None:
        return

    try:
        run(config, args.split or ["val", "test"], args.sequence)
    except FileNotFoundError as error:
        print(f"[ERROR] {error}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
