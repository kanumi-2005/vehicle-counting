#!/usr/bin/env python3
import argparse
import json
import os
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

from config_utils import (
    add_common_config_args,
    load_config_from_args,
    project_path,
)


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


def get_yolo_bbox(box_attributes, image_width, image_height):
    left = float(box_attributes["left"])
    top = float(box_attributes["top"])
    width = float(box_attributes["width"])
    height = float(box_attributes["height"])
    x_center = (left + width / 2) / image_width
    y_center = (top + height / 2) / image_height
    return x_center, y_center, width / image_width, height / image_height


def get_image_size(root, config):
    sequence_attributes = root.find("sequence_attribute")
    if (
        sequence_attributes is not None
        and "width" in sequence_attributes.attrib
        and "height" in sequence_attributes.attrib
    ):
        return (
            float(sequence_attributes.attrib["width"]),
            float(sequence_attributes.attrib["height"]),
        )

    dataset_config = config["dataset"]
    return float(dataset_config["image_width"]), float(dataset_config["image_height"])


def link_or_copy_image(source_image_path, output_image_path, copy_images):
    if output_image_path.exists() or output_image_path.is_symlink():
        return

    output_image_path.parent.mkdir(parents=True, exist_ok=True)
    if copy_images:
        shutil.copy2(source_image_path, output_image_path)
    else:
        os.symlink(source_image_path.resolve(), output_image_path)


def process_sequence(xml_path, image_sequence_dir, split_name, config, copy_images):
    if not image_sequence_dir.exists():
        print(f"[WARNING] Image directory not found: {image_sequence_dir}")
        return

    tree = ET.parse(xml_path)
    root = tree.getroot()
    sequence_name = root.attrib.get("name", xml_path.stem)
    image_width, image_height = get_image_size(root, config)

    output_dir = project_path(config["paths"]["yolo_output_dir"])
    labels_dir = output_dir / split_name / "labels"
    images_dir = output_dir / split_name / "images"
    labels_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    class_map = config["yolo"]["class_map"]
    default_class_name = config["yolo"]["default_class_name"]
    default_class_id = class_map[default_class_name]

    written_frames = 0
    for frame in root.findall(".//frame"):
        frame_number = int(frame.attrib.get("num", 0))
        unique_name = f"{sequence_name}_img{frame_number:05d}"
        source_image_path = image_sequence_dir / f"img{frame_number:05d}.jpg"

        if not source_image_path.exists():
            continue

        yolo_lines = []
        for target in frame.findall(".//target"):
            box = target.find("box")
            if box is None:
                continue

            x_center, y_center, width, height = get_yolo_bbox(
                box.attrib,
                image_width,
                image_height,
            )
            attribute = target.find("attribute")
            class_name = (
                attribute.attrib.get("vehicle_type")
                if attribute is not None
                else default_class_name
            )
            class_id = class_map.get(class_name, default_class_id)
            yolo_lines.append(
                f"{class_id} {x_center:.6f} {y_center:.6f} "
                f"{width:.6f} {height:.6f}"
            )

        label_path = labels_dir / f"{unique_name}.txt"
        with open(label_path, "w", encoding="utf-8") as file_out:
            file_out.write("\n".join(yolo_lines))
            if yolo_lines:
                file_out.write("\n")

        output_image_path = images_dir / f"{unique_name}.jpg"
        link_or_copy_image(source_image_path, output_image_path, copy_images)
        written_frames += 1

    print(f"[OK] YOLO {split_name}: {sequence_name} ({written_frames} frames)")


def get_yolo_class_names(config):
    class_map = config["yolo"]["class_map"]
    id_to_name = {class_id: name for name, class_id in class_map.items()}
    return [id_to_name[class_id] for class_id in sorted(id_to_name)]


def write_yolo_data_yaml(config, data_yaml_path=None):
    output_dir = project_path(config["paths"]["yolo_output_dir"])
    yaml_path = (
        project_path(data_yaml_path)
        if data_yaml_path
        else project_path(config["paths"]["yolo_data_yaml_path"])
    )
    class_names = get_yolo_class_names(config)

    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"path: {output_dir}",
        "train: train/images",
        "val: val/images",
        "test: test/images",
        f"nc: {len(class_names)}",
        "names:",
    ]
    lines.extend(
        f"  {class_id}: {class_name}"
        for class_id, class_name in enumerate(class_names)
    )

    with open(yaml_path, "w", encoding="utf-8") as file_out:
        file_out.write("\n".join(lines))
        file_out.write("\n")

    print(f"[OK] YOLO data config saved: {yaml_path}")


def run(config, split_names, sequence_filter, copy_images, data_yaml_path=None):
    manifest = load_split_manifest(config)
    selected_splits = ["train", "val", "test"] if "all" in split_names else split_names
    requested_sequences = set(sequence_filter or [])

    train_annotation_dir = project_path(config["paths"]["raw_train_annotations_dir"])
    test_annotation_dir = project_path(config["paths"]["raw_test_annotations_dir"])
    preprocessed_images_dir = project_path(config["paths"]["preprocessed_images_dir"])
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
        sequence_names = manifest["splits"].get(split_name, [])
        if requested_sequences:
            sequence_names = [
                name for name in sequence_names if name in requested_sequences
            ]

        print(f"\n=== CONVERT YOLO {split_name.upper()} ({len(sequence_names)} SEQS) ===")
        for sequence_name in sequence_names:
            xml_path = sequence_lookup.get(sequence_name)
            if xml_path is None:
                print(f"[WARNING] XML not found for sequence: {sequence_name}")
                continue
            process_sequence(
                xml_path=xml_path,
                image_sequence_dir=preprocessed_images_dir / sequence_name,
                split_name=split_name,
                config=config,
                copy_images=copy_images,
            )

    write_yolo_data_yaml(config, data_yaml_path)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert UA-DETRAC annotations to YOLO format."
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
        "--copy-images",
        action="store_true",
        help="Copy images instead of creating symlinks."
    )
    parser.add_argument(
        "--sequence",
        action="append",
        default=[],
        help="Only convert this sequence. Can be passed multiple times."
    )
    parser.add_argument(
        "--data-yaml-path",
        help="Override paths.yolo_data_yaml_path for the generated data.yaml."
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config_from_args(args)
    if config is None:
        return

    try:
        run(
            config=config,
            split_names=args.split or ["all"],
            sequence_filter=args.sequence,
            copy_images=args.copy_images,
            data_yaml_path=args.data_yaml_path,
        )
    except FileNotFoundError as error:
        print(f"[ERROR] {error}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
