#!/usr/bin/env python3
import argparse
import json
import random
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2

from config_utils import (
    add_common_config_args,
    load_config_from_args,
    project_path,
)


def get_sequence_name(xml_path):
    root = ET.parse(xml_path).getroot()
    return root.attrib.get("name", xml_path.stem)


def get_ignore_boxes(root):
    ignore_boxes = []
    ignored_region = root.find("ignored_region")
    if ignored_region is None:
        return ignore_boxes

    for box in ignored_region.findall("box"):
        left = int(float(box.get("left", 0)))
        top = int(float(box.get("top", 0)))
        width = int(float(box.get("width", 0)))
        height = int(float(box.get("height", 0)))
        ignore_boxes.append((left, top, left + width, top + height))

    return ignore_boxes


def apply_ignore_mask(image, ignore_boxes, mode, kernel_size, sigma):
    image_height, image_width = image.shape[:2]

    if mode not in {"black", "gaussian", "none"}:
        raise ValueError(
            "preprocessing.ignore_mask_mode must be black, gaussian, or none"
        )

    if mode == "none":
        return image

    if kernel_size % 2 == 0:
        kernel_size += 1

    for x1, y1, x2, y2 in ignore_boxes:
        x1 = max(0, min(x1, image_width))
        x2 = max(0, min(x2, image_width))
        y1 = max(0, min(y1, image_height))
        y2 = max(0, min(y2, image_height))

        if x2 <= x1 or y2 <= y1:
            continue

        if mode == "black":
            image[y1:y2, x1:x2] = 0
        elif mode == "gaussian":
            roi = image[y1:y2, x1:x2]
            image[y1:y2, x1:x2] = cv2.GaussianBlur(
                roi,
                (kernel_size, kernel_size),
                sigmaX=sigma
            )

    return image


def list_xml_files(annotation_dir):
    if not annotation_dir.exists():
        return []
    return sorted(annotation_dir.glob("*.xml"), key=lambda path: path.name)


def build_split_manifest(config):
    train_annotation_dir = project_path(config["paths"]["raw_train_annotations_dir"])
    test_annotation_dir = project_path(config["paths"]["raw_test_annotations_dir"])
    train_ratio = config["split"]["train_ratio"]
    random_seed = config["split"]["random_seed"]

    train_xml_files = list_xml_files(train_annotation_dir)
    train_sequences = [get_sequence_name(path) for path in train_xml_files]

    shuffled_sequences = sorted(train_sequences)
    random.Random(random_seed).shuffle(shuffled_sequences)
    split_index = int(len(shuffled_sequences) * train_ratio)

    test_sequences = [
        get_sequence_name(path) for path in list_xml_files(test_annotation_dir)
    ]

    return {
        "random_seed": random_seed,
        "train_ratio": train_ratio,
        "splits": {
            "train": sorted(shuffled_sequences[:split_index]),
            "val": sorted(shuffled_sequences[split_index:]),
            "test": sorted(test_sequences)
        }
    }


def write_split_manifest(config, manifest):
    manifest_path = project_path(config["paths"]["split_manifest_path"])
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as file_out:
        json.dump(manifest, file_out, indent=2)
        file_out.write("\n")
    print(f"[OK] Split manifest saved: {manifest_path}")


def load_split_manifest(config):
    manifest_path = project_path(config["paths"]["split_manifest_path"])
    with open(manifest_path, "r", encoding="utf-8") as file_in:
        manifest = json.load(file_in)
    print(f"[OK] Reusing split manifest: {manifest_path}")
    return manifest


def process_sequence(xml_path, config):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    sequence_name = root.attrib.get("name", xml_path.stem)

    raw_images_dir = project_path(config["paths"]["raw_images_dir"])
    output_images_dir = project_path(config["paths"]["preprocessed_images_dir"])
    source_sequence_dir = raw_images_dir / sequence_name
    output_sequence_dir = output_images_dir / sequence_name

    if not source_sequence_dir.exists():
        print(f"[WARNING] Image directory not found: {source_sequence_dir}")
        return

    output_sequence_dir.mkdir(parents=True, exist_ok=True)
    ignore_boxes = get_ignore_boxes(root)
    preprocessing_config = config["preprocessing"]

    image_files = sorted(source_sequence_dir.glob("*.jpg"), key=lambda p: p.name)
    for image_path in image_files:
        output_image_path = output_sequence_dir / image_path.name
        image = cv2.imread(str(image_path))
        if image is None:
            continue

        image = apply_ignore_mask(
            image=image,
            ignore_boxes=ignore_boxes,
            mode=preprocessing_config["ignore_mask_mode"],
            kernel_size=preprocessing_config["gaussian_kernel_size"],
            sigma=preprocessing_config["gaussian_sigma"],
        )
        cv2.imwrite(str(output_image_path), image)

    print(f"[OK] Preprocessed: {sequence_name} ({len(image_files)} images)")


def run(
    config,
    split_names,
    sequence_filter=None,
    manifest_only=False,
    overwrite_splits=False,
):
    manifest_path = project_path(config["paths"]["split_manifest_path"])
    if manifest_path.exists() and not overwrite_splits:
        manifest = load_split_manifest(config)
    else:
        if manifest_path.exists():
            print(f"[INFO] Overwriting split manifest: {manifest_path}")
        manifest = build_split_manifest(config)
        write_split_manifest(config, manifest)

    if manifest_only:
        return

    train_annotation_dir = project_path(config["paths"]["raw_train_annotations_dir"])
    test_annotation_dir = project_path(config["paths"]["raw_test_annotations_dir"])
    split_to_annotation_dir = {
        "train": train_annotation_dir,
        "val": train_annotation_dir,
        "test": test_annotation_dir,
    }

    selected_splits = ["train", "val", "test"] if "all" in split_names else split_names
    requested_sequences = set(sequence_filter or [])
    processed_sequences = set()

    for split_name in selected_splits:
        sequence_names = manifest["splits"].get(split_name, [])
        if requested_sequences:
            sequence_names = [
                name for name in sequence_names if name in requested_sequences
            ]
        annotation_dir = split_to_annotation_dir[split_name]
        sequence_lookup = {
            get_sequence_name(path): path for path in list_xml_files(annotation_dir)
        }

        print(f"\n=== PREPROCESS {split_name.upper()} ({len(sequence_names)} SEQS) ===")
        for sequence_name in sequence_names:
            if sequence_name in processed_sequences:
                continue
            xml_path = sequence_lookup.get(sequence_name)
            if xml_path is None:
                print(f"[WARNING] XML not found for sequence: {sequence_name}")
                continue
            process_sequence(xml_path, config)
            processed_sequences.add(sequence_name)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Apply UA-DETRAC ignore masks and create deterministic splits."
    )
    add_common_config_args(parser)
    parser.add_argument(
        "--split",
        action="append",
        choices=["train", "val", "test", "all"],
        default=None,
        help="Split to preprocess. Can be passed multiple times."
    )
    parser.add_argument(
        "--manifest-only",
        action="store_true",
        help="Only load or write the deterministic split manifest."
    )
    parser.add_argument(
        "--overwrite-splits",
        action="store_true",
        help="Regenerate split manifest even if it already exists."
    )
    parser.add_argument(
        "--sequence",
        action="append",
        default=[],
        help="Only preprocess this sequence. Can be passed multiple times."
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config_from_args(args)
    if config is None:
        return
    run(
        config=config,
        split_names=args.split or ["all"],
        sequence_filter=args.sequence,
        manifest_only=args.manifest_only,
        overwrite_splits=args.overwrite_splits,
    )


if __name__ == "__main__":
    main()
