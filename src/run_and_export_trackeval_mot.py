#!/usr/bin/env python3
import argparse
import glob
import sys

import cv2

from config_utils import (
    add_common_config_args,
    get_tracker_output_name,
    load_config_from_args,
    project_path,
)


def get_sequences(config, split_name):
    trackeval_data_dir = project_path(config["paths"]["trackeval_data_dir"])
    benchmark_name = config["dataset"]["benchmark_name"]
    seqmap_path = (
        trackeval_data_dir
        / "gt"
        / "seqmaps"
        / f"{benchmark_name}-{split_name}.txt"
    )

    if not seqmap_path.exists():
        return []

    with open(seqmap_path, "r", encoding="utf-8") as file_in:
        lines = [line.strip() for line in file_in if line.strip()]

    if lines and lines[0] == "name":
        return lines[1:]
    return lines


def run_sequence(sequence_name, split_name, pipeline, config):
    preprocessed_images_dir = project_path(config["paths"]["preprocessed_images_dir"])
    image_dir = preprocessed_images_dir / sequence_name
    image_paths = sorted(glob.glob(str(image_dir / "img*.jpg")))

    if not image_paths:
        print(f"[WARNING] No images found for sequence: {sequence_name}")
        return

    results = []
    for frame_id, image_path in enumerate(image_paths, start=1):
        frame = cv2.imread(image_path)
        if frame is None:
            continue

        detections = pipeline.detect(frame)
        tracked_detections = pipeline.track(detections)

        if tracked_detections.tracker_id is None:
            continue

        for box, track_id in zip(
            tracked_detections.xyxy,
            tracked_detections.tracker_id,
        ):
            if int(track_id) < 0:
                continue

            x1, y1, x2, y2 = box
            results.append(
                f"{frame_id},{int(track_id)},"
                f"{x1:.2f},{y1:.2f},{x2 - x1:.2f},{y2 - y1:.2f},"
                "1,1,-1"
            )

    tracker_output_dir = project_path(config["paths"]["tracker_output_dir"])
    benchmark_name = config["dataset"]["benchmark_name"]
    tracker_name = get_tracker_output_name(config)
    output_dir = tracker_output_dir / f"{benchmark_name}-{split_name}" / tracker_name / "data"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{sequence_name}.txt"

    with open(output_path, "w", encoding="utf-8") as file_out:
        file_out.write("\n".join(results))
        if results:
            file_out.write("\n")

    print(f"[OK] {split_name} | {sequence_name} -> {len(results)} lines")


def build_pipeline(config):
    from tracking_pipeline import TrackingPipeline

    detection_config = config["detection"]
    tracking_config = config["tracking"]
    return TrackingPipeline(
        model_path=str(project_path(config["paths"]["model_path"])),
        confidence_threshold=detection_config["confidence_threshold"],
        device=detection_config["device"],
        verbose=detection_config["verbose"],
        tracker_type=tracking_config["tracker_type"],
        lost_track_buffer=tracking_config["lost_track_buffer"],
        frame_rate=config["dataset"]["frame_rate"],
        track_activation_threshold=tracking_config["track_activation_threshold"],
        minimum_consecutive_frames=tracking_config["minimum_consecutive_frames"],
        minimum_iou_threshold=tracking_config["minimum_iou_threshold"],
        high_conf_detection_threshold=tracking_config[
            "high_conf_detection_threshold"
        ],
    )


def run(config, split_names, sequence_filter):
    selected_splits = ["val", "test"] if "all" in split_names else split_names
    requested_sequences = set(sequence_filter or [])
    pipeline = build_pipeline(config)

    for split_name in selected_splits:
        sequence_names = get_sequences(config, split_name)
        if requested_sequences:
            sequence_names = [
                name for name in sequence_names if name in requested_sequences
            ]

        if not sequence_names:
            print(f"\n[SKIP] No sequences for {split_name}.")
            continue

        print(f"\n=== RUN TRACKING {split_name.upper()} ({len(sequence_names)} SEQS) ===")
        for sequence_name in sequence_names:
            run_sequence(sequence_name, split_name, pipeline, config)

    print("\n[DONE] Finished running tracking.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run detection/tracking and export TrackEval MOT files."
    )
    add_common_config_args(parser)
    parser.add_argument(
        "--split",
        action="append",
        choices=["train", "val", "test", "all"],
        default=None,
        help="Split to run. Can be passed multiple times."
    )
    parser.add_argument(
        "--sequence",
        action="append",
        default=[],
        help="Run only this sequence. Can be passed multiple times."
    )
    parser.add_argument("--model-path", help="Override paths.model_path.")
    parser.add_argument("--detector", help="Override detection.detector_name.")
    parser.add_argument(
        "--tracker",
        choices=["byte", "sort"],
        help="Override tracking.tracker_type."
    )
    parser.add_argument(
        "--tracker-name",
        help=(
            "Override TrackEval tracker folder name. By default it is "
            "<detector>-<tracker>."
        )
    )
    parser.add_argument("--device", help="Override detection.device.")
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        help="Override detection.confidence_threshold."
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config_from_args(args)
    if config is None:
        return

    if args.model_path:
        config["paths"]["model_path"] = args.model_path
    if args.detector:
        config["detection"]["detector_name"] = args.detector
    if args.tracker:
        config["tracking"]["tracker_type"] = args.tracker
    if args.tracker_name:
        config["dataset"]["tracker_name"] = args.tracker_name
    if args.device:
        config["detection"]["device"] = args.device
    if args.confidence_threshold is not None:
        config["detection"]["confidence_threshold"] = args.confidence_threshold

    model_path = project_path(config["paths"]["model_path"])
    if not model_path.exists():
        print(f"[ERROR] Model not found: {model_path}", file=sys.stderr)
        sys.exit(1)

    print(f"[INFO] Tracker output name: {get_tracker_output_name(config)}")
    run(config, args.split or ["all"], args.sequence)


if __name__ == "__main__":
    main()
