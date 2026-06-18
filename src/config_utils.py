import argparse
import json
from copy import deepcopy
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "data" / "config.json"


DEFAULT_CONFIG = {
    "paths": {
        "raw_train_annotations_dir": "data/raw/DETRAC-Train-Annotations-XML",
        "raw_test_annotations_dir": "data/raw/DETRAC-Test-Annotations-XML",
        "raw_images_dir": "data/raw/DETRAC-Images",
        "preprocessed_images_dir": "data/preprocessed",
        "split_manifest_path": "data/preprocessed/splits.json",
        "yolo_output_dir": "data/yolo",
        "yolo_data_yaml_path": "data/yolo/data.yaml",
        "trackeval_data_dir": "data/trackeval/data",
        "tracker_output_dir": "data/trackeval/data/trackers",
        "model_path": "weights/best.pt"
    },
    "dataset": {
        "benchmark_name": "ua-detrac",
        "tracker_name": None,
        "frame_rate": 25,
        "image_width": 960,
        "image_height": 540,
        "image_ext": ".jpg"
    },
    "preprocessing": {
        "ignore_mask_mode": "black",
        "gaussian_kernel_size": 51,
        "gaussian_sigma": 15
    },
    "split": {
        "train_ratio": 0.8,
        "random_seed": 42
    },
    "yolo": {
        "class_map": {
            "car": 0,
            "bus": 1,
            "van": 2,
            "others": 3
        },
        "default_class_name": "others"
    },
    "detection": {
        "detector_name": "yolo",
        "confidence_threshold": 0.2,
        "device": "auto",
        "verbose": False
    },
    "tracking": {
        "tracker_type": "byte",
        "lost_track_buffer": 75,
        "track_activation_threshold": 0.572,
        "minimum_consecutive_frames": 2,
        "minimum_iou_threshold": 0.1,
        "high_conf_detection_threshold": 0.572
    },
    "trackeval": {
        "metrics": ["HOTA", "CLEAR", "Identity"],
        "print_config": False
    }
}


def project_path(path_value):
    path = Path(path_value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def deep_merge(base, override):
    merged = deepcopy(base)
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(config_path=None):
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    if not path.is_absolute():
        path = PROJECT_ROOT / path

    if not path.exists() or path.stat().st_size == 0:
        return deepcopy(DEFAULT_CONFIG)

    with open(path, "r", encoding="utf-8") as file_in:
        user_config = json.load(file_in)

    return deep_merge(DEFAULT_CONFIG, user_config)


def write_default_config(config_path=None):
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file_out:
        json.dump(DEFAULT_CONFIG, file_out, indent=2)
        file_out.write("\n")


def parse_value(raw_value):
    try:
        return json.loads(raw_value)
    except json.JSONDecodeError:
        return raw_value


def set_nested_value(config, dotted_key, value):
    current = config
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        if part not in current or not isinstance(current[part], dict):
            current[part] = {}
        current = current[part]
    current[parts[-1]] = value


def apply_overrides(config, overrides):
    updated = deepcopy(config)
    for override in overrides or []:
        if "=" not in override:
            raise argparse.ArgumentTypeError(
                f"Override must use key=value format: {override}"
            )
        key, raw_value = override.split("=", 1)
        set_nested_value(updated, key, parse_value(raw_value))
    return updated


def add_common_config_args(parser):
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to JSON config file."
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override config values, e.g. --set split.random_seed=123."
    )
    parser.add_argument(
        "--write-default-config",
        action="store_true",
        help="Write the default config JSON and exit."
    )


def load_config_from_args(args):
    if args.write_default_config:
        write_default_config(args.config)
        return None
    return apply_overrides(load_config(args.config), args.set)


def get_tracker_output_name(config):
    tracker_name = config["dataset"].get("tracker_name")
    if tracker_name:
        return tracker_name

    detector_name = config["detection"]["detector_name"]
    tracker_type = config["tracking"]["tracker_type"]
    return f"{detector_name}-{tracker_type}"
