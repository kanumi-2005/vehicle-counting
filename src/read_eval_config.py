import argparse
import shlex

from config_utils import get_tracker_output_name, load_config, project_path


def shell_assign(name, value):
    return f"{name}={shlex.quote(str(value))}"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Print shell assignments for TrackEval bash defaults."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--detector", default="")
    parser.add_argument("--tracker", default="")
    parser.add_argument("--tracker-name", default="")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    if args.detector:
        config["detection"]["detector_name"] = args.detector
    if args.tracker:
        config["tracking"]["tracker_type"] = args.tracker
    if args.tracker_name:
        config["dataset"]["tracker_name"] = args.tracker_name

    trackeval_data_dir = project_path(config["paths"]["trackeval_data_dir"])
    metrics = " ".join(config["trackeval"]["metrics"])
    print_config = str(config["trackeval"]["print_config"])

    assignments = {
        "CONFIG_BENCHMARK": config["dataset"]["benchmark_name"],
        "CONFIG_TRACKER": get_tracker_output_name(config),
        "CONFIG_METRICS": metrics,
        "CONFIG_GT_DIR": trackeval_data_dir / "gt",
        "CONFIG_TRACKER_DIR": project_path(config["paths"]["tracker_output_dir"]),
        "CONFIG_PRINT_CONFIG": print_config,
    }
    for name, value in assignments.items():
        print(shell_assign(name, value))


if __name__ == "__main__":
    main()
