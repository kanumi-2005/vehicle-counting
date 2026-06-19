#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

CONFIG_PATH="$PROJECT_ROOT/data/config.json"
SPLIT="val"
BENCHMARK=""
DETECTOR=""
TRACKER=""
TRACKER_NAME=""
METRICS=""
GT_DIR=""
TRACKER_DIR=""
PRINT_CONFIG=""
TRACKEVAL_SCRIPT="$PROJECT_ROOT/third_party/TrackEval/scripts/run_mot_challenge.py"

usage() {
    echo "Usage: $0 [OPTIONS]"
    echo "Options:"
    echo "  --config PATH              Path to JSON config file"
    echo "  --split NAME               Split to evaluate (default: val)"
    echo "  --benchmark NAME           Override dataset.benchmark_name"
    echo "  --detector NAME            Override detection.detector_name"
    echo "  --tracker [byte|sort]      Override tracking.tracker_type"
    echo "  --tracker-name NAME        Override TrackEval tracker folder name"
    echo "  --metrics \"A B C\"          Override trackeval.metrics"
    echo "  --gt-dir PATH              Override TrackEval gt folder"
    echo "  --tracker-dir PATH         Override TrackEval trackers folder"
    echo "  --trackeval-script PATH    Path to run_mot_challenge.py"
    echo "  --print-config BOOL        Override trackeval.print_config"
    echo "  -h, --help                 Show this help"
}

while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --config) CONFIG_PATH="$2"; shift ;;
        --split) SPLIT="$2"; shift ;;
        --benchmark) BENCHMARK="$2"; shift ;;
        --detector) DETECTOR="$2"; shift ;;
        --tracker) TRACKER="$2"; shift ;;
        --tracker-name|--tracker_name) TRACKER_NAME="$2"; shift ;;
        --metrics) METRICS="$2"; shift ;;
        --gt-dir|--gt_dir) GT_DIR="$2"; shift ;;
        --tracker-dir|--trk-dir|--trk_dir) TRACKER_DIR="$2"; shift ;;
        --trackeval-script) TRACKEVAL_SCRIPT="$2"; shift ;;
        --print-config|--print_config) PRINT_CONFIG="$2"; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "[ERROR] Invalid parameter: $1"; usage; exit 1 ;;
    esac
    shift
done

if [[ -n "$TRACKER" && "$TRACKER" != "byte" && "$TRACKER" != "sort" ]]; then
    echo "[ERROR] --tracker must be one of: byte, sort"
    exit 1
fi

eval "$(
    python "$SCRIPT_DIR/read_eval_config.py" \
        --config "$CONFIG_PATH" \
        --project-root "$PROJECT_ROOT" \
        --detector "$DETECTOR" \
        --tracker "$TRACKER" \
        --tracker-name "$TRACKER_NAME"
)"

BENCHMARK="${BENCHMARK:-$CONFIG_BENCHMARK}"
TRACKER_TO_EVAL="$CONFIG_TRACKER"
METRICS="${METRICS:-$CONFIG_METRICS}"
GT_DIR="${GT_DIR:-$CONFIG_GT_DIR}"
TRACKER_DIR="${TRACKER_DIR:-$CONFIG_TRACKER_DIR}"
PRINT_CONFIG="${PRINT_CONFIG:-$CONFIG_PRINT_CONFIG}"

echo "=========================================="
echo "STARTING EVALUATION"
echo "------------------------------------------"
echo "  Benchmark     : $BENCHMARK"
echo "  Detector      : ${DETECTOR:-from config}"
echo "  Tracker       : ${TRACKER:-from config}"
echo "  Tracker Name  : $TRACKER_TO_EVAL"
echo "  Split         : $SPLIT"
echo "  GT Folder     : $GT_DIR"
echo "  Tracker Folder: $TRACKER_DIR"
echo "  Metrics       : $METRICS"
echo "=========================================="

python "$TRACKEVAL_SCRIPT" \
    --BENCHMARK "$BENCHMARK" \
    --SPLIT_TO_EVAL "$SPLIT" \
    --TRACKERS_TO_EVAL "$TRACKER_TO_EVAL" \
    --GT_FOLDER "$GT_DIR" \
    --TRACKERS_FOLDER "$TRACKER_DIR" \
    --METRICS $METRICS \
    --PRINT_CONFIG "$PRINT_CONFIG"

echo
echo "[DONE] Evaluation finished."
