#!/bin/bash

# ==========================================
# DEFAULT VALUES
# ==========================================
SPLIT="val"
BENCHMARK="ua-detrac"
TRACKER="my_tracker"
METRICS="HOTA CLEAR Identity"

# Default paths (update these to your actual base paths if desired)
GT_DIR="./datasets/trackeval/data/gt/"
TRK_DIR="./datasets/trackeval/data/trackers/"

# ==========================================
# USAGE FUNCTION
# ==========================================
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo "Options:"
    echo "  --split     [val|test]"
    echo "  --benchmark BENCHMARK_NAME (e.g., ua-detrac)"
    echo "  --tracker   TRACKER_NAME"
    echo "  --gt_dir    Path to ground truth folder"
    echo "  --trk_dir   Path to trackers folder"
    exit 1
}

# ==========================================
# PARSE COMMAND LINE ARGUMENTS
# ==========================================
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --split) SPLIT="$2"; shift ;;
        --benchmark) BENCHMARK="$2"; shift ;;
        --tracker) TRACKER="$2"; shift ;;
        --gt_dir) GT_DIR="$2"; shift ;;
        --trk_dir) TRK_DIR="$2"; shift ;;
        -h|--help) usage ;;
        *) echo "[ERROR] Invalid parameter: $1"; usage ;;
    esac
    shift
done

echo "=========================================="
echo "STARTING EVALUATION (CUSTOM DATASET)"
echo "------------------------------------------"
echo "  Benchmark : $BENCHMARK"
echo "  Tracker   : $TRACKER"
echo "  Split     : $SPLIT"
echo "  GT Folder : $GT_DIR"
echo "  Trk Folder: $TRK_DIR"
echo "=========================================="

# ==========================================
# CALL PYTHON SCRIPT
# ==========================================
python third_party/TrackEval/scripts/run_mot_challenge.py \
    --BENCHMARK "$BENCHMARK" \
    --SPLIT_TO_EVAL "$SPLIT" \
    --TRACKERS_TO_EVAL "$TRACKER" \
    --GT_FOLDER "$GT_DIR" \
    --TRACKERS_FOLDER "$TRK_DIR" \
    --METRICS $METRICS \
    --PRINT_CONFIG False

echo -e "\n[DONE] Evaluation finished!"
