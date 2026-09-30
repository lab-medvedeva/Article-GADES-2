#!/bin/bash
# Usage: run_benchmark.sh generated-dense|generated-sparse|real|huge <datasets dir> <results dir>
#
#   generated-dense   <datasets dir>/<cells>_cells_<features>_features.csv
#   generated-sparse  <datasets dir>/<cells>_cells_<features>_features/<sparsity>.mtx
#   real              <datasets dir>/<dataset>.mtx
#   huge              <datasets dir>/<dataset>.h5ad and, for the baselines, <dataset>.mtx (h5ad_to_mtx.py)
#
# Needs in this directory: test.R, python_benchmarking.py and the built test_armadillo and test_arrayfire.
# GADES_ROOT points to the built GADES source tree; python is the environment with cuML, pandas and SciPy.
set -u

SUITE=$1
DATASETS_DIR=$(realpath "$2")
RESULTS_DIR=$(realpath -m "$3")
cd "$(dirname "$0")"

BATCH_SIZE=${BATCH_SIZE:-5000}
THREADS=${THREADS:-24}
export MEM_LIMIT_GB=${MEM_LIMIT_GB:-50}

ALL_METRICS="euclidean cosine pearson manhattan spearman kendall"
FOUR_METRICS="euclidean cosine pearson manhattan"
PER_CELL_PAIR_METRICS="kendall spearman manhattan"

GENERATED_CELLS="10 100 1000 10000"
GENERATED_FEATURES="10 100 1000 10000 100000"
GENERATED_SPARSITIES="0.5 0.75 0.9 0.95 0.99"
REAL_DATASETS="HLCA_aorta B_CD8T Camp TCells B_T Jester LastFM HLCA_lung PBMC_all CellLines HLCA_marrow HSC ModCloth Chen PBMC5K FibrocardRNA TaFeng Pinterest Anime BeerAdvocate RateBeer MovieLens20M"
HUGE_DATASETS="MouseAtlas TabulaMuris TabulaSapiensV1 FibrocardATAC AIDA NYTimes Netflix BookCrossing Pokec AmazonVideoGames"
FEATURES_BY_CELLS="HLCA_aorta B_CD8T Camp TCells B_T HLCA_lung PBMC_all CellLines HLCA_marrow HSC Chen PBMC5K FibrocardRNA"

contains() {
    local item=$1
    shift
    local candidate
    for candidate in "$@"; do
        if [[ "$candidate" == "$item" ]]; then
            return 0
        fi
    done
    return 1
}

repetitions() {
    local method=$1
    local metric=$2
    if [[ -n "${TIMES:-}" ]]; then
        echo "$TIMES"
        return
    fi
    case "$SUITE" in
        generated-dense)
            if [[ "$method" == "pandas" || "$method" == "pythonic" ]]; then
                echo 3
            elif [[ "$metric" == "kendall" ]]; then
                echo 5
            else
                echo 10
            fi
            ;;
        generated-sparse)
            echo 5
            ;;
        *)
            if [[ "$method" == "GPU" || "$method" == "raft" ]]; then
                echo 2
            else
                echo 1
            fi
            ;;
    esac
}

run_case() {
    local label=$1
    local name=$2
    shift 2
    local log="$RESULTS_DIR/logs/${label}_${name}.log"
    local start=$(date +%s)
    timeout --kill-after=30 "$TIME_LIMIT" "$@" > "$log" 2>&1
    local code=$?
    local elapsed=$(( $(date +%s) - start ))
    if [[ $code -eq 0 ]]; then
        echo "finished   ${elapsed}s  $label $name"
    elif [[ $code -eq 124 || $code -eq 137 ]]; then
        echo "time limit ${elapsed}s  $label $name"
    else
        echo "failed     ${elapsed}s  $label $name (exit code $code)"
    fi
}

run_r() {
    local label=$1
    local input=$2
    local output_dir=$3
    local method=$4
    local mode=$5
    local metric=$6
    local transpose=$7
    local sparse=FALSE
    local layout=default
    if [[ "$mode" == "sparse" ]]; then
        sparse=TRUE
        if contains "$metric" $PER_CELL_PAIR_METRICS; then
            layout=per_cell_pair
        fi
    fi
    local times=$(repetitions "$method" "$metric")
    run_case "$label" "${mode}_${method}_${metric}" \
        Rscript test.R "$input" "$method" "$times" "$metric" "$BATCH_SIZE" "$output_dir/$mode" "$sparse" FALSE "$transpose" "$layout"
}

run_armadillo() {
    local label=$1
    local input=$2
    local output_dir=$3
    local mode=$4
    local metric=$5
    local transpose=$6
    local sparse=FALSE
    if [[ "$mode" == "sparse" ]]; then
        sparse=TRUE
    fi
    local times=$(repetitions armadillo "$metric")
    run_case "$label" "${mode}_armadillo_${metric}" \
        ./test_armadillo "$input" armadillo "$times" "$metric" "$BATCH_SIZE" "$output_dir/$mode" "$sparse" "$transpose"
}

run_arrayfire() {
    local label=$1
    local input=$2
    local output_dir=$3
    local backend=$4
    local mode=$5
    local metric=$6
    local transpose=$7
    local sparse=FALSE
    if [[ "$mode" == "sparse" ]]; then
        sparse=TRUE
    fi
    local times=$(repetitions arrayfire "$metric")
    run_case "$label" "${mode}_arrayfire${backend}_${metric}" \
        ./test_arrayfire "$input" "$backend" "$times" "$metric" "$BATCH_SIZE" "$output_dir/$mode" "$sparse" "$transpose" TRUE
}

run_python() {
    local label=$1
    local input=$2
    local output_dir=$3
    local method=$4
    local mode=$5
    local metric=$6
    local transpose=$7
    local options=()
    if [[ "$mode" == "sparse" ]]; then
        options+=(--sparse)
    fi
    if [[ "$transpose" == "TRUE" ]]; then
        options+=(--no-transpose)
    fi
    local times=$(repetitions "$method" "$metric")
    run_case "$label" "${mode}_${method}_${metric}" \
        python python_benchmarking.py --num_threads "$THREADS" --metric "$metric" --input "$input" --times "$times" \
        --output "$output_dir/${mode}_${method}_${metric}.csv" --method "$method" "${options[@]}"
}

transpose_flag() {
    if contains "$1" $FEATURES_BY_CELLS; then
        echo FALSE
    else
        echo TRUE
    fi
}

run_generated_dense() {
    TIME_LIMIT=${TIME_LIMIT:-86400}
    local cells
    local features
    local metric
    local method
    for cells in $GENERATED_CELLS; do
        for features in $GENERATED_FEATURES; do
            if [[ $(( cells * features )) -gt 10000000 ]]; then
                continue
            fi
            local dataset="${cells}_cells_${features}_features"
            local input="$DATASETS_DIR/$dataset.csv"
            local output_dir="$RESULTS_DIR/$dataset"
            mkdir -p "$output_dir"
            for metric in $ALL_METRICS; do
                for method in GPU CPU amap factoextra; do
                    run_r "$dataset" "$input" "$output_dir" "$method" dense "$metric" TRUE
                done
                for method in pandas pythonic; do
                    run_python "$dataset" "$input" "$output_dir" "$method" dense "$metric" TRUE
                done
            done
            for metric in $FOUR_METRICS; do
                run_arrayfire "$dataset" "$input" "$output_dir" cuda dense "$metric" TRUE
                run_arrayfire "$dataset" "$input" "$output_dir" cpu dense "$metric" TRUE
                run_python "$dataset" "$input" "$output_dir" raft dense "$metric" TRUE
                run_armadillo "$dataset" "$input" "$output_dir" dense "$metric" TRUE
            done
        done
    done
}

run_generated_sparse() {
    TIME_LIMIT=${TIME_LIMIT:-86400}
    local directory
    local sparsity
    local metric
    for directory in "$DATASETS_DIR"/*_cells_*_features; do
        local dataset=$(basename "$directory")
        for sparsity in $GENERATED_SPARSITIES; do
            local input="$directory/$sparsity.mtx"
            local output_dir="$RESULTS_DIR/$dataset/$sparsity"
            local label="${dataset}_${sparsity}"
            mkdir -p "$output_dir"
            for metric in $ALL_METRICS; do
                run_r "$label" "$input" "$output_dir" GPU sparse "$metric" TRUE
                run_r "$label" "$input" "$output_dir" CPU sparse "$metric" TRUE
            done
            for metric in $FOUR_METRICS; do
                run_arrayfire "$label" "$input" "$output_dir" cuda sparse "$metric" TRUE
                run_python "$label" "$input" "$output_dir" raft sparse "$metric" TRUE
                run_armadillo "$label" "$input" "$output_dir" sparse "$metric" TRUE
            done
        done
    done
}

run_real() {
    TIME_LIMIT=${TIME_LIMIT:-86400}
    local dataset
    local mode
    local metric
    local method
    for dataset in $REAL_DATASETS; do
        local input="$DATASETS_DIR/$dataset.mtx"
        local output_dir="$RESULTS_DIR/$dataset"
        local transpose=$(transpose_flag "$dataset")
        mkdir -p "$output_dir"
        for mode in dense sparse; do
            for metric in $ALL_METRICS; do
                run_r "$dataset" "$input" "$output_dir" GPU "$mode" "$metric" "$transpose"
                run_r "$dataset" "$input" "$output_dir" CPU "$mode" "$metric" "$transpose"
            done
            for metric in $FOUR_METRICS; do
                run_arrayfire "$dataset" "$input" "$output_dir" cuda "$mode" "$metric" "$transpose"
                run_python "$dataset" "$input" "$output_dir" raft "$mode" "$metric" "$transpose"
                run_armadillo "$dataset" "$input" "$output_dir" "$mode" "$metric" "$transpose"
            done
        done
        for metric in $ALL_METRICS; do
            for method in amap factoextra; do
                run_r "$dataset" "$input" "$output_dir" "$method" dense "$metric" "$transpose"
            done
            for method in pandas pythonic; do
                run_python "$dataset" "$input" "$output_dir" "$method" dense "$metric" "$transpose"
            done
        done
    done
}

run_huge() {
    TIME_LIMIT=${TIME_LIMIT:-172800}
    local dataset
    local metric
    for dataset in $HUGE_DATASETS; do
        local h5ad="$DATASETS_DIR/$dataset.h5ad"
        local mtx="$DATASETS_DIR/$dataset.mtx"
        local output_dir="$RESULTS_DIR/$dataset"
        mkdir -p "$output_dir"
        for metric in $ALL_METRICS; do
            run_r "$dataset" "$h5ad" "$output_dir" GPU sparse "$metric" FALSE
            run_r "$dataset" "$h5ad" "$output_dir" GPU dense "$metric" FALSE
            run_r "$dataset" "$h5ad" "$output_dir" CPU sparse "$metric" FALSE
        done
        for metric in $FOUR_METRICS; do
            run_arrayfire "$dataset" "$mtx" "$output_dir" cuda sparse "$metric" TRUE
            run_python "$dataset" "$mtx" "$output_dir" raft sparse "$metric" TRUE
            run_armadillo "$dataset" "$mtx" "$output_dir" sparse "$metric" TRUE
        done
    done
}

mkdir -p "$RESULTS_DIR/logs"
case "$SUITE" in
    generated-dense)
        run_generated_dense
        ;;
    generated-sparse)
        run_generated_sparse
        ;;
    real)
        run_real
        ;;
    huge)
        run_huge
        ;;
    *)
        echo "unknown suite: $SUITE" >&2
        exit 1
        ;;
esac
