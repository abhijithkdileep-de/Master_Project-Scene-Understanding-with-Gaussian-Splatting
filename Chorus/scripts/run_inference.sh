#!/bin/bash

set -u
shopt -s nullglob nocasematch

SCRIPT_DIR=$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)

INPUT_ROOT="/path/to/dir/with/multiple/input"
OUTPUT_DIR="/path/to/output"
CONFIG="chorus_3dgs"
CHECKPOINT="lang-dino-enc-pretrain-scan-ppv2-mp3d-mcmc" # local pth path or model name from HF
PCA_VIS=false
DISABLE_OUTLIER_FILTER=false

declare -A USED_OUTPUT_NAMES=()

usage() {
    cat <<'EOF'
Usage: bash scripts/run_inference.sh [options]

Options:
  --input-root PATH   Batch input root containing scene folders and/or .ply files.
  --output-dir PATH   Batch output root for per-scene outputs.
  --config VALUE      Inference config path or alias. Default: chorus_3dgs
  --checkpoint VALUE  Local checkpoint path or Hugging Face model name.
  --pca-vis           Run PCA visualization for each batch item.
  --disable-outlier-filter
                      Disable raw-Ply outlier pruning for each batch item.
  -h, --help          Show this help message.
EOF
}

require_value() {
    local flag_name="$1"
    local flag_value="${2:-}"
    if [[ -z "$flag_value" ]] || [[ "$flag_value" == --* ]]; then
        echo "Missing value for ${flag_name}" >&2
        usage >&2
        exit 1
    fi
}

is_supported_input() {
    local input_path="$1"
    [[ -d "$input_path" ]] || [[ -f "$input_path" && "$input_path" == *.ply ]]
}

base_output_name() {
    local input_path="$1"
    local base_name

    base_name=$(basename "$input_path")
    if [[ -f "$input_path" ]]; then
        base_name="${base_name%.*}"
    fi
    printf '%s\n' "$base_name"
}

resolve_output_name() {
    local input_path="$1"
    local base_name output_name suffix

    base_name=$(base_output_name "$input_path")
    output_name="$base_name"

    if [[ -n "${USED_OUTPUT_NAMES[$output_name]+x}" ]]; then
        if [[ -f "$input_path" ]]; then
            output_name="${base_name}_ply"
        fi
        suffix=2
        while [[ -n "${USED_OUTPUT_NAMES[$output_name]+x}" ]]; do
            if [[ -f "$input_path" ]]; then
                output_name="${base_name}_ply_${suffix}"
            else
                output_name="${base_name}_${suffix}"
            fi
            suffix=$((suffix + 1))
        done
    fi

    USED_OUTPUT_NAMES["$output_name"]=1
    printf '%s\n' "$output_name"
}

run_inference() {
    local input_path="$1"
    local output_dir="$2"
    local input_name
    local cmd

    input_name=$(basename "$input_path")
    echo "Processing: $input_name"
    echo "Input: $input_path"
    echo "Output: $output_dir"

    cmd=(
        python
        -m tools.lang_inference
        --config "$CONFIG"
        --checkpoint "$CHECKPOINT"
        --input-root "$input_path"
        --output-dir "$output_dir"
    )
    if [[ "$PCA_VIS" == true ]]; then
        cmd+=(--pca_vis)
    fi
    if [[ "$DISABLE_OUTLIER_FILTER" == true ]]; then
        cmd+=(--disable-outlier-filter)
    fi

    if "${cmd[@]}"; then
        echo "Finished: $input_name"
        echo "---"
        return 0
    fi

    echo "Failed: $input_name" >&2
    echo "---"
    return 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --input-root)
            require_value "$1" "${2:-}"
            INPUT_ROOT="$2"
            shift 2
            ;;
        --output-dir)
            require_value "$1" "${2:-}"
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --config)
            require_value "$1" "${2:-}"
            CONFIG="$2"
            shift 2
            ;;
        --checkpoint)
            require_value "$1" "${2:-}"
            CHECKPOINT="$2"
            shift 2
            ;;
        --pca-vis)
            PCA_VIS=true
            shift
            ;;
        --disable-outlier-filter)
            DISABLE_OUTLIER_FILTER=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

if [[ ! -d "$INPUT_ROOT" ]]; then
    echo "Input root directory not found: $INPUT_ROOT" >&2
    exit 1
fi

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
cd "$REPO_ROOT"

batch_inputs=()
skipped=0
for input_path in "$INPUT_ROOT"/*; do
    if is_supported_input "$input_path"; then
        batch_inputs+=("$input_path")
    else
        skipped=$((skipped + 1))
        echo "Skipping unsupported input: $input_path"
    fi
done

if [[ ${#batch_inputs[@]} -eq 0 ]]; then
    echo "No supported scene folders or .ply files found under: $INPUT_ROOT" >&2
    exit 1
fi

processed=0
succeeded=0
failed=0

for input_path in "${batch_inputs[@]}"; do
    output_name=$(resolve_output_name "$input_path")
    output_dir="${OUTPUT_DIR}/${output_name}"
    processed=$((processed + 1))

    if run_inference "$input_path" "$output_dir"; then
        succeeded=$((succeeded + 1))
    else
        failed=$((failed + 1))
    fi
done

echo "Batch summary: processed=${processed}, succeeded=${succeeded}, failed=${failed}, skipped=${skipped}"

if [[ $failed -gt 0 ]]; then
    exit 1
fi
