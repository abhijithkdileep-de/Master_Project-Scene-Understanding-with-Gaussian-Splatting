#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)

TRACK=""
WITH_VIEWER=false
PYTHON_BIN="${PYTHON:-python}"

usage() {
    cat <<'EOF'
Usage: bash scripts/package/install_package_deps.sh --track {cu124|cu126} [options]

Options:
  --track VALUE      Required. Tested dependency track: cu124 or cu126.
  --viewer           Also install optional Mini Viewer runtime dependencies.
  --python PATH      Python executable to use. Defaults to $PYTHON or python.
  -h, --help         Show this help message.

Examples:
  bash scripts/package/install_package_deps.sh --track cu124
  bash scripts/package/install_package_deps.sh --track cu126 --viewer
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

while [[ $# -gt 0 ]]; do
    case "$1" in
        --track)
            require_value "$1" "${2:-}"
            TRACK="$2"
            shift 2
            ;;
        --viewer)
            WITH_VIEWER=true
            shift
            ;;
        --python)
            require_value "$1" "${2:-}"
            PYTHON_BIN="$2"
            shift 2
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

case "$TRACK" in
    cu124|cu126)
        REQUIREMENTS_FILE="${SCRIPT_DIR}/requirements-${TRACK}.txt"
        ;;
    "")
        echo "Missing required --track." >&2
        usage >&2
        exit 1
        ;;
    *)
        echo "Unsupported --track '${TRACK}'. Use cu124 or cu126." >&2
        exit 1
        ;;
esac

"${PYTHON_BIN}" - <<'PY'
import sys

if sys.version_info[:2] != (3, 10):
    raise SystemExit(
        "Chorus package-mode helper expects Python 3.10 because the pinned "
        f"FlashAttention wheels are cp310. Current Python: {sys.version.split()[0]}"
    )
PY

echo "Installing Chorus package-mode dependencies for track: ${TRACK}"
"${PYTHON_BIN}" -m pip install -r "${REQUIREMENTS_FILE}"

echo "Installing Chorus package in editable mode without dependency resolution."
"${PYTHON_BIN}" -m pip install -e "${REPO_ROOT}" --no-deps

if [[ "$WITH_VIEWER" == true ]]; then
    echo "Installing optional Mini Viewer dependencies."
    "${PYTHON_BIN}" -m pip install -r "${SCRIPT_DIR}/requirements-viewer.txt"
fi

cat <<EOF

Package-mode install finished.

Recommended verification:
  ${PYTHON_BIN} -c "import torch, spconv, torch_scatter, flash_attn, chorus; print(torch.__version__, chorus.__version__)"
  chorus-encode --help
EOF
