#!/bin/bash
# =============================================================================
# AdiFind Docker Entrypoint Script
# =============================================================================
# This script validates the environment and runs AdiFind.
# =============================================================================

set -e

DEFAULT_MODELS_ROOT="/app/models"
MODELS_VALID=true

echo "=========================================="
echo "  AdiFind - Automated Adipocyte Detection"
echo "=========================================="
echo ""

if [ "${ADIFIND_USE_GPU}" != "false" ]; then
    if command -v nvidia-smi >/dev/null 2>&1; then
        echo "GPU detected:"
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
        echo ""
    else
        echo "WARNING: CUDA/NVIDIA runtime not detected for the bundled GPU container."
        echo "         Falling back to CPU-only mode for this container image."
        echo ""
        export ADIFIND_USE_GPU=false
    fi
else
    echo "Running in CPU-only mode."
    echo ""
fi

configure_model_dir() {
    local model_label="$1"
    local env_var="$2"
    local bundled_dir="$3"
    local checkpoint_name="$4"
    local configured_dir

    configured_dir="$(printenv "$env_var" || true)"
    if [ -n "$configured_dir" ]; then
        if [ ! -f "$configured_dir/$checkpoint_name" ]; then
            echo "  ERROR: $model_label model directory '$configured_dir' must contain '$checkpoint_name'."
            MODELS_VALID=false
            return
        fi
        echo "  OK: $model_label model directory set to $configured_dir/$checkpoint_name"
        return
    fi

    if [ -f "$bundled_dir/$checkpoint_name" ]; then
        export "$env_var=$bundled_dir"
        echo "  OK: Using bundled $model_label model at $bundled_dir/$checkpoint_name"
    else
        echo "  INFO: No bundled $model_label model found; AdiFind will attempt to download '$checkpoint_name' on demand."
    fi
}

echo "Checking model configuration..."
configure_model_dir "Adipocyte" "ADIFIND_ADIPOCYTE_MODEL_DIR" "$DEFAULT_MODELS_ROOT/adipocyte" "adifind_adipocyte.pth"
configure_model_dir "Tumor" "ADIFIND_TUMOR_MODEL_DIR" "$DEFAULT_MODELS_ROOT/tumor" "adifind_tumor.pth"
configure_model_dir "Tissue guidance" "ADIFIND_TISSUE_MODEL_DIR" "$DEFAULT_MODELS_ROOT/tissue" "adifind_tissue_guidance.pth"
echo ""

if [ "$MODELS_VALID" = false ]; then
    echo "=========================================="
    echo "CRITICAL: Invalid model directory configuration."
    echo ""
    echo "Supported bundled or mounted checkpoint names are:"
    echo "  - docker/models/adipocyte/adifind_adipocyte.pth"
    echo "  - docker/models/tumor/adifind_tumor.pth"
    echo "  - docker/models/tissue/adifind_tissue_guidance.pth"
    echo ""
    echo "Legacy checkpoint filenames are not supported."
    echo "=========================================="
    exit 1
fi

echo "Starting AdiFind..."
echo "Arguments: $@"
echo ""

exec adifind "$@"
