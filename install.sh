#!/usr/bin/env bash
# AdiFind Installation Script (Linux / macOS)
# Usage: bash install.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "========================================"
echo " AdiFind Installation"
echo "========================================"
echo ""

# Check for conda
if ! command -v conda > /dev/null 2>&1; then
    echo "ERROR: conda not found. Please install Miniconda or Anaconda first:"
    echo "  https://docs.anaconda.com/miniconda/"
    exit 1
fi

# Check for git (required by environment.yml for Detectron2)
if ! command -v git > /dev/null 2>&1; then
    echo "ERROR: git not found. Detectron2 is installed from a Git URL in environment.yml."
    echo "  Install git: https://git-scm.com/downloads"
    exit 1
fi

echo "Creating conda environment 'adifind'..."
if ! conda env create -f environment.yml; then
    echo ""
    echo "ERROR: Environment creation failed. See errors above."
    echo "If Detectron2 failed to install, confirm that git is available and retry."
    exit 1
fi

echo ""
echo "Running post-install validation from the repository root..."
if ! conda run -n adifind python -c "import torch, detectron2, openslide; print('PyTorch', torch.__version__, 'CUDA:', torch.cuda.is_available(), '| Detectron2', detectron2.__version__, '| OpenSlide', openslide.__library_version__)"; then
    echo ""
    echo "ERROR: Base import validation failed."
    echo "Check the OpenSlide runtime on your platform if the failure mentions openslide."
    exit 1
fi

if ! conda run -n adifind python -c "import openslide; slide = openslide.OpenSlide('example_data/K106942.svs'); print(slide.dimensions); slide.close()"; then
    echo ""
    echo "ERROR: Bundled slide readability validation failed."
    echo "Expected command:"
    echo "  conda run -n adifind python -c \"import openslide; slide = openslide.OpenSlide('example_data/K106942.svs'); print(slide.dimensions); slide.close()\""
    exit 1
fi

if ! conda run -n adifind adifind --help > /dev/null; then
    echo ""
    echo "ERROR: Installed CLI validation failed: adifind --help"
    exit 1
fi

if ! conda run -n adifind python code/main.py example_data/K106942.svs --tissue_guidance --dry_run; then
    echo ""
    echo "ERROR: Repo-root source-tree validation failed."
    echo "Expected command:"
    echo "  conda run -n adifind python code/main.py example_data/K106942.svs --tissue_guidance --dry_run"
    exit 1
fi

echo ""
echo "========================================"
echo " Installation and Validation Complete!"
echo "========================================"
echo ""
echo "To get started:"
echo "  conda activate adifind"
echo "  adifind --help"
echo ""
echo "Repo-root validation commands:"
echo "  conda run -n adifind adifind --help"
echo "  conda run -n adifind python code/main.py --help"
echo "  conda run -n adifind python -c \"import openslide; slide = openslide.OpenSlide('example_data/K106942.svs'); print(slide.dimensions); slide.close()\""
echo "  conda run -n adifind adifind example_data/K106942.svs --tissue_guidance --dry_run"
echo "  conda run -n adifind python code/main.py example_data/K106942.svs --tissue_guidance --dry_run"
echo ""
echo "Model downloads:"
echo "  Authenticated Hugging Face access downloads models automatically."
echo "  Otherwise, use local canonical checkpoint files:"
echo "    adifind_adipocyte.pth"
echo "    adifind_tumor.pth"
echo "    adifind_tissue_guidance.pth"
echo ""
