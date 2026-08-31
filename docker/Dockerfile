# =============================================================================
# AdiFind Docker Image - GPU Version
# =============================================================================
# Automated Adipocyte Detection for Whole Slide Images
#
# Build:
#   docker build -t adifind:gpu -f docker/Dockerfile .
#
# Run:
#   docker run --gpus all -v /path/to/images:/data/input -v /path/to/output:/data/output \
#       adifind:gpu /data/input/slide.svs --output_dir /data/output
# =============================================================================

FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

LABEL maintainer="Martin Eide Lien"
LABEL description="AdiFind - Automated Adipocyte Detection for Whole Slide Images"
LABEL version="1.0"

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Set Python to run unbuffered (better for Docker logs)
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# =============================================================================
# SYSTEM DEPENDENCIES
# =============================================================================

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# =============================================================================
# PYTHON DEPENDENCIES
# =============================================================================

WORKDIR /app

# Install runtime-specific dependencies first.
RUN pip install --no-cache-dir 'numpy<2'
RUN pip install --no-cache-dir 'setuptools<82'
RUN pip install --no-cache-dir openslide-bin
RUN pip install --no-cache-dir 'git+https://github.com/facebookresearch/detectron2.git'
RUN pip install --no-cache-dir cupy-cuda12x

# Copy project metadata and source for editable install.
COPY pyproject.toml README.md /app/
COPY code/ /app/code/
RUN pip install --no-cache-dir -e .

# =============================================================================
# MODEL FILES
# =============================================================================
# Canonical AdiFind model directories are bundled when present.
# Structure:
#   /app/models/adipocyte/adifind_adipocyte.pth
#   /app/models/tumor/adifind_tumor.pth
#   /app/models/tissue/adifind_tissue_guidance.pth

COPY docker/models/ /app/models/

# =============================================================================
# ENVIRONMENT CONFIGURATION
# =============================================================================

ENV ADIFIND_USE_GPU=true
ENV PYTHONPATH=/app/code

# =============================================================================
# ENTRYPOINT
# =============================================================================

COPY docker/entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

RUN mkdir -p /data/input /data/output

WORKDIR /app/code

ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["--help"]
