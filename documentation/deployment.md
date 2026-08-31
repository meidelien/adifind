Back to [Documentation Index](index.md)

# Deployment

Use this page for the canonical deployment guide. It covers containerized local runs and shared-cluster execution, plus the shared model and cache concerns that matter in both environments.

---

## Choose a Deployment Path

| Path | Best for | Start here |
|:-----|:---------|:-----------|
| Docker | Reproducible local or VM-based runs | [Docker Deployment](#docker-deployment) |
| Apptainer / Singularity | Shared academic HPC clusters | [HPC Deployment](#hpc-deployment) |
| Local workstation install | Direct Conda or manual setup | [INSTALL.md](INSTALL.md) |

---

## Docker Deployment

AdiFind provides GPU and CPU Docker images for reproducible, containerized execution.

The bundled GPU image targets the CUDA/NVIDIA runtime path. GPU inference may also be possible in custom ROCm-capable environments, but that backend is not packaged in these container examples.
Both Docker images use pip-installed `openslide-bin` so the CPU and GPU containers share the same OpenSlide runtime path while avoiding Conda or distro OpenSlide drift inside the container images.

### Prerequisites

| Requirement | GPU image | CPU image |
|:------------|:----------|:----------|
| Docker | Required | Required |
| CUDA/NVIDIA-compatible container runtime | Required for the provided GPU image | Not needed |
| NVIDIA Container Toolkit | Required for the provided GPU image | Not needed |

### Build images

From the repository root:

```bash
docker build -t adifind:gpu -f docker/Dockerfile .
docker build -t adifind:cpu -f docker/Dockerfile.cpu .
```

### Run a single slide

GPU:

```bash
docker run --rm --gpus all \
    -v /path/to/slides:/data/input:ro \
    -v /path/to/output:/data/output \
    adifind:gpu /data/input/slide.svs \
    --output_dir /data/output --tissue_guidance
```

CPU:

```bash
docker run --rm \
    -v /path/to/slides:/data/input:ro \
    -v /path/to/output:/data/output \
    adifind:cpu /data/input/slide.svs \
    --output_dir /data/output --tissue_guidance
```

### Batch processing

```bash
docker run --rm --gpus all \
    -v /path/to/slides:/data/input:ro \
    -v /path/to/output:/data/output \
    adifind:gpu /data/input \
    --output_dir /data/output --tissue_guidance
```

### Tumor-aware processing

```bash
docker run --rm --gpus all \
    -v /path/to/slides:/data/input:ro \
    -v /path/to/output:/data/output \
    adifind:gpu /data/input/K106942.svs \
    --output_dir /data/output \
    --tissue_guidance --tumor_segmentation --save_distance_map
```

### Docker Compose

```bash
cd docker
docker compose build adifind-gpu
docker compose build adifind-cpu
docker compose run --rm adifind-gpu /data/input/slide.svs --output_dir /data/output --tissue_guidance
docker compose run --rm adifind-cpu /data/input/slide.svs --output_dir /data/output --tissue_guidance
```

### Container model paths

The images auto-use canonical bundled model files when they exist under `/app/models`. If no bundled or mounted model files are present, AdiFind will attempt Hugging Face download. The current model repo is private, so unauthenticated container runs should use bundled or mounted canonical model files.

Bundled or mounted model layout:

```text
docker/models/
|- adipocyte/adifind_adipocyte.pth
|- tumor/adifind_tumor.pth
`- tissue/adifind_tissue_guidance.pth
```

Mounted models:

```bash
docker run --rm --gpus all \
    -v /path/to/models:/app/models:ro \
    -v /path/to/slides:/data/input:ro \
    -v /path/to/output:/data/output \
    adifind:gpu /data/input/slide.svs \
    --output_dir /data/output --tissue_guidance
```

If you set `ADIFIND_ADIPOCYTE_MODEL_DIR`, `ADIFIND_TUMOR_MODEL_DIR`, or `ADIFIND_TISSUE_MODEL_DIR` explicitly, the referenced directory must contain the canonical filename for that model. Legacy checkpoint filenames are not supported.

### Verification

```bash
docker run --rm adifind:cpu --help
docker run --rm adifind:gpu --help
docker run --rm --entrypoint python adifind:cpu -c "import openslide; print(openslide.__library_version__)"
docker run --rm --entrypoint python -v /path/to/slides:/data/input:ro adifind:cpu -c "import openslide; slide = openslide.OpenSlide('/data/input/K106942.svs'); print(slide.dimensions); slide.close()"
docker run --rm --entrypoint python adifind:gpu -c "import openslide; print(openslide.__library_version__)"
docker run --rm --entrypoint python -v /path/to/slides:/data/input:ro adifind:gpu -c "import openslide; slide = openslide.OpenSlide('/data/input/K106942.svs'); print(slide.dimensions); slide.close()"
docker run --rm --gpus all \
    -v /path/to/slides:/data/input:ro \
    -v /path/to/output:/data/output \
    adifind:gpu /data/input --output_dir /data/output --tissue_guidance --dry_run
```

---

## HPC Deployment

Use Apptainer or Singularity when Docker is not permitted on shared infrastructure.

### Build the image

From the definition file:

```bash
apptainer build adifind.sif docker/adifind.def
```

The Apptainer definition follows the same pip-native OpenSlide path as the Docker images and intentionally avoids Conda OpenSlide packages inside the `pytorch/pytorch` base image.

### Run a single slide

GPU:

```bash
apptainer run --nv \
    --bind /path/to/slides:/data \
    --bind /path/to/output:/output \
    adifind.sif /data/slide.svs \
    --output_dir /output --tissue_guidance
```

CPU:

```bash
apptainer run \
    --bind /path/to/slides:/data \
    --bind /path/to/output:/output \
    adifind.sif /data/slide.svs \
    --output_dir /output --tissue_guidance --disable_gpu_accel
```

### Batch processing

```bash
apptainer run --nv \
    --bind /path/to/slides:/data \
    --bind /path/to/output:/output \
    adifind.sif /data \
    --output_dir /output --tissue_guidance
```

### Shared model cache for offline nodes

Pre-cache models on a node with internet access and Hugging Face access:

```bash
export ADIFIND_CACHE_DIR=/shared/project/adifind_models
apptainer exec adifind.sif python -c "from model_downloader import ensure_all_models; ensure_all_models()"
```

Then bind the same cache on offline compute nodes:

```bash
export ADIFIND_CACHE_DIR=/shared/project/adifind_models

apptainer run --nv \
    --bind /shared/project/adifind_models:/shared/project/adifind_models \
    --bind /path/to/slides:/data \
    --bind /path/to/output:/output \
    adifind.sif /data/slide.svs \
    --output_dir /output --tissue_guidance
```

If you set `ADIFIND_*_MODEL_DIR` explicitly inside Apptainer, the referenced directory must contain the canonical `adifind_*` filename for that model. Legacy checkpoint filenames are not supported.

### Example SLURM job

```bash
#!/bin/bash
#SBATCH --job-name=adifind
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=adifind_%j.log

module load apptainer
export ADIFIND_CACHE_DIR=/shared/project/adifind_models

apptainer run --nv \
    --bind /shared/project/adifind_models:/shared/project/adifind_models \
    --bind /scratch/slides:/data:ro \
    --bind /scratch/output:/output \
    /shared/containers/adifind.sif \
    /data/slide.svs \
    --output_dir /output \
    --tissue_guidance \
    --tumor_segmentation
```

### Verification

```bash
apptainer run adifind.sif --help
apptainer run --nv \
    --bind /path/to/slides:/data \
    --bind /path/to/output:/output \
    adifind.sif /data --output_dir /output --tissue_guidance --dry_run
```

### Resource guidance

| Slide size | GPU VRAM | RAM | CPUs |
|:-----------|:---------|:----|:-----|
| Small (< 1 GB) | 8 GB | 32 GB | 4 |
| Medium (1-5 GB) | 16 GB | 64 GB | 8 |
| Large (5-20 GB) | 24 GB | 128 GB | 16 |
| Very large (> 20 GB) | 24+ GB | 256 GB | 16 |

Use `--tissue_guidance` for nearly every run. On constrained nodes, combine it with `--low_memory` and `--memmap_mask`.

---

## Models, Cache, and Environment

- Model directories and cache paths are documented in [Configuration](configuration.md).
- The current Hugging Face model repo is private. Until it is public, use local canonical model files or authenticated Hugging Face access for first-time downloads.
- For Docker and HPC, prefer explicit model directories or a shared `ADIFIND_CACHE_DIR` so repeated jobs do not re-download assets.
- Canonical checkpoint filenames are:
  - `adifind_adipocyte.pth`
  - `adifind_tumor.pth`
  - `adifind_tissue_guidance.pth`

---

## Troubleshooting Pointers

| Problem | Where to look |
|:--------|:--------------|
| GPU not visible in container | [Troubleshooting](troubleshooting.md) and your container runtime configuration |
| Model download or cache issues | [Configuration](configuration.md) |
| Slow cluster performance | [Performance Tuning](performance-tuning.md) |
| Resume and retry behavior for large runs | [CLI Workflows](cli-workflows.md#batch-workflows) |

---

## Next Steps

- [Installation Guide](INSTALL.md) for non-container installation
- [CLI Workflows](cli-workflows.md) for common run patterns
- [Configuration](configuration.md) for model paths, cache locations, and runtime variables
- [Performance Tuning](performance-tuning.md) for optimization guidance

---

Back to [Documentation Index](index.md)
