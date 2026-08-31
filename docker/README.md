# AdiFind Docker Deployment

This directory contains the Docker assets for running AdiFind in a containerized environment.

## Quick Start

### 1. Build an image

From the repository root:

```bash
docker build -t adifind:gpu -f docker/Dockerfile .
docker build -t adifind:cpu -f docker/Dockerfile.cpu .
```

Both Docker images intentionally use the pip-native `openslide-bin` path. This keeps the CPU and GPU containers on the same OpenSlide runtime strategy and avoids Conda or distro OpenSlide drift across images.

### 2. Optionally bundle canonical model files

If you want model weights baked into the image, place them under `docker/models/` before building:

```text
docker/models/
|- adipocyte/adifind_adipocyte.pth
|- tumor/adifind_tumor.pth
`- tissue/adifind_tissue_guidance.pth
```

Legacy checkpoint filenames are not supported. If you do not bundle the files, AdiFind will try to download the canonical `adifind_*` checkpoints on first use. The current Hugging Face model repo is private, so no-token container runs should use bundled or mounted canonical model files.

### 3. Run analysis

Single slide with GPU:

```bash
docker run --rm --gpus all \
    -v /path/to/slides:/data/input:ro \
    -v /path/to/output:/data/output \
    adifind:gpu /data/input/slide.svs \
    --output_dir /data/output --tissue_guidance
```

Single slide with CPU:

```bash
docker run --rm \
    -v /path/to/slides:/data/input:ro \
    -v /path/to/output:/data/output \
    adifind:cpu /data/input/slide.svs \
    --output_dir /data/output --tissue_guidance
```

Batch run from a directory:

```bash
docker run --rm --gpus all \
    -v /path/to/slides:/data/input:ro \
    -v /path/to/output:/data/output \
    adifind:gpu /data/input \
    --output_dir /data/output --tissue_guidance
```

Tumor-aware run:

```bash
docker run --rm --gpus all \
    -v /path/to/slides:/data/input:ro \
    -v /path/to/output:/data/output \
    adifind:gpu /data/input/K106942.svs \
    --output_dir /data/output \
    --tissue_guidance --tumor_segmentation --save_distance_map
```

## Docker Compose

Build:

```bash
cd docker
docker compose build adifind-gpu
docker compose build adifind-cpu
```

Run:

```bash
docker compose run --rm adifind-gpu /data/input/slide.svs --output_dir /data/output --tissue_guidance
docker compose run --rm adifind-cpu /data/input/slide.svs --output_dir /data/output --tissue_guidance
```

## Mounted Model Directories

You can mount a model tree instead of bundling checkpoints into the image:

```bash
docker run --rm --gpus all \
    -v /path/to/models:/app/models:ro \
    -v /path/to/slides:/data/input:ro \
    -v /path/to/output:/data/output \
    adifind:gpu /data/input/slide.svs \
    --output_dir /data/output --tissue_guidance
```

Mounted model directories must use the canonical layout:

```text
/path/to/models/
|- adipocyte/adifind_adipocyte.pth
|- tumor/adifind_tumor.pth
`- tissue/adifind_tissue_guidance.pth
```

If you set `ADIFIND_*_MODEL_DIR` explicitly, the referenced directory must contain the canonical filename for that model.

## Verification

Container sanity checks:

```bash
docker run --rm adifind:cpu --help
docker run --rm adifind:gpu --help
docker run --rm --entrypoint python adifind:cpu -c "import openslide; print(openslide.__library_version__)"
docker run --rm --entrypoint python -v /path/to/slides:/data/input:ro adifind:cpu -c "import openslide; slide = openslide.OpenSlide('/data/input/K106942.svs'); print(slide.dimensions); slide.close()"
docker run --rm --entrypoint python adifind:gpu -c "import openslide; print(openslide.__library_version__)"
docker run --rm --entrypoint python -v /path/to/slides:/data/input:ro adifind:gpu -c "import openslide; slide = openslide.OpenSlide('/data/input/K106942.svs'); print(slide.dimensions); slide.close()"
```

Dry-run example:

```bash
docker run --rm --gpus all \
    -v /path/to/slides:/data/input:ro \
    -v /path/to/output:/data/output \
    adifind:gpu /data/input \
    --output_dir /data/output --tissue_guidance --dry_run
```

## Troubleshooting

### Invalid model directory configuration

If the container reports an invalid model directory, ensure the directory contains only the canonical filenames:

- `adifind_adipocyte.pth`
- `adifind_tumor.pth`
- `adifind_tissue_guidance.pth`

### GPU not detected

- Ensure NVIDIA drivers are installed: `nvidia-smi`
- Ensure NVIDIA Container Toolkit is installed
- Use `--gpus all` for GPU runs

### No bundled model files present

That is supported only when the runtime has access to the model repo. Today that means Hugging Face authentication or a future public repo; otherwise mount or bundle the canonical checkpoint files.
