# AdiFind Docker Quick Start (Windows)

A short Windows guide for building the **GPU** and **CPU** Docker images with **Docker Desktop**, then testing the **GPU image** with the example data.

---

## Requirements

- Windows
- Docker Desktop installed and running
- The AdiFind model files already present in:

```text
%LOCALAPPDATA%\adifind\models
```

Expected files:

- `adifind_adipocyte.pth`
- `adifind_tumor.pth`
- `adifind_tissue_guidance.pth`

> For the GPU test command below, Docker Desktop GPU support must be available on your machine.

---

## 1) Create the Docker images

Run this in **PowerShell** from the repository root:

```powershell
$modelCache = Join-Path $env:LOCALAPPDATA 'adifind\models'

New-Item -ItemType Directory -Force .\docker\models | Out-Null

Copy-Item (Join-Path $modelCache 'adifind_adipocyte.pth') .\docker\models\
Copy-Item (Join-Path $modelCache 'adifind_tumor.pth') .\docker\models\
Copy-Item (Join-Path $modelCache 'adifind_tissue_guidance.pth') .\docker\models\

docker build -t adifind:gpu-bundled -f .\docker\Dockerfile .
docker build -t adifind:cpu-bundled -f .\docker\Dockerfile.cpu .
```

---

## 2) Confirm the images were created

```powershell
docker image ls | Select-String adifind
```

You should see both:

- `adifind:gpu-bundled`
- `adifind:cpu-bundled`

---

## 3) Test the GPU image with the example data

Run this in **PowerShell**:

```powershell
$slides = (Resolve-Path .\example_data).Path
$gpuOut = (New-Item -ItemType Directory -Force .\temp\docker-gpu-out).FullName

docker run --rm `
  --gpus all `
  --entrypoint adifind `
  -e ADIFIND_USE_GPU=true `
  -e ADIFIND_ADIPOCYTE_MODEL_DIR=/app/models `
  -e ADIFIND_TUMOR_MODEL_DIR=/app/models `
  -e ADIFIND_TISSUE_MODEL_DIR=/app/models `
  -v "${slides}:/data/input:ro" `
  -v "${gpuOut}:/data/output" `
  adifind:gpu-bundled `
  /data/input/K106942.svs `
  --output_dir /data/output `
  --tissue_guidance
```

### Notes

- This command uses `--entrypoint adifind` directly.
- It also sets the model directories explicitly to `/app/models`.
- That avoids the current bundled entrypoint/model-path mismatch in the image and runs the CLI directly.

---

## 4) Optional: quick GPU check

Before running the full example, you can check that the container can see the GPU:

```powershell
docker run --rm --gpus all --entrypoint nvidia-smi adifind:gpu-bundled
```

---

## 5) Output location

The GPU test output will be written to:

```text
.\temp\docker-gpu-out
```

---

## 6) Common issues

### `docker : The term 'docker' is not recognized`
Docker Desktop is either:

- not installed
- not running
- not on your PATH yet

### `--gpus all` fails
Docker Desktop GPU access is not available yet on the current machine.

### Model copy fails
Check that these files exist:

```text
%LOCALAPPDATA%\adifind\models\adifind_adipocyte.pth
%LOCALAPPDATA%\adifind\models\adifind_tumor.pth
%LOCALAPPDATA%\adifind\models\adifind_tissue_guidance.pth
```

---

## 7) One-line summary

- Build both images
- Confirm they exist
- Run the GPU image against `example_data\K106942.svs`
- Collect results from `temp\docker-gpu-out`
