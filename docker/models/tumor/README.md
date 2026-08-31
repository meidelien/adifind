# Tumor Detection Model

## Optional File

Place the following file in this directory if you want to bundle the tumor model into a Docker image:

- **`adifind_tumor.pth`** - The canonical AdiFind tumor checkpoint

## Notes

- This model is **optional**.
- The file must keep the canonical name `adifind_tumor.pth`.
- Legacy checkpoint filenames are not supported.
- If you do not bundle the file, AdiFind will try to download it when tumor segmentation is requested. The current model repo is private, so no-token runs should bundle or mount the canonical file instead.

## After Adding the Model

Rebuild the image:

```bash
docker compose build adifind-gpu
docker compose build adifind-cpu
```
