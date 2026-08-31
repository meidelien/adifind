# Adipocyte Detection Model

## Required File

Place the following file in this directory if you want to bundle the adipocyte model into a Docker image:

- **`adifind_adipocyte.pth`** - The canonical AdiFind adipocyte checkpoint

## Notes

- This model is **required** for actual adipocyte inference.
- The file must keep the canonical name `adifind_adipocyte.pth`.
- Legacy checkpoint filenames are not supported.
- If you do not bundle the file, AdiFind will try to download it from Hugging Face on first use. The current model repo is private, so no-token runs should bundle or mount the canonical file instead.

## After Adding the Model

Rebuild the image:

```bash
docker compose build adifind-gpu
docker compose build adifind-cpu
```
