#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model Downloader
================

Automatic model downloading and caching from HuggingFace Hub.
Models are downloaded on first use and cached locally.
"""

import os
import sys
import logging
import platform
import tempfile
import time
from pathlib import Path

import requests
from huggingface_hub import hf_hub_url
from model_registry import DEFAULT_HF_REPO as BUILTIN_HF_REPO, MODEL_FILENAMES

logger = logging.getLogger(__name__)

# Default HuggingFace repository for AdiFind models.
# Override with ADIFIND_HF_REPO environment variable.
DEFAULT_HF_REPO = os.environ.get("ADIFIND_HF_REPO", BUILTIN_HF_REPO)

# Model definitions: model name → checkpoint filename on HuggingFace
_MODELS = MODEL_FILENAMES


def _default_cache_dir() -> Path:
    """Return the platform-appropriate default cache directory."""
    if platform.system() == "Windows":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    else:
        base = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return base / "adifind" / "models"


def get_cache_dir() -> Path:
    """Return the model cache directory, creating it if needed."""
    cache_dir = Path(os.environ.get("ADIFIND_CACHE_DIR", str(_default_cache_dir())))
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def _fmt_bytes(n: float) -> str:
    """Format a byte count as a human-readable string."""
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n) < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def _download_with_progress(repo_id: str, filename: str, cache_dir: Path,
                            token: str = None, label: str = None) -> Path:
    """Download a file from HuggingFace Hub with a live progress display.

    Uses an IPython display handle in notebooks (updates in-place without
    clearing other cell output) and carriage returns in plain terminals.
    """
    # Detect notebook environment for clean single-line progress updates
    try:
        from IPython.display import display as ipy_display, HTML
        _in_notebook = True
        # Each download gets its own display handle — updates replace only
        # this widget, so logger messages and prior completions persist.
        _handle = ipy_display(HTML(""), display_id=True)
    except ImportError:
        _in_notebook = False
        _handle = None

    display_name = label or filename

    def _show(text, final=False):
        """Render a progress line, replacing only its own output in notebooks."""
        if _in_notebook and _handle is not None:
            html_text = text.replace("\n", "<br>")
            _handle.update(HTML(f"<pre>{html_text}</pre>"))
        elif final:
            print(f"\r{text}", flush=True)
        else:
            print(f"\r{text}", end="", flush=True)

    _show(f"⬇️  Downloading {display_name} ...")

    url = hf_hub_url(repo_id=repo_id, filename=filename)
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    dest = cache_dir / filename
    tmp_path = None

    try:
        with requests.get(url, headers=headers, stream=True, timeout=30) as resp:
            resp.raise_for_status()
            total = int(resp.headers.get("content-length", 0))

            fd, tmp_path = tempfile.mkstemp(dir=str(cache_dir), suffix=".part")
            os.close(fd)

            downloaded = 0
            start = time.time()
            last_update = 0.0

            with open(tmp_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=1024 * 1024):
                    f.write(chunk)
                    downloaded += len(chunk)
                    now = time.time()

                    # Throttle updates to every 0.5s to avoid output spam
                    if now - last_update < 0.5 and downloaded < total:
                        continue
                    last_update = now

                    elapsed = now - start
                    speed = downloaded / elapsed if elapsed > 0 else 0

                    if total:
                        pct = downloaded / total * 100
                        bar_len = 30
                        filled = int(bar_len * downloaded / total)
                        bar = "█" * filled + "░" * (bar_len - filled)
                        eta = (total - downloaded) / speed if speed > 0 else 0
                        _show(
                            f"⬇️  {display_name}\n"
                            f"  {bar} {pct:5.1f}%  "
                            f"{_fmt_bytes(downloaded)}/{_fmt_bytes(total)}  "
                            f"{_fmt_bytes(speed)}/s  "
                            f"ETA {eta:.0f}s"
                        )
                    else:
                        _show(
                            f"⬇️  {display_name}\n"
                            f"  {_fmt_bytes(downloaded)} downloaded  "
                            f"{_fmt_bytes(speed)}/s"
                        )

            # Final done message
            elapsed = time.time() - start
            _show(
                f"✅ {display_name}: {_fmt_bytes(downloaded)} in {elapsed:.1f}s "
                f"({_fmt_bytes(downloaded / elapsed if elapsed > 0 else 0)}/s)",
                final=True,
            )

        # Atomic rename on success
        Path(tmp_path).replace(dest)
        return dest

    except BaseException:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise


def ensure_model(model_name: str, checkpoint: str = None, repo_id: str = None, token: str = None) -> Path:
    """
    Ensure a model checkpoint is available locally, downloading from
    HuggingFace Hub if necessary.

    Supports private repositories. Authentication is resolved in order:
      1. ``token`` argument passed directly
      2. ``HF_TOKEN`` environment variable
      3. Token saved by ``huggingface-cli login``

    Args:
        model_name: One of "adipocyte", "tumor", "tissue".
        checkpoint: Override checkpoint filename. Defaults to the
                    standard filename for the given model.
        repo_id:    HuggingFace repo ID. Defaults to ADIFIND_HF_REPO
                    env var or the built-in default.
        token:      HuggingFace API token for private repos. Falls back
                    to HF_TOKEN env var or stored login token.

    Returns:
        Path to the local checkpoint file.

    Raises:
        ValueError: If model_name is unknown.
        OSError: If download fails and no cached copy exists.
    """
    if model_name not in MODEL_FILENAMES:
        raise ValueError(
            f"Unknown model '{model_name}'. Choose from: {list(MODEL_FILENAMES.keys())}"
        )

    checkpoint = checkpoint or MODEL_FILENAMES[model_name]
    repo_id = repo_id or DEFAULT_HF_REPO

    # Check if the file is already in the local cache (avoids re-download)
    cache_dir = get_cache_dir()
    local_path = cache_dir / checkpoint

    if local_path.exists():
        logger.info(f"📦 Using cached {model_name} model: {local_path}")
        return local_path

    # Download from AdiFind on HuggingFace Hub
    logger.info(f"\u2b07\ufe0f  Downloading {model_name} model from AdiFind on HuggingFace (https://huggingface.co/{repo_id})...")
    logger.info(f"   File: {checkpoint}")

    # Resolve authentication token (explicit > env var > stored login)
    hf_token = token or os.environ.get("HF_TOKEN")
    
    # Friendly display names for the progress output
    _LABELS = {"adipocyte": "Adipocyte model", "tissue": "Tissue guidance model", "tumor": "Tumor model"}

    try:
        downloaded = _download_with_progress(
            repo_id=repo_id,
            filename=checkpoint,
            cache_dir=cache_dir,
            token=hf_token,
            label=_LABELS.get(model_name, model_name),
        )
        logger.info(f"\u2705 {model_name.capitalize()} model downloaded to: {downloaded}")
        return downloaded
    except requests.HTTPError as e:
        status_code = getattr(getattr(e, "response", None), "status_code", None)
        if status_code in (401, 403):
            message = (
                f"Access to Hugging Face repo '{repo_id}' was denied while downloading "
                f"'{checkpoint}'. The model repo is currently private. Authenticate with "
                "HF_TOKEN or a prior huggingface-cli login, or provide the canonical local "
                f"checkpoint via ADIFIND_{model_name.upper()}_MODEL_DIR."
            )
            logger.error(message)
            raise PermissionError(message) from e
        raise
    except Exception as e:
        logger.error(f"❌ Failed to download {model_name} model: {e}")
        logger.error(f"   You can manually place the checkpoint at: {local_path}")
        logger.error(f"   Or set ADIFIND_{model_name.upper()}_MODEL_DIR environment variable.")
        raise


def ensure_all_models(repo_id: str = None) -> dict:
    """
    Download all models. Useful for pre-caching before offline use.

    Returns:
        Dict mapping model name to local Path.
    """
    results = {}
    for name in MODEL_FILENAMES:
        try:
            results[name] = ensure_model(name, repo_id=repo_id)
        except Exception as e:
            logger.warning(f"⚠️  Could not download {name} model: {e}")
    return results
