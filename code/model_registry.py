#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Canonical model filenames and resolution helpers for AdiFind."""

from importlib.util import find_spec
from pathlib import Path
from typing import Optional, Tuple


DEFAULT_HF_REPO = "letarg/adifind"
DETECTRON2_BASE_CONFIG = "COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"

MODEL_FILENAMES = {
    "adipocyte": "adifind_adipocyte.pth",
    "tumor": "adifind_tumor.pth",
    "tissue": "adifind_tissue_guidance.pth",
}

MODEL_LABELS = {
    "adipocyte": "Adipocyte",
    "tumor": "Tumor",
    "tissue": "Tissue guidance",
}


def get_model_filename(model_name: str) -> str:
    """Return the canonical checkpoint filename for a named model."""
    try:
        return MODEL_FILENAMES[model_name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown model '{model_name}'. Choose from: {list(MODEL_FILENAMES.keys())}"
        ) from exc


def validate_model_checkpoint_name(model_name: str, model_checkpoint: Optional[str] = None) -> str:
    """Reject non-canonical checkpoint filenames."""
    canonical = get_model_filename(model_name)
    if model_checkpoint in (None, canonical):
        return canonical

    label = MODEL_LABELS.get(model_name, model_name.capitalize())
    raise ValueError(
        f"{label} checkpoints must be named '{canonical}'. "
        f"Legacy or custom checkpoint filenames are not supported."
    )


def resolve_model_path(
    model_name: str,
    model_dir: Optional[str] = None,
    model_checkpoint: Optional[str] = None,
) -> Tuple[str, str]:
    """Resolve a model path, enforcing canonical filenames for local directories."""
    canonical = validate_model_checkpoint_name(model_name, model_checkpoint)

    if model_dir is None:
        from model_downloader import ensure_model

        model_path = Path(ensure_model(model_name, checkpoint=canonical))
        return str(model_path), str(model_path.parent)

    model_path = Path(model_dir) / canonical
    if not model_path.is_file():
        label = MODEL_LABELS.get(model_name, model_name.capitalize())
        raise FileNotFoundError(
            f"{label} model directory '{model_dir}' must contain '{canonical}'. "
            f"Legacy checkpoint filenames are not supported."
        )

    return str(model_path), str(model_path.parent)


def get_detectron2_builtin_config_file(config_name: str = DETECTRON2_BASE_CONFIG) -> str:
    """Return a Detectron2 bundled config path without importing detectron2.model_zoo."""
    spec = find_spec("detectron2")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("Detectron2 is installed, but its package path could not be resolved.")

    detectron2_root = Path(next(iter(spec.submodule_search_locations)))
    config_file = detectron2_root / "model_zoo" / "configs" / config_name
    if not config_file.is_file():
        raise RuntimeError(
            f"Detectron2 built-in config file not found: {config_file}. "
            "Reinstall Detectron2 or use a build that includes model_zoo configs."
        )

    return str(config_file)


def merge_detectron2_builtin_config(cfg, config_name: str = DETECTRON2_BASE_CONFIG) -> str:
    """Merge AdiFind's Detectron2 base config and return the resolved config path."""
    config_file = get_detectron2_builtin_config_file(config_name)
    cfg.merge_from_file(config_file)
    return config_file


__all__ = [
    "DEFAULT_HF_REPO",
    "DETECTRON2_BASE_CONFIG",
    "MODEL_FILENAMES",
    "MODEL_LABELS",
    "get_model_filename",
    "validate_model_checkpoint_name",
    "resolve_model_path",
    "get_detectron2_builtin_config_file",
    "merge_detectron2_builtin_config",
]
