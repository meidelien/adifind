#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared tumour instance color helpers."""

import numpy as np

from config import config


DEFAULT_TUMOR_INSTANCE_COLORS_RGB = (
    (230, 57, 70),
    (29, 185, 84),
    (0, 118, 182),
    (255, 183, 3),
    (131, 56, 236),
    (251, 133, 0),
    (42, 157, 143),
    (255, 0, 110),
    (88, 129, 87),
    (67, 97, 238),
)


def get_tumor_instance_alpha():
    """Return the configured tumour overlay alpha clamped to [0, 1]."""
    try:
        alpha = float(getattr(config, "TUMOR_INSTANCE_ALPHA", 0.35))
    except (TypeError, ValueError):
        alpha = 0.35
    if not np.isfinite(alpha):
        alpha = 0.35
    return max(0.0, min(1.0, alpha))


def get_tumor_instance_color(tumor_id):
    """Return an RGB color for a one-based tumour instance id."""
    palette = getattr(config, "TUMOR_INSTANCE_COLORS_RGB", DEFAULT_TUMOR_INSTANCE_COLORS_RGB)
    if not palette:
        palette = DEFAULT_TUMOR_INSTANCE_COLORS_RGB
    try:
        label = max(1, int(tumor_id))
    except (TypeError, ValueError):
        label = 1
    color = palette[(label - 1) % len(palette)]
    if len(color) < 3:
        color = DEFAULT_TUMOR_INSTANCE_COLORS_RGB[(label - 1) % len(DEFAULT_TUMOR_INSTANCE_COLORS_RGB)]
    return np.array(
        [max(0, min(255, int(channel))) for channel in color[:3]],
        dtype=np.uint8,
    )


def build_tumor_instance_overlay(label_mask):
    """Build an RGB overlay image from a labelled tumour mask."""
    labels = np.asarray(label_mask)
    overlay = np.zeros((*labels.shape[:2], 3), dtype=np.uint8)
    if labels.size == 0:
        return overlay

    for tumor_id in np.unique(labels):
        tumor_id = int(tumor_id)
        if tumor_id <= 0:
            continue
        overlay[labels == tumor_id] = get_tumor_instance_color(tumor_id)
    return overlay


__all__ = [
    "DEFAULT_TUMOR_INSTANCE_COLORS_RGB",
    "get_tumor_instance_alpha",
    "get_tumor_instance_color",
    "build_tumor_instance_overlay",
]
