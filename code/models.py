#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Deep Learning Models Module
============================

Handles model configuration, initialization, and inference
for adipocyte and tumor detection in AdiFind WSI analysis.
"""

import logging
import torch
from torchvision.ops import nms

# Detectron2 imports
from detectron2.config import get_cfg
from detectron2.engine.defaults import DefaultPredictor
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
import detectron2.data.transforms as T
from detectron2.data import MetadataCatalog

# Import configuration
from config import config, paths
from model_registry import merge_detectron2_builtin_config, resolve_model_path


# ================================================================
# CUSTOM PREDICTOR
# ================================================================

class CustomBatchPredictor(DefaultPredictor):
    """
    Custom predictor that supports batch inference.
    """
    def __init__(self, cfg):
        super().__init__(cfg)
        self.device = torch.device(cfg.MODEL.DEVICE)
        self.model.to(self.device)
        self.cfg = cfg  # Store config for cloning

    def clone(self):
        """
        Create a copy of this predictor for use in multiprocessing.
        Returns a new instance with the same configuration.
        """
        return CustomBatchPredictor(self.cfg)

    def __call__(self, original_images):
        """
        Args:
            original_images (list[np.ndarray]): a list of images, each of shape (H, W, C) (in BGR order).
        Returns:
            predictions (list[dict]):
                the output of the model for each image.
        """
        with torch.no_grad():
            inputs = []
            for original_image in original_images:
                if self.input_format == "RGB":
                    original_image = original_image[:, :, ::-1]
                height, width = original_image.shape[:2]
                image = self.aug.get_transform(original_image).apply_image(original_image)
                image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1)).to(self.device)
                inputs.append({"image": image, "height": height, "width": width})

            predictions = self.model(inputs)
            return predictions


# ================================================================
# MODEL CONFIGURATION FUNCTIONS
# ================================================================

def configure_adipocyte_model(model_dir=None, model_checkpoint=None):
    """Configure and initialize the adipocyte detection model."""
    model_path, model_dir = resolve_model_path(
        "adipocyte",
        model_dir=model_dir or paths.ADIPOCYTE_MODEL_DIR,
        model_checkpoint=model_checkpoint or paths.ADIPOCYTE_MODEL_CHECKPOINT,
    )
    
    logging.debug("Configuring adipocyte detection model...")
    cfg = get_cfg()
    cfg.OUTPUT_DIR = model_dir
    merge_detectron2_builtin_config(cfg)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # Adipocytes
    cfg.TEST.DETECTIONS_PER_IMAGE = 100000  # Cap number of detections per tile
    cfg.MODEL.WEIGHTS = model_path
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = config.CONFIDENCE_THRESHOLD
    cfg.SOLVER.AMP.ENABLED = True  # Enable mixed precision training
    cfg.MODEL.DEVICE = 'cuda' if config.USE_GPU_INFERENCE else 'cpu'
    
    device_type = 'GPU' if config.USE_GPU_INFERENCE else 'CPU'
    logging.debug(f"? Adipocyte model configured successfully. Using {device_type}.")
    
    return CustomBatchPredictor(cfg)


def configure_tumor_model(model_dir=None, model_checkpoint=None):
    """Configure and initialize the tumor detection model."""
    model_path, model_dir = resolve_model_path(
        "tumor",
        model_dir=model_dir or paths.TUMOR_MODEL_DIR,
        model_checkpoint=model_checkpoint or paths.TUMOR_MODEL_CHECKPOINT,
    )
    
    logging.debug("Configuring tumor detection model...")
    cfg = get_cfg()
    cfg.OUTPUT_DIR = model_dir
    merge_detectron2_builtin_config(cfg)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # Tumor
    cfg.TEST.DETECTIONS_PER_IMAGE = 1000
    cfg.MODEL.WEIGHTS = model_path
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.2
    cfg.SOLVER.AMP.ENABLED = True
    cfg.MODEL.DEVICE = 'cuda' if config.USE_GPU_INFERENCE else 'cpu'
    
    device_type = 'GPU' if config.USE_GPU_INFERENCE else 'CPU'
    logging.debug(f"? Tumor model configured successfully. Using {device_type}.")
    
    return CustomBatchPredictor(cfg)


def configure_tissue_model(model_dir=None, model_checkpoint=None):
    """Configure and initialize the tissue guidance model."""
    logging.debug("Configuring tissue guidance model...")

    model_path, model_dir = resolve_model_path(
        "tissue",
        model_dir=model_dir or paths.TISSUE_MODEL_DIR,
        model_checkpoint=model_checkpoint or paths.TISSUE_MODEL_CHECKPOINT,
    )
    logging.debug(f"   Using tissue model from: {model_dir}")

    cfg = get_cfg()
    cfg.OUTPUT_DIR = model_dir
    merge_detectron2_builtin_config(cfg)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # Tissue
    cfg.TEST.DETECTIONS_PER_IMAGE = 1000

    cfg.MODEL.WEIGHTS = model_path
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = config.TISSUE_CONFIDENCE_THRESHOLD
    cfg.SOLVER.AMP.ENABLED = True
    cfg.MODEL.DEVICE = 'cuda' if config.USE_GPU_INFERENCE else 'cpu'

    device_type = 'GPU' if config.USE_GPU_INFERENCE else 'CPU'
    logging.debug(f"? Tissue guidance model configured successfully. Using {device_type}.")

    return CustomBatchPredictor(cfg)


# ================================================================
# LEGACY FUNCTION NAMES (for backward compatibility)
# ================================================================

# Keep the original function name for backward compatibility
def configure_model(model_dir):
    """Legacy function name - use configure_adipocyte_model instead."""
    return configure_adipocyte_model(model_dir)


def configure_tumour_model(model_dir):
    """Legacy function name - use configure_tumor_model instead."""
    return configure_tumor_model(model_dir)


# ================================================================
# EXPORTS
# ================================================================

__all__ = [
    'CustomBatchPredictor',
    'configure_adipocyte_model',
    'configure_tumor_model', 
    'configure_tissue_model',
    'configure_model',  # Legacy
    'configure_tumour_model'  # Legacy
]
