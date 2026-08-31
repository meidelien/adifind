#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Image Processing and IO Module
===============================

Handles image reading, processing, and slide format compatibility
for AdiFind WSI analysis.
"""

import os
import warnings
import logging
import math
import xml.etree.ElementTree as ET
import numpy as np
import cv2
from PIL import Image, ImageOps
from PIL.Image import Resampling
import torch

# Detectron2 imports
from detectron2.engine import DefaultPredictor

# Import configuration
from config import config

# Suppress common warnings for cleaner output
warnings.filterwarnings("ignore", message="torch.meshgrid: in an upcoming release")
warnings.filterwarnings("ignore", category=FutureWarning, module="fvcore.common.checkpoint")

# OpenSlide configuration
# Override with OPENSLIDE_PATH env var. On conda installs, OpenSlide is on PATH automatically.
OPENSLIDE_PATH = os.environ.get("OPENSLIDE_PATH", r"C:\OpenSlide\bin")
if hasattr(os, "add_dll_directory"):
    # Windows (Python 3.8+): add DLL directory if it exists
    if os.path.isdir(OPENSLIDE_PATH):
        with os.add_dll_directory(OPENSLIDE_PATH):
            import openslide
            from openslide import PROPERTY_NAME_MPP_X, PROPERTY_NAME_MPP_Y, OpenSlide, OpenSlideError
    else:
        # OpenSlide may be on PATH (e.g., conda install) — try direct import
        import openslide
        from openslide import PROPERTY_NAME_MPP_X, PROPERTY_NAME_MPP_Y, OpenSlide, OpenSlideError
else:
    import openslide
    from openslide import PROPERTY_NAME_MPP_X, PROPERTY_NAME_MPP_Y, OpenSlide, OpenSlideError

# Additional WSI format support
try:
    import slideio
    SLIDEIO_AVAILABLE = True
except ImportError:
    print("??  SlideIO not available. Some WSI formats may not be supported.")
    SLIDEIO_AVAILABLE = False


# ================================================================
# IMAGE HANDLER CLASSES
# ================================================================

def is_digital_slide(file_path):
    """Check if file is a compatible digital slide format."""
    try:
        slide = OpenSlide(file_path)
        slide.close()
        return True
    except OpenSlideError:
        return False


class ImageHandler:
    """Unified image handler for various WSI formats and regular images."""
    
    def __init__(self, file_path, desired_level=0):
        self.file_path = file_path
        self.desired_level = desired_level
        self.is_slide = is_digital_slide(file_path)
        self.is_ome_tiff = file_path.lower().endswith(('.ome.tif', '.ome.tiff', '.tif', '.tiff'))
        
        if self.is_slide:
            self.slide = OpenSlide(file_path)
            self.width, self.height = self.slide.level_dimensions[desired_level]
        elif self.is_ome_tiff:
            # Use slideio to open OME-TIFF files with the GDAL driver
            self.slide = slideio.open_slide(file_path, "GDAL")
            self.scene = self.slide.get_scene(0)
            self.width = self.scene.rect[2]
            self.height = self.scene.rect[3]
        else:
            self.image = Image.open(file_path)
            self.width, self.height = self.image.size

    def read_region(self, location, level, size):
        """Read a region from the image."""
        x, y = location
        w, h = size
        
        if self.is_slide:
            return self.slide.read_region(location, level, size)
        elif self.is_ome_tiff:
            rect = (x, y, w, h)
            region = self.scene.read_block(rect)
            # Convert from BGR to RGB
            region = cv2.cvtColor(region, cv2.COLOR_BGR2RGB)
            return Image.fromarray(region)
        else:
            box = (x, y, x + w, y + h)
            return self.image.crop(box)

    def get_best_level_for_downsample(self, target_downsample):
        """Get the best pyramid level for a target downsample factor."""
        if hasattr(self, 'slide') and hasattr(self.slide, 'level_downsamples'):
            downsamples = self.slide.level_downsamples
            level_diffs = [abs(ds - target_downsample) for ds in downsamples]
            best_level = level_diffs.index(min(level_diffs))
            return best_level
        else:
            return 0  # Default to level 0 for non-slide images

    def close(self):
        """Close the image handler and free resources."""
        if self.is_slide:
            self.slide.close()
        elif hasattr(self, 'image') and isinstance(self.image, Image.Image):
            self.image.close()
        # slideio does not require explicit closing


# ================================================================
# IMAGE PROCESSING UTILITIES
# ================================================================

_MICRON_UNIT_FACTORS = {
    "\u00b5m": 1.0,
    "um": 1.0,
    "micron": 1.0,
    "microns": 1.0,
    "micrometer": 1.0,
    "micrometers": 1.0,
    "micrometre": 1.0,
    "micrometres": 1.0,
    "nm": 0.001,
    "nanometer": 0.001,
    "nanometers": 0.001,
    "nanometre": 0.001,
    "nanometres": 0.001,
    "mm": 1000.0,
    "millimeter": 1000.0,
    "millimeters": 1000.0,
    "millimetre": 1000.0,
    "millimetres": 1000.0,
    "cm": 10000.0,
    "centimeter": 10000.0,
    "centimeters": 10000.0,
    "centimetre": 10000.0,
    "centimetres": 10000.0,
    "m": 1000000.0,
    "meter": 1000000.0,
    "meters": 1000000.0,
    "metre": 1000000.0,
    "metres": 1000000.0,
}


def _normalize_ome_unit(unit):
    """Normalize OME length units for lookup."""
    if unit is None or str(unit).strip() == "":
        return "um"
    return str(unit).strip().casefold().replace("\u03bc", "\u00b5")


def _ome_unit_to_microns(unit):
    """Return the multiplier from an OME unit to micrometers."""
    return _MICRON_UNIT_FACTORS.get(_normalize_ome_unit(unit))


def _xml_local_name(tag):
    """Return an XML tag's local name without its namespace."""
    return tag.rsplit("}", 1)[-1] if "}" in tag else tag


def _average_valid_mpp(mpp_x, mpp_y, source):
    """Return the scalar MPP average when X/Y values are valid."""
    try:
        mpp_x_val = float(mpp_x)
        mpp_y_val = float(mpp_y)
    except (TypeError, ValueError):
        logging.warning("Invalid MPP values from %s (x=%r, y=%r)", source, mpp_x, mpp_y)
        return None

    if (
        not math.isfinite(mpp_x_val)
        or not math.isfinite(mpp_y_val)
        or mpp_x_val <= 0
        or mpp_y_val <= 0
    ):
        logging.warning("Invalid MPP values from %s (x=%r, y=%r)", source, mpp_x, mpp_y)
        return None

    return (mpp_x_val + mpp_y_val) / 2


def _extract_openslide_mpp(slide, source):
    """Extract scalar MPP from OpenSlide properties, if present and valid."""
    mpp_x = slide.properties.get(PROPERTY_NAME_MPP_X)
    mpp_y = slide.properties.get(PROPERTY_NAME_MPP_Y)
    if mpp_x is None or mpp_y is None:
        return None

    mpp = _average_valid_mpp(mpp_x, mpp_y, source)
    if mpp is not None:
        logging.debug("Extracted OpenSlide MPP from %s: %.4f um/pixel", source, mpp)
    return mpp


def _extract_mpp_from_ome_xml(ome_xml):
    """Extract scalar MPP from the first OME Pixels element with physical sizes."""
    if not ome_xml:
        return None

    try:
        root = ET.fromstring(ome_xml)
    except ET.ParseError as exc:
        logging.debug("Unable to parse OME-XML metadata: %s", exc)
        return None

    for elem in root.iter():
        if _xml_local_name(elem.tag) != "Pixels":
            continue

        physical_size_x = elem.attrib.get("PhysicalSizeX")
        physical_size_y = elem.attrib.get("PhysicalSizeY")
        if physical_size_x is None or physical_size_y is None:
            continue

        unit_x_factor = _ome_unit_to_microns(elem.attrib.get("PhysicalSizeXUnit"))
        unit_y_factor = _ome_unit_to_microns(elem.attrib.get("PhysicalSizeYUnit"))
        if unit_x_factor is None or unit_y_factor is None:
            logging.warning(
                "Unsupported OME pixel size units (x=%r, y=%r)",
                elem.attrib.get("PhysicalSizeXUnit"),
                elem.attrib.get("PhysicalSizeYUnit"),
            )
            return None

        try:
            mpp_x = float(physical_size_x) * unit_x_factor
            mpp_y = float(physical_size_y) * unit_y_factor
        except (TypeError, ValueError):
            logging.warning(
                "Invalid OME physical pixel sizes (x=%r, y=%r)",
                physical_size_x,
                physical_size_y,
            )
            return None
        return _average_valid_mpp(mpp_x, mpp_y, "OME-XML")

    return None


def _extract_ome_tiff_mpp(file_path):
    """Read OME-XML metadata from a TIFF header and extract MPP."""
    if not file_path:
        return None

    try:
        import tifffile
    except ImportError:
        logging.debug("tifffile is unavailable; cannot inspect OME-TIFF metadata")
        return None

    try:
        with tifffile.TiffFile(file_path) as tiff:
            ome_xml = tiff.ome_metadata
    except Exception as exc:
        logging.debug("Unable to read OME-TIFF metadata from %s: %s", file_path, exc)
        return None

    mpp = _extract_mpp_from_ome_xml(ome_xml)
    if mpp is not None:
        logging.debug("Extracted OME-TIFF MPP from %s: %.4f um/pixel", file_path, mpp)
    return mpp

def is_window_predominantly_black(window, threshold=0.95):
    """
    Check if the given window is predominantly black.

    Args:
        window (PIL.Image or np.ndarray): The image window to check.
        threshold (float): Proportion of black pixels to determine if window is predominantly black.

    Returns:
        bool: True if the window is predominantly black, False otherwise.
    """
    # Convert window to NumPy array if it's a PIL image
    if not isinstance(window, np.ndarray):
        window = np.array(window)

    # Convert to grayscale
    gray_window = cv2.cvtColor(window, cv2.COLOR_RGB2GRAY)

    # Count the number of black pixels
    black_pixel_count = np.sum(gray_window == 0)
    total_pixels = gray_window.size

    # Calculate the proportion of black pixels
    black_pixel_ratio = black_pixel_count / total_pixels

    # Determine if the window is predominantly black
    return black_pixel_ratio >= threshold


def get_mpp(svs_file):
    """
    Extract microns per pixel (MPP) from slide metadata.
    
    Args:
        svs_file: Path to the slide file, or an ImageHandler-like object
        
    Returns:
        float: Microns per pixel value
    """
    file_path = getattr(svs_file, "file_path", svs_file)
    mpp = None

    try:
        slide = None
        close_slide = False
        if hasattr(svs_file, "slide") and hasattr(svs_file.slide, "properties"):
            slide = svs_file.slide
        else:
            slide = OpenSlide(file_path)
            close_slide = True
        try:
            mpp = _extract_openslide_mpp(slide, file_path)
        finally:
            if close_slide:
                slide.close()
    except Exception as e:
        logging.debug("OpenSlide MPP extraction unavailable for %s: %s", file_path, e)

    if mpp is None:
        mpp = _extract_ome_tiff_mpp(file_path)

    if mpp is None:
        logging.warning("MPP metadata not found for %s. Using default: %s", file_path, config.DEFAULT_MPP)
        mpp = config.DEFAULT_MPP
    
    # Final safety check to ensure we never return zero or negative MPP
    if mpp <= 0:
        logging.error("Invalid MPP value (%s) detected. Using safe default: %s", mpp, config.DEFAULT_MPP)
        mpp = config.DEFAULT_MPP
    
    return mpp


def generate_sliding_windows(width, height, window_size, stride):
    """Generate sliding window coordinates for image processing."""
    window_width, window_height = window_size
    stride_x, stride_y = stride
    x_positions = list(range(0, width - window_width + 1, stride_x))
    y_positions = list(range(0, height - window_height + 1, stride_y))

    # Ensure the last window covers the end of the image
    if x_positions[-1] + window_width < width:
        x_positions.append(width - window_width)
    if y_positions[-1] + window_height < height:
        y_positions.append(height - window_height)

    for y in y_positions:
        for x in x_positions:
            yield x, y


def generate_tumour_thumbnail_for_inference(image_handler, thumbnail_size=(2000, 2000)):
    """
    Generate a 2000x2000 thumbnail for tumour inference using OpenSlide's get_thumbnail,
    matching the training logic. Falls back to PIL resize if not a digital slide.
    
    Args:
        image_handler: ImageHandler object for the WSI.
        thumbnail_size: tuple, size of the thumbnail (width, height).
        
    Returns:
        np.ndarray: Thumbnail RGB image for tumour inference.
    """
    if hasattr(image_handler, 'slide') and isinstance(image_handler.slide, OpenSlide):
        # Use OpenSlide's get_thumbnail for digital slides
        thumbnail_pil = image_handler.slide.get_thumbnail(thumbnail_size)
        thumbnail_pil = thumbnail_pil.convert('RGB')
    else:
        # Fallback: use PIL resize for non-slides
        full_img = image_handler.read_region((0, 0), 0, (image_handler.width, image_handler.height))
        thumbnail_pil = full_img.copy()
        thumbnail_pil.thumbnail(thumbnail_size, Resampling.BILINEAR)
    
    thumbnail_np = np.array(thumbnail_pil)[:, :, :3]
    return thumbnail_np


class OptimalImageReader:
    """Optimized image reading utilities for memory efficiency."""
    
    @staticmethod
    def read_optimal_image(image_handler, width, height, scaling_factor, desired_level):
        """
        ?? OPTIMIZED image reading helper function.
        
        Consolidates duplicate image reading logic used across 6+ functions.
        Automatically selects the best pyramid level to minimize I/O and memory usage.
        
        Args:
            image_handler: OpenSlide image handler
            width, height: Original image dimensions  
            scaling_factor: Target scaling factor
            desired_level: Fallback level for non-slide images
            
        Returns:
            numpy.ndarray: RGB image array with shape (target_height, target_width, 3)
        """
        from PIL import Image
        
        target_width = int(width * scaling_factor)
        target_height = int(height * scaling_factor)
        
        # Find the best level that's closest to our target resolution
        if hasattr(image_handler, 'slide') and hasattr(image_handler.slide, 'level_downsamples'):
            downsamples = image_handler.slide.level_downsamples
            level_factors = [1 / ds for ds in downsamples]
            level_diffs = [abs(factor - scaling_factor) for factor in level_factors]
            best_level = level_diffs.index(min(level_diffs))
            level_dims = image_handler.slide.level_dimensions[best_level]
            
            # Read at optimal level (reduces I/O significantly)
            full_image_pil = image_handler.read_region((0, 0), best_level, level_dims)
            
            # Only resize if necessary
            if level_dims != (target_width, target_height):
                full_image_pil = full_image_pil.resize((target_width, target_height), Image.Resampling.LANCZOS)
        else:
            # Fallback for non-slide images
            full_image_pil = image_handler.read_region((0, 0), desired_level, (width, height))
            full_image_pil = full_image_pil.resize((target_width, target_height), Image.Resampling.LANCZOS)
        
        return np.array(full_image_pil, dtype=np.uint8)[:, :, :3]


# ================================================================
# IMAGE STITCHING UTILITIES
# ================================================================

def save_full_postprocessed_image(windows, window_coords, wsi_shape, scaling_factor, output_path):
    """
    Stitch all post-processed window images into a single WSI-sized image, resize, and save.
    Applies the same pre-inference processing (inversion, Sobel, bilateral) as in inference_worker.
    
    Args:
        windows: List of np.ndarray, each is a raw window (H, W, 3 or 1)
        window_coords: List of (x, y) tuples, top-left pixel of each window in WSI
        wsi_shape: (height, width, channels) of the full WSI
        scaling_factor: float, resize output by this factor
        output_path: str, where to save the stitched image
    """
    import tifffile
    
    channels = windows[0].shape[2] if windows[0].ndim == 3 else 1
    stitched = np.zeros(wsi_shape, dtype=windows[0].dtype)

    for win, (x, y) in zip(windows, window_coords):
        win_proc = win.copy()
        if config.APPLY_IMAGE_INVERSION:
            win_proc = 255 - win_proc
        if config.APPLY_SOBEL_FILTER:
            win_cv = cv2.cvtColor(win_proc, cv2.COLOR_RGB2BGR)
            win_cv = win_cv.astype(np.float32)
            sobelx = cv2.Sobel(win_cv, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(win_cv, cv2.CV_64F, 0, 1, ksize=3)
            sobel_magnitude = np.sqrt(sobelx ** 2 + sobely ** 2)
            sobel_magnitude = np.uint8(np.clip(sobel_magnitude, 0, 255))
            if config.APPLY_BILATERAL_FILTER:
                win_cv = cv2.bilateralFilter(sobel_magnitude, 20, 20, 20)
            else:
                win_cv = sobel_magnitude
            win_proc = cv2.cvtColor(win_cv, cv2.COLOR_BGR2RGB)
        else:
            if config.APPLY_BILATERAL_FILTER:
                win_proc = cv2.bilateralFilter(win_proc, 20, 20, 20)
            win_proc = win_proc[:, :, :3] if win_proc.ndim == 3 else win_proc
        
        h, w = win_proc.shape[:2]
        if channels == 1:
            stitched[y:y+h, x:x+w] = win_proc
        else:
            stitched[y:y+h, x:x+w, ...] = win_proc

    # Resize if needed
    if scaling_factor != 1.0:
        new_size = (int(wsi_shape[1] * scaling_factor), int(wsi_shape[0] * scaling_factor))
        stitched = cv2.resize(stitched, new_size, interpolation=cv2.INTER_NEAREST)

    tifffile.imwrite(output_path, stitched, bigtiff=True)
    logging.info(f"Full post-processed image saved at {output_path}")


def save_full_unprocessed_image(windows, window_coords, wsi_shape, scaling_factor, output_path):
    """
    Stitch all original (unprocessed) window images into a single WSI-sized image, resize, and save.
    
    Args:
        windows: List of np.ndarray, each is a raw window (H, W, 3 or 1)
        window_coords: List of (x, y) tuples, top-left pixel of each window in WSI
        wsi_shape: (height, width, channels) of the full WSI
        scaling_factor: float, resize output by this factor
        output_path: str, where to save the stitched image
    """
    import tifffile
    
    channels = windows[0].shape[2] if windows[0].ndim == 3 else 1
    stitched = np.zeros(wsi_shape, dtype=windows[0].dtype)

    for win, (x, y) in zip(windows, window_coords):
        win_copy = win.copy()
        h, w = win_copy.shape[:2]
        if channels == 1:
            stitched[y:y+h, x:x+w] = win_copy
        else:
            stitched[y:y+h, x:x+w, ...] = win_copy

    # Resize if needed
    if scaling_factor != 1.0:
        new_size = (int(wsi_shape[1] * scaling_factor), int(wsi_shape[0] * scaling_factor))
        stitched = cv2.resize(stitched, new_size, interpolation=cv2.INTER_NEAREST)

    tifffile.imwrite(output_path, stitched, bigtiff=True)
    logging.info(f"Full unprocessed image saved at {output_path}")


# ================================================================
# DETECTRON2 PREDICTOR CLASSES
# ================================================================

class CustomBatchPredictor(DefaultPredictor):
    """
    Custom predictor that supports batch inference (from original BIBLE).
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


def calculate_optimal_window_params(mpp, args_window_size, args_stride):
    """
    Calculate optimal window size and stride based on MPP and command line arguments.
    
    Args:
        mpp: Microns per pixel value
        args_window_size: Window size from command line arguments  
        args_stride: Stride from command line arguments
        
    Returns:
        tuple: (final_window_size, final_stride)
    """
    import logging
    
    # Set window size and stride based on MPP (exactly as in original)
    if mpp >= 0.40:
        dynamic_window_size = (1100, 1100)
        dynamic_stride = (900, 900)
    else:
        dynamic_window_size = (2000, 2000) 
        dynamic_stride = (1700, 1700)
    
    # Override with command line args if provided
    final_window_size = tuple(args_window_size) if args_window_size != [2048, 2048] else dynamic_window_size
    final_stride = tuple(args_stride) if args_stride != [1024, 1024] else dynamic_stride
    
    logging.debug("?? Dynamic window size: %s", final_window_size)
    logging.debug("?? Dynamic stride: %s", final_stride)
    
    return final_window_size, final_stride


# Export commonly used classes and functions
__all__ = [
    'ImageHandler',
    'is_window_predominantly_black',
    'get_mpp',
    'generate_sliding_windows',
    'generate_tumour_thumbnail_for_inference',
    'OptimalImageReader',
    'save_full_postprocessed_image',
    'save_full_unprocessed_image',
    'CustomBatchPredictor',
    'calculate_optimal_window_params'
]
