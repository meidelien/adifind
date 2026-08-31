#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Regression tests for post-detection robustness paths."""

import logging

import numpy as np


class _ArrayTensor:
    def __init__(self, array):
        self._array = np.asarray(array)

    def cpu(self):
        return self

    def numpy(self):
        return self._array


class _TumorInstances:
    def __init__(self, masks, scores=None):
        self.pred_masks = _ArrayTensor(masks)
        if scores is not None:
            self.scores = _ArrayTensor(scores)

    def __len__(self):
        return len(self.pred_masks.numpy())


class _TumorPredictor:
    def __init__(self, masks, scores=None):
        self._instances = _TumorInstances(masks, scores)

    def __call__(self, _images):
        return [{"instances": self._instances}]


def test_tumor_segmentation_preserves_instance_labels(monkeypatch):
    import tumor_detection

    monkeypatch.setattr(tumor_detection.config, "MIN_TUMOR_AREA_PIXELS", 1)

    low_score_mask = np.zeros((6, 6), dtype=bool)
    low_score_mask[1:3, 1:3] = True
    high_score_mask = np.zeros((6, 6), dtype=bool)
    high_score_mask[3:5, 3:5] = True

    tumor_mask, _analysis_mask, _fullres_mask, num_tumors, tumor_areas, tumor_centroids = (
        tumor_detection.optimized_segment_tumour_on_thumbnail(
            np.zeros((6, 6, 3), dtype=np.uint8),
            _TumorPredictor(
                np.stack([low_score_mask, high_score_mask]),
                scores=np.array([0.2, 0.9], dtype=np.float32),
            ),
            full_shape=(12, 12),
        )
    )

    assert num_tumors == 2
    assert tumor_mask.dtype == np.uint16
    assert set(np.unique(tumor_mask)) == {0, 1, 2}
    assert np.all(tumor_mask[3:5, 3:5] == 1)
    assert np.all(tumor_mask[1:3, 1:3] == 2)
    assert tumor_areas == [4, 4]
    assert tumor_centroids == [(3.5, 3.5), (1.5, 1.5)]


def test_labelled_tumor_distance_metrics_return_closest_ids(monkeypatch):
    import tumor_detection

    monkeypatch.setattr(tumor_detection.config, "USE_CUPY", False)
    monkeypatch.setattr(tumor_detection.config, "MIN_TUMOR_PIXELS_FOR_DISTANCE", 1)

    full_mask = np.zeros((320, 320), dtype=np.uint32)
    full_mask[:160, :160] = 1
    full_mask[160:, 160:] = 2

    tumor_mask_analysis = np.array(
        [
            [1, 0],
            [0, 2],
        ],
        dtype=np.uint16,
    )

    distances, closest_ids = tumor_detection.optimized_compute_adipocyte_distance_metrics(
        mask_areas={1: 100, 2: 100},
        full_mask=full_mask,
        tumor_mask_analysis=tumor_mask_analysis,
        mpp=0.5,
        full_shape=(320, 320),
        analysis_downsample=160,
    )

    assert distances == {1: 0.0, 2: 0.0}
    assert closest_ids == {1: 1, 2: 2}


def test_tumor_distance_profiling_handles_zero_elapsed_time(monkeypatch, caplog):
    import tumor_detection

    monkeypatch.setattr(tumor_detection.config, "USE_CUPY", False)
    monkeypatch.setattr(tumor_detection.config, "MIN_TUMOR_PIXELS_FOR_DISTANCE", 1)
    monkeypatch.setattr(tumor_detection.time, "time", lambda: 100.0)

    full_mask = np.ones((160, 160), dtype=np.uint32)
    tumor_mask_analysis = np.ones((1, 1), dtype=np.uint8)

    with caplog.at_level(logging.INFO):
        distances = tumor_detection.optimized_compute_adipocyte_distances(
            mask_areas={1: 25600},
            full_mask=full_mask,
            tumor_mask_analysis=tumor_mask_analysis,
            mpp=0.5,
            full_shape=(160, 160),
            analysis_downsample=160,
        )

    assert distances == {1: 0.0}
    assert "Processing Efficiency" in caplog.text


def test_optional_tumor_distance_failure_returns_plain_warning(caplog):
    import main

    def failing_distance_func(**_kwargs):
        raise RuntimeError("synthetic distance failure")

    with caplog.at_level(logging.WARNING):
        distances, warning = main._compute_optional_tumor_distances(
            mask_areas={1: 10},
            full_mask=np.ones((2, 2), dtype=np.uint32),
            tumor_mask_analysis=np.ones((1, 1), dtype=np.uint8),
            mpp=0.5,
            full_shape=(2, 2),
            image_name="slide",
            distance_func=failing_distance_func,
            stats_func=lambda _distances: None,
        )

    assert distances == {}
    assert warning == main.TUMOR_DISTANCE_WARNING_MESSAGE
    assert any(record.exc_info for record in caplog.records)


def test_batch_completed_warning_is_kept_in_error_message():
    from batch_processing import BatchProcessor, BatchState, ImageProcessingResult

    image_result = ImageProcessingResult(
        image_name="slide",
        image_path="slide.svs",
        output_dir="out",
    )
    processor = object.__new__(BatchProcessor)
    processor.batch_state = BatchState(
        batch_id="batch",
        start_time="2026-01-01T00:00:00",
        base_output_dir="out",
        summary_dir="summary",
        total_images=1,
        image_results=[image_result],
    )
    processor._get_adipocyte_size_stats = lambda _output_dir, _image_name: (12.0, 15.0)
    processor._save_batch_state = lambda: None
    processor._update_summary_csv = lambda: None
    processor._log_progress = lambda _message: None
    processor._print_progress_update = lambda: None

    processor.mark_image_completed(
        "slide.svs",
        {
            "total_adipocytes": 3,
            "num_tumors": 1,
            "total_time": 2.5,
            "warning_message": "Adipocytes saved; tumour-distance values unavailable.",
        },
    )

    assert image_result.processing_status == "completed"
    assert image_result.total_adipocytes == 3
    assert image_result.error_message == "Adipocytes saved; tumour-distance values unavailable."


def test_finalize_mask_view_with_no_overlaps_preserves_mask_labels(monkeypatch):
    import core_processing

    monkeypatch.setattr(core_processing.config, "MEMORY_EFFICIENT_MODE", True)
    monkeypatch.setattr(core_processing.config, "MAX_FULL_MASK_PIXELS", 0)
    monkeypatch.setattr(core_processing.config, "ENABLE_GPU_LABEL_MAPPING", False)
    monkeypatch.setattr(
        core_processing.memory_manager,
        "acquire_label_mapping_lock",
        lambda timeout=600: "lock",
    )
    monkeypatch.setattr(
        core_processing.memory_manager,
        "release_label_mapping_lock",
        lambda _lock: None,
    )

    base_mask = np.zeros((2, 2), dtype=np.uint32)
    base_mask[0, 0] = 1
    mask_view = base_mask[:, :]

    full_mask, mask_areas, adipocyte_ids, *_rest = core_processing._finalize_processed_mask(
        full_mask=mask_view,
        parent={},
        adipocyte_properties={
            1: {
                "area": 1,
                "centroid_x": 0.0,
                "centroid_y": 0.0,
                "bbox": (0, 0, 1, 1),
            }
        },
        min_area_threshold_pixels=1,
        max_area_threshold_pixels=10,
        height=2,
        width=2,
        image_handler=None,
        async_enabled=False,
        gpu_profiler=None,
        mask_cleanup=lambda: None,
        postprocessed_windows=[],
        processed_window_coords=[],
        diagnostics={},
    )

    assert full_mask.shape == (2, 2)
    assert int(np.count_nonzero(full_mask == 1)) == 1
    assert mask_areas == {1: 1}
    assert adipocyte_ids == [1]
