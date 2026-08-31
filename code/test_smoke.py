#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Smoke tests for AdiFind installation and basic functionality."""

import logging
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import requests


ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*m")
RUNTIME_CONFIG_FIELDS = (
    "ENABLE_TISSUE_GUIDANCE",
    "ENABLE_TUMOR_SEGMENTATION",
    "USE_GPU_INFERENCE",
    "USE_CUPY",
    "USE_GPU_PREPROCESSING",
    "ENABLE_GPU_LABEL_MAPPING",
    "ENABLE_QUPATH_EXPORT",
    "SAVE_QUPATH_GEOJSON",
    "SAVE_DISTANCE_COLORED_IMAGE",
    "SAVE_ANNOTATED_IMAGE",
    "SAVE_TUMOR_ZONE_OVERLAY_IMAGE",
    "TUMOR_INSTANCE_ALPHA",
    "TUMOR_INSTANCE_COLORS_RGB",
)


def _legacy_image_annotation_flag():
    return "--skip_" + "_".join(("annotated", "image"))


def _legacy_qupath_annotation_flag():
    return "--skip_" + "_".join(("qupath", "annotations"))


def _legacy_image_annotation_attr():
    return "_".join(("save", "annotated", "image"))


def _legacy_image_skip_attr():
    return "_".join(("skip", "annotated", "image"))


def _legacy_qupath_save_attr():
    return "_".join(("save", "qupath", "annotations"))


def _legacy_qupath_skip_attr():
    return "_".join(("skip", "qupath", "annotations"))


def _strip_ansi(text):
    return ANSI_ESCAPE_RE.sub("", text)


def _parse_args(monkeypatch, *extra_args):
    from argument_parser import parse_arguments

    monkeypatch.setattr(sys, "argv", ["main.py", "example_data/K106942.svs", *extra_args])
    return parse_arguments()


class EncodingConstrainedStream:
    """Text stream that raises when the configured encoding cannot represent input."""

    def __init__(self, encoding="cp1252", allow_reconfigure=False):
        self.encoding = encoding
        self.errors = "strict"
        self._allow_reconfigure = allow_reconfigure
        self._buffer = []

    def write(self, text):
        text.encode(self.encoding, self.errors)
        self._buffer.append(text)
        return len(text)

    def flush(self):
        return None

    def getvalue(self):
        return "".join(self._buffer)

    def reconfigure(self, encoding=None, errors=None):
        if not self._allow_reconfigure:
            raise OSError("reconfigure not supported")
        if encoding is not None:
            self.encoding = encoding
        if errors is not None:
            self.errors = errors


@pytest.fixture(autouse=True)
def restore_runtime_config():
    from config import config

    snapshot = {name: getattr(config, name) for name in RUNTIME_CONFIG_FIELDS}
    yield
    for name, value in snapshot.items():
        setattr(config, name, value)


@pytest.fixture
def restore_root_logger():
    root_logger = logging.getLogger()
    original_handlers = list(root_logger.handlers)
    original_level = root_logger.level
    yield root_logger
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)
        try:
            handler.close()
        except Exception:
            pass
    for handler in original_handlers:
        root_logger.addHandler(handler)
    root_logger.setLevel(original_level)


@pytest.fixture
def desktop_qt(monkeypatch, tmp_path):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6")
    from PySide6 import QtCore, QtWidgets

    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
        app.setQuitOnLastWindowClosed(False)

    QtCore.QSettings.setDefaultFormat(QtCore.QSettings.IniFormat)
    QtCore.QSettings.setPath(
        QtCore.QSettings.IniFormat,
        QtCore.QSettings.UserScope,
        str(tmp_path),
    )
    settings = QtCore.QSettings("adi", "adifind_desktop")
    settings.clear()
    settings.sync()
    yield app, QtCore
    settings = QtCore.QSettings("adi", "adifind_desktop")
    settings.clear()
    settings.sync()


# ================================================================
# IMPORT TESTS
# ================================================================


class TestImports:
    """Verify all required packages can be imported."""

    def test_torch_import(self):
        import torch

        assert hasattr(torch, "__version__")

    def test_detectron2_import(self):
        import detectron2

        assert hasattr(detectron2, "__version__")

    def test_detectron2_builtin_config_merge(self):
        from detectron2.config import get_cfg
        from model_registry import get_detectron2_builtin_config_file, merge_detectron2_builtin_config

        cfg = get_cfg()
        config_path = Path(merge_detectron2_builtin_config(cfg))
        direct_path = Path(get_detectron2_builtin_config_file())
        normalized = str(config_path).replace("\\", "/")

        assert config_path == direct_path
        assert config_path.is_file()
        assert "detectron2/model_zoo/configs" in normalized
        assert cfg.MODEL.META_ARCHITECTURE == "GeneralizedRCNN"
        assert cfg.MODEL.MASK_ON is True

    def test_openslide_import(self):
        import openslide

        assert hasattr(openslide, "__library_version__")

    def test_numpy_import(self):
        import numpy

        assert hasattr(numpy, "__version__")

    def test_scipy_import(self):
        import scipy

        assert hasattr(scipy, "__version__")

    def test_pandas_import(self):
        import pandas

        assert hasattr(pandas, "__version__")

    def test_cv2_import(self):
        import cv2

        assert hasattr(cv2, "__version__")

    def test_huggingface_hub_import(self):
        import huggingface_hub

        assert hasattr(huggingface_hub, "__version__")


# ================================================================
# ADIFIND MODULE TESTS
# ================================================================


class TestAdiFind:
    """Verify AdiFind modules load correctly."""

    def test_config_import(self):
        from config import Config, Paths, config, paths

        assert isinstance(config, Config)
        assert isinstance(paths, Paths)

    def test_config_defaults(self):
        from config import config

        assert config.CONFIDENCE_THRESHOLD > 0
        assert config.WINDOW_SIZE == (2000, 2000)
        assert config.USE_GPU_INFERENCE is True
        assert config.USE_CUPY is True
        assert config.USE_GPU_PREPROCESSING is True

    def test_paths_no_hardcoded_windows(self):
        """Ensure default model paths don't point to C:\\models."""
        from config import paths

        for attr in ["ADIPOCYTE_MODEL_DIR", "TUMOR_MODEL_DIR", "TISSUE_MODEL_DIR"]:
            val = getattr(paths, attr)
            if val is not None:
                assert not val.startswith(r"C:\models"), (
                    f"paths.{attr} still has hardcoded Windows path: {val}"
                )

    def test_version_defined(self):
        from __init__ import __version__

        assert __version__
        assert "." in __version__

    def test_argument_parser(self):
        from argument_parser import parse_arguments

        assert callable(parse_arguments)

    def test_model_downloader(self):
        from model_downloader import ensure_model, get_cache_dir

        assert callable(ensure_model)
        cache_dir = get_cache_dir()
        assert cache_dir.exists()


class TestMppExtraction:
    """Verify OpenSlide and OME-TIFF MPP extraction paths."""

    def _ome_xml(self, pixels_attrs):
        return f"""<?xml version="1.0" encoding="UTF-8"?>
<OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06">
  <Image ID="Image:0">
    <Pixels ID="Pixels:0" DimensionOrder="XYCZT" Type="uint8"
            SizeX="5" SizeY="4" SizeC="1" SizeZ="1" SizeT="1"
            {pixels_attrs}>
      <Channel ID="Channel:0:0" SamplesPerPixel="1"/>
      <TiffData IFD="0" PlaneCount="1"/>
    </Pixels>
  </Image>
</OME>"""

    def test_ome_xml_parser_reads_namespaced_physical_sizes(self):
        import image_processing

        ome_xml = self._ome_xml(
            'PhysicalSizeX="0.25" PhysicalSizeXUnit="&#181;m" '
            'PhysicalSizeY="0.35" PhysicalSizeYUnit="&#181;m"'
        )

        assert image_processing._extract_mpp_from_ome_xml(ome_xml) == pytest.approx(0.30)

    def test_ome_xml_parser_defaults_missing_units_to_microns(self):
        import image_processing

        ome_xml = self._ome_xml('PhysicalSizeX="0.20" PhysicalSizeY="0.40"')

        assert image_processing._extract_mpp_from_ome_xml(ome_xml) == pytest.approx(0.30)

    def test_ome_xml_parser_converts_nanometers_to_microns(self):
        import image_processing

        ome_xml = self._ome_xml(
            'PhysicalSizeX="250" PhysicalSizeXUnit="nm" '
            'PhysicalSizeY="500" PhysicalSizeYUnit="nm"'
        )

        assert image_processing._extract_mpp_from_ome_xml(ome_xml) == pytest.approx(0.375)

    def test_ome_xml_parser_rejects_invalid_physical_sizes(self):
        import image_processing

        zero_xml = self._ome_xml('PhysicalSizeX="0" PhysicalSizeY="0.40"')
        negative_xml = self._ome_xml('PhysicalSizeX="0.20" PhysicalSizeY="-0.40"')

        assert image_processing._extract_mpp_from_ome_xml(zero_xml) is None
        assert image_processing._extract_mpp_from_ome_xml(negative_xml) is None

    def test_get_mpp_prefers_valid_openslide_metadata(self, monkeypatch):
        import image_processing

        class FakeSlide:
            def __init__(self):
                self.properties = {
                    image_processing.PROPERTY_NAME_MPP_X: "0.20",
                    image_processing.PROPERTY_NAME_MPP_Y: "0.40",
                }
                self.closed = False

            def close(self):
                self.closed = True

        fake_slide = FakeSlide()
        monkeypatch.setattr(image_processing, "OpenSlide", lambda _: fake_slide)

        def fail_if_called(_):
            raise AssertionError("OME fallback should not run for valid OpenSlide MPP")

        monkeypatch.setattr(image_processing, "_extract_ome_tiff_mpp", fail_if_called)

        assert image_processing.get_mpp("slide.svs") == pytest.approx(0.30)
        assert fake_slide.closed is True

    def test_get_mpp_falls_back_to_ome_tiff_metadata(self, monkeypatch):
        import image_processing

        tifffile = pytest.importorskip("tifffile")
        ome_xml = self._ome_xml(
            'PhysicalSizeX="0.25" PhysicalSizeXUnit="um" '
            'PhysicalSizeY="0.50" PhysicalSizeYUnit="um"'
        )
        ome_path = Path(__file__).resolve().parent / f"_tmp_ome_mpp_{os.getpid()}.ome.tif"
        try:
            tifffile.imwrite(ome_path, np.zeros((4, 5), dtype=np.uint8), description=ome_xml)

            def raise_openslide_error(_):
                raise RuntimeError("not an OpenSlide file")

            monkeypatch.setattr(image_processing, "OpenSlide", raise_openslide_error)

            assert image_processing.get_mpp(ome_path) == pytest.approx(0.375)
        finally:
            if ome_path.exists():
                ome_path.unlink()

    def test_get_mpp_uses_default_when_metadata_is_missing(self, monkeypatch):
        import image_processing
        from config import config

        def raise_openslide_error(_):
            raise RuntimeError("not an OpenSlide file")

        monkeypatch.setattr(image_processing, "OpenSlide", raise_openslide_error)
        monkeypatch.setattr(image_processing, "_extract_ome_tiff_mpp", lambda _: None)

        assert image_processing.get_mpp("missing.ome.tif") == config.DEFAULT_MPP

    def test_system_utils_get_mpp_reuses_image_processing_logic(self, monkeypatch):
        import image_processing
        import system_utils

        calls = []

        def fake_get_mpp(image_handler):
            calls.append(image_handler)
            return 0.42

        sentinel = object()
        monkeypatch.setattr(image_processing, "get_mpp", fake_get_mpp)

        assert system_utils.get_mpp(sentinel) == pytest.approx(0.42)
        assert calls == [sentinel]


# ================================================================
# MODEL REGISTRY TESTS
# ================================================================


class TestModelRegistry:
    """Verify canonical checkpoint naming is enforced."""

    def test_resolve_model_path_accepts_canonical_checkpoint(self):
        from model_registry import MODEL_FILENAMES, resolve_model_path

        code_dir = Path(__file__).resolve().parent
        with tempfile.TemporaryDirectory(dir=code_dir) as temp_dir:
            model_dir = Path(temp_dir)
            model_file = model_dir / MODEL_FILENAMES["adipocyte"]
            model_file.touch()

            model_path, resolved_dir = resolve_model_path("adipocyte", model_dir=str(model_dir))

            assert model_path == str(model_file)
            assert resolved_dir == str(model_dir)

    def test_resolve_model_path_rejects_legacy_filename(self):
        from model_registry import resolve_model_path

        code_dir = Path(__file__).resolve().parent
        with tempfile.TemporaryDirectory(dir=code_dir) as temp_dir:
            model_dir = Path(temp_dir)
            (model_dir / "model_final.pth").touch()

            with pytest.raises(FileNotFoundError, match="adifind_adipocyte.pth"):
                resolve_model_path("adipocyte", model_dir=str(model_dir))

    def test_resolve_model_path_rejects_noncanonical_override(self):
        from model_registry import resolve_model_path

        with pytest.raises(ValueError, match="adifind_tumor.pth"):
            resolve_model_path("tumor", model_dir="unused", model_checkpoint="model_final.pth")

    def test_resolve_model_path_uses_auto_download_for_canonical_name(self, monkeypatch):
        import model_downloader
        from model_registry import MODEL_FILENAMES, resolve_model_path

        code_dir = Path(__file__).resolve().parent
        with tempfile.TemporaryDirectory(dir=code_dir) as temp_dir:
            model_path = Path(temp_dir) / MODEL_FILENAMES["tissue"]
            model_path.touch()

            def fake_ensure_model(model_name, checkpoint=None, repo_id=None, token=None):
                assert model_name == "tissue"
                assert checkpoint == MODEL_FILENAMES["tissue"]
                return model_path

            monkeypatch.setattr(model_downloader, "ensure_model", fake_ensure_model)

            resolved_path, resolved_dir = resolve_model_path("tissue")

            assert resolved_path == str(model_path)
            assert resolved_dir == str(model_path.parent)


class TestModelDownloader:
    """Verify downloader errors stay actionable for private Hugging Face repos."""

    def test_private_repo_auth_error_is_clear(self, monkeypatch):
        import model_downloader
        from model_registry import MODEL_FILENAMES

        class DummyResponse:
            status_code = 403

        http_error = requests.HTTPError("forbidden")
        http_error.response = DummyResponse()

        def fake_download(**kwargs):
            raise http_error

        code_dir = Path(__file__).resolve().parent
        with tempfile.TemporaryDirectory(dir=code_dir) as temp_dir:
            monkeypatch.setenv("ADIFIND_CACHE_DIR", temp_dir)
            monkeypatch.setattr(model_downloader, "_download_with_progress", fake_download)

            with pytest.raises(PermissionError, match="HF_TOKEN|private"):
                model_downloader.ensure_model(
                    "adipocyte",
                    checkpoint=MODEL_FILENAMES["adipocyte"],
                    repo_id="private/repo",
                )


# ================================================================
# CLI TESTS
# ================================================================


class TestCLI:
    """Verify the CLI entry point works."""

    def test_help_flag(self):
        result = subprocess.run(
            [sys.executable, "main.py", "--help"],
            capture_output=True,
            text=True,
            cwd=os.path.dirname(__file__),
        )
        assert result.returncode == 0
        assert "AdiFind" in result.stdout

    def test_version_flag(self):
        result = subprocess.run(
            [sys.executable, "main.py", "--version"],
            capture_output=True,
            text=True,
            cwd=os.path.dirname(__file__),
        )
        assert result.returncode == 0

    def test_image_annotation_flags_parse_to_new_argument_names(self, monkeypatch):
        args = _parse_args(monkeypatch, "--save_image_annotation")

        assert args.save_image_annotation is True
        assert args.skip_image_annotation is False
        assert not hasattr(args, _legacy_image_annotation_attr())
        assert not hasattr(args, _legacy_image_skip_attr())

    def test_qupath_annotation_flags_parse_to_new_argument_names(self, monkeypatch):
        args = _parse_args(monkeypatch, "--save_qupath_annotation")

        assert args.save_qupath_annotation is True
        assert args.skip_qupath_annotation is False
        assert not hasattr(args, _legacy_qupath_save_attr())
        assert not hasattr(args, _legacy_qupath_skip_attr())

    def test_hidden_gpu_probe_flag_parses(self, monkeypatch):
        args = _parse_args(monkeypatch, "--gpu_probe_only")

        assert args.gpu_probe_only is True

    def test_annotated_image_flags_are_mutually_exclusive(self, monkeypatch):
        with pytest.raises(SystemExit):
            _parse_args(monkeypatch, "--save_image_annotation", "--skip_image_annotation")

    def test_qupath_annotation_flags_are_mutually_exclusive(self, monkeypatch):
        with pytest.raises(SystemExit):
            _parse_args(monkeypatch, "--save_qupath_annotation", "--skip_qupath_annotation")

    def test_legacy_image_annotation_flags_are_rejected(self, monkeypatch):
        with pytest.raises(SystemExit):
            _parse_args(monkeypatch, _legacy_image_annotation_flag())

    def test_legacy_qupath_annotation_flags_are_rejected(self, monkeypatch):
        with pytest.raises(SystemExit):
            _parse_args(monkeypatch, _legacy_qupath_annotation_flag())

    def test_save_mode_rejects_unsupported_value(self, monkeypatch):
        with pytest.raises(SystemExit):
            _parse_args(monkeypatch, "--save_mode", "unsupported_legacy_mode")


class TestAcceleratedVisualization:
    """Verify accelerated image-saving mode validation."""

    def test_accelerated_save_rejects_unknown_mode(self, tmp_path):
        from accelerated_visualization import save_annotated_image_accelerated

        image = np.zeros((2, 2, 3), dtype=np.uint8)
        with pytest.raises(ValueError, match="Unsupported image save mode"):
            save_annotated_image_accelerated(
                image,
                str(tmp_path),
                "sample",
                "unsupported_legacy_mode",
            )


class TestDesktopCliGeneration:
    """Verify desktop GUI settings map to the expected CLI flags."""

    @staticmethod
    def _build_cli(**overrides):
        pytest.importorskip("PySide6")
        from adifind_desktop import RunConfig, build_cli_argv

        cfg = RunConfig(image_path="example_data/K106942.svs", **overrides)
        return build_cli_argv(cfg)

    def test_desktop_cli_emits_explicit_annotation_overrides(self):
        cli = self._build_cli(save_image_annotation=True, save_qupath_annotation=False)

        assert "--save_image_annotation" in cli
        assert "--skip_image_annotation" not in cli
        assert "--skip_qupath_annotation" in cli
        assert "--save_qupath_annotation" not in cli

    def test_desktop_cli_debug_modes_map_to_expected_flags(self):
        off_cli = self._build_cli(debug_mode="off")
        processed_cli = self._build_cli(debug_mode="processed")
        unprocessed_cli = self._build_cli(debug_mode="unprocessed")

        assert "--debug" not in off_cli
        assert processed_cli.count("--debug") == 1
        assert "unprocessed" not in processed_cli
        assert unprocessed_cli.count("--debug") == 1
        assert "unprocessed" in unprocessed_cli

    def test_desktop_cli_emits_output_and_runtime_flags(self):
        cli = self._build_cli(
            save_mode="fast",
            profiling=True,
            verbose=True,
            save_tissue_window_grid=True,
            roi_max_dim=4096,
            roi_min_coverage=0.35,
        )

        assert "--save_mode" in cli
        assert "fast" in cli
        assert "--profiling" in cli
        assert "--verbose" in cli
        assert "--save_tissue_window_grid" in cli
        assert "--roi_max_dim" in cli
        assert "4096" in cli
        assert "--roi_min_coverage" in cli
        assert "0.35" in cli

    def test_desktop_cli_keeps_legacy_benchmark_flag_support(self):
        cli = self._build_cli(benchmark_saving=True)

        assert "--benchmark_saving" in cli

    @pytest.mark.parametrize(
        ("gpu_mode", "expected_flag", "unexpected_flags"),
        [
            ("full_gpu", None, ("--disable_gpu_preprocessing", "--disable_gpu_ops", "--disable_gpu_accel")),
            ("disable_gpu_preprocessing", "--disable_gpu_preprocessing", ("--disable_gpu_ops", "--disable_gpu_accel")),
            ("disable_gpu_ops", "--disable_gpu_ops", ("--disable_gpu_preprocessing", "--disable_gpu_accel")),
            ("cpu_only", "--disable_gpu_accel", ("--disable_gpu_preprocessing", "--disable_gpu_ops")),
        ],
    )
    def test_desktop_cli_gpu_modes_are_mutually_exclusive(self, gpu_mode, expected_flag, unexpected_flags):
        cli = self._build_cli(gpu_mode=gpu_mode)

        if expected_flag is None:
            assert "--disable_gpu_preprocessing" not in cli
            assert "--disable_gpu_ops" not in cli
            assert "--disable_gpu_accel" not in cli
        else:
            assert expected_flag in cli
        for flag in unexpected_flags:
            assert flag not in cli

    @pytest.mark.parametrize(
        ("memory_mode", "expected_flag", "unexpected_flag"),
        [
            ("auto", None, None),
            ("memmap_mask", "--memmap_mask", "--low_memory"),
            ("low_memory", "--low_memory", "--memmap_mask"),
        ],
    )
    def test_desktop_cli_memory_modes_are_mutually_exclusive(self, memory_mode, expected_flag, unexpected_flag):
        cli = self._build_cli(memory_mode=memory_mode)

        if expected_flag is None:
            assert "--memmap_mask" not in cli
            assert "--low_memory" not in cli
        else:
            assert expected_flag in cli
            assert unexpected_flag not in cli

    def test_desktop_cli_does_not_emit_retired_window_stride_flags(self):
        cli = self._build_cli()

        assert "--window_size" not in cli
        assert "--stride" not in cli

    def test_desktop_cli_always_targets_repo_local_main_script(self):
        pytest.importorskip("PySide6")
        from adifind_desktop import DEFAULT_SCRIPT_PATH

        cli = self._build_cli()

        assert cli[1] == DEFAULT_SCRIPT_PATH

    def test_desktop_probe_cli_appends_hidden_probe_flag(self):
        pytest.importorskip("PySide6")
        from adifind_desktop import RunConfig, build_cli_argv

        cli = build_cli_argv(RunConfig(image_path="example_data/K106942.svs"), gpu_probe_only=True)

        assert "--gpu_probe_only" in cli

    def test_desktop_probe_runs_for_gpu_modes_and_skips_cpu_only(self):
        pytest.importorskip("PySide6")
        from adifind_desktop import RunConfig, should_run_gpu_probe

        assert should_run_gpu_probe(RunConfig(gpu_mode="full_gpu")) is True
        assert should_run_gpu_probe(RunConfig(gpu_mode="disable_gpu_ops")) is True
        assert should_run_gpu_probe(RunConfig(gpu_mode="disable_gpu_preprocessing")) is True
        assert should_run_gpu_probe(RunConfig(gpu_mode="cpu_only")) is False

    def test_desktop_launch_uses_script_directory_as_working_dir(self, desktop_qt, monkeypatch):
        app, _ = desktop_qt
        from adifind_desktop import (
            DEFAULT_SCRIPT_PATH,
            RunConfig,
            RunWorker,
            build_cli_argv,
            resolve_analysis_cwd,
        )

        captured = {}

        class DummyProc:
            def __init__(self):
                self.stdout = []

            def poll(self):
                return 0

        def fake_popen(argv, **kwargs):
            captured["argv"] = argv
            captured["cwd"] = kwargs.get("cwd")
            return DummyProc()

        monkeypatch.setattr(subprocess, "Popen", fake_popen)

        worker = RunWorker(RunConfig(image_path="example_data/K106942.svs"))
        worker._start_subprocess(build_cli_argv(worker.cfg))
        app.processEvents()

        assert captured["argv"][1] == DEFAULT_SCRIPT_PATH
        assert captured["cwd"] == resolve_analysis_cwd(DEFAULT_SCRIPT_PATH)


class TestDesktopUi:
    """Verify the desktop window exposes the intended simple/advanced experience."""

    @staticmethod
    def _new_window(app):
        from adifind_desktop import MainWindow

        win = MainWindow()
        win.show()
        app.processEvents()
        return win

    def test_simple_mode_hides_advanced_sections_by_default(self, desktop_qt):
        app, _ = desktop_qt
        win = self._new_window(app)

        try:
            assert not win.btnAdvancedToggle.isChecked()
            assert win.outputGroup.isVisibleTo(win)
            assert not win.advancedContainer.isVisibleTo(win)
            assert not win.diagnosticsGroup.isVisibleTo(win)
            assert not win.batchGroup.isVisibleTo(win)
            assert not win.runtimeGroup.isVisibleTo(win)
            assert not win.runDetailsContainer.isVisibleTo(win)
            assert win.executionPane.isVisibleTo(win)
            assert win.pbBatch.isVisibleTo(win)
            assert win.pbImage.isVisibleTo(win)
        finally:
            win.hide()
            win.deleteLater()
            app.processEvents()

    def test_advanced_expander_reveals_hidden_sections_and_persists(self, desktop_qt):
        app, _ = desktop_qt
        win = self._new_window(app)

        try:
            win.btnAdvancedToggle.setChecked(True)
            app.processEvents()
            win._save_settings()

            assert win.advancedContainer.isVisibleTo(win)
            assert win.diagnosticsGroup.isVisibleTo(win)
            assert win.batchGroup.isVisibleTo(win)
            assert win.runtimeGroup.isVisibleTo(win)
        finally:
            win.hide()
            win.deleteLater()
            app.processEvents()

        win2 = self._new_window(app)
        try:
            assert win2.btnAdvancedToggle.isChecked()
            assert win2.advancedContainer.isVisibleTo(win2)
        finally:
            win2.hide()
            win2.deleteLater()
            app.processEvents()

    def test_execution_pane_stays_visible_when_advanced_section_expands(self, desktop_qt):
        app, _ = desktop_qt
        win = self._new_window(app)

        try:
            assert win.mainSplitter.widget(1) is win.executionPane
            assert win.executionPane.minimumHeight() >= 220

            win.btnAdvancedToggle.setChecked(True)
            app.processEvents()
            assert win.executionPane.isVisibleTo(win)
            assert win.pbBatch.isVisibleTo(win)
            assert win.pbImage.isVisibleTo(win)

            win.btnAdvancedToggle.setChecked(False)
            app.processEvents()
            assert win.executionPane.isVisibleTo(win)
            assert win.pbBatch.isVisibleTo(win)
            assert win.pbImage.isVisibleTo(win)
        finally:
            win.hide()
            win.deleteLater()
            app.processEvents()

    def test_run_details_toggle_survives_advanced_expand_collapse(self, desktop_qt):
        app, _ = desktop_qt
        win = self._new_window(app)

        try:
            win.btnRunDetailsToggle.setChecked(True)
            app.processEvents()
            assert win.runDetailsContainer.isVisibleTo(win)

            win.btnAdvancedToggle.setChecked(True)
            app.processEvents()
            assert win.runDetailsContainer.isVisibleTo(win)

            win.btnAdvancedToggle.setChecked(False)
            app.processEvents()
            assert win.runDetailsContainer.isVisibleTo(win)
        finally:
            win.hide()
            win.deleteLater()
            app.processEvents()

    def test_splitter_sizes_persist_across_restart(self, desktop_qt):
        app, _ = desktop_qt
        win = self._new_window(app)

        try:
            win.mainSplitter.setSizes([420, 280])
            app.processEvents()
            win._save_settings()
        finally:
            win.hide()
            win.deleteLater()
            app.processEvents()

        win2 = self._new_window(app)
        try:
            sizes = win2.mainSplitter.sizes()
            assert len(sizes) == 2
            assert sizes[1] > 0
        finally:
            win2.hide()
            win2.deleteLater()
            app.processEvents()

    def test_batch_progress_label_shows_processed_and_remaining_counts(self, desktop_qt):
        app, _ = desktop_qt
        win = self._new_window(app)

        try:
            win._set_running_progress()
            win._set_batch_max(2)
            win._set_batch_val(1)
            app.processEvents()

            assert win.lblBatch.text() == "Batch progress: 1 / 2 processed, 1 remaining"
        finally:
            win.hide()
            win.deleteLater()
            app.processEvents()

    def test_error_finish_preserves_last_known_progress_and_formats_native_crash(self, desktop_qt, monkeypatch):
        app, _ = desktop_qt
        win = self._new_window(app)

        try:
            from PySide6 import QtWidgets

            captured = {}
            monkeypatch.setattr(
                QtWidgets.QMessageBox,
                "warning",
                lambda *args, **kwargs: captured.setdefault("message", args[2]),
            )
            win._set_running_progress()
            win._set_batch_max(2)
            win._set_batch_val(1)
            win._set_current_image("example.svs")
            win.worker = type(
                "WorkerStub",
                (),
                {
                    "_probe_failed": False,
                    "_last_probe_stage": "cupy",
                    "_last_runtime_stage": "runtime_model_setup",
                    "_probe_error_message": "",
                },
            )()

            win._run_finished(-1073741819)
            app.processEvents()

            assert win.lblBatch.text() == "Batch progress: 1 / 2 processed, 1 remaining"
            assert win.lblImage.text() == "Current image: example.svs"
            assert win.lblRunState.text() == "Error"
            assert "access violation" in win.status.currentMessage().lower()
            assert "model setup" in captured["message"].lower()
            assert "disable gpu ops" in captured["message"].lower()
        finally:
            win.hide()
            win.deleteLater()
            app.processEvents()

    def test_gpu_probe_failure_message_includes_stage_and_probe_detail(self, desktop_qt):
        app, _ = desktop_qt
        win = self._new_window(app)

        try:
            win.worker = type(
                "WorkerStub",
                (),
                {
                    "_probe_failed": True,
                    "_last_probe_stage": "gpu_preprocessing",
                    "_last_runtime_stage": "",
                    "_probe_error_message": "gpu_preprocessing: CUDA kernel launch failed",
                },
            )()

            msg = win._format_exit_message(2)
            assert "gpu probe failed" in msg.lower()
            assert "gpu preprocessing micro-path" in msg.lower()
            assert "cuda kernel launch failed" in msg.lower()
            assert "disable gpu ops" in msg.lower()
        finally:
            win.hide()
            win.deleteLater()
            app.processEvents()

    def test_output_controls_disable_when_annotated_image_saving_is_off(self, desktop_qt):
        app, _ = desktop_qt
        win = self._new_window(app)

        try:
            win.cbSaveAnnotatedImage.setChecked(False)
            app.processEvents()

            assert not win.lblAnnotatedScale.isEnabled()
            assert not win.dsbAnnotatedScale.isEnabled()
            assert not win.lblSaveMode.isEnabled()
            assert not win.cmbSaveMode.isEnabled()

            win.cbSaveAnnotatedImage.setChecked(True)
            app.processEvents()

            assert win.lblAnnotatedScale.isEnabled()
            assert win.dsbAnnotatedScale.isEnabled()
            assert win.lblSaveMode.isEnabled()
            assert win.cmbSaveMode.isEnabled()
        finally:
            win.hide()
            win.deleteLater()
            app.processEvents()

    def test_distance_map_requires_tumour_segmentation(self, desktop_qt):
        app, _ = desktop_qt
        win = self._new_window(app)

        try:
            win.cbTumor.setChecked(False)
            app.processEvents()
            assert not win.cbSaveDistanceMap.isEnabled()

            win.cbTumor.setChecked(True)
            app.processEvents()
            assert win.cbSaveDistanceMap.isEnabled()
        finally:
            win.hide()
            win.deleteLater()
            app.processEvents()

    def test_stale_save_mode_is_normalized_in_editable_ui(self, desktop_qt):
        app, QtCore = desktop_qt
        settings = QtCore.QSettings("adi", "adifind_desktop")
        settings.setValue("save_mode", "unsupported_legacy_mode")
        settings.sync()

        win = self._new_window(app)
        try:
            assert win.cmbSaveMode.currentData() == "balanced"
            assert [
                win.cmbSaveMode.itemData(i)
                for i in range(win.cmbSaveMode.count())
            ] == ["fast", "balanced", "high_quality"]
            assert win.cmbSaveMode.findData("unsupported_legacy_mode") == -1
        finally:
            win.hide()
            win.deleteLater()
            app.processEvents()

    def test_legacy_benchmark_setting_does_not_resurface_in_ui(self, desktop_qt):
        app, QtCore = desktop_qt
        settings = QtCore.QSettings("adi", "adifind_desktop")
        settings.setValue("benchmark_saving", True)
        settings.sync()

        win = self._new_window(app)
        try:
            assert not hasattr(win, "cbBenchmarkSaving")
            assert win._collect_cfg().benchmark_saving is False
        finally:
            win.hide()
            win.deleteLater()
            app.processEvents()

    def test_stale_script_path_setting_is_removed_on_startup(self, desktop_qt):
        app, QtCore = desktop_qt
        settings = QtCore.QSettings("adi", "adifind_desktop")
        settings.setValue("script_path", "C:/stale/old_main.py")
        settings.sync()

        win = self._new_window(app)
        try:
            settings.sync()
            assert not settings.contains("script_path")
        finally:
            win.hide()
            win.deleteLater()
            app.processEvents()

    def test_environment_settings_dialog_exposes_fixed_repo_script_path(self, desktop_qt):
        app, _ = desktop_qt
        pytest.importorskip("PySide6")
        from adifind_desktop import DEFAULT_SCRIPT_PATH

        win = self._new_window(app)
        try:
            dialog = win._build_environment_settings_dialog()
            script_edit = dialog.findChild(type(dialog._scriptPathEdit), "environmentScriptPath")
            assert script_edit is not None
            assert script_edit.isReadOnly()
            assert script_edit.text() == DEFAULT_SCRIPT_PATH
            dialog.close()
        finally:
            win.hide()
            win.deleteLater()
            app.processEvents()

    def test_resume_mode_rejects_unsupported_save_mode(self, desktop_qt):
        app, _ = desktop_qt
        win = self._new_window(app)

        try:
            batch_state = {
                "batch_id": "unsupported-save-mode-batch",
                "total_images": 10,
                "processed_images": 2,
                "args_snapshot": {
                    "tumor_segmentation": True,
                    "save_mode": "unsupported_legacy_mode",
                },
            }

            with pytest.raises(ValueError, match="Unsupported save mode"):
                win._apply_batch_state_settings(batch_state)
        finally:
            win.hide()
            win.deleteLater()
            app.processEvents()

    def test_resume_mode_preserves_batch_benchmark_option(self, desktop_qt):
        app, _ = desktop_qt
        win = self._new_window(app)

        try:
            batch_state = {
                "batch_id": "benchmark-batch",
                "total_images": 10,
                "processed_images": 2,
                "args_snapshot": {
                    "tumor_segmentation": True,
                    "save_mode": "high_quality",
                    "benchmark_saving": True,
                },
            }

            win._apply_batch_state_settings(batch_state)
            win._set_resume_mode(True)
            cfg = win._collect_cfg()

            assert win.btnAdvancedToggle.isChecked()
            assert win.advancedContainer.isVisibleTo(win)
            assert win.cmbSaveMode.currentData() == "high_quality"
            assert cfg.save_mode == "high_quality"
            assert cfg.benchmark_saving is True
        finally:
            win.hide()
            win.deleteLater()
            app.processEvents()


class TestRuntimeConfiguration:
    """Verify runtime config overrides and banner rendering stay in sync."""

    def test_probe_stage_plan_matches_full_gpu_defaults(self):
        from main import planned_gpu_probe_stages

        assert planned_gpu_probe_stages(
            use_cupy=True,
            use_gpu_preprocessing=True,
            enable_gpu_label_mapping=False,
        ) == [
            "cuda_device",
            "cuda_tensor",
            "gpu_inference",
            "cupy",
            "gpu_preprocessing",
        ]

    def test_probe_stage_plan_skips_optional_gpu_ops_when_disabled(self):
        from main import planned_gpu_probe_stages

        assert planned_gpu_probe_stages(
            use_cupy=False,
            use_gpu_preprocessing=False,
            enable_gpu_label_mapping=False,
        ) == [
            "cuda_device",
            "cuda_tensor",
            "gpu_inference",
        ]

    def test_probe_stage_plan_skips_gpu_preprocessing_only(self):
        from main import planned_gpu_probe_stages

        assert planned_gpu_probe_stages(
            use_cupy=True,
            use_gpu_preprocessing=False,
            enable_gpu_label_mapping=False,
        ) == [
            "cuda_device",
            "cuda_tensor",
            "gpu_inference",
            "cupy",
        ]

    def test_update_config_defaults_disable_opt_in_features(self, monkeypatch):
        from config import config
        from configuration_manager import update_config_from_args

        args = _parse_args(monkeypatch)
        update_config_from_args(args)

        assert config.ENABLE_TISSUE_GUIDANCE is False
        assert config.ENABLE_TUMOR_SEGMENTATION is False

    def test_update_config_rejects_injected_unsupported_save_mode(self, monkeypatch):
        from configuration_manager import update_config_from_args

        args = _parse_args(monkeypatch)
        args.save_mode = "unsupported_legacy_mode"

        with pytest.raises(ValueError, match="Unsupported image save mode"):
            update_config_from_args(args)

    def test_update_config_enables_tissue_guidance(self, monkeypatch):
        from config import config
        from configuration_manager import update_config_from_args

        args = _parse_args(monkeypatch, "--tissue_guidance")
        update_config_from_args(args)

        assert config.ENABLE_TISSUE_GUIDANCE is True
        assert config.ENABLE_TUMOR_SEGMENTATION is False

    def test_update_config_enables_tumor_segmentation(self, monkeypatch):
        from config import config
        from configuration_manager import update_config_from_args

        args = _parse_args(monkeypatch, "--tumor_segmentation")
        update_config_from_args(args)

        assert config.ENABLE_TUMOR_SEGMENTATION is True
        assert config.ENABLE_TISSUE_GUIDANCE is False

    def test_update_config_preserves_existing_annotated_image_default(self, monkeypatch):
        from config import config
        from configuration_manager import update_config_from_args

        config.SAVE_ANNOTATED_IMAGE = False

        args = _parse_args(monkeypatch)
        update_config_from_args(args)

        assert config.SAVE_ANNOTATED_IMAGE is False

    def test_update_config_can_force_annotated_image_save_on(self, monkeypatch):
        from config import config
        from configuration_manager import update_config_from_args

        config.SAVE_ANNOTATED_IMAGE = False

        args = _parse_args(monkeypatch, "--save_image_annotation")
        update_config_from_args(args)

        assert config.SAVE_ANNOTATED_IMAGE is True

    def test_update_config_can_force_annotated_image_save_off(self, monkeypatch):
        from config import config
        from configuration_manager import update_config_from_args

        config.SAVE_ANNOTATED_IMAGE = True

        args = _parse_args(monkeypatch, "--skip_image_annotation")
        update_config_from_args(args)

        assert config.SAVE_ANNOTATED_IMAGE is False

    def test_update_config_preserves_existing_qupath_defaults(self, monkeypatch):
        from config import config
        from configuration_manager import update_config_from_args

        config.ENABLE_QUPATH_EXPORT = False
        config.SAVE_QUPATH_GEOJSON = False

        args = _parse_args(monkeypatch)
        update_config_from_args(args)

        assert config.ENABLE_QUPATH_EXPORT is False
        assert config.SAVE_QUPATH_GEOJSON is False

    def test_update_config_can_force_qupath_export_on(self, monkeypatch):
        from config import config
        from configuration_manager import update_config_from_args

        config.ENABLE_QUPATH_EXPORT = False
        config.SAVE_QUPATH_GEOJSON = False

        args = _parse_args(monkeypatch, "--save_qupath_annotation")
        update_config_from_args(args)

        assert config.ENABLE_QUPATH_EXPORT is True
        assert config.SAVE_QUPATH_GEOJSON is True

    def test_update_config_can_force_qupath_export_off(self, monkeypatch):
        from config import config
        from configuration_manager import update_config_from_args

        config.ENABLE_QUPATH_EXPORT = True
        config.SAVE_QUPATH_GEOJSON = True

        args = _parse_args(monkeypatch, "--skip_qupath_annotation")
        update_config_from_args(args)

        assert config.ENABLE_QUPATH_EXPORT is False

    def test_disable_gpu_accel_turns_off_gpu_status_flags(self, monkeypatch):
        from config import config
        from configuration_manager import update_config_from_args

        args = _parse_args(monkeypatch, "--disable_gpu_accel")
        update_config_from_args(args)

        assert config.USE_GPU_INFERENCE is False
        assert config.USE_CUPY is False
        assert config.USE_GPU_PREPROCESSING is False
        assert config.ENABLE_GPU_LABEL_MAPPING is False

    def test_disable_gpu_ops_keeps_inference_enabled(self, monkeypatch):
        from config import config
        from configuration_manager import update_config_from_args

        args = _parse_args(monkeypatch, "--disable_gpu_ops")
        update_config_from_args(args)

        assert config.USE_GPU_INFERENCE is True
        assert config.USE_CUPY is False
        assert config.USE_GPU_PREPROCESSING is False
        assert config.ENABLE_GPU_LABEL_MAPPING is False

    def test_disable_gpu_preprocessing_only_affects_preprocessing(self, monkeypatch):
        from config import config
        from configuration_manager import update_config_from_args

        args = _parse_args(monkeypatch, "--disable_gpu_preprocessing")
        update_config_from_args(args)

        assert config.USE_GPU_INFERENCE is True
        assert config.USE_CUPY is True
        assert config.USE_GPU_PREPROCESSING is False
        assert config.ENABLE_GPU_LABEL_MAPPING is False

    def test_legacy_use_gpu_acceleration_write_disables_all_gpu_paths(self):
        from config import config

        config.USE_GPU_ACCELERATION = False

        assert config.USE_GPU_INFERENCE is False
        assert config.USE_CUPY is False
        assert config.USE_GPU_PREPROCESSING is False
        assert config.ENABLE_GPU_LABEL_MAPPING is False

    def test_legacy_enable_gpu_acceleration_write_disables_non_inference_gpu_paths(self):
        from config import config

        config.USE_GPU_INFERENCE = True
        config.USE_CUPY = True
        config.USE_GPU_PREPROCESSING = True
        config.ENABLE_GPU_LABEL_MAPPING = True

        config.ENABLE_GPU_ACCELERATION = False

        assert config.USE_GPU_INFERENCE is True
        assert config.USE_CUPY is False
        assert config.USE_GPU_PREPROCESSING is False
        assert config.ENABLE_GPU_LABEL_MAPPING is False

    def test_banner_shows_disabled_runtime_settings(self, monkeypatch, capsys):
        from configuration_manager import update_config_from_args
        from main import display_startup_banner

        args = _parse_args(monkeypatch)
        update_config_from_args(args)
        monkeypatch.delenv("ADIFIND_NO_BANNER", raising=False)
        monkeypatch.setenv("ADIFIND_NO_ANIM", "1")

        display_startup_banner()
        output = _strip_ansi(capsys.readouterr().out)

        assert "Tumor segmentation : disabled" in output
        assert "Tissue-Guidance    : disabled" in output
        assert "GPU inference      : enabled" in output
        assert "CuPy ops           : enabled" in output
        assert "GPU preprocessing  : enabled" in output
        assert "GPU label mapping" not in output
        assert "QuPath output      : enabled" in output

    def test_banner_shows_enabled_and_disabled_runtime_settings(self, monkeypatch, capsys):
        from config import config
        from configuration_manager import update_config_from_args
        from main import display_startup_banner

        args = _parse_args(
            monkeypatch,
            "--tissue_guidance",
            "--tumor_segmentation",
            "--disable_gpu_accel",
        )
        update_config_from_args(args)
        config.ENABLE_QUPATH_EXPORT = False
        monkeypatch.delenv("ADIFIND_NO_BANNER", raising=False)
        monkeypatch.setenv("ADIFIND_NO_ANIM", "1")

        display_startup_banner()
        output = _strip_ansi(capsys.readouterr().out)

        assert "Tumor segmentation : enabled" in output
        assert "Tissue-Guidance    : enabled" in output
        assert "GPU inference      : disabled (CPU only)" in output
        assert "CuPy ops           : disabled" in output
        assert "GPU preprocessing  : disabled" in output
        assert "GPU label mapping" not in output
        assert "QuPath output      : disabled" in output

    def test_banner_reflects_qupath_cli_override(self, monkeypatch, capsys):
        from config import config
        from configuration_manager import update_config_from_args
        from main import display_startup_banner

        config.ENABLE_QUPATH_EXPORT = False
        config.SAVE_QUPATH_GEOJSON = False

        args = _parse_args(monkeypatch, "--save_qupath_annotation")
        update_config_from_args(args)
        monkeypatch.delenv("ADIFIND_NO_BANNER", raising=False)
        monkeypatch.setenv("ADIFIND_NO_ANIM", "1")

        display_startup_banner()
        output = _strip_ansi(capsys.readouterr().out)

        assert "QuPath output      : enabled" in output

    def test_banner_reflects_qupath_geojson_setting(self, monkeypatch, capsys):
        from config import config
        from configuration_manager import update_config_from_args
        from main import display_startup_banner

        config.ENABLE_QUPATH_EXPORT = True
        config.SAVE_QUPATH_GEOJSON = False

        args = _parse_args(monkeypatch)
        update_config_from_args(args)
        monkeypatch.delenv("ADIFIND_NO_BANNER", raising=False)
        monkeypatch.setenv("ADIFIND_NO_ANIM", "1")

        display_startup_banner()
        output = _strip_ansi(capsys.readouterr().out)

        assert "QuPath output      : disabled" in output


class TestQuPathGeoJsonOutputs:
    """Verify QuPath GeoJSON export gating follows config and CLI-resolved state."""

    def test_main_helper_requires_enable_and_geojson_flags(self):
        from config import config
        from main import _should_export_qupath_geojson

        config.ENABLE_QUPATH_EXPORT = False
        config.SAVE_QUPATH_GEOJSON = True
        assert _should_export_qupath_geojson() is False

        config.ENABLE_QUPATH_EXPORT = True
        config.SAVE_QUPATH_GEOJSON = False
        assert _should_export_qupath_geojson() is False

        config.ENABLE_QUPATH_EXPORT = True
        config.SAVE_QUPATH_GEOJSON = True
        assert _should_export_qupath_geojson() is True

    def test_export_qupath_annotations_returns_early_when_qupath_is_disabled(self, tmp_path):
        from config import config
        import visualization

        config.ENABLE_QUPATH_EXPORT = False
        config.SAVE_QUPATH_GEOJSON = True

        qupath_path = tmp_path / "slide_qupath_annotations.geojson"
        visualization.export_qupath_annotations(
            {1: 12},
            np.zeros((2, 2), dtype=np.uint32),
            str(tmp_path),
            "slide",
            precomputed_properties={1: {"bbox": (0, 0, 1, 1), "centroid_x": 0.5, "centroid_y": 0.5}},
        )

        assert not qupath_path.exists()

    def test_export_qupath_annotations_returns_early_when_geojson_is_disabled(self, tmp_path):
        from config import config
        import visualization

        config.ENABLE_QUPATH_EXPORT = True
        config.SAVE_QUPATH_GEOJSON = False

        qupath_path = tmp_path / "slide_qupath_annotations.geojson"
        visualization.export_qupath_annotations(
            {1: 12},
            np.zeros((2, 2), dtype=np.uint32),
            str(tmp_path),
            "slide",
            precomputed_properties={1: {"bbox": (0, 0, 1, 1), "centroid_x": 0.5, "centroid_y": 0.5}},
        )

        assert not qupath_path.exists()

    def test_main_gate_skips_qupath_export_when_disabled(self, monkeypatch):
        from config import config
        from main import _maybe_export_qupath_geojson

        config.ENABLE_QUPATH_EXPORT = False
        config.SAVE_QUPATH_GEOJSON = True

        exported = []
        monkeypatch.setattr("main.export_qupath_annotations", lambda *args: exported.append(args))

        assert _maybe_export_qupath_geojson({1: 12}, np.zeros((2, 2), dtype=np.uint32), "unused", "slide", {}) is False
        assert exported == []

    def test_main_gate_calls_qupath_export_when_enabled(self, monkeypatch):
        from config import config
        from main import _maybe_export_qupath_geojson

        config.ENABLE_QUPATH_EXPORT = True
        config.SAVE_QUPATH_GEOJSON = True

        exported = []
        monkeypatch.setattr("main.export_qupath_annotations", lambda *args: exported.append(args))

        assert _maybe_export_qupath_geojson({1: 12}, np.zeros((2, 2), dtype=np.uint32), "unused", "slide", {}) is True
        assert len(exported) == 1


class TestCsvExtendedProperties:
    """Verify optional morphometric CSV fields are emitted consistently."""

    @staticmethod
    def _read_single_csv_row(csv_path):
        import csv

        with open(csv_path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 1
        return rows[0]

    @staticmethod
    def _read_summary_json(summary_path):
        import json

        with open(summary_path, encoding="utf-8") as f:
            return json.load(f)

    def test_precomputed_csv_export_writes_extended_properties_when_enabled(self, tmp_path, monkeypatch):
        from config import config
        import visualization

        monkeypatch.setattr(config, "CALCULATE_EXTENDED_PROPERTIES", True)

        full_mask = np.zeros((6, 6), dtype=np.uint32)
        full_mask[1:4, 1:5] = 1
        precomputed_properties = {
            1: {
                "area": 12,
                "centroid_x": 2.5,
                "centroid_y": 2.0,
                "bbox": (1, 1, 4, 5),
            }
        }

        visualization.export_results_csv(
            {1: 12},
            full_mask,
            str(tmp_path),
            "slide",
            precomputed_properties=precomputed_properties,
            mpp=0.5,
        )

        row = self._read_single_csv_row(tmp_path / "adipocyte_information_slide.csv")

        for column in visualization.EXTENDED_PROPERTY_COLUMNS:
            assert column in row
            assert row[column] != ""
            float(row[column])

        assert "bbox_min_x" in row
        assert "eccentricity" not in row
        assert "equivalent_diameter" not in row

        summary = self._read_summary_json(tmp_path / "slide_summary_stats.json")
        assert summary["extended_properties_enabled"] is True

    def test_precomputed_csv_export_omits_extended_properties_when_disabled(self, tmp_path, monkeypatch):
        from config import config
        import visualization

        monkeypatch.setattr(config, "CALCULATE_EXTENDED_PROPERTIES", False)

        full_mask = np.zeros((6, 6), dtype=np.uint32)
        full_mask[1:4, 1:5] = 1
        precomputed_properties = {
            1: {
                "area": 12,
                "centroid_x": 2.5,
                "centroid_y": 2.0,
                "bbox": (1, 1, 4, 5),
            }
        }

        visualization.export_results_csv(
            {1: 12},
            full_mask,
            str(tmp_path),
            "slide",
            precomputed_properties=precomputed_properties,
            mpp=0.5,
        )

        row = self._read_single_csv_row(tmp_path / "adipocyte_information_slide.csv")

        for column in visualization.EXTENDED_PROPERTY_COLUMNS:
            assert column not in row
        assert "bbox_min_x" not in row

        summary = self._read_summary_json(tmp_path / "slide_summary_stats.json")
        assert summary["extended_properties_enabled"] is False

    def test_csv_export_uses_real_closest_tumour_ids(self, tmp_path, monkeypatch):
        from config import config
        import visualization

        monkeypatch.setattr(config, "CALCULATE_EXTENDED_PROPERTIES", False)

        full_mask = np.zeros((6, 10), dtype=np.uint32)
        full_mask[1:3, 1:3] = 1
        full_mask[1:3, 7:9] = 2
        precomputed_properties = {
            1: {
                "area": 4,
                "centroid_x": 1.5,
                "centroid_y": 1.5,
                "bbox": (1, 1, 3, 3),
            },
            2: {
                "area": 4,
                "centroid_x": 7.5,
                "centroid_y": 1.5,
                "bbox": (1, 7, 3, 9),
            },
        }

        visualization.export_results_csv(
            {1: 4, 2: 4},
            full_mask,
            str(tmp_path),
            "slide",
            adipocyte_distances={1: 12.0, 2: 24.0},
            precomputed_properties=precomputed_properties,
            mpp=1.0,
            adipocyte_closest_tumor_ids={1: 1, 2: 2},
        )

        import csv

        with open(tmp_path / "adipocyte_information_slide.csv", newline="", encoding="utf-8") as f:
            rows = {int(row["Adipocyte_ID"]): row for row in csv.DictReader(f)}

        assert rows[1]["Closest_Tumour_ID"] == "1"
        assert rows[2]["Closest_Tumour_ID"] == "2"


class TestBatchResumeArgs:
    """Verify resume state args do not override explicit current opt-ins."""

    def test_saved_batch_args_do_not_turn_off_current_extended_properties_flag(self):
        from types import SimpleNamespace
        from main import _apply_saved_batch_args

        args = SimpleNamespace(
            resume_batch="state.json",
            extended_properties=True,
            output_dir="current_output",
        )

        _apply_saved_batch_args(
            args,
            {
                "resume_batch": None,
                "extended_properties": False,
                "output_dir": "saved_output",
            },
        )

        assert args.extended_properties is True
        assert args.resume_batch == "state.json"
        assert args.output_dir == "saved_output"


class TestAnnotatedImageOutputs:
    """Verify annotated-image save toggles and overlay behavior."""

    class _DummyImageHandler:
        width = 10
        height = 10

    @staticmethod
    def _stub_annotation_render(monkeypatch):
        import system_utils
        import visualization

        monkeypatch.setattr(system_utils, "get_mpp", lambda _: 0.5)
        monkeypatch.setattr(
            visualization,
            "create_optimized_annotated_image",
            lambda *args, **kwargs: np.zeros((4, 4, 3), dtype=np.uint8),
        )

    def test_main_helper_skips_annotation_phase_when_all_outputs_are_off(self):
        from config import config
        from main import _should_run_annotated_image_phase

        config.SAVE_ANNOTATED_IMAGE = False
        config.SAVE_TUMOR_ZONE_OVERLAY_IMAGE = False
        config.ENABLE_TUMOR_SEGMENTATION = False

        assert _should_run_annotated_image_phase() is False

    def test_main_helper_runs_annotation_phase_for_tumor_overlay(self):
        from config import config
        from main import _should_run_annotated_image_phase

        config.SAVE_ANNOTATED_IMAGE = False
        config.SAVE_TUMOR_ZONE_OVERLAY_IMAGE = True
        config.ENABLE_TUMOR_SEGMENTATION = True

        assert _should_run_annotated_image_phase() is True

    def test_annotated_image_generation_skips_when_all_outputs_are_disabled(self, monkeypatch):
        from config import config
        import visualization

        config.SAVE_ANNOTATED_IMAGE = False
        config.SAVE_TUMOR_ZONE_OVERLAY_IMAGE = False
        config.ENABLE_TUMOR_SEGMENTATION = False

        rendered = []
        written = []
        monkeypatch.setattr(
            visualization,
            "create_optimized_annotated_image",
            lambda *args, **kwargs: rendered.append(True),
        )
        monkeypatch.setattr(
            visualization.tifffile,
            "imwrite",
            lambda *args, **kwargs: written.append(args[0]),
        )

        result = visualization.annotate_image_with_adipocytes(
            self._DummyImageHandler(),
            {1: 12},
            np.zeros((2, 2), dtype=np.uint32),
            "unused_output",
            (1, 1),
            image_name="slide",
            precomputed_properties={1: {"centroid_y": 1.0, "centroid_x": 1.0, "area": 12}},
        )

        assert rendered == []
        assert written == []
        assert result["generated"] is False
        assert result["saved_annotated_image"] is False
        assert result["saved_tumor_zone_overlay"] is False

    def test_overlay_can_still_save_when_base_annotated_tiff_is_disabled(self, monkeypatch):
        from config import config
        import visualization

        config.SAVE_ANNOTATED_IMAGE = False
        config.SAVE_TUMOR_ZONE_OVERLAY_IMAGE = True
        config.ENABLE_TUMOR_SEGMENTATION = True

        self._stub_annotation_render(monkeypatch)
        monkeypatch.setattr(
            visualization,
            "add_tumor_distance_zones_overlay",
            lambda image, tumor_mask, scaling_factor, mpp: (np.copy(image), {}),
        )

        written = []
        monkeypatch.setattr(
            visualization.tifffile,
            "imwrite",
            lambda path, *_args, **_kwargs: written.append(Path(path).name),
        )

        result = visualization.annotate_image_with_adipocytes(
            self._DummyImageHandler(),
            {1: 12},
            np.zeros((2, 2), dtype=np.uint32),
            "unused_output",
            (1, 1),
            tumor_mask_fullres=np.ones((2, 2), dtype=np.uint8),
            image_name="slide",
            precomputed_properties={1: {"centroid_y": 1.0, "centroid_x": 1.0, "area": 12}},
        )

        assert written == ["slide_adifind_annotated_tumor_zones.tiff"]
        assert result["generated"] is True
        assert result["saved_annotated_image"] is False
        assert result["saved_tumor_zone_overlay"] is True

    def test_base_annotated_tiff_still_saves_when_enabled(self, monkeypatch):
        from config import config
        import visualization

        config.SAVE_ANNOTATED_IMAGE = True
        config.SAVE_TUMOR_ZONE_OVERLAY_IMAGE = False
        config.ENABLE_TUMOR_SEGMENTATION = False

        self._stub_annotation_render(monkeypatch)

        written = []
        monkeypatch.setattr(
            visualization.tifffile,
            "imwrite",
            lambda path, *_args, **_kwargs: written.append(Path(path).name),
        )

        result = visualization.annotate_image_with_adipocytes(
            self._DummyImageHandler(),
            {1: 12},
            np.zeros((2, 2), dtype=np.uint32),
            "unused_output",
            (1, 1),
            image_name="slide",
            precomputed_properties={1: {"centroid_y": 1.0, "centroid_x": 1.0, "area": 12}},
        )

        assert written == ["slide_adifind_annotated.tiff"]
        assert result["generated"] is True
        assert result["saved_annotated_image"] is True
        assert result["saved_tumor_zone_overlay"] is False

    def test_tumor_overlay_uses_instance_colors(self, monkeypatch):
        from config import config
        import visualization

        monkeypatch.setattr(config, "SHOW_TUMOR_BOUNDARIES", False)
        monkeypatch.setattr(config, "TUMOR_INSTANCE_ALPHA", 1.0)
        monkeypatch.setattr(
            config,
            "TUMOR_INSTANCE_COLORS_RGB",
            (
                (10, 20, 30),
                (40, 50, 60),
            ),
        )

        image = np.zeros((4, 4, 3), dtype=np.uint8)
        tumor_mask = np.zeros((4, 4), dtype=np.uint16)
        tumor_mask[0:2, 0:2] = 1
        tumor_mask[2:4, 2:4] = 2

        rendered = visualization.add_optimized_tumor_overlay(
            image,
            tumor_mask,
            scaling_factor=1.0,
            mpp=0.5,
        )

        assert tuple(rendered[0, 0]) == (10, 20, 30)
        assert tuple(rendered[3, 3]) == (40, 50, 60)
        assert tuple(rendered[0, 3]) == (0, 0, 0)

    def test_tumor_overlay_handles_legacy_binary_mask(self, monkeypatch):
        from config import config
        import visualization

        monkeypatch.setattr(config, "SHOW_TUMOR_BOUNDARIES", False)
        monkeypatch.setattr(config, "TUMOR_INSTANCE_ALPHA", 1.0)
        monkeypatch.setattr(config, "TUMOR_INSTANCE_COLORS_RGB", ((11, 22, 33),))

        image = np.zeros((3, 3, 3), dtype=np.uint8)
        tumor_mask = np.zeros((3, 3), dtype=np.uint8)
        tumor_mask[1, 1] = 1

        rendered = visualization.add_optimized_tumor_overlay(
            image,
            tumor_mask,
            scaling_factor=1.0,
            mpp=0.5,
        )

        assert tuple(rendered[1, 1]) == (11, 22, 33)
        assert tuple(rendered[0, 0]) == (0, 0, 0)


class TestVisualizationSummarySafety:
    """Verify visualization summary math stays safe when phases complete instantly."""

    def test_safe_phase_pct_returns_zero_for_zero_denominator(self):
        from main import _safe_phase_pct

        assert _safe_phase_pct(1.5, 0.0) == 0.0

    def test_safe_phase_pct_returns_zero_for_zero_zero(self):
        from main import _safe_phase_pct

        assert _safe_phase_pct(0.0, 0.0) == 0.0

    def test_safe_phase_pct_returns_expected_percentage(self):
        from main import _safe_phase_pct

        assert _safe_phase_pct(2.0, 8.0) == 25.0

    def test_visualization_summary_handles_zero_total_time(self, caplog):
        from main import _log_visualization_summary

        with caplog.at_level(logging.INFO):
            _log_visualization_summary(
                props_extraction_time=0.0,
                annotated_image_time=0.0,
                distance_image_time=0.0,
                plain_image_time=0.0,
                sobel_image_time=0.0,
                total_visualization_time=0.0,
                adipocyte_count=42,
                image_width=100,
                image_height=100,
            )

        messages = [record.getMessage() for record in caplog.records]
        assert any("Properties Extraction:" in message and "(  0.0%)" in message for message in messages)
        assert any("Visualization efficiency: N/A" in message for message in messages)
        assert any("Visualization density:" in message for message in messages)


class TestLoggingUtilities:
    """Verify ASCII-only logging behavior on Windows-style consoles."""

    def test_console_handler_sanitizes_surrogate_message_to_ascii(self):
        from logging_utils import _create_console_handler

        constrained_stream = EncodingConstrainedStream(encoding="cp1252", allow_reconfigure=False)
        handler = _create_console_handler(stream=constrained_stream)
        logger = logging.getLogger("ascii_safe_console_surrogate_test")
        logger.handlers[:] = []
        logger.setLevel(logging.INFO)
        logger.addHandler(handler)
        logger.propagate = False

        try:
            logger.info("\uD83D\uDDA5\uFE0F Using GPU %d: %s", 0, "GPU")
        finally:
            handler.close()
            logger.handlers[:] = []

        output = constrained_stream.getvalue()
        assert output.isascii()
        assert "Using GPU 0: GPU" in output

    def test_console_handler_sanitizes_real_unicode_message_to_ascii(self):
        from logging_utils import _create_console_handler

        constrained_stream = EncodingConstrainedStream(encoding="cp1252", allow_reconfigure=False)
        handler = _create_console_handler(stream=constrained_stream)
        logger = logging.getLogger("ascii_safe_console_unicode_test")
        logger.handlers[:] = []
        logger.setLevel(logging.INFO)
        logger.addHandler(handler)
        logger.propagate = False

        try:
            logger.info("\U0001F527 Processed 20 windows | GPU:   3%")
        finally:
            handler.close()
            logger.handlers[:] = []

        output = constrained_stream.getvalue()
        assert output.isascii()
        assert "Processed 20 windows | GPU:   3%" in output

    def test_console_handler_maps_common_non_ascii_symbols(self):
        from logging_utils import _create_console_handler

        constrained_stream = EncodingConstrainedStream(encoding="cp1252", allow_reconfigure=False)
        handler = _create_console_handler(stream=constrained_stream)
        logger = logging.getLogger("ascii_safe_console_symbol_test")
        logger.handlers[:] = []
        logger.setLevel(logging.INFO)
        logger.addHandler(handler)
        logger.propagate = False

        try:
            logger.info("\u2022 Area 250 \u00B5m\u00B2 \u00D7 3")
        finally:
            handler.close()
            logger.handlers[:] = []

        output = constrained_stream.getvalue()
        assert output.isascii()
        assert "- Area 250 um^2 x 3" in output

    def test_file_handler_writes_ascii_only_for_surrogate_messages(self):
        from logging_utils import _create_file_handler

        code_dir = Path(__file__).resolve().parent
        with tempfile.TemporaryDirectory(dir=code_dir) as temp_dir:
            log_path = Path(temp_dir) / "unicode.log"
            handler = _create_file_handler(log_path)
            logger = logging.getLogger("ascii_safe_file_handler_test")
            logger.handlers[:] = []
            logger.setLevel(logging.INFO)
            logger.addHandler(handler)
            logger.propagate = False

            try:
                logger.info("\uD83D\uDD27 Processed 20 windows | GPU: %3d%%", 3)
            finally:
                handler.close()
                logger.handlers[:] = []

            content = log_path.read_text(encoding="utf-8")
            assert content.isascii()
            assert "Processed 20 windows | GPU:   3%" in content

    def test_enable_console_logging_does_not_duplicate_handlers(
        self, monkeypatch, restore_root_logger
    ):
        from logging_utils import UnicodeSafeStreamHandler, enable_console_logging

        constrained_stream = EncodingConstrainedStream(encoding="cp1252", allow_reconfigure=False)
        monkeypatch.setattr(sys, "stderr", constrained_stream)
        restore_root_logger.handlers[:] = []

        enable_console_logging()
        enable_console_logging()

        console_handlers = [
            handler
            for handler in restore_root_logger.handlers
            if isinstance(handler, UnicodeSafeStreamHandler)
        ]
        assert len(console_handlers) == 1


# ================================================================
# GPU AVAILABILITY TEST (not required to pass)
# ================================================================


class TestGPU:
    """GPU availability checks - skipped if no GPU present."""

    @pytest.mark.skipif(
        not __import__("torch").cuda.is_available(),
        reason="No CUDA GPU available",
    )
    def test_cuda_available(self):
        import torch

        assert torch.cuda.device_count() > 0
        name = torch.cuda.get_device_name(0)
        assert len(name) > 0
