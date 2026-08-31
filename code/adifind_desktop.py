# adifind_desktop.py
# ------------------------------------------------------------
# GUI runner for AdiFind that launches the analysis script
# in a separate Python process (your adifind conda env).
# - No importlib of the analysis code in this process
# - Avoids typing.Self / torch/torchvision import issues
# - Streams logs live, parses total images + per-image progress
# - Lets you override key Config/Paths safely without editing
#   the analysis script (done in a tiny launcher in the subprocess)
# ------------------------------------------------------------

import sys
import os
import re
import json
import subprocess
from dataclasses import dataclass
from typing import Optional, Dict, Any

from PySide6 import QtCore, QtWidgets, QtGui


# ---------- Defaults (adjust if you want) ----------
DEFAULT_PYTHON_EXE = r"C:\ProgramData\anaconda3\envs\adifind\python.exe"
DEFAULT_SCRIPT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "main.py"))
DEFAULT_SCRIPT_CWD = os.path.dirname(DEFAULT_SCRIPT_PATH)

# Typical things people want to tweak per run
# These defaults match config.py defaults where applicable
DEFAULTS = {
    "image_path": r"",                # empty = leave script default
    "output_dir": r"",                # empty = leave script default
    "enable_tumor": False,
    "enable_tissue": True,
    "min_area_um2": 250,              # matches config.MIN_ADIPOCYTE_AREA_MICRONS
    "max_area_um2": 25000,            # matches config.MAX_ADIPOCYTE_AREA_MICRONS
    # CLI-style options mirrored from main.py / argument_parser.py
    "debug_mode": "off",
    "verbose": False,
    "save_distance_map": False,
    "extended_properties": False,
    "save_image_annotation": False,
    "save_qupath_annotation": True,
    "save_tissue_window_grid": False,
    "annotated_scale": 0.3,
    "save_mode": "balanced",
    "benchmark_saving": False,
    "profiling": False,
    "gpu_id": 0,
    "gpu_mode": "full_gpu",
    "memory_mode": "auto",
    "batch_size": None,               # None = use config.py default (6)
    # Visualization toggles (CLI: --show_adipocyte_ids/--hide_adipocyte_ids, --show_grid/--hide_grid)
    "show_adipocyte_ids": False,      # matches config.SHOW_ADIPOCYTE_IDS default
    "show_grid": False,               # matches config.SHOW_GRID default
    # ROI guidance
    "roi_freehand": False,            # interactive ROI selection (--roi_freehand)
    "roi_polygon_file": "",           # path to pre-saved ROI polygon JSON (--roi_polygon_file)
    "roi_max_dim": 2048,
    "roi_min_coverage": 0.2,
    # Batch resume options
    "resume_batch": "",               # path to batch state file (--resume_batch)
    "resume_failed": False,           # retry failed images (--resume_failed)
    "dry_run": False,                 # show what would be processed (--dry_run)
    "allow_resume": False,
    "show_advanced": False,
}

# Regex helpers to parse progress from your script logs
RE_TOTAL_IMAGES = re.compile(r"Found\s+(\d+)\s+images\s+to\s+process", re.IGNORECASE)
RE_BATCH_REMAINING_IMAGES = re.compile(
    r"Processing\s+(\d+)\s+remaining images out of\s+(\d+)\s+total",
    re.IGNORECASE,
)
RE_PROCESSING_IMAGE = re.compile(
    r"Processing image\s+(\d+)\s*/\s*(\d+):\s*(.+)$",
    re.IGNORECASE,
)
RE_BATCH_SUMMARY = re.compile(
    r"Successfully processed:\s*(\d+)\s*/\s*(\d+)\s+images",
    re.IGNORECASE,
)
RE_TQDM = re.compile(r"(\d+)\s*/\s*(\d+)")
RE_GPU_STAGE = re.compile(r"GPU_(PROBE|RUNTIME)_STAGE:\s*([A-Za-z0-9_]+)")
RE_GPU_PROBE_ERROR = re.compile(r"GPU_PROBE_ERROR:\s*(.+)")
# Strip ANSI escape sequences (colors, cursor moves) from logs
RE_ANSI = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")

DEBUG_MODE_OPTIONS = [
    ("Off", "off"),
    ("Processed outputs", "processed"),
    ("Processed + raw windows", "unprocessed"),
]
GPU_MODE_OPTIONS = [
    ("Full GPU", "full_gpu"),
    ("Disable GPU preprocessing", "disable_gpu_preprocessing"),
    ("Disable GPU ops", "disable_gpu_ops"),
    ("CPU only", "cpu_only"),
]
MEMORY_MODE_OPTIONS = [
    ("Auto", "auto"),
    ("Memory-mapped mask", "memmap_mask"),
    ("Low-memory mode", "low_memory"),
]
SAVE_MODE_OPTIONS = [
    ("Fast (JPEG)", "fast"),
    ("Balanced", "balanced"),
    ("High quality", "high_quality"),
]
SAVE_MODE_VALUES = tuple(value for _, value in SAVE_MODE_OPTIONS)

GPU_STAGE_LABELS = {
    "cuda_device": "CUDA device selection",
    "cuda_tensor": "CUDA tensor allocation/sync",
    "gpu_inference": "GPU inference model/device preparation",
    "cupy": "CuPy alloc/free/sync",
    "gpu_preprocessing": "GPU preprocessing micro-path",
    "gpu_label_mapping": "GPU label-mapping micro-path",
    "runtime_startup": "runtime startup",
    "runtime_image_load": "image load",
    "runtime_model_setup": "model setup",
    "runtime_window_generation": "window generation",
    "runtime_window_execution": "window execution",
}


def _first_existing_path(candidates) -> Optional[str]:
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def resolve_header_logo_path() -> Optional[str]:
    """Return the best header logo path from common repo locations."""
    here = os.path.dirname(__file__)
    parent = os.path.dirname(here)
    candidates = [
        # Prefer cropped version (no empty space)
        os.path.join(parent, "media", "adifind_logo_cropped.png"),        # repo/media/...
        os.path.join(here, "media", "adifind_logo_cropped.png"),          # code/media/...
        # Fall back to original
        os.path.join(parent, "media", "adifind_logo.png"),                # repo/media/...
        os.path.join(here, "media", "adifind_logo.png"),                  # code/media/...
        os.path.join(here, "media", "adifind_color_logo_cropped.png"),    # code/media fallback
        os.path.join(parent, "media", "adifind_color_logo_cropped.png"),  # repo/media fallback
    ]
    return _first_existing_path(candidates)


def resolve_app_icon_path() -> Optional[str]:
    """Return the square application icon path for the taskbar and window."""
    here = os.path.dirname(__file__)
    parent = os.path.dirname(here)
    candidates = [
        os.path.join(parent, "media", "adifind_logo.png"),
        os.path.join(here, "media", "adifind_logo.png"),
        resolve_header_logo_path(),
    ]
    return _first_existing_path(candidates)


def _normalize_debug_mode(value: Any) -> str:
    if isinstance(value, bool):
        return "processed" if value else "off"

    text = str(value or "").strip().lower()
    if text in {"processed", "unprocessed", "off"}:
        return text
    if text in {"true", "1", "yes"}:
        return "processed"
    if text in {"false", "0", "no", ""}:
        return "off"
    return DEFAULTS["debug_mode"]


def _set_combo_to_value(combo: QtWidgets.QComboBox, value: Any, fallback: str) -> None:
    target = str(value if value is not None else fallback)
    idx = combo.findData(target)
    if idx < 0:
        idx = combo.findData(fallback)
    if idx >= 0:
        combo.setCurrentIndex(idx)


def _mode_label(options, value: str) -> str:
    for label, key in options:
        if key == value:
            return label
    return value


def _normalize_desktop_save_mode(value: Any) -> str:
    target = str(value or DEFAULTS["save_mode"]).strip()
    if target in SAVE_MODE_VALUES:
        return target
    return DEFAULTS["save_mode"]


def _require_supported_save_mode(value: Any, context: str = "save mode") -> str:
    target = str(value or DEFAULTS["save_mode"]).strip()
    if target in SAVE_MODE_VALUES:
        return target
    supported_modes = ", ".join(SAVE_MODE_VALUES)
    raise ValueError(
        f"Unsupported save mode in {context}: {target!r}. Supported modes: {supported_modes}"
    )


def _resolve_cli_override(enable_flag: Any, disable_flag: Any, default: bool) -> bool:
    if bool(enable_flag):
        return True
    if bool(disable_flag):
        return False
    return default


def _gpu_mode_from_args(args: Dict[str, Any]) -> str:
    if bool(args.get("disable_gpu_accel")):
        return "cpu_only"
    if bool(args.get("disable_gpu_ops")):
        return "disable_gpu_ops"
    if bool(args.get("disable_gpu_preprocessing")):
        return "disable_gpu_preprocessing"
    return DEFAULTS["gpu_mode"]


def _memory_mode_from_args(args: Dict[str, Any]) -> str:
    if bool(args.get("low_memory")):
        return "low_memory"
    if bool(args.get("memmap_mask")):
        return "memmap_mask"
    return DEFAULTS["memory_mode"]


def _normalize_path_for_cli(path: str) -> str:
    text = str(path or "").strip()
    if not text:
        return ""
    return os.path.abspath(text)


def resolve_analysis_cwd(script_path: Optional[str] = None) -> str:
    return os.path.dirname(os.path.abspath(script_path or DEFAULT_SCRIPT_PATH))


def should_run_gpu_probe(cfg: "RunConfig") -> bool:
    return str(getattr(cfg, "gpu_mode", "") or DEFAULTS["gpu_mode"]) != "cpu_only"


def describe_gpu_stage(stage: str) -> str:
    return GPU_STAGE_LABELS.get(stage, stage.replace("_", " ").strip())


@dataclass
class RunConfig:
    python_exe_path: str = DEFAULT_PYTHON_EXE

    # User overrides (optional; empty/None means "use script defaults")
    image_path: str = DEFAULTS["image_path"]
    output_dir: str = DEFAULTS["output_dir"]
    enable_tumor: bool = DEFAULTS["enable_tumor"]
    enable_tissue: bool = DEFAULTS["enable_tissue"]
    min_area_um2: int = DEFAULTS["min_area_um2"]
    max_area_um2: int = DEFAULTS["max_area_um2"]
    debug_mode: str = DEFAULTS["debug_mode"]
    # CLI-style options mirrored from main.py
    verbose: bool = DEFAULTS["verbose"]
    save_distance_map: bool = DEFAULTS["save_distance_map"]
    extended_properties: bool = DEFAULTS["extended_properties"]
    save_image_annotation: bool = DEFAULTS["save_image_annotation"]
    save_qupath_annotation: bool = DEFAULTS["save_qupath_annotation"]
    save_tissue_window_grid: bool = DEFAULTS["save_tissue_window_grid"]
    annotated_scale: float = DEFAULTS["annotated_scale"]
    save_mode: str = DEFAULTS["save_mode"]
    benchmark_saving: bool = DEFAULTS["benchmark_saving"]
    profiling: bool = DEFAULTS["profiling"]
    gpu_id: int = DEFAULTS["gpu_id"]
    gpu_mode: str = DEFAULTS["gpu_mode"]
    memory_mode: str = DEFAULTS["memory_mode"]
    batch_size: Optional[int] = DEFAULTS["batch_size"]
    # Visualization toggles
    show_adipocyte_ids: bool = DEFAULTS["show_adipocyte_ids"]
    show_grid: bool = DEFAULTS["show_grid"]
    # ROI guidance
    roi_freehand: bool = DEFAULTS["roi_freehand"]
    roi_polygon_file: str = DEFAULTS["roi_polygon_file"]
    roi_max_dim: int = DEFAULTS["roi_max_dim"]
    roi_min_coverage: float = DEFAULTS["roi_min_coverage"]
    # Batch resume options
    resume_batch: str = DEFAULTS["resume_batch"]
    resume_failed: bool = DEFAULTS["resume_failed"]
    dry_run: bool = DEFAULTS["dry_run"]
    # Allow running in resume/list mode without providing an image_path
    allow_resume: bool = DEFAULTS["allow_resume"]


def build_cli_argv(cfg: RunConfig, *, gpu_probe_only: bool = False) -> list[str]:
    """Build the command line used to launch the analysis script."""
    cli = [cfg.python_exe_path or sys.executable, DEFAULT_SCRIPT_PATH]

    if gpu_probe_only:
        cli.append("--gpu_probe_only")

    if cfg.resume_batch:
        cli += ["--resume_batch", cfg.resume_batch]
    elif cfg.image_path:
        cli.append(cfg.image_path)
    elif cfg.allow_resume:
        cli.append("--list_resumable")

    if cfg.output_dir:
        cli += ["--output_dir", cfg.output_dir]

    if cfg.min_area_um2:
        cli += ["--min_area", str(cfg.min_area_um2)]
    if cfg.max_area_um2:
        cli += ["--max_area", str(cfg.max_area_um2)]

    if cfg.enable_tissue:
        cli.append("--tissue_guidance")
    if cfg.enable_tumor:
        cli.append("--tumor_segmentation")

    if cfg.save_distance_map:
        cli.append("--save_distance_map")
    if cfg.extended_properties:
        cli.append("--extended_properties")
    if cfg.save_tissue_window_grid:
        cli.append("--save_tissue_window_grid")

    if cfg.annotated_scale is not None:
        cli += ["--annotated_scale", str(cfg.annotated_scale)]
    if cfg.save_mode:
        cli += [
            "--save_mode",
            _require_supported_save_mode(cfg.save_mode, "desktop run configuration"),
        ]
    if cfg.save_image_annotation:
        cli.append("--save_image_annotation")
    else:
        cli.append("--skip_image_annotation")
    if cfg.save_qupath_annotation:
        cli.append("--save_qupath_annotation")
    else:
        cli.append("--skip_qupath_annotation")
    if cfg.benchmark_saving:
        cli.append("--benchmark_saving")

    if cfg.verbose:
        cli.append("--verbose")
    if cfg.profiling:
        cli.append("--profiling")

    debug_mode = _normalize_debug_mode(cfg.debug_mode)
    if debug_mode == "processed":
        cli.append("--debug")
    elif debug_mode == "unprocessed":
        cli += ["--debug", "unprocessed"]

    if cfg.gpu_id is not None:
        cli += ["--gpu_id", str(cfg.gpu_id)]
    if cfg.batch_size is not None:
        cli += ["--batch_size", str(cfg.batch_size)]

    if cfg.gpu_mode == "disable_gpu_preprocessing":
        cli.append("--disable_gpu_preprocessing")
    elif cfg.gpu_mode == "disable_gpu_ops":
        cli.append("--disable_gpu_ops")
    elif cfg.gpu_mode == "cpu_only":
        cli.append("--disable_gpu_accel")

    if cfg.memory_mode == "memmap_mask":
        cli.append("--memmap_mask")
    elif cfg.memory_mode == "low_memory":
        cli.append("--low_memory")

    if cfg.show_adipocyte_ids:
        cli.append("--show_adipocyte_ids")
    else:
        cli.append("--hide_adipocyte_ids")
    if cfg.show_grid:
        cli.append("--show_grid")
    else:
        cli.append("--hide_grid")

    if cfg.roi_polygon_file:
        cli += ["--roi_polygon_file", cfg.roi_polygon_file]
    elif cfg.roi_freehand:
        cli.append("--roi_freehand")
    cli += ["--roi_max_dim", str(cfg.roi_max_dim)]
    cli += ["--roi_min_coverage", str(cfg.roi_min_coverage)]

    if cfg.resume_failed:
        cli.append("--resume_failed")
    if cfg.dry_run:
        cli.append("--dry_run")

    return cli


class RunWorker(QtCore.QThread):
    """RunWorker executes the analysis script in a subprocess and
    streams its stdout/stderr back to the GUI via signals.
    """

    logLine = QtCore.Signal(str)
    setBatchMax = QtCore.Signal(int)
    setBatchValue = QtCore.Signal(int)
    setImageIndeterminate = QtCore.Signal(bool)
    setImageProgress = QtCore.Signal(int, int)
    setCurrentImage = QtCore.Signal(str)
    runFinished = QtCore.Signal(int)

    def __init__(self, cfg: RunConfig):
        super().__init__()
        self.cfg = cfg
        self._proc: Optional[subprocess.Popen] = None
        self._stop_requested = False
        self._launch_cwd = resolve_analysis_cwd(DEFAULT_SCRIPT_PATH)
        self._probe_requested = False
        self._probe_failed = False
        self._probe_error_message = ""
        self._last_probe_stage = ""
        self._last_runtime_stage = ""

        # internal counters
        self._batch_total = 0
        self._batch_done = 0
        self._current_image_windows_total = 0

    def stop(self):
        self._stop_requested = True
        try:
            if self._proc and self._proc.poll() is None:
                self._proc.terminate()
        except Exception:
            pass

    def _emit_launch_preview(self, argv, *, title: str):
        try:
            python_exe = argv[0] if argv else ""
            script_path = argv[1] if len(argv) > 1 else ""
            cwd = resolve_analysis_cwd(script_path or DEFAULT_SCRIPT_PATH)
            arg_lines = "\n  ".join(map(str, argv[2:])) if len(argv) > 2 else "(none)"
            pretty = (
                f"{title}:\n"
                f"  Python: {python_exe}\n"
                f"  Script: {script_path}\n"
                f"  Working dir: {cwd}\n"
                f"  Arguments:\n  {arg_lines}"
            )
            self.logLine.emit(pretty)
        except Exception:
            pass

    def _start_subprocess(self, argv: list[str]) -> subprocess.Popen:
        script_path = argv[1] if len(argv) > 1 else DEFAULT_SCRIPT_PATH
        self._launch_cwd = resolve_analysis_cwd(script_path)
        env = os.environ.copy()
        # Prefer Python's UTF-8 mode in the child (Python 3.7+)
        env.setdefault("PYTHONUTF8", "1")
        # Make stdio decoding explicit
        env.setdefault("PYTHONIOENCODING", "utf-8")
        # Force unbuffered output so progress reaches the GUI immediately
        env["PYTHONUNBUFFERED"] = "1"
        return subprocess.Popen(
            argv,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            bufsize=1,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
            cwd=self._launch_cwd,
        )

    def _capture_gpu_stage(self, line: str) -> None:
        stage_match = RE_GPU_STAGE.search(line)
        if stage_match:
            source = stage_match.group(1).lower()
            stage = stage_match.group(2).strip()
            if source == "probe":
                self._last_probe_stage = stage
            else:
                self._last_runtime_stage = stage

        probe_error_match = RE_GPU_PROBE_ERROR.search(line)
        if probe_error_match:
            self._probe_error_message = probe_error_match.group(1).strip()

    def _run_subprocess(self, argv: list[str], *, title: str) -> int:
        self._emit_launch_preview(argv, title=title)
        try:
            self._proc = self._start_subprocess(argv)
        except Exception as e:
            self.logLine.emit(f"Failed to start process: {e}")
            return 1

        try:
            for raw in self._proc.stdout:
                if raw is None:
                    break
                line = RE_ANSI.sub("", raw.rstrip())
                self._capture_gpu_stage(line)

                # parse "Found N images to process"
                m = RE_TOTAL_IMAGES.search(line)
                if m:
                    try:
                        self._batch_total = int(m.group(1))
                        self.setBatchMax.emit(self._batch_total)
                        self.setImageIndeterminate.emit(True)
                    except Exception:
                        pass

                # parse "Processing X remaining images out of Y total"
                m = RE_BATCH_REMAINING_IMAGES.search(line)
                if m:
                    try:
                        remaining = int(m.group(1))
                        total = int(m.group(2))
                        if total > 0:
                            self._batch_total = total
                            self._batch_done = max(total - remaining, 0)
                            self.setBatchMax.emit(total)
                            self.setBatchValue.emit(self._batch_done)
                    except Exception:
                        pass

                # parse "Processing image I/N: <name>"
                m = RE_PROCESSING_IMAGE.search(line)
                if m:
                    try:
                        current_index = int(m.group(1))
                        total = int(m.group(2))
                        name = os.path.basename(m.group(3).strip())
                        if total > 0:
                            self._batch_total = total
                            # The current image has started, so completed work is the prior index.
                            self._batch_done = max(current_index - 1, 0)
                            self.setBatchMax.emit(total)
                            self.setBatchValue.emit(self._batch_done)
                        self.setCurrentImage.emit(name)
                        self.setImageIndeterminate.emit(True)
                        self._current_image_windows_total = 0
                    except Exception:
                        pass

                m = RE_BATCH_SUMMARY.search(line)
                if m:
                    try:
                        completed = int(m.group(1))
                        total = int(m.group(2))
                        if total > 0:
                            self._batch_total = total
                            self._batch_done = max(completed, 0)
                            self.setBatchMax.emit(total)
                            self.setBatchValue.emit(self._batch_done)
                    except Exception:
                        pass

                # parse tqdm-style "  123/456 ..."
                m = RE_TQDM.search(line)
                if m:
                    try:
                        cur = int(m.group(1))
                        tot = int(m.group(2))
                        if tot > 0:
                            self._current_image_windows_total = tot
                            self.setImageIndeterminate.emit(False)
                            self.setImageProgress.emit(cur, tot)
                    except Exception:
                        pass

                # Suppress noisy tqdm redraws unless it's a Label Mapping line
                if (
                    "Building Label Mapping" not in line
                    and RE_TQDM.search(line)
                    and ("it/s" in line or "|" in line or "%" in line)
                ):
                    continue

                # emit cleaned line
                self.logLine.emit(line)

                # handle cancel request
                if self._stop_requested:
                    try:
                        if self._proc and self._proc.poll() is None:
                            self._proc.terminate()
                    except Exception:
                        pass
                    break

            if self._stop_requested:
                return 130
            return self._proc.wait()
        except Exception as e:
            self.logLine.emit(f"ERROR: {e}")
            return 1
        finally:
            try:
                if self._proc and self._proc.poll() is None:
                    self._proc.terminate()
            except Exception:
                pass

    def run(self):
        if should_run_gpu_probe(self.cfg):
            self._probe_requested = True
            probe_cli = build_cli_argv(self.cfg, gpu_probe_only=True)
            probe_ret = self._run_subprocess(probe_cli, title="Launching GPU probe subprocess")
            if probe_ret != 0:
                self._probe_failed = probe_ret != 130
                self.runFinished.emit(probe_ret)
                return

        cli = build_cli_argv(self.cfg)
        ret = self._run_subprocess(cli, title="Launching analysis subprocess")
        self.runFinished.emit(ret)



class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("AdiFind - Desktop")
        self.resize(1000, 720)

        self.cfg = RunConfig()

        # Track batch progress for label (x / y)
        self._batch_total_ui = 0
        self._batch_done_ui = 0
        self._image_total_ui = 0
        self._image_done_ui = 0
        self._image_progress_busy = False
        self._run_state = "Ready"
        self._logo_path: Optional[str] = None
        self._pending_close = False
        
        # Resume mode state
        self._resume_mode = False
        self._resume_batch_state: Optional[Dict[str, Any]] = None

        self._build_ui()
        self._wire_events()
        self._load_settings()

        self.worker: Optional[RunWorker] = None

    def _resolve_logo_path(self) -> Optional[str]:
        return resolve_header_logo_path()

    def _resolve_app_icon_path(self) -> Optional[str]:
        return resolve_app_icon_path()

    def _populate_combo(self, combo: QtWidgets.QComboBox, options) -> None:
        for label, value in options:
            combo.addItem(label, value)

    # ---------- UI ----------
    def _build_ui(self):
        root = QtWidgets.QWidget()
        self.setCentralWidget(root)

        rootLayout = QtWidgets.QVBoxLayout(root)
        rootLayout.setContentsMargins(12, 8, 12, 12)
        rootLayout.setSpacing(8)

        icon_path = self._resolve_app_icon_path()
        if icon_path:
            try:
                self.setWindowIcon(QtGui.QIcon(icon_path))
            except Exception:
                pass

        self._python_exe_path = self.cfg.python_exe_path
        self._save_distance_map_tooltip = (
            "Save distance transform map as additional output.\n"
            "Requires tumour segmentation to produce a result."
        )
        self._save_distance_map_disabled_tooltip = (
            "Save distance transform map as additional output.\n"
            "Enable tumour segmentation to make this option available."
        )
        self._resume_benchmark_saving_override: Optional[bool] = None

        self.mainSplitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        self.mainSplitter.setChildrenCollapsible(False)
        self.mainSplitter.setHandleWidth(8)
        rootLayout.addWidget(self.mainSplitter, 1)

        self.configScroll = QtWidgets.QScrollArea()
        self.configScroll.setWidgetResizable(True)
        self.configScroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.mainSplitter.addWidget(self.configScroll)

        configHost = QtWidgets.QWidget()
        self.configScroll.setWidget(configHost)

        g = QtWidgets.QGridLayout(configHost)
        g.setContentsMargins(0, 0, 0, 0)
        g.setHorizontalSpacing(8)
        g.setVerticalSpacing(8)
        g.setColumnStretch(1, 1)
        row = 0

        header = QtWidgets.QHBoxLayout()
        lblLogo = QtWidgets.QLabel()
        lblLogo.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)

        logo_path = self._resolve_logo_path()
        if logo_path:
            try:
                pix = QtGui.QPixmap(logo_path)
                if not pix.isNull():
                    scaled = pix.scaledToHeight(56, QtCore.Qt.SmoothTransformation)
                    lblLogo.setPixmap(scaled)
                    lblLogo.setFixedSize(scaled.size())
                    self._logo_path = logo_path
                else:
                    lblLogo.setText("AdiFind")
            except Exception:
                lblLogo.setText("AdiFind")
        else:
            lblLogo.setText("AdiFind")

        header.addWidget(lblLogo)
        header.addStretch(1)
        g.addLayout(header, row, 0, 1, 3)
        row += 1

        self.leInput = QtWidgets.QLineEdit(self.cfg.image_path)
        self.leInput.setPlaceholderText("Select image file or folder (required)")
        self.leInput.setToolTip(
            "Select a single whole slide image file (.svs, .ndpi, .tiff, etc.)\n"
            "OR a folder containing multiple images to batch process."
        )
        self.btnInputFile = QtWidgets.QPushButton("File")
        self.btnInputFile.setToolTip("Select a single image file")
        self.btnInputFolder = QtWidgets.QPushButton("Folder")
        self.btnInputFolder.setToolTip("Select a folder containing images")

        inputBtnLayout = QtWidgets.QHBoxLayout()
        inputBtnLayout.setSpacing(4)
        inputBtnLayout.addWidget(self.btnInputFile)
        inputBtnLayout.addWidget(self.btnInputFolder)

        g.addWidget(QtWidgets.QLabel("Input image or folder"), row, 0)
        g.addWidget(self.leInput, row, 1)
        g.addLayout(inputBtnLayout, row, 2)
        row += 1

        self.leOutput = QtWidgets.QLineEdit(self.cfg.output_dir)
        self.leOutput.setToolTip(
            "Where to save analysis results.\n"
            "If empty, results are saved next to the input images."
        )
        self.btnOutput = QtWidgets.QPushButton("Browse")
        g.addWidget(QtWidgets.QLabel("Output folder"), row, 0)
        g.addWidget(self.leOutput, row, 1)
        g.addWidget(self.btnOutput, row, 2)
        row += 1

        self.sbBatchSize = QtWidgets.QSpinBox()
        self.sbBatchSize.setRange(1, 32)
        self.sbBatchSize.setValue(self.cfg.batch_size if self.cfg.batch_size else 6)
        self.sbBatchSize.setSpecialValueText("Auto")
        self.sbBatchSize.setToolTip("Number of windows to batch for GPU inference (default: 6)")

        self.sbMinArea = QtWidgets.QSpinBox()
        self.sbMinArea.setRange(1, 10_000_000)
        self.sbMinArea.setValue(self.cfg.min_area_um2)
        self.sbMinArea.setSuffix(" \u00B5m\u00B2")
        self.sbMinArea.setToolTip(
            "Minimum adipocyte area in square micrometers.\n"
            "Detected cells smaller than this are excluded.\n"
            "Default: 250 \u00B5m\u00B2"
        )

        self.sbMaxArea = QtWidgets.QSpinBox()
        self.sbMaxArea.setRange(1, 10_000_000)
        self.sbMaxArea.setValue(self.cfg.max_area_um2)
        self.sbMaxArea.setSuffix(" \u00B5m\u00B2")
        self.sbMaxArea.setToolTip(
            "Maximum adipocyte area in square micrometers.\n"
            "Detected cells larger than this are excluded.\n"
            "Default: 25,000 \u00B5m\u00B2"
        )

        params_layout = QtWidgets.QHBoxLayout()

        colBatchSize = QtWidgets.QVBoxLayout()
        colBatchSize.addWidget(QtWidgets.QLabel("Batch size"))
        colBatchSize.addWidget(self.sbBatchSize)

        colMinArea = QtWidgets.QVBoxLayout()
        colMinArea.addWidget(QtWidgets.QLabel("Min area"))
        colMinArea.addWidget(self.sbMinArea)

        colMaxArea = QtWidgets.QVBoxLayout()
        colMaxArea.addWidget(QtWidgets.QLabel("Max area"))
        colMaxArea.addWidget(self.sbMaxArea)

        params_layout.addLayout(colBatchSize)
        params_layout.addSpacing(12)
        params_layout.addLayout(colMinArea)
        params_layout.addSpacing(12)
        params_layout.addLayout(colMaxArea)
        params_layout.addStretch(1)

        self.lblAutoWindowing = QtWidgets.QLabel("Windowing is automatic from slide metadata.")
        self.lblAutoWindowing.setWordWrap(True)
        self.lblAutoWindowing.setStyleSheet("color: #9aa0a6;")

        self.basicGroup = QtWidgets.QGroupBox("Analysis basics")
        basicLayout = QtWidgets.QVBoxLayout(self.basicGroup)
        basicLayout.addLayout(params_layout)
        basicLayout.addWidget(self.lblAutoWindowing)
        g.addWidget(self.basicGroup, row, 0, 1, 3)
        row += 1

        self.cbTumor = QtWidgets.QCheckBox("Enable tumour segmentation")
        self.cbTumor.setChecked(self.cfg.enable_tumor)
        self.cbTumor.setToolTip(
            "Segment and exclude tumour regions from adipocyte analysis.\n"
            "Useful for samples containing both tumour and adipose tissue."
        )

        self.cbTissue = QtWidgets.QCheckBox("Enable tissue guidance")
        self.cbTissue.setChecked(self.cfg.enable_tissue)
        self.cbTissue.setToolTip(
            "Use tissue detection to focus analysis on tissue regions only.\n"
            "Skips empty/background areas for faster processing.\n"
            "Recommended: ON"
        )

        self.cbShowAdipocyteIds = QtWidgets.QCheckBox("Show adipocyte IDs")
        self.cbShowAdipocyteIds.setChecked(self.cfg.show_adipocyte_ids)
        self.cbShowAdipocyteIds.setToolTip(
            "Display numeric IDs on each detected adipocyte in output images.\n"
            "IDs match the rows in the CSV results file."
        )

        self.cbShowGrid = QtWidgets.QCheckBox("Show analysis grid")
        self.cbShowGrid.setChecked(self.cfg.show_grid)
        self.cbShowGrid.setToolTip(
            "Overlay the analysis window grid on output images.\n"
            "Useful for understanding how the image was processed."
        )

        self.cbRoiFreehand = QtWidgets.QCheckBox("Enable ROI selection")
        self.cbRoiFreehand.setChecked(self.cfg.roi_freehand)
        self.cbRoiFreehand.setToolTip(
            "Draw or load a Region of Interest before analysis.\n"
            "Only adipocytes inside the ROI will be counted.\n"
            "An interactive drawing window will appear when the run starts."
        )

        self.leRoiPolygonFile = QtWidgets.QLineEdit()
        self.leRoiPolygonFile.setPlaceholderText("ROI polygon file (optional, skips drawing)")
        self.leRoiPolygonFile.setToolTip(
            "Path to a previously saved ROI polygon JSON file.\n"
            "If provided, the interactive ROI dialog is skipped."
        )
        self.leRoiPolygonFile.setReadOnly(True)

        self.btnLoadRoi = QtWidgets.QPushButton("Load ROI")
        self.btnLoadRoi.setToolTip("Load a previously saved ROI polygon file")
        self.btnLoadRoi.clicked.connect(self._choose_roi_polygon_file)

        self.btnClearRoi = QtWidgets.QPushButton("Clear")
        self.btnClearRoi.setToolTip("Clear the loaded ROI polygon file")
        self.btnClearRoi.clicked.connect(self._clear_roi_polygon_file)
        self.btnClearRoi.setMaximumWidth(60)

        self.cbRoiFreehand.toggled.connect(self._on_roi_toggled)
        self._on_roi_toggled(self.cbRoiFreehand.isChecked())

        self.sbRoiMaxDim = QtWidgets.QSpinBox()
        self.sbRoiMaxDim.setRange(256, 16384)
        self.sbRoiMaxDim.setSingleStep(256)
        self.sbRoiMaxDim.setValue(self.cfg.roi_max_dim)
        self.sbRoiMaxDim.setToolTip(
            "Maximum dimension for the interactive ROI thumbnail.\n"
            "Higher values improve drawing detail but use more memory."
        )

        self.dsbRoiMinCoverage = QtWidgets.QDoubleSpinBox()
        self.dsbRoiMinCoverage.setRange(0.0, 1.0)
        self.dsbRoiMinCoverage.setDecimals(2)
        self.dsbRoiMinCoverage.setSingleStep(0.05)
        self.dsbRoiMinCoverage.setValue(self.cfg.roi_min_coverage)
        self.dsbRoiMinCoverage.setToolTip(
            "Minimum ROI overlap required for a window to be processed.\n"
            "Lower values include more edge-touching windows."
        )

        self.processingGroup = QtWidgets.QGroupBox("Analysis options")
        processingLayout = QtWidgets.QVBoxLayout(self.processingGroup)
        processingRow1 = QtWidgets.QHBoxLayout()
        processingRow1.addWidget(self.cbTumor)
        processingRow1.addSpacing(8)
        processingRow1.addWidget(self.cbTissue)
        processingRow1.addStretch(1)
        processingLayout.addLayout(processingRow1)

        processingRow2 = QtWidgets.QHBoxLayout()
        processingRow2.addWidget(self.cbRoiFreehand)
        processingRow2.addSpacing(8)
        processingRow2.addWidget(self.leRoiPolygonFile, 1)
        processingRow2.addWidget(self.btnLoadRoi)
        processingRow2.addWidget(self.btnClearRoi)
        processingLayout.addLayout(processingRow2)
        g.addWidget(self.processingGroup, row, 0, 1, 3)
        row += 1

        self.outputGroup = QtWidgets.QGroupBox("Results to save")
        outputLayout = QtWidgets.QGridLayout(self.outputGroup)

        self.cbSaveAnnotatedImage = QtWidgets.QCheckBox("Save annotated image")
        self.cbSaveAnnotatedImage.setChecked(self.cfg.save_image_annotation)
        self.cbSaveAnnotatedImage.setToolTip(
            "Save the base annotated adipocyte image.\n"
            "This emits an explicit CLI override for each run."
        )

        self.cbSaveQuPathAnnotation = QtWidgets.QCheckBox("Save QuPath annotations")
        self.cbSaveQuPathAnnotation.setChecked(self.cfg.save_qupath_annotation)
        self.cbSaveQuPathAnnotation.setToolTip(
            "Export QuPath GeoJSON annotations for the run.\n"
            "This emits an explicit CLI override for each run."
        )

        self.cbSaveTissueWindowGrid = QtWidgets.QCheckBox("Save tissue window grid")
        self.cbSaveTissueWindowGrid.setChecked(self.cfg.save_tissue_window_grid)
        self.cbSaveTissueWindowGrid.setToolTip(
            "Save a low-resolution thumbnail showing tissue-guided windows."
        )

        self.cbSaveDistanceMap = QtWidgets.QCheckBox("Save distance map")
        self.cbSaveDistanceMap.setChecked(self.cfg.save_distance_map)
        self.cbSaveDistanceMap.setToolTip(self._save_distance_map_tooltip)

        self.cbExtendedProps = QtWidgets.QCheckBox("Extended properties")
        self.cbExtendedProps.setChecked(self.cfg.extended_properties)
        self.cbExtendedProps.setToolTip(
            "Calculate additional morphological properties for each adipocyte.\n"
            "Includes eccentricity, solidity, orientation, and more."
        )

        self.lblAnnotatedScale = QtWidgets.QLabel("Annotated scale")
        self.dsbAnnotatedScale = QtWidgets.QDoubleSpinBox()
        self.dsbAnnotatedScale.setRange(0.01, 2.0)
        self.dsbAnnotatedScale.setSingleStep(0.05)
        self.dsbAnnotatedScale.setValue(self.cfg.annotated_scale)
        self.dsbAnnotatedScale.setToolTip(
            "Scale factor for annotated output images.\n"
            "Lower values reduce file size at the cost of detail."
        )

        self.lblSaveMode = QtWidgets.QLabel("Save mode")
        self.cmbSaveMode = QtWidgets.QComboBox()
        self._populate_combo(self.cmbSaveMode, SAVE_MODE_OPTIONS)
        _set_combo_to_value(
            self.cmbSaveMode,
            _normalize_desktop_save_mode(self.cfg.save_mode),
            DEFAULTS["save_mode"],
        )
        self.cmbSaveMode.setToolTip(
            "Choose the annotated image output format and compression strategy."
        )

        outputLayout.addWidget(self.cbSaveAnnotatedImage, 0, 0)
        outputLayout.addWidget(self.cbSaveQuPathAnnotation, 0, 1)
        outputLayout.addWidget(self.cbSaveTissueWindowGrid, 0, 2)
        outputLayout.addWidget(self.cbSaveDistanceMap, 1, 0)
        outputLayout.addWidget(self.cbExtendedProps, 1, 1)
        outputLayout.addWidget(self.lblAnnotatedScale, 2, 0)
        outputLayout.addWidget(self.dsbAnnotatedScale, 2, 1)
        outputLayout.addWidget(self.lblSaveMode, 2, 2)
        outputLayout.addWidget(self.cmbSaveMode, 2, 3)
        outputLayout.setColumnStretch(3, 1)
        g.addWidget(self.outputGroup, row, 0, 1, 3)
        row += 1

        self.btnAdvancedToggle = QtWidgets.QToolButton()
        self.btnAdvancedToggle.setCheckable(True)
        self.btnAdvancedToggle.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
        self.btnAdvancedToggle.setStyleSheet("QToolButton { font-weight: 600; padding: 4px 0; }")
        self.btnAdvancedToggle.toggled.connect(self._set_advanced_visible)
        g.addWidget(self.btnAdvancedToggle, row, 0, 1, 3)
        row += 1

        self.advancedContainer = QtWidgets.QWidget()
        advancedLayout = QtWidgets.QVBoxLayout(self.advancedContainer)
        advancedLayout.setContentsMargins(0, 0, 0, 0)
        advancedLayout.setSpacing(8)

        self.diagnosticsGroup = QtWidgets.QGroupBox("Display overlays")
        diagnosticsLayout = QtWidgets.QHBoxLayout(self.diagnosticsGroup)
        diagnosticsLayout.addWidget(self.cbShowAdipocyteIds)
        diagnosticsLayout.addSpacing(8)
        diagnosticsLayout.addWidget(self.cbShowGrid)
        diagnosticsLayout.addStretch(1)
        advancedLayout.addWidget(self.diagnosticsGroup)

        self.roiAdvancedGroup = QtWidgets.QGroupBox("ROI details")
        roiAdvancedLayout = QtWidgets.QHBoxLayout(self.roiAdvancedGroup)
        roiAdvancedLayout.addWidget(QtWidgets.QLabel("ROI thumbnail max"))
        roiAdvancedLayout.addWidget(self.sbRoiMaxDim)
        roiAdvancedLayout.addSpacing(8)
        roiAdvancedLayout.addWidget(QtWidgets.QLabel("ROI min coverage"))
        roiAdvancedLayout.addWidget(self.dsbRoiMinCoverage)
        roiAdvancedLayout.addStretch(1)
        advancedLayout.addWidget(self.roiAdvancedGroup)

        self.batchGroup = QtWidgets.QGroupBox("Resume batch")
        batchLayout = QtWidgets.QGridLayout(self.batchGroup)

        self.leResumeBatch = QtWidgets.QLineEdit()
        self.leResumeBatch.setPlaceholderText("Path to batch state file (optional)")
        self.leResumeBatch.setToolTip(
            "Path to a batch_state.json file from a previous run.\n"
            "Use this to resume an interrupted analysis."
        )
        self.leResumeBatch.setReadOnly(True)

        self.btnResumeBatch = QtWidgets.QPushButton("Browse")
        self.btnResumeBatch.clicked.connect(self._choose_resume_batch)

        self.btnClearResume = QtWidgets.QPushButton("Clear")
        self.btnClearResume.setToolTip("Clear resume batch and restore normal settings")
        self.btnClearResume.clicked.connect(self._clear_resume_batch)
        self.btnClearResume.setEnabled(False)
        self.btnClearResume.setMaximumWidth(60)

        self.btnBatchInfo = QtWidgets.QPushButton("?")
        self.btnBatchInfo.setToolTip("View batch information")
        self.btnBatchInfo.clicked.connect(self._show_batch_info)
        self.btnBatchInfo.setEnabled(False)
        self.btnBatchInfo.setMaximumWidth(30)

        self.cbResumeFailed = QtWidgets.QCheckBox("Retry failed images")
        self.cbResumeFailed.setChecked(self.cfg.resume_failed)
        self.cbResumeFailed.setToolTip(
            "When resuming, also retry images that failed previously.\n"
            "Otherwise, only pending images are processed."
        )

        self.cbDryRun = QtWidgets.QCheckBox("Dry run (preview only)")
        self.cbDryRun.setChecked(self.cfg.dry_run)
        self.cbDryRun.setToolTip(
            "Show what would be processed without actually running analysis.\n"
            "Useful for verifying settings before a long run."
        )

        self.cbAllowResume = QtWidgets.QCheckBox("List resumable batches")
        self.cbAllowResume.setChecked(self.cfg.allow_resume)
        self.cbAllowResume.setToolTip(
            "List all batch state files that can be resumed.\n"
            "No input folder required when this is checked."
        )

        batchLayout.addWidget(QtWidgets.QLabel("Resume batch"), 0, 0)
        batchLayout.addWidget(self.leResumeBatch, 0, 1)

        batchBtnLayout = QtWidgets.QHBoxLayout()
        batchBtnLayout.addWidget(self.btnResumeBatch)
        batchBtnLayout.addWidget(self.btnClearResume)
        batchBtnLayout.addWidget(self.btnBatchInfo)
        batchLayout.addLayout(batchBtnLayout, 0, 2)

        batchOptsLayout = QtWidgets.QHBoxLayout()
        batchOptsLayout.addWidget(self.cbResumeFailed)
        batchOptsLayout.addSpacing(8)
        batchOptsLayout.addWidget(self.cbDryRun)
        batchOptsLayout.addSpacing(8)
        batchOptsLayout.addWidget(self.cbAllowResume)
        batchOptsLayout.addStretch(1)
        batchLayout.addLayout(batchOptsLayout, 1, 0, 1, 3)
        advancedLayout.addWidget(self.batchGroup)

        self.runtimeGroup = QtWidgets.QGroupBox("Runtime options")
        runtimeLayout = QtWidgets.QGridLayout(self.runtimeGroup)

        self.sbGpuId = QtWidgets.QSpinBox()
        self.sbGpuId.setRange(0, 16)
        self.sbGpuId.setValue(self.cfg.gpu_id)
        self.sbGpuId.setToolTip(
            "CUDA GPU device ID to use for inference.\n"
            "Use 0 for the first GPU, 1 for the second, etc."
        )

        self.cmbGpuMode = QtWidgets.QComboBox()
        self._populate_combo(self.cmbGpuMode, GPU_MODE_OPTIONS)
        _set_combo_to_value(self.cmbGpuMode, self.cfg.gpu_mode, DEFAULTS["gpu_mode"])
        self.cmbGpuMode.setToolTip("Choose how much of the GPU acceleration stack to use.")

        self.cmbMemoryMode = QtWidgets.QComboBox()
        self._populate_combo(self.cmbMemoryMode, MEMORY_MODE_OPTIONS)
        _set_combo_to_value(self.cmbMemoryMode, self.cfg.memory_mode, DEFAULTS["memory_mode"])
        self.cmbMemoryMode.setToolTip("Choose the memory-management mode for large image runs.")

        self.cmbDebugMode = QtWidgets.QComboBox()
        self._populate_combo(self.cmbDebugMode, DEBUG_MODE_OPTIONS)
        _set_combo_to_value(
            self.cmbDebugMode,
            _normalize_debug_mode(self.cfg.debug_mode),
            DEFAULTS["debug_mode"],
        )
        self.cmbDebugMode.setToolTip(
            "Choose whether to save processed debug outputs only or processed + raw windows."
        )

        self.cbVerbose = QtWidgets.QCheckBox("Verbose logging")
        self.cbVerbose.setChecked(self.cfg.verbose)
        self.cbVerbose.setToolTip("Show detailed startup diagnostics and extra runtime logging.")

        self.cbProfiling = QtWidgets.QCheckBox("Enable profiling")
        self.cbProfiling.setChecked(self.cfg.profiling)
        self.cbProfiling.setToolTip("Collect timing and profiling information during the run.")

        runtimeLayout.addWidget(QtWidgets.QLabel("GPU id"), 0, 0)
        runtimeLayout.addWidget(self.sbGpuId, 0, 1)
        runtimeLayout.addWidget(QtWidgets.QLabel("GPU mode"), 0, 2)
        runtimeLayout.addWidget(self.cmbGpuMode, 0, 3)
        runtimeLayout.addWidget(QtWidgets.QLabel("Memory mode"), 0, 4)
        runtimeLayout.addWidget(self.cmbMemoryMode, 0, 5)
        runtimeLayout.addWidget(QtWidgets.QLabel("Debug mode"), 1, 0)
        runtimeLayout.addWidget(self.cmbDebugMode, 1, 1)
        runtimeLayout.addWidget(self.cbVerbose, 1, 2)
        runtimeLayout.addWidget(self.cbProfiling, 1, 3)
        runtimeLayout.setColumnStretch(5, 1)
        advancedLayout.addWidget(self.runtimeGroup)

        g.addWidget(self.advancedContainer, row, 0, 1, 3)
        row += 1

        g.setRowStretch(row, 1)

        self.executionPane = QtWidgets.QWidget()
        self.executionPane.setMinimumHeight(220)
        self.executionPane.setSizePolicy(
            QtWidgets.QSizePolicy.Preferred,
            QtWidgets.QSizePolicy.MinimumExpanding,
        )
        executionLayout = QtWidgets.QVBoxLayout(self.executionPane)
        executionLayout.setContentsMargins(0, 0, 0, 0)
        executionLayout.setSpacing(8)

        executionTopRow = QtWidgets.QHBoxLayout()
        self.lblRunState = QtWidgets.QLabel("Ready")
        self.lblRunState.setObjectName("runStateLabel")
        self.lblRunState.setStyleSheet("font-weight: 600; color: #9ad0ff;")
        executionTopRow.addWidget(self.lblRunState)
        executionTopRow.addStretch(1)

        self.btnRun = QtWidgets.QPushButton("Run")
        self.btnStop = QtWidgets.QPushButton("Stop")
        self.btnStop.setEnabled(False)
        executionTopRow.addWidget(self.btnRun)
        executionTopRow.addWidget(self.btnStop)
        executionLayout.addLayout(executionTopRow)

        progressGrid = QtWidgets.QGridLayout()
        progressGrid.setHorizontalSpacing(8)
        progressGrid.setVerticalSpacing(8)

        self.lblBatch = QtWidgets.QLabel("Batch progress")
        self.pbBatch = QtWidgets.QProgressBar()
        self.pbBatch.setTextVisible(True)
        self.pbBatch.setMinimumHeight(20)

        self.lblImage = QtWidgets.QLabel("Current image: -")
        self.pbImage = QtWidgets.QProgressBar()
        self.pbImage.setTextVisible(True)
        self.pbImage.setMinimumHeight(20)

        progressGrid.addWidget(self.lblBatch, 0, 0)
        progressGrid.addWidget(self.pbBatch, 0, 1)
        progressGrid.addWidget(self.lblImage, 1, 0)
        progressGrid.addWidget(self.pbImage, 1, 1)
        progressGrid.setColumnStretch(1, 1)
        executionLayout.addLayout(progressGrid)

        self.btnRunDetailsToggle = QtWidgets.QToolButton()
        self.btnRunDetailsToggle.setCheckable(True)
        self.btnRunDetailsToggle.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
        self.btnRunDetailsToggle.setStyleSheet("QToolButton { font-weight: 600; padding: 4px 0; }")
        self.btnRunDetailsToggle.toggled.connect(self._set_run_details_visible)
        executionLayout.addWidget(self.btnRunDetailsToggle, 0, QtCore.Qt.AlignLeft)

        self.runDetailsContainer = QtWidgets.QWidget()
        runDetailsLayout = QtWidgets.QVBoxLayout(self.runDetailsContainer)
        runDetailsLayout.setContentsMargins(0, 0, 0, 0)

        self.teLog = QtWidgets.QPlainTextEdit()
        self.teLog.setReadOnly(True)
        mono = self.font()
        mono.setFamily("Consolas")
        self.teLog.setFont(mono)
        self.teLog.setMaximumBlockCount(5000)
        self.teLog.setMinimumHeight(150)
        runDetailsLayout.addWidget(self.teLog)
        executionLayout.addWidget(self.runDetailsContainer, 1)

        self.mainSplitter.addWidget(self.executionPane)

        self._set_idle_progress()
        self._set_advanced_visible(DEFAULTS["show_advanced"])
        self._set_run_details_visible(False)

        self.cbSaveAnnotatedImage.toggled.connect(self._update_annotated_output_state)
        self.cbTumor.toggled.connect(self._update_distance_map_state)

        self.status = QtWidgets.QStatusBar()
        self.setStatusBar(self.status)
        self.status.showMessage("Ready")

        menubar = self.menuBar()

        self.btnSettings = QtWidgets.QToolButton()
        self.btnSettings.setText("Settings")
        self.btnSettings.setToolTip("Configure Python environment and AdiFind settings")
        self.btnSettings.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        self.btnSettings.setStyleSheet(
            "QToolButton { padding: 2px 8px; border: 1px solid #5f6368; border-radius: 4px; }"
        )

        settingsMenu = QtWidgets.QMenu(self.btnSettings)
        self.btnSettings.setMenu(settingsMenu)

        actEnvSettings = QtGui.QAction("Environment settings", self)
        actEnvSettings.setToolTip("Configure the Python executable used for analysis")
        actEnvSettings.triggered.connect(self._show_environment_settings)
        settingsMenu.addAction(actEnvSettings)

        settingsMenu.addSeparator()

        actResetSettings = QtGui.QAction("Reset all settings", self)
        actResetSettings.setToolTip("Reset all settings to defaults")
        actResetSettings.triggered.connect(self._reset_all_settings)
        settingsMenu.addAction(actResetSettings)

        menubar.setCornerWidget(self.btnSettings, QtCore.Qt.TopRightCorner)

    def _set_advanced_visible(self, visible: bool):
        self.advancedContainer.setVisible(visible)
        self.btnAdvancedToggle.setChecked(visible)
        self.btnAdvancedToggle.setText(
            "Hide advanced options" if visible else "Show advanced options"
        )
        self.btnAdvancedToggle.setArrowType(
            QtCore.Qt.DownArrow if visible else QtCore.Qt.RightArrow
        )

    def _set_run_details_visible(self, visible: bool):
        self.runDetailsContainer.setVisible(visible)
        self.btnRunDetailsToggle.setChecked(visible)
        self.btnRunDetailsToggle.setText("Hide run details" if visible else "Show run details")
        self.btnRunDetailsToggle.setArrowType(
            QtCore.Qt.DownArrow if visible else QtCore.Qt.RightArrow
        )

    def _restore_splitter_sizes(self, sizes: Any) -> None:
        normalized = []
        if isinstance(sizes, (list, tuple)):
            normalized = [int(v) for v in sizes if int(v) > 0]
        elif sizes:
            try:
                normalized = [int(v) for v in list(sizes) if int(v) > 0]
            except Exception:
                normalized = []

        if len(normalized) == 2:
            self.mainSplitter.setSizes(normalized)
            return

        total = max(self.height(), 720)
        bottom = max(self.executionPane.minimumHeight(), 260)
        top = max(total - bottom, 320)
        self.mainSplitter.setSizes([top, bottom])

    def _set_run_state(self, state: str) -> None:
        self._run_state = state
        self.lblRunState.setText(state)

    def _update_batch_progress_display(self) -> None:
        if self.pbBatch.maximum() == 0:
            self.lblBatch.setText("Batch progress: discovering images...")
            self.pbBatch.setFormat("Discovering images...")
            return

        if self._batch_total_ui > 0:
            done = min(self._batch_done_ui, self._batch_total_ui)
            remaining = max(self._batch_total_ui - done, 0)
            percent = int((done / self._batch_total_ui) * 100)
            self.lblBatch.setText(
                f"Batch progress: {done} / {self._batch_total_ui} processed, {remaining} remaining"
            )
            self.pbBatch.setFormat(f"{done} / {self._batch_total_ui} images ({percent}%)")
        else:
            self.lblBatch.setText(f"Batch progress: {self._batch_done_ui} processed")
            self.pbBatch.setFormat(f"{self._batch_done_ui} / ? images")

    def _update_image_progress_display(self) -> None:
        if self._image_progress_busy:
            self.pbImage.setFormat("Preparing image windows...")
            return

        if self._image_total_ui > 0:
            done = min(self._image_done_ui, self._image_total_ui)
            percent = int((done / self._image_total_ui) * 100)
            self.pbImage.setFormat(f"{done} / {self._image_total_ui} windows ({percent}%)")
        else:
            self.pbImage.setFormat("Idle" if self._image_done_ui == 0 else f"{self._image_done_ui} / ? windows")

    def _last_gpu_stage_details(self) -> tuple[str, str]:
        if self.worker and getattr(self.worker, "_last_runtime_stage", ""):
            stage = str(getattr(self.worker, "_last_runtime_stage", "")).strip()
            if stage:
                return ("runtime", stage)
        if self.worker and getattr(self.worker, "_last_probe_stage", ""):
            stage = str(getattr(self.worker, "_last_probe_stage", "")).strip()
            if stage:
                return ("probe", stage)
        return ("", "")

    def _gpu_workaround_text(self) -> str:
        return (
            "Try this order: Disable GPU ops, Disable GPU preprocessing, then CPU only."
        )

    def _format_exit_message(self, code: int) -> str:
        stage_source, stage_key = self._last_gpu_stage_details()
        stage_suffix = ""
        if stage_key:
            stage_suffix = f"\nLast GPU {stage_source} stage: {describe_gpu_stage(stage_key)}"

        probe_error = ""
        if self.worker and getattr(self.worker, "_probe_error_message", ""):
            probe_error = f"\nProbe detail: {getattr(self.worker, '_probe_error_message')}"

        if self.worker and getattr(self.worker, "_probe_failed", False):
            base = f"GPU probe failed (exit {code})"
            if code == -1073741819:
                base = "GPU probe failed (Windows access violation / exit -1073741819)"
            return f"{base}{stage_suffix}{probe_error}\n{self._gpu_workaround_text()}"

        if code == -1073741819:
            return (
                "Finished with errors (Windows access violation / exit -1073741819)"
                f"{stage_suffix}\n{self._gpu_workaround_text()}"
            )
        return f"Finished with errors (exit {code})"

    def _update_annotated_output_state(self):
        enabled = self.cbSaveAnnotatedImage.isChecked() and not self._resume_mode
        self.lblAnnotatedScale.setEnabled(enabled)
        self.dsbAnnotatedScale.setEnabled(enabled)
        self.lblSaveMode.setEnabled(enabled)
        self.cmbSaveMode.setEnabled(enabled)

    def _update_distance_map_state(self):
        tumor_enabled = self.cbTumor.isChecked()
        if not tumor_enabled and not self._resume_mode:
            self.cbSaveDistanceMap.setChecked(False)
        self.cbSaveDistanceMap.setEnabled(tumor_enabled and not self._resume_mode)
        self.cbSaveDistanceMap.setToolTip(
            self._save_distance_map_tooltip if tumor_enabled else self._save_distance_map_disabled_tooltip
        )

    def _sync_contextual_state(self):
        self._on_roi_toggled(self.cbRoiFreehand.isChecked())
        self._update_annotated_output_state()
        self._update_distance_map_state()

    def _should_expand_run_details(self, line: str) -> bool:
        text = str(line or "").lower()
        return any(token in text for token in ("error", "warning", "traceback", "failed"))

    def _wire_events(self):
        self.btnInputFile.clicked.connect(self._choose_input_file)
        self.btnInputFolder.clicked.connect(self._choose_input_folder)
        self.btnOutput.clicked.connect(self._choose_output)
        self.btnRun.clicked.connect(self._on_run)
        self.btnStop.clicked.connect(self._on_stop)

    # ---------- Settings ----------
    def _load_settings(self):
        st = QtCore.QSettings("adi", "adifind_desktop")

        # Load Python executable path (hidden from main UI)
        stored_python = st.value("python_exe_path", DEFAULT_PYTHON_EXE)
        try:
            stored_python = str(stored_python or "").strip()
        except Exception:
            stored_python = ""
        self._python_exe_path = stored_python if stored_python else DEFAULT_PYTHON_EXE
        st.remove("script_path")

        self.leInput.setText(st.value("image_path", self.leInput.text()))
        self.leOutput.setText(st.value("output_dir", self.leOutput.text()))
        self.cbTumor.setChecked(st.value("enable_tumor", self.cbTumor.isChecked(), type=bool))
        self.cbTissue.setChecked(st.value("enable_tissue", self.cbTissue.isChecked(), type=bool))
        
        # Numeric controls
        try:
            self.sbBatchSize.setValue(st.value("batch_size", self.sbBatchSize.value(), type=int))
        except Exception:
            pass
        try:
            self.sbMinArea.setValue(st.value("min_area_um2", self.sbMinArea.value(), type=int))
        except Exception:
            pass
        try:
            self.sbMaxArea.setValue(st.value("max_area_um2", self.sbMaxArea.value(), type=int))
        except Exception:
            pass

        # Debug/runtime settings
        _set_combo_to_value(
            self.cmbDebugMode,
            _normalize_debug_mode(st.value("debug_mode", self.cmbDebugMode.currentData())),
            DEFAULTS["debug_mode"],
        )
        try:
            self.cbVerbose.setChecked(st.value("verbose", self.cbVerbose.isChecked(), type=bool))
        except Exception:
            pass
        try:
            self.cbProfiling.setChecked(st.value("profiling", self.cbProfiling.isChecked(), type=bool))
        except Exception:
            pass
        try:
            self.sbGpuId.setValue(st.value("gpu_id", self.sbGpuId.value(), type=int))
        except Exception:
            pass
        _set_combo_to_value(
            self.cmbGpuMode,
            st.value("gpu_mode", self.cmbGpuMode.currentData()),
            DEFAULTS["gpu_mode"],
        )
        _set_combo_to_value(
            self.cmbMemoryMode,
            st.value("memory_mode", self.cmbMemoryMode.currentData()),
            DEFAULTS["memory_mode"],
        )

        # Output settings
        try:
            self.dsbAnnotatedScale.setValue(float(st.value("annotated_scale", self.dsbAnnotatedScale.value())))
        except Exception:
            pass
        _set_combo_to_value(
            self.cmbSaveMode,
            _normalize_desktop_save_mode(st.value("save_mode", self.cmbSaveMode.currentData())),
            DEFAULTS["save_mode"],
        )
        try:
            self.cbSaveDistanceMap.setChecked(st.value("save_distance_map", self.cbSaveDistanceMap.isChecked(), type=bool))
        except Exception:
            pass
        try:
            self.cbExtendedProps.setChecked(st.value("extended_properties", self.cbExtendedProps.isChecked(), type=bool))
        except Exception:
            pass
        try:
            self.cbSaveAnnotatedImage.setChecked(
                st.value("save_image_annotation", self.cbSaveAnnotatedImage.isChecked(), type=bool)
            )
        except Exception:
            pass
        try:
            self.cbSaveQuPathAnnotation.setChecked(
                st.value("save_qupath_annotation", self.cbSaveQuPathAnnotation.isChecked(), type=bool)
            )
        except Exception:
            pass
        try:
            self.cbSaveTissueWindowGrid.setChecked(
                st.value("save_tissue_window_grid", self.cbSaveTissueWindowGrid.isChecked(), type=bool)
            )
        except Exception:
            pass

        # Visualization toggles
        try:
            self.cbShowAdipocyteIds.setChecked(st.value("show_adipocyte_ids", self.cbShowAdipocyteIds.isChecked(), type=bool))
        except Exception:
            pass
        try:
            self.cbShowGrid.setChecked(st.value("show_grid", self.cbShowGrid.isChecked(), type=bool))
        except Exception:
            pass

        # ROI guidance
        try:
            self.cbRoiFreehand.setChecked(st.value("roi_freehand", self.cbRoiFreehand.isChecked(), type=bool))
        except Exception:
            pass
        try:
            self.sbRoiMaxDim.setValue(st.value("roi_max_dim", self.sbRoiMaxDim.value(), type=int))
        except Exception:
            pass
        try:
            self.dsbRoiMinCoverage.setValue(
                float(st.value("roi_min_coverage", self.dsbRoiMinCoverage.value()))
            )
        except Exception:
            pass
        # Note: ROI polygon file is NOT restored on app start (session-only, like resume_batch)
        self._on_roi_toggled(self.cbRoiFreehand.isChecked())

        # Batch resume options
        # Note: We don't restore resume mode on app start - user must re-select batch file
        # The resume path is intentionally NOT restored to prevent stale state issues
        try:
            self.cbResumeFailed.setChecked(st.value("resume_failed", self.cbResumeFailed.isChecked(), type=bool))
        except Exception:
            pass
        try:
            self.cbDryRun.setChecked(st.value("dry_run", self.cbDryRun.isChecked(), type=bool))
        except Exception:
            pass
        try:
            self.cbAllowResume.setChecked(st.value("allow_resume", self.cbAllowResume.isChecked(), type=bool))
        except Exception:
            pass
        try:
            show_advanced = st.value("show_advanced", DEFAULTS["show_advanced"], type=bool)
        except Exception:
            show_advanced = DEFAULTS["show_advanced"]
        self._set_advanced_visible(bool(show_advanced))
        self._restore_splitter_sizes(st.value("main_splitter_sizes", []))
        self._sync_contextual_state()

    def _save_settings(self):
        st = QtCore.QSettings("adi", "adifind_desktop")
        st.setValue("python_exe_path", self._python_exe_path)
        st.setValue("image_path", self.leInput.text())
        st.setValue("output_dir", self.leOutput.text())
        st.setValue("enable_tumor", self.cbTumor.isChecked())
        st.setValue("enable_tissue", self.cbTissue.isChecked())
        st.setValue("batch_size", self.sbBatchSize.value())
        st.setValue("min_area_um2", self.sbMinArea.value())
        st.setValue("max_area_um2", self.sbMaxArea.value())

        st.setValue("debug_mode", self.cmbDebugMode.currentData())
        st.setValue("verbose", self.cbVerbose.isChecked())
        st.setValue("profiling", self.cbProfiling.isChecked())
        st.setValue("gpu_id", self.sbGpuId.value())
        st.setValue("gpu_mode", self.cmbGpuMode.currentData())
        st.setValue("memory_mode", self.cmbMemoryMode.currentData())

        st.setValue("annotated_scale", self.dsbAnnotatedScale.value())
        st.setValue("save_mode", self.cmbSaveMode.currentData())
        st.setValue("save_distance_map", self.cbSaveDistanceMap.isChecked())
        st.setValue("extended_properties", self.cbExtendedProps.isChecked())
        st.setValue("save_image_annotation", self.cbSaveAnnotatedImage.isChecked())
        st.setValue("save_qupath_annotation", self.cbSaveQuPathAnnotation.isChecked())
        st.setValue("save_tissue_window_grid", self.cbSaveTissueWindowGrid.isChecked())
        # Visualization toggles
        st.setValue("show_adipocyte_ids", self.cbShowAdipocyteIds.isChecked())
        st.setValue("show_grid", self.cbShowGrid.isChecked())
        # ROI guidance
        st.setValue("roi_freehand", self.cbRoiFreehand.isChecked())
        st.setValue("roi_max_dim", self.sbRoiMaxDim.value())
        st.setValue("roi_min_coverage", self.dsbRoiMinCoverage.value())
        st.setValue("show_advanced", self.btnAdvancedToggle.isChecked())
        st.setValue("main_splitter_sizes", self.mainSplitter.sizes())
        # Note: ROI polygon file path is session-only, not persisted
        # Batch resume options (don't save resume_batch path - it's session-only)
        st.setValue("resume_failed", self.cbResumeFailed.isChecked())
        st.setValue("dry_run", self.cbDryRun.isChecked())
        st.setValue("allow_resume", self.cbAllowResume.isChecked())

        # Remove stale settings from the retired manual window/stride controls.
        for key in ("window_w", "window_h", "stride_x", "stride_y"):
            st.remove(key)
        st.remove("script_path")
        st.remove("benchmark_saving")

    # ---------- File pickers ----------
    def _build_environment_settings_dialog(self):
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Environment Settings")
        dialog.setMinimumWidth(600)

        layout = QtWidgets.QVBoxLayout(dialog)

        # Info label
        infoLabel = QtWidgets.QLabel(
            "Configure the Python environment used to run analysis.\n"
            "The analysis script is fixed to this AdiFind checkout."
        )
        infoLabel.setWordWrap(True)
        layout.addWidget(infoLabel)

        # Python executable
        pyLayout = QtWidgets.QHBoxLayout()
        pyLabel = QtWidgets.QLabel("Python executable:")
        pyLabel.setMinimumWidth(120)
        lePy = QtWidgets.QLineEdit(self._python_exe_path)
        lePy.setToolTip("Path to Python interpreter in your adifind conda environment")
        btnPy = QtWidgets.QPushButton("Browse")

        def choose_python():
            fn, _ = QtWidgets.QFileDialog.getOpenFileName(
                dialog, "Select python.exe", lePy.text() or "",
                "Executables (*.exe);;All files (*)"
            )
            if fn:
                lePy.setText(fn)

        btnPy.clicked.connect(choose_python)
        pyLayout.addWidget(pyLabel)
        pyLayout.addWidget(lePy)
        pyLayout.addWidget(btnPy)
        layout.addLayout(pyLayout)

        # Analysis script is fixed to the current repo checkout.
        scriptLayout = QtWidgets.QHBoxLayout()
        scriptLabel = QtWidgets.QLabel("Analysis script:")
        scriptLabel.setMinimumWidth(120)
        leScript = QtWidgets.QLineEdit(DEFAULT_SCRIPT_PATH)
        leScript.setObjectName("environmentScriptPath")
        leScript.setReadOnly(True)
        leScript.setToolTip("Repo-local analysis script used for every desktop run")
        scriptLayout.addWidget(scriptLabel)
        scriptLayout.addWidget(leScript)
        layout.addLayout(scriptLayout)

        # Reset to defaults button
        resetLayout = QtWidgets.QHBoxLayout()
        btnReset = QtWidgets.QPushButton("Reset to Defaults")
        btnReset.setToolTip("Reset paths to default values")

        def reset_defaults():
            lePy.setText(DEFAULT_PYTHON_EXE)

        btnReset.clicked.connect(reset_defaults)
        resetLayout.addWidget(btnReset)
        resetLayout.addStretch(1)
        layout.addLayout(resetLayout)

        layout.addSpacing(10)

        # OK/Cancel buttons
        btnBox = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        btnBox.accepted.connect(dialog.accept)
        btnBox.rejected.connect(dialog.reject)
        layout.addWidget(btnBox)

        dialog._pythonPathEdit = lePy  # type: ignore[attr-defined]
        dialog._scriptPathEdit = leScript  # type: ignore[attr-defined]
        return dialog

    def _show_environment_settings(self):
        """Show dialog to configure the Python executable used for analysis."""
        dialog = self._build_environment_settings_dialog()
        if dialog.exec() == QtWidgets.QDialog.Accepted:
            lePy = dialog._pythonPathEdit  # type: ignore[attr-defined]
            self._python_exe_path = lePy.text().strip() or DEFAULT_PYTHON_EXE
            self._save_settings()
            self.status.showMessage("Environment settings updated", 3000)

    def _reset_all_settings(self):
        """Reset all settings to defaults after confirmation."""
        resp = QtWidgets.QMessageBox.question(
            self,
            "Reset Settings",
            "Are you sure you want to reset all settings to defaults?\n\n"
            "This will clear saved paths and restore default options.",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No,
        )
        if resp != QtWidgets.QMessageBox.Yes:
            return
        
        # Clear QSettings
        st = QtCore.QSettings("adi", "adifind_desktop")
        st.clear()
        
        # Reset instance variables and UI to defaults
        self._python_exe_path = DEFAULT_PYTHON_EXE
        self.leInput.clear()
        self.leOutput.clear()
        self.cbTumor.setChecked(DEFAULTS["enable_tumor"])
        self.cbTissue.setChecked(DEFAULTS["enable_tissue"])
        self.sbBatchSize.setValue(DEFAULTS["batch_size"] or 6)
        self.sbMinArea.setValue(DEFAULTS["min_area_um2"])
        self.sbMaxArea.setValue(DEFAULTS["max_area_um2"])
        _set_combo_to_value(self.cmbDebugMode, DEFAULTS["debug_mode"], DEFAULTS["debug_mode"])
        self.cbVerbose.setChecked(DEFAULTS["verbose"])
        self.cbProfiling.setChecked(DEFAULTS["profiling"])
        self.sbGpuId.setValue(DEFAULTS["gpu_id"])
        _set_combo_to_value(self.cmbGpuMode, DEFAULTS["gpu_mode"], DEFAULTS["gpu_mode"])
        _set_combo_to_value(self.cmbMemoryMode, DEFAULTS["memory_mode"], DEFAULTS["memory_mode"])
        self.dsbAnnotatedScale.setValue(DEFAULTS["annotated_scale"])
        _set_combo_to_value(self.cmbSaveMode, DEFAULTS["save_mode"], DEFAULTS["save_mode"])
        self.cbSaveDistanceMap.setChecked(DEFAULTS["save_distance_map"])
        self.cbExtendedProps.setChecked(DEFAULTS["extended_properties"])
        self.cbSaveAnnotatedImage.setChecked(DEFAULTS["save_image_annotation"])
        self.cbSaveQuPathAnnotation.setChecked(DEFAULTS["save_qupath_annotation"])
        self.cbSaveTissueWindowGrid.setChecked(DEFAULTS["save_tissue_window_grid"])
        self.cbShowAdipocyteIds.setChecked(DEFAULTS["show_adipocyte_ids"])
        self.cbShowGrid.setChecked(DEFAULTS["show_grid"])
        self.cbRoiFreehand.setChecked(DEFAULTS["roi_freehand"])
        self.sbRoiMaxDim.setValue(DEFAULTS["roi_max_dim"])
        self.dsbRoiMinCoverage.setValue(DEFAULTS["roi_min_coverage"])
        self.leRoiPolygonFile.clear()
        self.leResumeBatch.clear()
        self.cbResumeFailed.setChecked(DEFAULTS["resume_failed"])
        self.cbDryRun.setChecked(DEFAULTS["dry_run"])
        self.cbAllowResume.setChecked(DEFAULTS["allow_resume"])
        self._resume_benchmark_saving_override = None
        self._set_advanced_visible(DEFAULTS["show_advanced"])
        self._set_run_details_visible(False)
        self._restore_splitter_sizes([])
        self._sync_contextual_state()

        # Clear resume mode if active
        if self._resume_mode:
            self._resume_batch_state = None
            self._set_resume_mode(False)
        
        self.status.showMessage("All settings reset to defaults", 3000)

    # ---------- ROI helpers ----------
    def _on_roi_toggled(self, checked: bool):
        """Enable/disable the ROI polygon file widgets based on ROI checkbox."""
        editable = checked and not self._resume_mode
        self.leRoiPolygonFile.setEnabled(editable)
        self.btnLoadRoi.setEnabled(editable)
        self.btnClearRoi.setEnabled(editable and bool(self.leRoiPolygonFile.text()))

    def _choose_roi_polygon_file(self):
        """Open file dialog to select a saved ROI polygon JSON."""
        start_dir = os.path.dirname(self.leRoiPolygonFile.text()) if self.leRoiPolygonFile.text() else ""
        if not start_dir and self.leInput.text():
            start_dir = os.path.dirname(self.leInput.text())
        fn, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select ROI polygon file", start_dir,
            "ROI Polygon Files (*.json);;All Files (*)"
        )
        if fn:
            self.leRoiPolygonFile.setText(fn)
            self.btnClearRoi.setEnabled(True)
            self.status.showMessage(f"ROI polygon loaded: {os.path.basename(fn)}", 3000)

    def _clear_roi_polygon_file(self):
        """Clear the loaded ROI polygon file."""
        self.leRoiPolygonFile.clear()
        self.btnClearRoi.setEnabled(False)

    def _choose_input_file(self):
        """Open file dialog to select a single image file."""
        # Supported WSI and standard image formats
        filters = (
            "Whole Slide Images (*.svs *.ndpi *.tiff *.tif *.vms *.vmu *.scn *.mrxs *.svslide *.bif *.czi);;"
            "Standard Images (*.png *.jpg *.jpeg);;"
            "All Files (*)"
        )
        start_dir = self.leInput.text() or ""
        # If current value is a file, start in its directory
        if start_dir and os.path.isfile(start_dir):
            start_dir = os.path.dirname(start_dir)
        
        fn, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select image file", start_dir, filters
        )
        if fn:
            self.leInput.setText(fn)

    def _choose_input_folder(self):
        """Open folder dialog to select a folder containing images."""
        start_dir = self.leInput.text() or ""
        # If current value is a file, start in its directory
        if start_dir and os.path.isfile(start_dir):
            start_dir = os.path.dirname(start_dir)
        
        d = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select folder with images", start_dir
        )
        if d:
            self.leInput.setText(d)

    def _choose_output(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "Select output folder", self.leOutput.text() or "")
        if d:
            self.leOutput.setText(d)

    def _choose_resume_batch(self):
        """Open file dialog to select a batch state JSON file for resuming."""
        fn, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select batch state file",
            self.leResumeBatch.text() or self.leOutput.text() or "",
            "JSON files (*.json);;All files (*)"
        )
        if fn:
            # Try to load and validate the batch state
            try:
                batch_state = self._load_batch_state(fn)
            except ValueError as e:
                QtWidgets.QMessageBox.warning(
                    self,
                    "Invalid batch state",
                    str(e),
                )
                return
            if batch_state is None:
                QtWidgets.QMessageBox.warning(
                    self,
                    "Invalid batch state",
                    f"Could not load batch state from:\n{fn}\n\n"
                    "The file may be corrupted or not a valid batch_state.json file."
                )
                return
            
            # Show warning about strict restore
            batch_id = batch_state.get("batch_id", "Unknown")
            total = batch_state.get("total_images", 0)
            processed = batch_state.get("processed_images", 0)
            remaining = total - processed
            
            msg = (
                f"Resume batch: {batch_id}\n\n"
                f"Progress: {processed}/{total} images processed\n"
                f"Remaining: {remaining} images\n\n"
                "When resuming a batch, all analysis settings will be restored "
                "from the original run and locked to ensure consistent results.\n\n"
                "Do you want to load this batch?"
            )
            
            resp = QtWidgets.QMessageBox.question(
                self,
                "Resume Batch",
                msg,
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                QtWidgets.QMessageBox.Yes,
            )
            
            if resp != QtWidgets.QMessageBox.Yes:
                return
            
            # Accept the batch state
            self._resume_batch_state = batch_state
            self.leResumeBatch.setText(fn)
            self._apply_batch_state_settings(batch_state)
            self._set_resume_mode(True)
            
            self.status.showMessage(f"Loaded batch: {batch_id} ({remaining} images remaining)", 5000)

    def _load_batch_state(self, path: str) -> Optional[Dict[str, Any]]:
        """Load and validate a batch state JSON file."""
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # Basic validation - check for required fields
            required = ["batch_id", "total_images", "processed_images"]
            if not all(k in data for k in required):
                return None

            args = data.get("args_snapshot", {}) or {}
            _require_supported_save_mode(
                args.get("save_mode", DEFAULTS["save_mode"]),
                "batch state",
            )
            
            return data
        except ValueError:
            raise
        except Exception:
            return None

    def _apply_batch_state_settings(self, batch_state: Dict[str, Any]):
        """Apply settings from a batch state to the UI controls."""
        args = batch_state.get("args_snapshot", {}) or {}
        config = batch_state.get("config_snapshot", {}) or {}
        self._resume_benchmark_saving_override = None

        if args:
            if args.get("min_area") is not None:
                self.sbMinArea.setValue(int(args["min_area"]))
            if args.get("max_area") is not None:
                self.sbMaxArea.setValue(int(args["max_area"]))

            self.cbTissue.setChecked(bool(args.get("tissue_guidance", self.cbTissue.isChecked())))
            self.cbTumor.setChecked(bool(args.get("tumor_segmentation", self.cbTumor.isChecked())))
            _set_combo_to_value(
                self.cmbDebugMode,
                _normalize_debug_mode(args.get("debug", self.cmbDebugMode.currentData())),
                DEFAULTS["debug_mode"],
            )
            self.cbVerbose.setChecked(bool(args.get("verbose", self.cbVerbose.isChecked())))
            self.cbProfiling.setChecked(bool(args.get("profiling", self.cbProfiling.isChecked())))

            self.cbShowAdipocyteIds.setChecked(
                bool(args.get("show_adipocyte_ids", self.cbShowAdipocyteIds.isChecked()))
            )
            self.cbShowGrid.setChecked(bool(args.get("show_grid", self.cbShowGrid.isChecked())))

            if args.get("annotated_scale") is not None:
                self.dsbAnnotatedScale.setValue(float(args["annotated_scale"]))
            raw_save_mode = _require_supported_save_mode(
                args.get("save_mode", DEFAULTS["save_mode"]),
                "batch state",
            )
            _set_combo_to_value(
                self.cmbSaveMode,
                raw_save_mode,
                DEFAULTS["save_mode"],
            )
            self.cbSaveDistanceMap.setChecked(
                bool(args.get("save_distance_map", self.cbSaveDistanceMap.isChecked()))
            )
            self.cbExtendedProps.setChecked(
                bool(args.get("extended_properties", self.cbExtendedProps.isChecked()))
            )
            self.cbSaveAnnotatedImage.setChecked(
                _resolve_cli_override(
                    args.get("save_image_annotation"),
                    args.get("skip_image_annotation"),
                    DEFAULTS["save_image_annotation"],
                )
            )
            self.cbSaveQuPathAnnotation.setChecked(
                _resolve_cli_override(
                    args.get("save_qupath_annotation"),
                    args.get("skip_qupath_annotation"),
                    DEFAULTS["save_qupath_annotation"],
                )
            )
            self.cbSaveTissueWindowGrid.setChecked(
                bool(args.get("save_tissue_window_grid", self.cbSaveTissueWindowGrid.isChecked()))
            )
            if "benchmark_saving" in args:
                self._resume_benchmark_saving_override = bool(args.get("benchmark_saving"))

            if args.get("gpu_id") is not None:
                self.sbGpuId.setValue(int(args["gpu_id"]))
            _set_combo_to_value(self.cmbGpuMode, _gpu_mode_from_args(args), DEFAULTS["gpu_mode"])
            _set_combo_to_value(self.cmbMemoryMode, _memory_mode_from_args(args), DEFAULTS["memory_mode"])
            if "batch_size" in args:
                self.sbBatchSize.setValue(int(args["batch_size"]) if args["batch_size"] else 1)

            self.cbRoiFreehand.setChecked(bool(args.get("roi_freehand", self.cbRoiFreehand.isChecked())))
            self.leRoiPolygonFile.setText(str(args.get("roi_polygon_file") or "").strip())
            if args.get("roi_max_dim") is not None:
                self.sbRoiMaxDim.setValue(int(args["roi_max_dim"]))
            if args.get("roi_min_coverage") is not None:
                self.dsbRoiMinCoverage.setValue(float(args["roi_min_coverage"]))

            if args.get("output_dir"):
                self.leOutput.setText(args["output_dir"])

        if config:
            if (
                not args.get("save_image_annotation")
                and not args.get("skip_image_annotation")
                and "SAVE_ANNOTATED_IMAGE" in config
            ):
                self.cbSaveAnnotatedImage.setChecked(bool(config["SAVE_ANNOTATED_IMAGE"]))
            if (
                not args.get("save_qupath_annotation")
                and not args.get("skip_qupath_annotation")
                and ("ENABLE_QUPATH_EXPORT" in config or "SAVE_QUPATH_GEOJSON" in config)
            ):
                self.cbSaveQuPathAnnotation.setChecked(
                    bool(
                        config.get("ENABLE_QUPATH_EXPORT", False)
                        and config.get("SAVE_QUPATH_GEOJSON", False)
                    )
                )

        self.btnClearRoi.setEnabled(self.cbRoiFreehand.isChecked() and bool(self.leRoiPolygonFile.text()))
        self._sync_contextual_state()

    def _set_resume_mode(self, enabled: bool):
        """Enable or disable resume mode, locking/unlocking settings controls."""
        self._resume_mode = enabled
        
        # List of controls to lock when in resume mode
        settings_controls = [
            self.sbBatchSize, self.sbMinArea, self.sbMaxArea,
            self.cbTumor, self.cbTissue,
            self.cbShowAdipocyteIds, self.cbShowGrid,
            self.cbRoiFreehand, self.leRoiPolygonFile, self.btnLoadRoi, self.btnClearRoi,
            self.sbRoiMaxDim, self.dsbRoiMinCoverage,
            self.cbSaveAnnotatedImage, self.cbSaveQuPathAnnotation,
            self.cbSaveTissueWindowGrid, self.cbSaveDistanceMap,
            self.cbExtendedProps,
            self.dsbAnnotatedScale, self.cmbSaveMode,
            self.sbGpuId, self.cmbGpuMode, self.cmbMemoryMode,
            self.cmbDebugMode, self.cbVerbose, self.cbProfiling,
        ]
        
        # Enable/disable controls
        for ctrl in settings_controls:
            ctrl.setEnabled(not enabled)
        
        # Update group box titles to indicate locked state
        if enabled:
            self.basicGroup.setTitle("Analysis basics (locked from batch)")
            self.processingGroup.setTitle("Analysis options (locked from batch)")
            self.outputGroup.setTitle("Results to save (locked from batch)")
            self.diagnosticsGroup.setTitle("Display overlays (locked from batch)")
            self.roiAdvancedGroup.setTitle("ROI details (locked from batch)")
            self.batchGroup.setTitle("Resume batch")
            self.runtimeGroup.setTitle("Runtime options (locked from batch)")
        else:
            self.basicGroup.setTitle("Analysis basics")
            self.processingGroup.setTitle("Analysis options")
            self.outputGroup.setTitle("Results to save")
            self.diagnosticsGroup.setTitle("Display overlays")
            self.roiAdvancedGroup.setTitle("ROI details")
            self.batchGroup.setTitle("Resume batch")
            self.runtimeGroup.setTitle("Runtime options")
        
        # Update clear/info button states
        self.btnClearResume.setEnabled(enabled)
        self.btnBatchInfo.setEnabled(enabled)
        
        # Input folder is optional when resuming (batch has the paths)
        if enabled:
            self.leInput.setPlaceholderText("Optional when resuming batch")
            self._set_advanced_visible(True)
        else:
            self.leInput.setPlaceholderText("Select image file or folder (required)")
        self._sync_contextual_state()

    def _clear_resume_batch(self):
        """Clear the resume batch and restore normal editing mode."""
        if not self._resume_mode:
            return
        
        resp = QtWidgets.QMessageBox.question(
            self,
            "Clear Resume",
            "This will clear the batch resume and unlock settings.\n\n"
            "Your current settings will be preserved. Continue?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.Yes,
        )
        
        if resp != QtWidgets.QMessageBox.Yes:
            return
        
        self._resume_batch_state = None
        self._resume_benchmark_saving_override = None
        self.leResumeBatch.clear()
        self._set_resume_mode(False)
        self.status.showMessage("Resume batch cleared", 3000)

    def _show_batch_info(self):
        """Show detailed information about the loaded batch state."""
        if not self._resume_batch_state:
            return
        
        bs = self._resume_batch_state
        args = bs.get("args_snapshot", {})
        
        # Build info text
        batch_id = bs.get("batch_id", "Unknown")
        start_time = bs.get("start_time", "Unknown")
        total = bs.get("total_images", 0)
        processed = bs.get("processed_images", 0)
        successful = bs.get("successful_images", 0)
        failed = bs.get("failed_images", 0)
        remaining = total - processed

        min_area = args.get("min_area", 250)
        max_area = args.get("max_area", 25000)
        tissue = "Yes" if args.get("tissue_guidance") else "No"
        tumor = "Yes" if args.get("tumor_segmentation") else "No"
        debug_mode = _mode_label(
            DEBUG_MODE_OPTIONS,
            _normalize_debug_mode(args.get("debug", DEFAULTS["debug_mode"])),
        )
        gpu_mode = _mode_label(GPU_MODE_OPTIONS, _gpu_mode_from_args(args))
        memory_mode = _mode_label(MEMORY_MODE_OPTIONS, _memory_mode_from_args(args))
        save_mode_key = _require_supported_save_mode(
            args.get("save_mode", DEFAULTS["save_mode"]),
            "batch state",
        )
        save_mode = _mode_label(SAVE_MODE_OPTIONS, save_mode_key)
        save_annotated = "Yes" if _resolve_cli_override(
            args.get("save_image_annotation"),
            args.get("skip_image_annotation"),
            DEFAULTS["save_image_annotation"],
        ) else "No"
        save_qupath = "Yes" if _resolve_cli_override(
            args.get("save_qupath_annotation"),
            args.get("skip_qupath_annotation"),
            DEFAULTS["save_qupath_annotation"],
        ) else "No"
        save_grid = "Yes" if args.get("save_tissue_window_grid") else "No"
        save_distance = "Yes" if args.get("save_distance_map") else "No"
        extended_props = "Yes" if args.get("extended_properties") else "No"
        verbose = "Yes" if args.get("verbose") else "No"
        profiling = "Yes" if args.get("profiling") else "No"
        roi_max_dim = args.get("roi_max_dim", DEFAULTS["roi_max_dim"])
        roi_min_coverage = float(args.get("roi_min_coverage", DEFAULTS["roi_min_coverage"]))
        benchmark_row = ""
        if "benchmark_saving" in args:
            benchmark_saving = "Yes" if args.get("benchmark_saving") else "No"
            benchmark_row = (
                f"<tr><td><b>Benchmark saving (legacy):</b></td><td>{benchmark_saving}</td></tr>"
            )
        
        info = f"""<h3>Batch Information</h3>
<table>
<tr><td><b>Batch ID:</b></td><td>{batch_id}</td></tr>
<tr><td><b>Started:</b></td><td>{start_time}</td></tr>
<tr><td colspan="2"><hr></td></tr>
<tr><td><b>Total images:</b></td><td>{total}</td></tr>
<tr><td><b>Processed:</b></td><td>{processed}</td></tr>
<tr><td><b>Successful:</b></td><td>{successful}</td></tr>
<tr><td><b>Failed:</b></td><td>{failed}</td></tr>
<tr><td><b>Remaining:</b></td><td>{remaining}</td></tr>
<tr><td colspan="2"><hr></td></tr>
<tr><td><b>Windowing:</b></td><td>Automatic from slide metadata</td></tr>
<tr><td><b>Min area:</b></td><td>{min_area} \u00B5m\u00B2</td></tr>
<tr><td><b>Max area:</b></td><td>{max_area} \u00B5m\u00B2</td></tr>
<tr><td><b>Tissue guidance:</b></td><td>{tissue}</td></tr>
<tr><td><b>Tumor segmentation:</b></td><td>{tumor}</td></tr>
<tr><td><b>Save annotated image:</b></td><td>{save_annotated}</td></tr>
<tr><td><b>Save QuPath annotations:</b></td><td>{save_qupath}</td></tr>
<tr><td><b>Save tissue window grid:</b></td><td>{save_grid}</td></tr>
<tr><td><b>Save distance map:</b></td><td>{save_distance}</td></tr>
<tr><td><b>Extended properties:</b></td><td>{extended_props}</td></tr>
<tr><td><b>Save mode:</b></td><td>{save_mode}</td></tr>
<tr><td><b>Debug mode:</b></td><td>{debug_mode}</td></tr>
<tr><td><b>Verbose logging:</b></td><td>{verbose}</td></tr>
<tr><td><b>Profiling:</b></td><td>{profiling}</td></tr>
<tr><td><b>GPU id:</b></td><td>{args.get("gpu_id", DEFAULTS["gpu_id"])}</td></tr>
<tr><td><b>GPU mode:</b></td><td>{gpu_mode}</td></tr>
<tr><td><b>Memory mode:</b></td><td>{memory_mode}</td></tr>
<tr><td><b>ROI thumbnail max:</b></td><td>{roi_max_dim}</td></tr>
<tr><td><b>ROI min coverage:</b></td><td>{roi_min_coverage:.2f}</td></tr>
{benchmark_row}
</table>
"""
        
        msgBox = QtWidgets.QMessageBox(self)
        msgBox.setWindowTitle("Batch Information")
        msgBox.setTextFormat(QtCore.Qt.RichText)
        msgBox.setText(info)
        msgBox.setIcon(QtWidgets.QMessageBox.Information)
        msgBox.exec()

    # ---------- Run control ----------
    def _collect_cfg(self) -> RunConfig:
        # Handle batch_size - if at minimum (1), treat as "Auto" (None)
        batch_size_val = self.sbBatchSize.value()
        if batch_size_val == 1:
            batch_size_val = None  # Use config.py default
        save_mode = _require_supported_save_mode(
            self.cmbSaveMode.currentData(),
            "desktop run configuration",
        )
        benchmark_saving = False
        if self._resume_mode and self._resume_benchmark_saving_override is not None:
            benchmark_saving = self._resume_benchmark_saving_override
        
        cfg = RunConfig(
            python_exe_path=self._python_exe_path,
            image_path=_normalize_path_for_cli(self.leInput.text()),
            output_dir=_normalize_path_for_cli(self.leOutput.text()),
            enable_tumor=self.cbTumor.isChecked(),
            enable_tissue=self.cbTissue.isChecked(),
            min_area_um2=self.sbMinArea.value(),
            max_area_um2=self.sbMaxArea.value(),
            debug_mode=str(self.cmbDebugMode.currentData()),
            verbose=self.cbVerbose.isChecked(),
            gpu_id=int(self.sbGpuId.value()),
            gpu_mode=str(self.cmbGpuMode.currentData()),
            memory_mode=str(self.cmbMemoryMode.currentData()),
            annotated_scale=float(self.dsbAnnotatedScale.value()),
            save_mode=save_mode,
            save_distance_map=bool(self.cbSaveDistanceMap.isChecked()),
            extended_properties=bool(self.cbExtendedProps.isChecked()),
            save_image_annotation=bool(self.cbSaveAnnotatedImage.isChecked()),
            save_qupath_annotation=bool(self.cbSaveQuPathAnnotation.isChecked()),
            save_tissue_window_grid=bool(self.cbSaveTissueWindowGrid.isChecked()),
            benchmark_saving=benchmark_saving,
            profiling=bool(self.cbProfiling.isChecked()),
            batch_size=batch_size_val,
            # Visualization toggles
            show_adipocyte_ids=self.cbShowAdipocyteIds.isChecked(),
            show_grid=self.cbShowGrid.isChecked(),
            # ROI guidance
            roi_freehand=self.cbRoiFreehand.isChecked(),
            roi_polygon_file=_normalize_path_for_cli(self.leRoiPolygonFile.text()),
            roi_max_dim=int(self.sbRoiMaxDim.value()),
            roi_min_coverage=float(self.dsbRoiMinCoverage.value()),
            # Batch resume options
            resume_batch=_normalize_path_for_cli(self.leResumeBatch.text()),
            resume_failed=self.cbResumeFailed.isChecked(),
            dry_run=self.cbDryRun.isChecked(),
            allow_resume=self.cbAllowResume.isChecked(),
        )
        return cfg

    def _on_run(self):
        if self.worker and self.worker.isRunning():
            QtWidgets.QMessageBox.warning(self, "Running", "A run is already in progress.")
            return

        # Collect configuration and validate required input path (unless resume/list mode)
        cfg = self._collect_cfg()
        img_path = cfg.image_path
        resume_batch = cfg.resume_batch
        
        # Allow running if resume_batch is set (no input file/folder required)
        if not img_path and not resume_batch and not cfg.allow_resume:
            QtWidgets.QMessageBox.warning(
                self,
                "Input required",
                "The analysis script requires an input image or folder.\n\n"
                "Please select a file or folder, provide a resume batch file, or enable 'List resumable batches'.",
            )
            self.leInput.setFocus()
            return
        
        # Validate input path if provided (can be file or folder)
        if img_path and not os.path.exists(img_path):
            QtWidgets.QMessageBox.warning(
                self,
                "Invalid input path",
                f"The selected input path does not exist:\n{img_path}",
            )
            self.leInput.setFocus()
            return
        
        # Validate resume batch path if provided
        if resume_batch and not os.path.isfile(resume_batch):
            QtWidgets.QMessageBox.warning(
                self,
                "Invalid resume batch path",
                f"The selected batch state file does not exist:\n{resume_batch}",
            )
            self.leResumeBatch.setFocus()
            return

        # Prepare UI for a new run
        self.teLog.clear()
        self._set_running_progress()
        self.lblImage.setText("Current image: -")
        self._set_current_image("-")
        self._set_run_state("Preparing")

        # Save user settings before launching
        self._save_settings()

        # Start worker thread that launches subprocess
        self.worker = RunWorker(cfg)
        self.worker.logLine.connect(self._append_log)
        self.worker.setBatchMax.connect(self._set_batch_max)
        self.worker.setBatchValue.connect(self._set_batch_val)
        self.worker.setImageIndeterminate.connect(self._image_indeterminate)
        self.worker.setImageProgress.connect(self._image_progress)
        she = self.worker.setCurrentImage.connect(self._set_current_image)  # noqa: F841 (keep binding)
        self.worker.runFinished.connect(self._run_finished)

        self.btnRun.setEnabled(False)
        self.btnStop.setEnabled(True)
        self.status.showMessage("Running.")
        self.worker.start()

    def _on_stop(self):
        if not self.worker:
            return
        # Ask for confirmation before stopping
        resp = QtWidgets.QMessageBox.question(
            self,
            "Stop run",
            "Are you sure you want to stop the current run?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No,
        )
        if resp != QtWidgets.QMessageBox.Yes:
            return
        self.worker.stop()
        self.btnStop.setEnabled(False)
        self._set_run_state("Stopping")
        self.status.showMessage("Stopping.")

    # ---------- Helpers ----------
    def _set_idle_progress(self):
        self.pbBatch.setEnabled(True)
        self.pbImage.setEnabled(True)
        self.pbBatch.setMinimum(0)
        self.pbBatch.setMaximum(1)
        self.pbBatch.setValue(0)
        self.pbBatch.resetFormat()
        self.pbImage.setMinimum(0)
        self.pbImage.setMaximum(1)
        self.pbImage.setValue(0)
        self.pbImage.resetFormat()
        self._batch_done_ui = 0
        self._batch_total_ui = 0
        self._image_done_ui = 0
        self._image_total_ui = 0
        self._image_progress_busy = False
        self._update_batch_progress_display()
        self._update_image_progress_display()

    def _set_running_progress(self):
        self.pbBatch.setEnabled(True)
        self.pbImage.setEnabled(True)
        self.pbBatch.setMinimum(0)
        self.pbBatch.setMaximum(0)
        self.pbBatch.setValue(0)
        self.pbImage.setMinimum(0)
        self.pbImage.setMaximum(0)
        self.pbImage.setValue(0)
        self._image_progress_busy = True
        self._batch_done_ui = 0
        self._batch_total_ui = 0
        self._image_done_ui = 0
        self._image_total_ui = 0
        self._update_batch_progress_display()
        self._update_image_progress_display()

    # ---------- Slots ----------
    @QtCore.Slot(str)
    def _append_log(self, line: str):
        self.teLog.appendPlainText(line)
        if self._should_expand_run_details(line):
            self._set_run_details_visible(True)
        line_text = str(line or "").lower()
        if self._run_state != "Error" and any(token in line_text for token in ("error", "traceback", "failed")):
            self._set_run_state("Error")
        # autoscroll
        sb = self.teLog.verticalScrollBar()
        sb.setValue(sb.maximum())

    @QtCore.Slot(int)
    def _set_batch_max(self, total: int):
        self._batch_total_ui = max(0, int(total))
        if total <= 0:
            self.pbBatch.setMaximum(0)
            self.pbBatch.setValue(0)
        else:
            self.pbBatch.setMinimum(0)
            self.pbBatch.setMaximum(total)
            self.pbBatch.setValue(0)
        self._update_batch_progress_display()

    @QtCore.Slot(int)
    def _set_batch_val(self, val: int):
        self._batch_done_ui = max(0, int(val))
        if self.pbBatch.maximum() == 0:
            self.pbBatch.setMaximum(max(val, 1))
        self.pbBatch.setValue(val)
        self._update_batch_progress_display()

    @QtCore.Slot(bool)
    def _image_indeterminate(self, on: bool):
        self._image_progress_busy = bool(on)
        if on:
            self._image_done_ui = 0
            self._image_total_ui = 0
            self.pbImage.setMaximum(0)
            self.pbImage.setValue(0)
        else:
            if self.pbImage.maximum() == 0:
                self.pbImage.setMinimum(0)
                self.pbImage.setMaximum(1)
        self._update_image_progress_display()

    @QtCore.Slot(int, int)
    def _image_progress(self, cur: int, tot: int):
        self._image_progress_busy = False
        self._image_done_ui = max(0, int(cur))
        self._image_total_ui = max(0, int(tot))
        self.pbImage.setMinimum(0)
        self.pbImage.setMaximum(tot)
        self.pbImage.setValue(cur)
        self._update_image_progress_display()

    @QtCore.Slot(str)
    def _set_current_image(self, name: str):
        self.lblImage.setText(f"Current image: {name}")
        if name and name != "-" and self._run_state != "Error":
            self._set_run_state("Running")

    @QtCore.Slot(int)
    def _run_finished(self, code: int):
        self.btnRun.setEnabled(True)
        self.btnStop.setEnabled(False)
        if code == 0:
            self._set_idle_progress()
            self.lblImage.setText("Current image: -")
            msg = "Finished OK"
            self._set_run_state("Finished")
            self.status.showMessage(msg, 5000)
        elif code == 130:
            self._set_idle_progress()
            self.lblImage.setText("Current image: -")
            msg = "Canceled by user"
            self._set_run_state("Ready")
            self.status.showMessage(msg, 5000)
            # Do not show warning dialog on cancel
            if self._pending_close:
                QtWidgets.QApplication.instance().quit()
            return
        else:
            if self.worker and getattr(self.worker, "_probe_failed", False):
                self._set_idle_progress()
                self.lblImage.setText("Current image: -")
            msg = self._format_exit_message(code)
            self._set_run_state("Error")
            self._set_run_details_visible(True)
            self.status.showMessage(msg, 5000)
            QtWidgets.QMessageBox.warning(self, "Run finished", msg)
        # If a close was requested while running, quit now that we're finished
        if self._pending_close:
            QtWidgets.QApplication.instance().quit()

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        # Always confirm on close; if a run is active, offer to stop first
        if self.worker and self.worker.isRunning():
            try:
                self.showNormal()
                self.raise_()
                self.activateWindow()
            except Exception:
                pass
            resp = QtWidgets.QMessageBox.question(
                self,
                "Stop run",
                "Are you sure you want to stop the current run?",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                QtWidgets.QMessageBox.No,
            )
            if resp != QtWidgets.QMessageBox.Yes:
                event.ignore()
                return
            # request stop and delay closing until the worker finishes
            self._pending_close = True
            try:
                self.worker.stop()
            except Exception:
                pass
            self.btnStop.setEnabled(False)
            self.status.showMessage("Stopping.")
            event.ignore()
            return
        else:
            try:
                self.showNormal()
                self.raise_()
                self.activateWindow()
            except Exception:
                pass
            resp = QtWidgets.QMessageBox.question(
                self,
                "Exit",
                "Are you sure you want to exit?",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                QtWidgets.QMessageBox.No,
            )
            if resp != QtWidgets.QMessageBox.Yes:
                event.ignore()
                return
        self._save_settings()
        event.accept()


def main():
    app = QtWidgets.QApplication(sys.argv)

    # Use a square icon for the taskbar/window while keeping the wide logo in the header.
    try:
        icon_path = resolve_app_icon_path()
        if icon_path and os.path.exists(icon_path):
            app.setWindowIcon(QtGui.QIcon(icon_path))
    except Exception:
        pass

    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
