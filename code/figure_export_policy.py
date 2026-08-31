from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional

import matplotlib.pyplot as plt
from matplotlib import font_manager

try:
    import seaborn as sns
except Exception:  # pragma: no cover - optional dependency in some contexts
    sns = None


class PDFPolicyError(RuntimeError):
    """Raised when exported PDFs violate Illustrator-friendly policy."""


@dataclass(frozen=True)
class PDFFontRecord:
    name: str
    font_type: str
    encoding: str
    embedded: bool
    subset: bool
    unicode_map: bool
    object_id: str
    generation: str


_PDFFONTS_ROW_RE = re.compile(
    r"^(?P<name>\S+)\s+"
    r"(?P<font_type>.+?)\s+"
    r"(?P<encoding>\S+)\s+"
    r"(?P<embedded>yes|no)\s+"
    r"(?P<subset>yes|no)\s+"
    r"(?P<unicode_map>yes|no)\s+"
    r"(?P<object_id>\d+)\s+"
    r"(?P<generation>\d+)\s*$"
)


def _as_bool(text: str) -> bool:
    return text.strip().lower() == "yes"


def _ensure_font_available(font_family: str) -> None:
    try:
        font_manager.findfont(
            font_manager.FontProperties(family=[font_family]),
            fallback_to_default=False,
        )
    except Exception as exc:  # pragma: no cover - environment dependent
        raise PDFPolicyError(
            f"{font_family} font is required but not available."
        ) from exc


def apply_publication_style(
    *,
    font_family: str = "Arial",
    font_size: float = 12.0,
    stroke_pt: float = 1.0,
    fig_dpi: int = 300,
    transparent: bool = True,
    style: Optional[str] = "whitegrid",
    require_font: bool = True,
    extra_rc: Optional[Mapping[str, object]] = None,
) -> Dict[str, object]:
    """
    Apply common publication defaults with Illustrator-safe export settings.
    """
    if require_font:
        _ensure_font_available(font_family)

    if style:
        if sns is not None:
            sns.set_theme(style=style)
        else:
            try:
                plt.style.use(f"seaborn-v0_8-{style}")
            except Exception:
                pass

    facecolor = "none" if transparent else "white"
    rc: Dict[str, object] = {
        "font.family": font_family,
        "font.size": float(font_size),
        "axes.labelsize": float(font_size),
        "axes.titlesize": float(font_size),
        "xtick.labelsize": float(font_size),
        "ytick.labelsize": float(font_size),
        "legend.fontsize": float(font_size),
        "axes.linewidth": float(stroke_pt),
        "lines.linewidth": float(stroke_pt),
        "patch.linewidth": float(stroke_pt),
        "xtick.major.width": float(stroke_pt),
        "ytick.major.width": float(stroke_pt),
        "xtick.minor.width": float(stroke_pt),
        "ytick.minor.width": float(stroke_pt),
        "savefig.dpi": int(fig_dpi),
        "savefig.transparent": bool(transparent),
        "figure.facecolor": facecolor,
        "axes.facecolor": facecolor,
        "savefig.facecolor": facecolor,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "text.usetex": False,
    }
    if extra_rc:
        rc.update(dict(extra_rc))
    plt.rcParams.update(rc)
    return rc


def _run_pdffonts(pdf_path: Path, pdffonts_bin: str = "pdffonts") -> str:
    if shutil.which(pdffonts_bin) is None:
        raise PDFPolicyError(
            "Could not find 'pdffonts'. Install Poppler and ensure 'pdffonts' is on PATH."
        )
    try:
        completed = subprocess.run(
            [pdffonts_bin, str(pdf_path)],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.strip() if exc.stderr else ""
        raise PDFPolicyError(
            f"pdffonts failed for {pdf_path}: {stderr or exc}"
        ) from exc
    return completed.stdout


def parse_pdffonts_output(output: str) -> List[PDFFontRecord]:
    rows: List[PDFFontRecord] = []
    for raw_line in output.splitlines():
        line = raw_line.rstrip()
        if not line or line.lower().startswith("name "):
            continue
        if line.startswith("-"):
            continue
        m = _PDFFONTS_ROW_RE.match(line)
        if m is None:
            continue
        rows.append(
            PDFFontRecord(
                name=m.group("name"),
                font_type=m.group("font_type").strip(),
                encoding=m.group("encoding"),
                embedded=_as_bool(m.group("embedded")),
                subset=_as_bool(m.group("subset")),
                unicode_map=_as_bool(m.group("unicode_map")),
                object_id=m.group("object_id"),
                generation=m.group("generation"),
            )
        )
    return rows


def assert_pdf_illustrator_friendly(
    pdf_path: Path | str,
    *,
    require_no_type3: bool = True,
    require_embedded: bool = True,
    pdffonts_bin: str = "pdffonts",
) -> List[PDFFontRecord]:
    """
    Validate a PDF's font table for Illustrator-friendly editing behavior.
    """
    p = Path(pdf_path)
    if not p.exists():
        raise PDFPolicyError(f"PDF does not exist: {p}")
    output = _run_pdffonts(p, pdffonts_bin=pdffonts_bin)
    rows = parse_pdffonts_output(output)
    if not rows:
        raise PDFPolicyError(f"No fonts detected in PDF: {p}")

    if require_no_type3:
        bad = [r for r in rows if "type 3" in r.font_type.lower()]
        if bad:
            details = ", ".join(f"{r.name} ({r.font_type})" for r in bad)
            raise PDFPolicyError(f"Type 3 fonts found in {p}: {details}")

    if require_embedded:
        not_embedded = [r for r in rows if not r.embedded]
        if not_embedded:
            details = ", ".join(f"{r.name} ({r.font_type})" for r in not_embedded)
            raise PDFPolicyError(f"Non-embedded fonts found in {p}: {details}")

    return rows


def _resolve_out_base(out_base: Path | str) -> Path:
    base = Path(out_base)
    if base.suffix.lower() in {".png", ".svg", ".pdf"}:
        base = base.with_suffix("")
    if not base.parent.exists():
        base.parent.mkdir(parents=True, exist_ok=True)
    return base


def save_publication_figure(
    fig: plt.Figure,
    out_base: Path | str,
    *,
    save_png: bool = True,
    save_svg: bool = True,
    save_pdf: bool = True,
    dpi: Optional[int] = None,
    bbox_inches: str = "tight",
    pad_inches: Optional[float] = None,
    transparent: bool = True,
    facecolor: Optional[str] = None,
    check_pdf: bool = True,
    require_no_type3: bool = True,
    require_embedded: bool = True,
    pdffonts_bin: str = "pdffonts",
) -> Dict[str, Path]:
    """
    Save a figure with centralized export policy and optional PDF validation.
    """
    base = _resolve_out_base(out_base)
    face = "none" if transparent else "white"
    if facecolor is not None:
        face = facecolor

    common_kwargs: Dict[str, object] = {
        "bbox_inches": bbox_inches,
        "transparent": transparent,
        "facecolor": face,
    }
    if pad_inches is not None:
        common_kwargs["pad_inches"] = float(pad_inches)
    if dpi is not None:
        common_kwargs["dpi"] = int(dpi)

    saved: Dict[str, Path] = {}

    if save_png:
        png_path = base.with_suffix(".png")
        fig.savefig(png_path, **common_kwargs)
        saved["png"] = png_path

    if save_svg:
        svg_path = base.with_suffix(".svg")
        with plt.rc_context({"svg.fonttype": "none", "text.usetex": False}):
            fig.savefig(svg_path, **common_kwargs)
        saved["svg"] = svg_path

    if save_pdf:
        pdf_path = base.with_suffix(".pdf")
        with plt.rc_context({"pdf.fonttype": 42, "ps.fonttype": 42, "text.usetex": False}):
            fig.savefig(pdf_path, format="pdf", **common_kwargs)
        if check_pdf:
            assert_pdf_illustrator_friendly(
                pdf_path,
                require_no_type3=require_no_type3,
                require_embedded=require_embedded,
                pdffonts_bin=pdffonts_bin,
            )
        saved["pdf"] = pdf_path

    return saved
