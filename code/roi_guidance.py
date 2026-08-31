#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ROI Guidance Module
===================

Interactive freehand ROI selection on a downscaled thumbnail and
window filtering based on ROI coverage.

Provides a PySide6 QGraphicsView-based GUI with:
  - Freehand, rectangle, ellipse, and polygon drawing tools
  - Zoom (scroll wheel) and pan (middle-click or spacebar+drag)
  - Per-polygon undo/redo, selection, deletion
  - Load/save ROI polygons from/to JSON
  - Keyboard shortcuts for all actions
"""

import json
import logging
import os
from typing import Iterable, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Thumbnail builder (no GUI dependency)
# ---------------------------------------------------------------------------

def _build_thumbnail(image_handler, roi_max_dim: int):
    """Create a downscaled thumbnail and return (thumbnail, scale_x, scale_y)."""
    from image_processing import OptimalImageReader

    width, height = image_handler.width, image_handler.height
    if width <= 0 or height <= 0:
        raise ValueError("Invalid image dimensions for ROI selection")

    max_dim = max(width, height)
    if roi_max_dim is None or roi_max_dim <= 0:
        scale = 1.0
    else:
        scale = min(1.0, float(roi_max_dim) / float(max_dim))

    thumbnail = OptimalImageReader.read_optimal_image(
        image_handler, width, height, scale, desired_level=0
    )
    if thumbnail.ndim == 3 and thumbnail.shape[2] > 3:
        thumbnail = thumbnail[:, :, :3]

    thumb_h, thumb_w = thumbnail.shape[:2]
    scale_x = float(thumb_w) / float(width)
    scale_y = float(thumb_h) / float(height)
    return thumbnail, scale_x, scale_y


# ---------------------------------------------------------------------------
# PySide6 ROI Selector — lazily imported
# ---------------------------------------------------------------------------

# Drawing-tool constants
_TOOL_FREEHAND = "freehand"
_TOOL_RECTANGLE = "rect"
_TOOL_ELLIPSE = "ellipse"
_TOOL_POLYGON = "polygon"

# Colours
_ACCENT = "#4a90e2"
_ROI_PEN_COLOR = "#e53935"
_ROI_FILL_COLOR = "#33e53935"

_QT_CLASSES_READY = False


def _numpy_to_qimage(arr: np.ndarray):
    """Convert an RGB uint8 numpy array to QImage (always copies data)."""
    from PySide6.QtGui import QImage

    h, w = arr.shape[:2]
    if arr.ndim == 2:
        return QImage(arr.data, w, h, w, QImage.Format_Grayscale8).copy()
    channels = arr.shape[2]
    if channels == 4:
        bpl = 4 * w
        return QImage(arr.data, w, h, bpl, QImage.Format_RGBA8888).copy()
    # RGB -> RGBA so QImage row alignment is safe
    rgba = np.empty((h, w, 4), dtype=np.uint8)
    rgba[:, :, :3] = arr[:, :, :3]
    rgba[:, :, 3] = 255
    bpl = 4 * w
    return QImage(rgba.data, w, h, bpl, QImage.Format_RGBA8888).copy()


# We store the lazily-built Qt classes in this dict so the module stays importable
# without PySide6.
_QT = {}


def _ensure_qt_classes():
    """Define all PySide6 classes on first use."""
    global _QT_CLASSES_READY
    if _QT_CLASSES_READY:
        return
    _QT_CLASSES_READY = True

    from PySide6 import QtCore, QtGui, QtWidgets
    from PySide6.QtCore import Qt, QPointF, QRectF
    from PySide6.QtGui import (
        QBrush, QColor, QKeySequence, QPainter, QPainterPath, QPen,
        QPixmap, QPolygonF, QShortcut,
    )
    from PySide6.QtWidgets import (
        QApplication, QButtonGroup, QDialog, QFileDialog,
        QGraphicsPathItem, QGraphicsScene, QGraphicsView,
        QHBoxLayout, QMessageBox, QPushButton, QStatusBar,
        QToolButton, QVBoxLayout,
    )

    # ------------------------------------------------------------------
    # Helper
    # ------------------------------------------------------------------
    def _qpath_to_points(path: QPainterPath) -> list:
        pts = []
        for i in range(path.elementCount()):
            el = path.elementAt(i)
            pts.append(QPointF(el.x, el.y))
        return pts

    # ------------------------------------------------------------------
    # QGraphicsView with zoom / pan / drawing
    # ------------------------------------------------------------------
    class ROIGraphicsView(QGraphicsView):
        def __init__(self, scene, parent=None):
            super().__init__(scene, parent)
            self.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
            self.setDragMode(QGraphicsView.NoDrag)
            self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
            self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
            self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
            self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
            self.setBackgroundBrush(QBrush(QColor("#2b2b2b")))
            self.setMinimumSize(400, 300)

            self._zoom = 1.0
            self._panning = False
            self._pan_start = QPointF()
            self._space_held = False

            # Drawing
            self.current_tool = _TOOL_FREEHAND
            self._drawing = False
            self._cur_item = None
            self._cur_path = None
            self._origin = None
            self._poly_verts: list = []
            self._poly_preview = None

            # Callbacks (set by dialog)
            self.on_shape_finished = None
            self.on_status = None

        # ---- zoom ----
        def wheelEvent(self, ev):
            angle = ev.angleDelta().y()
            if angle == 0:
                return
            factor = 1.15 if angle > 0 else 1.0 / 1.15
            nz = self._zoom * factor
            if nz < 0.05 or nz > 80.0:
                return
            self._zoom = nz
            self.scale(factor, factor)

        def fit_in_view_padded(self):
            self.fitInView(self.sceneRect(), Qt.KeepAspectRatio)
            self._zoom = self.transform().m11()

        # ---- pan helpers ----
        def keyPressEvent(self, ev):
            if ev.key() == Qt.Key_Space:
                self._space_held = True
                self.setCursor(Qt.OpenHandCursor)
            super().keyPressEvent(ev)

        def keyReleaseEvent(self, ev):
            if ev.key() == Qt.Key_Space:
                self._space_held = False
                if not self._panning:
                    self.setCursor(Qt.CrossCursor)
            super().keyReleaseEvent(ev)

        # ---- mouse ----
        def mousePressEvent(self, ev):
            if ev.button() == Qt.MiddleButton or (
                ev.button() == Qt.LeftButton and self._space_held
            ):
                self._panning = True
                self._pan_start = ev.position()
                self.setCursor(Qt.ClosedHandCursor)
                ev.accept()
                return
            if ev.button() != Qt.LeftButton:
                return super().mousePressEvent(ev)
            sp = self.mapToScene(ev.position().toPoint())
            tool = self.current_tool
            if tool == _TOOL_FREEHAND:
                self._begin_freehand(sp)
            elif tool == _TOOL_RECTANGLE:
                self._begin_rect(sp)
            elif tool == _TOOL_ELLIPSE:
                self._begin_ellipse(sp)
            elif tool == _TOOL_POLYGON:
                self._polygon_click(sp)
            ev.accept()

        def mouseMoveEvent(self, ev):
            if self._panning:
                d = ev.position() - self._pan_start
                self._pan_start = ev.position()
                self.horizontalScrollBar().setValue(
                    int(self.horizontalScrollBar().value() - d.x()))
                self.verticalScrollBar().setValue(
                    int(self.verticalScrollBar().value() - d.y()))
                ev.accept()
                return
            sp = self.mapToScene(ev.position().toPoint())
            if self._drawing:
                tool = self.current_tool
                if tool == _TOOL_FREEHAND:
                    self._move_freehand(sp)
                elif tool == _TOOL_RECTANGLE:
                    self._move_rect(sp)
                elif tool == _TOOL_ELLIPSE:
                    self._move_ellipse(sp)
            elif self.current_tool == _TOOL_POLYGON and self._poly_verts:
                self._polygon_preview(sp)
            if self.on_status:
                self.on_status(f"({sp.x():.0f}, {sp.y():.0f})  Zoom: {self._zoom:.1f}x")
            ev.accept()

        def mouseReleaseEvent(self, ev):
            if self._panning and ev.button() in (Qt.MiddleButton, Qt.LeftButton):
                self._panning = False
                self.setCursor(Qt.OpenHandCursor if self._space_held else Qt.CrossCursor)
                ev.accept()
                return
            if ev.button() != Qt.LeftButton:
                return super().mouseReleaseEvent(ev)
            tool = self.current_tool
            if tool == _TOOL_FREEHAND:
                self._end_freehand()
            elif tool == _TOOL_RECTANGLE:
                self._end_rect()
            elif tool == _TOOL_ELLIPSE:
                self._end_ellipse()
            ev.accept()

        def mouseDoubleClickEvent(self, ev):
            if ev.button() == Qt.LeftButton and self.current_tool == _TOOL_POLYGON:
                self._end_polygon()
                ev.accept()
                return
            super().mouseDoubleClickEvent(ev)

        # ---- pen helpers ----
        def _pen(self):
            p = QPen(QColor(_ROI_PEN_COLOR), 2.0)
            p.setCosmetic(True)
            return p

        def _brush(self):
            return QBrush(QColor(_ROI_FILL_COLOR))

        # ---- freehand ----
        def _begin_freehand(self, pos):
            self._drawing = True
            self._cur_path = QPainterPath(pos)
            self._cur_item = QGraphicsPathItem(self._cur_path)
            self._cur_item.setPen(self._pen())
            self._cur_item.setBrush(self._brush())
            self.scene().addItem(self._cur_item)

        def _move_freehand(self, pos):
            if self._cur_path is None:
                return
            self._cur_path.lineTo(pos)
            self._cur_item.setPath(self._cur_path)

        def _end_freehand(self):
            self._drawing = False
            if self._cur_path is None:
                return
            self._cur_path.closeSubpath()
            self._cur_item.setPath(self._cur_path)
            pts = _qpath_to_points(self._cur_path)
            self._emit(pts)
            self._cur_path = None
            self._cur_item = None

        # ---- rectangle ----
        def _begin_rect(self, pos):
            self._drawing = True
            self._origin = pos
            self._cur_item = self.scene().addRect(
                QRectF(pos, pos), self._pen(), self._brush())

        def _move_rect(self, pos):
            if self._origin is None:
                return
            self._cur_item.setRect(QRectF(self._origin, pos).normalized())

        def _end_rect(self):
            self._drawing = False
            if self._origin is None:
                return
            r = self._cur_item.rect()
            pts = [QPointF(r.left(), r.top()), QPointF(r.right(), r.top()),
                   QPointF(r.right(), r.bottom()), QPointF(r.left(), r.bottom())]
            self._emit(pts)
            self._origin = None
            self._cur_item = None

        # ---- ellipse ----
        def _begin_ellipse(self, pos):
            self._drawing = True
            self._origin = pos
            self._cur_item = self.scene().addEllipse(
                QRectF(pos, pos), self._pen(), self._brush())

        def _move_ellipse(self, pos):
            if self._origin is None:
                return
            self._cur_item.setRect(QRectF(self._origin, pos).normalized())

        def _end_ellipse(self):
            self._drawing = False
            if self._origin is None:
                return
            r = self._cur_item.rect()
            cx, cy = r.center().x(), r.center().y()
            rx, ry = r.width() / 2.0, r.height() / 2.0
            n = 64
            pts = [
                QPointF(cx + rx * np.cos(2 * np.pi * i / n),
                        cy + ry * np.sin(2 * np.pi * i / n))
                for i in range(n)
            ]
            self._emit(pts)
            self._origin = None
            self._cur_item = None

        # ---- polygon (click to place, double-click to close) ----
        def _polygon_click(self, pos):
            self._poly_verts.append(pos)
            r = 3.0 / max(self._zoom, 0.1)
            m = self.scene().addEllipse(
                pos.x() - r, pos.y() - r, 2 * r, 2 * r,
                self._pen(), QBrush(QColor(_ROI_PEN_COLOR)))
            m.setData(0, "poly_marker")

        def _polygon_preview(self, pos):
            if self._poly_preview:
                self.scene().removeItem(self._poly_preview)
                self._poly_preview = None
            if not self._poly_verts:
                return
            path = QPainterPath(self._poly_verts[0])
            for p in self._poly_verts[1:]:
                path.lineTo(p)
            path.lineTo(pos)
            self._poly_preview = QGraphicsPathItem(path)
            pen = QPen(QColor(_ROI_PEN_COLOR), 1.5)
            pen.setCosmetic(True)
            pen.setStyle(Qt.DashLine)
            self._poly_preview.setPen(pen)
            self.scene().addItem(self._poly_preview)

        def _end_polygon(self):
            if self._poly_preview:
                self.scene().removeItem(self._poly_preview)
                self._poly_preview = None
            for item in list(self.scene().items()):
                if item.data(0) == "poly_marker":
                    self.scene().removeItem(item)
            if len(self._poly_verts) < 3:
                self._poly_verts.clear()
                return
            poly = QPolygonF(self._poly_verts)
            self.scene().addPolygon(poly, self._pen(), self._brush())
            self._emit(list(self._poly_verts))
            self._poly_verts = []

        def _emit(self, pts):
            if self.on_shape_finished and len(pts) >= 3:
                self.on_shape_finished(pts)

    # ------------------------------------------------------------------
    # ROI Dialog
    # ------------------------------------------------------------------
    class ROIDialog(QDialog):
        def __init__(self, thumbnail, scale_x, scale_y,
                     output_dir=None, load_path=None, parent=None):
            super().__init__(parent)
            self.setWindowTitle("AdiFind \u2014 Draw ROI")
            self.setMinimumSize(800, 600)
            self.resize(1100, 800)

            self._thumbnail = thumbnail
            self._scale_x = scale_x
            self._scale_y = scale_y
            self._output_dir = output_dir

            self._polygons: List[list] = []
            self._redo_stack: List[list] = []
            self._accepted = False

            self._build_ui()
            self._wire_shortcuts()

            if load_path and os.path.isfile(load_path):
                self._load_from_file(load_path)

        def _build_ui(self):
            layout = QVBoxLayout(self)
            layout.setContentsMargins(6, 6, 6, 6)

            # -- toolbar --
            tb = QHBoxLayout()
            tb.setSpacing(4)

            self._tool_btns = {}
            grp = QButtonGroup(self)
            grp.setExclusive(True)
            for tid, label, tip in [
                (_TOOL_FREEHAND, "\u270D Freehand", "Draw freehand (F)"),
                (_TOOL_RECTANGLE, "\u25AD Rectangle", "Draw rectangle (R)"),
                (_TOOL_ELLIPSE, "\u25CB Ellipse", "Draw ellipse (E)"),
                (_TOOL_POLYGON, "\u2B21 Polygon", "Click vertices, double-click to close (P)"),
            ]:
                b = QToolButton()
                b.setText(label)
                b.setCheckable(True)
                b.setToolTip(tip)
                b.setMinimumWidth(100)
                b.setStyleSheet(f"""
                    QToolButton {{
                        border: 1px solid #999; border-radius: 4px;
                        padding: 5px 10px; background: #f5f5f5; color: black;
                        font-size: 10pt;
                    }}
                    QToolButton:checked {{
                        background: {_ACCENT}; color: white;
                        border-color: {_ACCENT}; font-weight: bold;
                    }}
                    QToolButton:hover {{ background: #e3f2fd; color: black; }}
                    QToolButton:checked:hover {{ background: #3a80d2; color: white; }}
                """)
                grp.addButton(b)
                b.clicked.connect(lambda _, t=tid: self._set_tool(t))
                self._tool_btns[tid] = b
                tb.addWidget(b)

            self._tool_btns[_TOOL_FREEHAND].setChecked(True)
            tb.addSpacing(20)

            for label, tip, cb in [
                ("Undo", "Undo last polygon (Ctrl+Z)", self._undo),
                ("Redo", "Redo (Ctrl+Y)", self._redo),
                ("Clear", "Clear all ROIs (X)", self._clear_all),
                ("Load\u2026", "Load ROI from JSON", self._load_dialog),
                ("Save\u2026", "Save ROI to JSON", self._save_dialog),
            ]:
                btn = QPushButton(label)
                btn.setToolTip(tip)
                btn.setMinimumWidth(60)
                btn.clicked.connect(cb)
                tb.addWidget(btn)

            tb.addStretch(1)

            self._btn_accept = QPushButton("\u2714  Accept")
            self._btn_accept.setToolTip("Accept ROI and proceed (Enter)")
            self._btn_accept.setMinimumWidth(100)
            self._btn_accept.setStyleSheet(f"""
                QPushButton {{
                    background: {_ACCENT}; color: white; border: none;
                    border-radius: 4px; padding: 6px 16px;
                    font-weight: bold; font-size: 11pt;
                }}
                QPushButton:hover {{ background: #3a80d2; }}
                QPushButton:pressed {{ background: #2a70c2; }}
            """)
            self._btn_accept.clicked.connect(self._on_accept)

            btn_cancel = QPushButton("Cancel")
            btn_cancel.setToolTip("Cancel ROI selection (Escape)")
            btn_cancel.setMinimumWidth(80)
            btn_cancel.clicked.connect(self._confirm_cancel)

            tb.addWidget(btn_cancel)
            tb.addWidget(self._btn_accept)
            layout.addLayout(tb)

            # -- graphics view --
            self._scene = QGraphicsScene(self)
            qimg = _numpy_to_qimage(self._thumbnail)
            self._bg = self._scene.addPixmap(QPixmap.fromImage(qimg))

            self._view = ROIGraphicsView(self._scene, self)
            self._view.on_shape_finished = self._on_shape_finished
            self._view.on_status = self._on_status
            self._view.setCursor(Qt.CrossCursor)
            layout.addWidget(self._view, 1)

            # -- status bar --
            self._status = QStatusBar(self)
            self._status.showMessage(
                "Draw ROI regions on the image. Accept when done. "
                "Scroll to zoom, middle-click or Space+drag to pan."
            )
            layout.addWidget(self._status)

            QtCore.QTimer.singleShot(50, self._view.fit_in_view_padded)

        def _wire_shortcuts(self):
            QShortcut(QKeySequence("Ctrl+Z"), self, self._undo)
            QShortcut(QKeySequence("Ctrl+Y"), self, self._redo)
            QShortcut(QKeySequence("Ctrl+Shift+Z"), self, self._redo)
            QShortcut(QKeySequence("X"), self, self._clear_all)
            QShortcut(QKeySequence("F"), self, lambda: self._set_tool(_TOOL_FREEHAND))
            QShortcut(QKeySequence("R"), self, lambda: self._set_tool(_TOOL_RECTANGLE))
            QShortcut(QKeySequence("E"), self, lambda: self._set_tool(_TOOL_ELLIPSE))
            QShortcut(QKeySequence("P"), self, lambda: self._set_tool(_TOOL_POLYGON))
            QShortcut(QKeySequence("Return"), self, self._on_accept)
            QShortcut(QKeySequence("Enter"), self, self._on_accept)
            QShortcut(QKeySequence("Escape"), self, self._confirm_cancel)
            QShortcut(QKeySequence("Delete"), self, self._delete_last)
            QShortcut(QKeySequence("Backspace"), self, self._delete_last)

        # ---- tool ----
        def _set_tool(self, tid):
            self._view.current_tool = tid
            b = self._tool_btns.get(tid)
            if b and not b.isChecked():
                b.setChecked(True)
            self._status.showMessage(f"Tool: {tid}")

        # ---- callbacks ----
        def _on_shape_finished(self, pts):
            if len(pts) < 3:
                return
            self._polygons.append(pts)
            self._redo_stack.clear()
            self._update_status()

        def _on_status(self, msg):
            self._status.showMessage(msg)

        # ---- undo / redo / clear ----
        def _undo(self):
            if not self._polygons:
                return
            self._redo_stack.append(self._polygons.pop())
            self._redraw()
            self._update_status()

        def _redo(self):
            if not self._redo_stack:
                return
            self._polygons.append(self._redo_stack.pop())
            self._redraw()
            self._update_status()

        def _clear_all(self):
            self._polygons.clear()
            self._redo_stack.clear()
            self._redraw()
            self._update_status()

        def _delete_last(self):
            if self._polygons:
                self._polygons.pop()
                self._redraw()
                self._update_status()

        def _redraw(self):
            from PySide6.QtGui import QBrush, QColor, QPen, QPolygonF
            for item in list(self._scene.items()):
                if item is not self._bg:
                    self._scene.removeItem(item)
            pen = QPen(QColor(_ROI_PEN_COLOR), 2.0)
            pen.setCosmetic(True)
            brush = QBrush(QColor(_ROI_FILL_COLOR))
            for pts in self._polygons:
                self._scene.addPolygon(QPolygonF(pts), pen, brush)

        def _update_status(self):
            n = len(self._polygons)
            self._status.showMessage(
                f"{n} ROI region{'s' if n != 1 else ''} drawn. "
                "Accept when done, or keep drawing."
            )

        # ---- load / save ----
        def _load_dialog(self):
            fn, _ = QFileDialog.getOpenFileName(
                self, "Load ROI polygons", self._output_dir or "",
                "JSON files (*.json);;All files (*)")
            if fn:
                self._load_from_file(fn)

        def _load_from_file(self, path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                polys_raw = data.get("polygons_full_res", [])
                if not polys_raw:
                    self._status.showMessage("No polygons found in file.")
                    return
                from PySide6.QtCore import QPointF
                for raw in polys_raw:
                    pts = []
                    for x, y in raw:
                        pts.append(QPointF(float(x) * self._scale_x,
                                           float(y) * self._scale_y))
                    if len(pts) >= 3:
                        self._polygons.append(pts)
                self._redo_stack.clear()
                self._redraw()
                self._update_status()
                self._status.showMessage(
                    f"Loaded {len(polys_raw)} ROI(s) from {os.path.basename(path)}")
            except Exception as e:
                QMessageBox.warning(self, "Load failed", f"Could not load ROI file:\n{e}")

        def _save_dialog(self):
            if not self._polygons:
                QMessageBox.information(self, "Nothing to save", "Draw at least one ROI first.")
                return
            fn, _ = QFileDialog.getSaveFileName(
                self, "Save ROI polygons",
                os.path.join(self._output_dir or "", "roi_polygon_fullres.json"),
                "JSON files (*.json);;All files (*)")
            if fn:
                polys = self._to_full_res()
                try:
                    with open(fn, "w", encoding="utf-8") as f:
                        json.dump({"polygons_full_res": polys}, f, indent=2)
                    self._status.showMessage(
                        f"Saved {len(polys)} ROI(s) to {os.path.basename(fn)}")
                except Exception as e:
                    QMessageBox.warning(self, "Save failed", f"Could not save:\n{e}")

        # ---- cancel confirmation ----
        def _confirm_cancel(self):
            """Ask the user to confirm before discarding ROI work."""
            if self._polygons:
                resp = QMessageBox.question(
                    self,
                    "Cancel ROI Selection",
                    "You have drawn ROI region(s).\n\n"
                    "Are you sure you want to cancel and discard them?",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No,
                )
                if resp != QMessageBox.Yes:
                    return
            super(ROIDialog, self).reject()

        def reject(self):
            """Override reject so that the window close button also confirms."""
            self._confirm_cancel()

        # ---- result ----
        def _on_accept(self):
            self._accepted = True
            self.accept()

        def _to_full_res(self):
            result = []
            for pts in self._polygons:
                poly = []
                for p in pts:
                    fx = float(p.x()) / self._scale_x if self._scale_x > 0 else float(p.x())
                    fy = float(p.y()) / self._scale_y if self._scale_y > 0 else float(p.y())
                    poly.append([fx, fy])
                result.append(poly)
            return result

        def get_result(self):
            if not self._accepted or not self._polygons:
                return None
            return {"full_res_polygons": self._to_full_res()}

    # Store in module-level dict
    _QT["ROIDialog"] = ROIDialog
    _QT["ROIGraphicsView"] = ROIGraphicsView


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def select_freehand_roi(
    image_handler,
    output_dir: Optional[str] = None,
    roi_max_dim: int = 2048,
    load_polygon_file: Optional[str] = None,
):
    """
    Launch an interactive ROI selector on a downscaled thumbnail.

    Parameters
    ----------
    image_handler : ImageHandler
        The opened slide.
    output_dir : str, optional
        Where to save roi_mask.png and roi_polygon_fullres.json.
    roi_max_dim : int
        Max dimension for the thumbnail shown in the GUI.
    load_polygon_file : str, optional
        Path to a previously saved roi_polygon_fullres.json.
        If provided *without* a display, polygons are loaded headlessly.
        If a display is available the file is pre-loaded into the editor.

    Returns
    -------
    tuple or None
        ``(roi_mask, scale_x, scale_y, full_res_polygons)``
        roi_mask is uint8 0/1 at thumbnail resolution.
        Returns None if cancelled or empty.
    """
    thumbnail, scale_x, scale_y = _build_thumbnail(image_handler, roi_max_dim)
    thumb_h, thumb_w = thumbnail.shape[:2]

    # Headless path — just load from file, skip GUI
    if load_polygon_file and os.path.isfile(load_polygon_file):
        try:
            with open(load_polygon_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            polys_raw = data.get("polygons_full_res", [])
            if polys_raw:
                roi_mask = _rasterize_polygons(polys_raw, thumb_w, thumb_h, scale_x, scale_y)
                if output_dir:
                    save_roi_artifacts(output_dir, roi_mask, polys_raw)
                return roi_mask, scale_x, scale_y, polys_raw
        except Exception as e:
            logging.warning(f"Could not load ROI polygon file: {e}")

    # GUI path
    _ensure_qt_classes()
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance()
    if app is None:
        app = QApplication([])

    dialog = _QT["ROIDialog"](
        thumbnail, scale_x, scale_y,
        output_dir=output_dir,
        load_path=load_polygon_file,
    )
    dialog.exec()
    result = dialog.get_result()

    # Clean up Qt to prevent interference with CUDA/GPU inference
    dialog.close()
    dialog.deleteLater()
    app.processEvents()
    if app is not None:
        app.quit()
        app.processEvents()

    if result is None:
        logging.warning("ROI selection canceled.")
        return None

    full_res_polygons = result["full_res_polygons"]
    if not full_res_polygons:
        logging.warning("ROI selection empty.")
        return None

    roi_mask = _rasterize_polygons(full_res_polygons, thumb_w, thumb_h, scale_x, scale_y)

    if output_dir:
        save_roi_artifacts(output_dir, roi_mask, full_res_polygons)

    return roi_mask, scale_x, scale_y, full_res_polygons


# ---------------------------------------------------------------------------
# Rasterization — cv2.fillPoly > skimage > matplotlib fallback
# ---------------------------------------------------------------------------

def _rasterize_polygons(
    full_res_polygons: List[List[List[float]]],
    thumb_w: int, thumb_h: int,
    scale_x: float, scale_y: float,
) -> np.ndarray:
    """Rasterize full-res polygons into a thumbnail-resolution binary mask."""
    roi_mask = np.zeros((thumb_h, thumb_w), dtype=np.uint8)

    thumb_polys = []
    for poly in full_res_polygons:
        pts = np.array(poly, dtype=np.float64)
        pts[:, 0] *= scale_x
        pts[:, 1] *= scale_y
        thumb_polys.append(pts)

    # Try OpenCV (fastest)
    try:
        import cv2
        for pts in thumb_polys:
            cv2.fillPoly(roi_mask, [np.round(pts).astype(np.int32)], 1)
        return roi_mask
    except ImportError:
        pass

    # Try scikit-image
    try:
        from skimage.draw import polygon as ski_polygon
        for pts in thumb_polys:
            rr, cc = ski_polygon(pts[:, 1], pts[:, 0], shape=(thumb_h, thumb_w))
            roi_mask[rr, cc] = 1
        return roi_mask
    except ImportError:
        pass

    # Fallback: matplotlib
    from matplotlib.path import Path as MplPath
    yy, xx = np.mgrid[0:thumb_h, 0:thumb_w]
    points = np.vstack((xx.ravel(), yy.ravel())).T
    for pts in thumb_polys:
        path = MplPath(pts)
        mask = path.contains_points(points, radius=0.5).reshape((thumb_h, thumb_w))
        roi_mask |= mask.astype(np.uint8)
    return roi_mask


# ---------------------------------------------------------------------------
# Artifact saving
# ---------------------------------------------------------------------------

def save_roi_artifacts(output_dir: str, roi_mask: np.ndarray,
                       full_res_polygons: List[List[List[float]]]):
    """Save ROI mask PNG and full-res polygon JSON."""
    try:
        from PIL import Image
        os.makedirs(output_dir, exist_ok=True)
        Image.fromarray((roi_mask * 255).astype(np.uint8)).save(
            os.path.join(output_dir, "roi_mask.png"))
        with open(os.path.join(output_dir, "roi_polygon_fullres.json"), "w",
                  encoding="utf-8") as f:
            json.dump({"polygons_full_res": full_res_polygons}, f, indent=2)
    except Exception as e:
        logging.warning(f"Could not save ROI artifacts: {e}")


# ---------------------------------------------------------------------------
# Window filtering (integral-image approach — unchanged)
# ---------------------------------------------------------------------------

def _integral_image(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask.astype(np.uint8), ((1, 0), (1, 0)),
                    mode="constant", constant_values=0)
    integral = np.cumsum(padded, axis=0, dtype=np.int64)
    integral = np.cumsum(integral, axis=1, dtype=np.int64)
    return integral


def _sum_region(integral: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> int:
    return (int(integral[y2, x2]) - int(integral[y1, x2])
            - int(integral[y2, x1]) + int(integral[y1, x1]))


def filter_windows_by_roi(
    window_coords: Iterable[Tuple[int, int]],
    window_size: Tuple[int, int],
    roi_mask: np.ndarray,
    scale_x: float,
    scale_y: float,
    min_coverage: float = 0.2,
) -> List[Tuple[int, int]]:
    """Filter windows by ROI coverage using an integral image."""
    if roi_mask is None or roi_mask.size == 0:
        return list(window_coords)

    thumb_h, thumb_w = roi_mask.shape[:2]
    win_w, win_h = window_size
    integral = _integral_image(roi_mask)
    filtered = []

    for x, y in window_coords:
        tx1 = max(0, min(int(round(x * scale_x)), thumb_w))
        ty1 = max(0, min(int(round(y * scale_y)), thumb_h))
        tx2 = max(0, min(int(round((x + win_w) * scale_x)), thumb_w))
        ty2 = max(0, min(int(round((y + win_h) * scale_y)), thumb_h))

        if tx2 <= tx1 or ty2 <= ty1:
            coverage = 0.0
        else:
            roi_px = _sum_region(integral, tx1, ty1, tx2, ty2)
            coverage = roi_px / float((tx2 - tx1) * (ty2 - ty1))

        if coverage >= min_coverage:
            filtered.append((x, y))

    return filtered


def filter_detections_by_roi(
    final_properties: dict,
    full_mask: np.ndarray,
    mask_areas: dict,
    adipocyte_ids: list,
    roi_mask: np.ndarray,
    scale_x: float,
    scale_y: float,
) -> Tuple[np.ndarray, dict, list, dict]:
    """Remove adipocytes whose centroids fall outside the ROI mask.

    After inference, windows near the ROI boundary may contain detections
    that extend beyond the drawn ROI.  This function checks each adipocyte
    centroid and removes those outside the ROI, then re-labels the remaining
    IDs to consecutive integers starting at 1.

    Returns
    -------
    tuple
        (full_mask, mask_areas, adipocyte_ids, final_properties) — updated.
    """
    if roi_mask is None or roi_mask.size == 0 or not final_properties:
        return full_mask, mask_areas, adipocyte_ids, final_properties

    thumb_h, thumb_w = roi_mask.shape[:2]
    keep_ids = []

    for aid, props in final_properties.items():
        cx = props.get("centroid_x", 0.0)
        cy = props.get("centroid_y", 0.0)
        tx = int(round(cx * scale_x))
        ty = int(round(cy * scale_y))
        tx = max(0, min(tx, thumb_w - 1))
        ty = max(0, min(ty, thumb_h - 1))
        if roi_mask[ty, tx] > 0:
            keep_ids.append(aid)

    if len(keep_ids) == len(final_properties):
        return full_mask, mask_areas, adipocyte_ids, final_properties

    # Build old→new ID mapping (consecutive from 1)
    keep_set = set(keep_ids)
    old_to_new = {}
    for new_id, old_id in enumerate(sorted(keep_set), start=1):
        old_to_new[old_id] = new_id

    # Relabel full_mask
    lut = np.zeros(full_mask.max() + 1, dtype=full_mask.dtype)
    for old_id, new_id in old_to_new.items():
        if old_id < len(lut):
            lut[old_id] = new_id
    full_mask = lut[full_mask]

    # Rebuild output dicts with new IDs
    new_properties = {}
    new_areas = {}
    new_ids = []
    for old_id in sorted(keep_set):
        new_id = old_to_new[old_id]
        new_properties[new_id] = final_properties[old_id]
        if old_id in mask_areas:
            new_areas[new_id] = mask_areas[old_id]
        new_ids.append(new_id)

    return full_mask, new_areas, new_ids, new_properties


__all__ = [
    "select_freehand_roi",
    "filter_windows_by_roi",
    "filter_detections_by_roi",
    "save_roi_artifacts",
]
