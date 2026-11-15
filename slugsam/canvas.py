"""Zoomable image view that turns mouse gestures into SAM prompts."""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QGraphicsScene, QGraphicsView

DRAG_THRESHOLD = 5  # px of movement before a left press becomes a box drag


def make_lut(n: int = 65536) -> np.ndarray:
    """RGBA colour per label id; id 0 is transparent. Golden-ratio hues keep neighbours distinct."""
    i = np.arange(n, dtype=np.float64)
    h = (i * 0.61803398875) % 1.0
    s = 0.65 + 0.35 * ((i * 0.37) % 1.0)
    v = 1.0
    k = (h * 6.0).astype(int) % 6
    f = h * 6.0 - np.floor(h * 6.0)
    p, q, t = v * (1 - s), v * (1 - f * s), v * (1 - (1 - f) * s)
    vv = np.full(n, v)
    r = np.choose(k, [vv, q, p, p, t, vv])
    g = np.choose(k, [t, vv, vv, q, p, p])
    b = np.choose(k, [p, p, t, vv, vv, q])
    lut = np.empty((n, 4), np.uint8)
    lut[:, 0], lut[:, 1], lut[:, 2], lut[:, 3] = r * 255, g * 255, b * 255, 255
    lut[0] = 0
    return lut


def _pixmap(arr: np.ndarray, fmt: QImage.Format) -> QPixmap:
    arr = np.ascontiguousarray(arr)
    h, w = arr.shape[:2]
    return QPixmap.fromImage(QImage(arr.data, w, h, arr.strides[0], fmt))


class Canvas(QGraphicsView):
    pointClicked = Signal(float, float, bool)  # x, y, positive
    modClicked = Signal(float, float, bool, bool)  # x, y, ctrl, shift: click with a modifier held
    boxDrawn = Signal(float, float, float, float)
    hovered = Signal(float, float)  # image coords, or (-1, -1) when off the image

    def __init__(self, parent=None):
        super().__init__(parent)
        scene = QGraphicsScene(self)
        self.setScene(scene)
        self._img = scene.addPixmap(QPixmap())
        self._lab = scene.addPixmap(QPixmap())
        self._sel = scene.addPixmap(QPixmap())
        self._prev = scene.addPixmap(QPixmap())
        self._lab.setOpacity(0.45)
        self._shape = (0, 0)
        self.points: list[tuple[float, float, bool]] = []
        self.box: tuple[float, float, float, float] | None = None
        self._drag_box: QRectF | None = None
        self._press = None  # viewport pos of a left press that may become a click or a box
        self._pan = None
        self._space = False

        self.setBackgroundBrush(QColor("#1b1b1f"))
        self.setFrameShape(QGraphicsView.Shape.NoFrame)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.FullViewportUpdate)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.NoContextMenu)
        self.setCursor(Qt.CursorShape.CrossCursor)

    # ---- content -------------------------------------------------------

    def set_image(self, rgb: np.ndarray):
        new_shape = rgb.shape[:2] != self._shape
        self._shape = rgb.shape[:2]
        self._img.setPixmap(_pixmap(rgb, QImage.Format.Format_RGB888))
        if new_shape:
            h, w = self._shape
            self.setSceneRect(QRectF(-w / 2, -h / 2, 2 * w, 2 * h))  # room to pan past edges
            self.fit()

    def clear(self):
        for item in (self._img, self._lab, self._sel, self._prev):
            item.setPixmap(QPixmap())
        self._shape = (0, 0)
        self.set_prompts([], None)

    def set_labels(self, rgba: np.ndarray):
        self._lab.setPixmap(_pixmap(rgba, QImage.Format.Format_RGBA8888))

    def set_label_opacity(self, opacity: float):
        self._lab.setOpacity(opacity)

    def set_labels_visible(self, visible: bool):
        self._lab.setVisible(visible)

    def set_preview(self, mask: np.ndarray | None, color=(255, 255, 255), fill_alpha=110):
        """Show a candidate mask as a translucent fill with a solid outline."""
        self._show_mask(self._prev, mask, color, fill_alpha)

    def set_selection(self, mask: np.ndarray | None):
        """Outline the selected object."""
        self._show_mask(self._sel, mask, (255, 255, 255), 0)

    @staticmethod
    def _show_mask(item, mask, color, fill_alpha):
        if mask is None:
            item.setPixmap(QPixmap())
            return
        ys, xs = np.flatnonzero(mask.any(1)), np.flatnonzero(mask.any(0))
        if not len(ys):
            item.setPixmap(QPixmap())
            return
        y0, y1, x0, x1 = ys[0], ys[-1] + 1, xs[0], xs[-1] + 1
        m = np.pad(mask[y0:y1, x0:x1], 1)
        inner = m[1:-1, 1:-1] & m[:-2, 1:-1] & m[2:, 1:-1] & m[1:-1, :-2] & m[1:-1, 2:]
        m = m[1:-1, 1:-1]
        rgba = np.zeros((*m.shape, 4), np.uint8)
        rgba[m] = (*color, fill_alpha)
        rgba[m & ~inner] = (*color, 255)
        item.setPixmap(_pixmap(rgba, QImage.Format.Format_RGBA8888))
        item.setOffset(float(x0), float(y0))

    def set_prompts(self, points, box):
        self.points, self.box = list(points), box
        self.viewport().update()

    def fit(self):
        h, w = self._shape
        if h and w:
            self.fitInView(QRectF(0, 0, w, h), Qt.AspectRatioMode.KeepAspectRatio)
            self._update_smoothing()

    def _update_smoothing(self):
        smooth = self.transform().m11() < 1.0  # crisp pixels when zoomed in
        mode = Qt.TransformationMode.SmoothTransformation if smooth else Qt.TransformationMode.FastTransformation
        self._img.setTransformationMode(mode)

    # ---- helpers -------------------------------------------------------

    def _image_pos(self, view_pos) -> QPointF:
        return self.mapToScene(view_pos)

    def _inside(self, p: QPointF) -> bool:
        h, w = self._shape
        return 0 <= p.x() < w and 0 <= p.y() < h

    # ---- events --------------------------------------------------------

    def wheelEvent(self, ev):
        dy = ev.angleDelta().y()
        if not dy:
            return
        factor = 1.0015**dy
        scale = self.transform().m11() * factor
        if 0.01 < scale < 80:
            self.scale(factor, factor)
            self._update_smoothing()

    def mousePressEvent(self, ev):
        self.setFocus()
        pos = ev.position().toPoint()
        btn = ev.button()
        if btn == Qt.MouseButton.MiddleButton or (btn == Qt.MouseButton.LeftButton and self._space):
            self._pan = pos
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
        elif btn == Qt.MouseButton.LeftButton:
            ctrl = bool(ev.modifiers() & Qt.KeyboardModifier.ControlModifier)
            shift = bool(ev.modifiers() & Qt.KeyboardModifier.ShiftModifier)
            if ctrl or shift:
                p = self._image_pos(pos)
                if self._inside(p):
                    self.modClicked.emit(p.x(), p.y(), ctrl, shift)
            else:
                self._press = pos
        elif btn == Qt.MouseButton.RightButton:
            p = self._image_pos(pos)
            if self._inside(p):
                self.pointClicked.emit(p.x(), p.y(), False)

    def mouseMoveEvent(self, ev):
        pos = ev.position().toPoint()
        if self._pan is not None:
            d = pos - self._pan
            self._pan = pos
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - d.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - d.y())
            return
        if self._press is not None:
            if self._drag_box is None and (pos - self._press).manhattanLength() < DRAG_THRESHOLD:
                return
            self._drag_box = QRectF(self._image_pos(self._press), self._image_pos(pos)).normalized()
            self.viewport().update()
            return
        p = self._image_pos(pos)
        if self._inside(p):
            self.hovered.emit(p.x(), p.y())
        else:
            self.hovered.emit(-1, -1)

    def mouseReleaseEvent(self, ev):
        pos = ev.position().toPoint()
        if self._pan is not None and ev.button() in (Qt.MouseButton.MiddleButton, Qt.MouseButton.LeftButton):
            self._pan = None
            self.setCursor(Qt.CursorShape.OpenHandCursor if self._space else Qt.CursorShape.CrossCursor)
        elif ev.button() == Qt.MouseButton.LeftButton and self._press is not None:
            self._press = None
            if self._drag_box is not None:
                h, w = self._shape
                r = self._drag_box.intersected(QRectF(0, 0, w, h))
                self._drag_box = None
                self.viewport().update()
                if r.width() >= 2 and r.height() >= 2:
                    self.boxDrawn.emit(r.left(), r.top(), r.right(), r.bottom())
            else:
                p = self._image_pos(pos)
                if self._inside(p):
                    self.pointClicked.emit(p.x(), p.y(), True)

    def leaveEvent(self, ev):
        self.hovered.emit(-1, -1)
        super().leaveEvent(ev)

    def keyPressEvent(self, ev):
        if ev.key() == Qt.Key.Key_Space and not ev.isAutoRepeat():
            self._space = True
            if self._pan is None:
                self.setCursor(Qt.CursorShape.OpenHandCursor)
        elif ev.key() != Qt.Key.Key_Space:
            ev.ignore()  # leave everything else to the window's shortcuts

    def keyReleaseEvent(self, ev):
        if ev.key() == Qt.Key.Key_Space and not ev.isAutoRepeat():
            self._space = False
            if self._pan is None:
                self.setCursor(Qt.CursorShape.CrossCursor)
        else:
            ev.ignore()

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        if not ev.oldSize().isValid() or ev.oldSize().isEmpty():
            self.fit()

    # ---- prompt markers, drawn in screen space so they keep their size ----

    def drawForeground(self, painter: QPainter, rect):
        painter.save()
        painter.resetTransform()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        box = self._drag_box
        if box is None and self.box is not None:
            x0, y0, x1, y1 = self.box
            box = QRectF(x0, y0, x1 - x0, y1 - y0)
        if box is not None:
            r = QRectF(self.mapFromScene(box.topLeft()), self.mapFromScene(box.bottomRight()))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(QPen(QColor(0, 0, 0, 160), 3))
            painter.drawRect(r)
            painter.setPen(QPen(QColor("white"), 1.5, Qt.PenStyle.DashLine))
            painter.drawRect(r)
        for x, y, positive in self.points:
            c = QPointF(self.mapFromScene(QPointF(x, y)))
            painter.setPen(QPen(QColor("white"), 1.5))
            painter.setBrush(QColor("#2ecc71") if positive else QColor("#e74c3c"))
            painter.drawEllipse(c, 5, 5)
        painter.restore()
