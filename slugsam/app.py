"""SlugSam main window."""

from __future__ import annotations

import argparse
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
from PySide6.QtCore import QSettings, Qt, QTimer
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QSpinBox,
    QFileDialog,
    QInputDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QSlider,
    QToolBar,
)

from . import tiles
from .auto import grid_masks
from .canvas import Canvas, make_lut
from .data import IMAGE_FILTER, ImageStack, load_image, load_labels, next_image, save_labels, save_mask
from .engine import EmbedWorker, SamModel, find_models, free_gpu_cache, load_model, memory_usage

EMBEDDING_CACHE_BYTES = 768 * 2**20  # device memory for cached embeddings: 4 MB (SAM) or 10 MB (SAM 2) per slice
PREFETCH_RADIUS = 3  # neighbouring slices embedded ahead of time
DEFAULT_UNDO_LIMIT = 50  # object edits kept for undo; 0 means unlimited
VIDEO_ABSENT_FRAMES = 5  # SAM 2 video tracking gives up on an object unseen for this many slices
DEFAULT_MIN_IOU = 0.3  # a tracked object is lost when it overlaps its previous slice less


def interior_point(mask: np.ndarray) -> tuple[float, float, bool]:
    """A positive point prompt on the mask pixel nearest its centroid."""
    ys, xs = np.nonzero(mask)
    k = np.argmin((ys - ys.mean()) ** 2 + (xs - xs.mean()) ** 2)
    return (xs[k] + 0.5, ys[k] + 0.5, True)

HELP = """
<table cellspacing=6>
<tr><td><b>Left click</b></td><td>add a positive point</td></tr>
<tr><td><b>Right click</b></td><td>add a negative point</td></tr>
<tr><td><b>Left drag</b></td><td>draw a box prompt</td></tr>
<tr><td><b>Enter</b></td><td>commit the current mask as a new object</td></tr>
<tr><td><b>Tab</b></td><td>cycle through SAM's alternative masks</td></tr>
<tr><td><b>Esc</b></td><td>stop propagation, else clear prompts, else deselect</td></tr>
<tr><td><b>Ctrl+Z</b></td><td>undo last prompt, or last object edit</td></tr>
<tr><td><b>Ctrl+click</b></td><td>delete the object under the cursor (<b>+Shift</b>: on all slices)</td></tr>
<tr><td><b>Shift+click</b></td><td>select an object: Enter then redraws it, on any slice, keeping its id</td></tr>
<tr><td><b>P / Shift+P</b></td><td>with prompts placed: apply the same prompts on this and all
following / previous slices.<br>Otherwise: track the selected (or hovered, or last) object
through following / previous slices. <i>Tracking</i> in the toolbar picks how: <i>Overlap</i>
re-prompts each slice from the previous mask until the overlap drops below <i>IoU</i>;
<i>SAM 2 video</i> (SAM 2 models only) follows the object with the model's video memory</td></tr>
<tr><td><b>Ctrl+P / Ctrl+Shift+P</b></td><td>track all objects on this slice forward / backward</td></tr>
<tr><td><b>A</b></td><td>auto-segment: prompt a grid of points over the view or the slice</td></tr>
<tr><td><b>Wheel</b></td><td>zoom; on large images, zooming in also makes new masks finer
(<i>View ▸ Adaptive detail</i>)</td></tr>
<tr><td><b>Middle drag / Space+drag</b></td><td>pan</td></tr>
<tr><td><b>R</b></td><td>fit image to window</td></tr>
<tr><td><b>V</b></td><td>show / hide objects</td></tr>
<tr><td><b>H</b></td><td>toggle hover preview</td></tr>
<tr><td><b>, / .</b> or <b>← / →</b></td><td>previous / next slice</td></tr>
<tr><td><b>Ctrl+Shift+O</b></td><td>open the next image in the folder</td></tr>
<tr><td><b>Ctrl+W</b></td><td>close the image and free its memory</td></tr>
<tr><td><b>Ctrl+S</b></td><td>save labels as <i>&lt;image&gt;_labels.tif</i></td></tr>
<tr><td><b>Ctrl+E</b></td><td>export all objects as a binary mask, <i>mask_&lt;image&gt;.tif</i></td></tr>
</table>
"""


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.settings = QSettings("SlugSam", "SlugSam")
        self.lut = make_lut()
        self.model: SamModel | None = None
        self.stack: ImageStack | None = None
        self.stack_id = 0
        self.idx = 0
        self.channel: int | None = None  # None: composite of the first three channels
        self.active: int | None = None  # selected object id; commits redraw it instead of adding
        self.last_id: int | None = None
        self.prop: dict | None = None  # state of a running propagation or auto-segmentation
        self.wanted_tile: tiles.Tile | None = None  # tile worth embedding next, for finer masks
        self.prompt_level: int | None = None  # detail level the object being drawn started at
        self.rgb: np.ndarray | None = None
        self.labels: np.ndarray | None = None
        self.next_id = 1
        self.dirty = False
        self.embeddings: OrderedDict = OrderedDict()
        self.pending_key = None
        # prompt state for the object being drawn
        self.points: list[tuple[float, float, bool]] = []
        self.box = None
        self.choice: int | None = None
        self.prompt_undo: list = []
        self.mask: np.ndarray | None = None
        self.label_undo: list = []

        self.canvas = Canvas()
        self.setCentralWidget(self.canvas)
        self.canvas.pointClicked.connect(self.add_point)
        self.canvas.boxDrawn.connect(self.set_box)
        self.canvas.modClicked.connect(self.on_mod_click)
        self.canvas.hovered.connect(self.on_hover)
        self._view_timer = QTimer(self, singleShot=True, interval=250)
        self._view_timer.timeout.connect(self._view_settled)
        self.canvas.viewChanged.connect(self._view_timer.start)

        self.worker = EmbedWorker(self)
        self.worker.done.connect(self.on_embedded)
        self.worker.failed.connect(self.on_embed_failed)
        self.worker.start()

        self._hover_pos = None
        self._hover_timer = QTimer(self, singleShot=True, interval=10)
        self._hover_timer.timeout.connect(self._do_hover)

        self._build_ui()
        self.setAcceptDrops(True)
        self.resize(1200, 850)
        self._update_title()
        self.statusBar().showMessage("Open an image (Ctrl+O) or drop one here")

    # ---- UI ------------------------------------------------------------

    def _action(self, text, slot, *shortcuts, checkable=False, checked=False):
        act = QAction(text, self)
        if shortcuts:
            act.setShortcuts([QKeySequence(s) for s in shortcuts])
        if checkable:
            act.setCheckable(True)
            act.setChecked(checked)
            act.toggled.connect(slot)
        else:
            act.triggered.connect(lambda _=False: slot())
        self.addAction(act)
        return act

    def _build_ui(self):
        a_open = self._action("&Open image…", self.open_dialog, QKeySequence.StandardKey.Open)
        a_next_img = self._action("Open &next image in folder", self.open_next, "Ctrl+Shift+O")
        a_close = self._action("&Close image", self.close_image, "Ctrl+W")
        a_save = self._action("&Save labels", self.save, QKeySequence.StandardKey.Save)
        a_save_as = self._action("Save labels &as…", self.save_as, "Ctrl+Shift+S")
        a_export = self._action("&Export binary mask", self.export_mask, "Ctrl+E")
        a_export.setToolTip("Save all objects as one 0/255 mask, mask_<image>.tif")
        a_clear_labels = self._action("Clear la&bels", self.clear_labels)
        a_clear_labels.setToolTip("Remove all objects (Ctrl+Z restores them)")
        a_quit = self._action("&Quit", self.close, "Ctrl+Q")
        self.a_lazy = self._action(
            "&Lazy loading (keep stacks on disk)",
            lambda on: self.settings.setValue("lazy", on),
            checkable=True,
            checked=self.settings.value("lazy", False, type=bool),
        )
        self.a_lazy.setToolTip("Applies to the next image opened")
        a_undo_limit = self._action("Undo history &limit…", self.ask_undo_limit)
        a_commit = self._action("&Commit object", self.commit, Qt.Key.Key_Return, Qt.Key.Key_Enter)
        a_cycle = self._action("Next &alternative mask", self.cycle_mask, Qt.Key.Key_Tab)
        a_clear = self._action("C&lear prompts", self.clear_prompts, Qt.Key.Key_Escape)
        a_undo = self._action("&Undo", self.undo, QKeySequence.StandardKey.Undo)
        a_prop = [
            self._action("Propagate object &forward", lambda: self.propagate(1), "P"),
            self._action("Propagate object &backward", lambda: self.propagate(-1), "Shift+P"),
            self._action("Propagate all objects forward", lambda: self.propagate(1, True), "Ctrl+P"),
            self._action("Propagate all objects backward", lambda: self.propagate(-1, True), "Ctrl+Shift+P"),
        ]
        a_fit = self._action("&Fit to window", self.canvas.fit, "R")
        self.a_show = self._action(
            "Show &objects", self.canvas.set_labels_visible, "V", checkable=True, checked=True
        )
        hover_on = self.settings.value("hover", True, type=bool)
        self.a_hover = self._action(
            "&Hover preview", self._hover_toggled, "H", checkable=True, checked=hover_on
        )
        self.a_detail = self._action(
            "Adaptive &detail (finer masks when zoomed in)",
            self._detail_toggled,
            checkable=True,
            checked=self.settings.value("detail", True, type=bool),
        )
        a_auto = self._action("Auto-&segment…", self.ask_auto, "A")
        a_prev = self._action("&Previous slice", lambda: self.step_slice(-1), ",", Qt.Key.Key_Left)
        a_next = self._action("&Next slice", lambda: self.step_slice(1), ".", Qt.Key.Key_Right)
        a_help = self._action("&Shortcuts", self.show_help, "F1")

        mb = self.menuBar()
        m = mb.addMenu("&File")
        m.addActions([a_open, a_next_img, a_close])
        m.addSeparator()
        m.addActions([a_save, a_save_as, a_export])
        m.addSeparator()
        m.addAction(self.a_lazy)
        m.addSeparator()
        m.addAction(a_quit)
        m = mb.addMenu("&Edit")
        m.addActions([a_commit, a_cycle, a_clear, a_undo])
        m.addAction(a_undo_limit)
        m.addAction(a_clear_labels)
        m.addSeparator()
        m.addAction(a_auto)
        m.addActions(a_prop)
        m = mb.addMenu("&View")
        m.addActions([a_fit, self.a_show, self.a_hover, self.a_detail])
        m.addSeparator()
        m.addActions([a_prev, a_next])
        mb.addMenu("&Help").addAction(a_help)

        tb = QToolBar("Main")
        tb.setMovable(False)
        self.addToolBar(tb)
        tb.addAction(a_open)
        tb.addAction(a_save)
        tb.addAction(a_clear_labels)
        tb.addSeparator()

        tb.addWidget(QLabel(" Model "))
        self.model_combo = QComboBox()
        self.model_combo.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.model_combo.setMinimumWidth(130)
        for p in find_models():
            self.model_combo.addItem(p.name, str(p))
        self.model_combo.addItem("Browse…", None)
        self.model_combo.setCurrentIndex(-1)
        self.model_combo.activated.connect(self._model_chosen)
        tb.addWidget(self.model_combo)
        self.channel_combo = QComboBox()
        self.channel_combo.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.channel_combo.setToolTip("Channel shown and used for segmentation")
        self.channel_combo.activated.connect(self._channel_chosen)
        self.channel_actions = [tb.addWidget(QLabel(" Channel ")), tb.addWidget(self.channel_combo)]
        for a in self.channel_actions:
            a.setVisible(False)
        tb.addSeparator()

        tb.addWidget(QLabel(" Opacity "))
        self.opacity = QSlider(Qt.Orientation.Horizontal)
        self.opacity.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.opacity.setRange(0, 100)
        self.opacity.setValue(45)
        self.opacity.setFixedWidth(110)
        self.opacity.valueChanged.connect(lambda v: self.canvas.set_label_opacity(v / 100))
        tb.addWidget(self.opacity)

        self.hover_box = QCheckBox("Hover preview")
        self.hover_box.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.hover_box.setChecked(hover_on)
        self.hover_box.toggled.connect(self.a_hover.setChecked)
        self.a_hover.toggled.connect(self.hover_box.setChecked)
        tb.addWidget(self.hover_box)

        self.slice_actions = [tb.addSeparator(), tb.addWidget(QLabel(" Slice "))]
        self.slice_slider = QSlider(Qt.Orientation.Horizontal)
        self.slice_slider.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.slice_slider.setMinimumWidth(160)
        self.slice_slider.valueChanged.connect(self.set_slice)
        self.slice_actions.append(tb.addWidget(self.slice_slider))
        self.slice_label = QLabel()
        self.slice_actions.append(tb.addWidget(self.slice_label))
        self.slice_actions.append(tb.addWidget(QLabel(" Tracking ")))
        self.track_mode = QComboBox()
        self.track_mode.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.track_mode.addItems(["Overlap", "SAM 2 video"])
        self.track_mode.setToolTip(
            "How P follows an object through the stack when no prompts are placed.\n"
            "Overlap: prompt each slice from the previous mask; stops below the IoU threshold.\n"
            "SAM 2 video: SAM 2's video memory; needs a SAM 2 model, copes with faster motion."
        )
        self.track_mode.setCurrentIndex(self.settings.value("track_mode", 0, type=int))
        self.track_mode.currentIndexChanged.connect(self._track_mode_changed)
        self.slice_actions.append(tb.addWidget(self.track_mode))
        self.slice_actions.append(tb.addWidget(QLabel(" IoU ≥ ")))
        self.min_iou = QDoubleSpinBox()
        self.min_iou.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
        self.min_iou.setRange(0.0, 0.95)
        self.min_iou.setSingleStep(0.05)
        self.min_iou.setValue(self.settings.value("min_iou", DEFAULT_MIN_IOU, type=float))
        self.min_iou.setToolTip(
            "Tracking (P) drops an object when its mask overlaps the previous slice's mask\n"
            "less than this (intersection over union). 0 keeps following whatever is found."
        )
        self.min_iou.valueChanged.connect(lambda v: self.settings.setValue("min_iou", v))
        self.slice_actions.append(tb.addWidget(self.min_iou))
        self.min_iou.setEnabled(self.track_mode.currentIndex() == 0)
        for a in self.slice_actions:
            a.setVisible(False)

        self.info = QLabel()
        self.statusBar().addPermanentWidget(self.info)
        self.mem = QLabel()
        self.statusBar().addPermanentWidget(self.mem)
        self._mem_timer = QTimer(self, interval=2000)
        self._mem_timer.timeout.connect(self._update_memory)
        self._mem_timer.start()
        self._update_memory()

    def _update_memory(self):
        m, gb = memory_usage(), 2**30
        parts, tip = [], []
        if "gpu_ours" in m:
            held = f"held by SlugSam: {m['gpu_ours'] / gb:.2f} GB, of which live " \
                   f"(model + {len(self.embeddings)} cached embeddings): {m['gpu_live'] / gb:.2f} GB"
            if "gpu_total" in m:
                parts.append(f"GPU {m['gpu_used'] / gb:.1f}/{m['gpu_total'] / gb:.1f} GB")
                tip.append(f"GPU memory in use on the whole device: {m['gpu_used'] / gb:.2f} GB\n  {held}")
            else:
                parts.append(f"GPU {m['gpu_ours'] / gb:.1f} GB")
                tip.append(f"GPU memory {held}")
        if "ram" in m:
            parts.append(f"RAM {m['ram'] / gb:.1f} GB")
            tip.append(f"Main memory used by SlugSam: {m['ram'] / gb:.2f} GB")
        self.mem.setText("  " + " · ".join(parts) + " ")
        self.mem.setToolTip("\n".join(tip))

    def _track_mode_changed(self, i: int):
        self.settings.setValue("track_mode", i)
        self.min_iou.setEnabled(i == 0)

    def _update_title(self):
        name = f"{self.stack.path.name}{'*' if self.dirty else ''} — " if self.stack else ""
        self.setWindowTitle(f"{name}SlugSam")
        parts = []
        if self.labels is not None:
            parts.append(f"{len(np.unique(self.labels[self.idx])) - (0 in self.labels[self.idx])} objects")
        if self.active:
            parts.insert(0, f"editing object {self.active} (Esc to deselect)")
        if self.model:
            parts.append(f"{self.model.model_type} on {self.model.device}")
        self.info.setText("  |  ".join(parts))

    def show_help(self):
        QMessageBox.information(self, "SlugSam shortcuts", HELP)

    def _hover_toggled(self, on: bool):
        self.settings.setValue("hover", on)
        if not on and not self._has_prompts():
            self.canvas.set_preview(None)

    # ---- model ---------------------------------------------------------

    def _model_chosen(self, i: int):
        path = self.model_combo.itemData(i)
        if path is None:  # Browse…
            start = self.settings.value("model_dir", str(Path.home()))
            path, _ = QFileDialog.getOpenFileName(self, "Select SAM checkpoint", start)
            if not path:
                self._sync_model_combo()
                return
            self.settings.setValue("model_dir", str(Path(path).parent))
        self.load_model(path)

    def _sync_model_combo(self):
        i = self.model_combo.findData(str(self.model.path)) if self.model else -1
        self.model_combo.setCurrentIndex(i)

    def load_model(self, path: str | Path):
        path = Path(path)
        self.statusBar().showMessage(f"Loading {path.name}…")
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        QApplication.processEvents()
        try:
            model = load_model(path)
        except Exception as e:
            QMessageBox.warning(self, "SlugSam", f"Could not load {path.name}:\n{e}")
            self._sync_model_combo()
            self.statusBar().clearMessage()
            return
        finally:
            QApplication.restoreOverrideCursor()
        self.model = model
        self.embeddings.clear()
        self.pending_key = None
        free_gpu_cache()  # hand the previous model's memory back
        if self.model_combo.findData(str(path)) < 0:
            self.model_combo.insertItem(self.model_combo.count() - 1, path.name, str(path))
        self._sync_model_combo()
        self.settings.setValue("model", str(path))
        self.statusBar().showMessage(f"Loaded {path.name}", 3000)
        self._update_title()
        self._ensure_embedding()
        self._update_prediction()

    def load_default_model(self, name: str | None = None):
        """Pick a model by name/path, else the last one used, else a sensible cached one."""
        if name:
            if Path(name).is_file():
                return self.load_model(name)
            i = self.model_combo.findText(name)
            if i >= 0:
                return self.load_model(self.model_combo.itemData(i))
            self.statusBar().showMessage(f"Model '{name}' not found")
        last = self.settings.value("model", "")
        if last and Path(last).is_file():
            return self.load_model(last)
        for pref in ("vit_b_lm", "vit_b", "vit_l_lm", "vit_l"):
            i = self.model_combo.findText(pref)
            if i >= 0:
                return self.load_model(self.model_combo.itemData(i))
        if self.model_combo.count() > 1:
            return self.load_model(self.model_combo.itemData(0))
        self.statusBar().showMessage("No SAM checkpoint found — pick one via Model ▸ Browse…")

    # ---- image ---------------------------------------------------------

    def open_dialog(self):
        start = self.settings.value("image_dir", str(Path.home()))
        path, _ = QFileDialog.getOpenFileName(self, "Open image", start, IMAGE_FILTER)
        if path:
            self.open_image(path)

    def open_next(self):
        if self.stack is None:
            self.open_dialog()
            return
        nxt = next_image(self.stack.path)
        if nxt is None:
            self.statusBar().showMessage(f"{self.stack.path.name} is the last image in its folder", 4000)
        else:
            self.open_image(nxt)

    def close_image(self):
        if self.stack is None:
            return
        if self.prop:
            self._propagate_finish(stopped=True)
        if not self._confirm_discard():
            return
        self._reset_prompts()
        self.stack.close()
        self.stack = self.labels = self.rgb = None
        self.stack_id += 1  # embeddings still being computed for it are dropped on arrival
        self.embeddings.clear()
        self.pending_key = None
        self.label_undo.clear()
        self.active = self.last_id = None
        self.dirty = False
        self.canvas.clear()
        for a in self.slice_actions + self.channel_actions:
            a.setVisible(False)
        free_gpu_cache()
        self._update_title()
        self._update_memory()
        self.statusBar().showMessage("Image closed")

    def open_image(self, path: str | Path):
        if self.prop:
            self._propagate_finish(stopped=True)
        if not self._confirm_discard():
            return
        path = Path(path)
        try:
            stack = load_image(path, lazy=self.a_lazy.isChecked())
        except Exception as e:
            QMessageBox.warning(self, "SlugSam", f"Could not open {path.name}:\n{e}")
            return
        self.settings.setValue("image_dir", str(path.parent))
        if self.stack is not None:
            self.stack.close()
        self.stack = stack
        self.stack_id += 1
        self.embeddings.clear()
        self.pending_key = None
        free_gpu_cache()
        n, shape = len(stack), stack.frame_shape
        existing = load_labels(stack.labels_path, n, shape)
        # A fresh label array is left untouched: the OS only backs the slices that get edited.
        self.labels = existing if existing is not None else np.zeros((n, *shape), np.uint16)
        self.next_id = int(existing.max()) + 1 if existing is not None else 1
        self.label_undo.clear()
        self.dirty = False
        self.active = self.last_id = self.prop = None
        c = stack.channels
        self.channel_combo.clear()
        if c > 1:
            self.channel_combo.addItems([f"{k + 1}" for k in range(c)] + ["Composite"])
            last = self.settings.value("channel", 0, type=int)
            self.channel = None if stack.is_rgb else min(last, c - 1)
            self.channel_combo.setCurrentIndex(c if self.channel is None else self.channel)
        else:
            self.channel = None
        for a in self.channel_actions:
            a.setVisible(c > 1)
        for a in self.slice_actions:
            a.setVisible(n > 1)
        self.slice_slider.blockSignals(True)
        self.slice_slider.setRange(0, n - 1)
        self.slice_slider.setValue(0)
        self.slice_slider.blockSignals(False)
        self.set_slice(0)
        msg = [f"{path.name}: {shape[1]}×{shape[0]}" + (f", {n} slices" if n > 1 else "")]
        if existing is not None:
            msg.append(f"loaded existing {stack.labels_path.name}")
        if c > 1 and not stack.is_rgb:
            msg.append(f"{c} channels, using channel {self.channel + 1}")
        if stack.mode != "in memory":
            msg.append(stack.mode)
        self.statusBar().showMessage(" — ".join(msg), 6000)

    def set_slice(self, i: int):
        if self.stack is None:
            return
        self.idx = i
        self.slice_label.setText(f" {i + 1}/{len(self.stack)} ")
        self.rgb = self.stack.frame_rgb(i, self.channel)
        self.canvas.set_image(self.rgb)
        self._reset_prompts()
        self._refresh_labels()
        self.wanted_tile = self._view_tile()
        self._ensure_embedding()

    def _channel_chosen(self, i: int):
        if self.stack is None or self.prop:
            return
        self.channel = None if i >= self.stack.channels else i
        if self.channel is not None:
            self.settings.setValue("channel", self.channel)
        self.embeddings.clear()
        self.pending_key = None
        free_gpu_cache()
        self.set_slice(self.idx)

    def step_slice(self, d: int):
        if self.stack is not None and len(self.stack) > 1 and not self.prop:
            self.slice_slider.setValue(self.slice_slider.value() + d)

    def dragEnterEvent(self, ev):
        if ev.mimeData().hasUrls():
            ev.acceptProposedAction()

    def dropEvent(self, ev):
        urls = [u for u in ev.mimeData().urls() if u.isLocalFile()]
        if urls:
            self.open_image(urls[0].toLocalFile())

    # ---- embeddings ----------------------------------------------------
    # Each slice is embedded as a whole (tile None). On large images, parts of the current
    # slice are additionally embedded as tiles at higher resolution, chosen by the zoom.

    def _key(self, idx: int | None = None, tile: tiles.Tile | None = None):
        return (id(self.model), self.stack_id, self.channel, self.idx if idx is None else idx, tile)

    def _embedding(self, tile: tiles.Tile | None = None):
        if self.model is None or self.stack is None:
            return None
        key = self._key(tile=tile)
        emb = self.embeddings.get(key)
        if emb is not None:
            self.embeddings.move_to_end(key)
        return emb

    def _request(self, key, rgb):
        if key != self.pending_key:
            self.pending_key = key
            self.worker.request(key, self.model, rgb)

    def _crop(self, tile: tiles.Tile) -> np.ndarray:
        y, x, h, w = tile
        return np.ascontiguousarray(self.rgb[y : y + h, x : x + w])

    def _ensure_embedding(self):
        """Ask for the most useful missing embedding: this slice, then a finer tile, then neighbours."""
        if self.model is None or self.stack is None:
            return
        if self.prop:
            return  # propagation and auto-segmentation embed what they need themselves
        if self._embedding() is None:
            self._request(self._key(), self.rgb)
            self.statusBar().showMessage("Computing image embedding… (you can already place prompts)")
        elif self.wanted_tile is not None and self._key(tile=self.wanted_tile) not in self.embeddings:
            self._request(self._key(tile=self.wanted_tile), self._crop(self.wanted_tile))
        else:
            self._prefetch()

    def on_embedded(self, key, emb):
        if key == self.pending_key:
            self.pending_key = None
        if key[:3] == self._key()[:3]:  # else the model, image or channel changed meanwhile
            self._store_embedding(key, emb)
            self._ensure_embedding()
        if self.pending_key is None:
            # The encoder needs gigabytes of scratch space that PyTorch would otherwise keep
            # reserved; give it back once there is nothing left to embed.
            free_gpu_cache()

    def _cache(self, key, emb):
        self.embeddings[key] = emb
        while (
            len(self.embeddings) > 2 * PREFETCH_RADIUS + 1
            and sum(e.nbytes for e in self.embeddings.values()) > EMBEDDING_CACHE_BYTES
        ):
            self.embeddings.popitem(last=False)

    def _embedding_now(self, tile: tiles.Tile | None = None):
        """The embedding of the current slice or one of its tiles, computed on the spot if missing."""
        emb = self._embedding(tile)
        if emb is None:
            emb = self.model.embed(self.rgb if tile is None else self._crop(tile))
            self._cache(self._key(tile=tile), emb)
        return emb

    def _store_embedding(self, key, emb):
        self._cache(key, emb)
        if key[3] != self.idx or self.prop:
            return
        if self._has_prompts():
            self._update_prediction()  # first mask, or the same mask in finer detail
        else:
            if key[4] is None:
                self.statusBar().showMessage("Ready", 2000)
            self._do_hover()

    # ---- detail levels ---------------------------------------------------

    def _detail_toggled(self, on: bool):
        self.settings.setValue("detail", on)
        self._view_settled()

    def _view_level(self) -> int:
        """Detail level matching the zoom: the finest whose tiles still span what is on screen."""
        if self.stack is None or not self.a_detail.isChecked():
            return 0
        shape = self.stack.frame_shape
        x0, y0, x1, y1 = self.canvas.visible_rect()
        span = max(x1 - x0, y1 - y0, 1.0)
        level = 0
        while level < tiles.max_level(shape) and tiles.tile_side(shape, level + 1) >= span:
            level += 1
        return level

    def _view_tile(self) -> tiles.Tile | None:
        """The tile at the zoom's detail level around the centre of the view."""
        if self.stack is None:
            return None
        x0, y0, x1, y1 = self.canvas.visible_rect()
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        return tiles.tile_for(self.stack.frame_shape, self._view_level(), (cx, cy, cx, cy))

    def _view_settled(self):
        if self.stack is None or self.prop:
            return
        if not self._has_prompts():
            self.wanted_tile = self._view_tile()
        self._ensure_embedding()

    def _predict(self, points, box=None, choice=None, level: int | None = None, wait: bool = False):
        """Decode prompts on the finest embedded tile that holds them all.

        Returns (full-frame mask, score, number of candidates, detail factor), or None while
        nothing is embedded yet. If a finer tile than the one used would fit, it is requested
        and the caller is run again when it arrives; with wait=True it is computed first.
        """
        shape = self.stack.frame_shape
        level = self._view_level() if level is None else level
        xs = [p[0] for p in points] + ([box[0], box[2]] if box else [])
        ys = [p[1] for p in points] + ([box[1], box[3]] if box else [])
        bbox = (min(xs), min(ys), max(xs), max(ys))
        if wait:
            tile = tiles.finest_tile(shape, bbox, level)
            return self._predict_on(self._embedding_now(tile), tile, points, box, choice)
        emb, tile, wanted = self._embedding(), None, None
        for lvl in range(level, 0, -1):
            t = tiles.tile_for(shape, lvl, bbox)
            if t is None:
                continue
            wanted = wanted or t
            e = self._embedding(t)
            if e is not None:
                emb, tile = e, t
                break
        if wanted is not None and wanted != tile:
            self.wanted_tile = wanted
            self._ensure_embedding()
        if emb is None:
            return None
        return self._predict_on(emb, tile, points, box, choice)

    def _predict_on(self, emb, tile, points, box, choice):
        shape = self.stack.frame_shape
        if tile is None:
            return (*self.model.predict(emb, points, box, choice), 1)
        y, x, h, w = tile
        points = [(px - x, py - y, pos) for px, py, pos in points]
        if box:
            box = (box[0] - x, box[1] - y, box[2] - x, box[3] - y)
        part, score, n = self.model.predict(emb, points, box, choice)
        mask = np.zeros(shape, bool)
        mask[y : y + h, x : x + w] = part
        return mask, score, n, round(max(shape) / max(h, w))

    def on_embed_failed(self, key, msg):
        if key == self.pending_key:
            self.pending_key = None
        self.statusBar().showMessage(f"Embedding failed: {msg}")

    def _prefetch(self):
        """Embed the nearest not-yet-embedded neighbouring slice while the user works."""
        if self.pending_key is not None or len(self.stack) == 1:
            return
        for d in range(1, PREFETCH_RADIUS + 1):
            for j in (self.idx + d, self.idx - d):
                if 0 <= j < len(self.stack) and self._key(j) not in self.embeddings:
                    self.pending_key = self._key(j)
                    rgb = self.stack.frame_rgb(j, self.channel)
                    self.worker.request(self.pending_key, self.model, rgb)
                    return

    # ---- prompts -------------------------------------------------------

    def _has_prompts(self):
        return bool(self.points) or self.box is not None

    def _snapshot(self):
        if not self._has_prompts():
            self.prompt_level = self._view_level()  # zooming later must not change this object
        self.prompt_undo.append((list(self.points), self.box, self.choice))

    def add_point(self, x: float, y: float, positive: bool):
        if self.stack is None or self.prop:
            return
        self._snapshot()
        self.points.append((x, y, positive))
        self._update_prediction()

    def set_box(self, x0, y0, x1, y1):
        if self.stack is None or self.prop:
            return
        self._snapshot()
        self.box = (x0, y0, x1, y1)
        self._update_prediction()

    def cycle_mask(self):
        if not self._has_prompts() or self._embedding() is None:
            return
        self._snapshot()
        self.choice = ((self.choice or 0) + 1) % 3
        self._update_prediction()

    def clear_prompts(self):
        """Esc: stop a running propagation, else clear prompts, else deselect."""
        if self.prop:
            self._propagate_finish(stopped=True)
            return
        if not self._has_prompts() and self.active:
            self.active = None
            self._refresh_selection()
            self._update_title()
        self._reset_prompts()
        self.statusBar().clearMessage()

    def _reset_prompts(self):
        self.points, self.box, self.choice, self.mask = [], None, None, None
        self.prompt_level = None
        self.prompt_undo.clear()
        self.canvas.set_prompts([], None)
        self.canvas.set_preview(None)

    def _color(self):
        return tuple(int(c) for c in self.lut[(self.active or self.next_id) & 0xFFFF][:3])

    def _update_prediction(self):
        self.canvas.set_prompts(self.points, self.box)
        if not self._has_prompts():
            self.mask = None
            self.canvas.set_preview(None)
            return
        if self.model is None:
            return
        result = self._predict(self.points, self.box, self.choice, self.prompt_level)
        if result is None:
            return
        self.mask, score, n, detail = result
        self.canvas.set_preview(self.mask, self._color(), 110)
        alt = f"alternative {(self.choice or 0) + 1}/{n}, " if n > 1 else ""
        if detail > 1:
            alt += f"detail ×{detail}, "
        prop = ", P to apply these prompts on following slices" if len(self.stack) > 1 else ""
        self.statusBar().showMessage(
            f"Mask: {alt}score {score:.2f} — Enter to commit{prop}, Tab for alternatives, Esc to clear"
        )

    def on_hover(self, x: float, y: float):
        self._hover_pos = (x, y)
        if not self._hover_timer.isActive():
            self._hover_timer.start()

    def _do_hover(self):
        if self._has_prompts() or self.prop or not self.a_hover.isChecked():
            return
        if self.model is None or self.stack is None or self._hover_pos is None:
            return
        x, y = self._hover_pos
        if x < 0 or self.labels[self.idx][int(y), int(x)]:
            self.canvas.set_preview(None)
            return
        result = self._predict([(x, y, True)])
        if result is not None:
            self.canvas.set_preview(result[0], self._color(), 45)

    # ---- objects -------------------------------------------------------

    def _refresh_labels(self):
        self.canvas.set_labels(self.lut[self.labels[self.idx] & 0xFFFF])
        self._refresh_selection()
        self._update_title()

    def _refresh_selection(self):
        mask = self.labels[self.idx] == self.active if self.active else None
        self.canvas.set_selection(mask)

    def _edit_labels(self, idx: int, region: np.ndarray, value: int, group: list):
        """Write value into labels[idx][region], appending an undo patch to group."""
        ys, xs = np.flatnonzero(region.any(1)), np.flatnonzero(region.any(0))
        if not len(ys):
            return
        sl = (slice(ys[0], ys[-1] + 1), slice(xs[0], xs[-1] + 1))
        lab = self.labels[idx]
        group.append((idx, sl, lab[sl].copy()))
        lab[sl][region[sl]] = value
        self.dirty = True

    def _redraw_object(self, idx: int, obj: int, mask: np.ndarray, group: list) -> np.ndarray | None:
        """Make mask the shape of obj on slice idx, without overwriting other objects."""
        lab = self.labels[idx]
        region = mask & ((lab == 0) | (lab == obj))
        if not region.any():
            return None
        self._edit_labels(idx, lab == obj, 0, group)
        self._edit_labels(idx, region, obj, group)
        return region

    def _target_id(self) -> int:
        """Id the current mask will be written as: the selected object, or a new one."""
        obj = self.active or self.next_id
        if obj > np.iinfo(self.labels.dtype).max:
            self.labels = self.labels.astype(np.uint32)
        return obj

    def _push_undo(self, group: list, next_id: int):
        if group:
            self.label_undo.append((group, next_id))
            self._trim_undo()

    def _trim_undo(self):
        limit = self.settings.value("undo_limit", DEFAULT_UNDO_LIMIT, type=int)
        if limit > 0:
            del self.label_undo[:-limit]

    def ask_undo_limit(self):
        limit, ok = QInputDialog.getInt(
            self,
            "Undo history",
            "Number of object edits kept for undo (0 = unlimited).\n"
            "A propagation counts as one edit.",
            self.settings.value("undo_limit", DEFAULT_UNDO_LIMIT, type=int),
            0,
            100000,
        )
        if ok:
            self.settings.setValue("undo_limit", limit)
            self._trim_undo()

    def commit(self):
        if self.mask is None or self.prop:
            return
        obj = self._target_id()
        group = []
        if self._redraw_object(self.idx, obj, self.mask, group) is None:
            self.statusBar().showMessage("Nothing to add: the mask is empty or fully covered", 4000)
            return
        self._push_undo(group, self.next_id)
        if self.active:
            self.statusBar().showMessage(f"Redrew object {obj} on this slice", 2000)
        else:
            self.statusBar().showMessage(f"Added object {obj}", 2000)
            self.next_id += 1
        self.last_id = obj
        self._reset_prompts()
        self._refresh_labels()

    def on_mod_click(self, x: float, y: float, ctrl: bool, shift: bool):
        if self.labels is None or self.prop:
            return
        obj = int(self.labels[self.idx][int(y), int(x)])
        if ctrl:
            if obj:
                self.delete_object(obj, all_slices=shift)
        else:
            self.active = obj or None
            self._reset_prompts()
            self._refresh_selection()
            self._update_title()
            if obj:
                self.statusBar().showMessage(
                    f"Object {obj} selected: Enter redraws it here, P propagates it", 5000
                )

    def delete_object(self, obj: int, all_slices: bool = False):
        group = []
        for j in range(len(self.stack)) if all_slices else [self.idx]:
            self._edit_labels(j, self.labels[j] == obj, 0, group)
        self._push_undo(group, self.next_id)
        where = f"on {len(group)} slices" if all_slices else "on this slice"
        self.statusBar().showMessage(f"Deleted object {obj} {where} (Ctrl+Z to restore)", 3000)
        self._refresh_labels()

    def undo(self):
        if self.prop:
            return
        if self.prompt_undo:
            self.points, self.box, self.choice = self.prompt_undo.pop()
            self._update_prediction()
            if not self._has_prompts():
                self.statusBar().clearMessage()
        elif self.label_undo:
            group, next_id = self.label_undo.pop()
            for idx, sl, patch in reversed(group):
                self.labels[idx][sl] = patch
            self.next_id = next_id
            self.dirty = True
            touched = {g[0] for g in group}
            if self.idx not in touched:
                self.slice_slider.setValue(group[0][0])
            else:
                self._refresh_labels()
            self.statusBar().showMessage(f"Undid edit on {len(touched)} slice(s)", 2000)

    # ---- propagation across slices --------------------------------------

    def propagate(self, direction: int, all_objects: bool = False):
        """Follow objects from this slice through the stack, re-prompting SAM from each previous mask."""
        # Objects must overlap themselves between consecutive slices to be followed.
        if self.stack is None or self.model is None or self.prop or len(self.stack) == 1:
            return
        lab = self.labels[self.idx]
        if self._has_prompts() and not all_objects:
            self._propagate_static(direction)
            return
        if all_objects:
            ids = [int(i) for i in np.unique(lab) if i]
        else:
            obj = self.active
            if not obj and self._hover_pos and self._hover_pos[0] >= 0:
                obj = int(lab[int(self._hover_pos[1]), int(self._hover_pos[0])])
            obj = obj or self.last_id
            ids = [obj] if obj and (lab == obj).any() else []
            if ids:
                self.active = obj  # stays selected so it can be fixed where tracking stops
        if not ids:
            self.statusBar().showMessage(
                "Nothing to propagate: select (Shift+click) or hover an object on this slice", 5000
            )
            return
        self._reset_prompts()
        if self.track_mode.currentIndex() == 1:
            self._propagate_video(direction, ids)
            return
        self.slice_slider.setEnabled(False)
        self.prop = {
            "dir": direction,
            "masks": {i: lab == i for i in ids},
            "n": len(ids),
            "group": [],
            "next_id": self.next_id,
        }
        self._propagate_advance()

    def _propagate_static(self, direction: int):
        """Apply the current prompts unchanged on this and every following slice, as one object."""
        if self.mask is None:
            return  # embedding not ready yet
        obj, group, next_id = self._target_id(), [], self.next_id
        if self._redraw_object(self.idx, obj, self.mask, group) is None:
            self.statusBar().showMessage("Nothing to add: the mask is empty or fully covered", 4000)
            return
        if not self.active:
            self.next_id += 1
        self.active = self.last_id = obj
        static = (list(self.points), self.box, self.choice, self.prompt_level)
        self._reset_prompts()
        self._refresh_labels()
        self.slice_slider.setEnabled(False)
        self.prop = {
            "dir": direction,
            "masks": {obj: None},
            "n": 1,
            "group": group,
            "next_id": next_id,
            "static": static,
        }
        self._propagate_advance()

    def _propagate_advance(self):
        p = self.prop
        if p is None:
            return
        j = self.idx + p["dir"]
        if not p["masks"] or not 0 <= j < len(self.stack):
            self._propagate_finish()
            return
        self.slice_slider.setValue(j)
        QTimer.singleShot(0, self._propagate_work)

    def _track_by_overlap(self, prev: np.ndarray):
        """Re-find an object on the current slice from its mask on the previous one.

        Prompts with a point inside the previous mask and keeps whichever of the model's
        candidates matches that mask best. No box: it would pin a moving object in place.
        Works on the finest tile that holds the object with room to move, so the result is as
        detailed as a mask drawn zoomed in. Returns (full-frame mask, IoU with prev).
        """
        shape = self.stack.frame_shape
        ys, xs = np.flatnonzero(prev.any(1)), np.flatnonzero(prev.any(0))
        mh, mw = (ys[-1] - ys[0] + 1) / 2, (xs[-1] - xs[0] + 1) / 2  # margin: half its size each way
        bbox = (
            max(xs[0] - mw, 0),
            max(ys[0] - mh, 0),
            min(xs[-1] + 1 + mw, shape[1]),
            min(ys[-1] + 1 + mh, shape[0]),
        )
        level = tiles.max_level(shape) if self.a_detail.isChecked() else 0
        y, x, h, w = tile = tiles.finest_tile(shape, bbox, level) or (0, 0, *shape)
        prev_part = prev[y : y + h, x : x + w]
        px, py, _ = interior_point(prev_part)
        emb = self._embedding_now(tile if (h, w) != shape else None)
        cands = self.model.predict_candidates(emb, [(px, py, True)])
        ious = [(m & prev_part).sum() / max((m | prev_part).sum(), 1) for m in cands]
        mask = np.zeros(shape, bool)
        mask[y : y + h, x : x + w] = cands[int(np.argmax(ious))]
        return mask, max(ious)

    def _propagate_work(self):
        p = self.prop
        if p is None or "video" in p or "auto" in p:
            return
        try:
            self._propagate_slice(p)
        except Exception as e:  # e.g. CUDA out of memory
            self._propagate_finish(error=f"{type(e).__name__}: {e}")
            return
        self._refresh_labels()
        self.statusBar().showMessage(
            f"Propagating {len(p['masks'])}/{p['n']} object(s), slice {self.idx + 1} — Esc to stop"
        )
        QTimer.singleShot(0, self._propagate_advance)

    def _propagate_slice(self, p):
        static = p.get("static")
        if static:
            (obj,) = p["masks"]
            points, box, choice, level = static
            self.canvas.set_prompts(points, box)
            mask = self._predict(points, box, choice, level, wait=True)[0]
            if self._redraw_object(self.idx, obj, mask, p["group"]) is None:
                p["masks"].clear()  # empty or fully covered: stop here
            return
        min_iou = self.min_iou.value()
        for obj, prev in list(p["masks"].items()):
            mask, iou = self._track_by_overlap(prev)
            region = self._redraw_object(self.idx, obj, mask, p["group"]) if iou >= min_iou else None
            if region is None:
                del p["masks"][obj]  # lost
            else:
                p["masks"][obj] = region

    def _propagate_video(self, direction: int, ids: list[int]):
        """Track objects with SAM 2's video memory, seeded with their masks on this slice."""
        if not self.model.supports_tracking:
            self.statusBar().showMessage(
                "SAM 2 video tracking needs a SAM 2 model: pick one, or set Tracking to Overlap", 6000
            )
            return
        lab, stack, channel = self.labels[self.idx], self.stack, self.channel
        frames = self.model.track(
            lambda i: stack.frame_rgb(i, channel),
            len(stack),
            self.idx,
            {i: lab == i for i in ids},
            reverse=direction < 0,
        )
        self.slice_slider.setEnabled(False)
        self.prop = {
            "dir": direction,
            "masks": dict.fromkeys(ids),  # objects still being followed
            "absent": dict.fromkeys(ids, 0),  # consecutive slices each one was not found on
            "n": len(ids),
            "group": [],
            "next_id": self.next_id,
            "video": frames,
            "start": self.idx,
        }
        self.statusBar().showMessage("Starting SAM 2 video tracking…")
        QTimer.singleShot(0, self._propagate_video_step)

    def _propagate_video_step(self):
        p = self.prop
        if p is None or "video" not in p:
            return
        try:
            frame, masks = next(p["video"])
        except StopIteration:
            self._propagate_finish()
            return
        except Exception as e:  # e.g. CUDA out of memory
            self._propagate_finish(error=f"{type(e).__name__}: {e}")
            return
        if frame != p["start"]:  # the start slice keeps the mask it was seeded with
            for obj in list(p["masks"]):
                mask = masks[obj]
                if mask.any() and self._redraw_object(frame, obj, mask, p["group"]) is not None:
                    p["absent"][obj] = 0
                    p["last"] = frame
                else:
                    p["absent"][obj] += 1
                    if p["absent"][obj] >= VIDEO_ABSENT_FRAMES:
                        del p["masks"][obj]
            self.slice_slider.setValue(frame)
            self.statusBar().showMessage(
                f"Tracking {len(p['masks'])}/{p['n']} object(s), slice {frame + 1} — Esc to stop"
            )
        if p["masks"]:
            QTimer.singleShot(0, self._propagate_video_step)
        else:
            self._propagate_finish()

    # ---- automatic segmentation -------------------------------------------

    def ask_auto(self):
        """Dialog for the grid-prompt segmentation settings; runs it on OK."""
        if self.stack is None or self.model is None or self.prop:
            return
        st, shape = self.settings, self.stack.frame_shape
        dlg = QDialog(self)
        dlg.setWindowTitle("Auto-segment")
        form = QFormLayout(dlg)
        scope = QComboBox()
        scope.addItems(["Visible area", "Whole slice"])
        scope.setCurrentIndex(st.value("auto/scope", 0, type=int))
        detail = QComboBox()
        for lvl in range(tiles.max_level(shape) + 1):
            n = len(tiles.tiles(shape, lvl))
            detail.addItem(f"×{2**lvl}  ({n} tile{'s' if n > 1 else ''} per slice)" if lvl else "×1  (slice as a whole)")
        detail.setCurrentIndex(self._view_level())
        detail.setToolTip("Higher detail finds smaller objects, in more and smaller tiles.\nDefaults to the current zoom.")
        grid = QSpinBox()
        grid.setRange(4, 64)
        grid.setValue(st.value("auto/grid", 32, type=int))
        grid.setToolTip("Points prompted along the long side of each tile")
        score = QDoubleSpinBox()
        score.setRange(0.0, 1.0)
        score.setSingleStep(0.05)
        score.setValue(st.value("auto/score", 0.8, type=float))
        score.setToolTip("Minimum quality the model predicts for a mask.\nFine-tuned models often score lower: try 0.5 if little is found.")
        min_area = QSpinBox()
        min_area.setRange(0, 10**7)
        min_area.setSuffix(" px")
        min_area.setValue(st.value("auto/min_area", 30, type=int))
        max_area = QDoubleSpinBox()
        max_area.setRange(0.01, 100.0)
        max_area.setSuffix(" % of slice")
        max_area.setValue(st.value("auto/max_area", 25.0, type=float))
        form.addRow("Area", scope)
        form.addRow("Detail", detail)
        form.addRow("Grid points per side", grid)
        form.addRow("Min. quality", score)
        form.addRow("Min. object size", min_area)
        form.addRow("Max. object size", max_area)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        form.addRow(buttons)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        st.setValue("auto/scope", scope.currentIndex())
        st.setValue("auto/grid", grid.value())
        st.setValue("auto/score", score.value())
        st.setValue("auto/min_area", min_area.value())
        st.setValue("auto/max_area", max_area.value())
        self.auto_segment(
            visible_only=scope.currentIndex() == 0,
            level=detail.currentIndex(),
            points_per_side=grid.value(),
            min_score=score.value(),
            min_area=min_area.value(),
            max_area_pct=max_area.value(),
        )

    def auto_segment(self, visible_only=True, level=0, points_per_side=32, min_score=0.8, min_area=30, max_area_pct=25.0):
        """Add an object for everything a grid of point prompts finds; existing objects are kept."""
        if self.stack is None or self.model is None or self.prop or self._embedding() is None:
            return
        self._reset_prompts()
        shape = self.stack.frame_shape
        area = self.canvas.visible_rect() if visible_only else (0.0, 0.0, float(shape[1]), float(shape[0]))
        # Each tile handles the grid points in its core, clipped to the requested area.
        todo = []
        for t in tiles.tiles(shape, level) if level else [(0, 0, *shape)]:
            c = tiles.core(shape, t)
            region = (max(c[0], area[0]), max(c[1], area[1]), min(c[2], area[2]), min(c[3], area[3]))
            if region[0] < region[2] and region[1] < region[3]:
                todo.append((t, region))
        self.slice_slider.setEnabled(False)
        self.prop = {"group": [], "next_id": self.next_id, "count": 0}
        self.prop["auto"] = self._auto_steps(
            self.prop, todo, level, area, points_per_side, min_score, min_area, max_area_pct / 100 * shape[0] * shape[1]
        )
        QTimer.singleShot(0, self._auto_step)

    def _auto_steps(self, p, todo, level, area, points_per_side, min_score, min_area, max_area):
        """Generator doing the work in small pieces; yields a status text after each."""
        shape, idx = self.stack.frame_shape, self.idx
        for k, (tile, region) in enumerate(todo):
            where = f"tile {k + 1}/{len(todo)}" if len(todo) > 1 else "slice"
            y, x, h, w = tile
            region = (region[0] - x, region[1] - y, region[2] - x, region[3] - y)
            key = self._key(tile=tile if level else None)
            emb = self.embeddings.get(key)
            if emb is None:
                yield f"Auto-segmenting {where}: embedding… — Esc to stop"
                emb = self.embeddings[key] = self.model.embed(self._crop(tile))
            # masks may only touch a tile border where that is also the border of the image
            edges = (y == 0, x == 0, y + h == shape[0], x + w == shape[1])
            found = []
            for step in grid_masks(self.model, emb, points_per_side, min_score, min_area, max_area, edges, region):
                if isinstance(step, list):
                    found = step
                else:
                    yield f"Auto-segmenting {where}: {step:.0%}, {p['count']} objects so far — Esc to stop"
            if self.next_id + len(found) > np.iinfo(self.labels.dtype).max:
                self.labels = self.labels.astype(np.uint32)
            lab = self.labels[idx]
            for mask, my, mx, _ in found:  # best first
                sl = (slice(y + my, y + my + mask.shape[0]), slice(x + mx, x + mx + mask.shape[1]))
                cy, cx = (sl[0].start + sl[0].stop) / 2, (sl[1].start + sl[1].stop) / 2
                if not (area[0] <= cx <= area[2] and area[1] <= cy <= area[3]):
                    continue
                free = mask & (lab[sl] == 0)
                n = int(free.sum())
                if n < 0.5 * mask.sum() or n < min_area:
                    continue  # mostly an object that is already there
                p["group"].append((idx, sl, lab[sl].copy()))
                lab[sl][free] = self.next_id
                self.next_id += 1
                p["count"] += 1
            self.dirty = True
            self._refresh_labels()
            free_gpu_cache()
            yield f"Auto-segmenting: {where} done, {p['count']} objects so far — Esc to stop"

    def _auto_step(self):
        p = self.prop
        if p is None or "auto" not in p:
            return
        try:
            self.statusBar().showMessage(next(p["auto"]))
        except StopIteration:
            self._propagate_finish()
            return
        except Exception as e:  # e.g. CUDA out of memory
            self._propagate_finish(error=f"{type(e).__name__}: {e}")
            return
        QTimer.singleShot(0, self._auto_step)

    def _auto_finish(self, p, stopped: bool, error: str):
        p["auto"].close()
        self.slice_slider.setEnabled(True)
        self._push_undo(p["group"], p["next_id"])
        self.last_id = None
        self._refresh_labels()
        free_gpu_cache()
        msg = f"Auto-segmentation added {p['count']} object(s)"
        msg += f", failed: {error}" if error else ", stopped" if stopped else ""
        self._ensure_embedding()
        self.statusBar().showMessage(msg + (" — Ctrl+Z undoes it" if p["count"] else ""))

    def _propagate_finish(self, stopped: bool = False, error: str = ""):
        p, self.prop = self.prop, None
        if "auto" in p:
            self._auto_finish(p, stopped, error)
            return
        self.slice_slider.setEnabled(True)
        self.canvas.set_prompts([], None)
        self._push_undo(p["group"], p["next_id"])
        n = len({g[0] for g in p["group"]})
        lost = p["n"] - len(p["masks"])
        msg = f"Propagated {p['n']} object(s) onto {n} slice(s)"
        if "video" in p:
            p["video"].close()  # drops the tracker's memory of the stack
            free_gpu_cache()
            if not stopped and "last" in p:
                self.slice_slider.setValue(p["last"])  # back to where something was last found
        if error:
            msg += f", failed: {error}"
        elif stopped:
            msg += ", stopped"
        elif lost and "video" in p:
            msg += f"; {lost} no longer found" if p["n"] > 1 else f"; last found on slice {self.idx + 1}"
        elif lost:
            msg += f"; {lost} lost on the way" if p["n"] > 1 else f"; lost at slice {self.idx + 1}"
        self._ensure_embedding()
        self.statusBar().showMessage(msg + " — Ctrl+Z undoes it")

    # ---- saving --------------------------------------------------------

    def save(self, path: Path | None = None) -> bool:
        if self.stack is None:
            return False
        path = Path(path) if path else self.stack.labels_path
        try:
            save_labels(path, self.labels)
        except Exception as e:
            QMessageBox.warning(self, "SlugSam", f"Could not save {path}:\n{e}")
            return False
        self.dirty = False
        self._update_title()
        self.statusBar().showMessage(f"Saved {path}", 4000)
        return True

    def export_mask(self):
        if self.stack is None:
            return
        path = self.stack.mask_path
        try:
            save_mask(path, self.labels)
        except Exception as e:
            QMessageBox.warning(self, "SlugSam", f"Could not save {path}:\n{e}")
            return
        self.statusBar().showMessage(f"Exported binary mask to {path}", 4000)

    def clear_labels(self):
        """Remove all objects from this slice or, on request, from the whole stack."""
        if self.stack is None or self.prop:
            return
        slices = [self.idx]
        if len(self.stack) > 1:
            box = QMessageBox(QMessageBox.Icon.Question, "Clear labels", "Remove all objects from:", parent=self)
            this = box.addButton("This slice", QMessageBox.ButtonRole.AcceptRole)
            every = box.addButton("All slices", QMessageBox.ButtonRole.DestructiveRole)
            box.addButton(QMessageBox.StandardButton.Cancel)
            box.exec()
            if box.clickedButton() is every:
                slices = range(len(self.stack))
            elif box.clickedButton() is not this:
                return
        self._clear_labels(slices)

    def _clear_labels(self, slices):
        group, next_id = [], self.next_id
        for j in slices:
            self._edit_labels(j, self.labels[j] != 0, 0, group)
        if not group:
            self.statusBar().showMessage("No objects to remove", 3000)
            return
        if len(slices) == len(self.stack):
            self.next_id = 1  # nothing is left: start numbering afresh
        self._push_undo(group, next_id)
        self.active = self.last_id = None
        self._reset_prompts()
        self._refresh_labels()
        self.statusBar().showMessage(f"Removed all objects from {len(group)} slice(s) — Ctrl+Z restores them")

    def save_as(self):
        if self.stack is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save labels", str(self.stack.labels_path), "TIFF (*.tif *.tiff)"
        )
        if path:
            self.save(path)

    def _confirm_discard(self) -> bool:
        if not self.dirty:
            return True
        B = QMessageBox.StandardButton
        r = QMessageBox.question(
            self, "SlugSam", "Save changes to the labels?", B.Save | B.Discard | B.Cancel, B.Save
        )
        if r == B.Save:
            return self.save()
        return r == B.Discard

    def closeEvent(self, ev):
        if not self._confirm_discard():
            ev.ignore()
            return
        self.worker.stop()
        ev.accept()


def main():
    ap = argparse.ArgumentParser(description="Interactive Segment Anything for microscopy images")
    ap.add_argument("image", nargs="?", help="image to open")
    ap.add_argument("--lazy", action="store_true", help="keep stacks on disk and read slices on demand")
    ap.add_argument("-m", "--model", help="checkpoint path, or name of a cached model (e.g. vit_b_lm)")
    args = ap.parse_args()

    app = QApplication(sys.argv[:1])
    app.setApplicationName("SlugSam")
    win = MainWindow()
    if args.lazy:
        win.a_lazy.setChecked(True)
    win.show()
    if args.image:
        win.open_image(args.image)
    QTimer.singleShot(0, lambda: win.load_default_model(args.model))
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
