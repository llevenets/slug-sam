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
    QFileDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QSlider,
    QToolBar,
)

from .canvas import Canvas, make_lut
from .data import IMAGE_FILTER, ImageStack, load_image, load_labels, save_labels
from .engine import EmbedWorker, SamModel, find_models

MAX_CACHED_EMBEDDINGS = 128  # ~4 MB each, kept on the model device
PREFETCH_RADIUS = 3  # neighbouring slices embedded ahead of time

HELP = """
<table cellspacing=6>
<tr><td><b>Left click</b></td><td>add a positive point</td></tr>
<tr><td><b>Right click</b></td><td>add a negative point</td></tr>
<tr><td><b>Left drag</b></td><td>draw a box prompt</td></tr>
<tr><td><b>Enter</b></td><td>commit the current mask as a new object</td></tr>
<tr><td><b>Tab</b></td><td>cycle through SAM's alternative masks</td></tr>
<tr><td><b>Esc</b></td><td>clear prompts</td></tr>
<tr><td><b>Ctrl+Z</b></td><td>undo last prompt, or last object edit</td></tr>
<tr><td><b>Ctrl+click</b></td><td>delete the object under the cursor</td></tr>
<tr><td><b>Wheel</b></td><td>zoom</td></tr>
<tr><td><b>Middle drag / Space+drag</b></td><td>pan</td></tr>
<tr><td><b>R</b></td><td>fit image to window</td></tr>
<tr><td><b>V</b></td><td>show / hide objects</td></tr>
<tr><td><b>H</b></td><td>toggle hover preview</td></tr>
<tr><td><b>, / .</b> or <b>← / →</b></td><td>previous / next slice</td></tr>
<tr><td><b>Ctrl+S</b></td><td>save labels as <i>&lt;image&gt;_labels.tif</i></td></tr>
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
        self.canvas.pickClicked.connect(self.delete_object_at)
        self.canvas.hovered.connect(self.on_hover)

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
        a_save = self._action("&Save labels", self.save, QKeySequence.StandardKey.Save)
        a_save_as = self._action("Save labels &as…", self.save_as, "Ctrl+Shift+S")
        a_quit = self._action("&Quit", self.close, "Ctrl+Q")
        a_commit = self._action("&Commit object", self.commit, Qt.Key.Key_Return, Qt.Key.Key_Enter)
        a_cycle = self._action("Next &alternative mask", self.cycle_mask, Qt.Key.Key_Tab)
        a_clear = self._action("C&lear prompts", self.clear_prompts, Qt.Key.Key_Escape)
        a_undo = self._action("&Undo", self.undo, QKeySequence.StandardKey.Undo)
        a_fit = self._action("&Fit to window", self.canvas.fit, "R")
        self.a_show = self._action(
            "Show &objects", self.canvas.set_labels_visible, "V", checkable=True, checked=True
        )
        hover_on = self.settings.value("hover", True, type=bool)
        self.a_hover = self._action(
            "&Hover preview", self._hover_toggled, "H", checkable=True, checked=hover_on
        )
        a_prev = self._action("&Previous slice", lambda: self.step_slice(-1), ",", Qt.Key.Key_Left)
        a_next = self._action("&Next slice", lambda: self.step_slice(1), ".", Qt.Key.Key_Right)
        a_help = self._action("&Shortcuts", self.show_help, "F1")

        mb = self.menuBar()
        m = mb.addMenu("&File")
        m.addActions([a_open, a_save, a_save_as])
        m.addSeparator()
        m.addAction(a_quit)
        m = mb.addMenu("&Edit")
        m.addActions([a_commit, a_cycle, a_clear, a_undo])
        m = mb.addMenu("&View")
        m.addActions([a_fit, self.a_show, self.a_hover])
        m.addSeparator()
        m.addActions([a_prev, a_next])
        mb.addMenu("&Help").addAction(a_help)

        tb = QToolBar("Main")
        tb.setMovable(False)
        self.addToolBar(tb)
        tb.addAction(a_open)
        tb.addAction(a_save)
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
        for a in self.slice_actions:
            a.setVisible(False)

        self.info = QLabel()
        self.statusBar().addPermanentWidget(self.info)

    def _update_title(self):
        name = f"{self.stack.path.name}{'*' if self.dirty else ''} — " if self.stack else ""
        self.setWindowTitle(f"{name}SlugSam")
        parts = []
        if self.labels is not None:
            parts.append(f"{len(np.unique(self.labels[self.idx])) - (0 in self.labels[self.idx])} objects")
        if self.model:
            parts.append(f"{self.model.path.name} ({self.model.model_type}, {self.model.device})")
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
            model = SamModel(path)
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

    def open_image(self, path: str | Path):
        if not self._confirm_discard():
            return
        path = Path(path)
        try:
            stack = load_image(path)
        except Exception as e:
            QMessageBox.warning(self, "SlugSam", f"Could not open {path.name}:\n{e}")
            return
        self.settings.setValue("image_dir", str(path.parent))
        self.stack = stack
        self.stack_id += 1
        self.embeddings.clear()
        self.pending_key = None
        n, shape = len(stack), stack.frame_shape
        existing = load_labels(stack.labels_path, n, shape)
        self.labels = existing if existing is not None else np.zeros((n, *shape), np.uint16)
        self.next_id = int(self.labels.max()) + 1
        self.label_undo.clear()
        self.dirty = False
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
        if stack.note:
            msg.append(stack.note)
        self.statusBar().showMessage(" — ".join(msg), 6000)

    def set_slice(self, i: int):
        if self.stack is None:
            return
        self.idx = i
        self.slice_label.setText(f" {i + 1}/{len(self.stack)} ")
        self.rgb = self.stack.frame_rgb(i)
        self.canvas.set_image(self.rgb)
        self._reset_prompts()
        self._refresh_labels()
        self._ensure_embedding()

    def step_slice(self, d: int):
        if self.stack is not None and len(self.stack) > 1:
            self.slice_slider.setValue(self.slice_slider.value() + d)

    def dragEnterEvent(self, ev):
        if ev.mimeData().hasUrls():
            ev.acceptProposedAction()

    def dropEvent(self, ev):
        urls = [u for u in ev.mimeData().urls() if u.isLocalFile()]
        if urls:
            self.open_image(urls[0].toLocalFile())

    # ---- embeddings ----------------------------------------------------

    def _key(self, idx: int | None = None):
        return (id(self.model), self.stack_id, self.idx if idx is None else idx)

    def _embedding(self):
        if self.model is None or self.stack is None:
            return None
        emb = self.embeddings.get(self._key())
        if emb is not None:
            self.embeddings.move_to_end(self._key())
        return emb

    def _ensure_embedding(self):
        if self.model is None or self.stack is None:
            return
        if self._embedding() is not None:
            self._prefetch()
            return
        key = self._key()
        if key != self.pending_key:
            self.pending_key = key
            self.worker.request(key, self.model, self.rgb)
        self.statusBar().showMessage("Computing image embedding… (you can already place prompts)")

    def on_embedded(self, key, emb):
        if key == self.pending_key:
            self.pending_key = None
        if key[:2] != self._key()[:2]:
            return  # model or image changed meanwhile
        self.embeddings[key] = emb
        while len(self.embeddings) > MAX_CACHED_EMBEDDINGS:
            self.embeddings.popitem(last=False)
        if key == self._key():
            self.statusBar().showMessage("Ready", 2000)
            self._update_prediction()
            self._prefetch()
        elif self._embedding() is None:
            self._ensure_embedding()  # a prefetch finished while the current slice still waits
        else:
            self._prefetch()

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
                    self.worker.request(self.pending_key, self.model, self.stack.frame_rgb(j))
                    return

    # ---- prompts -------------------------------------------------------

    def _has_prompts(self):
        return bool(self.points) or self.box is not None

    def _snapshot(self):
        self.prompt_undo.append((list(self.points), self.box, self.choice))

    def add_point(self, x: float, y: float, positive: bool):
        if self.stack is None:
            return
        self._snapshot()
        self.points.append((x, y, positive))
        self._update_prediction()

    def set_box(self, x0, y0, x1, y1):
        if self.stack is None:
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
        self._reset_prompts()
        self.statusBar().clearMessage()

    def _reset_prompts(self):
        self.points, self.box, self.choice, self.mask = [], None, None, None
        self.prompt_undo.clear()
        self.canvas.set_prompts([], None)
        self.canvas.set_preview(None)

    def _color(self):
        return tuple(int(c) for c in self.lut[self.next_id & 0xFFFF][:3])

    def _update_prediction(self):
        self.canvas.set_prompts(self.points, self.box)
        if not self._has_prompts():
            self.mask = None
            self.canvas.set_preview(None)
            return
        emb = self._embedding()
        if emb is None:
            return
        self.mask, score, n = self.model.predict(emb, self.points, self.box, self.choice)
        self.canvas.set_preview(self.mask, self._color(), 110)
        alt = f"alternative {(self.choice or 0) + 1}/{n}, " if n > 1 else ""
        self.statusBar().showMessage(
            f"Mask: {alt}score {score:.2f} — Enter to commit, Tab for alternatives, Esc to clear"
        )

    def on_hover(self, x: float, y: float):
        self._hover_pos = (x, y)
        if not self._hover_timer.isActive():
            self._hover_timer.start()

    def _do_hover(self):
        if self._has_prompts() or not self.a_hover.isChecked():
            return
        emb = self._embedding()
        if emb is None:
            return
        x, y = self._hover_pos
        if x < 0 or self.labels[self.idx][int(y), int(x)]:
            self.canvas.set_preview(None)
            return
        mask, _, _ = self.model.predict(emb, [(x, y, True)])
        self.canvas.set_preview(mask, self._color(), 45)

    # ---- objects -------------------------------------------------------

    def _refresh_labels(self):
        self.canvas.set_labels(self.lut[self.labels[self.idx] & 0xFFFF])
        self._update_title()

    def _edit_labels(self, region: np.ndarray, value: int):
        """Write value into labels[idx][region], recording a patch for undo."""
        ys, xs = np.flatnonzero(region.any(1)), np.flatnonzero(region.any(0))
        sl = (slice(ys[0], ys[-1] + 1), slice(xs[0], xs[-1] + 1))
        lab = self.labels[self.idx]
        self.label_undo.append((self.idx, sl, lab[sl].copy(), self.next_id))
        lab[sl][region[sl]] = value
        self.dirty = True

    def commit(self):
        if self.mask is None:
            return
        region = self.mask & (self.labels[self.idx] == 0)  # never overwrite existing objects
        if not region.any():
            self.statusBar().showMessage("Nothing to add: the mask is empty or fully covered", 4000)
            return
        if self.next_id > np.iinfo(self.labels.dtype).max:
            self.labels = self.labels.astype(np.uint32)
        self._edit_labels(region, self.next_id)
        self.statusBar().showMessage(f"Added object {self.next_id}", 2000)
        self.next_id += 1
        self._reset_prompts()
        self._refresh_labels()

    def delete_object_at(self, x: float, y: float):
        if self.labels is None:
            return
        obj = int(self.labels[self.idx][int(y), int(x)])
        if not obj:
            return
        self._edit_labels(self.labels[self.idx] == obj, 0)
        self.statusBar().showMessage(f"Deleted object {obj} (Ctrl+Z to restore)", 3000)
        self._refresh_labels()

    def undo(self):
        if self.prompt_undo:
            self.points, self.box, self.choice = self.prompt_undo.pop()
            self._update_prediction()
            if not self._has_prompts():
                self.statusBar().clearMessage()
        elif self.label_undo:
            idx, sl, patch, next_id = self.label_undo.pop()
            self.labels[idx][sl] = patch
            self.next_id = next_id
            self.dirty = True
            if idx != self.idx and len(self.stack) > 1:
                self.slice_slider.setValue(idx)
            else:
                self._refresh_labels()

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
    ap.add_argument("-m", "--model", help="checkpoint path, or name of a cached model (e.g. vit_b_lm)")
    args = ap.parse_args()

    app = QApplication(sys.argv[:1])
    app.setApplicationName("SlugSam")
    win = MainWindow()
    win.show()
    if args.image:
        win.open_image(args.image)
    QTimer.singleShot(0, lambda: win.load_default_model(args.model))
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
