"""Image loading and label I/O."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import tifffile

IMAGE_FILTER = "Images (*.tif *.tiff *.png *.jpg *.jpeg *.bmp);;All files (*)"


class ImageStack:
    """An image as N frames of (H, W) or (H, W, C); a plain 2D image has N == 1."""

    def __init__(self, path: Path, data: np.ndarray, note: str = ""):
        self.path = path
        self.data = data
        self.note = note

    def __len__(self):
        return self.data.shape[0]

    @property
    def frame_shape(self) -> tuple[int, int]:
        return self.data.shape[1:3]

    def frame_rgb(self, i: int) -> np.ndarray:
        """Frame i as contiguous (H, W, 3) uint8, contrast-stretched unless already uint8."""
        f = self.data[i]
        if f.ndim == 2:
            f = f[..., None]
        if f.dtype == bool:
            f = f.astype(np.uint8) * 255
        elif f.dtype != np.uint8:
            f = np.stack([_stretch(f[..., c]) for c in range(f.shape[-1])], axis=-1)
        c = f.shape[-1]
        if c == 1:
            f = np.repeat(f, 3, axis=-1)
        elif c == 2:
            f = np.concatenate([f, np.zeros_like(f[..., :1])], axis=-1)
        elif c > 3:
            f = f[..., :3]
        return np.ascontiguousarray(f)

    @property
    def labels_path(self) -> Path:
        return self.path.with_name(self.path.stem + "_labels.tif")


def _stretch(ch: np.ndarray, lo_pct=0.5, hi_pct=99.9) -> np.ndarray:
    step = max(1, min(ch.shape) // 512)
    sample = ch[::step, ::step].astype(np.float32)
    lo, hi = np.nanpercentile(sample, (lo_pct, hi_pct))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros(ch.shape, np.uint8)
    out = (ch.astype(np.float32) - lo) * (255.0 / (hi - lo))
    return np.clip(np.nan_to_num(out), 0, 255).astype(np.uint8)


def load_image(path: str | Path) -> ImageStack:
    path = Path(path)
    if path.suffix.lower() in (".tif", ".tiff"):
        with tifffile.TiffFile(path) as tf:
            series = tf.series[0]
            arr, axes = series.asarray(), series.axes
    else:
        from PIL import Image

        arr = np.asarray(Image.open(path))
        axes = "YX" if arr.ndim == 2 else "YXS"
    return _to_stack(path, arr, axes)


def _to_stack(path: Path, arr: np.ndarray, axes: str) -> ImageStack:
    if len(axes) != arr.ndim or "Y" not in axes or "X" not in axes:
        # unknown layout: assume (..., Y, X), or (..., Y, X, S) for a short last axis
        rgb = arr.ndim >= 3 and arr.shape[-1] in (3, 4) and arr.shape[-3] > 4
        axes = "Q" * (arr.ndim - (3 if rgb else 2)) + ("YXS" if rgb else "YX")
    ch = next((a for a in "SC" if a in axes), None)
    order = [i for i, a in enumerate(axes) if a not in ("Y", "X", ch)]
    order += [axes.index("Y"), axes.index("X")]
    if ch:
        order.append(axes.index(ch))
    arr = np.transpose(arr, order)
    tail = arr.shape[-3:] if ch else arr.shape[-2:]
    arr = arr.reshape((-1, *tail))
    note = ""
    if ch and arr.shape[-1] > 3:
        note = f"{arr.shape[-1]} channels: showing the first 3"
    return ImageStack(path, arr, note)


def load_labels(path: Path, n: int, shape: tuple[int, int]) -> np.ndarray | None:
    """Existing label file, if it matches the image; else None."""
    if not path.is_file():
        return None
    try:
        lab = tifffile.imread(path)
    except Exception:
        return None
    if lab.size != n * shape[0] * shape[1] or lab.dtype.kind not in "ui":
        return None
    lab = lab.reshape((n, *shape))
    return lab.astype(np.uint16 if lab.max(initial=0) < 65536 else np.uint32)


def save_labels(path: Path, labels: np.ndarray):
    out = labels[0] if labels.shape[0] == 1 else labels
    tifffile.imwrite(path, out, compression="zlib")
