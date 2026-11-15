"""Image loading and label I/O."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import tifffile

IMAGE_FILTER = "Images (*.tif *.tiff *.png *.jpg *.jpeg *.bmp);;All files (*)"


IMAGE_SUFFIXES = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}


def next_image(path: Path) -> Path | None:
    """The image after path in its folder, in natural name order, skipping label files."""
    def key(p: Path):
        return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", p.name)]

    files = sorted(
        (
            f
            for f in path.parent.iterdir()
            if f.suffix.lower() in IMAGE_SUFFIXES and not f.stem.endswith("_labels") and f.is_file()
        ),
        key=key,
    )
    later = [f for f in files if key(f) > key(path)]
    return later[0] if later else None


class ImageStack:
    """An image as N frames of (H, W) pixels, each with one or more channels.

    The file keeps its own axis order. `read` fetches one (H, W) plane, or (H, W, S) for
    colour-sample images, given an index over the leading (non-Y/X/S) axes, so the pixels
    can live in memory, in a memory-mapped file, or be decoded from the file on demand.
    """

    def __init__(self, path, read, lead_shape, ch_axis, plane_shape, mode="in memory", close=None):
        self.path = Path(path)
        self._read = read
        self._lead_shape = tuple(lead_shape)
        self._ch_axis = ch_axis  # position of the channel axis among the leading axes, or None
        self.frame_shape = tuple(plane_shape[:2])
        self.is_rgb = len(plane_shape) == 3  # colour samples rather than separate channels
        self.mode = mode
        self._close = close
        self._frames_shape = tuple(n for k, n in enumerate(self._lead_shape) if k != ch_axis)
        if self.is_rgb:
            self.channels = plane_shape[2]
        else:
            self.channels = self._lead_shape[ch_axis] if ch_axis is not None else 1

    def __len__(self):
        return int(np.prod(self._frames_shape, dtype=np.int64))

    def close(self):
        if self._close:
            self._close()
        self._read = self._close = None

    def frame(self, i: int, channel: int | None = None) -> np.ndarray:
        """Frame i as (H, W) for one channel, or (H, W, C) with up to three channels for None."""
        idx = list(np.unravel_index(i, self._frames_shape)) if self._frames_shape else []
        if self._ch_axis is None:
            f = np.asarray(self._read(tuple(idx)))
            return f[..., channel] if self.is_rgb and channel is not None else f
        idx.insert(self._ch_axis, 0)
        chans = range(min(self.channels, 3)) if channel is None else [channel]
        planes = []
        for c in chans:
            idx[self._ch_axis] = c
            planes.append(np.asarray(self._read(tuple(idx))))
        return planes[0] if channel is not None else np.stack(planes, axis=-1)

    def frame_rgb(self, i: int, channel: int | None = None) -> np.ndarray:
        """Frame i as contiguous (H, W, 3) uint8, contrast-stretched unless already uint8.

        channel selects a single channel, shown in grey; None composites the first three.
        """
        f = self.frame(i, channel)
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


def load_image(path: str | Path, lazy: bool = False) -> ImageStack:
    """Open an image. With lazy=True a TIFF's pixels stay on disk and are read slice by slice."""
    path = Path(path)
    if path.suffix.lower() not in (".tif", ".tiff"):
        from PIL import Image

        arr = np.asarray(Image.open(path))
        return _array_stack(path, arr, "YX" if arr.ndim == 2 else "YXS")
    if not lazy:
        with tifffile.TiffFile(path) as tf:
            series = tf.series[0]
            return _array_stack(path, series.asarray(), series.axes)

    tf = tifffile.TiffFile(path)
    try:
        series = tf.series[0]
        axes, shape = series.axes, series.shape
        try:  # uncompressed, contiguous files can be mapped straight into memory
            arr = tifffile.memmap(path, series=0, mode="r")
        except Exception:
            arr = None
        if arr is not None:
            tf.close()
            return _array_stack(path, arr, axes, mode="lazy, memory-mapped")
        # otherwise decode single pages on demand, if the file stores one page per plane
        if _plane_last(axes, len(shape)):
            n_plane = 3 if axes.endswith("S") else 2
            lead = shape[:-n_plane]
            n_pages = int(np.prod(lead, dtype=np.int64))
            pages = series.pages
            if len(pages) == n_pages and tuple(pages[0].shape) == tuple(shape[-n_plane:]):

                def read(idx):
                    k = int(np.ravel_multi_index(idx, lead)) if lead else 0
                    return pages[k].asarray()

                ch = axes.index("C") if "C" in axes[:-n_plane] and n_plane == 2 else None
                return ImageStack(
                    path, read, lead, ch, shape[-n_plane:], "lazy, decoded per slice", tf.close
                )
        stack = _array_stack(path, series.asarray(), axes, mode="in memory (file layout prevents lazy loading)")
        tf.close()
        return stack
    except Exception:
        tf.close()
        raise


def _plane_last(axes: str, ndim: int) -> bool:
    return len(axes) == ndim and (axes.endswith("YX") or axes.endswith("YXS"))


def _array_stack(path: Path, arr: np.ndarray, axes: str, mode: str = "in memory") -> ImageStack:
    """Wrap an array (in memory or memory-mapped) without copying it."""
    if len(axes) != arr.ndim or "Y" not in axes or "X" not in axes:
        # unknown layout: assume (..., Y, X), or (..., Y, X, S) for a short last axis
        rgb = arr.ndim >= 3 and arr.shape[-1] in (3, 4) and arr.shape[-3] > 4
        axes = "Q" * (arr.ndim - (3 if rgb else 2)) + ("YXS" if rgb else "YX")
    if not _plane_last(axes, arr.ndim):  # move the image plane last; a view, not a copy
        tail = [a for a in "YXS" if a in axes]
        order = [a for a in axes if a not in tail] + tail
        arr = np.transpose(arr, [axes.index(a) for a in order])
        axes = "".join(order)
    n_plane = 3 if axes.endswith("S") else 2
    lead_axes = axes[:-n_plane]
    ch = lead_axes.index("C") if "C" in lead_axes and n_plane == 2 else None
    return ImageStack(
        path, lambda idx: arr[idx], arr.shape[:-n_plane], ch, arr.shape[-n_plane:], mode
    )


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
