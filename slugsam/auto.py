"""Automatic segmentation by prompting a model with a regular grid of points."""

from __future__ import annotations

from typing import Iterator

import numpy as np
import torch
from torchvision.ops import masks_to_boxes, nms

from .engine import Embedding, SamModel

STABILITY = 0.9  # a mask must keep this share of its area when the cut-off is moved by +-1
BOX_NMS_IOU = 0.7  # masks whose boxes overlap more than this are duplicates


@torch.inference_mode()
def _grid_batch(model, emb, xy, min_score, min_area, max_area, keep_edges):
    """Best acceptable candidate mask for each grid point: (low-res logits, scores, boxes)."""
    low, iou = model._decode_points(emb, xy)  # (B, 3, h, w), (B, 3)
    hv, wv = model._valid_low_res(emb)
    v = low[:, :, :hv, :wv]
    m = v > 0
    px = emb.original_size[0] * emb.original_size[1] / (hv * wv)  # image pixels per low-res pixel
    area = m.sum((2, 3)) * px
    stable = (v > 1).sum((2, 3)) / (v > -1).sum((2, 3)).clamp(min=1)
    ok = (iou >= min_score) & (stable >= STABILITY) & (area >= max(min_area, px)) & (area <= max_area)
    # A mask cut by the border of a tile is incomplete; a neighbouring tile holds it whole.
    top, left, bottom, right = keep_edges
    if not top:
        ok &= ~m[:, :, 0, :].any(-1)
    if not left:
        ok &= ~m[:, :, :, 0].any(-1)
    if not bottom:
        ok &= ~m[:, :, -1, :].any(-1)
    if not right:
        ok &= ~m[:, :, :, -1].any(-1)
    score, best = torch.where(ok, iou, -1.0).max(1)
    sel = score > 0
    rows = torch.arange(len(best), device=best.device)[sel]
    best = best[sel]
    return low[rows, best].half(), score[sel], masks_to_boxes(m[rows, best])


@torch.inference_mode()
def _upsample(model, emb, logits):
    """Full-resolution masks, each cropped to its bounding box: list of (mask, y0, x0) or None."""
    masks = model._upsample(logits.float()[:, None], emb)[:, 0] > 0
    out = []
    for mask in masks:
        if not mask.any():
            out.append(None)
            continue
        x0, y0, x1, y1 = (int(v) for v in masks_to_boxes(mask[None])[0])
        out.append((mask[y0 : y1 + 1, x0 : x1 + 1].cpu().numpy(), y0, x0))
    return out


def grid_masks(
    model: SamModel,
    emb: Embedding,
    points_per_side: int = 32,
    min_score: float = 0.8,
    min_area: float = 0,
    max_area: float = float("inf"),
    keep_edges: tuple[bool, bool, bool, bool] = (True, True, True, True),
    region: tuple[float, float, float, float] | None = None,
    batch: int = 32,
) -> Iterator[float | list]:
    """Segment everything in an embedded image (or tile).

    Each grid point is prompted as a single click, and the best candidate that passes the
    quality and size limits is kept. keep_edges = (top, left, bottom, right) says on which
    sides a mask may touch the border. region (x0, y0, x1, y1) limits the grid points, not
    the masks, to part of the image.
    A generator, so the caller stays responsive: yields progress in 0..1, then finally a
    list of (mask, y0, x0, score) with each bool mask cropped to its bounding box at
    (y0, x0), best score first.
    """
    h, w = emb.original_size
    n = max(h, w)
    ys = (np.arange(max(1, round(points_per_side * h / n))) + 0.5) * h / max(1, round(points_per_side * h / n))
    xs = (np.arange(max(1, round(points_per_side * w / n))) + 0.5) * w / max(1, round(points_per_side * w / n))
    if region is not None:
        xs = xs[(xs >= region[0]) & (xs < region[2])]
        ys = ys[(ys >= region[1]) & (ys < region[3])]
    grid = np.stack(np.meshgrid(xs, ys), -1).reshape(-1, 2).astype(np.float32)
    if not len(grid):
        yield []
        return

    logits, scores, boxes = [], [], []
    for i in range(0, len(grid), batch):
        lo, sc, bx = _grid_batch(model, emb, grid[i : i + batch], min_score, min_area, max_area, keep_edges)
        logits.append(lo)
        scores.append(sc)
        boxes.append(bx)
        yield 0.8 * min(1.0, (i + batch) / len(grid))
    logits, scores, boxes = torch.cat(logits), torch.cat(scores), torch.cat(boxes)
    keep = nms(boxes.float(), scores.float(), BOX_NMS_IOU)  # many points hit the same object

    out = []
    step = int(np.clip(64e6 // (h * w), 1, 16))  # full-resolution masks are large: few at a time
    for i in range(0, len(keep), step):
        k = keep[i : i + step]
        for found, score in zip(_upsample(model, emb, logits[k]), scores[k].tolist()):
            if found is not None:
                out.append((*found, score))
        yield 0.8 + 0.2 * min(1.0, (i + step) / len(keep))
    yield out
