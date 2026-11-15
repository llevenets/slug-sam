"""Thin wrapper around segment_anything: load a checkpoint, embed images, decode prompts."""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PySide6.QtCore import QThread, Signal
from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide

_EMBED_DIM_TO_TYPE = {768: "vit_b", 1024: "vit_l", 1280: "vit_h"}


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def find_models() -> list[Path]:
    """Checkpoints in the micro_sam cache, ./models and $SLUGSAM_MODELS."""
    dirs = [Path.home() / ".cache" / "micro_sam" / "models", Path.cwd() / "models"]
    if os.environ.get("SLUGSAM_MODELS"):
        dirs.append(Path(os.environ["SLUGSAM_MODELS"]))
    found = []
    for d in dirs:
        if not d.is_dir():
            continue
        for f in sorted(d.iterdir()):
            if not f.is_file() or f.suffix not in ("", ".pth", ".pt"):
                continue
            if f.name.startswith("tmp") or f.stem.endswith("_decoder"):
                continue
            if f.stat().st_size < 30e6:
                continue
            found.append(f)
    return found


@dataclass
class Embedding:
    features: torch.Tensor  # (1, 256, 64, 64) on the model device
    original_size: tuple[int, int]  # (H, W)
    input_size: tuple[int, int]  # (H, W) after resizing the long side to 1024


class SamModel:
    def __init__(self, path: str | Path, device: str | None = None):
        self.path = Path(path)
        self.device = device or pick_device()
        state = torch.load(self.path, map_location="cpu", weights_only=True)
        # micro_sam fine-tuning checkpoints wrap the weights
        if "model_state" in state:
            state = {k.removeprefix("sam."): v for k, v in state["model_state"].items()}
        elif "model" in state and isinstance(state["model"], dict):
            state = state["model"]
        try:
            dim = state["image_encoder.pos_embed"].shape[-1]
            self.model_type = _EMBED_DIM_TO_TYPE[dim]
        except KeyError:
            raise ValueError(
                f"{self.path.name} is not a ViT-B/L/H Segment Anything checkpoint"
            ) from None
        sam = sam_model_registry[self.model_type](checkpoint=None)
        sam.load_state_dict(state)
        self.sam = sam.to(self.device).eval()
        self.transform = ResizeLongestSide(sam.image_encoder.img_size)

    @torch.inference_mode()
    def embed(self, rgb: np.ndarray) -> Embedding:
        """rgb: (H, W, 3) uint8."""
        resized = self.transform.apply_image(rgb)
        x = torch.as_tensor(resized, device=self.device).permute(2, 0, 1)[None].float()
        input_size = tuple(x.shape[-2:])
        features = self.sam.image_encoder(self.sam.preprocess(x))
        return Embedding(features, rgb.shape[:2], input_size)

    @torch.inference_mode()
    def predict(self, emb: Embedding, points=(), box=None, choice: int | None = None):
        """Decode a mask from prompts.

        points: sequence of (x, y, is_positive); box: (x0, y0, x1, y1) or None.
        choice: None picks automatically; an int forces multi-mask output and
        selects the choice-th candidate, ranked by predicted quality.
        Returns (mask (H, W) bool, score, number of candidates).
        """
        sam, dev = self.sam, self.device
        pts = None
        if points:
            xy = np.array([[p[0], p[1]] for p in points], dtype=np.float32)
            xy = self.transform.apply_coords(xy, emb.original_size)
            lab = [1 if p[2] else 0 for p in points]
            pts = (
                torch.as_tensor(xy, dtype=torch.float, device=dev)[None],
                torch.as_tensor(lab, dtype=torch.int, device=dev)[None],
            )
        bx = None
        if box is not None:
            b = self.transform.apply_boxes(np.array([box], dtype=np.float32), emb.original_size)
            bx = torch.as_tensor(b, dtype=torch.float, device=dev)
        # a single click is ambiguous (part / object / group): let SAM propose three
        multimask = choice is not None or (box is None and len(points) == 1)
        sparse, dense = sam.prompt_encoder(points=pts, boxes=bx, masks=None)
        low_res, iou = sam.mask_decoder(
            image_embeddings=emb.features,
            image_pe=sam.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense,
            multimask_output=multimask,
        )
        iou = iou[0]
        order = torch.argsort(iou, descending=True)
        k = int(order[(choice or 0) % len(order)])
        mask = sam.postprocess_masks(low_res[:, k : k + 1], emb.input_size, emb.original_size)
        return (mask[0, 0] > 0).cpu().numpy(), float(iou[k]), len(order)


class EmbedWorker(QThread):
    """Computes embeddings off the GUI thread. Only the newest request is kept."""

    done = Signal(object, object)  # key, Embedding
    failed = Signal(object, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._cv = threading.Condition()
        self._job = None
        self._quit = False

    def request(self, key, model: SamModel, rgb: np.ndarray):
        with self._cv:
            self._job = (key, model, rgb)
            self._cv.notify()

    def stop(self):
        with self._cv:
            self._quit = True
            self._cv.notify()
        self.wait()

    def run(self):
        while True:
            with self._cv:
                while self._job is None and not self._quit:
                    self._cv.wait()
                if self._quit:
                    return
                key, model, rgb = self._job
                self._job = None
            try:
                emb = model.embed(rgb)
            except Exception as e:  # e.g. CUDA out of memory
                self.failed.emit(key, f"{type(e).__name__}: {e}")
            else:
                self.done.emit(key, emb)
