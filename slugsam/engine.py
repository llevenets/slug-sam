"""Model backends (SAM, SAM 2): load a checkpoint, embed images, decode prompts."""

from __future__ import annotations

import math
import os
import threading
import warnings
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator

import numpy as np
import platformdirs
import psutil
import torch
import torch.nn.functional as F
from PySide6.QtCore import QThread, Signal

_EMBED_DIM_TO_TYPE = {768: "vit_b", 1024: "vit_l", 1280: "vit_h"}


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _micro_sam_cache_dirs() -> list[Path]:
    """Where micro_sam keeps its downloads: $MICROSAM_CACHEDIR, else the per-OS user cache."""
    dirs = [Path(platformdirs.user_cache_dir("micro_sam"))]  # same rule micro_sam uses (via pooch)
    if os.environ.get("MICROSAM_CACHEDIR"):
        dirs.insert(0, Path(os.environ["MICROSAM_CACHEDIR"]).expanduser())
    return dirs


def find_models() -> list[Path]:
    """Checkpoints in the micro_sam cache, the models folders and $SLUGSAM_MODELS."""
    dirs = [
        *(d / "models" for d in _micro_sam_cache_dirs()),
        Path(__file__).resolve().parent.parent / "models",  # next to the source checkout
        Path.cwd() / "models",
    ]
    if os.environ.get("SLUGSAM_MODELS"):
        dirs.append(Path(os.environ["SLUGSAM_MODELS"]))
    found = []
    for d in dict.fromkeys(p.resolve() for p in dirs):
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


def free_gpu_cache():
    """Return memory PyTorch holds but no longer uses (e.g. encoder scratch space) to the GPU."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()


def memory_usage() -> dict:
    """Current memory figures in bytes; keys are absent where a figure is unavailable."""
    out = {}
    try:
        out["ram"] = psutil.Process().memory_info().rss
    except psutil.Error:
        pass
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        out["gpu_used"] = total - free  # whole device, other programs included
        out["gpu_total"] = total
        out["gpu_ours"] = torch.cuda.memory_reserved()
        out["gpu_live"] = torch.cuda.memory_allocated()
    elif torch.backends.mps.is_available():  # Apple GPUs share main memory: no device total
        out["gpu_ours"] = torch.mps.driver_allocated_memory()
        out["gpu_live"] = torch.mps.current_allocated_memory()
    return out


@dataclass
class Embedding:
    features: Any  # backend-specific tensors on the model device
    original_size: tuple[int, int]  # (H, W)
    input_size: tuple[int, int]  # (H, W) of the image as fed to the encoder
    nbytes: int = 0  # device memory held by features


def _nbytes(*tensors: torch.Tensor) -> int:
    return sum(t.numel() * t.element_size() for t in tensors)


def _prompt_arrays(points, box):
    """Prompts as float arrays in image pixels: (xy (N, 2), labels (N,)) or None, box (4,) or None."""
    pts = None
    if points:
        xy = np.array([[p[0], p[1]] for p in points], dtype=np.float32)
        pts = (xy, np.array([1 if p[2] else 0 for p in points], dtype=np.int32))
    return pts, None if box is None else np.array(box, dtype=np.float32)


class SamModel:
    """A loaded promptable segmentation model. Use load_model() to get the right backend.

    Backends provide embed(), _decode() and _upsample(); everything above is shared.
    """

    path: Path
    device: str
    model_type: str  # short human-readable description
    supports_tracking = False  # whether track() is available

    def embed(self, rgb: np.ndarray) -> Embedding:
        """rgb: (H, W, 3) uint8."""
        raise NotImplementedError

    def _decode(self, emb: Embedding, points, box, multimask: bool):
        """Low-res mask logits (1, k, h, w) and predicted quality (k,) for the prompts."""
        raise NotImplementedError

    def _upsample(self, low_res: torch.Tensor, emb: Embedding) -> torch.Tensor:
        """Mask logits (n, k, h, w) resampled to the original image size."""
        raise NotImplementedError

    def _decode_points(self, emb: Embedding, xy: np.ndarray):
        """One independent click per row of xy (B, 2): logits (B, 3, h, w) and quality (B, 3)."""
        raise NotImplementedError

    def _valid_low_res(self, emb: Embedding) -> tuple[int, int]:
        """Height and width of the part of the low-res logits that shows the image."""
        raise NotImplementedError

    @torch.inference_mode()
    def predict(self, emb: Embedding, points=(), box=None, choice: int | None = None):
        """Decode a mask from prompts.

        points: sequence of (x, y, is_positive); box: (x0, y0, x1, y1) or None.
        choice: None picks automatically; an int forces multi-mask output and
        selects the choice-th candidate, ranked by predicted quality.
        Returns (mask (H, W) bool, score, number of candidates).
        """
        # a single click is ambiguous (part / object / group): let SAM propose three
        multimask = choice is not None or (box is None and len(points) == 1)
        low_res, iou = self._decode(emb, points, box, multimask)
        order = torch.argsort(iou, descending=True)
        k = int(order[(choice or 0) % len(order)])
        mask = self._upsample(low_res[:, k : k + 1], emb)
        return (mask[0, 0] > 0).cpu().numpy(), float(iou[k]), len(order)

    @torch.inference_mode()
    def predict_candidates(self, emb: Embedding, points=(), box=None) -> np.ndarray:
        """All three candidate masks for the prompts, as (3, H, W) bool."""
        low_res, _ = self._decode(emb, points, box, True)
        return (self._upsample(low_res, emb)[0] > 0).cpu().numpy()


class Sam1Model(SamModel):
    """Original Segment Anything (ViT-B/L/H), including micro_sam fine-tuned weights."""

    def __init__(self, path: Path, state: dict, device: str):
        from segment_anything import sam_model_registry
        from segment_anything.utils.transforms import ResizeLongestSide

        self.path, self.device = path, device
        try:
            vit = _EMBED_DIM_TO_TYPE[state["image_encoder.pos_embed"].shape[-1]]
        except KeyError:
            raise ValueError(f"{path.name}: unsupported SAM image encoder") from None
        self.model_type = f"SAM {vit}"
        sam = sam_model_registry[vit](checkpoint=None)
        sam.load_state_dict(state)
        self.sam = sam.to(device).eval()
        self.transform = ResizeLongestSide(sam.image_encoder.img_size)

    @torch.inference_mode()
    def embed(self, rgb: np.ndarray) -> Embedding:
        resized = self.transform.apply_image(rgb)
        x = torch.as_tensor(resized, device=self.device).permute(2, 0, 1)[None].float()
        input_size = tuple(x.shape[-2:])
        features = self.sam.image_encoder(self.sam.preprocess(x))
        return Embedding(features, rgb.shape[:2], input_size, _nbytes(features))

    def _decode(self, emb, points, box, multimask):
        sam, dev = self.sam, self.device
        pts, bx = _prompt_arrays(points, box)
        if pts is not None:
            xy = self.transform.apply_coords(pts[0], emb.original_size)
            pts = (
                torch.as_tensor(xy, dtype=torch.float, device=dev)[None],
                torch.as_tensor(pts[1], dtype=torch.int, device=dev)[None],
            )
        if bx is not None:
            bx = self.transform.apply_boxes(bx[None], emb.original_size)
            bx = torch.as_tensor(bx, dtype=torch.float, device=dev)
        sparse, dense = sam.prompt_encoder(points=pts, boxes=bx, masks=None)
        low_res, iou = sam.mask_decoder(
            image_embeddings=emb.features,
            image_pe=sam.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense,
            multimask_output=multimask,
        )
        return low_res, iou[0]

    def _upsample(self, low_res, emb):
        return self.sam.postprocess_masks(low_res, emb.input_size, emb.original_size)

    def _decode_points(self, emb, xy):
        sam, dev = self.sam, self.device
        xy = self.transform.apply_coords(xy, emb.original_size)
        pts = (
            torch.as_tensor(xy, dtype=torch.float, device=dev)[:, None],
            torch.ones((len(xy), 1), dtype=torch.int, device=dev),
        )
        sparse, dense = sam.prompt_encoder(points=pts, boxes=None, masks=None)
        return sam.mask_decoder(
            image_embeddings=emb.features,
            image_pe=sam.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense,
            multimask_output=True,
        )

    def _valid_low_res(self, emb):  # the image is padded to a square at the bottom / right
        scale = 256 / self.sam.image_encoder.img_size
        return math.ceil(emb.input_size[0] * scale), math.ceil(emb.input_size[1] * scale)


# (patch embedding width, number of trunk blocks) -> config name suffix
_SAM2_SIZES = {(96, 12): "t", (96, 16): "s", (112, 24): "b+", (144, 48): "l"}
_SAM2_NAMES = {"t": "tiny", "s": "small", "b+": "base+", "l": "large"}


class Sam2Model(SamModel):
    """SAM 2 / SAM 2.1 (Hiera).

    Used per image like SAM (embed once, then decode prompts), and through track() as a
    video model that follows objects across frames with its memory.
    """

    supports_tracking = True

    def __init__(self, path: Path, state: dict, device: str):
        try:
            import sam2.sam2_video_predictor
            from sam2.build_sam import build_sam2
            from sam2.utils.transforms import SAM2Transforms
        except ImportError:
            raise ValueError(
                f"{path.name} is a SAM 2 checkpoint, but the 'sam2' package is not installed"
            ) from None

        # the tracker wraps its frame loop in a terminal progress bar; the app shows its own
        sam2.sam2_video_predictor.tqdm = lambda iterable, **_: iterable
        self.path, self.device = path, device
        width = state["image_encoder.trunk.patch_embed.proj.weight"].shape[0]
        blocks = len({k.split(".")[3] for k in state if k.startswith("image_encoder.trunk.blocks.")})
        try:
            size = _SAM2_SIZES[(width, blocks)]
        except KeyError:
            raise ValueError(f"{path.name}: unrecognised SAM 2 model size") from None
        version = "sam2.1" if "no_obj_embed_spatial" in state else "sam2"
        self.model_type = f"SAM {version[3:]} {_SAM2_NAMES[size]}"
        # Build the video predictor: it is the same network plus the tracking logic.
        model = build_sam2(
            f"configs/{version}/{version}_hiera_{size}.yaml",
            None,
            device=device,
            hydra_overrides_extra=[
                "++model._target_=sam2.sam2_video_predictor.SAM2VideoPredictor",
                # remember seed masks exactly as given, not a softened version
                "++model.binarize_mask_from_pts_for_mem_enc=true",
            ],
        )
        model.load_state_dict(state)
        self.sam = model
        with warnings.catch_warnings():  # sam2 scripts its transforms with a deprecated API
            warnings.simplefilter("ignore", FutureWarning)
            self.transform = SAM2Transforms(resolution=model.image_size, mask_threshold=0.0)
        n = model.image_size // 4  # the three feature maps have strides 4, 8 and 16
        self._feat_sizes = [(n, n), (n // 2, n // 2), (n // 4, n // 4)]

    @torch.inference_mode()
    def embed(self, rgb: np.ndarray) -> Embedding:
        m = self.sam
        x = self.transform(rgb)[None].to(self.device)  # squashed to a square, unlike SAM 1
        _, feats, _, _ = m._prepare_backbone_features(m.forward_image(x))
        if m.directly_add_no_mem_embed:
            feats[-1] = feats[-1] + m.no_mem_embed
        feats = [
            f.permute(1, 2, 0).reshape(1, -1, *size) for f, size in zip(feats, self._feat_sizes)
        ]
        # the decoder also uses the two finer maps; halve their footprint, they dominate it
        features = {"image_embed": feats[-1], "high_res": [f.half() for f in feats[:-1]]}
        size = (m.image_size, m.image_size)
        return Embedding(features, rgb.shape[:2], size, _nbytes(feats[-1], *features["high_res"]))

    def _decode(self, emb, points, box, multimask):
        m, dev = self.sam, self.device
        h, w = emb.original_size
        scale = np.array([m.image_size / w, m.image_size / h], dtype=np.float32)
        pts, bx = _prompt_arrays(points, box)
        coords, labels = [], []
        if bx is not None:  # a box is passed as two specially labelled corner points
            coords.append(bx.reshape(2, 2) * scale)
            labels.append(np.array([2, 3], dtype=np.int32))
        if pts is not None:
            coords.append(pts[0] * scale)
            labels.append(pts[1])
        concat = (
            torch.as_tensor(np.concatenate(coords), dtype=torch.float, device=dev)[None],
            torch.as_tensor(np.concatenate(labels), dtype=torch.int, device=dev)[None],
        )
        sparse, dense = m.sam_prompt_encoder(points=concat, boxes=None, masks=None)
        low_res, iou, _, _ = m.sam_mask_decoder(
            image_embeddings=emb.features["image_embed"],
            image_pe=m.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense,
            multimask_output=multimask,
            repeat_image=False,
            high_res_features=[f.float() for f in emb.features["high_res"]],
        )
        return low_res, iou[0]

    def track(
        self,
        frame_rgb: Callable[[int], np.ndarray],
        n_frames: int,
        start: int,
        masks: dict[int, np.ndarray],
        reverse: bool = False,
    ) -> Iterator[tuple[int, dict[int, np.ndarray]]]:
        """Follow objects through a stack with SAM 2's video memory.

        frame_rgb(i) returns frame i as (H, W, 3) uint8 and is only called for frames that
        are visited. masks maps object id to its (H, W) bool mask on frame `start`.
        Yields (frame index, {object id: mask}) for `start` and then each following
        (or, with reverse, preceding) frame. Close the generator to stop early.
        """
        model, transform = self.sam, self.transform

        class Frames:  # what the predictor indexes instead of a decoded video
            def __len__(self):
                return n_frames

            def __getitem__(self, i):
                return transform(frame_rgb(i))

        height, width = next(iter(masks.values())).shape
        device = torch.device(self.device)
        state = {  # mirrors SAM2VideoPredictor.init_state(), minus loading a video file
            "images": Frames(),
            "num_frames": n_frames,
            "offload_video_to_cpu": True,
            "offload_state_to_cpu": False,
            "video_height": height,
            "video_width": width,
            "device": device,
            "storage_device": device,
            "point_inputs_per_obj": {},
            "mask_inputs_per_obj": {},
            "cached_features": {},
            "constants": {},
            "obj_id_to_idx": OrderedDict(),
            "obj_idx_to_id": OrderedDict(),
            "obj_ids": [],
            "output_dict_per_obj": {},
            "temp_output_dict_per_obj": {},
            "frames_tracked_per_obj": {},
        }
        for obj, mask in masks.items():
            model.add_new_mask(state, start, obj, torch.from_numpy(mask))
        frames = model.propagate_in_video(state, start_frame_idx=start, reverse=reverse)
        for frame, ids, logits in frames:
            found = (logits[:, 0] > 0).cpu().numpy()
            yield frame, {obj: found[k] for k, obj in enumerate(ids)}

    def _upsample(self, low_res, emb):
        return F.interpolate(low_res.float(), emb.original_size, mode="bilinear", align_corners=False)

    def _decode_points(self, emb, xy):
        m, dev = self.sam, self.device
        h, w = emb.original_size
        xy = xy * np.array([m.image_size / w, m.image_size / h], dtype=np.float32)
        pts = (
            torch.as_tensor(xy, dtype=torch.float, device=dev)[:, None],
            torch.ones((len(xy), 1), dtype=torch.int, device=dev),
        )
        sparse, dense = m.sam_prompt_encoder(points=pts, boxes=None, masks=None)
        low_res, iou, _, _ = m.sam_mask_decoder(
            image_embeddings=emb.features["image_embed"],
            image_pe=m.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense,
            multimask_output=True,
            repeat_image=True,
            high_res_features=[f.float() for f in emb.features["high_res"]],
        )
        return low_res, iou

    def _valid_low_res(self, emb):  # the whole image is squashed into the square
        n = self.sam.image_size // 4
        return n, n


def load_model(path: str | Path, device: str | None = None) -> SamModel:
    """Load a checkpoint, picking the backend from its contents."""
    path = Path(path)
    device = device or pick_device()
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise ValueError(f"{path.name} is not a model checkpoint")
    if "model_state" in state:  # micro_sam fine-tuning checkpoints wrap the weights
        state = {k.removeprefix("sam."): v for k, v in state["model_state"].items()}
    elif isinstance(state.get("model"), dict):
        state = state["model"]
    if "image_encoder.pos_embed" in state:
        return Sam1Model(path, state, device)
    if "image_encoder.trunk.patch_embed.proj.weight" in state:
        return Sam2Model(path, state, device)
    raise ValueError(f"{path.name} is not a SAM or SAM 2 checkpoint")


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
