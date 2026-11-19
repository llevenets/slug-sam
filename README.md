# SlugSAM

Standalone PySide6-based frontend for simple interactive image segmentation with Segment Anything Models

## Installation

Requires `python >=3.11`

**GPU support**:
- For Linux-based systems with CUDA-compatible NVIDIA GPU should work as is
- For macOS (>=14) on Apple Silicon GPU supported via MPS
- For Windows:
    - currently PyPi has only Torch CPU builds for Windows, so for Windows platforms  specifically `pyproject.toml` has separate entry pointing to PyTorch own pakage index. So you should still automatically get GPU CUDA build through `uv sync --locked`
    - if you want to upgrade PyTorch you might need to manually amend name and URL in `pyproject.toml` after running `uv lock --upgrade` (e.g. you want to use CUDA 13.4 instead of current CUDA 13.2)
    - it does not check whether or not you actually have compatible GPU, if you want CPU-only build for Windows consider commenting out `[[tool.uv.index]]` and `[[tool.uv.sources]]` lines in `pyproject.toml`


Fallbacks on CPU if no compatible GPU is detected

### Via `uv`

1. [Install UV](https://docs.astral.sh/uv/getting-started/installation/)
2. `git clone` repository, inside cloned folder run `uv sync --locked`
3. Run GUI with `uv run slugsam`

### Via `pip` 

1. `git clone` repository, create separate environment for it using your preferred way
2. With active project environment run `pip install -r requirements.txt` followed by `pip install -e ..`
3. Run GUI by calling `slugsam` or `python -m slugsam`

On Windows pip-install stays CPU-only unless you additionally manually pull torch GPU libraries with `--index-url https://download.pytorch.org/whl/cu132` during installation (see [PyTorch install guide](https://pytorch.org/get-started/locally/))

### Additional notes

- For minimal Linux systems you might need to additionally install `libxcb-cursor0` for PySide6 to work properly
- `segment-anything` and `sam2` are build from source on the first installation, it can take some time and disk space (`sam2` will re-download PyTorch libraries into temporary directory to build itself) 
- `sam2` will also try to compile some optional CUDA extensions and (likely) fail on Win/macOS systems, the build should still carry on and be mostly functional
- if you want full functionality of `sam2` on Windows I would advise to try to run it under WSL

## Model Checkpoints

`slugsam` works with SAM, SAM2/2.1 and fine-tuned models build on those

You can get model checkpoints from their official repos:
- original [SAM](https://github.com/facebookresearch/segment-anything)
- [SAM2 and SAM2.1](https://github.com/facebookresearch/sam2)
- For [Segment Anything for Microscopy (µSAM1 and µSAM2)](https://github.com/computational-cell-analytics/micro-sam) see [finetuned-models section](https://computational-cell-analytics.github.io/micro-sam/micro_sam.html#finetuned-models) in their official docs

Once you have checkpoint file (normally `.pth` or `.pt` extension) you can either manually provide path to file-containing folder in GUI model-selection dropdown menu or put checkpoints into `models` folder inside project directory, so they will be recognized automatically on start

Checkpoint files should also be automatically picked up from cache directory labeled `micro_sam`, where µSAM package usually puts downloaded models (or at least it did before)

## What's currently in

Supports loading of `png`, `jpg`, `bmp` and `tif` files, including TIFF multichannel time- and Z-stacks. For multichannel images number of channels is unlimited but currently only first three channels are taken for Composite view. For image stacks whatever is currently rendered in GUI window (T-, Z-position, specific channel or composite view) is treated as model input. 

### Shortcuts

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

### Minimal memory handling

- supports lazy loading of TIFF stacks, with small trade-off for processing speed
- for image stacks embedding is done per viewed frame, this allows for reasonably quick switch between different models during analysis
- embeddings are cached, cleared on new image loading, channel or model change (cache size currently capped of ≈700 Mb, handling around 192 embedding slices for SAM1 or 76 for SAM2)
- RAM memory automatically handled back when image is closed or switched to the next one in folder
- GPU memory automatically handled back once embedding worker goes idle
- Memory readout in the status bar (together with current model and processor)

### Object linkage for stacks

Time and Z-axis are flattened together during object propagation

Current options for stack processing:
1. Static simple point propagation (picks highest scoring mask under the same prompt across all time slices) for cases where you have static (or large and slow moving) objects across frames
2. Linkage via simple IoU threshold 
3. Linkage via SAM2 video predictor (only for SAM2 models), can handle complex or fast moving things but scales badly with number of tracked objects

### Auto-segmenting and mask resolution for larger images

SAM internally rescales images to a fixed size before embedding, which might be disadvantageous for certain cases (e.g few small cells in large FoV). For images of raw size above ≈1000 px zooming in will recalculate embedding to this tile specifically, allowing to segment finer details. Simple model-agnostic auto-segmentation is implemented via grid-prompt and tile embedding.
