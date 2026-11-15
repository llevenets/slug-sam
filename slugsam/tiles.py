"""Tile geometry for embedding parts of an image at higher resolution.

SAM-type models see every input at roughly 1024 px, so on a large image small objects
are lost. At detail level L the image is covered by half-overlapping tiles with a side
of 1/2**L of the image's long side; each tile is embedded on its own.
A tile is (y0, x0, height, width) in image pixels; None stands for the whole image.
"""

from __future__ import annotations

import math

MIN_TILE = 512  # tiles are not made smaller than this: below it there is no detail left to gain

Tile = tuple[int, int, int, int]


def max_level(shape: tuple[int, int]) -> int:
    return max(0, int(math.log2(max(shape) / MIN_TILE)))


def tile_side(shape: tuple[int, int], level: int) -> int:
    return math.ceil(max(shape) / 2**level)


def _origins(length: int, side: int) -> list[int]:
    if side >= length:
        return [0]
    return [*range(0, length - side, max(1, side // 2)), length - side]


def tiles(shape: tuple[int, int], level: int) -> list[Tile]:
    """All tiles of a level, row by row."""
    side = tile_side(shape, level)
    h, w = min(side, shape[0]), min(side, shape[1])
    return [(y, x, h, w) for y in _origins(shape[0], side) for x in _origins(shape[1], side)]


def tile_for(shape: tuple[int, int], level: int, bbox: tuple[float, float, float, float]) -> Tile | None:
    """The tile of a level that contains bbox (x0, y0, x1, y1) most centrally, if any does."""
    if level <= 0:
        return None
    side = tile_side(shape, level)
    out = []
    for length, lo, hi in ((shape[0], bbox[1], bbox[3]), (shape[1], bbox[0], bbox[2])):
        size = min(side, length)
        fits = [o for o in _origins(length, side) if o <= lo and hi <= o + size]
        if not fits:
            return None
        out.append((min(fits, key=lambda o: abs(o + size / 2 - (lo + hi) / 2)), size))
    (y, h), (x, w) = out
    return (y, x, h, w)


def core(shape: tuple[int, int], tile: Tile) -> tuple[float, float, float, float]:
    """The central part (x0, y0, x1, y1) of a tile, which it does not share with its neighbours.

    The cores of a level's tiles cover the image; anything centred in a core and smaller
    than half a tile lies completely inside that tile.
    """
    y, x, h, w = tile
    y0 = 0 if y == 0 else y + h / 4
    x0 = 0 if x == 0 else x + w / 4
    y1 = shape[0] if y + h >= shape[0] else y + 3 * h / 4
    x1 = shape[1] if x + w >= shape[1] else x + 3 * w / 4
    return (x0, y0, x1, y1)
