# MIT License

# Copyright (c) 2026 Jaime Álvarez Díaz
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the “Software”), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
# of the Software, and to permit persons to whom the Software is furnished to do
# so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


import numpy as np
from shapely.geometry import Polygon

from ..logs import vimi_logger


def norm_coords(coords: np.ndarray, img_size: tuple[int, int]) -> np.ndarray:
    """Normalize an array for a given img_size. It will scale even columns with the image width and odd columns with the image height.
    Args:
        coords (np.ndarray): n x [x1, y1, x2, y2, ...] array with the coordinates to normalize.
        img_size (tuple[int, int]): (w, h) tuple with the size of the image.
    Raises:
        ValueError: On input shape not like (n, even).
    Returns:
        np.ndarray: Normalized array.
    """
    if not isinstance(coords, np.ndarray):
        msg: str = f'coords must be a np.ndarray, but got {type(coords)}.'
        vimi_logger.error(msg)
        raise ValueError(msg)
    if coords.shape[-1] % 2 != 0:
        msg: str = f'coords must have an even number of columns, but got {type(coords.shape[-1])}.'
        vimi_logger.error(msg)
        raise ValueError(msg)
    coords_pairs: int = coords.shape[-1] // 2
    norm_array: np.ndarray = np.array(img_size * coords_pairs)
    return coords / norm_array

def xywh2xyxy(xywh: np.ndarray) -> np.ndarray:
    """Get a box representation by the top-left and bottom-right corners.
    Args:
        xywh (np.ndarray): n x [xc, yc, w, h, ...] array boxes center-dimensions representation.
    Raises:
        ValueError: On input shape not like (n, 4+).
    Returns:
        np.ndarray: n x [x1, y1, x2, y2, ...] array with boxes two-corner representation.
    """
    if not isinstance(xywh, np.ndarray):
        msg: str = f'xywh must be a np.ndarray, but got {type(xywh)}.'
        vimi_logger.error(msg)
        raise ValueError(msg)
    shape = xywh.shape
    if xywh.ndim != 2 or shape[1] < 4:
        msg: str = f'xywh shape must be (n, 4+), but got {shape}.'
        vimi_logger.error(msg)
        raise ValueError(msg)
    extra_cols: np.ndarray | None = xywh[..., 4:] if shape[1] > 4 else None
    x0y0 = xywh[:, :2] - xywh[:, 2:] / 2
    x1y1 = x0y0 + xywh[:, 2:]
    res: np.ndarray = np.concatenate((x0y0, x1y1), axis=1)
    if extra_cols is not None:
        res = np.concatenate([res, extra_cols], axis= 1)
    return res

def xyxy2xywh(xyxy: np.ndarray) -> np.ndarray:
    """Get a box representated by the center, width and height.
    Args:
        xyxy (np.ndarray): n x [x1, y1, x2, y2, ...] array with boxes two-corner representation.
    Raises:
        ValueError: On input shape not like (n, 4+).
    Returns:
        np.ndarray: n x [xc, yc, w, h, ...] array boxes center-dimensions representation.
    """
    if not isinstance(xyxy, np.ndarray):
        msg: str = f'xyxy must be a np.ndarray, but got {type(xyxy)}.'
        vimi_logger.error(msg)
        raise ValueError(msg)
    shape = xyxy.shape
    if xyxy.ndim != 2 or shape[1] < 4:
        msg: str = f'xyxy shape must be (n, 4+), but got {shape}.'
        vimi_logger.error(msg)
        raise ValueError(msg)
    extra_cols: np.ndarray | None = xyxy[..., 4:] if shape[1] > 4 else None
    wh = xyxy[:, 2:] - xyxy[:, :2]
    xy = xyxy[:, :2] + wh / 2
    res: np.ndarray = np.concatenate((xy, wh), axis=1)
    if extra_cols is not None:
        res = np.concatenate([res, extra_cols], axis= 1)
    return res

def xywhr2xyxyxyxy(xywhr: np.ndarray) -> np.ndarray:
    """Get the coordinates of the corners of an array of rotated boxes.
    Args:
        xywhr (np.ndarray): n x [xc, yc, w, h, rad, ...] array with rotated-boxes representation.
    Raises:
        ValueError: On input shape not like (n, 5+).
    Returns:
        np.ndarray: n x [x1, y1, x2, y2, x3, y3, x4, y4] with the corners representation.
    """
    if not isinstance(xywhr, np.ndarray):
        msg: str = f'xywhr must be a np.ndarray, but got {type(xywhr)}.'
        vimi_logger.error(msg)
        raise ValueError(msg)
    shape = xywhr.shape
    if xywhr.ndim != 2 or shape[1] < 5:
        msg: str = f'xywhr shape must be (n, 5+), but got {shape}.'
        vimi_logger.error(msg)
        raise ValueError(msg)
    extra_cols: np.ndarray | None = xywhr[..., 5:] if shape[1] > 5 else None
    ctr: np.ndarray = xywhr[..., :2]
    w: np.ndarray
    h: np.ndarray
    angle: np.ndarray
    w, h, angle = (xywhr[..., i : i + 1] for i in range(2, 5))
    cos_value: np.ndarray = np.cos(angle)
    sin_value: np.ndarray = np.sin(angle)
    vec1: list[np.ndarray] = [w / 2 * cos_value, w / 2 * sin_value]
    vec2: list[np.ndarray] = [-h / 2 * sin_value, h / 2 * cos_value]
    vec1_array: np.ndarray = np.concatenate(vec1, -1)
    vec2_array: np.ndarray = np.concatenate(vec2, -1)
    pt1 = ctr + vec1_array + vec2_array
    pt2 = ctr + vec1_array - vec2_array
    pt3 = ctr - vec1_array - vec2_array
    pt4 = ctr - vec1_array + vec2_array
    res: np.ndarray = np.stack([pt1, pt2, pt3, pt4], -2)
    res = res.reshape(shape[0], -1)
    if extra_cols is not None:
        res = np.concatenate([res, extra_cols], axis= 1)
    return res

def xyxyxyxy2poly(xyxyxyxy: np.ndarray) -> Polygon:
    """Get a shapely Polygon from an xyxyxyxy array.
    Args:
        xyxyxyxy (np.ndarray): [x1, y1, x2, y2, x3, y3, x4, y4, ...] array with box corners.
    Raises:
        ValueError: On input shape not like (8+,).
    Returns:
        Polygon: a shapely.geometry.Polygon object.
    """
    if xyxyxyxy.shape[0] < 8 and xyxyxyxy.ndim != 1:
        msg: str = f'box shape must be (8+,), but got {xyxyxyxy.shape}.'
        vimi_logger.error(msg)
        raise ValueError(msg)
    return Polygon(xyxyxyxy[:8].reshape(4, 2))

def iou(first: Polygon, second: Polygon) -> float:
    """Compute the Intersection Over Union (IoU) from two shapely Polygons.
    Args:
        first (Polygon): The first Polygon.
        second (Polygon): The second Polygon.
    Returns:
        float: The IoU value.
    """
    if not first.intersects(second):
        return 0.
    intersection: float = first.intersection(second).area
    union: float = first.area + second.area - intersection
    iou: float = intersection / union if union > 0 else 0.
    return iou
