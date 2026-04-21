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


def xywhr2xyxyxyxy(xywhr: np.ndarray) -> np.ndarray:
    """Get the coordinates of the corners of an array of rotated boxes.
    Args:
        xywhr (np.ndarray): n x [xc, yc, w, h, rad] array with rotated boxes representation.
    Returns:
        np.ndarray: n x [x1, y1, x2, y2, x3, y3, x4, y4] with the corners coords of the box.
    """
    if not isinstance(xywhr, np.ndarray):
        msg: str = f'xywhr must be a np.ndarrya, but got {type(xywhr)}.'
        vimi_logger.error(msg)
        raise ValueError(msg)
    shape = xywhr.shape
    if xywhr.ndim != 2 or shape[0] == 0 or shape[1] < 5:
        msg: str = f'xywhr shape must be (n,5+), but got {shape}.'
        vimi_logger.error(msg)
        raise ValueError(msg)
    ctr: np.ndarray = xywhr[..., :2]
    w: float
    h: float
    angle: float
    w, h, angle = (float(xywhr[..., i : i + 1]) for i in range(2, 5))
    cos_value: float = np.cos(angle)
    sin_value: float = np.sin(angle)
    vec1: list[float] = [w / 2 * cos_value, w / 2 * sin_value]
    vec2: list[float] = [-h / 2 * sin_value, h / 2 * cos_value]
    vec1_array: np.ndarray = np.concatenate(vec1, -1)
    vec2_array: np.ndarray = np.concatenate(vec2, -1)
    pt1 = ctr + vec1_array + vec2
    pt2 = ctr + vec1_array - vec2_array
    pt3 = ctr - vec1_array - vec2_array
    pt4 = ctr - vec1_array + vec2_array
    return np.stack([pt1, pt2, pt3, pt4], -2)

def xyxyxyxy2poly(box: np.ndarray) -> Polygon:
    if box.shape != (8,):
        msg: str = f'box shape must be (8,), but got {box.shape}.'
        vimi_logger.error(msg)
        raise ValueError(msg)
    return Polygon(box.reshape(4, 2))

def iou(first: Polygon, second: Polygon, threshold: float) -> bool:
    if not first.intersects(second):
        return False
    intersection: float = first.intersection(second).area
    union: float = first.area + second.area - intersection
    iou: float = intersection / union if union > 0 else 0.
    return iou < threshold
