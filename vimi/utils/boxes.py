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


def poly_from_xyxyxyxy(box: np.ndarray) -> Polygon:
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
