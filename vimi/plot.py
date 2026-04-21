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


import cv2
import numpy as np

from .logs import vimi_logger

Color = tuple[int, int, int]


PALLETTE: tuple[Color, ...]= (
    (79, 68, 255),
    (68, 243, 0),
    (255, 42, 4),
    (235, 219, 11),
    (243, 243, 243),
    (183, 223, 0),
    (104, 31, 17),
    (221, 111, 255),
    (0, 237, 204),
    (255, 0, 189),
    (128, 128, 128),
    (0, 128, 0),
    (255, 255, 0),
    (0, 255, 255),
    (255, 0, 0),
    (255, 192, 203),
    (0, 0, 128),
    (64, 224, 208),
    (255, 20, 147)
)

def get_color(i: int, pallette: tuple[Color, ...] = PALLETTE) -> Color:
    return pallette[i % len(pallette)]

def get_font_color(color: Color) -> Color:
    b: int
    g: int
    r: int
    b, g, r = color
    brightness: float = 0.299 * r + 0.587 * g + 0.114 * b
    return (0, 0, 0) if brightness > 127.5 else (255, 255, 255)

def plot_polygon(
    img: np.ndarray,
    vertex: np.ndarray,  # [px0, py0, px1, py1, px2, py2, ...]
    cls: int,
    line_width: float | None = None,
    pallette: tuple[Color, ...] = PALLETTE
) -> None:
    if line_width is None:
        line_width = float(np.min(img.shape[:2])) * 0.004
    color: Color = get_color(cls, pallette)
    if len(vertex) % 2 != 0:
        msg: str = f'A vertex array should contain an even number of elements. Got {len(vertex)}. Polygon won\'t be plotted.'
        vimi_logger.error(msg)
        return None
    vertex = vertex.reshape(-1, 2).astype(np.int32)
    cv2.polylines(
        img,
        [vertex],
        isClosed= True,
        color= color,
        thickness= int(line_width)
    )

def plot_label(
    img: np.ndarray,
    p0: np.ndarray,
    cls: int,
    conf_val: float,
    names: dict[int, str],
    conf: bool = True,
    labels: bool = True,
    font_size: float | None = None,
    line_width: float | None = None,
    pallette: tuple[Color, ...] = PALLETTE
) -> None:
    # Constants
    if font_size is None:
        font_size = float(np.min(img.shape[:2])) * 0.001
    if line_width is None:
        line_width = float(np.min(img.shape[:2])) * 0.004
    color: Color = get_color(cls, pallette)
    font: int = cv2.FONT_HERSHEY_SIMPLEX
    font_color: tuple = get_font_color(color)
    img_shape: tuple = img.shape
    # Text content
    text: str = ''
    if labels:
        try:
            text: str = f'{names[cls]}'
        except Exception as e:
            text: str = f'{cls}'
    if conf:
        text += f' {conf_val:.2f}'
    text = text.strip()
    # Text size
    (txt_w, txt_h), _ = cv2.getTextSize(
        text,
        font,
        font_size,
        1
    )
    # Text p0
    inf_left_corner: list[int] = [
        int(p0[0]),
        int(p0[1] - line_width)
    ]
    text_p0: list[int] = [
        int(p0[0] + line_width),
        int(p0[1] - 2 * line_width)
    ]
    sup_right_corner: list[int] = [
        int(p0[0] + txt_w + line_width),
        int(p0[1] - txt_h - 3 * line_width)
    ]
    # Fix text out of borders X
    if sup_right_corner[0] > img_shape[1]:
        shift: int = sup_right_corner[0] - img_shape[1]
        inf_left_corner[0] -= shift
        text_p0[0] -= shift
        sup_right_corner[0] -= shift
    if inf_left_corner[0] < 0:
        shift: int = - inf_left_corner[0]
        inf_left_corner[0] += shift
        text_p0[0] += shift
        sup_right_corner[0] += shift
    # Fix text out of borders Y
    if inf_left_corner[1] > img_shape[0]:
        shift: int = inf_left_corner[1] - img_shape[0]
        inf_left_corner[1] -= shift
        text_p0[1] -= shift
        sup_right_corner[1] -= shift
    if sup_right_corner[1] < 0:
        shift: int = - sup_right_corner[1]
        inf_left_corner[1] += shift
        text_p0[1] += shift
        sup_right_corner[1] += shift
    # Plot
    cv2.rectangle(
        img= img,
        pt1= inf_left_corner,
        pt2= sup_right_corner,
        color= color,
        thickness= -1
    )
    cv2.putText(
        img= img,
        text= text,
        org= text_p0,
        fontFace= font,
        fontScale= font_size,
        color= font_color,
        thickness= int(float(np.min(img.shape[:2])) * 0.002),
        lineType= cv2.LINE_AA
    )
