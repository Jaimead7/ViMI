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


from collections.abc import Callable
from typing import Any, ClassVar, Protocol

import cv2
import numpy as np

from .logs import vimi_logger


class CoordsTransformer:
    def __init__(
        self,
        scale: tuple[float, float] = (1, 1),
        offset: tuple[int, int] = (0, 0)
    ) -> None:
        self.scaleX: float
        self.scaleY: float
        self.scaleX, self.scaleY = scale
        self.offsetX: float
        self.offsetY: float
        self.offsetX, self.offsetY = offset

    def res2org(self, values: np.ndarray) -> np.ndarray:
        values = values.copy()
        values[:,0] = (values[:, 0] / self.scaleX) + self.offsetX
        values[:,1] = (values[:, 1] / self.scaleY) + self.offsetY
        values[:,2] = (values[:, 2] / self.scaleX) + self.offsetX
        values[:,3] = (values[:, 3] / self.scaleY) + self.offsetY
        return values

    def org2res(self, values: np.ndarray) -> np.ndarray:
        values = values.copy()
        values[:,0] = (values[:, 0] - self.offsetX) * self.scaleX
        values[:,1] = (values[:, 1] - self.offsetY) * self.scaleY
        values[:,2] = (values[:, 2] - self.offsetX) * self.scaleX
        values[:,3] = (values[:, 3] - self.offsetY) * self.scaleY
        return values

    def xywh2org(self, xywh: np.ndarray) -> np.ndarray:
        values: np.ndarray = xywh.copy()
        values[:,0] = (values[:, 0] / self.scaleX) + self.offsetX
        values[:,1] = (values[:, 1] / self.scaleY) + self.offsetY
        values[:,2] = values[:, 2] / self.scaleX
        values[:,3] = values[:, 3] / self.scaleY
        return values

    def org2xywh(self, xywh: np.ndarray) -> np.ndarray:
        values: np.ndarray = xywh.copy()
        values[:,0] = (values[:, 0] - self.offsetX) * self.scaleX
        values[:,1] = (values[:, 1] - self.offsetY) * self.scaleY
        values[:,2] = values[:, 2] * self.scaleX
        values[:,3] = values[:, 3] * self.scaleY
        return values


class ImageFilter(Protocol):
    def __call__(
        self,
        img: np.ndarray,
        *args: Any,
        **kwargs: Any
    ) -> tuple[np.ndarray, CoordsTransformer]:
        ...


class ImageFiltersReg:
    _filters: ClassVar[dict[str, ImageFilter]] = {}

    def __new__(cls) -> None:
        msg: str = f'Class "{cls.__name__}" is not instantiable.'
        vimi_logger.critical(msg)
        raise TypeError(msg)

    @staticmethod
    def no_filter(img: np.ndarray) -> tuple[np.ndarray, CoordsTransformer]:
        return img, CoordsTransformer()

    @classmethod
    def register(cls, name: str) -> Callable[[ImageFilter], ImageFilter]:
        def decorator(func: ImageFilter) -> ImageFilter:
            if name.upper() in cls._filters:
                vimi_logger.warning(f'ImageFilter "{name.upper()}" is already registered. It will be overwritten.')
            cls._filters[name.upper()] = func
            vimi_logger.debug(f'{name.upper()} registered in {cls.__name__} for function {func}.')
            return func
        return decorator

    @classmethod
    def unregister(cls, name: str) -> None:
        cls._filters.pop(name.upper(), None)
        vimi_logger.debug(f'{name.upper()} unregistered from {cls.__name__}.')

    @classmethod
    def get(cls, name: str) -> ImageFilter:
        return cls._filters.get(name.upper(), cls.no_filter)

    @classmethod
    def list(cls) -> list[str]:
        return sorted(cls._filters.keys())

    @classmethod
    def clear(cls) -> None:
        cls._filters.clear()
        vimi_logger.debug(f'{cls.__name__} cleared.')


@ImageFiltersReg.register('RESIZE')
def resize(
    img: np.ndarray,
    width: int = 640,
    height: int = 640
) -> tuple[np.ndarray, CoordsTransformer]:
    img_resized: np.ndarray = cv2.resize(
        img,
        (width, height),
        interpolation= cv2.INTER_LINEAR
    )
    org_h: int
    org_w: int
    org_h, org_w = img.shape[:2]
    transformer: CoordsTransformer = CoordsTransformer(
        scale= (width/org_w, height/org_h),
        offset= (0, 0)
    )
    return img_resized, transformer

@ImageFiltersReg.register('REDIM')
def redim(
    img: np.ndarray,
    height: int = 640,
    width: int = 640,
    gray: int = 114
) -> tuple[np.ndarray, CoordsTransformer]:
    org_h: int
    org_w: int
    org_h, org_w = img.shape[:2]
    scale: float = min(width/org_w, height/org_h)
    new_w = int(org_w * scale)
    new_h = int(org_h * scale)
    img_resized: np.ndarray = cv2.resize(
        img,
        (new_w, new_h),
        interpolation= cv2.INTER_LINEAR
    )
    result: np.ndarray = np.ones((height, width, 3), dtype= np.uint8) * gray
    result[:new_h, :new_w] = img_resized
    transformer: CoordsTransformer = CoordsTransformer(
        scale= (scale, scale),
        offset= (0, 0)
    )
    return result, transformer

@ImageFiltersReg.register('CUT')
def cut(
    img: np.ndarray,
    p0: tuple[int, int] = (0, 0),
    width: int = 640,
    height: int = 640
) -> tuple[np.ndarray, CoordsTransformer]:
    p1: tuple[int, int] = (p0[0] + width, p0[1] + height)
    h: int
    w: int
    h, w = img.shape[:2]
    y0: int = max(0, min(p0[1], h))
    y1: int = max(0, min(p1[1], h))
    x0: int = max(0, min(p0[0], w))
    x1: int = max(0, min(p1[0], w))
    result: np.ndarray = img[y0:y1, x0:x1]
    transformer: CoordsTransformer = CoordsTransformer(
        scale= (1, 1),
        offset= (x0, y0)
    )
    return result.copy(), transformer

@ImageFiltersReg.register('GRAY')
def bgr2gray(
    img: np.ndarray
) -> tuple[np.ndarray, CoordsTransformer]:
    if len(img.shape) == 2:
        return img, CoordsTransformer()
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), CoordsTransformer()

@ImageFiltersReg.register('COLOR')
def gray2bgr(
    img: np.ndarray
) -> tuple[np.ndarray, CoordsTransformer]:
    if len(img.shape) == 3:
        return img, CoordsTransformer()
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), CoordsTransformer()

@ImageFiltersReg.register('RGB')
def bgr2rgb(
    img: np.ndarray
) -> tuple[np.ndarray, CoordsTransformer]:
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB), CoordsTransformer()

@ImageFiltersReg.register('BGR')
def rgb2bgr(
    img: np.ndarray
) -> tuple[np.ndarray, CoordsTransformer]:
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR), CoordsTransformer()
