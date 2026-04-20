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


from functools import lru_cache
from math import degrees
from typing import Any, Optional

import numpy as np
from typing_extensions import Self, TypedDict

from .plot import PALLETTE, Color, plot_label, plot_polygon


def _parse_np_str(array: np.ndarray) -> str:
    return np.array2string(
        array,
        separator=', ',
        precision=4,
        suppress_small=True
    )


class SpeedDict(TypedDict):
    preprocess: Optional[float]
    inference: Optional[float]
    postprocess: Optional[float]


class ResultDataWrapper:
    def __init__(self, data: np.ndarray, orig_shape: tuple[int, int]) -> None:
        self.data: np.ndarray = data
        self.orig_shape: tuple[int, int] = orig_shape[:2]

    def __getitem__(self, idx: int | slice) -> Self:
        return self.__class__(self.data[idx], self.orig_shape)


class Boxes(ResultDataWrapper):
    def __init__(
        self,
        boxes: np.ndarray,  # [x0, y0, x1, y1, conf, id] x n
        orig_shape: tuple[int, int]  # (h, w)
    ) -> None:
        if boxes.ndim == 1:
            boxes = boxes[None, :]
        assert boxes.ndim == 2
        assert boxes.shape[-1] == 6
        super().__init__(data= boxes, orig_shape= orig_shape)

    def __repr__(self) -> str:
        result: str = 'Boxes object:\n'
        result += f'cls: {self.cls}\n'
        result += f'data: {_parse_np_str(self.data)}\n'
        result += f'orig_shape: {self.orig_shape}\n'
        result += f'xywh: {_parse_np_str(self.xywh)}\n'
        result += f'xywhn: {_parse_np_str(self.xywhn)}\n'
        result += f'xyxy: {_parse_np_str(self.xyxy)}\n'
        result += f'xyxyn: {_parse_np_str(self.xyxyn)}'
        return result

    @property
    @lru_cache(maxsize=2)
    def xyxy(self) -> np.ndarray:
        return self.data[:, :4]

    @property
    @lru_cache(maxsize=2)
    def conf(self) -> np.ndarray:
        return self.data[:, -2]

    @property
    @lru_cache(maxsize=2)
    def cls(self) -> np.ndarray:
        return self.data[:, -1]

    @property
    @lru_cache(maxsize=2)
    def xywh(self) -> np.ndarray:
        return self.xyxy2xywh(self.xyxy)

    @property
    @lru_cache(maxsize=2)
    def xyxyn(self) -> np.ndarray:
        return self.norm_coords(self.xyxy, self.orig_shape)

    @property
    @lru_cache(maxsize=2)
    def xywhn(self) -> np.ndarray:
        return self.norm_coords(self.xywh, self.orig_shape)

    @property
    @lru_cache(maxsize=2)
    def vertex(self) -> np.ndarray:
        xyxy: np.ndarray = self.xyxy.copy()
        x1: np.ndarray = xyxy[:, 0]
        y1: np.ndarray = xyxy[:, 1]
        x2: np.ndarray = xyxy[:, 2]
        y2: np.ndarray = xyxy[:, 3]
        return np.column_stack([
            np.minimum(x1, x2),
            np.minimum(y1, y2),
            np.maximum(x1, x2),
            np.minimum(y1, y2),
            np.minimum(x1, x2),
            np.maximum(y1, y2),
            np.maximum(x1, x2),
            np.maximum(y1, y2)
        ])

    @staticmethod
    def xyxy2xywh(xyxy: np.ndarray) -> np.ndarray:
        wh = xyxy[:, 2:] - xyxy[:, :2]
        xy = xyxy[:, :2] + wh / 2
        return np.concatenate((xy, wh), axis=1)

    @staticmethod
    def xywh2xyxy(xywh: np.ndarray) -> np.ndarray:
        x0y0 = xywh[:, :2] - xywh[:, 2:] / 2
        x1y1 = x0y0 + xywh[:, 2:]
        return np.concatenate((x0y0, x1y1), axis=1)

    @staticmethod
    def norm_coords(coords: np.ndarray, img_size: tuple[int, int]) -> np.ndarray:
        norm_array: np.ndarray = np.array(img_size + img_size)
        return coords / norm_array

    def plot(
        self,
        img: np.ndarray,
        names: dict[int, str],
        conf: bool = True,
        line_width: float | None = None,
        font_size: float | None = None,
        labels: bool = True,
        boxes: bool = True,
        pallette: tuple[Color, ...] = PALLETTE,
        *args,
        **kwargs
    ) -> np.ndarray:
        if not(boxes or labels or conf):
            return img
        for i in range(self.data):
            if boxes:
                plot_polygon(
                    img= img,
                    vertex= self.vertex[i],
                    cls= self.cls[i],
                    line_width= line_width,
                    pallette= pallette
                )
            if labels or conf:
                plot_label(
                    img= img,
                    p0= self.xyxy[:2],
                    cls= self.cls[i],
                    conf_val= self.conf[i],
                    names= names,
                    conf= conf,
                    labels= labels,
                    font_size= font_size,
                    line_width= line_width,
                    pallette= pallette
                )
        return img


class Masks(ResultDataWrapper):
    #TODO
    def __init__(
        self,
        masks: np.ndarray,  # 
        orig_shape: tuple[int, int]  # (h, w)
    ) -> None:
        if masks.ndim == 2:
            masks = masks[None, :]
        assert masks.ndim == 2
        super().__init__(data= masks, orig_shape= orig_shape)

    def plot(
        self,
        img: np.ndarray,
        names: dict[int, str],
        conf: bool = True,
        line_width: float | None = None,
        font_size: float | None = None,
        font: str = 'Arial.ttf',
        pil: bool = False,
        im_gpu: Any = None,
        kpt_radius: int = 5,
        kpt_line: bool = True,
        labels: bool = True,
        boxes: bool = True,
        masks: bool = True,
        probs: bool = True,
        show: bool = False,
        save: bool = False,
        filename: str | None = None,
        color_mode: str = 'class',
        txt_color: Color = (255, 255, 255),
        pallette: tuple[Color, ...] = PALLETTE,
        *args,
        **kwargs
    ) -> np.ndarray:
        if not masks:
            return img
        ...
        return img


class Probs(ResultDataWrapper):
    def __init__(
        self,
        probs: np.ndarray,  # [prob0, prob1, prob2, prob3, ...]
        orig_shape: tuple[int, int]  # (h, w)
    ) -> None:
        super().__init__(data= probs, orig_shape= orig_shape)

    def __repr__(self) -> str:
        result: str = 'Probs object:\n'
        result += f'data: {_parse_np_str(self.data)}\n'
        result += f'top1: {self.top1}({_parse_np_str(self.data[self.top1])})\n'
        result += f'orig_shape: {self.orig_shape}\n'
        return result

    @property
    @lru_cache(maxsize= 1)
    def top1(self) -> int:
        return int(self.data.argmax())

    @property
    @lru_cache(maxsize= 1)
    def top5(self) -> list[int]:
        return (-self.data).argsort(axis= 0)[:5].tolist()

    @property
    @lru_cache(maxsize= 1)
    def top1conf(self) -> float:
        return float(self.data[self.top1])

    @property
    @lru_cache(maxsize= 1)
    def top4conf(self) -> np.ndarray:
        return self.data[self.top5]

    @property
    @lru_cache(maxsize= 1)
    def vertex(self) -> np.ndarray:
        return np.array((
            0,
            0,
            self.orig_shape[1],
            0,
            self.orig_shape[1],
            self.orig_shape[0],
            0,
            self.orig_shape[0]
        ))

    def plot(
        self,
        img: np.ndarray,
        names: dict[int, str],
        conf: bool = True,
        line_width: float | None = None,
        font_size: float | None = None,
        font: str = 'Arial.ttf',
        pil: bool = False,
        im_gpu: Any = None,
        kpt_radius: int = 5,
        kpt_line: bool = True,
        labels: bool = True,
        boxes: bool = True,
        masks: bool = True,
        probs: bool = True,
        show: bool = False,
        save: bool = False,
        filename: str | None = None,
        color_mode: str = 'class',
        txt_color: Color = (255, 255, 255),
        pallette: tuple[Color, ...] = PALLETTE,
        *args,
        **kwargs
    ) -> np.ndarray:
        if not(probs or labels or conf):
            return img
        if probs:
            plot_polygon(
                img= img,
                vertex= self.vertex,
                cls= self.top1,
                line_width= line_width,
                pallette= pallette
            )
        if labels or conf:
            plot_label(
                img= img,
                p0= self.vertex[:2],
                cls= self.top1,
                conf_val= self.top1conf,
                names= names,
                conf= conf,
                labels= labels,
                font_size= font_size,
                line_width= line_width,
                pallette= pallette
            )
        return img


class Keypoints(ResultDataWrapper):
    def __init__(
        self,
        keypoints: np.ndarray,  # 
        orig_shape: tuple[int, int]  # (h, w)
    ) -> None:
        if keypoints.ndim == 2:
            keypoints = keypoints[None, :]
        assert keypoints.ndim == 2
        #TODO
        super().__init__(data= keypoints, orig_shape= orig_shape)

    def plot(
        self,
        img: np.ndarray,
        names: dict[int, str],
        conf: bool = True,
        line_width: float | None = None,
        font_size: float | None = None,
        font: str = 'Arial.ttf',
        pil: bool = False,
        im_gpu: Any = None,
        kpt_radius: int = 5,
        kpt_line: bool = True,
        labels: bool = True,
        boxes: bool = True,
        masks: bool = True,
        probs: bool = True,
        show: bool = False,
        save: bool = False,
        filename: str | None = None,
        color_mode: str = 'class',
        txt_color: Color = (255, 255, 255),
        pallette: tuple[Color, ...] = PALLETTE,
        *args,
        **kwargs
    ) -> np.ndarray:
        ...
        return img


class OBB(ResultDataWrapper):
    def __init__(
        self,
        obb: np.ndarray,  # [xc, yc, w, h, rad, conf, id] x n
        orig_shape: tuple[int, int]  # (h, w)
    ) -> None:
        if obb.ndim == 1:
            obb = obb[None, :]
        assert obb.ndim == 2
        assert obb.shape[-1] == 6
        super().__init__(data= obb, orig_shape= orig_shape)

    def __repr__(self) -> str:
        result: str = 'Boxes object:\n'
        result += f'cls: {self.cls}\n'
        result += f'conf: {self.conf}\n'
        result += f'data: {_parse_np_str(self.data)}\n'
        result += f'orig_shape: {self.orig_shape}\n'
        result += f'xywhr: {_parse_np_str(self.xywhr)}\n'
        result += f'xyxyxyxy: {_parse_np_str(self.xyxyxyxy)}\n'
        result += f'xyxyxyxyn: {_parse_np_str(self.xyxyxyxyn)}\n'
        result += f'xyxy: {_parse_np_str(self.xyxy)}\n'
        return result

    @property
    @lru_cache(maxsize=2)
    def xywhr(self) -> np.ndarray:
        return self.data[:, :5]

    @property
    @lru_cache(maxsize=2)
    def conf(self) -> np.ndarray:
        return self.data[:, -2]

    @property
    @lru_cache(maxsize=2)
    def cls(self) -> np.ndarray:
        return self.data[:, -1]

    @property
    @lru_cache(maxsize=2)
    def r(self) -> float:
        return self.data[4]

    @property
    @lru_cache(maxsize=2)
    def r_deg(self) -> float:
        return degrees(self.r)

    @property
    @lru_cache(maxsize=2)
    def xyxyxyxy(self) -> np.ndarray:
        return OBB.xywhr2xyxyxyxy(self.xywhr)

    @property
    @lru_cache(maxsize=2)
    def xyxyxyxyn(self) -> np.ndarray:
        xyxyxyxyn: np.ndarray = self.xyxyxyxy.copy()
        xyxyxyxyn[..., 0] /= self.orig_shape[1]
        xyxyxyxyn[..., 1] /= self.orig_shape[0]
        return xyxyxyxyn

    @property
    @lru_cache(maxsize=2)
    def xyxy(self) -> np.ndarray:
        x: np.ndarray = self.xyxyxyxy[..., 0]
        y: np.ndarray = self.xyxyxyxy[..., 1]
        return np.stack([x.min(1), y.min(1), x.max(1), y.max(1)], -1)

    @staticmethod
    def xywhr2xyxyxyxy(xywhr: np.ndarray) -> np.ndarray:
        ctr: np.ndarray = xywhr[..., :2]
        w: float
        h: float
        angle: float
        w, h, angle = (float(xywhr[..., i : i + 1]) for i in range(2, 5))
        cos_value: float
        sin_value: float
        cos_value, sin_value = np.cos(angle), np.sin(angle)
        vec1: list[float] = [w / 2 * cos_value, w / 2 * sin_value]
        vec2: list[float] = [-h / 2 * sin_value, h / 2 * cos_value]
        vec1_array: np.ndarray = np.concatenate(vec1, -1)
        vec2_array: np.ndarray = np.concatenate(vec2, -1)
        pt1 = ctr + vec1_array + vec2
        pt2 = ctr + vec1_array - vec2_array
        pt3 = ctr - vec1_array - vec2_array
        pt4 = ctr - vec1_array + vec2_array
        return np.stack([pt1, pt2, pt3, pt4], -2)

    def plot(
        self,
        img: np.ndarray,
        names: dict[int, str],
        conf: bool = True,
        line_width: float | None = None,
        font_size: float | None = None,
        font: str = 'Arial.ttf',
        pil: bool = False,
        im_gpu: Any = None,
        kpt_radius: int = 5,
        kpt_line: bool = True,
        labels: bool = True,
        boxes: bool = True,
        masks: bool = True,
        probs: bool = True,
        show: bool = False,
        save: bool = False,
        filename: str | None = None,
        color_mode: str = 'class',
        txt_color: Color = (255, 255, 255),
        pallette: tuple[Color, ...] = PALLETTE,
        *args,
        **kwargs
    ) -> np.ndarray:
        if not(boxes or labels or conf):
            return img
        for i in range(self.data):
            if boxes:
                plot_polygon(
                    img= img,
                    vertex= self.xyxyxyxy[i],
                    cls= int(self.cls[i]),
                    line_width= line_width,
                    pallette= pallette
                )
            if labels or conf:
                plot_label(
                    img= img,
                    p0= self.xyxyxyxy[i][:2],
                    cls= int(self.cls[i]),
                    conf_val= self.conf[i],
                    names= names,
                    conf= conf,
                    labels= labels,
                    font_size= font_size,
                    line_width= line_width,
                    pallette= pallette
                )
        return img


class Results:
    def __init__(
        self,
        orig_img: np.ndarray,
        path: str,
        names: dict[int, str],
        boxes: Optional[np.ndarray] = None,  # Detection boxes: [x1, y1, x2, y2, conf, id] x n,
        masks: Optional[np.ndarray] = None,  # Segmentetion masks: 
        probs: Optional[np.ndarray] = None,  # Classification probs: [prob1, prob2, prob3, ...]
        keypoints: Optional[np.ndarray] = None,  # Keypoints: 
        obb: Optional[np.ndarray] = None,  # Oriented boxes: [xc, yc, w, h, rad, conf, id] x n
        speed: Optional[SpeedDict] = None
    ) -> None:
        self.orig_img: np.ndarray = orig_img
        self.orig_shape: tuple[int, int] = orig_img.shape[:2]  # (h, w)
        self.path: str = path
        self.names: dict[int, str] = names
        self.set_boxes(boxes= boxes)
        self.set_masks(masks= masks)
        self.set_probs(probs= probs)
        self.set_keypoints(keypoints= keypoints)
        self.set_obb(obb= obb)
        self.speed: SpeedDict = speed if speed is not None else SpeedDict(preprocess= None, inference= None, postprocess= None)
        self._keys = ('boxes', 'masks', 'probs', 'keypoints', 'obb')

    def __repr__(self) -> str:
        result: str = 'Results object:\n'
        result += f'cls: {self.names}\n'
        result += f'orig_shape: {self.orig_shape}\n'
        result += 'probs: ' + str(self.probs).replace('\n', ' ') + '\n'
        result += f'boxes: ' + str(self.boxes).replace('\n', ' ') + '\n'
        result += f'masks: ' + str(self.masks).replace('\n', ' ') + '\n'
        result += f'keypoints: ' + str(self.keypoints).replace('\n', ' ') + '\n'
        result += f'obb: {self.obb}\n'
        result += f'speed: {self.speed}\n'
        return result

    def __getitem__(self, idx: int, slice) -> Self:
        return self._apply('__getitem__', idx)

    def _apply(self, fn: str, *args, **kwargs) -> Self:
        r: Self = self.copy()
        for k in self._keys:
            v: Boxes | Masks | Probs | Keypoints | OBB | None = getattr(self, k)
            if v is None:
                continue
            setattr(r, k, getattr(v, fn)(*args, **kwargs))
        return r

    def set_boxes(self, boxes: Optional[np.ndarray] = None) -> None:
        self.boxes: Optional[Boxes] = Boxes(boxes= boxes, orig_shape= self.orig_shape) if boxes is not None else None

    def set_masks(self, masks: Optional[np.ndarray] = None) -> None:
        self.masks: Optional[Masks] = Masks(masks= masks, orig_shape= self.orig_shape) if masks is not None else None

    def set_probs(self, probs: Optional[np.ndarray] = None) -> None:
        self.probs: Optional[Probs] = Probs(probs= probs, orig_shape= self.orig_shape) if probs is not None else None

    def set_keypoints(self, keypoints: Optional[np.ndarray] = None) -> None:
        self.keypoints: Optional[Keypoints] = Keypoints(keypoints= keypoints, orig_shape= self.orig_shape) if keypoints is not None else None

    def set_obb(self, obb: Optional[np.ndarray] = None) -> None:
        self.obb: Optional[OBB] = OBB(obb= obb, orig_shape= self.orig_shape) if obb is not None else None

    def copy(self) -> Self:
        return self.__class__(
            orig_img= self.orig_img,
            path= self.path,
            names= self.names,
            speed= self.speed
        )

    def plot(
        self,
        img: Optional[np.ndarray] = None,
        conf: bool = True,
        line_width: float | None = None,
        font_size: float | None = None,
        font: str = 'Arial.ttf',
        pil: bool = False,
        im_gpu: Any = None,
        kpt_radius: int = 5,
        kpt_line: bool = True,
        labels: bool = True,
        boxes: bool = True,
        masks: bool = True,
        probs: bool = True,
        show: bool = False,
        save: bool = False,
        filename: str | None = None,
        color_mode: str = 'class',
        txt_color: Color = (255, 255, 255),
        pallette: tuple[Color, ...] = PALLETTE,
    ) -> np.ndarray:
        if img is None:
            img = self.orig_img
        for k in self._keys:
            v: Boxes | Masks | Probs | Keypoints | OBB | None = getattr(self, k)
            if v is None:
                continue
            img = v.plot(
                img= img,
                names= self.names,
                conf= conf,
                line_width= line_width,
                font_size= font_size,
                font= font,
                pil= pil,
                im_gpu= im_gpu,
                kpt_radius= kpt_radius,
                kpt_line= kpt_line,
                labels= labels,
                boxes= boxes,
                masks= masks,
                probs= probs,
                show= show,
                save= save,
                filename= filename,
                color_mode= color_mode,
                txt_color= txt_color,
                pallette= pallette
            )
        return img
