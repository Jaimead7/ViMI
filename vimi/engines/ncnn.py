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


from abc import abstractmethod
from collections.abc import Callable, Iterable, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, ClassVar, Generator, Optional

import cv2
import ncnn
import numpy as np

from ..filesystem import NCNNModelFolder
from ..filters import CoordsTransformer, bgr2rgb, gray2bgr, redim
from ..logs import vimi_logger
from ..results import Results, SpeedDict
from ..utils.boxes import iou, xywh2xyxy, xywhr2xyxyxyxy, xyxyxyxy2poly
from .model_engine import EnginesReg


class Postprocessor:
    @abstractmethod
    def __call__(
            self,
            result: Results,
            ncc_out: ncnn.Mat,
            transformers: Sequence[CoordsTransformer]
        ) -> Results: ...


class PostprocessorsReg:
    _postprocessors: ClassVar[dict[str, type[Postprocessor]]] = {}

    class NoProcess(Postprocessor):
        def __call__(
            self,
            result: Results,
            ncc_out: ncnn.Mat,
            transformers: Sequence[CoordsTransformer]
        ) -> Results:
            return result

    @classmethod
    def register(cls, name: str) -> Callable[..., type[Postprocessor]]:
        def decorator(postprocessor: type[Postprocessor]) -> type[Postprocessor]:
            cls._postprocessors[name.upper()] = postprocessor
            return postprocessor
        return decorator

    @classmethod
    def unregister(cls, name: str) -> None:
        cls._postprocessors.pop(name.upper(), None)

    @classmethod
    def get(cls, name: str) -> type[Postprocessor]:
        return cls._postprocessors.get(name.upper(), cls.NoProcess)

    @classmethod
    def list(cls) -> list[str]:
        return sorted(cls._postprocessors.keys())

    @classmethod
    def clear(cls) -> None:
        cls._postprocessors.clear()


@PostprocessorsReg.register('CLASSIFY')
class ClassifyPostProcessor(Postprocessor):
    def __call__(
        self,
        result: Results,
        ncc_out: ncnn.Mat,
        transformers: Sequence[CoordsTransformer]
    ) -> Results:
        probs: np.ndarray = np.array(ncc_out)
        result.set_probs(probs= probs)
        return result


@PostprocessorsReg.register('DETECT')
class DetectPostProcessor(Postprocessor):
    GLOBAL_CONF_THRESHOLD: float = 0.25
    IOU_THRESHOLD: float = 0.75

    def __call__(
        self,
        result: Results,
        ncc_out: ncnn.Mat,
        transformers: Sequence[CoordsTransformer]
    ) -> Results:
        out_array: np.ndarray = self.parse_ncnn_out(ncc_out)
        boxes: np.ndarray = self.filter_boxes(out_array)
        for transformer in transformers[::-1]:
            boxes = transformer.res2org(boxes)
        result.set_boxes(boxes)
        return result

    @classmethod
    def parse_ncnn_out(
        cls,
        model_out: ncnn.Mat
    ) -> np.ndarray:
        model_out_array: np.ndarray = np.array(model_out).T
        xywh: np.ndarray = model_out_array[:, :4]
        xyxy: np.ndarray = xywh2xyxy(xywh)
        all_conf: np.ndarray = model_out_array[:, 4:]
        max_conf: np.ndarray = np.max(all_conf, axis= 1, keepdims= True)
        max_index: np.ndarray = np.argmax(all_conf, axis= 1, keepdims= True)
        return np.hstack((xyxy, max_conf, max_index))

    @classmethod
    def filter_boxes(
        cls,
        boxes: np.ndarray
    ) -> np.ndarray:
        mask = boxes[:, 4] > cls.GLOBAL_CONF_THRESHOLD
        pot_boxes: np.ndarray = boxes[mask]
        filter_boxes: np.ndarray = cls.nms(pot_boxes)
        return filter_boxes

    @classmethod
    def nms(
        cls,
        boxes: np.ndarray
    ) -> np.ndarray:
        x0: np.ndarray = boxes[:, 0]
        y0: np.ndarray = boxes[:, 1]
        x1: np.ndarray = boxes[:, 2]
        y1: np.ndarray = boxes[:, 3]
        scores: np.ndarray = boxes[:, -2]
        areas: np.ndarray = (x1 - x0) * (y1 - y0)
        order: np.ndarray = scores.argsort()[::-1]
        keep: list[int] = []
        while order.size > 0:
            i: int = order[0]
            keep.append(i)
            xx0: np.ndarray = np.maximum(x0[i], x0[order[1:]])
            yy0: np.ndarray = np.maximum(y0[i], y0[order[1:]])
            xx1: np.ndarray = np.minimum(x1[i], x1[order[1:]])
            yy1: np.ndarray = np.minimum(y1[i], y1[order[1:]])
            w: np.ndarray = np.maximum(0.0, xx1 - xx0)
            h: np.ndarray = np.maximum(0.0, yy1 - yy0)
            intersection: np.ndarray = w * h
            union: np.ndarray = areas[i] + areas[order[1:]] - intersection
            iou: np.ndarray = intersection / union
            inds = np.where(iou <= cls.IOU_THRESHOLD)[0]
            order = order[inds + 1]
        return boxes[keep]


@PostprocessorsReg.register('OBB')
class OBBPostProcessor(Postprocessor):
    GLOBAL_CONF_THRESHOLD: float = 0.25
    IOU_THRESHOLD: float = 0.75

    def __call__(
        self,
        result: Results,
        ncc_out: ncnn.Mat,
        transformers: Sequence[CoordsTransformer]
    ) -> Results:
        out_array: np.ndarray = self.parse_ncnn_out(ncc_out)
        boxes: np.ndarray = self.filter_boxes(out_array)
        for transformer in transformers[::-1]:
            boxes = transformer.xywh2org(boxes)
        result.set_obb(boxes)
        return result

    @classmethod
    def parse_ncnn_out(
        cls,
        model_out: ncnn.Mat
    ) -> np.ndarray:
        model_out_array: np.ndarray = np.array(model_out).T
        xywh: np.ndarray = model_out_array[:, :4]
        r: np.ndarray = model_out_array[:, -1, np.newaxis]
        all_conf: np.ndarray = model_out_array[:, 4:-1]
        max_conf: np.ndarray = np.max(all_conf, axis= 1, keepdims= True)
        max_index: np.ndarray = np.argmax(all_conf, axis= 1, keepdims= True)
        return np.hstack((xywh, r, max_conf, max_index))

    @classmethod
    def filter_boxes(
        cls,
        boxes: np.ndarray  # [xc, yc, w, h, rad, conf, id] x n
    ) -> np.ndarray:
        mask = boxes[:, -2] > cls.GLOBAL_CONF_THRESHOLD
        pot_boxes: np.ndarray = boxes[mask]
        filter_boxes: np.ndarray = cls.nms(pot_boxes)
        return filter_boxes

    @classmethod
    def nms(
        cls,
        boxes: np.ndarray
    ) -> np.ndarray:
        scores: np.ndarray = boxes[:, -2]
        order: np.ndarray = scores.argsort()[::-1]
        xyxyxyxy: np.ndarray = xywhr2xyxyxyxy(boxes)
        poly_array: np.ndarray = np.array([xyxyxyxy2poly(coords) for coords in xyxyxyxy])
        keep: list[int] = []
        while order.size > 0:
            i: int = order[0]
            keep.append(i)
            iou_array: np.ndarray = np.array([
                iou(poly_array[i], other)
                for other in poly_array[order[1:]]
            ])
            inds = np.where(iou_array <= cls.IOU_THRESHOLD)[0]
            order = order[inds + 1]
        return boxes[keep]


@EnginesReg.register('NCNN')
class NCCEngine:
    def __init__(
        self,
        folder_path: Path,
        *args: Any,
        **kwargs: Any
    ) -> None:
        self.ncnn_folder = NCNNModelFolder(path= folder_path)
        self.postporcessor: Postprocessor = PostprocessorsReg.get(self.ncnn_folder.task)()
        self._load()

    def __call__(
        self,
        source: np.ndarray | str | Path | list | tuple
    ) -> list[Results]:
        # Get images arrays
        sources: Iterable[np.ndarray | str | Path]
        if not isinstance(source, tuple | list):
            sources = [source]
        else:
            sources = source
        raw_imgs: Generator[np.ndarray, None, None] = self.get_raw_imgs(sources)
        # Process images
        results: list[Results] = []
        for raw_img in raw_imgs:
            speed: SpeedDict = SpeedDict(
                preprocess= 0,
                inference= 0,
                postprocess= 0
            )
            # Pre-process
            start_time: datetime = datetime.now(timezone.utc)
            transformers: list[CoordsTransformer] = []
            filter_img: np.ndarray
            aux_tl: list[CoordsTransformer]
            filter_img, aux_tl = self.apply_model_filters(img= raw_img)
            transformers.extend(aux_tl)
            redim_img: np.ndarray
            aux_t: CoordsTransformer
            redim_img, aux_t = self.redim(filter_img)
            transformers.append(aux_t)
            in_img: ncnn.Mat = self.preprocess(redim_img)
            speed.preprocess = (datetime.now(timezone.utc) - start_time).microseconds / 1000.
            # Inference
            start_time = datetime.now(timezone.utc)
            with self._net.create_extractor() as ex:
                ex.input('in0', in_img)
                out0: ncnn.Mat
                ret: int
                ret, out0 = ex.extract('out0')
            speed.inference = (datetime.now(timezone.utc) - start_time).microseconds / 1000.
            # Prost-process
            start_time = datetime.now(timezone.utc)
            result: Results = Results(
                orig_img= raw_img,
                path= '',
                names= self.ncnn_folder.metadata.names,
                speed= speed
            )
            result = self.postporcessor(
                result= result,
                ncc_out= out0,
                transformers= transformers
            )
            result.speed.postprocess = (datetime.now(timezone.utc) - start_time).microseconds / 1000.
            results.append(result)
        return results

    def _load(self) -> None:
        self._net: ncnn.Net = ncnn.Net()
        self._net.load_param(str(self.ncnn_folder.param_path))
        self._net.load_model(str(self.ncnn_folder.bin_path))

    def get_raw_imgs(
        self,
        sources: Iterable[np.ndarray | str | Path]
    ) -> Generator[np.ndarray, None, None]:
        for source in sources:
            if isinstance(source, np.ndarray):
                yield source
                continue
            try:
                img: Optional[np.ndarray] = cv2.imread(str(source))
                if img is None:
                    raise ValueError(f'Could not read image from "{str(source)}".')
                yield img
            except Exception as e:
                msg: str = f'Error reading image "{source}": {str(e)}.'
                vimi_logger.error(msg)

    def apply_model_filters(
        self,
        img: np.ndarray
    ) -> tuple[np.ndarray, list[CoordsTransformer]]:
        filtered_img: np.ndarray = img.copy()
        transformers: list[CoordsTransformer] = []
        for filter_fn, filter_attrs in self.ncnn_folder.filters:
            transformer: CoordsTransformer
            filtered_img, transformer = filter_fn(filtered_img, **filter_attrs)
            transformers.append(transformer)
        return filtered_img, transformers

    def redim(self, img: np.ndarray) -> tuple[np.ndarray, CoordsTransformer]:
        exp_shape: tuple[int, int, int] = (
            *self.ncnn_folder.metadata.imgsz,
            self.ncnn_folder.metadata.channels
        )
        shape: tuple[int, ...] = img.shape
        if len(shape) == 2:
            img, _ = gray2bgr(img)
        if len(shape) != 3:
            msg: str = f'Input img should be (h, w, 3) not {shape}.'
            vimi_logger.error(msg)
            raise AttributeError(msg)
        if exp_shape == img.shape:
            return img, CoordsTransformer()
        redim_img: np.ndarray
        transformer: CoordsTransformer
        redim_img, transformer = redim(
            img= img,
            height= exp_shape[0],
            width= exp_shape[1]
        )
        return redim_img, transformer

    def preprocess(
        self,
        source: np.ndarray
    ) -> ncnn.Mat:
        img_proc: np.ndarray = source
        img_proc, _ = bgr2rgb(img_proc)
        img_proc = img_proc.transpose(2, 0, 1)
        img_proc = np.ascontiguousarray(img_proc)
        img_proc = img_proc.astype(np.float32) / 255.0
        return ncnn.Mat(img_proc).clone()
