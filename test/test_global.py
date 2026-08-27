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


from pathlib import Path

import cv2
import numpy as np
from pytest import fixture

from vimi import EnginesReg, ModelEngine, Results

MODELS_PATH: Path = Path(__file__).absolute().parent / 'models'
IMGS_PATH: Path = Path(__file__).absolute().parent / 'imgs'


@fixture
def class_model() -> ModelEngine:
    model_path: Path = MODELS_PATH / 'ClassModelNCNN'
    model: ModelEngine | None = EnginesReg.get_model(model_path)
    if model is None:
        raise SystemError('Error creating classification model.')
    return model

@fixture
def detect_model() -> ModelEngine:
    model_path: Path = MODELS_PATH / 'DetectModelNCNN'
    model: ModelEngine | None = EnginesReg.get_model(model_path)
    if model is None:
        raise SystemError('Error creating detection model.')
    return model

@fixture
def obb_model() -> ModelEngine:
    model_path: Path = MODELS_PATH / 'OBBModelNCNN'
    model: ModelEngine | None = EnginesReg.get_model(model_path)
    if model is None:
        raise SystemError('Error creating obb model.')
    return model

@fixture
def imagenet_imgs() -> dict[Path, dict]:
    imgs_path: Path = IMGS_PATH / 'imagenet'
    imgs: dict[Path, dict] = {
        imgs_path / 'jay.jpg': {
            'result': 'jay',
        },
        imgs_path / 'leopard.jpg': {
            'result': 'leopard',
        }
    }
    for img_path in imgs.keys():
        img: np.ndarray | None = cv2.imread(img_path)
        if img is None:
            continue
        imgs[img_path]['img'] = img
    return imgs

@fixture
def coco_imgs() -> dict[Path, dict]:
    imgs_path: Path = IMGS_PATH / 'coco'
    imgs: dict[Path, dict] = {
        imgs_path / 'bike.jpg': {
            'result': ['bicycle', 'horse'],
        },
        imgs_path / 'planes.jpg': {
            'result': ['airplane', 'airplane'],
        }
    }
    for img_path in imgs.keys():
        img: np.ndarray | None = cv2.imread(img_path)
        if img is None:
            continue
        imgs[img_path]['img'] = img
    return imgs

@fixture
def dota_imgs() -> dict[Path, dict]:
    imgs_path: Path = IMGS_PATH / 'dota'
    imgs: dict[Path, dict] = {
        imgs_path / 'airport.png': {
            'result': ['plane', 'plane', 'plane'],
        },
        imgs_path / 'roundabout.png': {
            'result': ['roundabout', 'small vehicle'],
        }
    }
    for img_path in imgs.keys():
        img: np.ndarray | None = cv2.imread(img_path)
        if img is None:
            continue
        imgs[img_path]['img'] = img
    return imgs


class TestGlobal:
    def test_class_model(
        self,
        class_model: ModelEngine,
        imagenet_imgs: dict[Path, dict]
    ) -> None:
        imgs: list[np.ndarray] = [img['img'] for img in imagenet_imgs.values()]
        results: list[Results] = class_model(imgs)
        for result, img_prop in zip(results, imagenet_imgs.values()):
            if result.probs is None:
                assert False
            assert result.names[result.probs.top1] == img_prop['result']

    def test_detect_model(
        self,
        detect_model: ModelEngine,
        coco_imgs: dict[Path, dict]
    ) -> None:
        imgs: list[np.ndarray] = [img['img'] for img in coco_imgs.values()]
        results: list[Results] = detect_model(imgs)
        for result, img_prop in zip(results, coco_imgs.values()):
            if result.boxes is None:
                assert False
            res_names: list[str] = [result.names[i] for i in result.boxes.cls]
            assert res_names == img_prop['result']

    def test_obb_model(
        self,
        obb_model: ModelEngine,
        dota_imgs: dict[Path, dict]
    ) -> None:
        imgs: list[np.ndarray] = [img['img'] for img in dota_imgs.values()]
        results: list[Results] = obb_model(imgs)
        for result, img_prop in zip(results, dota_imgs.values()):
            if result.obb is None:
                assert False
            res_names: list[str] = [result.names[i] for i in result.obb.cls]
            assert res_names == img_prop['result']
