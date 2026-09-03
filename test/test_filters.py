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


from collections.abc import Sequence
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
from pytest import fixture, mark, raises

from vimi.filters import (CoordsTransformer, ImageFilter, ImageFiltersReg,
                          bgr2gray, bgr2rgb, cut, gray2bgr, redim, resize,
                          rgb2bgr)


@fixture
def generic_array() -> np.ndarray:
    return np.array(
        [
            [0, 1, 2, 3],
            [4, 5, 6, 7],
            [8, 9, 10, 11]
        ]
    )

@fixture
def generic_filter() -> ImageFilter:
    def filter(img: np.ndarray) -> tuple[np.ndarray, CoordsTransformer]:
        return np.zeros((0, 0)), CoordsTransformer()
    return filter

@fixture
def generic_gray_img() -> np.ndarray:
    return np.array(
        [
            [0, 255, 0, 255],
            [255, 0, 255, 0],
            [0, 255, 0, 255],
            [255, 0, 255, 0]
        ]
    ).astype(np.uint8)

@fixture
def generic_bgr_img() -> np.ndarray:
    return np.array(
        [
            [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 0, 0]],
            [[0, 255, 0], [0, 0, 255], [255, 0, 0], [0, 255, 0]],
            [[0, 0, 255], [255, 0, 0], [0, 255, 0], [0, 0, 255]],
            [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 0, 0]]
        ]
    ).astype(np.uint8)

@fixture
def generic_rgb_img() -> np.ndarray:
    return np.array(
        [
            [[0, 0, 255], [0, 255, 0], [255, 0, 0], [0, 0, 255]],
            [[0, 255, 0], [255, 0, 0], [0, 0, 255], [0, 255, 0]],
            [[255, 0, 0], [0, 0, 255], [0, 255, 0], [255, 0, 0]],
            [[0, 0, 255], [0, 255, 0], [255, 0, 0], [0, 0, 255]]
        ]
    ).astype(np.uint8)


class TestCoordsTransformer:
    @mark.parametrize(
        'scale, offset, expected',
        [
            ((1, 1), (0, 0), (1, 1, 0, 0)),
            ((1, 1, 1), (0, 0, 0), (1, 1, 0, 0)),
            ((1.5, 1.1), (1.5, 1.1), (1.5, 1.1, 1, 1)),
            ((1, 1), (0, -2), (1, 1, 0, -2)),
            (('1.5', 1.1), (1, '1'), (1.5, 1.1, 1, 1)),
        ]
    )
    def test_constructor(
        self,
        scale: tuple[float, float],
        offset: tuple[int, int],
        expected: Sequence
    ) -> None:
        transformer: CoordsTransformer = CoordsTransformer(scale= scale, offset= offset)
        assert expected[0] == transformer.scaleX
        assert expected[1] == transformer.scaleY
        assert expected[2] == transformer.offsetX
        assert expected[3] == transformer.offsetY

    @mark.parametrize(
        'scale, offset, error',
        [
            ((1, 1), (0,), ValueError),
            ((1,), (0, 0), ValueError),
            ((1, 'foo'), (0, 0), TypeError),
            ((1, 1), (0, 'foo'), TypeError),
            ((0, 1), (0, 0), ValueError),
        ]
    )
    def test_constructor_errors(
        self,
        scale: tuple[float, float],
        offset: tuple[int, int],
        error: type[Exception]
    ) -> None:
        with raises(error):
            _: CoordsTransformer = CoordsTransformer(scale= scale, offset= offset)

    @mark.parametrize(
        'scale, offset, expected',
        [
            ((1, 1), (0, 0), np.array([[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]])),
            ((1, 1), (1, 2), np.array([[1, 3, 3, 5], [5, 7, 7, 9], [9, 11, 11, 13]])),
            ((1, 1), (-1, -2), np.array([[-1, -1, 1, 1], [3, 3, 5, 5], [7, 7, 9, 9]])),
            ((2, 4), (0, 0), np.array([[0, 0.25, 1, 0.75], [2, 1.25, 3, 1.75], [4, 2.25, 5, 2.75]])),
            ((0.5, 0.25), (0, 0), np.array([[0, 4, 4, 12], [8, 20, 12, 28], [16, 36, 20, 44]])),
        ]
    )
    def test_res2org(
        self,
        generic_array: np.ndarray,
        scale: tuple[float, float],
        offset: tuple[int, int],
        expected: Sequence
    ) -> None:
        transformer: CoordsTransformer = CoordsTransformer(scale= scale, offset= offset)
        assert np.array_equal(transformer.res2org(values= generic_array), expected)

    @mark.parametrize(
        'scale, offset, expected',
        [
            ((1, 1), (0, 0), np.array([[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]])),
            ((1, 1), (1, 2), np.array([[-1, -1, 1, 1], [3, 3, 5, 5], [7, 7, 9, 9]])),
            ((1, 1), (-1, -2), np.array([[1, 3, 3, 5], [5, 7, 7, 9], [9, 11, 11, 13]])),
            ((2, 4), (0, 0), np.array([[0, 4, 4, 12], [8, 20, 12, 28], [16, 36, 20, 44]])),
            ((0.5, 0.25), (0, 0), np.array([[0, 0.25, 1, 0.75], [2, 1.25, 3, 1.75], [4, 2.25, 5, 2.75]])),
        ]
    )
    def test_org2res(
        self,
        generic_array: np.ndarray,
        scale: tuple[float, float],
        offset: tuple[int, int],
        expected: Sequence
    ) -> None:
        transformer: CoordsTransformer = CoordsTransformer(scale= scale, offset= offset)
        assert np.array_equal(transformer.org2res(values= generic_array), expected)

    @mark.parametrize(
        'scale, offset, expected',
        [
            ((1, 1), (0, 0), np.array([[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]])),
            ((1, 1), (1, 2), np.array([[1, 3, 2, 3], [5, 7, 6, 7], [9, 11, 10, 11]])),
            ((1, 1), (-1, -2), np.array([[-1, -1, 2, 3], [3, 3, 6, 7], [7, 7, 10, 11]])),
            ((2, 4), (0, 0), np.array([[0, 0.25, 1, 0.75], [2, 1.25, 3, 1.75], [4, 2.25, 5, 2.75]])),
            ((0.5, 0.25), (0, 0), np.array([[0, 4, 4, 12], [8, 20, 12, 28], [16, 36, 20, 44]])),
        ]
    )
    def test_xywh2org(
        self,
        generic_array: np.ndarray,
        scale: tuple[float, float],
        offset: tuple[int, int],
        expected: Sequence
    ) -> None:
        transformer: CoordsTransformer = CoordsTransformer(scale= scale, offset= offset)
        assert np.array_equal(transformer.xywh2org(xywh= generic_array), expected)

    @mark.parametrize(
        'scale, offset, expected',
        [
            ((1, 1), (0, 0), np.array([[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]])),
            ((1, 1), (1, 2), np.array([[-1, -1, 2, 3], [3, 3, 6, 7], [7, 7, 10, 11]])),
            ((1, 1), (-1, -2), np.array([[1, 3, 2, 3], [5, 7, 6, 7], [9, 11, 10, 11]])),
            ((2, 4), (0, 0), np.array([[0, 4, 4, 12], [8, 20, 12, 28], [16, 36, 20, 44]])),
            ((0.5, 0.25), (0, 0), np.array([[0, 0.25, 1, 0.75], [2, 1.25, 3, 1.75], [4, 2.25, 5, 2.75]])),
        ]
    )
    def test_org2xywh(
        self,
        generic_array: np.ndarray,
        scale: tuple[float, float],
        offset: tuple[int, int],
        expected: Sequence
    ) -> None:
        transformer: CoordsTransformer = CoordsTransformer(scale= scale, offset= offset)
        assert np.array_equal(transformer.org2xywh(xywh= generic_array), expected)


class TestImageFiltersReg:
    def test_no_instantiable(self) -> None:
        with raises(RuntimeError):
            ImageFiltersReg()

    def test_register_unregister(self, generic_filter: ImageFilter) -> None:
        filters_backup: dict[str, ImageFilter] = ImageFiltersReg._filters.copy()
        try:
            name = 'test'
            decorator: Callable[[ImageFilter], ImageFilter] = ImageFiltersReg.register(name)
            decorator(generic_filter)
            assert ImageFiltersReg.get(name) == generic_filter
            ImageFiltersReg.unregister(name)
            assert ImageFiltersReg.get(name) == ImageFiltersReg.no_filter
        finally:
            ImageFiltersReg._filters = filters_backup

    def test_clear(self) -> None:
        filters_backup: dict[str, ImageFilter] = ImageFiltersReg._filters.copy()
        try:
            ImageFiltersReg.clear()
            assert ImageFiltersReg._filters == dict()
        finally:
            ImageFiltersReg._filters = filters_backup


class TestResize:
    @mark.parametrize(
            'w, h, img_expected, trans_exp',
            [
                (5, 5, np.array([[0, 178, 127, 76, 255], [179, 107, 127, 148, 76], [128, 127, 128, 127, 128], [76, 148, 127, 107 ,179], [255, 76, 127, 178, 0]], dtype= np.uint8), CoordsTransformer(scale= (1.25, 1.25))),
                (2, 2, np.array([[128, 128], [128, 128]], dtype= np.uint8), CoordsTransformer(scale= (0.5, 0.5))),
            ]
        )
    def test_resize_gray(
        self,
        w: int,
        h: int,
        img_expected: np.ndarray,
        trans_exp: CoordsTransformer,
        generic_gray_img: np.ndarray
    ) -> None:
        res: np.ndarray
        trans: CoordsTransformer
        res, trans = resize(img= generic_gray_img, width= w, height= h)
        assert np.array_equal(res, img_expected)
        assert trans == trans_exp

    @mark.parametrize(
            'w, h, img_expected, trans_exp',
            [
                (5, 5, np.array([[[255, 0, 0], [76, 178, 0], [0, 127, 127], [76, 0, 178], [255, 0, 0]], [[76, 179, 0], [23, 107, 125], [89, 38, 127], [148, 54, 54], [76, 179, 0]], [[0, 128, 128], [89, 38, 127], [128, 64, 64], [89, 127, 38], [0, 128, 128]], [[76, 0, 179], [148, 54, 54], [89, 127, 38], [23, 125, 107], [76, 0, 179]], [[255, 0, 0], [76, 178, 0], [0, 127, 127], [76, 0, 178], [255, 0, 0]]], dtype= np.uint8), CoordsTransformer(scale= (1.25, 1.25))),
                (2, 2, np.array([[[64, 128, 64], [128, 64, 64]], [[128, 64, 64], [64, 64, 128]]], dtype= np.uint8), CoordsTransformer(scale= (0.5, 0.5))),
            ]
        )
    def test_resize_bgr(
        self,
        w: int,
        h: int,
        img_expected: np.ndarray,
        trans_exp: CoordsTransformer,
        generic_bgr_img: np.ndarray
    ) -> None:
        res: np.ndarray
        trans: CoordsTransformer
        res, trans = resize(img= generic_bgr_img, width= w, height= h)
        assert np.array_equal(res, img_expected)
        assert trans == trans_exp


class TestRedim:
    @mark.parametrize(
            'w, h, img_expected, trans_exp',
            [
                (4, 5, np.array([[0, 255, 0, 255], [255, 0, 255, 0], [0, 255, 0, 255], [255, 0, 255, 0], [114, 114, 114, 114]], dtype= np.uint8), CoordsTransformer()),
                (2, 2, np.array([[128, 128], [128, 128]], dtype= np.uint8), CoordsTransformer(scale= (0.5, 0.5))),
                (3, 2, np.array([[128, 128, 114], [128, 128, 114]], dtype= np.uint8), CoordsTransformer(scale= (0.5, 0.5))),
            ]
        )
    def test_redim_gray(
        self,
        w: int,
        h: int,
        img_expected: np.ndarray,
        trans_exp: CoordsTransformer,
        generic_gray_img: np.ndarray
    ) -> None:
        res: np.ndarray
        trans: CoordsTransformer
        res, trans = redim(img= generic_gray_img, width= w, height= h)
        assert np.array_equal(res, img_expected)
        assert trans == trans_exp

    @mark.parametrize(
            'w, h, img_expected, trans_exp',
            [
                (4, 5, np.array([[[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 0, 0]], [[0, 255, 0], [0, 0, 255], [255, 0, 0], [0, 255, 0]], [[0, 0, 255], [255, 0, 0], [0, 255, 0], [0, 0, 255]], [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 0, 0]], [[114, 114, 114], [114, 114, 114], [114, 114, 114], [114, 114, 114]]], dtype= np.uint8), CoordsTransformer()),
                (2, 2, np.array([[[64, 128, 64], [128, 64, 64]], [[128, 64, 64], [64, 64, 128]]], dtype= np.uint8), CoordsTransformer(scale= (0.5, 0.5))),
                (3, 2, np.array([[[64, 128, 64], [128, 64, 64], [114, 114, 114]], [[128, 64, 64], [64, 64, 128], [114, 114, 114]]], dtype= np.uint8), CoordsTransformer(scale= (0.5, 0.5))),
            ]
        )
    def test_redim_bgr(
        self,
        w: int,
        h: int,
        img_expected: np.ndarray,
        trans_exp: CoordsTransformer,
        generic_bgr_img: np.ndarray
    ) -> None:
        res: np.ndarray
        trans: CoordsTransformer
        res, trans = redim(img= generic_bgr_img, width= w, height= h)
        assert np.array_equal(res, img_expected)
        assert trans == trans_exp


class TestCut:
    @mark.parametrize(
        'p0, w, h, img_expected, trans_exp',
        [
            ((1, 1), 2, 2, np.array([[0, 255], [255, 0]], dtype= np.uint8), CoordsTransformer(offset= (1, 1))),
            ((1, 1), 3, 2, np.array([[0, 255, 0], [255, 0, 255]], dtype= np.uint8), CoordsTransformer(offset= (1, 1))),
            ((1, 1), 10, 10, np.array([[0, 255, 0], [255, 0, 255], [0, 255, 0]], dtype= np.uint8), CoordsTransformer(offset= (1, 1))),
            ((10, 10), 10, 10, np.empty((0, 0), dtype= np.uint8), CoordsTransformer(offset= (4, 4))),
            ((-5, -5), 2, 2, np.empty((0, 0), dtype= np.uint8), CoordsTransformer()),
        ]
    )
    def test_cut_gray(
        self,
        p0: tuple[int, int],
        w: int,
        h: int,
        img_expected: np.ndarray,
        trans_exp: CoordsTransformer,
        generic_gray_img: np.ndarray
    ) -> None:
        res: np.ndarray
        trans: CoordsTransformer
        res, trans = cut(img= generic_gray_img, p0= p0, width= w, height= h)
        assert np.array_equal(res, img_expected)
        assert trans == trans_exp

    @mark.parametrize(
        'p0, w, h, img_expected, trans_exp',
        [
            ((1, 1), 2, 2, np.array([[[0, 0, 255], [255, 0, 0]], [[255, 0, 0], [0, 255, 0]]], dtype= np.uint8), CoordsTransformer(offset= (1, 1))),
            ((1, 1), 3, 2, np.array([[[0, 0, 255], [255, 0, 0], [0, 255, 0]], [[255, 0, 0], [0, 255, 0], [0, 0, 255]]], dtype= np.uint8), CoordsTransformer(offset= (1, 1))),
            ((1, 1), 10, 10, np.array([[[0, 0, 255], [255, 0, 0], [0, 255, 0]], [[255, 0, 0], [0, 255, 0], [0, 0, 255]], [[0, 255, 0], [0, 0, 255], [255, 0, 0]]], dtype= np.uint8), CoordsTransformer(offset= (1, 1))),
            ((10, 10), 10, 10, np.empty((0, 0, 3), dtype= np.uint8), CoordsTransformer(offset= (4, 4))),
            ((-5, -5), 2, 2, np.empty((0, 0, 3), dtype= np.uint8), CoordsTransformer()),
        ]
    )
    def test_cut_bgr(
        self,
        p0: tuple[int, int],
        w: int,
        h: int,
        img_expected: np.ndarray,
        trans_exp: CoordsTransformer,
        generic_bgr_img: np.ndarray
    ) -> None:
        res: np.ndarray
        trans: CoordsTransformer
        res, trans = cut(img= generic_bgr_img, p0= p0, width= w, height= h)
        assert np.array_equal(res, img_expected)
        assert trans == trans_exp


class TestBGR2GRAY:
    def test_bgr2gray(
        self,
        generic_gray_img: np.ndarray,
        generic_bgr_img: np.ndarray
    ) -> None:
        res: np.ndarray
        trans: CoordsTransformer
        res, trans = bgr2gray(img= generic_gray_img)
        assert np.array_equal(res, generic_gray_img)
        assert trans == CoordsTransformer()
        res, trans = bgr2gray(img= generic_bgr_img)
        expected: np.ndarray = np.array(
            [
                [29, 150, 76, 29],
                [150, 76, 29, 150],
                [76, 29, 150, 76],
                [29, 150, 76, 29]
            ],
            dtype= np.uint8
        )
        assert np.array_equal(res, expected)
        assert trans == CoordsTransformer()


class TestGRAY2BGR:
    def test_gray2bgr(
        self,
        generic_gray_img: np.ndarray,
        generic_bgr_img: np.ndarray
    ) -> None:
        res: np.ndarray
        trans: CoordsTransformer
        res, trans = gray2bgr(img= generic_gray_img)
        expected: np.ndarray = np.array(
            [
                [[0, 0, 0], [255, 255, 255], [0, 0, 0], [255, 255, 255]],
                [[255, 255, 255], [0, 0, 0], [255, 255, 255], [0, 0, 0]],
                [[0, 0, 0], [255, 255, 255], [0, 0, 0], [255, 255, 255]],
                [[255, 255, 255], [0, 0, 0], [255, 255, 255], [0, 0, 0]]
            ],
            dtype= np.uint8
        )
        assert np.array_equal(res, expected)
        assert trans == CoordsTransformer()
        res, trans = gray2bgr(img= generic_bgr_img)
        assert np.array_equal(res, generic_bgr_img)
        assert trans == CoordsTransformer()


class TestBGR2RGB:
    def test_bgr2rgb(
        self,
        generic_gray_img: np.ndarray,
        generic_bgr_img: np.ndarray,
        generic_rgb_img: np.ndarray
    ) -> None:
        res: np.ndarray
        trans: CoordsTransformer
        res, trans = gray2bgr(img= generic_gray_img)
        expected: np.ndarray = np.array(
            [
                [[0, 0, 0], [255, 255, 255], [0, 0, 0], [255, 255, 255]],
                [[255, 255, 255], [0, 0, 0], [255, 255, 255], [0, 0, 0]],
                [[0, 0, 0], [255, 255, 255], [0, 0, 0], [255, 255, 255]],
                [[255, 255, 255], [0, 0, 0], [255, 255, 255], [0, 0, 0]]
            ],
            dtype= np.uint8
        )
        assert np.array_equal(res, expected)
        res, trans = bgr2rgb(img= generic_bgr_img)
        assert np.array_equal(res, generic_rgb_img)
        assert trans == CoordsTransformer()


class TestRGB2BGR:
    def test_rgb2bgr(
        self,
        generic_gray_img: np.ndarray,
        generic_bgr_img: np.ndarray,
        generic_rgb_img: np.ndarray
    ) -> None:
        res: np.ndarray
        trans: CoordsTransformer
        res, trans = rgb2bgr(img= generic_gray_img)
        expected: np.ndarray = np.array(
            [
                [[0, 0, 0], [255, 255, 255], [0, 0, 0], [255, 255, 255]],
                [[255, 255, 255], [0, 0, 0], [255, 255, 255], [0, 0, 0]],
                [[0, 0, 0], [255, 255, 255], [0, 0, 0], [255, 255, 255]],
                [[255, 255, 255], [0, 0, 0], [255, 255, 255], [0, 0, 0]]
            ],
            dtype= np.uint8
        )
        assert np.array_equal(res, expected)
        res, trans = rgb2bgr(img= generic_rgb_img)
        assert np.array_equal(res, generic_bgr_img)
        assert trans == CoordsTransformer()
