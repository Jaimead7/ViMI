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
        scale: Sequence,
        offset: Sequence,
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
        scale: Sequence,
        offset: Sequence,
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
        scale: Sequence,
        offset: Sequence,
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
        scale: Sequence,
        offset: Sequence,
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
        scale: Sequence,
        offset: Sequence,
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
        scale: Sequence,
        offset: Sequence,
        expected: Sequence
    ) -> None:
        transformer: CoordsTransformer = CoordsTransformer(scale= scale, offset= offset)
        assert np.array_equal(transformer.org2xywh(xywh= generic_array), expected)
