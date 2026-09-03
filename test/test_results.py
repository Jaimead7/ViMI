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
from pytest import mark, raises

from vimi.results import SpeedDict, _parse_np_str


class TestParseArray:
    @mark.parametrize(
        'input, expected',
        [
            (np.array([1.23456, 2.34567]), '[1.2346, 2.3457]'),
            (np.array([-1.23456, -0.00012345]), '[-1.2346, -0.0001]'),
            (np.array([]), '[]')
        ]
    )
    def test_parse_np_str_1d(self, input: np.ndarray, expected: str) -> None:
        result: str = _parse_np_str(input)
        assert result == expected

    @mark.parametrize(
        'input, expected',
        [
            (
                np.array([[1.23456, 2.34567], [3.45678, 4.56789]]),
                '[[1.2346, 2.3457],\n [3.4568, 4.5679]]'
            ),
            (
                np.array([[0.000123, 0.000456], [0.000789, 0.000012]]),
                '[[0.0001, 0.0005],\n [0.0008, 0.    ]]'
            )
        ]
    )
    def test_parse_np_str_2d(self, input: np.ndarray, expected: str) -> None:
        result: str = _parse_np_str(input)
        assert result == expected

    @mark.parametrize(
        'input, expected',
        [
            (
                np.array([[[1.2345, 2.3456], [3.4567, 4.5678]]]),
                '[[[1.2345, 2.3456],\n  [3.4567, 4.5678]]]'
            ),
            (
                np.array([[[1.11, 2.2], [3.3, 4.4]], [[5.5, 6.6], [7.7, 8.8]]]),
                '[[[1.11, 2.2 ],\n  [3.3 , 4.4 ]],\n\n [[5.5 , 6.6 ],\n  [7.7 , 8.8 ]]]'
            )
        ]
    )
    def test_parse_np_str_3d(self, input: np.ndarray, expected: str) -> None:
        result: str = _parse_np_str(input)
        assert result == expected

    @mark.parametrize(
        'input, expected',
        [
            (
                np.array([0.00012345, 0.00000001, 1.0]),
                '[0.0001, 0.    , 1.    ]'
            ),
            (
                np.array([0.00005]),
                '[0.0001]'
            ),
            (
                np.array([0.00004]),
                '[0.]'
            ),
            (
                np.array([-0.00005]),
                '[-0.0001]'
            ),
            (
                np.array([-0.00004]),
                '[-0.]'
            ),
            (
                np.array([0.0004999]),
                '[0.0005]'
            )
        ]
    )
    def test_parse_np_str_small(self, input: np.ndarray, expected: str) -> None:
        result: str = _parse_np_str(input)
        assert result == expected

    @mark.parametrize(
        'input, expected',
        [
            (
                np.array([1.23456789], dtype=np.float32),
                '[1.2346]'
            ),
            (
                np.array([1.23456789], dtype=np.float64),
                '[1.2346]'
            ),
            (
                np.array([1, 2, 3], dtype=np.int32),
                '[1, 2, 3]'
            ),
            (
                np.array([1, 2, 3], dtype=np.int64),
                '[1, 2, 3]'
            )
        ]
    )
    def test_parse_np_str_types(self, input: np.ndarray, expected: str) -> None:
        result: str = _parse_np_str(input)
        assert result == expected


class TestSpeedDict:
    def test_key_access(self) -> None:
        res = SpeedDict()
        assert res['preprocess'] == 0
        assert res['inference'] == 0
        assert res['postprocess'] == 0

    def test_key_access_error(self) -> None:
        res = SpeedDict()
        with raises(KeyError):
            res['no_key']

    @mark.parametrize(
        'pre, inf, post',
        [
            (-1.107, 2.239, 3.476),
            (1.17, -3.2314, 4.1779),
            (6.124, 0.2147, -3.142),
        ]
    )
    def test_constructor_error(self, pre: float, inf: float, post: float) -> None:
        with raises(ValueError):
            SpeedDict(preprocess= pre, inference= inf, postprocess= post)

    @mark.parametrize(
        'pre, inf, post',
        [
            (-1.107, 2.239, 3.476),
            (1.17, -3.2314, 4.1779),
            (6.124, 0.2147, -3.142),
            (6.124, 0.2147, -3.142),
        ]
    )
    def test_assign_error(self, pre: float, inf: float, post: float) -> None:
        res = SpeedDict()
        with raises(ValueError):
            res.preprocess = pre
            res.inference = inf
            res.postprocess = post

    @mark.parametrize(
        'pre, inf, post',
        [
            (-1.107, 2.239, 3.476),
            (1.17, -3.2314, 4.1779),
            (6.124, 0.2147, -3.142),
        ]
    )
    def test_key_assign_error(self, pre: float, inf: float, post: float) -> None:
        res = SpeedDict()
        with raises(ValueError):
            res['preprocess'] = pre
            res['inference'] = inf
            res['postprocess'] = post


class TestResultDataWrapper:
    ... #TODO


class TestBoxes:
    ... #TODO


class TestMasks:
    ... #TODO


class TestProbs:
    ... #TODO


class TestKeypoints:
    ... #TODO


class TestOBB:
    ... #TODO


class TestResults:
    ... #TODO
