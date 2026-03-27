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
from functools import cached_property
from pathlib import Path

import yaml
from pydantic import field_validator

from ..logs import vimi_logger
from .filesystem import ModelFolder, ModelMetadata


class NCNNMetadata(ModelMetadata):
    imgsz: tuple[int, int]
    channels: int


class NCNNModelFolder(ModelFolder):
    @field_validator('path')
    def validate_structure(cls, path: Path) -> Path:
        if not path.is_dir():
            msg: str = f'"{path}" doesn\'t exists.'
            vimi_logger.error(msg)
            raise NotADirectoryError(msg)
        if not (path / 'metadata.yaml').is_file():
            msg: str = f'"{path}" doesn\'t have a metadata.yaml.'
            vimi_logger.error(msg)
            raise FileNotFoundError(msg)
        try:
            with open(path / 'metadata.yaml', 'r') as f:
                NCNNMetadata(**yaml.safe_load(f))
        except Exception:
            msg: str = f'"{path}" doesn\'t have a valid ncnn metadata.yaml.'
            vimi_logger.error(msg)
            raise ImportError(msg)
        return path

    @cached_property
    def bin_path(self) -> Path:
        return self.path / 'model.ncnn.bin'

    @cached_property
    def param_path(self) -> Path:
        return self.path / 'model.ncnn.param'

    @cached_property
    def metadata(self) -> NCNNMetadata:
        if not self.metadata_path.is_file():
            msg: str = f'"{self.metadata_path}" does\'t exists.'
            vimi_logger.error(msg)
            raise FileNotFoundError(msg)
        try:
            with open(self.metadata_path, 'r') as f:
                return NCNNMetadata(**yaml.safe_load(f))
        except Exception as e:
            msg: str = f'"{self.metadata_path}" is not a valid ncnn metadata file.'
            vimi_logger.error(msg)
            raise ImportError(msg)

    @cached_property
    def imgsz(self) -> Sequence[int]:
        return self.metadata.imgsz

    @cached_property
    def channels(self) -> int:
        return self.metadata.channels
