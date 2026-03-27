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
from datetime import datetime
from functools import cached_property
from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import BaseModel, ConfigDict, field_validator

from ..filters import ImageFilter, ImageFiltersReg
from ..logs import vimi_logger


class ModelMetadata(BaseModel):
    date: datetime
    model_type: str
    filters: Optional[list[str]] = None
    filters_attrs: Optional[dict[str, dict[str, Any]]] = None
    task: str
    names: dict[int, str]

    model_config = ConfigDict(
        extra= 'allow',
        frozen= True
    )


class ModelFolder(BaseModel):
    path: Path

    model_config = ConfigDict(
        frozen= True
    )

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
                ModelMetadata(**yaml.safe_load(f))
        except Exception:
            msg: str = f'"{path}" doesn\'t have a valid metadata.yaml.'
            vimi_logger.error(msg)
            raise ImportError(msg)
        return path

    @cached_property
    def name(self) -> str:
        return self.path.name

    @cached_property
    def metadata_path(self) -> Path:
        return self.path / 'metadata.yaml'

    @cached_property
    def metadata(self) -> ModelMetadata:
        if not self.metadata_path.is_file():
            msg: str = f'"{self.metadata_path}" does\'t exists.'
            vimi_logger.error(msg)
            raise FileNotFoundError(msg)
        try:
            with open(self.metadata_path, 'r') as f:
                return ModelMetadata(**yaml.safe_load(f))
        except Exception as e:
            msg: str = f'"{self.metadata_path}" is not a valid ncnn metadata file.'
            vimi_logger.error(msg)
            raise ImportError(msg)

    @cached_property
    def date(self) -> datetime:
        return self.metadata.date

    @cached_property
    def model_type(self) -> str:
        return self.metadata.model_type.upper()

    @cached_property
    def filters(self) -> Sequence[tuple[ImageFilter, dict[str, Any]]]:
        filters_names: Optional[list[str]] = self.metadata.filters
        filters_attrs: Optional[dict[str, dict[str, Any]]] = self.metadata.filters_attrs
        if filters_names is None or filters_attrs is None:
            return tuple()
        return tuple(
            (
                ImageFiltersReg.get(filter_name),
                filters_attrs[filter_name]
            )
            for filter_name in filters_names
        )

    @cached_property
    def task(self) -> str:
        return self.metadata.task.upper()

    @cached_property
    def names(self) -> dict[int, str]:
        return self.metadata.names
