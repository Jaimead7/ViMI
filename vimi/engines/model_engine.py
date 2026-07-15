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
from pathlib import Path
from typing import Any, ClassVar, Optional, Protocol

import numpy as np

from ..filesystem import ModelFolder
from ..logs import vimi_logger
from ..results import Results


class ModelEngine(Protocol):
    def __init__(
        self,
        folder_path: Path,
        *args: Any,
        **kwargs: Any
    ) -> None: ...

    def __call__(
        self,
        source: np.ndarray | str | Path | list | tuple,
        *args: Any,
        **kwargs: Any
    ) -> list[Results]: ...


class EnginesReg:
    _engine_cls: ClassVar[dict[str, type[ModelEngine]]] = {}
    _engines: ClassVar[dict[Path, ModelEngine]] = {}

    @classmethod
    def register(cls, name: str) -> Callable[..., type[ModelEngine]]:
        def decorator(engine_cls: type[ModelEngine]) -> type[ModelEngine]:
            cls._engine_cls[name.upper()] = engine_cls
            return engine_cls
        return decorator

    @classmethod
    def unregister(cls, name: str) -> None:
        cls._engine_cls.pop(name.upper(), None)

    @classmethod
    def get_model(cls, model_path: Path) -> Optional[ModelEngine]:
        model: Optional[ModelEngine] = cls._engines.get(model_path, None)
        if model is not None:
            return model
        try:
            model_folder: ModelFolder = ModelFolder(path= model_path)
            engine_cls: Optional[type[ModelEngine]] = cls._engine_cls.get(
                model_folder.model_type,
                None
            )
            if engine_cls is None:
                vimi_logger.error(f'Engine type "{model_folder.model_type}" not found.')
                return None
            model = engine_cls(folder_path= model_path)
            cls._engines[model_path] = model
            vimi_logger.debug(f'Model "{model_path}" loaded.')
            return model
        except Exception as e:
            vimi_logger.error(f'Unexpected error loading model "{model_path}". {e}')
        return None

    @classmethod
    def models_list(cls) -> list[Path]:
        return sorted(cls._engines.keys())

    @classmethod
    def clear_model(cls, model_path: Path) -> None:
        try:
            del cls._engines[model_path]
            vimi_logger.debug(f'Model "{model_path}" unloaded.')
        except KeyError:
            pass

    @classmethod
    def clear_models(cls) -> None:
        cls._engines.clear()
