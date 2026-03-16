from __future__ import annotations

import gc
from contextlib import contextmanager
from threading import Lock
from typing import Callable, Generic, Iterator, Optional, TypeVar


T = TypeVar("T")


class ModelSwapper(Generic[T]):
    """
    Lazy model loader that can unload heavy runtimes after each call.
    """

    def __init__(self, loader: Callable[[], T], keep_loaded: bool = False) -> None:
        self._loader = loader
        self._keep_loaded = keep_loaded
        self._model: Optional[T] = None
        self._lock = Lock()

    def _load(self) -> T:
        if self._model is None:
            self._model = self._loader()
        return self._model

    def _unload(self) -> None:
        self._model = None
        gc.collect()

    @contextmanager
    def session(self) -> Iterator[T]:
        with self._lock:
            model = self._load()
            try:
                yield model
            finally:
                if not self._keep_loaded:
                    self._unload()

