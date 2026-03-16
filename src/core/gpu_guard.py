from __future__ import annotations

from contextlib import contextmanager
from threading import Lock
from typing import Iterator


_GPU_LOCK = Lock()


def clear_cuda_cache() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        # Keep failures non-fatal in environments without CUDA/Torch.
        pass


@contextmanager
def gpu_guard() -> Iterator[None]:
    """
    Serializes heavy GPU sections and guarantees cache cleanup on failure.
    """
    with _GPU_LOCK:
        try:
            yield
        except Exception:
            clear_cuda_cache()
            raise

