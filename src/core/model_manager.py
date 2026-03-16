"""
CA_Monk v3 — Singleton Model Manager
======================================
Military-grade VRAM lifecycle controller.

Design contract:
  1. Only ONE heavy model lives on GPU at a time.
  2. Before loading a new model, the previous one is evicted → CPU / GC.
  3. If VRAM is exhausted, execution falls back to CPU via ONNX / OpenVINO.
  4. Pipeline stages are strictly sequential:
       Ingest → Biometrics → Forensics → Reconstruction → Reporting
"""

from __future__ import annotations

import gc
import logging
import os
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, Iterator, Optional

logger = logging.getLogger("ca_monk.model_manager")


# ---------------------------------------------------------------------------
# Hardware helpers
# ---------------------------------------------------------------------------

def _cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def _cuda_mem_free_mb() -> float:
    try:
        import torch
        if torch.cuda.is_available():
            free, _ = torch.cuda.mem_get_info()
            return free / (1024 ** 2)
    except Exception:
        pass
    return 0.0


def _cuda_empty_cache() -> None:
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except Exception:
        pass


def _force_gc() -> None:
    """Aggressive garbage collection + CUDA cache purge."""
    gc.collect()
    gc.collect()
    _cuda_empty_cache()


# ---------------------------------------------------------------------------
# Registry entry
# ---------------------------------------------------------------------------

class DeviceTarget(str, Enum):
    GPU = "gpu"
    CPU = "cpu"
    AUTO = "auto"          # GPU if VRAM sufficient, else CPU


@dataclass
class ModelSpec:
    """Blueprint for a lazy-loadable model."""
    key: str
    loader: Callable[[], Any]                 # Returns the loaded model / service
    vram_mb: float = 0.0                      # Approximate GPU footprint
    device: DeviceTarget = DeviceTarget.AUTO
    keep_loaded: bool = False                 # True = never auto-evict
    unloader: Optional[Callable[[Any], None]] = None  # Custom teardown


@dataclass
class _LoadedEntry:
    spec: ModelSpec
    instance: Any
    loaded_at: float = field(default_factory=time.time)
    device_used: str = "cpu"


# ---------------------------------------------------------------------------
# Singleton Model Manager
# ---------------------------------------------------------------------------

class ModelManager:
    """
    Thread-safe singleton that guarantees at most ONE heavy model on GPU.

    Usage::

        mgr = ModelManager.get()
        mgr.register("biometrics", ModelSpec(key="biometrics", loader=..., vram_mb=800))
        with mgr.acquire("biometrics") as svc:
            result = svc.analyze(...)
        # model is auto-evicted after context exit (unless keep_loaded=True)
    """

    _instance: Optional["ModelManager"] = None
    _init_lock = threading.Lock()

    def __init__(self) -> None:
        self._registry: Dict[str, ModelSpec] = {}
        self._loaded: Dict[str, _LoadedEntry] = {}
        self._gpu_lock = threading.Lock()
        self._vram_budget_mb = float(os.getenv("CA_MONK_VRAM_BUDGET_MB", "5500"))
        self._current_gpu_key: Optional[str] = None

    # --- Singleton accessor ---------------------------------------------------

    @classmethod
    def get(cls) -> "ModelManager":
        if cls._instance is None:
            with cls._init_lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Tear down for tests / full restart."""
        if cls._instance is not None:
            cls._instance.unload_all()
            cls._instance = None

    # --- Registration ---------------------------------------------------------

    def register(self, key: str, spec: ModelSpec) -> None:
        self._registry[key] = spec

    # --- Acquire / Release ----------------------------------------------------

    @contextmanager
    def acquire(self, key: str) -> Iterator[Any]:
        """
        Load *key*, yield the live instance, then evict unless keep_loaded.
        Only one GPU model lives at a time.
        """
        entry = self._load(key)
        try:
            yield entry.instance
        finally:
            if not entry.spec.keep_loaded:
                self._unload(key)

    def load_model(self, key: str) -> Any:
        """Eagerly load and return the model instance."""
        return self._load(key).instance

    def unload_model(self, key: str) -> None:
        self._unload(key)

    def unload_all(self) -> None:
        for k in list(self._loaded.keys()):
            self._unload(k)

    def garbage_collect(self) -> None:
        _force_gc()
        logger.debug(
            "GC complete — CUDA free: %.0f MB",
            _cuda_mem_free_mb(),
        )

    # --- Internal -------------------------------------------------------------

    def _load(self, key: str) -> _LoadedEntry:
        if key in self._loaded:
            return self._loaded[key]

        spec = self._registry.get(key)
        if spec is None:
            raise KeyError(f"Model '{key}' is not registered in ModelManager.")

        with self._gpu_lock:
            # Evict the current GPU tenant if we need space
            wants_gpu = spec.device in (DeviceTarget.GPU, DeviceTarget.AUTO) and _cuda_available()
            if wants_gpu and self._current_gpu_key and self._current_gpu_key != key:
                self._evict_gpu_tenant()

            _force_gc()

            # Decide actual device
            if wants_gpu and spec.vram_mb <= self._vram_budget_mb:
                if _cuda_mem_free_mb() >= spec.vram_mb * 1.15:
                    device_used = "gpu"
                else:
                    logger.warning(
                        "Insufficient VRAM for '%s' (need %.0f MB, have %.0f MB). Falling back to CPU.",
                        key, spec.vram_mb, _cuda_mem_free_mb(),
                    )
                    device_used = "cpu"
            elif spec.device == DeviceTarget.GPU and _cuda_available():
                device_used = "gpu"
            else:
                device_used = "cpu"

            logger.info("Loading model '%s' on %s …", key, device_used.upper())
            t0 = time.time()

            os.environ["CA_MONK_DEVICE"] = device_used
            instance = spec.loader()
            elapsed = time.time() - t0
            logger.info("Loaded '%s' in %.1f s", key, elapsed)

            entry = _LoadedEntry(spec=spec, instance=instance, device_used=device_used)
            self._loaded[key] = entry
            if device_used == "gpu":
                self._current_gpu_key = key
            return entry

    def _unload(self, key: str) -> None:
        entry = self._loaded.pop(key, None)
        if entry is None:
            return
        logger.info("Unloading model '%s'", key)
        if entry.spec.unloader:
            try:
                entry.spec.unloader(entry.instance)
            except Exception as exc:
                logger.warning("Custom unloader for '%s' raised: %s", key, exc)
        del entry.instance
        if self._current_gpu_key == key:
            self._current_gpu_key = None
        _force_gc()

    def _evict_gpu_tenant(self) -> None:
        if self._current_gpu_key and self._current_gpu_key in self._loaded:
            logger.info("Evicting GPU tenant '%s' to free VRAM.", self._current_gpu_key)
            self._unload(self._current_gpu_key)
