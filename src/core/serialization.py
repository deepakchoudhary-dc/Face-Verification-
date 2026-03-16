from __future__ import annotations

from typing import Any


def to_builtin(value: Any) -> Any:
    """
    Recursively convert runtime objects into plain JSON-safe Python types.

    This is required because several subsystems return NumPy scalars / arrays or
    Pydantic models, which are valid in-memory but break strict JSON encoding.
    """
    if hasattr(value, "model_dump"):
        try:
            return to_builtin(value.model_dump())
        except Exception:
            pass

    if hasattr(value, "item") and not isinstance(value, (str, bytes, bytearray)):
        try:
            return to_builtin(value.item())
        except Exception:
            pass

    if hasattr(value, "tolist") and not isinstance(value, (str, bytes, bytearray)):
        try:
            return to_builtin(value.tolist())
        except Exception:
            pass

    if isinstance(value, dict):
        return {str(k): to_builtin(v) for k, v in value.items()}

    if isinstance(value, (list, tuple, set)):
        return [to_builtin(v) for v in value]

    if hasattr(value, "__fspath__"):
        try:
            return value.__fspath__()
        except Exception:
            pass

    return value
