from __future__ import annotations

import logging
import os
from typing import Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger("ca_monk.recognition")


class AdaFaceRecognizer:
    """
    AdaFace / ArcFace ONNX recognizer with **MagFace quality scoring**.

    Quality tiers (embedding L2 norm):
      norm < 22  →  Low Fidelity  / Unreliable
      22 ≤ norm ≤ 28  →  Usable / Standard
      norm > 28  →  High Fidelity / Enrollment Quality
    """

    NORM_LOW = 22.0       # Below this → unreliable
    NORM_HIGH = 28.0      # Above this → enrollment quality

    def __init__(
        self,
        model_path: Optional[str] = None,
        magface_norm_threshold: float = 20.0,
    ) -> None:
        self.model_path = model_path or os.getenv(
            "ADAFACE_ONNX_PATH", "models/adaface_ir101_webface12m.onnx"
        )
        self.magface_norm_threshold = float(
            os.getenv("MAGFACE_NORM_THRESHOLD", str(magface_norm_threshold))
        )
        self.session = None
        self.input_name = None
        self._init_session()

    def _init_session(self) -> None:
        try:
            import onnxruntime as ort

            providers = self._select_providers()
            options = ort.SessionOptions()
            configured_threads = os.getenv("CA_MONK_ORT_THREADS", "").strip()
            if configured_threads.isdigit():
                options.intra_op_num_threads = max(1, int(configured_threads))
                options.inter_op_num_threads = 1
            self.session = ort.InferenceSession(
                self.model_path,
                sess_options=options,
                providers=providers,
            )
            active = self.session.get_providers()
            logger.info("AdaFace ONNX loaded — providers: %s", active)
            self.input_name = self.session.get_inputs()[0].name
        except Exception as exc:
            logger.warning("AdaFace ONNX unavailable (%s) — falling back to InsightFace embeddings.", exc)
            self.session = None
            self.input_name = None

    @staticmethod
    def _select_providers():
        """CPU-first provider policy with optional GPU opt-in for benchmarks."""
        import onnxruntime as ort
        avail = ort.get_available_providers()
        preferred = []
        if "OpenVINOExecutionProvider" in avail:
            preferred.append("OpenVINOExecutionProvider")
        allow_gpu = os.getenv("CA_MONK_ALLOW_GPU", "0").strip().lower() in {"1", "true", "yes"}
        if allow_gpu and "CUDAExecutionProvider" in avail:
            preferred.append("CUDAExecutionProvider")
        preferred.append("CPUExecutionProvider")
        return preferred

    def _preprocess(self, face_bgr: np.ndarray) -> np.ndarray:
        """Strict 112×112 aligned crop → normalised CHW tensor."""
        face = cv2.resize(face_bgr, (112, 112), interpolation=cv2.INTER_AREA)
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        face = (face - 0.5) / 0.5
        face = np.transpose(face, (2, 0, 1))
        return np.expand_dims(face, axis=0).astype(np.float32)

    def embedding(self, face_bgr: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[float]]:
        if self.session is None or self.input_name is None or face_bgr is None or face_bgr.size == 0:
            return None, None
        try:
            x = self._preprocess(face_bgr)
            outputs = self.session.run(None, {self.input_name: x})
            emb = np.asarray(outputs[0], dtype=np.float32).reshape(-1)
            norm = float(np.linalg.norm(emb))
            return emb, norm
        except Exception:
            return None, None

    def quality_label(self, embedding_norm: float) -> str:
        """MagFace-inspired quality tier from embedding magnitude."""
        n = float(embedding_norm)
        if n < float(self.magface_norm_threshold):
            return "unidentifiable_noise"
        if n < self.NORM_LOW:
            return "low_quality_unreliable"    # Low Fidelity / Unreliable
        if n > self.NORM_HIGH:
            return "reliable"                  # High Fidelity / Enrollment Quality
        return "reliable"                      # Usable / Standard

    def quality_description(self, embedding_norm: float) -> str:
        """Human-readable MagFace quality description."""
        n = float(embedding_norm)
        if n < float(self.magface_norm_threshold):
            return "Unidentifiable Noise — reject"
        if n < self.NORM_LOW:
            return f"Low Fidelity / Unreliable (norm={n:.1f} < {self.NORM_LOW})"
        if n > self.NORM_HIGH:
            return f"High Fidelity / Enrollment Quality (norm={n:.1f} > {self.NORM_HIGH})"
        return f"Standard Fidelity (norm={n:.1f})"

    def capabilities(self) -> dict:
        return {
            "model_path": self.model_path,
            "loaded": self.session is not None,
            "providers": list(self.session.get_providers()) if self.session is not None else [],
            "magface_norm_threshold": float(self.magface_norm_threshold),
        }
