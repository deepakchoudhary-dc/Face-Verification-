from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

logger = logging.getLogger("ca_monk.liveness")


class LivenessDetector:
    """
    CPU-first passive presentation-attack detector for still images.

    The detector now supports two execution modes:
    1. Local ONNX PAD model, if `CA_MONK_PAD_ONNX_PATH` (or a default path) exists.
    2. A fused heuristic fallback that remains fully CPU-safe.

    The public output always reports which path was active so the system never
    implies that a neural PAD model was used when it was not.
    """

    DEFAULT_MODEL_CANDIDATES = (
        "models/pad_silent_face.onnx",
        "models/mini_fasnet.onnx",
        "models/anti_spoof.onnx",
    )

    def __init__(self, model_path: Optional[str] = None) -> None:
        self.model_path = self._resolve_model_path(model_path)
        self.session = None
        self.input_name: Optional[str] = None
        self.output_names: List[str] = []
        self.providers: List[str] = []
        self.input_layout = "NCHW"
        self.input_size: Tuple[int, int] = (80, 80)
        self.input_channels = 3
        configured_live_index = os.getenv("CA_MONK_PAD_LIVE_INDEX", "").strip()
        self.live_index = int(configured_live_index) if configured_live_index.isdigit() else None
        self.attack_labels = [
            item.strip()
            for item in os.getenv("CA_MONK_PAD_ATTACK_LABELS", "").split(",")
            if item.strip()
        ]
        self._init_pad_model()

    def capabilities(self) -> Dict[str, Any]:
        if self.session is None:
            return {
                "backend": "heuristic_cpu_pad",
                "model_loaded": False,
                "model_path": str(self.model_path) if self.model_path else None,
                "providers": [],
                "input_size": list(self.input_size),
                "input_layout": self.input_layout,
            }
        return {
            "backend": "onnx_pad+heuristics",
            "model_loaded": True,
            "model_path": str(self.model_path) if self.model_path else None,
            "providers": list(self.providers),
            "input_size": list(self.input_size),
            "input_layout": self.input_layout,
        }

    def check_liveness(self, face_img: np.ndarray) -> Dict[str, Any]:
        if face_img is None or face_img.size == 0:
            return {
                "score": 0.0,
                "is_real": False,
                "signal_state": "spoof",
                "attack_indicators": ["empty_face_crop"],
                "backend": "invalid_input",
                "details": {"reason": "empty_face_crop"},
            }

        face = self._normalize(face_img)
        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

        heuristic = self._heuristic_analysis(face, gray)
        model = self._model_analysis(face) if self.session is not None else {
            "available": False,
            "backend": "heuristic_cpu_pad",
        }

        if model.get("available"):
            return self._fuse_model_and_heuristics(heuristic, model)
        return self._heuristic_decision(heuristic)

    def _init_pad_model(self) -> None:
        if not self.model_path or not self.model_path.is_file():
            return

        try:
            import onnxruntime as ort

            options = ort.SessionOptions()
            configured_threads = os.getenv("CA_MONK_ORT_THREADS", "").strip()
            if configured_threads.isdigit():
                options.intra_op_num_threads = max(1, int(configured_threads))
                options.inter_op_num_threads = 1

            self.session = ort.InferenceSession(
                str(self.model_path),
                sess_options=options,
                providers=self._select_providers(ort),
            )
            self.providers = list(self.session.get_providers())
            input_meta = self.session.get_inputs()[0]
            self.input_name = input_meta.name
            self.output_names = [item.name for item in self.session.get_outputs()]
            self.input_size, self.input_layout, self.input_channels = self._infer_input_spec(input_meta.shape)
            logger.info(
                "PAD ONNX loaded - providers=%s, input_size=%s, layout=%s",
                self.providers,
                self.input_size,
                self.input_layout,
            )
        except Exception as exc:
            logger.warning("PAD ONNX unavailable (%s) - using heuristic CPU PAD.", exc)
            self.session = None
            self.input_name = None
            self.output_names = []
            self.providers = []

    @staticmethod
    def _select_providers(ort: Any) -> List[str]:
        available = ort.get_available_providers()
        providers: List[str] = []
        if "OpenVINOExecutionProvider" in available:
            providers.append("OpenVINOExecutionProvider")
        allow_gpu = os.getenv("CA_MONK_ALLOW_GPU", "0").strip().lower() in {"1", "true", "yes"}
        if allow_gpu and "CUDAExecutionProvider" in available:
            providers.append("CUDAExecutionProvider")
        providers.append("CPUExecutionProvider")
        return providers

    @classmethod
    def _resolve_model_path(cls, explicit_path: Optional[str]) -> Optional[Path]:
        candidates: List[str] = []
        if explicit_path:
            candidates.append(explicit_path)
        env_path = os.getenv("CA_MONK_PAD_ONNX_PATH", "").strip()
        if env_path:
            candidates.append(env_path)
        candidates.extend(cls.DEFAULT_MODEL_CANDIDATES)

        for candidate in candidates:
            path = Path(candidate)
            if path.is_file():
                return path
        return Path(candidates[0]) if candidates else None

    @staticmethod
    def _infer_input_spec(shape: Sequence[Any]) -> Tuple[Tuple[int, int], str, int]:
        dims = list(shape or [])
        numeric = [int(item) for item in dims if isinstance(item, (int, np.integer)) and int(item) > 0]
        if len(dims) >= 4 and isinstance(dims[1], (int, np.integer)) and int(dims[1]) in {1, 3}:
            height = int(dims[2]) if isinstance(dims[2], (int, np.integer)) and int(dims[2]) > 0 else 80
            width = int(dims[3]) if isinstance(dims[3], (int, np.integer)) and int(dims[3]) > 0 else 80
            return (width, height), "NCHW", int(dims[1])
        if len(dims) >= 4 and isinstance(dims[-1], (int, np.integer)) and int(dims[-1]) in {1, 3}:
            height = int(dims[1]) if isinstance(dims[1], (int, np.integer)) and int(dims[1]) > 0 else 80
            width = int(dims[2]) if isinstance(dims[2], (int, np.integer)) and int(dims[2]) > 0 else 80
            return (width, height), "NHWC", int(dims[-1])
        if len(numeric) >= 2:
            side = int(numeric[-1])
            return (side, side), "NCHW", 3
        return (80, 80), "NCHW", 3

    @staticmethod
    def _normalize(face_img: np.ndarray) -> np.ndarray:
        if face_img.shape[0] < 96 or face_img.shape[1] < 96:
            return cv2.resize(face_img, (160, 160), interpolation=cv2.INTER_CUBIC)
        max_dim = max(face_img.shape[:2])
        if max_dim > 320:
            scale = 320.0 / max_dim
            return cv2.resize(
                face_img,
                (
                    max(32, int(face_img.shape[1] * scale)),
                    max(32, int(face_img.shape[0] * scale)),
                ),
                interpolation=cv2.INTER_AREA,
            )
        return face_img.copy()

    def _heuristic_analysis(self, face: np.ndarray, gray: np.ndarray) -> Dict[str, Any]:
        texture = self._texture_signal(gray)
        frequency = self._frequency_signal(gray)
        reflections = self._reflection_signal(face)
        banding = self._banding_signal(gray)

        signals = {
            "texture": texture["live_signal"],
            "frequency": frequency["live_signal"],
            "reflections": reflections["live_signal"],
            "banding": banding["live_signal"],
        }
        weights = {
            "texture": 0.34,
            "frequency": 0.28,
            "reflections": 0.20,
            "banding": 0.18,
        }
        score = float(sum(signals[name] * weights[name] for name in signals))

        attack_indicators: List[str] = []
        if texture["spoof_indicator"]:
            attack_indicators.append("texture_pattern")
        if frequency["spoof_indicator"]:
            attack_indicators.append("moire_or_fft_peaks")
        if reflections["spoof_indicator"]:
            attack_indicators.append("laminated_or_screen_reflections")
        if banding["spoof_indicator"]:
            attack_indicators.append("row_column_banding")

        return {
            "score": round(score, 4),
            "attack_indicators": attack_indicators,
            "details": {
                "texture": texture,
                "frequency": frequency,
                "reflections": reflections,
                "banding": banding,
            },
        }

    def _heuristic_decision(self, heuristic: Dict[str, Any]) -> Dict[str, Any]:
        score = float(heuristic.get("score", 0.0) or 0.0)
        attack_indicators = list(heuristic.get("attack_indicators", []) or [])

        if len(attack_indicators) >= 2 or score <= 0.38:
            signal_state = "spoof"
            is_real = False
        elif score >= 0.62 and not attack_indicators:
            signal_state = "live"
            is_real = True
        else:
            signal_state = "indeterminate"
            is_real = score >= 0.52 and len(attack_indicators) < 2

        return {
            "score": round(score, 4),
            "is_real": bool(is_real),
            "signal_state": signal_state,
            "attack_indicators": attack_indicators,
            "backend": "heuristic_cpu_pad",
            "model_loaded": False,
            "details": {
                "heuristic": heuristic,
            },
        }

    def _model_analysis(self, face: np.ndarray) -> Dict[str, Any]:
        if self.session is None or self.input_name is None:
            return {"available": False, "backend": "heuristic_cpu_pad"}

        try:
            tensor = self._pad_preprocess(face)
            outputs = self.session.run(None, {self.input_name: tensor})
            raw = self._extract_primary_scores(outputs)
            probabilities = self._normalize_scores(raw)
            live_index = self._resolve_live_index(len(probabilities))
            live_probability = float(probabilities[live_index])
            predicted_index = int(np.argmax(probabilities))
            labels = self._resolve_labels(len(probabilities), live_index)
            predicted_label = labels[predicted_index]

            attack_scores = {
                labels[index]: round(float(score), 4)
                for index, score in enumerate(probabilities)
                if index != live_index
            }
            return {
                "available": True,
                "backend": "onnx_pad",
                "providers": list(self.providers),
                "model_path": str(self.model_path) if self.model_path else None,
                "live_probability": round(live_probability, 4),
                "predicted_index": predicted_index,
                "predicted_label": predicted_label,
                "attack_class": None if predicted_index == live_index else predicted_label,
                "attack_scores": attack_scores,
                "raw_scores": [round(float(item), 6) for item in probabilities.tolist()],
            }
        except Exception as exc:
            logger.warning("PAD ONNX inference failed (%s) - falling back to heuristics.", exc)
            return {
                "available": False,
                "backend": "heuristic_cpu_pad",
                "error": str(exc),
            }

    def _pad_preprocess(self, face: np.ndarray) -> np.ndarray:
        width, height = self.input_size
        resized = cv2.resize(face, (width, height), interpolation=cv2.INTER_AREA)
        if self.input_channels == 1:
            resized = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            resized = np.expand_dims(resized, axis=-1)
        else:
            resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        array = resized.astype(np.float32) / 255.0
        mean = np.asarray([0.485, 0.456, 0.406][: self.input_channels], dtype=np.float32)
        std = np.asarray([0.229, 0.224, 0.225][: self.input_channels], dtype=np.float32)
        array = (array - mean) / std

        if self.input_layout == "NCHW":
            array = np.transpose(array, (2, 0, 1))
        return np.expand_dims(array, axis=0).astype(np.float32)

    @staticmethod
    def _extract_primary_scores(outputs: Sequence[Any]) -> np.ndarray:
        numeric: List[np.ndarray] = []
        for output in outputs:
            arr = np.asarray(output)
            if arr.size == 0 or not np.issubdtype(arr.dtype, np.number):
                continue
            numeric.append(arr.astype(np.float32).reshape(-1))
        if not numeric:
            raise ValueError("pad_model_returned_no_numeric_output")
        numeric.sort(key=lambda item: item.size, reverse=True)
        return numeric[0]

    @staticmethod
    def _normalize_scores(raw: np.ndarray) -> np.ndarray:
        scores = np.asarray(raw, dtype=np.float32).reshape(-1)
        if scores.size == 1:
            value = float(scores[0])
            if 0.0 <= value <= 1.0:
                live_probability = value
            else:
                live_probability = float(1.0 / (1.0 + np.exp(-value)))
            return np.asarray([1.0 - live_probability, live_probability], dtype=np.float32)

        if np.all(scores >= 0.0) and np.isclose(float(np.sum(scores)), 1.0, atol=1e-3):
            probs = scores
        else:
            scores = scores - float(np.max(scores))
            exp_scores = np.exp(scores)
            probs = exp_scores / max(float(np.sum(exp_scores)), 1e-8)
        return probs.astype(np.float32)

    def _resolve_live_index(self, class_count: int) -> int:
        if self.live_index is not None:
            return max(0, min(class_count - 1, int(self.live_index)))
        return class_count - 1 if class_count > 2 else 1

    def _resolve_labels(self, class_count: int, live_index: int) -> List[str]:
        if self.attack_labels and len(self.attack_labels) == class_count:
            return list(self.attack_labels)

        labels = [f"class_{idx}" for idx in range(class_count)]
        labels[live_index] = "live"
        attack_defaults = [
            "print_attack",
            "replay_attack",
            "mask_attack",
            "artifact_attack",
        ]
        attack_cursor = 0
        for idx in range(class_count):
            if idx == live_index:
                continue
            if attack_cursor < len(attack_defaults):
                labels[idx] = attack_defaults[attack_cursor]
                attack_cursor += 1
        return labels

    def _fuse_model_and_heuristics(
        self,
        heuristic: Dict[str, Any],
        model: Dict[str, Any],
    ) -> Dict[str, Any]:
        heuristic_score = float(heuristic.get("score", 0.0) or 0.0)
        model_live_probability = float(model.get("live_probability", 0.0) or 0.0)
        score = float(np.clip(0.62 * model_live_probability + 0.38 * heuristic_score, 0.0, 1.0))

        attack_indicators = list(heuristic.get("attack_indicators", []) or [])
        attack_class = model.get("attack_class")
        if attack_class:
            attack_indicators.append(f"model:{attack_class}")

        predicted_attack_strength = max(
            (float(value) for value in (model.get("attack_scores", {}) or {}).values()),
            default=0.0,
        )

        if model_live_probability <= 0.35 or (predicted_attack_strength >= 0.65 and heuristic_score <= 0.6):
            signal_state = "spoof"
            is_real = False
        elif model_live_probability >= 0.72 and heuristic_score >= 0.55 and not heuristic.get("attack_indicators"):
            signal_state = "live"
            is_real = True
        else:
            signal_state = "indeterminate"
            is_real = score >= 0.56 and predicted_attack_strength < 0.75

        return {
            "score": round(score, 4),
            "is_real": bool(is_real),
            "signal_state": signal_state,
            "attack_indicators": sorted(set(attack_indicators)),
            "backend": "onnx_pad+heuristics",
            "model_loaded": True,
            "details": {
                "heuristic": heuristic,
                "model": model,
            },
        }

    def _texture_signal(self, gray: np.ndarray) -> Dict[str, Any]:
        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        lbp = self._lbp(gray)
        hist, _ = np.histogram(lbp.ravel(), bins=256, range=(0, 256))
        hist = hist.astype(np.float64)
        hist /= max(hist.sum(), 1.0)
        entropy = float(-np.sum(hist * np.log2(hist + 1e-10)))
        peak_ratio = float(np.sum(hist > 0.04) / hist.size)

        sharp_signal = np.clip((lap_var - 25.0) / 120.0, 0.0, 1.0)
        entropy_signal = np.clip((entropy - 4.2) / 2.0, 0.0, 1.0)
        live_signal = float(np.clip(0.55 * sharp_signal + 0.45 * entropy_signal, 0.0, 1.0))
        spoof_indicator = bool((lap_var < 18.0 and entropy < 4.5) or peak_ratio > 0.11)

        return {
            "live_signal": round(live_signal, 4),
            "laplacian_variance": round(lap_var, 4),
            "lbp_entropy": round(entropy, 4),
            "lbp_peak_ratio": round(peak_ratio, 4),
            "spoof_indicator": spoof_indicator,
        }

    def _frequency_signal(self, gray: np.ndarray) -> Dict[str, Any]:
        spectrum = np.fft.fftshift(np.fft.fft2(gray.astype(np.float32)))
        magnitude = np.log1p(np.abs(spectrum))
        height, width = magnitude.shape
        center_y, center_x = height // 2, width // 2

        yy, xx = np.ogrid[:height, :width]
        dist2 = (yy - center_y) ** 2 + (xx - center_x) ** 2
        inner = dist2 <= int(min(height, width) * 0.10) ** 2
        ring = (dist2 >= int(min(height, width) * 0.18) ** 2) & (dist2 <= int(min(height, width) * 0.40) ** 2)
        ring_vals = magnitude[ring]

        if ring_vals.size == 0:
            return {
                "live_signal": 0.5,
                "moire_score": 0.0,
                "peak_density": 0.0,
                "spoof_indicator": False,
            }

        threshold = float(np.mean(ring_vals) + 2.5 * np.std(ring_vals))
        peaks = ring & (magnitude > threshold)
        peak_density = float(np.mean(peaks))
        center_energy = float(np.mean(magnitude[inner])) if np.any(inner) else 1.0
        ring_energy = float(np.mean(ring_vals))
        moire_score = float(
            np.clip(
                peak_density * 32.0 + max(0.0, ring_energy - center_energy) / max(center_energy, 1e-6),
                0.0,
                1.0,
            )
        )
        live_signal = float(np.clip(1.0 - moire_score, 0.0, 1.0))

        return {
            "live_signal": round(live_signal, 4),
            "moire_score": round(moire_score, 4),
            "peak_density": round(peak_density, 4),
            "spoof_indicator": bool(moire_score >= 0.55),
        }

    def _reflection_signal(self, face: np.ndarray) -> Dict[str, Any]:
        hsv = cv2.cvtColor(face, cv2.COLOR_BGR2HSV)
        value = hsv[:, :, 2].astype(np.float32)
        saturation = hsv[:, :, 1].astype(np.float32)

        highlight_mask = (value > 235) & (saturation < 50)
        highlight_ratio = float(np.mean(highlight_mask))
        specular_penalty = np.clip(highlight_ratio / 0.08, 0.0, 1.0)
        live_signal = float(np.clip(1.0 - specular_penalty, 0.0, 1.0))

        return {
            "live_signal": round(live_signal, 4),
            "highlight_ratio": round(highlight_ratio, 4),
            "spoof_indicator": bool(highlight_ratio > 0.06),
        }

    def _banding_signal(self, gray: np.ndarray) -> Dict[str, Any]:
        row_means = gray.astype(np.float32).mean(axis=1)
        col_means = gray.astype(np.float32).mean(axis=0)
        row_delta = np.abs(np.diff(row_means))
        col_delta = np.abs(np.diff(col_means))
        row_banding = float(np.percentile(row_delta, 95)) / 255.0 if row_delta.size else 0.0
        col_banding = float(np.percentile(col_delta, 95)) / 255.0 if col_delta.size else 0.0
        banding_score = float(np.clip(max(row_banding, col_banding) * 4.0, 0.0, 1.0))
        live_signal = float(np.clip(1.0 - banding_score, 0.0, 1.0))

        return {
            "live_signal": round(live_signal, 4),
            "row_banding": round(row_banding, 4),
            "col_banding": round(col_banding, 4),
            "spoof_indicator": bool(banding_score > 0.45),
        }

    @staticmethod
    def _lbp(gray: np.ndarray) -> np.ndarray:
        img = gray.astype(np.uint8)
        padded = np.pad(img, 1, mode="edge")
        center = padded[1:-1, 1:-1]

        lbp = (
            ((padded[:-2, :-2] >= center).astype(np.uint8) << 7)
            | ((padded[:-2, 1:-1] >= center).astype(np.uint8) << 6)
            | ((padded[:-2, 2:] >= center).astype(np.uint8) << 5)
            | ((padded[1:-1, 2:] >= center).astype(np.uint8) << 4)
            | ((padded[2:, 2:] >= center).astype(np.uint8) << 3)
            | ((padded[2:, 1:-1] >= center).astype(np.uint8) << 2)
            | ((padded[2:, :-2] >= center).astype(np.uint8) << 1)
            | (padded[1:-1, :-2] >= center).astype(np.uint8)
        )
        return lbp
