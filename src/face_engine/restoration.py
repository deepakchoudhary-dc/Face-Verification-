from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np


class CodeFormerONNXRestorer:
    """
    Optional CPU CodeFormer ONNX pre-restoration for blurry inputs.
    """

    def __init__(self, model_path: Optional[str] = None, blur_threshold: float = 60.0) -> None:
        self.model_path = model_path or os.getenv("CODEFORMER_ONNX_PATH", "models/codeformer.onnx")
        self.blur_threshold = float(os.getenv("CODEFORMER_BLUR_THRESHOLD", str(blur_threshold)))
        self.fidelity_weight = float(os.getenv("CODEFORMER_FIDELITY_WEIGHT", "0.7"))
        self.session = None
        self.input_name = None
        self.extra_inputs = {}
        self._init_session()

    def _init_session(self) -> None:
        try:
            import onnxruntime as ort

            self.session = ort.InferenceSession(
                self.model_path,
                providers=["CPUExecutionProvider"],
            )
            inputs = self.session.get_inputs()
            if inputs:
                self.input_name = inputs[0].name
            self.extra_inputs = {inp.name: inp for inp in inputs[1:]}
        except Exception:
            self.session = None
            self.input_name = None
            self.extra_inputs = {}

    def _blur_score(self, image: np.ndarray) -> float:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    def _mean_saturation(self, image: np.ndarray) -> float:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        return float(np.mean(hsv[:, :, 1]))

    def should_restore(self, image: np.ndarray) -> bool:
        if image is None or image.size == 0:
            return False
        return self._blur_score(image) < self.blur_threshold

    def _preprocess(self, image: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int]]:
        h, w = image.shape[:2]
        resized = cv2.resize(image, (512, 512), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        rgb = (rgb - 0.5) / 0.5
        chw = np.transpose(rgb, (2, 0, 1))
        return np.expand_dims(chw, axis=0).astype(np.float32), (w, h)

    def _postprocess(self, output: np.ndarray, original_size: Tuple[int, int]) -> np.ndarray:
        arr = np.asarray(output)
        if arr.ndim == 4:
            arr = arr[0]
        if arr.shape[0] in (1, 3):
            arr = np.transpose(arr, (1, 2, 0))
        arr = np.clip((arr * 0.5 + 0.5) * 255.0, 0, 255).astype(np.uint8)
        arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        return cv2.resize(arr, original_size, interpolation=cv2.INTER_CUBIC)

    def restore(self, image: np.ndarray) -> Tuple[np.ndarray, bool]:
        if self.session is None or self.input_name is None:
            return image, False
        try:
            x, original_size = self._preprocess(image)
            feed = {self.input_name: x}
            if "w" in self.extra_inputs:
                feed["w"] = np.array(self.fidelity_weight, dtype=np.float64)
            outputs = self.session.run(None, feed)
            restored = self._postprocess(outputs[0], original_size)
            return restored, True
        except Exception:
            return image, False

    def restore_if_beneficial(
        self,
        image: np.ndarray,
        min_sharpness_gain: float = 1.03,
        max_mean_shift: float = 14.0,
        max_saturation_shift: float = 18.0,
    ) -> Tuple[np.ndarray, bool, Dict[str, Any]]:
        if image is None or image.size == 0:
            return image, False, {"accepted": False, "reason": "empty_image"}
        if not self.should_restore(image):
            return image, False, {"accepted": False, "reason": "sharp_enough"}

        restored, ok = self.restore(image)
        if not ok:
            return image, False, {"accepted": False, "reason": "restore_failed"}

        pre_blur = self._blur_score(image)
        post_blur = self._blur_score(restored)
        sharpness_gain = post_blur / max(pre_blur, 1e-6)
        mean_shift = float(
            np.mean(
                np.abs(
                    restored.astype(np.float32) - image.astype(np.float32)
                )
            )
        )
        saturation_shift = abs(self._mean_saturation(restored) - self._mean_saturation(image))

        accepted = (
            sharpness_gain >= float(min_sharpness_gain)
            and mean_shift <= float(max_mean_shift)
            and saturation_shift <= float(max_saturation_shift)
        )
        metrics = {
            "accepted": accepted,
            "sharpness_gain": sharpness_gain,
            "mean_shift": mean_shift,
            "saturation_shift": saturation_shift,
        }
        return (restored if accepted else image), accepted, metrics
