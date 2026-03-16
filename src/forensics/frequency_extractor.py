from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np
from scipy.fftpack import dct


class FrequencyExtractor:
    """
    DCT-based frequency map extractor for frequency-aware deepfake checks.
    """

    def __init__(self, resize_to: Tuple[int, int] = (224, 224)) -> None:
        self.resize_to = resize_to

    def extract(self, image: np.ndarray) -> np.ndarray:
        if image is None or image.size == 0:
            raise ValueError("Input image is empty")

        img = cv2.resize(image, self.resize_to, interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        dct2 = dct(dct(gray.T, norm="ortho").T, norm="ortho")
        mag = np.log1p(np.abs(dct2))
        mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        freq_rgb = cv2.cvtColor(mag, cv2.COLOR_GRAY2BGR)
        return freq_rgb
