from __future__ import annotations

import logging
import os
from typing import Optional

import cv2
import numpy as np
from scipy.fftpack import dct

from src.forensics.frequency_extractor import FrequencyExtractor

logger = logging.getLogger("ca_monk.f3net")


class FrequencyAwareDeepfakeDetector:
    """
    CPU-only F3-Net Lite detector:
    - scipy DCT-based high-frequency projection
    - checkerboard artifact score (GAN upsampling signature)
    - Spectral Ghost Image generation for evidence cards
    """

    def __init__(self, threshold: float = 0.55) -> None:
        self.threshold = threshold
        self.extractor = FrequencyExtractor()

    def _checkerboard_score(self, image: np.ndarray) -> float:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        dct2 = dct(dct(gray.T, norm="ortho").T, norm="ortho")
        mag = np.abs(dct2)
        h, w = mag.shape

        # High-frequency block
        hf = mag[h // 2 :, w // 2 :]
        if hf.size == 0:
            return 0.0

        # Checkerboard artifacts produce alternating bin peaks.
        even_bins = hf[::2, ::2]
        odd_bins = hf[1::2, 1::2]
        cross_bins = hf[::2, 1::2]

        even_energy = float(np.mean(even_bins) + 1e-8)
        odd_energy = float(np.mean(odd_bins) + 1e-8)
        cross_energy = float(np.mean(cross_bins) + 1e-8)
        imbalance = abs(even_energy - odd_energy) / max(even_energy + odd_energy, 1e-8)
        ratio = (even_energy + odd_energy) / max(cross_energy, 1e-8)

        score = 0.6 * np.clip(imbalance, 0.0, 1.0) + 0.4 * np.clip((ratio - 1.0) / 2.0, 0.0, 1.0)
        return float(np.clip(score, 0.0, 1.0))

    def predict_probability(self, image: np.ndarray) -> float:
        try:
            freq_img = self.extractor.extract(image)
            score = self._checkerboard_score(freq_img)
            return float(np.clip(score, 0.0, 1.0))
        except Exception:
            return 0.0

    def is_deepfake(self, image: np.ndarray) -> bool:
        return self.predict_probability(image) >= self.threshold

    # ------------------------------------------------------------------
    # Spectral Ghost Image (for evidence card)
    # ------------------------------------------------------------------
    def generate_spectral_heatmap(
        self,
        image: np.ndarray,
        save_path: Optional[str] = None,
    ) -> np.ndarray:
        """
        Convert image to Frequency Domain → Apply High-Pass Filter →
        Inverse Transform → Generate 'Ghost Image'.

        If it shows a checkerboard pattern → GAN deepfake.

        Returns the heatmap as a BGR numpy array.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
        rows, cols = gray.shape
        crow, ccol = rows // 2, cols // 2

        # 1. FFT → shift zero-frequency to center
        f = np.fft.fft2(gray)
        fshift = np.fft.fftshift(f)

        # 2. High-Pass Filter: zero out low-frequency center disc
        radius = int(min(rows, cols) * 0.08)
        mask = np.ones((rows, cols), dtype=np.float32)
        y, x = np.ogrid[:rows, :cols]
        center_mask = ((x - ccol) ** 2 + (y - crow) ** 2) <= radius ** 2
        mask[center_mask] = 0.0
        fshift_filtered = fshift * mask

        # 3. Inverse FFT → ghost image (high-freq residual)
        f_ishift = np.fft.ifftshift(fshift_filtered)
        ghost = np.fft.ifft2(f_ishift)
        ghost_mag = np.abs(ghost)

        # 4. Normalize to 0–255
        ghost_norm = cv2.normalize(ghost_mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # 5. Apply JET colormap for dramatic visual
        heatmap = cv2.applyColorMap(ghost_norm, cv2.COLORMAP_JET)

        # 6. Overlay subtle original for context
        img_gray_3ch = cv2.cvtColor(
            cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8),
            cv2.COLOR_GRAY2BGR,
        )
        blended = cv2.addWeighted(heatmap, 0.7, img_gray_3ch, 0.3, 0)

        # 7. Add analysis text
        prob = self.predict_probability(image)
        label = "DEEPFAKE SUSPECTED" if prob >= self.threshold else "AUTHENTIC"
        color = (0, 0, 255) if prob >= self.threshold else (0, 255, 0)
        cv2.putText(blended, f"SPECTRAL ANALYSIS — {label}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
        cv2.putText(blended, f"Checkerboard Score: {prob:.3f}  |  Threshold: {self.threshold:.2f}",
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)

        if save_path:
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            cv2.imwrite(save_path, blended)
            logger.info("Spectral ghost image saved → %s", save_path)

        return blended
