from __future__ import annotations

import logging
import os
from typing import Optional, Tuple

import cv2
import numpy as np

from src.core.contracts import FaceBox, NoisePrintResult

logger = logging.getLogger("ca_monk.noiseprint")


class NoisePrintAnalyzer:
    """
    Document forgery detector — dual-branch EdgeDoc-inspired architecture:
      Branch 1: PRNU/NoisePrint variance-mismatch (face vs. background)
      Branch 2: Scharr high-pass edge residual for splice boundary localization
      + Error Level Analysis (ELA) — the "Paste Attack" detector

    Generates a pixel-level tampering heatmap for evidence cards.
    """

    def __init__(self, discrepancy_threshold: float = 0.35, correlation_threshold: float = 0.6) -> None:
        self.discrepancy_threshold = discrepancy_threshold
        self.correlation_threshold = correlation_threshold

    def _residual(self, image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
        denoised = cv2.bilateralFilter(gray, 7, 30, 30)
        residual = gray - denoised
        return residual

    def _scharr_residual(self, image: np.ndarray) -> np.ndarray:
        """
        High-pass Scharr edge residual — detects splice boundaries where
        two different camera sensors meet. Sharper boundary = higher splice
        likelihood.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
        sx = cv2.Scharr(gray, cv2.CV_32F, 1, 0)
        sy = cv2.Scharr(gray, cv2.CV_32F, 0, 1)
        edge_mag = np.sqrt(sx**2 + sy**2)
        return edge_mag

    def _split_regions(
        self, residual: np.ndarray, face_box: Optional[FaceBox]
    ) -> Tuple[np.ndarray, np.ndarray]:
        h, w = residual.shape
        if face_box is None:
            fw = int(w * 0.35)
            fh = int(h * 0.45)
            fx = (w - fw) // 2
            fy = int(h * 0.18)
        else:
            fx, fy, fw, fh = face_box.x, face_box.y, face_box.w, face_box.h
        fx = max(0, min(w - 1, fx))
        fy = max(0, min(h - 1, fy))
        fw = max(1, min(w - fx, fw))
        fh = max(1, min(h - fy, fh))

        face = residual[fy : fy + fh, fx : fx + fw]
        bg_mask = np.ones_like(residual, dtype=np.uint8)
        bg_mask[fy : fy + fh, fx : fx + fw] = 0
        background = residual[bg_mask > 0]
        return face, background

    def _correlation(self, face: np.ndarray, background: np.ndarray) -> float:
        if face.size == 0 or background.size == 0:
            return 0.0
        face_flat = face.flatten().astype(np.float32)
        bg_flat = background.flatten().astype(np.float32)
        target_len = min(face_flat.size, bg_flat.size, 20000)
        if target_len < 128:
            return 0.0
        face_sample = face_flat[:target_len]
        bg_sample = bg_flat[:target_len]
        face_sample -= np.mean(face_sample)
        bg_sample -= np.mean(bg_sample)
        denom = float(np.linalg.norm(face_sample) * np.linalg.norm(bg_sample))
        if denom <= 1e-8:
            return 0.0
        return float(np.dot(face_sample, bg_sample) / denom)

    # ------------------------------------------------------------------
    # Error Level Analysis (ELA) — Paste Attack detection
    # ------------------------------------------------------------------
    def compute_ela(
        self,
        image: np.ndarray,
        quality: int = 95,
        amplification: int = 15,
    ) -> np.ndarray:
        """
        ELA: Save image at `quality`%, subtract from original, amplify difference.
        Areas with inconsistent compression (pasted photos) glow brighter.
        """
        # Encode to JPEG in memory (no temp file needed)
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, encoded = cv2.imencode(".jpg", image, encode_param)
        resaved = cv2.imdecode(encoded, cv2.IMREAD_COLOR)

        if resaved is None:
            return np.zeros_like(image)

        diff = cv2.absdiff(image, resaved)
        ela = cv2.scaleAdd(diff, amplification, np.zeros_like(diff))
        return ela

    def generate_tamper_heatmap(
        self,
        image_path: str,
        face_box: Optional[FaceBox] = None,
        save_path: Optional[str] = None,
    ) -> Optional[str]:
        """
        Generate pixel-level ELA + Scharr dual-branch tampering heatmap.
        
        Spliced areas (e.g. pasted face on ID) glow bright red against
        a dark blue authentic background — forensic-grade localization.
        """
        image = cv2.imread(image_path)
        if image is None:
            return None

        # Branch 1: ELA (compression inconsistency)
        ela = self.compute_ela(image)
        ela_gray = cv2.cvtColor(ela, cv2.COLOR_BGR2GRAY).astype(np.float32)

        # Branch 2: Scharr edge residual (splice boundary detection)
        edge_residual = self._scharr_residual(image)
        # Normalize both to [0, 1]
        ela_norm = ela_gray / max(ela_gray.max(), 1.0)
        edge_norm = edge_residual / max(edge_residual.max(), 1.0)

        # Fuse: weighted combination (ELA detects content, Scharr detects boundaries)
        fused = 0.6 * ela_norm + 0.4 * edge_norm
        fused = np.clip(fused, 0, 1)

        # Gaussian blur for smooth heatmap (reduce pixel noise)
        fused_smooth = cv2.GaussianBlur(fused, (15, 15), 3.0)
        fused_u8 = np.clip(fused_smooth * 255, 0, 255).astype(np.uint8)

        # Compute face ROI vs background noise ratio for splice detection
        splice_ratio = 1.0
        if face_box:
            fx, fy, fw, fh = face_box.x, face_box.y, face_box.w, face_box.h
            fx = max(0, fx); fy = max(0, fy)
            fw = min(fw, image.shape[1] - fx); fh = min(fh, image.shape[0] - fy)
            if fw > 0 and fh > 0:
                face_roi = fused_smooth[fy:fy+fh, fx:fx+fw]
                bg_mask = np.ones(fused_smooth.shape, dtype=bool)
                bg_mask[fy:fy+fh, fx:fx+fw] = False
                bg_roi = fused_smooth[bg_mask]
                face_mean = float(np.mean(face_roi)) if face_roi.size else 0
                bg_mean = float(np.mean(bg_roi)) if bg_roi.size else 0.001
                splice_ratio = face_mean / max(bg_mean, 0.001)

        # Apply INFERNO colormap for dramatic forensic visual
        heatmap = cv2.applyColorMap(fused_u8, cv2.COLORMAP_INFERNO)

        # Overlay on dimmed original for context
        dimmed = (image.astype(np.float32) * 0.3).astype(np.uint8)
        blended = cv2.addWeighted(heatmap, 0.65, dimmed, 0.35, 0)

        # Draw face box + splice ratio annotation
        if face_box:
            cv2.rectangle(
                blended,
                (face_box.x, face_box.y),
                (face_box.x + face_box.w, face_box.y + face_box.h),
                (0, 255, 255), 2,
            )
            cv2.putText(blended, "FACE ROI", (face_box.x, face_box.y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
            ratio_color = (0, 0, 255) if splice_ratio > 1.5 else (0, 255, 0)
            cv2.putText(blended, f"Noise Ratio: {splice_ratio:.2f}x",
                        (face_box.x, face_box.y + face_box.h + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, ratio_color, 1, cv2.LINE_AA)

        # Compute ELA score
        score = float(np.mean(ela_gray)) / 255.0
        is_tampered = score > 0.06 or splice_ratio > 1.5
        label = "SPLICED_PHOTO" if splice_ratio > 1.5 else (
            "TAMPERING SUSPECTED" if score > 0.06 else "INTEGRITY OK"
        )
        color = (0, 0, 255) if is_tampered else (0, 255, 0)

        cv2.putText(blended, f"ELA+SCHARR DUAL-BRANCH — {label}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
        cv2.putText(blended, f"ELA: {score:.4f}  Splice Ratio: {splice_ratio:.2f}x", (10, 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)

        if save_path:
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            cv2.imwrite(save_path, blended)
            logger.info("Tamper heatmap saved → %s", save_path)
            return save_path
        return None

    def analyze(self, image_path: str, face_box: Optional[FaceBox] = None) -> NoisePrintResult:
        image = cv2.imread(image_path)
        if image is None:
            return NoisePrintResult(
                face_noise_variance=0.0,
                background_noise_variance=0.0,
                variance_discrepancy=0.0,
                suspected_splice=False,
            )

        # Branch 1: PRNU bilateral residual
        residual = self._residual(image)
        face, bg = self._split_regions(residual, face_box)

        face_var = float(np.var(face)) if face.size else 0.0
        bg_var = float(np.var(bg)) if bg.size else 0.0
        denom = max(bg_var, 1e-8)
        variance_discrepancy = abs(face_var - bg_var) / denom
        corr = self._correlation(face, bg)
        corr_gap = max(0.0, 1.0 - ((corr + 1.0) / 2.0))

        # Branch 2: Scharr edge residual
        edge_res = self._scharr_residual(image)
        edge_face, edge_bg = self._split_regions(edge_res, face_box)
        edge_face_var = float(np.var(edge_face)) if edge_face.size else 0.0
        edge_bg_var = float(np.var(edge_bg)) if edge_bg.size else 0.0
        edge_ratio = edge_face_var / max(edge_bg_var, 1e-8)

        # Fused discrepancy: bilateral + correlation + edge branches
        discrepancy = 0.45 * variance_discrepancy + 0.3 * corr_gap + 0.25 * min(edge_ratio, 3.0)

        # Splice detection: any branch can trigger
        suspected = bool(
            discrepancy >= self.discrepancy_threshold
            or corr < self.correlation_threshold
            or (face_var / max(bg_var, 1e-8)) > 1.5  # noise ratio > 1.5x = spliced photo
        )

        return NoisePrintResult(
            face_noise_variance=face_var,
            background_noise_variance=bg_var,
            variance_discrepancy=float(discrepancy),
            suspected_splice=suspected,
        )
