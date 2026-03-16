from __future__ import annotations

import os
from typing import Any, Dict

import cv2
import numpy as np

from src.core.contracts import FaceBox


def _crop(img: np.ndarray, box: FaceBox) -> np.ndarray:
    x1 = max(0, box.x)
    y1 = max(0, box.y)
    x2 = max(x1 + 1, x1 + box.w)
    y2 = max(y1 + 1, y1 + box.h)
    return img[y1:y2, x1:x2]


class SiameseGradCAM:
    """
    Practical XAI wrapper for ArcFace-like embeddings.
    ONNX inference does not expose gradients directly, so this uses
    an occlusion-based attribution map as a deployable fallback.
    """

    def __init__(self, patch_size: int = 24, stride: int = 12) -> None:
        self.patch_size = patch_size
        self.stride = stride

    def _cosine(self, v1: np.ndarray, v2: np.ndarray) -> float:
        n1 = float(np.linalg.norm(v1))
        n2 = float(np.linalg.norm(v2))
        denom = max(n1 * n2, 1e-8)
        return float(np.dot(v1, v2) / denom)

    def _handcrafted_embedding(self, face: np.ndarray) -> np.ndarray:
        # Lightweight fallback descriptor for attribution when direct model gradients
        # are unavailable from ONNX runtime.
        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (128, 128), interpolation=cv2.INTER_AREA)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag = cv2.magnitude(gx, gy)
        hist = cv2.calcHist([mag.astype(np.uint8)], [0], None, [64], [0, 256]).flatten()
        norm = np.linalg.norm(hist) + 1e-8
        return (hist / norm).astype(np.float32)

    def _occlusion_heatmap(self, face_a: np.ndarray, face_b: np.ndarray) -> np.ndarray:
        base_a = self._handcrafted_embedding(face_a)
        base_b = self._handcrafted_embedding(face_b)
        base_sim = self._cosine(base_a, base_b)

        h, w = face_a.shape[:2]
        heat = np.zeros((h, w), dtype=np.float32)

        for y in range(0, max(1, h - self.patch_size + 1), self.stride):
            for x in range(0, max(1, w - self.patch_size + 1), self.stride):
                occluded = face_a.copy()
                cv2.rectangle(
                    occluded,
                    (x, y),
                    (min(w - 1, x + self.patch_size), min(h - 1, y + self.patch_size)),
                    (0, 0, 0),
                    -1,
                )
                emb_occ = self._handcrafted_embedding(occluded)
                sim_occ = self._cosine(emb_occ, base_b)
                contribution = max(0.0, base_sim - sim_occ)
                heat[y : min(h, y + self.patch_size), x : min(w, x + self.patch_size)] += contribution

        if np.max(heat) > 0:
            heat = heat / np.max(heat)
        return heat

    def explain(
        self,
        image_a_path: str,
        box_a: FaceBox,
        image_b_path: str,
        box_b: FaceBox,
        save_path: str | None = None,
    ) -> Dict[str, Any]:
        img_a = cv2.imread(image_a_path)
        img_b = cv2.imread(image_b_path)
        if img_a is None or img_b is None:
            return {"error": "Unable to load one or both images for explainability map."}

        face_a = _crop(img_a, box_a)
        face_b = _crop(img_b, box_b)
        if face_a.size == 0 or face_b.size == 0:
            return {"error": "Invalid face crop for explainability map."}

        face_a = cv2.resize(face_a, (224, 224), interpolation=cv2.INTER_AREA)
        face_b = cv2.resize(face_b, (224, 224), interpolation=cv2.INTER_AREA)
        heat = self._occlusion_heatmap(face_a, face_b)
        heat_u8 = (heat * 255).astype(np.uint8)
        overlay = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
        blended = cv2.addWeighted(face_a, 0.55, overlay, 0.45, 0)
        overlay_path = None
        if save_path:
            try:
                os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
                cv2.imwrite(save_path, blended)
                overlay_path = save_path
            except Exception:
                overlay_path = None

        # Eyes + nose regions tend to dominate embedding similarity.
        regions = [
            {"region": "eyes", "weight": float(np.mean(heat[60:110, 40:180]))},
            {"region": "nose", "weight": float(np.mean(heat[90:160, 90:140]))},
            {"region": "mouth", "weight": float(np.mean(heat[145:205, 70:160]))},
        ]

        return {
            "method": "siamese_occlusion_cam",
            "overlay_path": overlay_path,
            "regions": sorted(regions, key=lambda x: x["weight"], reverse=True),
        }
