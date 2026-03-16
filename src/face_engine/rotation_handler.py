from __future__ import annotations

import os
import tempfile
from typing import Dict, Tuple

import cv2
import numpy as np


class RotationHandler:
    """
    Rotation selection without DeepFace dependency.
    Chooses angle maximizing frontal-face detector confidence proxy.
    """

    def __init__(self, angles: Tuple[int, ...] = (0, 90, 180, 270)) -> None:
        self.angles = angles
        self.temp_dir = os.path.join(tempfile.gettempdir(), "ca_monk_rot")
        os.makedirs(self.temp_dir, exist_ok=True)
        self.cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

    def rotate_image(self, image: np.ndarray, angle: float) -> np.ndarray:
        if angle % 360 == 0:
            return image
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        mat = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(image, mat, (w, h), flags=cv2.INTER_LINEAR)

    def _face_score(self, image: np.ndarray) -> float:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
        )
        if len(faces) == 0:
            return 0.0
        areas = [float(w * h) for (_, _, w, h) in faces]
        return max(areas)

    def find_best_rotation_array(self, image: np.ndarray) -> Tuple[np.ndarray, int]:
        best_angle = 0
        best_score = -1.0
        best_img = image
        for angle in self.angles:
            rotated = self.rotate_image(image, angle)
            score = self._face_score(rotated)
            if score > best_score:
                best_score = score
                best_angle = angle
                best_img = rotated
        return best_img, int(best_angle)

    def find_best_rotation(self, image_path: str) -> Dict[str, object]:
        img = cv2.imread(image_path)
        if img is None:
            return {"best_image_path": image_path, "best_angle": 0}
        rotated, angle = self.find_best_rotation_array(img)
        out_path = os.path.join(self.temp_dir, f"rot_{angle}_{os.path.basename(image_path)}")
        cv2.imwrite(out_path, rotated)
        return {"best_image_path": out_path, "best_angle": angle}

    def cleanup(self) -> None:
        # Files are ephemeral. Keep cleanup best-effort.
        try:
            for name in os.listdir(self.temp_dir):
                path = os.path.join(self.temp_dir, name)
                if os.path.isfile(path):
                    os.remove(path)
        except Exception:
            pass

