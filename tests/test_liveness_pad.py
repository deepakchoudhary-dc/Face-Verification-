import unittest

import cv2
import numpy as np

from src.face_engine.liveness import LivenessDetector


def _live_like_face(size: int = 192) -> np.ndarray:
    img = np.full((size, size, 3), (42, 44, 48), dtype=np.uint8)
    center = (size // 2, size // 2)
    axes = (size // 3, int(size * 0.42))
    cv2.ellipse(img, center, axes, 0, 0, 360, (125, 165, 198), -1)

    yy, xx = np.mgrid[0:size, 0:size]
    face_mask = np.zeros((size, size), dtype=np.uint8)
    cv2.ellipse(face_mask, center, axes, 0, 0, 360, 255, -1)

    vertical_gradient = ((yy - size * 0.25) / max(size * 0.65, 1.0) * 26.0).astype(np.float32)
    horizontal_gradient = ((xx - size / 2.0) / max(size / 2.0, 1.0) * 12.0).astype(np.float32)
    shading = vertical_gradient + horizontal_gradient
    for channel in range(3):
        channel_data = img[:, :, channel].astype(np.float32)
        channel_data[face_mask > 0] += shading[face_mask > 0]
        img[:, :, channel] = np.clip(channel_data, 0, 255).astype(np.uint8)

    cv2.circle(img, (size // 2 - 28, size // 2 - 18), 10, (58, 68, 82), -1)
    cv2.circle(img, (size // 2 + 28, size // 2 - 18), 10, (58, 68, 82), -1)
    cv2.ellipse(img, (size // 2, size // 2 + 18), (12, 28), 0, 0, 180, (108, 140, 170), 2)
    cv2.ellipse(img, (size // 2, size // 2 + 52), (24, 10), 0, 0, 180, (88, 108, 130), 2)

    noise = np.random.default_rng(7).normal(0.0, 6.0, size=(size, size, 3))
    blended = img.astype(np.float32)
    blended[face_mask > 0] += noise[face_mask > 0]
    return np.clip(blended, 0, 255).astype(np.uint8)


def _screen_replay_frame() -> tuple[np.ndarray, dict, np.ndarray]:
    face = _live_like_face(176)
    frame = np.full((320, 320, 3), 6, dtype=np.uint8)
    x, y, w, h = 72, 68, 176, 176
    frame[y:y + h, x:x + w] = cv2.GaussianBlur(face, (7, 7), 1.2)

    for row in range(y, y + h, 6):
        frame[row:row + 1, x:x + w] = np.clip(frame[row:row + 1, x:x + w].astype(np.int16) + 28, 0, 255)
    cv2.rectangle(frame, (x - 20, y - 20), (x + w + 20, y + h + 20), (0, 0, 0), 14)
    cv2.rectangle(frame, (x + 18, y + 20), (x + 70, y + 48), (255, 255, 255), -1)
    crop = frame[y:y + h, x:x + w].copy()
    return frame, {"x": x, "y": y, "w": w, "h": h}, crop


class StillPadTests(unittest.TestCase):
    def test_live_like_face_is_not_classified_as_spoof(self):
        detector = LivenessDetector(model_path="models/does_not_exist.onnx")
        face = _live_like_face()

        result = detector.check_liveness(face)

        self.assertNotEqual(result["signal_state"], "spoof")
        self.assertGreater(result["score"], 0.35)
        self.assertEqual(result["backend"], "advanced_heuristic_cpu_pad")

    def test_screen_replay_with_border_is_flagged(self):
        detector = LivenessDetector(model_path="models/does_not_exist.onnx")
        frame, box, crop = _screen_replay_frame()

        result = detector.check_liveness(crop, frame_img=frame, face_box=box)

        self.assertEqual(result["signal_state"], "spoof")
        self.assertGreater(result["spoof_score"], 0.45)
        self.assertIn(result["attack_type"], {"replay_attack", "artifact_attack", "print_attack"})
        self.assertTrue(
            any(
                token in result["attack_indicators"]
                for token in ["row_column_banding", "context_screen_border", "advanced_pad:screen_display"]
            )
        )


if __name__ == "__main__":
    unittest.main()
