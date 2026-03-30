from __future__ import annotations

from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
from scipy.spatial import Delaunay

from src.core.contracts import ExpressionTransferRequest


class ExpressionDirector:
    PRESETS: Dict[str, Dict[str, Any]] = {
        "donor_expression": {
            "description": "Identity-preserving donor transfer with minimal semantic exaggeration.",
            "controls": {
                "mouth_widen": 0.00,
                "mouth_corner_lift": 0.00,
                "jaw_drop": 0.00,
                "eye_widen": 0.00,
                "eye_squint": 0.00,
                "brow_raise": 0.00,
                "brow_lower": 0.00,
                "smirk": 0.00,
                "micro_yaw_deg": 3.0,
                "micro_pitch_deg": 1.2,
                "micro_roll_deg": 1.8,
                "default_yaw_deg": None,
                "default_pitch_deg": None,
                "default_roll_deg": None,
            },
        },
        "subtle_smile": {
            "description": "A restrained smile with small mouth corner lift and gentle eye compression.",
            "controls": {
                "mouth_widen": 0.12,
                "mouth_corner_lift": 0.18,
                "jaw_drop": 0.05,
                "eye_widen": 0.00,
                "eye_squint": 0.08,
                "brow_raise": 0.01,
                "brow_lower": 0.00,
                "smirk": 0.00,
                "micro_yaw_deg": 3.8,
                "micro_pitch_deg": 1.4,
                "micro_roll_deg": 2.0,
                "default_yaw_deg": None,
                "default_pitch_deg": None,
                "default_roll_deg": 2.0,
            },
        },
        "laughing": {
            "description": "Wide laugh with stronger jaw drop, cheek lift, and livelier head motion.",
            "controls": {
                "mouth_widen": 0.28,
                "mouth_corner_lift": 0.30,
                "jaw_drop": 0.24,
                "eye_widen": 0.00,
                "eye_squint": 0.16,
                "brow_raise": 0.02,
                "brow_lower": 0.00,
                "smirk": 0.00,
                "micro_yaw_deg": 5.5,
                "micro_pitch_deg": 2.0,
                "micro_roll_deg": 3.5,
                "default_yaw_deg": None,
                "default_pitch_deg": None,
                "default_roll_deg": 5.0,
            },
        },
        "frightened": {
            "description": "Fear response with widened eyes, raised brows, and slight mouth opening.",
            "controls": {
                "mouth_widen": 0.05,
                "mouth_corner_lift": -0.04,
                "jaw_drop": 0.10,
                "eye_widen": 0.20,
                "eye_squint": 0.00,
                "brow_raise": 0.22,
                "brow_lower": 0.00,
                "smirk": 0.00,
                "micro_yaw_deg": 2.4,
                "micro_pitch_deg": 1.8,
                "micro_roll_deg": 1.4,
                "default_yaw_deg": 12.0,
                "default_pitch_deg": 3.0,
                "default_roll_deg": 0.0,
            },
        },
        "surprised": {
            "description": "Raised brows, open eyes, and a larger jaw drop for a surprise look.",
            "controls": {
                "mouth_widen": 0.08,
                "mouth_corner_lift": 0.00,
                "jaw_drop": 0.22,
                "eye_widen": 0.16,
                "eye_squint": 0.00,
                "brow_raise": 0.18,
                "brow_lower": 0.00,
                "smirk": 0.00,
                "micro_yaw_deg": 3.0,
                "micro_pitch_deg": 1.6,
                "micro_roll_deg": 1.0,
                "default_yaw_deg": 0.0,
                "default_pitch_deg": 2.0,
                "default_roll_deg": 0.0,
            },
        },
        "smirk": {
            "description": "Asymmetric half-smile with a controlled corner lift and mild eye compression.",
            "controls": {
                "mouth_widen": 0.08,
                "mouth_corner_lift": 0.10,
                "jaw_drop": 0.04,
                "eye_widen": 0.00,
                "eye_squint": 0.06,
                "brow_raise": 0.00,
                "brow_lower": 0.00,
                "smirk": 0.24,
                "micro_yaw_deg": 3.0,
                "micro_pitch_deg": 1.2,
                "micro_roll_deg": 2.2,
                "default_yaw_deg": -10.0,
                "default_pitch_deg": 0.0,
                "default_roll_deg": 2.0,
            },
        },
        "sideview_left": {
            "description": "Keeps the donor expression but turns the head toward a left profile review.",
            "controls": {
                "mouth_widen": 0.00,
                "mouth_corner_lift": 0.00,
                "jaw_drop": 0.00,
                "eye_widen": 0.00,
                "eye_squint": 0.00,
                "brow_raise": 0.00,
                "brow_lower": 0.00,
                "smirk": 0.00,
                "micro_yaw_deg": 2.0,
                "micro_pitch_deg": 1.0,
                "micro_roll_deg": 1.0,
                "default_yaw_deg": 32.0,
                "default_pitch_deg": 0.0,
                "default_roll_deg": 0.0,
            },
        },
        "sideview_right": {
            "description": "Keeps the donor expression but turns the head toward a right profile review.",
            "controls": {
                "mouth_widen": 0.00,
                "mouth_corner_lift": 0.00,
                "jaw_drop": 0.00,
                "eye_widen": 0.00,
                "eye_squint": 0.00,
                "brow_raise": 0.00,
                "brow_lower": 0.00,
                "smirk": 0.00,
                "micro_yaw_deg": 2.0,
                "micro_pitch_deg": 1.0,
                "micro_roll_deg": 1.0,
                "default_yaw_deg": -32.0,
                "default_pitch_deg": 0.0,
                "default_roll_deg": 0.0,
            },
        },
    }

    @classmethod
    def available_presets(cls) -> Dict[str, Dict[str, Any]]:
        return {
            name: {
                "description": spec["description"],
                "defaults": spec["controls"],
            }
            for name, spec in cls.PRESETS.items()
        }

    @staticmethod
    def _clip_float(value: Any, lo: float, hi: float, default: float) -> float:
        try:
            return float(np.clip(float(value), lo, hi))
        except Exception:
            return default

    @staticmethod
    def _emotion_suggestions(dominant_expression: str) -> List[str]:
        mapping = {
            "smile": ["donor_expression", "subtle_smile", "laughing", "sideview_left"],
            "surprise": ["surprised", "frightened", "sideview_left", "sideview_right"],
            "jaw_drop": ["surprised", "frightened", "donor_expression", "sideview_right"],
            "smirk": ["smirk", "subtle_smile", "sideview_left", "sideview_right"],
            "blink_or_squint": ["donor_expression", "smirk", "sideview_left", "sideview_right"],
        }
        return mapping.get(dominant_expression, ["donor_expression", "subtle_smile", "sideview_left", "sideview_right"])

    @classmethod
    def resolve_controls(
        cls,
        req: ExpressionTransferRequest,
        source_capture: Dict[str, Any],
        expression_capture: Dict[str, Any],
    ) -> Dict[str, Any]:
        preset_name = req.expression_preset or "donor_expression"
        preset = cls.PRESETS.get(preset_name, cls.PRESETS["donor_expression"])
        controls = dict(preset["controls"])

        target_yaw = req.target_yaw_deg if req.target_yaw_deg is not None else controls.get("default_yaw_deg")
        target_pitch = req.target_pitch_deg if req.target_pitch_deg is not None else controls.get("default_pitch_deg")
        target_roll = req.target_roll_deg if req.target_roll_deg is not None else controls.get("default_roll_deg")

        dominant_expression = str((expression_capture or {}).get("dominant_expression", "neutral"))
        recommended = cls._emotion_suggestions(dominant_expression)
        if preset_name not in recommended:
            recommended = [preset_name] + [name for name in recommended if name != preset_name]

        return {
            "preset": preset_name,
            "preset_description": preset["description"],
            "expression_strength": cls._clip_float(req.expression_strength, 0.0, 2.5, 1.0),
            "blink_strength": cls._clip_float(req.blink_strength, 0.0, 1.0, 0.15),
            "animation_mode": str(req.animation_mode or "cinematic"),
            "animation_frames": int(max(10, min(int(req.animation_frames or 18), 32))),
            "sideview_angle_deg": cls._clip_float(req.sideview_angle_deg, 10.0, 65.0, 30.0),
            "target_yaw_deg": None if target_yaw is None else float(target_yaw),
            "target_pitch_deg": None if target_pitch is None else float(target_pitch),
            "target_roll_deg": None if target_roll is None else float(target_roll),
            "semantic_controls": controls,
            "recommended_presets": recommended[:4],
            "source_dominant_expression": str((source_capture or {}).get("dominant_expression", "neutral")),
            "expression_dominant_expression": dominant_expression,
        }

    @staticmethod
    def cinematic_schedule(frame_count: int, mode: str = "cinematic") -> List[Dict[str, float]]:
        frame_count = max(10, frame_count)
        rows: List[Dict[str, float]] = []
        for idx, t in enumerate(np.linspace(0.0, 1.0, frame_count)):
            eased = 0.5 - 0.5 * np.cos(np.pi * t)
            if mode == "linear":
                alpha = eased
            else:
                overshoot = 0.09 * np.exp(-((t - 0.72) / 0.12) ** 2)
                settle = 0.02 * np.sin(np.pi * np.clip((t - 0.75) / 0.25, 0.0, 1.0))
                alpha = np.clip(eased + overshoot - settle, 0.0, 1.08)
            blink = max(0.0, 1.0 - abs(t - 0.58) / 0.08)
            rows.append(
                {
                    "frame": float(idx),
                    "alpha": float(alpha),
                    "blink": float(blink),
                    "yaw_wave": float(np.sin(2.0 * np.pi * t)),
                    "pitch_wave": float(np.sin(4.0 * np.pi * t + 0.5)),
                    "roll_wave": float(np.sin(2.0 * np.pi * t + 1.1)),
                    "expression_wave": float(0.5 - 0.5 * np.cos(2.0 * np.pi * t)),
                }
            )
        hold_rows = rows[-3:]
        return rows + hold_rows + list(reversed(rows[1:-1]))

    @staticmethod
    def expression_loop_schedule(frame_count: int) -> List[Dict[str, float]]:
        frame_count = max(10, frame_count)
        up = np.linspace(0.0, 1.0, frame_count)
        peak_hold = [1.0, 1.0, 1.0]
        down = np.linspace(1.0, 0.0, frame_count)[1:]
        stages = list(up) + peak_hold + list(down)
        rows: List[Dict[str, float]] = []
        total = max(len(stages) - 1, 1)
        for idx, alpha in enumerate(stages):
            t = idx / total
            blink = max(0.0, 1.0 - abs(t - 0.55) / 0.06)
            rows.append(
                {
                    "frame": float(idx),
                    "alpha": float(0.5 - 0.5 * np.cos(np.pi * alpha)),
                    "blink": float(blink),
                    "yaw_wave": 0.0,
                    "pitch_wave": 0.0,
                    "roll_wave": 0.0,
                    "expression_wave": float(alpha),
                }
            )
        return rows

    @staticmethod
    def teaser_schedule(frame_count: int) -> List[Dict[str, float]]:
        rows: List[Dict[str, float]] = []
        for idx, phase in enumerate(np.linspace(0.0, 2.0 * np.pi, frame_count, endpoint=False)):
            rows.append(
                {
                    "frame": float(idx),
                    "yaw_wave": float(np.sin(phase)),
                    "pitch_wave": float(np.sin(phase * 2.0)),
                    "roll_wave": float(np.cos(phase)),
                    "expression_wave": float(0.5 + 0.5 * np.sin(phase + 0.8)),
                    "blink": float(max(0.0, 1.0 - abs(np.sin(phase * 1.5)) * 2.5)),
                }
            )
        return rows

    @staticmethod
    def turntable_schedule(frame_count: int = 24) -> List[float]:
        frame_count = max(12, frame_count)
        return [float(v) for v in np.linspace(0.0, 360.0, frame_count, endpoint=False)]

    @staticmethod
    def _face_scale(landmarks: np.ndarray) -> float:
        left_eye = np.mean(landmarks[36:42], axis=0)
        right_eye = np.mean(landmarks[42:48], axis=0)
        return float(np.linalg.norm(right_eye - left_eye))

    @staticmethod
    def _shift(points: np.ndarray, indices: List[int], dx: float = 0.0, dy: float = 0.0) -> None:
        points[indices, 0] += dx
        points[indices, 1] += dy

    @classmethod
    def controlled_landmarks(
        cls,
        landmarks: np.ndarray,
        controls: Dict[str, Any],
        phase_strength: float,
        blink_scale: float,
    ) -> np.ndarray:
        if phase_strength <= 1e-5 and blink_scale <= 1e-5:
            return np.asarray(landmarks, dtype=np.float32).copy()

        points = np.asarray(landmarks, dtype=np.float32).copy()
        face_scale = max(cls._face_scale(points), 1.0)
        semantic = controls.get("semantic_controls", {}) or {}
        strength = float(controls.get("expression_strength", 1.0)) * float(phase_strength)

        mouth_widen = face_scale * 0.18 * float(semantic.get("mouth_widen", 0.0)) * strength
        jaw_drop = face_scale * 0.22 * float(semantic.get("jaw_drop", 0.0)) * strength
        mouth_lift = face_scale * 0.16 * float(semantic.get("mouth_corner_lift", 0.0)) * strength
        eye_widen = face_scale * 0.10 * float(semantic.get("eye_widen", 0.0)) * strength
        eye_squint = face_scale * 0.08 * float(semantic.get("eye_squint", 0.0)) * strength
        brow_raise = face_scale * 0.10 * float(semantic.get("brow_raise", 0.0)) * strength
        brow_lower = face_scale * 0.08 * float(semantic.get("brow_lower", 0.0)) * strength
        smirk = face_scale * 0.12 * float(semantic.get("smirk", 0.0)) * strength

        cls._shift(points, [48, 49, 59, 60], dx=-mouth_widen * 0.55)
        cls._shift(points, [54, 53, 55, 64], dx=mouth_widen * 0.55)
        cls._shift(points, [61, 62, 63], dy=-mouth_lift * 0.35)
        cls._shift(points, [48, 49, 50], dy=-mouth_lift * 0.65)
        cls._shift(points, [54, 53, 52], dy=-mouth_lift * 0.65)
        cls._shift(points, [57, 58, 59, 65, 66, 67, 8, 9, 10], dy=jaw_drop)
        cls._shift(points, [48, 49, 50], dy=-smirk)
        cls._shift(points, [54, 53, 52], dy=smirk * 0.35)

        cls._shift(points, [37, 38, 43, 44], dy=-(eye_widen + blink_scale * face_scale * 0.06))
        cls._shift(points, [40, 41, 46, 47], dy=(eye_widen + blink_scale * face_scale * 0.06))
        cls._shift(points, [37, 38, 43, 44], dy=eye_squint + blink_scale * face_scale * 0.10)
        cls._shift(points, [40, 41, 46, 47], dy=-(eye_squint + blink_scale * face_scale * 0.10))

        cls._shift(points, list(range(17, 27)), dy=-brow_raise)
        cls._shift(points, list(range(17, 27)), dy=brow_lower)

        return points

    @staticmethod
    def _triangle_warp(
        src: np.ndarray,
        dst: np.ndarray,
        tri_src: np.ndarray,
        tri_dst: np.ndarray,
    ) -> None:
        rect_src = cv2.boundingRect(np.float32([tri_src]))
        rect_dst = cv2.boundingRect(np.float32([tri_dst]))

        x1, y1, w1, h1 = rect_src
        x2, y2, w2, h2 = rect_dst
        if w1 <= 1 or h1 <= 1 or w2 <= 1 or h2 <= 1:
            return

        tri_src_rect = tri_src - np.array([x1, y1], dtype=np.float32)
        tri_dst_rect = tri_dst - np.array([x2, y2], dtype=np.float32)

        src_patch = src[y1:y1 + h1, x1:x1 + w1]
        warp_mat = cv2.getAffineTransform(np.float32(tri_src_rect), np.float32(tri_dst_rect))
        warped_patch = cv2.warpAffine(
            src_patch,
            warp_mat,
            (w2, h2),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )

        mask = np.zeros((h2, w2, 3), dtype=np.float32)
        cv2.fillConvexPoly(mask, np.int32(tri_dst_rect), (1.0, 1.0, 1.0), 16, 0)
        dst_region = dst[y2:y2 + h2, x2:x2 + w2].astype(np.float32)
        dst[y2:y2 + h2, x2:x2 + w2] = np.clip(
            dst_region * (1.0 - mask) + warped_patch.astype(np.float32) * mask,
            0,
            255,
        ).astype(np.uint8)

    @classmethod
    def warp_frame(
        cls,
        image: np.ndarray,
        landmarks: np.ndarray,
        controls: Dict[str, Any],
        phase_strength: float,
        blink_scale: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        src_points = np.asarray(landmarks, dtype=np.float32)
        dst_points = cls.controlled_landmarks(src_points, controls, phase_strength, blink_scale)
        h, w = image.shape[:2]
        border = np.array(
            [
                [0.0, 0.0],
                [w * 0.5, 0.0],
                [w - 1.0, 0.0],
                [w - 1.0, h * 0.5],
                [w - 1.0, h - 1.0],
                [w * 0.5, h - 1.0],
                [0.0, h - 1.0],
                [0.0, h * 0.5],
            ],
            dtype=np.float32,
        )
        src_full = np.vstack([src_points, border])
        dst_full = np.vstack([dst_points, border])
        tri = Delaunay(src_full).simplices

        output = image.copy()
        for simplex in tri:
            tri_src = src_full[simplex]
            tri_dst = dst_full[simplex]
            cls._triangle_warp(image, output, tri_src, tri_dst)
        return output, dst_points
