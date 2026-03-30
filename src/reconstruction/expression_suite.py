from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Callable, Dict, Optional

import cv2
import numpy as np
import torch
from PIL import Image

from src.core.contracts import ExpressionTransferRequest, ExpressionTransferResponse
from src.forensics.coefficient_analysis import CoefficientForensics
from src.reconstruction.expression_director import ExpressionDirector
from src.reconstruction.expression_transfer import Deep3DExpressionTransferService
from src.reconstruction.deep3d_recon import Deep3DFaceReconstructor

logger = logging.getLogger("ca_monk.expression_suite")


class Deep3DExpressionSuiteService(Deep3DExpressionTransferService):
    """
    Extended expression workflow inspired by DECA's demo surface.

    It adds:
      - per-image expression capture summaries
      - interpolated expression animation
      - teaser-style reposing animation
    while leaving the current verification pipeline logic unchanged.
    """

    def __init__(
        self,
        deep3d_provider: Optional[Callable[[], Deep3DFaceReconstructor]] = None,
        output_dir: str = "evidence_cards",
    ) -> None:
        super().__init__(deep3d_provider=deep3d_provider, output_dir=output_dir)

    def capabilities(self) -> Dict[str, Any]:
        caps = super().capabilities()
        caps["method"] = "deep3d_bfm_expression_suite"
        caps["supports"] = [
            "expression_capture",
            "expression_transfer",
            "expression_animation",
            "teaser_repose_animation",
            "turntable_review",
            "directed_expression_presets",
            "pose_targeting",
            "preset_gallery",
        ]
        caps["animation_default"] = "cinematic"
        caps["available_presets"] = ExpressionDirector.available_presets()
        return caps

    @staticmethod
    def _with_ext(path: str, suffix: str, ext: str) -> str:
        root, _ = os.path.splitext(path)
        return f"{root}{suffix}{ext}"

    @staticmethod
    def _write_json(path: str, payload: Dict[str, Any]) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)

    @staticmethod
    def _distance(landmarks: np.ndarray, a: int, b: int) -> float:
        return float(np.linalg.norm(landmarks[a] - landmarks[b]))

    @staticmethod
    def _safe_ratio(num: float, den: float) -> float:
        den = float(den)
        if abs(den) < 1e-6:
            return 0.0
        return float(num) / den

    def _semantic_metrics(self, landmarks: np.ndarray) -> Dict[str, float]:
        lm = np.asarray(landmarks, dtype=np.float32)
        left_eye_width = self._distance(lm, 36, 39)
        right_eye_width = self._distance(lm, 42, 45)
        mouth_width = self._distance(lm, 48, 54)
        nose_width = self._distance(lm, 31, 35)

        left_eye_open = self._safe_ratio(
            (self._distance(lm, 37, 41) + self._distance(lm, 38, 40)) * 0.5,
            left_eye_width,
        )
        right_eye_open = self._safe_ratio(
            (self._distance(lm, 43, 47) + self._distance(lm, 44, 46)) * 0.5,
            right_eye_width,
        )
        mouth_open = self._safe_ratio(
            (self._distance(lm, 62, 66) + self._distance(lm, 63, 65)) * 0.5,
            mouth_width,
        )
        smile_ratio = self._safe_ratio(mouth_width, max(nose_width, 1.0))

        left_brow_y = float(np.mean(lm[17:22, 1]))
        right_brow_y = float(np.mean(lm[22:27, 1]))
        left_eye_y = float(np.mean(lm[36:42, 1]))
        right_eye_y = float(np.mean(lm[42:48, 1]))
        brow_lift_left = self._safe_ratio(left_eye_y - left_brow_y, left_eye_width)
        brow_lift_right = self._safe_ratio(right_eye_y - right_brow_y, right_eye_width)

        mouth_corner_tilt = self._safe_ratio(
            abs(float(lm[48, 1] - lm[54, 1])),
            mouth_width,
        )
        asymmetry = (
            abs(left_eye_open - right_eye_open)
            + abs(brow_lift_left - brow_lift_right)
            + mouth_corner_tilt
        )

        return {
            "left_eye_open": round(left_eye_open, 4),
            "right_eye_open": round(right_eye_open, 4),
            "mouth_open": round(mouth_open, 4),
            "smile_ratio": round(smile_ratio, 4),
            "brow_lift_left": round(brow_lift_left, 4),
            "brow_lift_right": round(brow_lift_right, 4),
            "mouth_corner_tilt": round(mouth_corner_tilt, 4),
            "asymmetry": round(asymmetry, 4),
        }

    @staticmethod
    def _dominant_expression(
        semantic: Dict[str, float],
        coeff_analysis: Dict[str, Any],
    ) -> str:
        mouth_open = float(semantic.get("mouth_open", 0.0))
        smile_ratio = float(semantic.get("smile_ratio", 0.0))
        brow_avg = (
            float(semantic.get("brow_lift_left", 0.0))
            + float(semantic.get("brow_lift_right", 0.0))
        ) * 0.5
        eye_avg = (
            float(semantic.get("left_eye_open", 0.0))
            + float(semantic.get("right_eye_open", 0.0))
        ) * 0.5
        asymmetry = float(semantic.get("asymmetry", 0.0))
        intensity = str(
            ((coeff_analysis.get("expression_analysis", {}) or {}).get("intensity", "neutral"))
        )

        if mouth_open > 0.28 and brow_avg > 0.34:
            return "surprise"
        if smile_ratio > 1.75 and mouth_open > 0.06:
            return "smile"
        if mouth_open > 0.26:
            return "jaw_drop"
        if eye_avg < 0.16:
            return "blink_or_squint"
        if asymmetry > 0.22 and smile_ratio > 1.55:
            return "smirk"
        if intensity == "neutral":
            return "neutral"
        return "subtle_expression"

    def _capture_summary(self, result: Dict[str, Any], label: str) -> Dict[str, Any]:
        coeffs = np.asarray(result["coefficients"], dtype=np.float32)
        coeff_dict = result.get("coeff_dict", {}) or {}
        coeff_analysis = CoefficientForensics().analyze(coeffs, coeff_dict=coeff_dict)
        semantic = self._semantic_metrics(result["landmarks_68"])
        dominant = self._dominant_expression(semantic, coeff_analysis)
        top_exp = np.asarray(coeff_dict.get("exp", coeffs[80:144]), dtype=np.float32)
        top_idx = np.argsort(np.abs(top_exp))[::-1][:8]
        top_components = [
            {"index": int(idx), "value": round(float(top_exp[idx]), 4)}
            for idx in top_idx
        ]

        return {
            "label": label,
            "dominant_expression": dominant,
            "intensity": str(
                ((coeff_analysis.get("expression_analysis", {}) or {}).get("intensity", "neutral"))
            ),
            "pose": coeff_analysis.get("pose", {}),
            "semantic_metrics": semantic,
            "expression_analysis": coeff_analysis.get("expression_analysis", {}),
            "top_expression_components": top_components,
        }

    def _draw_metric_bar(
        self,
        canvas: np.ndarray,
        label: str,
        value: float,
        max_value: float,
        x: int,
        y: int,
        width: int,
    ) -> None:
        cv2.putText(
            canvas,
            f"{label}: {value:.3f}",
            (x, y - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (220, 235, 245),
            1,
            cv2.LINE_AA,
        )
        cv2.rectangle(canvas, (x, y), (x + width, y + 18), (34, 54, 74), thickness=-1)
        fill = int(width * np.clip(value / max(max_value, 1e-6), 0.0, 1.0))
        cv2.rectangle(canvas, (x, y), (x + fill, y + 18), (125, 211, 252), thickness=-1)

    def _render_capture_card(
        self,
        result: Dict[str, Any],
        summary: Dict[str, Any],
    ) -> np.ndarray:
        canvas = np.full((720, 1280, 3), (10, 18, 28), dtype=np.uint8)
        aligned = cv2.resize(result["aligned_input"], (360, 360), interpolation=cv2.INTER_LANCZOS4)
        rendered = cv2.resize(result["rendered"], (360, 360), interpolation=cv2.INTER_LANCZOS4)
        canvas[40:400, 40:400] = self._label_tile(aligned, "Aligned Input")
        canvas[320:680, 40:400] = self._label_tile(rendered, "3D Render")

        cv2.putText(
            canvas,
            f"{summary['label']} Expression Capture",
            (440, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.95,
            (244, 247, 250),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            f"Dominant: {summary['dominant_expression']}  |  Intensity: {summary['intensity']}",
            (440, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (125, 211, 252),
            2,
            cv2.LINE_AA,
        )

        pose = summary.get("pose", {}) or {}
        cv2.putText(
            canvas,
            (
                f"Pose  yaw={pose.get('yaw_deg', 0.0)}  "
                f"pitch={pose.get('pitch_deg', 0.0)}  roll={pose.get('roll_deg', 0.0)}"
            ),
            (440, 145),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (184, 201, 216),
            1,
            cv2.LINE_AA,
        )

        semantic = summary.get("semantic_metrics", {}) or {}
        metrics = [
            ("Mouth Open", float(semantic.get("mouth_open", 0.0)), 0.40),
            ("Smile Ratio", float(semantic.get("smile_ratio", 0.0)), 2.20),
            ("Brow Lift L", float(semantic.get("brow_lift_left", 0.0)), 0.55),
            ("Brow Lift R", float(semantic.get("brow_lift_right", 0.0)), 0.55),
            ("Eye Open L", float(semantic.get("left_eye_open", 0.0)), 0.35),
            ("Eye Open R", float(semantic.get("right_eye_open", 0.0)), 0.35),
            ("Asymmetry", float(semantic.get("asymmetry", 0.0)), 0.35),
        ]
        y = 205
        for label, value, max_value in metrics:
            self._draw_metric_bar(canvas, label, value, max_value, 440, y, 430)
            y += 48

        cv2.putText(
            canvas,
            "Top Expression Coefficients",
            (910, 205),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (244, 247, 250),
            2,
            cv2.LINE_AA,
        )
        top_components = summary.get("top_expression_components", []) or []
        y = 245
        for component in top_components:
            idx = int(component.get("index", 0))
            val = float(component.get("value", 0.0))
            magnitude = min(int(abs(val) / 4.0 * 260), 260)
            cv2.putText(
                canvas,
                f"exp[{idx:02d}] {val:+.4f}",
                (910, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (220, 235, 245),
                1,
                cv2.LINE_AA,
            )
            cv2.rectangle(canvas, (1080, y - 14), (1080 + magnitude, y), (125, 211, 252), thickness=-1)
            y += 42

        return canvas

    @torch.no_grad()
    def _render_coefficients(
        self,
        base_result: Dict[str, Any],
        coefficients: np.ndarray,
        output_size: int = 512,
    ) -> Dict[str, Any]:
        coeff_tensor = torch.from_numpy(np.asarray(coefficients, dtype=np.float32)).unsqueeze(0).to(self.deep3d.device)
        recon = self.deep3d.bfm.reconstruct(coeff_tensor)

        verts_cam = recon["face_vertex"][0].detach().cpu().numpy()
        colors = recon["face_color"][0].detach().cpu().numpy()
        normals = recon["face_norm"][0].detach().cpu().numpy()
        face_buf = recon["face_buf"]
        if isinstance(face_buf, torch.Tensor):
            face_buf = face_buf.detach().cpu().numpy()

        render_224 = self.deep3d.renderer.render(
            verts_cam, face_buf, colors, normals, output_size=224
        )
        render_main = self.deep3d.renderer.render(
            verts_cam, face_buf, colors, normals, output_size=output_size
        )
        side_view = self.deep3d.renderer.render_rotated(
            verts_cam,
            face_buf,
            colors,
            normals,
            angle_y_deg=30.0,
            output_size=224,
        )

        depth_colored = cv2.applyColorMap(render_main["depth"], cv2.COLORMAP_INFERNO)
        depth_colored[render_main["mask"] == 0] = 0

        aligned_source = cv2.resize(
            base_result["aligned_input"],
            (output_size, output_size),
            interpolation=cv2.INTER_LANCZOS4,
        )
        mask_feathered = self._feather_mask(render_main["mask"])
        mask_3ch = np.stack([mask_feathered] * 3, axis=-1)
        overlay = (
            render_main["rendered"].astype(np.float32) * mask_3ch
            + aligned_source.astype(np.float32) * (1.0 - mask_3ch)
        )
        overlay = np.clip(overlay, 0, 255).astype(np.uint8)

        return {
            "rendered": render_main["rendered"],
            "rendered_224": render_224["rendered"],
            "overlay": overlay,
            "depth_colored": depth_colored,
            "geometry": render_main["geometry"],
            "normal_map": render_main["normal_map"],
            "side_view": side_view["rendered"],
            "verts_cam": verts_cam,
            "face_buf": face_buf,
            "face_colors": colors,
            "face_normals": normals,
            "mesh_vertices": recon["face_shape"][0].detach().cpu().numpy(),
            "mesh_faces": face_buf,
            "mesh_colors": colors,
            "landmarks_68": recon["landmarks_2d"][0].detach().cpu().numpy(),
            "coefficients": np.asarray(coefficients, dtype=np.float32),
            "coeff_dict": {
                key: value[0].detach().cpu().numpy()
                for key, value in recon["coeff_dict"].items()
            },
        }

    @staticmethod
    def _blend_coefficients(
        source_coeffs: np.ndarray,
        expression_coeffs: np.ndarray,
        alpha: float,
        transfer_pose: bool,
    ) -> np.ndarray:
        merged = np.asarray(source_coeffs, dtype=np.float32).copy()
        merged[80:144] = (
            (1.0 - alpha) * source_coeffs[80:144] + alpha * expression_coeffs[80:144]
        )
        if transfer_pose:
            merged[224:227] = (
                (1.0 - alpha) * source_coeffs[224:227] + alpha * expression_coeffs[224:227]
            )
        return merged

    @staticmethod
    def _animation_schedule(frame_count: int = 14) -> list[float]:
        forward = [0.5 - 0.5 * np.cos(np.pi * t) for t in np.linspace(0.0, 1.0, frame_count)]
        backward = list(reversed(forward[1:-1]))
        return [float(v) for v in forward + backward]

    @staticmethod
    def _save_gif(frames: list[np.ndarray], path: str, duration_ms: int = 70) -> None:
        pil_frames = [
            Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            for frame in frames
        ]
        pil_frames[0].save(
            path,
            save_all=True,
            append_images=pil_frames[1:],
            duration=duration_ms,
            loop=0,
            optimize=False,
        )

    def _save_keyframe_sheet(
        self,
        frames: list[np.ndarray],
        path: str,
        title: str,
        tile_size: int = 256,
        columns: int = 4,
    ) -> None:
        if not frames:
            return
        total = min(len(frames), 8)
        picks = np.linspace(0, len(frames) - 1, total).astype(int).tolist()
        tiles = []
        for idx in picks:
            tile = cv2.resize(frames[idx], (tile_size, tile_size), interpolation=cv2.INTER_AREA)
            tile = self._label_tile(tile, f"frame {idx:02d}")
            tiles.append(tile)

        rows = []
        for start in range(0, len(tiles), columns):
            row_tiles = tiles[start:start + columns]
            while len(row_tiles) < columns:
                row_tiles.append(np.zeros_like(tiles[0]))
            rows.append(np.concatenate(row_tiles, axis=1))
        sheet = np.concatenate(rows, axis=0)
        banner = np.full((54, sheet.shape[1], 3), (8, 18, 30), dtype=np.uint8)
        cv2.putText(
            banner,
            title,
            (16, 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (244, 247, 250),
            2,
            cv2.LINE_AA,
        )
        cv2.imwrite(path, np.concatenate([banner, sheet], axis=0))

    @staticmethod
    def _apply_target_pose(
        coeffs: np.ndarray,
        controls: Dict[str, Any],
        schedule_row: Optional[Dict[str, float]] = None,
    ) -> np.ndarray:
        updated = np.asarray(coeffs, dtype=np.float32).copy()
        target_pitch = controls.get("target_pitch_deg")
        target_yaw = controls.get("target_yaw_deg")
        target_roll = controls.get("target_roll_deg")
        if target_pitch is not None:
            updated[224] = np.radians(float(target_pitch))
        if target_yaw is not None:
            updated[225] = np.radians(float(target_yaw))
        if target_roll is not None:
            updated[226] = np.radians(float(target_roll))
        if schedule_row:
            semantic = controls.get("semantic_controls", {}) or {}
            updated[224] += np.radians(float(schedule_row.get("pitch_wave", 0.0)) * float(semantic.get("micro_pitch_deg", 0.0)))
            updated[225] += np.radians(float(schedule_row.get("yaw_wave", 0.0)) * float(semantic.get("micro_yaw_deg", 0.0)))
            updated[226] += np.radians(float(schedule_row.get("roll_wave", 0.0)) * float(semantic.get("micro_roll_deg", 0.0)))
        return updated

    def _render_controlled_frame(
        self,
        source_result: Dict[str, Any],
        expression_result: Dict[str, Any],
        transfer_pose: bool,
        controls: Dict[str, Any],
        alpha: float,
        output_size: int,
        schedule_row: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        source_coeffs = np.asarray(source_result["coefficients"], dtype=np.float32)
        expression_coeffs = np.asarray(expression_result["coefficients"], dtype=np.float32)
        coeffs = self._blend_coefficients(
            source_coeffs,
            expression_coeffs,
            alpha=float(np.clip(alpha * float(controls.get("expression_strength", 1.0)), 0.0, 1.35)),
            transfer_pose=transfer_pose,
        )
        coeffs = self._apply_target_pose(coeffs, controls, schedule_row=schedule_row)
        render = self._render_coefficients(source_result, coeffs, output_size=output_size)
        warp_strength = float(schedule_row.get("expression_wave", alpha)) if schedule_row else float(alpha)
        blink_strength = float(controls.get("blink_strength", 0.0)) * float(schedule_row.get("blink", 0.0) if schedule_row else 0.0)
        render["overlay"], warped_landmarks = ExpressionDirector.warp_frame(
            render["overlay"],
            render["landmarks_68"],
            controls,
            phase_strength=warp_strength,
            blink_scale=blink_strength,
        )
        render["rendered"], _ = ExpressionDirector.warp_frame(
            render["rendered"],
            render["landmarks_68"],
            controls,
            phase_strength=warp_strength,
            blink_scale=blink_strength,
        )
        render["landmarks_68"] = warped_landmarks
        side_angle = controls.get("target_yaw_deg")
        if side_angle is None or abs(float(side_angle)) < 10.0:
            side_angle = controls.get("sideview_angle_deg", 30.0)
        side_view = self.deep3d.renderer.render_rotated(
            render["verts_cam"],
            render["face_buf"],
            render["face_colors"],
            render["face_normals"],
            angle_y_deg=float(side_angle),
            output_size=224,
        )
        render["side_view"] = side_view["rendered"]
        render["preview_strip"] = self._build_preview_strip(
            source_result["aligned_input"],
            expression_result["aligned_input"],
            cv2.resize(render["rendered"], (224, 224), interpolation=cv2.INTER_AREA),
        )
        render["metadata"] = {
            "transfer_pose": bool(transfer_pose),
            "source_expression_norm": round(float(np.linalg.norm(source_coeffs[80:144])), 4),
            "expression_source_norm": round(float(np.linalg.norm(expression_coeffs[80:144])), 4),
            "transferred_expression_norm": round(float(np.linalg.norm(coeffs[80:144])), 4),
            "source_angles": source_coeffs[224:227].tolist(),
            "expression_angles": expression_coeffs[224:227].tolist(),
            "transferred_angles": coeffs[224:227].tolist(),
            "active_controls": controls,
        }
        return render

    def _build_preset_gallery(
        self,
        source_result: Dict[str, Any],
        expression_result: Dict[str, Any],
        transfer_pose: bool,
        save_path: str,
        base_controls: Dict[str, Any],
    ) -> Dict[str, Any]:
        gallery_json_path = self._with_ext(save_path, "_preset_gallery", ".json")
        gallery_image_path = self._with_suffix(save_path, "_preset_gallery")

        preview_presets = list(dict.fromkeys(
            [
                base_controls.get("preset", "donor_expression"),
                *base_controls.get("recommended_presets", []),
                "laughing",
                "frightened",
                "sideview_left",
                "sideview_right",
            ]
        ))[:6]

        tiles: list[np.ndarray] = []
        entries: list[Dict[str, Any]] = []
        for preset_name in preview_presets:
            preset_controls = dict(base_controls)
            preset_controls.update(
                ExpressionDirector.resolve_controls(
                    ExpressionTransferRequest(
                        source_image_path="source",
                        expression_image_path="expression",
                        expression_preset=preset_name,
                        expression_strength=base_controls.get("expression_strength", 1.0),
                        blink_strength=base_controls.get("blink_strength", 0.15),
                        animation_mode=base_controls.get("animation_mode", "cinematic"),
                        animation_frames=base_controls.get("animation_frames", 18),
                        sideview_angle_deg=base_controls.get("sideview_angle_deg", 30.0),
                    ),
                    source_capture={"dominant_expression": base_controls.get("source_dominant_expression", "neutral")},
                    expression_capture={"dominant_expression": base_controls.get("expression_dominant_expression", "neutral")},
                )
            )
            frame = self._render_controlled_frame(
                source_result=source_result,
                expression_result=expression_result,
                transfer_pose=transfer_pose,
                controls=preset_controls,
                alpha=1.0,
                output_size=320,
            )
            tile = cv2.resize(frame["overlay"], (320, 320), interpolation=cv2.INTER_AREA)
            label = preset_name.replace("_", " ")
            yaw = preset_controls.get("target_yaw_deg")
            if yaw is not None:
                label = f"{label} | yaw {float(yaw):.0f}"
            tiles.append(self._label_tile(tile, label))
            entries.append(
                {
                    "preset": preset_name,
                    "target_yaw_deg": preset_controls.get("target_yaw_deg"),
                    "target_pitch_deg": preset_controls.get("target_pitch_deg"),
                    "target_roll_deg": preset_controls.get("target_roll_deg"),
                    "expression_strength": preset_controls.get("expression_strength"),
                    "description": preset_controls.get("preset_description"),
                }
            )

        rows = []
        for start in range(0, len(tiles), 3):
            row_tiles = tiles[start:start + 3]
            while len(row_tiles) < 3:
                row_tiles.append(np.zeros_like(tiles[0]))
            rows.append(np.concatenate(row_tiles, axis=1))
        sheet = np.concatenate(rows, axis=0)
        banner = np.full((54, sheet.shape[1], 3), (8, 18, 30), dtype=np.uint8)
        cv2.putText(
            banner,
            "Directed Expression Presets",
            (16, 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (244, 247, 250),
            2,
            cv2.LINE_AA,
        )
        cv2.imwrite(gallery_image_path, np.concatenate([banner, sheet], axis=0))
        self._write_json(
            gallery_json_path,
            {
                "active_controls": base_controls,
                "presets": entries,
                "available_presets": ExpressionDirector.available_presets(),
            },
        )
        return {
            "preset_gallery_image_path": gallery_image_path,
            "preset_gallery_json_path": gallery_json_path,
        }

    def _build_animation_artifacts(
        self,
        source_result: Dict[str, Any],
        expression_result: Dict[str, Any],
        transfer_pose: bool,
        controls: Dict[str, Any],
        save_path: str,
    ) -> Dict[str, Any]:
        expression_controls = dict(controls)
        expression_semantic = dict(expression_controls.get("semantic_controls", {}) or {})
        expression_semantic["micro_yaw_deg"] = 0.0
        expression_semantic["micro_pitch_deg"] = 0.0
        expression_semantic["micro_roll_deg"] = 0.0
        expression_controls["semantic_controls"] = expression_semantic

        animation_frames: list[np.ndarray] = []
        expression_schedule = ExpressionDirector.expression_loop_schedule(
            frame_count=int(controls.get("animation_frames", 18)),
        )
        for row in expression_schedule:
            render = self._render_controlled_frame(
                source_result=source_result,
                expression_result=expression_result,
                transfer_pose=transfer_pose,
                controls=expression_controls,
                alpha=float(row.get("alpha", 1.0)),
                output_size=384,
                schedule_row=row,
            )
            label = f"{controls.get('preset', 'donor_expression').replace('_', ' ')} expression"
            animation_frames.append(self._label_tile(render["rendered"], label))

        animation_gif_path = self._with_ext(save_path, "_animation", ".gif")
        animation_keyframes_path = self._with_suffix(save_path, "_animation_keyframes")
        self._save_gif(animation_frames, animation_gif_path, duration_ms=96)
        self._save_keyframe_sheet(
            animation_frames,
            animation_keyframes_path,
            "Expression Loop Keyframes",
        )

        teaser_frames: list[np.ndarray] = []
        teaser_schedule = ExpressionDirector.teaser_schedule(frame_count=20)
        base_controls = dict(controls)
        side_angle = float(base_controls.get("sideview_angle_deg", 30.0))
        if base_controls.get("target_yaw_deg") is None:
            base_controls["target_yaw_deg"] = side_angle
        for row in teaser_schedule:
            render = self._render_controlled_frame(
                source_result=source_result,
                expression_result=expression_result,
                transfer_pose=transfer_pose,
                controls=base_controls,
                alpha=0.9 + 0.1 * float(row.get("expression_wave", 0.0)),
                output_size=384,
                schedule_row={
                    "yaw_wave": float(row.get("yaw_wave", 0.0)) * 0.45,
                    "pitch_wave": float(row.get("pitch_wave", 0.0)) * 0.35,
                    "roll_wave": float(row.get("roll_wave", 0.0)) * 0.30,
                    "expression_wave": 0.65 + 0.35 * float(row.get("expression_wave", 0.0)),
                    "blink": float(row.get("blink", 0.0)),
                },
            )
            teaser_label = f"profile swing | yaw {float(base_controls.get('target_yaw_deg', side_angle)):.0f}"
            teaser_frames.append(self._label_tile(render["rendered"], teaser_label))

        teaser_gif_path = self._with_ext(save_path, "_teaser", ".gif")
        teaser_keyframes_path = self._with_suffix(save_path, "_teaser_keyframes")
        self._save_gif(teaser_frames, teaser_gif_path, duration_ms=92)
        self._save_keyframe_sheet(
            teaser_frames,
            teaser_keyframes_path,
            "Profile Swing Keyframes",
        )

        turntable_render = self._render_controlled_frame(
            source_result=source_result,
            expression_result=expression_result,
            transfer_pose=transfer_pose,
            controls=expression_controls,
            alpha=1.0,
            output_size=384,
        )
        turntable_frames: list[np.ndarray] = []
        for angle in ExpressionDirector.turntable_schedule(frame_count=24):
            turn = self.deep3d.renderer.render_rotated(
                turntable_render["verts_cam"],
                turntable_render["face_buf"],
                turntable_render["face_colors"],
                turntable_render["face_normals"],
                angle_y_deg=float(angle),
                output_size=384,
            )
            turntable_frames.append(self._label_tile(turn["rendered"], f"turntable {int(angle):03d} deg"))

        turntable_gif_path = self._with_ext(save_path, "_turntable", ".gif")
        turntable_keyframes_path = self._with_suffix(save_path, "_turntable_keyframes")
        self._save_gif(turntable_frames, turntable_gif_path, duration_ms=92)
        self._save_keyframe_sheet(
            turntable_frames,
            turntable_keyframes_path,
            "Turntable 360 Keyframes",
        )

        return {
            "animation_gif_path": animation_gif_path,
            "animation_keyframes_path": animation_keyframes_path,
            "teaser_gif_path": teaser_gif_path,
            "teaser_keyframes_path": teaser_keyframes_path,
            "turntable_gif_path": turntable_gif_path,
            "turntable_keyframes_path": turntable_keyframes_path,
        }

    def generate(self, req: ExpressionTransferRequest) -> ExpressionTransferResponse:
        save_path = req.evidence_save_path or self._default_save_path(req)
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        warnings: list[str] = []
        t0 = time.time()

        source_image = cv2.imread(req.source_image_path)
        if source_image is None:
            return ExpressionTransferResponse(
                generated_image_path=None,
                warnings=[f"source_image_not_readable: {req.source_image_path}"],
            )
        expression_image = cv2.imread(req.expression_image_path)
        if expression_image is None:
            return ExpressionTransferResponse(
                generated_image_path=None,
                warnings=[f"expression_image_not_readable: {req.expression_image_path}"],
            )

        try:
            source_result = self.deep3d.reconstruct(source_image)
            expression_result = self.deep3d.reconstruct(expression_image)
        except FileNotFoundError as exc:
            return ExpressionTransferResponse(
                generated_image_path=None,
                warnings=[f"deep3d_model_missing: {str(exc)[:160]}"],
            )
        except Exception as exc:
            return ExpressionTransferResponse(
                generated_image_path=None,
                warnings=[f"expression_suite_failed: {str(exc)[:160]}"],
            )

        if source_result is None:
            return ExpressionTransferResponse(
                generated_image_path=None,
                warnings=["identity_source_face_not_detected"],
            )
        if expression_result is None:
            return ExpressionTransferResponse(
                generated_image_path=None,
                warnings=["expression_source_face_not_detected"],
            )

        source_capture = self._capture_summary(source_result, "Identity Source")
        expression_capture = self._capture_summary(expression_result, "Expression Source")
        source_capture_image = self._render_capture_card(source_result, source_capture)
        expression_capture_image = self._render_capture_card(expression_result, expression_capture)

        source_capture_json_path = self._with_ext(save_path, "_source_capture", ".json")
        source_capture_image_path = self._with_suffix(save_path, "_source_capture")
        expression_capture_json_path = self._with_ext(save_path, "_expression_capture", ".json")
        expression_capture_image_path = self._with_suffix(save_path, "_expression_capture")
        self._write_json(source_capture_json_path, source_capture)
        self._write_json(expression_capture_json_path, expression_capture)
        cv2.imwrite(source_capture_image_path, source_capture_image)
        cv2.imwrite(expression_capture_image_path, expression_capture_image)

        controls = ExpressionDirector.resolve_controls(
            req,
            source_capture=source_capture,
            expression_capture=expression_capture,
        )

        transfer_result = self._render_controlled_frame(
            source_result=source_result,
            expression_result=expression_result,
            transfer_pose=req.transfer_pose,
            controls=controls,
            alpha=1.0,
            output_size=512,
        )

        overlay_path = save_path
        rendered_path = self._with_suffix(save_path, "_rendered")
        preview_path = self._with_suffix(save_path, "_preview")
        depth_path = self._with_suffix(save_path, "_depth")
        geometry_path = self._with_suffix(save_path, "_geometry")
        normal_map_path = self._with_suffix(save_path, "_normals")
        side_view_path = self._with_suffix(save_path, "_sideview")
        mesh_path = self._with_ext(save_path, "_mesh", ".obj")

        cv2.imwrite(overlay_path, transfer_result["overlay"])
        cv2.imwrite(rendered_path, transfer_result["rendered"])
        cv2.imwrite(preview_path, transfer_result["preview_strip"])
        cv2.imwrite(depth_path, transfer_result["depth_colored"])
        cv2.imwrite(geometry_path, transfer_result["geometry"])
        cv2.imwrite(normal_map_path, transfer_result["normal_map"])
        cv2.imwrite(side_view_path, transfer_result["side_view"])
        self.deep3d.save_obj(transfer_result, mesh_path)

        preset_gallery_paths = self._build_preset_gallery(
            source_result=source_result,
            expression_result=expression_result,
            transfer_pose=req.transfer_pose,
            save_path=save_path,
            base_controls=controls,
        )

        animation_paths = self._build_animation_artifacts(
            source_result=source_result,
            expression_result=expression_result,
            transfer_pose=req.transfer_pose,
            controls=controls,
            save_path=save_path,
        )

        elapsed = time.time() - t0
        metadata = dict(transfer_result["metadata"])
        metadata.update(
            {
                "source_image_path": req.source_image_path,
                "expression_image_path": req.expression_image_path,
                "elapsed_ms": round(elapsed * 1000.0, 2),
                "source_capture": source_capture,
                "expression_capture": expression_capture,
                "expression_controls": controls,
            }
        )
        suite_json_path = self._with_ext(save_path, "_suite", ".json")
        self._write_json(
            suite_json_path,
            {
                "metadata": metadata,
                "paths": {
                    "generated_image_path": overlay_path,
                    "rendered_image_path": rendered_path,
                    "preview_image_path": preview_path,
                    "depth_map_path": depth_path,
                    "geometry_image_path": geometry_path,
                    "normal_map_path": normal_map_path,
                    "side_view_image_path": side_view_path,
                    "mesh_path": mesh_path,
                    "source_capture_json_path": source_capture_json_path,
                    "source_capture_image_path": source_capture_image_path,
                    "expression_capture_json_path": expression_capture_json_path,
                    "expression_capture_image_path": expression_capture_image_path,
                    **preset_gallery_paths,
                    **animation_paths,
                },
            },
        )

        warnings.extend(
            [
                "deep3d_expression_capture_complete",
                "deep3d_expression_transfer_complete",
                "deep3d_directed_expression_controls_complete",
                "deep3d_expression_animation_complete",
                "deep3d_teaser_animation_complete",
                "deep3d_turntable_complete",
                f"processing_time_{elapsed:.1f}s",
            ]
        )

        logger.info(
            "Expression suite complete in %.1fs: source=%s expression=%s pose=%s",
            elapsed,
            req.source_image_path,
            req.expression_image_path,
            req.transfer_pose,
        )

        return ExpressionTransferResponse(
            generated_image_path=overlay_path,
            rendered_image_path=rendered_path,
            preview_image_path=preview_path,
            depth_map_path=depth_path,
            geometry_image_path=geometry_path,
            normal_map_path=normal_map_path,
            side_view_image_path=side_view_path,
            mesh_path=mesh_path,
            animation_gif_path=animation_paths["animation_gif_path"],
            animation_keyframes_path=animation_paths["animation_keyframes_path"],
            teaser_gif_path=animation_paths["teaser_gif_path"],
            teaser_keyframes_path=animation_paths["teaser_keyframes_path"],
            turntable_gif_path=animation_paths["turntable_gif_path"],
            turntable_keyframes_path=animation_paths["turntable_keyframes_path"],
            source_capture_json_path=source_capture_json_path,
            source_capture_image_path=source_capture_image_path,
            expression_capture_json_path=expression_capture_json_path,
            expression_capture_image_path=expression_capture_image_path,
            preset_gallery_image_path=preset_gallery_paths["preset_gallery_image_path"],
            preset_gallery_json_path=preset_gallery_paths["preset_gallery_json_path"],
            suite_json_path=suite_json_path,
            warnings=warnings,
            metadata=metadata,
        )
