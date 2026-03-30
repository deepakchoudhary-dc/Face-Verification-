from __future__ import annotations

import logging
import os
import time
from datetime import datetime
from typing import Any, Callable, Dict, Optional

import cv2
import numpy as np
import torch

from src.core.contracts import ExpressionTransferRequest, ExpressionTransferResponse
from src.reconstruction.deep3d_recon import BFM_DIR, CHECKPOINT_PATH, Deep3DFaceReconstructor

logger = logging.getLogger("ca_monk.expression_transfer")


class Deep3DExpressionTransferService:
    """
    Expression-transfer surface built on the existing Deep3D/BFM stack.

    This keeps the current reconstruction pipeline intact while exposing a
    separate capability that swaps the source face's expression coefficients
    for those estimated from another image.
    """

    def __init__(
        self,
        deep3d_provider: Optional[Callable[[], Deep3DFaceReconstructor]] = None,
        output_dir: str = "evidence_cards",
    ) -> None:
        self.output_dir = output_dir
        self._deep3d_provider = deep3d_provider
        self._owned_deep3d: Deep3DFaceReconstructor | None = None

    @property
    def deep3d(self) -> Deep3DFaceReconstructor:
        if self._deep3d_provider is not None:
            return self._deep3d_provider()
        if self._owned_deep3d is None:
            self._owned_deep3d = Deep3DFaceReconstructor()
        return self._owned_deep3d

    def capabilities(self) -> Dict[str, Any]:
        bfm_model_path = os.path.join(BFM_DIR, "BFM_model_front.mat")
        return {
            "backend": self.__class__.__name__,
            "available": bool(
                os.path.isfile(CHECKPOINT_PATH) and os.path.isfile(bfm_model_path)
            ),
            "shared_backbone": "Deep3DFaceReconstructor",
            "checkpoint_present": os.path.isfile(CHECKPOINT_PATH),
            "bfm_model_present": os.path.isfile(bfm_model_path),
            "method": "deep3d_bfm_expression_coeff_transfer",
            "notes": [
                "Keeps the existing reconstruction pipeline unchanged.",
                "Transfers 64 BFM expression coefficients from the expression image.",
                "Optional pose transfer copies global pose only, not FLAME jaw pose.",
            ],
        }

    def _default_save_path(self, req: ExpressionTransferRequest) -> str:
        os.makedirs(self.output_dir, exist_ok=True)
        out_dir = os.path.join(self.output_dir, "expression_transfer")
        os.makedirs(out_dir, exist_ok=True)
        src_stem = os.path.splitext(os.path.basename(req.source_image_path))[0]
        exp_stem = os.path.splitext(os.path.basename(req.expression_image_path))[0]
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        return os.path.join(out_dir, f"{src_stem}_xfer_{exp_stem}_{ts}.jpg")

    @staticmethod
    def _with_suffix(path: str, suffix: str) -> str:
        root, ext = os.path.splitext(path)
        if not ext:
            ext = ".jpg"
        return f"{root}{suffix}{ext}"

    @staticmethod
    def _label_tile(image: np.ndarray, label: str) -> np.ndarray:
        tile = image.copy()
        cv2.rectangle(tile, (0, 0), (tile.shape[1], 32), (8, 18, 30), thickness=-1)
        cv2.putText(
            tile,
            label,
            (10, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (125, 211, 252),
            2,
            cv2.LINE_AA,
        )
        return tile

    def _build_preview_strip(
        self,
        source_aligned: np.ndarray,
        expression_aligned: np.ndarray,
        transferred_rendered: np.ndarray,
    ) -> np.ndarray:
        tiles = [
            self._label_tile(source_aligned, "Identity Source"),
            self._label_tile(expression_aligned, "Expression Source"),
            self._label_tile(transferred_rendered, "Transferred Result"),
        ]
        return np.concatenate(tiles, axis=1)

    @staticmethod
    def _feather_mask(mask: np.ndarray) -> np.ndarray:
        mask_f = mask.astype(np.float32) / 255.0
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask_f = cv2.erode(mask_f, kernel, iterations=1)
        return cv2.GaussianBlur(mask_f, (9, 9), 3.0)

    @torch.no_grad()
    def _render_transfer(
        self,
        source_result: Dict[str, Any],
        expression_result: Dict[str, Any],
        transfer_pose: bool,
    ) -> Dict[str, Any]:
        source_coeffs = np.asarray(source_result["coefficients"], dtype=np.float32).copy()
        expression_coeffs = np.asarray(
            expression_result["coefficients"], dtype=np.float32
        )

        source_coeffs[80:144] = expression_coeffs[80:144]
        if transfer_pose:
            source_coeffs[224:227] = expression_coeffs[224:227]

        coeff_tensor = torch.from_numpy(source_coeffs).unsqueeze(0).to(self.deep3d.device)
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
        render_512 = self.deep3d.renderer.render(
            verts_cam, face_buf, colors, normals, output_size=512
        )
        side_view = self.deep3d.renderer.render_rotated(
            verts_cam,
            face_buf,
            colors,
            normals,
            angle_y_deg=30.0,
            output_size=224,
        )

        depth_colored_512 = cv2.applyColorMap(render_512["depth"], cv2.COLORMAP_INFERNO)
        depth_colored_512[render_512["mask"] == 0] = 0

        aligned_source = source_result["aligned_input"]
        aligned_source_512 = cv2.resize(
            aligned_source, (512, 512), interpolation=cv2.INTER_LANCZOS4
        )
        mask_feathered = self._feather_mask(render_512["mask"])
        mask_3ch = np.stack([mask_feathered] * 3, axis=-1)
        overlay_512 = (
            render_512["rendered"].astype(np.float32) * mask_3ch
            + aligned_source_512.astype(np.float32) * (1.0 - mask_3ch)
        )
        overlay_512 = np.clip(overlay_512, 0, 255).astype(np.uint8)

        preview_strip = self._build_preview_strip(
            source_result["aligned_input"],
            expression_result["aligned_input"],
            render_224["rendered"],
        )

        coeff_dict = {
            key: value[0].detach().cpu().numpy()
            for key, value in recon["coeff_dict"].items()
        }
        metadata = {
            "transfer_pose": bool(transfer_pose),
            "source_expression_norm": round(
                float(np.linalg.norm(np.asarray(source_result["coefficients"])[80:144])),
                4,
            ),
            "expression_source_norm": round(
                float(np.linalg.norm(np.asarray(expression_result["coefficients"])[80:144])),
                4,
            ),
            "transferred_expression_norm": round(
                float(np.linalg.norm(source_coeffs[80:144])),
                4,
            ),
            "source_angles": np.asarray(source_result["coefficients"])[224:227].tolist(),
            "expression_angles": expression_coeffs[224:227].tolist(),
            "transferred_angles": source_coeffs[224:227].tolist(),
        }

        return {
            "rendered": render_512["rendered"],
            "overlay": overlay_512,
            "depth_colored": depth_colored_512,
            "geometry": render_512["geometry"],
            "normal_map": render_512["normal_map"],
            "side_view": side_view["rendered"],
            "preview_strip": preview_strip,
            "mesh_vertices": recon["face_shape"][0].detach().cpu().numpy(),
            "mesh_faces": face_buf,
            "mesh_colors": colors,
            "landmarks_68": recon["landmarks_2d"][0].detach().cpu().numpy(),
            "coefficients": source_coeffs,
            "coeff_dict": coeff_dict,
            "metadata": metadata,
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
                warnings=[f"expression_transfer_failed: {str(exc)[:160]}"],
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

        transfer_result = self._render_transfer(
            source_result=source_result,
            expression_result=expression_result,
            transfer_pose=req.transfer_pose,
        )

        overlay_path = save_path
        rendered_path = self._with_suffix(save_path, "_rendered")
        preview_path = self._with_suffix(save_path, "_preview")
        depth_path = self._with_suffix(save_path, "_depth")
        geometry_path = self._with_suffix(save_path, "_geometry")
        normal_map_path = self._with_suffix(save_path, "_normals")
        side_view_path = self._with_suffix(save_path, "_sideview")
        mesh_path = self._with_suffix(save_path, "_mesh")
        mesh_path = os.path.splitext(mesh_path)[0] + ".obj"

        cv2.imwrite(overlay_path, transfer_result["overlay"])
        cv2.imwrite(rendered_path, transfer_result["rendered"])
        cv2.imwrite(preview_path, transfer_result["preview_strip"])
        cv2.imwrite(depth_path, transfer_result["depth_colored"])
        cv2.imwrite(geometry_path, transfer_result["geometry"])
        cv2.imwrite(normal_map_path, transfer_result["normal_map"])
        cv2.imwrite(side_view_path, transfer_result["side_view"])
        self.deep3d.save_obj(transfer_result, mesh_path)

        elapsed = time.time() - t0
        warnings.append("deep3d_expression_transfer_complete")
        warnings.append(f"processing_time_{elapsed:.1f}s")

        metadata = dict(transfer_result["metadata"])
        metadata.update(
            {
                "source_image_path": req.source_image_path,
                "expression_image_path": req.expression_image_path,
                "elapsed_ms": round(elapsed * 1000.0, 2),
            }
        )

        logger.info(
            "Expression transfer complete in %.1fs: source=%s expression=%s pose=%s",
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
            warnings=warnings,
            metadata=metadata,
        )

    def cleanup(self) -> None:
        if self._deep3d_provider is None and self._owned_deep3d is not None:
            self._owned_deep3d.cleanup()
            self._owned_deep3d = None
