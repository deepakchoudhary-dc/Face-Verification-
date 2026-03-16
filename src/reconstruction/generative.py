"""
DEEP3D FORENSIC FACE RECONSTRUCTOR -- v5.1
============================================
True Deep3DFaceReconstruction integration for CA_MONK,
using ResNet50 -> BFM (Basel Face Model) -> CPU-rendered 3D face.

Architecture (from https://github.com/microsoft/Deep3DFaceReconstruction):
    1. ResNet50 backbone -> 257 BFM coefficients (~150ms CPU)
       - 80 identity, 64 expression, 80 texture
       - 3 rotation, 27 SH illumination, 3 translation
    2. Basel Face Model -> 35,709-vertex 3D mesh with per-vertex color
    3. CPU Software Renderer -> rendered face + depth map + face mask
    4. CodeFormer ONNX -> neural face restoration (~400MB ONNX)

Pipeline:
    Input -> InsightFace 5-point landmarks -> 224x224 alignment
    -> ResNet50 forward -> BFM coefficients -> 3D mesh + lit texture
    -> CPU rasterize -> rendered face + depth map
    -> Upscale to 512 -> OcclusionRemover -> LightingNormalizer
    -> ForensicReconstructor -> SuperResolution -> CodeFormer
    -> Save evidence chain + .obj mesh

Key advantages over previous pipeline:
    - ACTUAL 3D face reconstruction (Deep3D resnet50, not MediaPipe proxy)
    - Disentangled shape/expression/texture/illumination
    - Identity-preserving (trained on face recognition loss)
    - 0.3s reconstruction + 2-4s post-processing on CPU
    - Generates real .obj 3D mesh exportable to MeshLab
    - Proper depth map from real 3D geometry
    - Landmark projection from BFM fitting (accurate 68-point)

Author: CA_MONK Forensic Intelligence Unit
Version: 5.1.0 -- Deep3DFaceReconstruction Integration
"""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from src.core.contracts import ReconstructionRequest, ReconstructionResponse
from src.face_engine.restoration import CodeFormerONNXRestorer

logger = logging.getLogger("ca_monk.reconstruction")


# ============================================================================
#  Main Reconstructor -- Deep3D + Forensic Post-Processing
# ============================================================================
class OpenVINOForensicReconstructor:
    """
    Deep3D Forensic Face Reconstructor -- true 3D face reconstruction.

    Uses Microsoft's Deep3DFaceReconstruction (PyTorch ResNet50 + BFM09)
    for real 3D face mesh reconstruction, followed by forensic post-processing.

    Pipeline:
        Deep3D reconstruct -> Upscale -> OcclusionRemoval -> LightingNorm
        -> ForensicRecon -> SuperRes -> CodeFormer -> Evidence Chain

    Memory: ~600MB total (ResNet50 ~160MB + BFM ~50MB + CodeFormer ~400MB)
    Speed:  0.3s (Deep3D) + 2-5s (post-processing) on CPU
    """

    def __init__(
        self,
        base_model: Optional[str] = None,
        ip_adapter_repo: Optional[str] = None,
        evidence_dir: str = "evidence_cards",
    ) -> None:
        self.evidence_dir = evidence_dir
        self.runtime = "deep3d_bfm_resnet50"

        # Deep3D engine (lazy loaded on first use)
        self._deep3d = None

        # CodeFormer (ONNX neural face restoration)
        self.codeformer = CodeFormerONNXRestorer()

        # Lazy-loaded post-processing modules
        self._face_reconstructor = None
        self._occlusion_remover = None
        self._lighting_normalizer = None
        self._super_resolution = None
        self._identity_recognizer = None

        logger.info(
            "Deep3D Face Reconstructor initialized -- "
            "ResNet50 + BFM09 + CPU renderer. "
            "True 3D face reconstruction."
        )

    # -- Lazy module loading --

    @property
    def deep3d(self):
        """Lazy-load Deep3D engine (ResNet50 + BFM)."""
        if self._deep3d is None:
            from src.reconstruction.deep3d_recon import Deep3DFaceReconstructor
            self._deep3d = Deep3DFaceReconstructor()
        return self._deep3d

    @property
    def face_reconstructor(self):
        if self._face_reconstructor is None:
            from src.reconstruction.face_reconstructor import ForensicFaceReconstructor
            self._face_reconstructor = ForensicFaceReconstructor()
        return self._face_reconstructor

    @property
    def occlusion_remover(self):
        if self._occlusion_remover is None:
            from src.reconstruction.occlusion_remover import OcclusionRemover
            self._occlusion_remover = OcclusionRemover()
        return self._occlusion_remover

    @property
    def lighting_normalizer(self):
        if self._lighting_normalizer is None:
            from src.reconstruction.lighting_normalizer import LightingNormalizer
            self._lighting_normalizer = LightingNormalizer()
        return self._lighting_normalizer

    @property
    def super_resolution(self):
        if self._super_resolution is None:
            from src.reconstruction.super_resolution import SuperResolutionEngine
            self._super_resolution = SuperResolutionEngine(target_size=512)
        return self._super_resolution

    @property
    def aging_simulator(self):
        if not hasattr(self, '_aging_simulator') or self._aging_simulator is None:
            from src.reconstruction.aging_simulator import AgingSimulator
            self._aging_simulator = AgingSimulator()
        return self._aging_simulator

    @property
    def identity_recognizer(self):
        if self._identity_recognizer is None:
            from src.face_engine.recognition import AdaFaceRecognizer
            self._identity_recognizer = AdaFaceRecognizer()
        return self._identity_recognizer

    @staticmethod
    def _feather_face_mask(mask: Optional[np.ndarray], target_shape: Optional[Tuple[int, int]] = None) -> Optional[np.ndarray]:
        if mask is None or mask.size == 0:
            return None
        mask_f = mask.astype(np.float32)
        if mask_f.max() > 1.0:
            mask_f /= 255.0
        if target_shape is not None and mask_f.shape[:2] != target_shape:
            mask_f = cv2.resize(mask_f, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_LINEAR)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask_f = cv2.erode(mask_f, kernel, iterations=1)
        mask_f = cv2.GaussianBlur(mask_f, (17, 17), 4.0)
        return np.clip(mask_f, 0.0, 1.0)

    @staticmethod
    def _composite_face(face: np.ndarray, background: Optional[np.ndarray], mask: Optional[np.ndarray]) -> np.ndarray:
        if background is None or mask is None:
            return face.copy()
        if background.shape[:2] != face.shape[:2]:
            background = cv2.resize(background, (face.shape[1], face.shape[0]), interpolation=cv2.INTER_LANCZOS4)
        if mask.shape[:2] != face.shape[:2]:
            mask = cv2.resize(mask, (face.shape[1], face.shape[0]), interpolation=cv2.INTER_LINEAR)
        mask_3ch = np.stack([mask] * 3, axis=-1)
        composite = face.astype(np.float32) * mask_3ch + background.astype(np.float32) * (1.0 - mask_3ch)
        return np.clip(composite, 0, 255).astype(np.uint8)

    @staticmethod
    def _quality_metrics(image: np.ndarray, mask: Optional[np.ndarray] = None) -> Dict[str, float]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        sample = None
        if mask is not None and mask.size > 0:
            sample = mask > 0.2
            if int(np.count_nonzero(sample)) < 64:
                sample = None

        lap = cv2.Laplacian(gray, cv2.CV_64F)
        if sample is None:
            luma = gray.reshape(-1)
            sat = hsv[:, :, 1].reshape(-1)
            sharpness = float(lap.var())
        else:
            luma = gray[sample]
            sat = hsv[:, :, 1][sample]
            sharpness = float(np.var(lap[sample]))

        return {
            "mean_luma": float(np.mean(luma)),
            "std_luma": float(np.std(luma)),
            "mean_sat": float(np.mean(sat)),
            "sharpness": sharpness,
        }

    def _embedding_similarity(self, reference_face: np.ndarray, candidate_face: np.ndarray) -> Optional[float]:
        ref_emb, _ = self.identity_recognizer.embedding(reference_face)
        cand_emb, _ = self.identity_recognizer.embedding(candidate_face)
        if ref_emb is None or cand_emb is None:
            return None
        denom = float(np.linalg.norm(ref_emb) * np.linalg.norm(cand_emb)) + 1e-8
        return float(np.dot(ref_emb, cand_emb) / denom)

    def _maybe_restore_face(
        self,
        face: np.ndarray,
        reference_face: Optional[np.ndarray] = None,
        mask: Optional[np.ndarray] = None,
        min_sharpness_gain: float = 1.05,
        min_similarity: float = 0.90,
    ) -> Tuple[np.ndarray, bool, Dict[str, float]]:
        restored, applied = self.codeformer.restore(face)
        if not applied:
            return face.copy(), False, {}

        base_metrics = self._quality_metrics(face, mask=mask)
        restored_metrics = self._quality_metrics(restored, mask=mask)
        reference = reference_face if reference_face is not None else face
        similarity = self._embedding_similarity(reference, restored)

        if restored_metrics["sharpness"] < base_metrics["sharpness"] * min_sharpness_gain:
            return face.copy(), False, {
                "sharpness_before": base_metrics["sharpness"],
                "sharpness_after": restored_metrics["sharpness"],
            }
        if similarity is not None and similarity < min_similarity:
            return face.copy(), False, {
                "sharpness_before": base_metrics["sharpness"],
                "sharpness_after": restored_metrics["sharpness"],
                "identity_similarity": similarity,
            }

        details = {
            "sharpness_before": base_metrics["sharpness"],
            "sharpness_after": restored_metrics["sharpness"],
        }
        if similarity is not None:
            details["identity_similarity"] = similarity
        return restored, True, details

    def _prepare_age_seed_face(
        self,
        face: np.ndarray,
        mask: Optional[np.ndarray],
    ) -> Tuple[np.ndarray, List[str]]:
        seed = face.copy()
        notes: List[str] = []

        try:
            light_result = self.lighting_normalizer.normalize_lighting(seed)
            if light_result.get("success") and light_result.get("corrections_applied"):
                seed = light_result["normalized"]
                notes.append("age_seed_lighting_normalized")
        except Exception:
            pass

        seed = self.super_resolution._smart_unsharp_mask(
            seed, amount=0.2, radius=1.0, threshold=4
        )
        notes.append("age_seed_gentle_sharpen")

        restored, restored_ok, _ = self._maybe_restore_face(
            seed,
            reference_face=face,
            mask=mask,
            min_sharpness_gain=1.05,
            min_similarity=0.90,
        )
        if restored_ok:
            seed = restored
            notes.append("age_seed_codeformer")

        return seed, notes

    def _evaluate_age_variant(
        self,
        reference_face: np.ndarray,
        candidate_face: np.ndarray,
        current_age: int,
        target_age: int,
        mask: Optional[np.ndarray],
    ) -> Dict[str, Any]:
        age_delta = abs(int(target_age) - int(current_age))
        reference_metrics = self._quality_metrics(reference_face, mask=mask)
        candidate_metrics = self._quality_metrics(candidate_face, mask=mask)

        sharpness_ratio = candidate_metrics["sharpness"] / max(reference_metrics["sharpness"], 1e-6)
        saturation_delta = candidate_metrics["mean_sat"] - reference_metrics["mean_sat"]
        identity_similarity = self._embedding_similarity(reference_face, candidate_face)

        min_sharpness_ratio = 0.42 if age_delta <= 10 else 0.32 if age_delta <= 20 else 0.28
        min_candidate_sharpness = 60.0 if age_delta <= 20 else 45.0
        max_saturation_delta = 16.0 if age_delta <= 10 else 20.0 if age_delta <= 20 else 24.0
        min_identity_similarity = 0.88 if age_delta <= 10 else 0.84 if age_delta <= 20 else 0.80

        rejection_reasons: List[str] = []
        if sharpness_ratio < min_sharpness_ratio or candidate_metrics["sharpness"] < min_candidate_sharpness:
            rejection_reasons.append("soft")
        if abs(saturation_delta) > max_saturation_delta:
            rejection_reasons.append("color_shift")
        if identity_similarity is not None and identity_similarity < min_identity_similarity:
            rejection_reasons.append("identity_drift")

        result: Dict[str, Any] = {
            "accepted": not rejection_reasons,
            "age_delta": age_delta,
            "sharpness_ratio": round(float(sharpness_ratio), 4),
            "saturation_delta": round(float(saturation_delta), 4),
            "reference_sharpness": round(float(reference_metrics["sharpness"]), 2),
            "candidate_sharpness": round(float(candidate_metrics["sharpness"]), 2),
            "reference_saturation": round(float(reference_metrics["mean_sat"]), 2),
            "candidate_saturation": round(float(candidate_metrics["mean_sat"]), 2),
            "rejection_reasons": rejection_reasons,
        }
        if identity_similarity is not None:
            result["identity_similarity"] = round(float(identity_similarity), 4)
        return result

    # ------------------------------------------------------------------
    # Evidence helpers
    # ------------------------------------------------------------------
    def _default_evidence_path(self, req: ReconstructionRequest) -> str:
        os.makedirs(self.evidence_dir, exist_ok=True)
        stem = os.path.splitext(os.path.basename(req.image_path))[0]
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        return os.path.join(
            self.evidence_dir, f"{stem}_{req.mode}_{ts}_evidence.png"
        )

    def _save_evidence_steps(
        self,
        save_path: str,
        stages: Dict[str, np.ndarray],
    ) -> None:
        """Save final output + multi-stage visual evidence chain."""
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

        final = stages.get(
            "final", stages.get("enhanced", list(stages.values())[-1])
        )
        cv2.imwrite(save_path, final)

        valid_stages = [
            (k, v) for k, v in stages.items()
            if v is not None and v.size > 0
        ]
        if not valid_stages:
            return

        target_h = 256
        resized = []
        for label, img in valid_stages:
            h, w = img.shape[:2]
            ratio = target_h / h
            small = cv2.resize(
                img, (int(w * ratio), target_h), interpolation=cv2.INTER_AREA
            )
            if len(small.shape) == 2:
                small = cv2.cvtColor(small, cv2.COLOR_GRAY2BGR)
            cv2.putText(
                small, label, (6, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA,
            )
            resized.append(small)

        if resized:
            chain = np.concatenate(resized, axis=1)
            chain_path = save_path.replace(".jpg", "_chain.jpg").replace(
                ".png", "_chain.png"
            )
            cv2.imwrite(chain_path, chain)

    # ------------------------------------------------------------------
    # Main Deep3D reconstruction pipeline
    # ------------------------------------------------------------------
    def generate(self, req: ReconstructionRequest) -> ReconstructionResponse:
        """
        Full Deep3D forensic reconstruction pipeline.

        Pipeline stages:
            1. Load & validate input image
            2. Deep3D reconstruction (ResNet50 -> BFM -> CPU render)
               -> rendered 3D face, depth map, .obj mesh, 68 landmarks
            3. Upscale rendered face to 512x512
            4. Occlusion removal (glasses, masks, bandages)
            5. Lighting normalization
            6. Forensic face reconstruction (injury/scar aware)
            7. Super-resolution enhancement
            8. CodeFormer ONNX neural restoration
            9. Evidence chain + mesh save
        """
        t_start = time.time()
        warnings: List[str] = []
        guidance = getattr(req, "reconstruction_guidance", "") or ""
        stages: Dict[str, np.ndarray] = {}
        depth_map_colored = None
        forensic_3d_data: Dict[str, Any] = {}
        age_reference_face = None
        age_seed_face = None
        age_background = None
        age_mask = None

        logger.info(
            "Deep3D Reconstruction: mode=%s, image=%s, guidance_len=%d",
            req.mode, req.image_path, len(guidance),
        )

        # ---- Stage 1: Load image ----
        image = cv2.imread(req.image_path)
        if image is None:
            return ReconstructionResponse(
                generated_image_path=None,
                warnings=[f"image_not_readable: {req.image_path}"],
            )
        stages["01_original"] = image.copy()

        # ---- Stage 2: Deep3D 3D Face Reconstruction ----
        deep3d_result = None
        try:
            deep3d_result = self.deep3d.reconstruct(image)
        except FileNotFoundError as e:
            warnings.append(f"deep3d_model_missing: {str(e)[:120]}")
            logger.error("Deep3D model files missing: %s", e)
        except Exception as e:
            warnings.append(f"deep3d_failed: {str(e)[:120]}")
            logger.error("Deep3D reconstruction failed: %s", e)

        if deep3d_result is not None:
            d3d_ms = deep3d_result['elapsed'] * 1000
            warnings.append(f"deep3d_resnet50_bfm_reconstruction_{d3d_ms:.0f}ms")

            # Store Deep3D outputs as evidence stages
            stages["02_aligned_224"] = deep3d_result['aligned_input'].copy()
            stages["03_deep3d_textured"] = deep3d_result['rendered'].copy()
            stages["04_deep3d_geometry"] = deep3d_result.get('geometry', deep3d_result['rendered']).copy()
            stages["05_depth_map"] = deep3d_result['depth_colored'].copy()
            stages["06_side_view"] = deep3d_result.get('side_view', deep3d_result['rendered']).copy()
            stages["07_overlay"] = deep3d_result['overlay'].copy()
            depth_map_colored = deep3d_result['depth_colored']
            age_reference_face = deep3d_result['rendered'].copy()
            age_background = cv2.resize(
                deep3d_result['aligned_input'], (512, 512), interpolation=cv2.INTER_LANCZOS4
            )
            age_mask = self._feather_face_mask(
                deep3d_result.get('face_mask'), target_shape=(512, 512)
            )

            # Use the 512x512 overlay directly (already rendered at 512 by new renderer)
            current = deep3d_result['overlay'].copy()
            if current.shape[0] != 512 or current.shape[1] != 512:
                current = cv2.resize(current, (512, 512), interpolation=cv2.INTER_LANCZOS4)
            warnings.append("deep3d_native_512_render")

            logger.info(
                "Deep3D: %d vertices, 68 landmarks, %d triangles, "
                "%.0fms reconstruction, output=%dx%d",
                deep3d_result['mesh_vertices'].shape[0],
                deep3d_result['mesh_faces'].shape[0],
                d3d_ms,
                deep3d_result['rendered'].shape[1],
                deep3d_result['rendered'].shape[0],
            )
        else:
            # Fallback: just resize to 512 if Deep3D unavailable
            warnings.append("deep3d_unavailable_fallback_resize")
            current = cv2.resize(image, (512, 512), interpolation=cv2.INTER_LANCZOS4)
            stages["02_fallback_resize"] = current.copy()
            age_reference_face = current.copy()

        # ---- Stage 2.5: 3D Forensic Analysis (Anthropometry + Coefficients) ----
        if deep3d_result is not None:
            try:
                from src.forensics.anthropometry import ForensicAnthropometry
                from src.forensics.coefficient_analysis import CoefficientForensics

                _anthro = ForensicAnthropometry()
                _coeff = CoefficientForensics()

                lm_3d = deep3d_result.get("landmarks_3d")
                if lm_3d is not None:
                    forensic_3d_data["anthropometry"] = _anthro.measure(lm_3d)

                raw_coeffs = deep3d_result.get("coefficients")
                raw_dict = deep3d_result.get("coeff_dict")
                if raw_coeffs is not None:
                    forensic_3d_data["coefficient_analysis"] = _coeff.analyze(
                        raw_coeffs, raw_dict
                    )

                warnings.append("forensic_3d_anthropometry_and_coefficients_complete")
                logger.info(
                    "3D Forensic Analysis: anthropometry=%s, coefficients=%s",
                    "ok" if "anthropometry" in forensic_3d_data else "skip",
                    "ok" if "coefficient_analysis" in forensic_3d_data else "skip",
                )
            except Exception as exc:
                warnings.append(f"forensic_3d_analysis_failed: {str(exc)[:80]}")
                logger.warning("3D forensic analysis failed: %s", exc)

        # ---- Stage 3: Occlusion removal ----
        # ONLY run on fallback path. When Deep3D succeeds, the 3D mesh
        # already represents the face without occlusions — running 2D
        # heuristic detectors on a 3D render creates false positives.
        if deep3d_result is None:
            try:
                occ_result = self.occlusion_remover.remove_occlusions(current)
                if occ_result.get("success") and occ_result.get("occlusions_found"):
                    current = occ_result["cleaned"]
                    for occ in occ_result["occlusions_found"]:
                        warnings.append(f"occlusion_removed_{occ}")
                    stages["08_deoccluded"] = current.copy()
                else:
                    stages["08_no_occlusion"] = current.copy()
            except Exception as exc:
                warnings.append(f"occlusion_removal_skipped: {str(exc)[:80]}")
                stages["08_occ_skipped"] = current.copy()
        else:
            stages["08_deep3d_clean_mesh"] = current.copy()
            warnings.append("occlusion_removal_skipped_deep3d_provides_clean_mesh")

        # ---- Stage 4: Lighting normalization ----
        # Safe on both paths — corrects exposure/shadow issues.
        try:
            light_result = self.lighting_normalizer.normalize_lighting(current)
            if light_result.get("success") and light_result.get("corrections_applied"):
                current = light_result["normalized"]
                for corr in light_result["corrections_applied"]:
                    warnings.append(f"lighting_{corr}")
                stages["09_lit_normalized"] = current.copy()
            else:
                stages["09_lighting_ok"] = current.copy()
        except Exception as exc:
            warnings.append(f"lighting_normalization_skipped: {str(exc)[:80]}")
            stages["09_light_skipped"] = current.copy()

        # ---- Stage 5: Forensic face reconstruction ----
        # ONLY run on fallback (no Deep3D). The 3D BFM mesh IS the
        # reconstruction — running 2D scar/injury inpainting on a 3D
        # render destroys the image with false-positive anomaly detection.
        if deep3d_result is None:
            try:
                face_box = {
                    "x": 0, "y": 0,
                    "w": current.shape[1], "h": current.shape[0],
                }
                recon_result = self.face_reconstructor.analyze_and_reconstruct(
                    current, face_box
                )
                if recon_result.get("reconstructed_image") is not None:
                    probable = recon_result.get(
                        "probable_original", recon_result["reconstructed_image"]
                    )
                    current = probable
                    anomaly_count = len(recon_result.get("anomalies_detected", []))
                    if anomaly_count > 0:
                        warnings.append(f"forensic_reconstruction_{anomaly_count}_anomalies")
                    stages["10_forensic_recon"] = current.copy()
                else:
                    stages["10_recon_clean"] = current.copy()
            except Exception as exc:
                warnings.append(f"forensic_reconstruction_skipped: {str(exc)[:80]}")
                stages["10_recon_skipped"] = current.copy()
        else:
            stages["10_3d_is_reconstruction"] = current.copy()
            warnings.append("forensic_2d_reconstruction_skipped_3d_mesh_is_authoritative")

        # ---- Stage 6: Super-resolution enhancement ----
        # When Deep3D provides a clean 512px render, heavy SR processing
        # (sharpen+contrast+color) amplifies 3D render artifacts.
        # Apply ONLY gentle enhancement when Deep3D succeeded.
        try:
            if deep3d_result is not None:
                # Deep3D path: gentle sharpen only, no contrast/color mangling
                img_sr = current.copy()
                img_sr = self.super_resolution._smart_unsharp_mask(
                    img_sr, amount=0.3, radius=1.0, threshold=5
                )
                current = img_sr
                warnings.append("sr_gentle_sharpen_only")
                stages["11_gentle_enhanced"] = current.copy()
            else:
                # Fallback path: full SR pipeline
                sr_result = self.super_resolution.enhance_face(current, scale_factor=1.0)
                if sr_result.get("success") and sr_result.get("enhanced") is not None:
                    current = sr_result["enhanced"]
                    for enh in sr_result.get("enhancements", []):
                        warnings.append(f"sr_{enh}")
                    stages["11_enhanced"] = current.copy()
                else:
                    stages["11_sr_skipped"] = current.copy()
        except Exception as exc:
            warnings.append(f"super_resolution_skipped: {str(exc)[:80]}")
            stages["11_sr_err"] = current.copy()

        # ---- Stage 7: CodeFormer ONNX neural restoration ----
        cf_applied = False
        try:
            if self.codeformer.should_restore(current):
                cf_result = self.codeformer.restore(current)
                if cf_result[1]:
                    current = cf_result[0]
                    cf_applied = True
                    warnings.append("codeformer_onnx_applied")
                    stages["12_codeformer"] = current.copy()
                else:
                    warnings.append("codeformer_unavailable_or_skipped")
                    stages["12_cf_skipped"] = current.copy()
            else:
                warnings.append("codeformer_skipped_sharp_enough")
                stages["12_cf_skipped"] = current.copy()
        except Exception as exc:
            warnings.append(f"codeformer_skipped: {str(exc)[:80]}")
            stages["12_cf_err"] = current.copy()

        # ---- Final stage ----
        stages["final"] = current.copy()

        # Compute save path early (needed by age simulation + evidence save)
        save_path = req.evidence_save_path or self._default_evidence_path(req)
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

        # ---- Age Simulation (generates de-aged / aged variants) ----
        age_simulation: Dict[str, Any] = {}
        detected_age = getattr(req, 'estimated_age', None)
        detected_sex = getattr(req, 'sex', None) or 'unknown'
        age_context = getattr(req, "age_context", {}) or {}
        age_policy = str(age_context.get("variant_policy", "full"))
        max_age_delta = int(age_context.get("max_age_delta_years", 0) or 0)
        max_variants = int(age_context.get("max_variants", 0) or 0)
        allow_age_variants = bool(age_context.get("allow_age_variants", True))
        if req.mode != "age_progression":
            age_simulation = {
                "detected_age": detected_age,
                "detected_sex": detected_sex,
                "age_context": age_context,
                "variants": [],
                "rejected_variants": [],
                "skipped_reason": "mode_not_age_progression",
            }
            warnings.append("age_simulation_skipped_mode_not_age_progression")
        elif not allow_age_variants:
            age_simulation = {
                "detected_age": detected_age,
                "detected_sex": detected_sex,
                "age_context": age_context,
                "variants": [],
                "rejected_variants": [],
                "skipped_reason": "age_context_policy_disabled",
            }
            warnings.append("age_simulation_skipped_age_context_policy_disabled")
        elif detected_age is not None and detected_age > 0:
            try:
                age_simulation['detected_age'] = detected_age
                age_simulation['detected_sex'] = detected_sex
                age_simulation['age_context'] = age_context
                age_simulation['variants'] = []
                age_simulation['rejected_variants'] = []

                if age_reference_face is None:
                    age_reference_face = current.copy()
                if age_seed_face is None:
                    age_seed_face = age_reference_face.copy()

                if deep3d_result is not None:
                    age_seed_face, prep_notes = self._prepare_age_seed_face(
                        age_reference_face, age_mask
                    )
                    if prep_notes:
                        stages["12a_age_seed"] = age_seed_face.copy()
                        for note in prep_notes:
                            warnings.append(note)

                # Decide age targets based on current age
                targets: List[Tuple[int, str]] = []
                if detected_age >= 55:
                    # Elderly: show what they looked like younger
                    targets.append((detected_age - 10, f"minus_10yr_{detected_age - 10}"))
                    targets.append((detected_age - 20, f"minus_20yr_{detected_age - 20}"))
                    if detected_age >= 65:
                        targets.append((detected_age - 30, f"minus_30yr_{detected_age - 30}"))
                elif detected_age <= 30:
                    # Young: show what they will look like older
                    targets.append((detected_age + 10, f"plus_10yr_{detected_age + 10}"))
                    targets.append((detected_age + 20, f"plus_20yr_{detected_age + 20}"))
                    targets.append((detected_age + 30, f"plus_30yr_{detected_age + 30}"))
                else:
                    # Middle-aged: both directions
                    targets.append((detected_age - 15, f"minus_15yr_{detected_age - 15}"))
                    targets.append((detected_age + 15, f"plus_15yr_{detected_age + 15}"))
                    targets.append((detected_age + 25, f"plus_25yr_{detected_age + 25}"))

                # Clamp targets to [5, 95]
                targets = [(max(5, t), lbl) for t, lbl in targets]
                targets = [(min(95, t), lbl) for t, lbl in targets]
                if max_age_delta > 0:
                    targets = [
                        (t, lbl) for t, lbl in targets
                        if abs(int(t) - int(detected_age)) <= max_age_delta
                    ]
                if max_variants > 0:
                    targets = targets[:max_variants]

                for target_age, label in targets:
                    sim_result = self.aging_simulator.simulate_aging(
                        age_seed_face, detected_age, target_age
                    )
                    if sim_result.get('success') and sim_result.get('simulated') is not None:
                        sim_face = sim_result['simulated']
                        restored_face, restored_ok, _ = self._maybe_restore_face(
                            sim_face,
                            reference_face=age_reference_face,
                            mask=age_mask,
                            min_sharpness_gain=1.02,
                            min_similarity=0.84,
                        )
                        if restored_ok:
                            sim_face = restored_face
                            warnings.append(f"age_sim_{label}_codeformer")

                        quality_gate = self._evaluate_age_variant(
                            age_reference_face,
                            sim_face,
                            detected_age,
                            target_age,
                            age_mask,
                        )

                        variant_info = {
                            'target_age': target_age,
                            'label': label,
                            'confidence': sim_result.get('confidence', 0),
                            'transformations': sim_result.get('transformations', []),
                            'quality_gate': quality_gate,
                        }

                        if not quality_gate.get("accepted", False):
                            age_simulation['rejected_variants'].append(variant_info)
                            reason = "_".join(quality_gate.get("rejection_reasons", ["gated"]))
                            warnings.append(f"age_sim_{label}_rejected_{reason}")
                            continue

                        sim_img = self._composite_face(sim_face, age_background, age_mask)
                        stage_key = f"age_{label}"
                        stages[stage_key] = sim_img.copy()

                        # Save as separate evidence file
                        age_path = save_path.replace(
                            '.jpg', f'_age_{label}.jpg'
                        ).replace('.png', f'_age_{label}.png')
                        cv2.imwrite(age_path, sim_img)

                        variant_info['saved_path'] = os.path.basename(age_path)
                        age_simulation['variants'].append(variant_info)
                        warnings.append(f"age_sim_{label}_saved")

                direction = 'de-aging' if detected_age >= 55 else (
                    'aging' if detected_age <= 30 else 'bidirectional'
                )
                logger.info(
                    "Age Simulation: detected=%d, sex=%s, direction=%s, %d variants generated",
                    detected_age, detected_sex, direction, len(age_simulation['variants']),
                )
                warnings.append(f"age_simulation_{direction}_{len(age_simulation['variants'])}_variants")
            except Exception as exc:
                warnings.append(f"age_simulation_failed: {str(exc)[:80]}")
                logger.warning("Age simulation failed: %s", exc)
        else:
            warnings.append("age_simulation_skipped_no_age_detected")

        # ---- Save evidence ----
        try:
            self._save_evidence_steps(save_path, stages)
            logger.info("Deep3D reconstruction evidence saved: %s", save_path)
        except Exception as exc:
            warnings.append(f"evidence_save_failed: {exc}")
            return ReconstructionResponse(
                generated_image_path=None, warnings=warnings
            )

        # Save depth map separately
        if depth_map_colored is not None:
            try:
                depth_path = save_path.replace(
                    ".jpg", "_depth.jpg"
                ).replace(".png", "_depth.png")
                cv2.imwrite(depth_path, depth_map_colored)
                warnings.append(f"depth_map_saved:{os.path.basename(depth_path)}")
            except Exception:
                pass

        # Save geometry render (gray Lambertian shading — like GitHub demo)
        if deep3d_result is not None and deep3d_result.get('geometry') is not None:
            try:
                geo_path = save_path.replace(".jpg", "_geometry.jpg").replace(
                    ".png", "_geometry.png"
                )
                cv2.imwrite(geo_path, deep3d_result['geometry'])
                warnings.append(f"geometry_render_saved:{os.path.basename(geo_path)}")
            except Exception:
                pass

        # Save 3/4 side-view render
        if deep3d_result is not None and deep3d_result.get('side_view') is not None:
            try:
                side_path = save_path.replace(".jpg", "_sideview.jpg").replace(
                    ".png", "_sideview.png"
                )
                cv2.imwrite(side_path, deep3d_result['side_view'])
                warnings.append(f"side_view_saved:{os.path.basename(side_path)}")
            except Exception:
                pass

        # Save normal map
        if deep3d_result is not None and deep3d_result.get('normal_map') is not None:
            try:
                nmap_path = save_path.replace(".jpg", "_normals.jpg").replace(
                    ".png", "_normals.png"
                )
                cv2.imwrite(nmap_path, deep3d_result['normal_map'])
                warnings.append(f"normal_map_saved:{os.path.basename(nmap_path)}")
            except Exception:
                pass

        # Save .obj mesh if Deep3D succeeded
        if deep3d_result is not None:
            try:
                mesh_path = save_path.replace(
                    ".jpg", "_mesh.obj"
                ).replace(".png", "_mesh.obj")
                self.deep3d.save_obj(deep3d_result, mesh_path)
                warnings.append(f"mesh_saved:{os.path.basename(mesh_path)}")
                logger.info("3D mesh saved: %s", mesh_path)
            except Exception as exc:
                warnings.append(f"mesh_save_failed: {str(exc)[:80]}")

        elapsed = time.time() - t_start
        warnings.append("deep3d_forensic_pipeline_v5.1")
        warnings.append(f"processing_time_{elapsed:.1f}s")
        logger.info(
            "Deep3D Forensic Reconstruction complete in %.1fs -- "
            "%d stages, deep3d=%s, codeformer=%s, depth_map=%s, mesh=%s",
            elapsed,
            len(stages),
            "yes" if deep3d_result is not None else "no",
            "yes" if cf_applied else "no",
            "yes" if depth_map_colored is not None else "no",
            "yes" if deep3d_result is not None else "no",
        )

        return ReconstructionResponse(
            generated_image_path=save_path,
            warnings=warnings,
            forensic_3d=forensic_3d_data,
            age_simulation=age_simulation,
        )
