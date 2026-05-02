from __future__ import annotations

import asyncio
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.core.contracts import (
    BiometricsRequest,
    DocumentRequest,
    EmbeddingResult,
    ExpressionTransferRequest,
    ForensicsRequest,
    PairMatchRequest,
    ReconstructionRequest,
    ReportRequest,
)
from src.core.data_structures import Applicant
from src.document_processor.extractor import DocumentExtractor
from src.face_engine.analyzer import FaceAnalyzer
from src.face_engine.image_study import PrimaryImageStudy
from src.forensics.service import ForensicsService
from src.biometric_analysis.orchestrator import AdvancedBiometricOrchestrator
from src.face_engine.visualizer import ForensicVisualizer
from src.reconstruction.generative import OpenVINOForensicReconstructor
from src.reconstruction.expression_transfer import Deep3DExpressionTransferService
from src.reconstruction.expression_suite import Deep3DExpressionSuiteService
from src.reporting.llm_analyst import LlamaForensicAnalyst
from src.reporting.interactive_casefile import InteractiveCasefileBuilder
from src.core.serialization import to_builtin

logger = logging.getLogger("ca_monk.engine")


def _is_video_file(path: str) -> bool:
    return Path(path).suffix.lower() in {".mp4", ".mov", ".avi", ".webm", ".mkv"}


class VerificationEngine:
    """
    CA_Monk v5 — Military-Grade Biometric Intelligence Engine.

    Linear sequential pipeline (NO Ray Serve):
        Ingest → Biometrics (InsightFace/AdaFace) → Forensics (F3-Net/rPPG)
        → Document Intelligence (Donut/NoisePrint)
        → Reconstruction (Deep3D: MediaPipe Mesh + OpenCV + CodeFormer)
        → Reporting (Llama-3 CoT) → Evidence Package

    v5.0 — SD 1.5 Realistic Vision REMOVED. Reconstruction now uses:
        Deep3D Forensic Pipeline (MediaPipe 3D Mesh → InsightFace Align
        → Occlusion Removal → Lighting → Forensic Recon → 3D Depth Map
        → Super Resolution → CodeFormer ONNX)
        ~90% less CPU/RAM, ~20x faster, identity-preserving.

    Only ONE heavy model on GPU at a time via ModelManager.
    Generates evidence_package_{id}/ with deliverables:
        reconstruction_hq.jpg, spectral_analysis.jpg, tamper_heatmap.jpg,
        biometric_pulse.png, FINAL_REPORT.md, depth map
    """

    OUTPUT_DIR = "evidence_cards"
    EVIDENCE_ROOT = "forensic_output"

    def __init__(self) -> None:
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)
        os.makedirs(self.EVIDENCE_ROOT, exist_ok=True)
        self.extractor = DocumentExtractor()

        # Direct service instances — no Ray Serve overhead
        self.analyzer = FaceAnalyzer()
        self.forensics = ForensicsService()
        self.adv_biometrics = AdvancedBiometricOrchestrator()
        self.reconstruction = OpenVINOForensicReconstructor()
        self.expression_transfer = Deep3DExpressionTransferService(
            deep3d_provider=lambda: self.reconstruction.deep3d,
            output_dir=self.OUTPUT_DIR,
        )
        self.expression_suite = Deep3DExpressionSuiteService(
            deep3d_provider=lambda: self.reconstruction.deep3d,
            output_dir=self.OUTPUT_DIR,
        )
        self.reporting = LlamaForensicAnalyst()
        self.image_study = PrimaryImageStudy()
        self.visualizer = ForensicVisualizer(output_dir=self.OUTPUT_DIR)
        self.casefile_builder = InteractiveCasefileBuilder()
        self._quality_calibration_dir = os.getenv("CA_MONK_CALIBRATION_DIR")
        self._quality_calibrated = False
        logger.info("VerificationEngine v4 initialized — linear pipeline + adv biometrics.")

    # ------------------------------------------------------------------
    # Evidence package directory
    # ------------------------------------------------------------------
    def _create_evidence_dir(self, applicant_id: str) -> str:
        """Create evidence_package_{id}_{timestamp}/ under EVIDENCE_ROOT."""
        safe_id = "".join(ch for ch in applicant_id if ch.isalnum() or ch in ("_", "-"))
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        dirname = f"evidence_package_{safe_id}_{ts}"
        path = os.path.join(self.EVIDENCE_ROOT, dirname)
        os.makedirs(path, exist_ok=True)
        logger.info("Evidence package directory: %s", path)
        return path

    def _maybe_calibrate_quality_threshold(self) -> None:
        """Calibrate once from an explicit environment directory, never from bundled demo data."""
        if self._quality_calibrated:
            return

        calibrate_dir = self._quality_calibration_dir
        self._quality_calibrated = True
        if not calibrate_dir:
            return

        if not os.path.isdir(calibrate_dir):
            logger.warning("Calibration directory not found: %s", calibrate_dir)
            return

        try:
            threshold = self.analyzer.calibrate_quality_threshold(calibrate_dir)
            logger.info(
                "Embedding quality threshold calibrated from %s -> %.3f",
                calibrate_dir,
                threshold,
            )
        except Exception as exc:
            logger.warning("Embedding quality calibration failed: %s", exc)

    @staticmethod
    def _utc_now() -> str:
        return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

    def _build_run_metadata(self, applicant_role: str, evidence_dir: str) -> Dict[str, Any]:
        capabilities = self.runtime_capabilities()
        providers = list((capabilities.get("face_analyzer", {}) or {}).get("recognizer", {}).get("providers", []) or [])

        return {
            "run_id": Path(evidence_dir).name,
            "applicant_role": applicant_role,
            "started_utc": self._utc_now(),
            "pipeline_mode": "linear",
            "execution_profile": "cpu_first",
            "quality_threshold": round(float(getattr(self.analyzer, "quality_threshold", 0.0) or 0.0), 4),
            "embedding_providers": providers,
            "runtime_capabilities": capabilities,
            "stage_telemetry": [],
        }

    @staticmethod
    def _record_stage(
        stage_store: List[Dict[str, Any]],
        stage_name: str,
        start_time: float,
        status: str = "ok",
        details: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        entry: Dict[str, Any] = {
            "stage": stage_name,
            "duration_ms": round((time.perf_counter() - start_time) * 1000.0, 2),
            "status": status,
        }
        if details:
            entry["details"] = to_builtin(details)
        stage_store.append(entry)
        return entry

    def _finalize_evidence_package(self, evidence_dir: str, result: Dict[str, Any]) -> Dict[str, Any]:
        from src.core.evidence_integrity import EvidenceIntegrity

        integrity = EvidenceIntegrity()
        summary = {
            "algorithm": integrity.HASH_ALGORITHM.upper(),
            "manifest_path": os.path.join(evidence_dir, EvidenceIntegrity.MANIFEST_FILENAME),
            "verification": {"status": "pending_manifest_creation"},
        }
        result["evidence_integrity"] = dict(summary)
        casefile = self.casefile_builder.build(evidence_dir, result)
        manifest = integrity.create_manifest(evidence_dir)
        verification = integrity.verify_manifest(evidence_dir)
        summary = {
            "chain_root_hash": manifest.get("chain_root_hash"),
            "total_files": manifest.get("total_files"),
            "algorithm": manifest.get("algorithm", "SHA-256"),
            "manifest_path": os.path.join(evidence_dir, EvidenceIntegrity.MANIFEST_FILENAME),
            "verification": verification,
            "interactive_casefile": casefile,
        }
        result["interactive_casefile"] = casefile
        result["evidence_integrity"] = summary
        for comparison in result.get("comparisons", []):
            if isinstance(comparison, dict):
                comparison["interactive_casefile"] = casefile
                comparison["evidence_integrity"] = summary
        return summary

    def runtime_capabilities(self) -> Dict[str, Any]:
        return {
            "face_analyzer": self.analyzer.runtime_capabilities(),
            "forensics": {
                "deepfake_detector": getattr(getattr(self.forensics, "deepfake", None), "__class__", type(None)).__name__,
                "rppg_detector": getattr(getattr(self.forensics, "rppg", None), "__class__", type(None)).__name__,
            },
            "reconstruction": {
                "backend": getattr(self.reconstruction, "__class__", type(None)).__name__,
            },
            "expression_transfer": self.expression_transfer.capabilities(),
            "expression_suite": self.expression_suite.capabilities(),
            "reporting": {
                "backend": getattr(self.reporting, "__class__", type(None)).__name__,
                "llm_available": bool(getattr(self.reporting, "llm_available", False)),
                "model_name": getattr(self.reporting, "model_name", None),
            },
        }

    # ------------------------------------------------------------------
    # Service invocations (synchronous, sequential)
    # ------------------------------------------------------------------
    def _extract_biometrics(
        self, image_path: str, calibrate_dir: Optional[str] = None
    ) -> Tuple[List[EmbeddingResult], List[str]]:
        warnings: List[str] = []
        faces: List[EmbeddingResult] = []
        try:
            if calibrate_dir:
                self.analyzer.calibrate_quality_threshold(calibrate_dir)
            raw_faces = self.analyzer.get_face_embeddings(image_path)
            for f in raw_faces:
                c = self.analyzer.to_contract(f)
                if c is not None:
                    faces.append(c)
        except Exception as exc:
            warnings.append(f"biometrics_extract_failed: {exc}")
        return faces, warnings

    def _compare_pair(
        self, primary: EmbeddingResult, comparison: EmbeddingResult
    ) -> Dict[str, Any]:
        try:
            result = self.analyzer.compare_pair(primary, comparison)
            return to_builtin(result.model_dump())
        except Exception as exc:
            return {"warnings": [f"compare_failed: {exc}"], "verified": False, "cosine_similarity": 0.0}

    def _run_forensics(
        self, image_path: str, video_path: Optional[str], evidence_dir: Optional[str]
    ) -> Dict[str, Any]:
        try:
            req = ForensicsRequest(image_path=image_path, video_path=video_path)
            return to_builtin(self.forensics.analyze(req, evidence_dir=evidence_dir).model_dump())
        except Exception as exc:
            return {"warnings": [f"forensics_failed: {exc}"]}

    def _run_document_analysis(
        self, image_path: str, face_box: Any, evidence_dir: Optional[str]
    ) -> Dict[str, Any]:
        try:
            return to_builtin(
                self.extractor.parse_id_document_with_evidence(
                    image_path, face_box, evidence_dir=evidence_dir
                ).model_dump()
            )
        except Exception:
            # Fallback to original method without evidence dir
            try:
                return to_builtin(self.extractor.parse_id_document(image_path, face_box).model_dump())
            except Exception as exc:
                return {"warnings": [f"document_analysis_failed: {exc}"]}

    def _run_reconstruction(
        self,
        image_path: str,
        embedding: Optional[List[float]],
        mode: str,
        save_path: Optional[str],
        reconstruction_guidance: Optional[str] = None,
        estimated_age: Optional[int] = None,
        sex: Optional[str] = None,
        age_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        try:
            req = ReconstructionRequest(
                image_path=image_path,
                face_embedding=embedding,
                mode=mode,
                evidence_save_path=save_path,
                reconstruction_guidance=reconstruction_guidance,
                estimated_age=estimated_age,
                sex=sex,
                age_context=age_context or {},
            )
            return to_builtin(self.reconstruction.generate(req).model_dump())
        except Exception as exc:
            return {"warnings": [f"reconstruction_failed: {exc}"]}

    def run_expression_transfer(
        self,
        source_image_path: str,
        expression_image_path: str,
        save_path: Optional[str] = None,
        transfer_pose: bool = False,
    ) -> Dict[str, Any]:
        try:
            req = ExpressionTransferRequest(
                source_image_path=source_image_path,
                expression_image_path=expression_image_path,
                evidence_save_path=save_path,
                transfer_pose=transfer_pose,
            )
            return to_builtin(self.expression_transfer.generate(req).model_dump())
        except Exception as exc:
            return {"warnings": [f"expression_transfer_failed: {exc}"]}

    def run_expression_suite(
        self,
        source_image_path: str,
        expression_image_path: str,
        save_path: Optional[str] = None,
        transfer_pose: bool = False,
    ) -> Dict[str, Any]:
        try:
            req = ExpressionTransferRequest(
                source_image_path=source_image_path,
                expression_image_path=expression_image_path,
                evidence_save_path=save_path,
                transfer_pose=transfer_pose,
            )
            return to_builtin(self.expression_suite.generate(req).model_dump())
        except Exception as exc:
            return {"warnings": [f"expression_suite_failed: {exc}"]}

    @staticmethod
    def _normalize_age_band(aging_features: Dict[str, Any]) -> Tuple[Optional[int], Optional[int], Optional[int]]:
        if not aging_features:
            return None, None, None
        min_age = aging_features.get("range_min")
        max_age = aging_features.get("range_max")
        if min_age is None or max_age is None:
            age_band = str(aging_features.get("estimated_age_range", "")).strip()
            if age_band == "18-30":
                min_age, max_age = 18, 30
            elif age_band == "30-50":
                min_age, max_age = 30, 50
            elif age_band == "50+":
                min_age, max_age = 50, 80
        midpoint = aging_features.get("range_midpoint")
        if midpoint is None and min_age is not None and max_age is not None:
            midpoint = int(round((int(min_age) + int(max_age)) / 2))
        return (
            int(min_age) if min_age is not None else None,
            int(max_age) if max_age is not None else None,
            int(midpoint) if midpoint is not None else None,
        )

    def _build_age_context(
        self,
        demographics: Optional[Dict[str, Any]],
        age_evidence: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        demographics = demographics or {}
        age_evidence = age_evidence or {}
        aging_features = age_evidence.get("aging_features", {}) or {}
        quality = age_evidence.get("quality", {}) or {}
        lighting = age_evidence.get("lighting", {}) or {}

        detector_age_raw = demographics.get("estimated_age")
        detector_age = None
        if detector_age_raw is not None:
            try:
                detector_age = int(detector_age_raw)
            except Exception:
                detector_age = None
        sex = demographics.get("sex")

        study_min, study_max, study_mid = self._normalize_age_band(aging_features)
        study_conf = float(aging_features.get("confidence", 0.0) or 0.0)
        reasons: List[str] = []

        if quality.get("blur_score", 999.0) < 60.0:
            study_conf *= 0.7
            reasons.append("age_study_blur_limited")
        if lighting.get("shadow_ratio", 0.0) > 0.45 or lighting.get("uneven", False):
            study_conf *= 0.8
            reasons.append("age_study_lighting_limited")

        estimated_age = None
        confidence = 0.0
        consistency = "none"
        variant_policy = "disabled"
        max_age_delta_years = 0
        max_variants = 0
        agreement_years = None

        if detector_age is not None and study_mid is not None:
            agreement_years = abs(detector_age - study_mid)
            expanded_min = study_min - 8 if study_min is not None else None
            expanded_max = study_max + 8 if study_max is not None else None

            if expanded_min is not None and expanded_max is not None and expanded_min <= detector_age <= expanded_max:
                consistency = "high" if study_min <= detector_age <= study_max else "medium"
                estimated_age = int(round(detector_age * 0.8 + study_mid * 0.2))
                confidence = min(0.92, 0.68 + study_conf * 0.5)
                variant_policy = "full" if confidence >= 0.75 else "conservative"
                max_age_delta_years = 30 if variant_policy == "full" else 10
                max_variants = 3 if variant_policy == "full" else 1
            elif agreement_years <= 25:
                consistency = "medium"
                estimated_age = int(round(detector_age * 0.88 + study_mid * 0.12))
                confidence = min(0.72, 0.56 + study_conf * 0.15)
                variant_policy = "conservative"
                max_age_delta_years = 10
                max_variants = 1
                reasons.append(f"age_sources_diverge_{agreement_years}y")
            else:
                consistency = "low"
                estimated_age = int(detector_age)
                confidence = 0.46
                variant_policy = "conservative"
                max_age_delta_years = 10
                max_variants = 1
                reasons.append(f"age_sources_conflict_{agreement_years}y")
        elif detector_age is not None:
            consistency = "single_source_detector"
            estimated_age = int(detector_age)
            confidence = 0.62
            variant_policy = "conservative"
            max_age_delta_years = 10
            max_variants = 1
            reasons.append("age_detector_only")
        elif study_mid is not None:
            consistency = "single_source_study"
            estimated_age = int(study_mid)
            confidence = max(0.25, study_conf)
            variant_policy = "disabled"
            max_age_delta_years = 0
            max_variants = 0
            reasons.append("age_study_only")

        context = {
            "estimated_age": estimated_age,
            "detector_age": detector_age,
            "study_age_range": aging_features.get("estimated_age_range"),
            "study_range_min": study_min,
            "study_range_max": study_max,
            "study_midpoint": study_mid,
            "study_confidence": round(study_conf, 3),
            "confidence": round(confidence, 3),
            "consistency": consistency,
            "variant_policy": variant_policy,
            "max_age_delta_years": max_age_delta_years,
            "max_variants": max_variants,
            "allow_age_variants": bool(estimated_age is not None and variant_policy != "disabled"),
            "sex": sex,
            "reasons": reasons,
        }
        if agreement_years is not None:
            context["agreement_years"] = int(agreement_years)
        return context

    def _run_reporting(
        self,
        applicant_id: str,
        biometrics: Dict[str, Any],
        forensics: Dict[str, Any],
        document: Dict[str, Any],
        reconstruction: Dict[str, Any],
        primary_image_study: Optional[Dict[str, Any]] = None,
        advanced_biometrics: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        try:
            req = ReportRequest(
                applicant_id=applicant_id,
                biometrics=biometrics,
                forensics=forensics,
                document=document,
                reconstruction=reconstruction,
                primary_image_study=primary_image_study or {},
                advanced_biometrics=advanced_biometrics or {},
            )
            return to_builtin(self.reporting.generate(req).model_dump())
        except Exception as exc:
            return {"warnings": [f"reporting_failed: {exc}"]}

    def _generate_explainability(
        self,
        primary_face: EmbeddingResult,
        comparison_face: EmbeddingResult,
        save_path: str,
    ) -> Dict[str, Any]:
        try:
            return self.analyzer.explain_similarity(
                image_a_path=primary_face.source_path,
                box_a=primary_face.box.model_dump(),
                image_b_path=comparison_face.source_path,
                box_b=comparison_face.box.model_dump(),
                save_path=save_path,
            )
        except Exception as exc:
            return {"warnings": [f"explainability_failed: {exc}"]}

    # ------------------------------------------------------------------
    # Stage 3.5 — Advanced Biometric Analysis
    # ------------------------------------------------------------------
    def _run_advanced_biometrics_single(
        self,
        image_path: str,
        face_box: Dict[str, Any],
        landmarks: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Run 7-module biometric sweep on a single face."""
        try:
            import cv2
            crop = self.adv_biometrics.crop_face(image_path, face_box)
            if crop is None:
                return {"warnings": ["adv_biometrics_crop_failed"]}
            return self.adv_biometrics.analyze_single(crop, face_box, landmarks=landmarks)
        except Exception as exc:
            return {"warnings": [f"adv_biometrics_failed: {exc}"]}

    def _run_advanced_biometrics_pair(
        self,
        primary_path: str,
        primary_box: Dict[str, Any],
        comparison_path: str,
        comparison_box: Dict[str, Any],
        face_match_score: float,
        primary_landmarks: Optional[Dict[str, Any]] = None,
        comparison_landmarks: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Run cross-image biometric comparison using all 7 modules."""
        try:
            import cv2
            crop1 = self.adv_biometrics.crop_face(primary_path, primary_box)
            crop2 = self.adv_biometrics.crop_face(comparison_path, comparison_box)
            if crop1 is None or crop2 is None:
                return {"warnings": ["adv_biometrics_crop_failed"]}
            return self.adv_biometrics.compare_pair(
                crop1, primary_box, crop2, comparison_box, face_match_score,
                primary_landmarks, comparison_landmarks,
            )
        except Exception as exc:
            return {"warnings": [f"adv_biometrics_pair_failed: {exc}"]}

    # ------------------------------------------------------------------
    # Write FINAL_REPORT.md
    # ------------------------------------------------------------------
    def _write_final_report(
        self, evidence_dir: str, report: Dict[str, Any], match: Dict[str, Any],
        primary_study: Optional[Dict[str, Any]] = None,
        advanced_biometrics: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Write a markdown report into the evidence package."""
        report_path = os.path.join(evidence_dir, "FINAL_REPORT.md")
        warnings = report.get("warnings", []) if isinstance(report, dict) else []
        verdict = report.get("verdict", "Manual Review") if isinstance(report, dict) else "Manual Review"
        confidence = report.get("confidence", 0.0) if isinstance(report, dict) else 0.0
        summary = report.get("summary", "") if isinstance(report, dict) else ""
        steps = report.get("reasoning_steps", []) if isinstance(report, dict) else []
        if not summary and warnings:
            summary = (
                "Report generation degraded. Warning-only output was produced: "
                + "; ".join(str(w) for w in warnings)
            )
        if verdict in (None, "", "Unknown") and warnings:
            verdict = "Manual Review"

        lines = [
            "# CA_MONK — FORENSIC INTELLIGENCE DOSSIER",
            "",
            f"**Verdict:** {verdict}",
            f"**Confidence:** {confidence:.2f}",
            "",
            "---",
            "",
            "## Executive Summary",
            "",
            summary,
            "",
        ]

        # --- Primary Image Deep Study Section ---
        if primary_study and primary_study.get("face_detected"):
            lines.extend([
                "---",
                "",
                "## Primary Image Deep Study",
                "",
                "### Findings",
                "",
            ])
            for finding in primary_study.get("findings", []):
                lines.append(f"- {finding}")
            lines.append("")

            issues = primary_study.get("issues_detected", [])
            if issues:
                lines.extend([
                    "### Issues Detected",
                    "",
                ])
                for issue in issues:
                    lines.append(f"- `{issue}`")
                lines.append("")

            marks = primary_study.get("marks_and_injuries", {})
            if marks.get("total_marks", 0) > 0:
                lines.extend([
                    "### Marks & Injuries Analysis",
                    "",
                    f"- **Total marks detected:** {marks.get('total_marks', 0)}",
                    f"- **Dark spots (scars/moles):** {marks.get('dark_spots', 0)}",
                    f"- **Red spots (bruises/inflammation):** {marks.get('red_spots', 0)}",
                    f"- **Scar-like linear regions:** {marks.get('scar_like_regions', 0)}",
                    "",
                ])

            quality = primary_study.get("quality", {})
            if quality:
                lines.extend([
                    "### Image Quality Metrics",
                    "",
                    f"- **Blur Score:** {quality.get('blur_score', 0):.1f}",
                    f"- **Noise Level:** {quality.get('noise_level', 0):.1f}",
                    f"- **Mean Brightness:** {quality.get('mean_brightness', 0):.0f}",
                    f"- **Resolution:** {quality.get('resolution', 'unknown')}",
                    "",
                ])

            skin = primary_study.get("skin_analysis", {})
            if skin:
                lines.extend([
                    "### Skin Analysis",
                    "",
                    f"- **Uniformity:** {skin.get('uniformity', 0):.2f}",
                    f"- **Redness Ratio:** {skin.get('redness_ratio', 0):.3f}",
                    f"- **Texture Roughness:** {skin.get('texture_roughness', 0):.1f}",
                    "",
                ])

            symmetry = primary_study.get("symmetry", {})
            aging = primary_study.get("aging_features", {})
            if symmetry or aging:
                lines.extend([
                    "### Biometric Features",
                    "",
                    f"- **Facial Symmetry Score:** {symmetry.get('symmetry_score', 0):.2f}",
                    f"- **Wrinkle Density:** {aging.get('wrinkle_density', 0):.2f}",
                    f"- **Estimated Age Range:** {aging.get('estimated_age_range', 'unknown')}",
                    f"- **Consensus Estimated Age:** {aging.get('consensus_estimated_age', 'unknown')}",
                    f"- **Age Evidence Consistency:** {aging.get('age_consistency', 'unknown')}",
                    "",
                ])

            guidance = primary_study.get("reconstruction_guidance", "")
            if guidance:
                lines.extend([
                    "### Reconstruction Guidance",
                    "",
                    f"> {guidance}",
                    "",
                ])

        # --- Advanced Biometrics Section ---
        if advanced_biometrics:
            pair = advanced_biometrics.get("pair_analysis", {})
            primary_bio = advanced_biometrics.get("primary", {})
            comp_bio = advanced_biometrics.get("comparison", {})

            lines.extend([
                "---",
                "",
                "## Advanced Biometric Analysis (Stage 3.5)",
                "",
                f"- **Primary Threat Level:** {primary_bio.get('threat_level', 'N/A')} "
                f"(score: {primary_bio.get('threat_score', 0):.3f})",
                f"- **Comparison Threat Level:** {comp_bio.get('threat_level', 'N/A')} "
                f"(score: {comp_bio.get('threat_score', 0):.3f})",
                f"- **Pair Verdict:** {pair.get('final_verdict', pair.get('verdict', 'N/A'))}",
                f"- **Pair Confidence:** {pair.get('confidence', 0):.2f}%",
                "",
            ])

            # Module-level alerts
            for src_label, src_bio in [("Primary", primary_bio), ("Comparison", comp_bio)]:
                alerts = src_bio.get("alerts", [])
                if alerts:
                    lines.append(f"### {src_label} Alerts")
                    lines.append("")
                    for a in alerts:
                        lines.append(f"- {a}")
                    lines.append("")

            # Sub-module details
            lines.extend([
                "### Module Results (Primary Image)",
                "",
                f"- **Tampering:** {'DETECTED' if primary_bio.get('tampering', {}).get('tampering_detected') else 'CLEAR'}",
                f"- **Morphing:** {'DETECTED' if primary_bio.get('morphing', {}).get('is_morphed') else 'CLEAR'}",
                f"- **Disguise/Makeup:** {'DETECTED' if primary_bio.get('makeup_disguise', {}).get('disguise_detected') else 'CLEAR'}",
                f"- **Iris Spoof:** {'SUSPECTED' if primary_bio.get('iris', {}).get('anti_spoofing', {}).get('contact_lens_detected') else 'CLEAR'}",
                f"- **Uniqueness Score:** {primary_bio.get('uniqueness', {}).get('uniqueness_score', 0):.2f}",
                f"- **Facial Markers:** {primary_bio.get('facial_markers', {}).get('markers_detected', 0)} detected",
                "",
            ])

            pair_recs = pair.get("recommendations", [])
            if pair_recs:
                lines.extend(["### Pair Recommendations", ""])
                for r in pair_recs:
                    lines.append(f"- {r}")
                lines.append("")

        lines.extend([
            "---",
            "",
            "## Chain-of-Thought Analysis",
            "",
        ])
        for i, step in enumerate(steps, 1):
            lines.append(f"### Step {i}")
            lines.append(step)
            lines.append("")

        lines.extend([
            "---",
            "",
            "## Match Details",
            "",
            f"- **Cosine Similarity:** {match.get('cosine_similarity', 'N/A')}",
            f"- **Verified:** {match.get('verified', 'N/A')}",
            f"- **Threshold:** {match.get('threshold', 'N/A')}",
            f"- **Rationale:** {match.get('rationale', 'N/A')}",
            "",
            "---",
            "",
            "## Evidence Artifacts",
            "",
            "| Artifact | Description |",
            "| --- | --- |",
            "| `reconstruction_hq.jpg` | Deep3D forensic reconstructed face (MediaPipe + CodeFormer) |",
            "| `primary_reconstruction_hq.jpg` | Primary image guided reconstruction |",
            "| `reconstruction_hq_depth.jpg` | 3D depth map (MediaPipe mesh interpolation) |",
            "| `spectral_analysis.jpg` | FFT high-pass ghost image heatmap |",
            "| `tamper_heatmap.jpg` | ELA splice/tampering heatmap |",
            "| `biometric_pulse.png` | rPPG liveness pulse graph |",
            "",
            "---",
            "",
            f"*Generated by CA_MONK Tier-1 Biometric Intelligence Platform — "
            f"{datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}*",
        ])

        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        logger.info("FINAL_REPORT.md written to %s", report_path)
        return report_path

    # ------------------------------------------------------------------
    # Best-pair selection
    # ------------------------------------------------------------------
    def _best_pair(
        self,
        primary_faces: List[EmbeddingResult],
        comparison_faces: List[EmbeddingResult],
    ) -> Optional[Tuple[EmbeddingResult, EmbeddingResult]]:
        best: Optional[Tuple[EmbeddingResult, EmbeddingResult]] = None
        best_rank = (float("-inf"), float("-inf"))
        for p in primary_faces:
            for c in comparison_faces:
                if not p.embedding or not c.embedding:
                    continue
                try:
                    match = self.analyzer.compare_pair(p, c)
                    rank = (float(match.fusion_score), float(match.confidence))
                except Exception:
                    rank = (_np_cosine(p.embedding, c.embedding), 0.0)
                if rank > best_rank:
                    best_rank = rank
                    best = (p, c)
        return best

    @staticmethod
    def _match_confidence_percent(match: Dict[str, Any]) -> float:
        """Normalize whichever confidence representation is available into a 0-100 score."""
        for key in ("confidence", "fusion_score"):
            value = match.get(key)
            if value is None:
                continue
            try:
                numeric = float(value)
            except Exception:
                continue
            if numeric <= 1.0:
                return round(max(0.0, min(1.0, numeric)) * 100.0, 2)
            return round(max(0.0, min(100.0, numeric)), 2)

        cosine = float(match.get("cosine_similarity", 0.0) or 0.0)
        return round((cosine + 1.0) * 50.0, 2)

    # ------------------------------------------------------------------
    # Safe evidence path
    # ------------------------------------------------------------------
    def _safe_evidence_path(self, applicant_role: str, doc_name: str) -> str:
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        role = "".join(ch for ch in applicant_role if ch.isalnum() or ch in ("_", "-"))
        stem = Path(doc_name).stem
        stem = "".join(ch for ch in stem if ch.isalnum() or ch in ("_", "-"))
        return os.path.join(self.OUTPUT_DIR, f"{role}_{stem}_{ts}_evidence.png")

    @staticmethod
    def _face_evidence_snapshot(face: EmbeddingResult) -> Dict[str, Any]:
        return {
            "source_path": face.source_path,
            "model_name": face.model_name,
            "embedding_norm": round(float(face.embedding_norm), 4),
            "quality": face.quality,
            "detector_confidence": round(float(face.detector_confidence), 4),
            "box": face.box.model_dump() if face.box else {},
            "demographics": to_builtin(face.demographics or {}),
            "liveness": to_builtin(face.liveness or {}),
            "landmark_counts": {
                "kps5": len(((face.landmarks or {}).get("kps5", []) or [])),
                "points_106": len(((face.landmarks or {}).get("points_106", []) or [])),
            },
        }

    # ------------------------------------------------------------------
    # Main pipeline — synchronous, sequential, one model at a time
    # ------------------------------------------------------------------
    def process_applicant(self, applicant: Applicant) -> Dict[str, Any]:
        """
        Full linear pipeline for one applicant:
            Ingest → Biometrics → Forensics → Documents → Reconstruction → Report
        """
        logger.info("Processing applicant: %s", applicant.role)
        run_start_perf = time.perf_counter()
        evidence_dir = self._create_evidence_dir(applicant.role)
        self._maybe_calibrate_quality_threshold()
        run_metadata = self._build_run_metadata(applicant.role, evidence_dir)

        # --- Ingest ---
        ingest_start = time.perf_counter()
        primary_image_paths: List[str] = []
        comparison_docs: List[Tuple[str, str, List[str]]] = []
        comparison_video_paths: List[str] = []
        for doc in applicant.primary_docs:
            primary_image_paths.extend(list(self.extractor.extract_images(doc.file_path)))
        for doc in applicant.comparison_docs:
            paths = list(self.extractor.extract_images(doc.file_path))
            comparison_docs.append((doc.original_filename or doc.file_path, doc.file_path, paths))
            if _is_video_file(doc.file_path):
                comparison_video_paths.append(doc.file_path)
        self._record_stage(
            run_metadata["stage_telemetry"],
            "ingest_documents",
            ingest_start,
            details={
                "primary_images": len(primary_image_paths),
                "comparison_documents": len(comparison_docs),
                "comparison_videos": len(comparison_video_paths),
            },
        )

        # --- Biometrics: extract primary faces ---
        primary_extract_start = time.perf_counter()
        all_primary_faces: List[EmbeddingResult] = []
        primary_warnings: List[str] = []
        for path in primary_image_paths:
            faces, warns = self._extract_biometrics(path)
            all_primary_faces.extend(faces)
            primary_warnings.extend(warns)
        self._record_stage(
            run_metadata["stage_telemetry"],
            "extract_primary_biometrics",
            primary_extract_start,
            status="degraded" if primary_warnings else "ok",
            details={"faces_found": len(all_primary_faces), "warning_count": len(primary_warnings)},
        )

        result: Dict[str, Any] = {
            "role": applicant.role,
            "primary_faces_found": len(all_primary_faces),
            "evidence_dir": evidence_dir,
            "warnings": list(primary_warnings),
            "comparisons": [],
            "run_metadata": run_metadata,
            "runtime_capabilities": run_metadata.get("runtime_capabilities", {}),
        }

        for doc_name, doc_file_path, image_paths in comparison_docs:
            doc_stage_telemetry: List[Dict[str, Any]] = []
            # --- Biometrics: extract comparison faces ---
            comparison_extract_start = time.perf_counter()
            comparison_faces: List[EmbeddingResult] = []
            comp_warnings: List[str] = []
            for path in image_paths:
                faces, warns = self._extract_biometrics(path)
                comparison_faces.extend(faces)
                comp_warnings.extend(warns)
            self._record_stage(
                doc_stage_telemetry,
                "extract_comparison_biometrics",
                comparison_extract_start,
                status="degraded" if comp_warnings else "ok",
                details={"faces_found": len(comparison_faces), "warning_count": len(comp_warnings)},
            )

            doc_row: Dict[str, Any] = {
                "filename": doc_name,
                "faces_found": len(comparison_faces),
                "match": None,
                "forensics": None,
                "document_intelligence": None,
                "reconstruction": None,
                "report": None,
                "explainability": None,
                "warnings": comp_warnings,
                "stage_telemetry": doc_stage_telemetry,
            }

            if not all_primary_faces or not comparison_faces:
                doc_row["warnings"].append("insufficient_faces_for_matching")
                result["comparisons"].append(doc_row)
                continue

            pair_select_start = time.perf_counter()
            best = self._best_pair(all_primary_faces, comparison_faces)
            self._record_stage(
                doc_stage_telemetry,
                "select_best_pair",
                pair_select_start,
                status="ok" if best is not None else "degraded",
                details={
                    "primary_candidates": len(all_primary_faces),
                    "comparison_candidates": len(comparison_faces),
                },
            )
            if best is None:
                doc_row["warnings"].append("unable_to_compute_similarity")
                result["comparisons"].append(doc_row)
                continue

            primary_face, comparison_face = best
            primary_box_dict = primary_face.box.model_dump() if primary_face.box else {"x": 0, "y": 0, "w": 100, "h": 100}
            comparison_box_dict = comparison_face.box.model_dump() if comparison_face.box else {"x": 0, "y": 0, "w": 100, "h": 100}
            doc_row["face_evidence"] = {
                "primary": self._face_evidence_snapshot(primary_face),
                "comparison": self._face_evidence_snapshot(comparison_face),
            }

            # --- Deep Study on Primary Image ---
            logger.info("Running deep primary image study...")
            primary_study_start = time.perf_counter()
            primary_study: Dict[str, Any] = {}
            primary_study_status = "ok"
            try:
                primary_study = self.image_study.analyze(
                    primary_face.source_path, face_box=primary_box_dict
                )
            except Exception as exc:
                primary_study = {"findings": [f"study_failed: {exc}"], "issues_detected": [], "reconstruction_guidance": ""}
                primary_study_status = "degraded"
                logger.warning("Primary image study failed: %s", exc)
            self._record_stage(
                doc_stage_telemetry,
                "primary_image_study",
                primary_study_start,
                status=primary_study_status,
                details={"finding_count": len(primary_study.get("findings", []) or [])},
            )

            comparison_age_evidence: Dict[str, Any] = {}
            comparison_age_start = time.perf_counter()
            comparison_age_status = "ok"
            try:
                comparison_age_evidence = self.image_study.estimate_age_evidence(
                    comparison_face.source_path, face_box=comparison_box_dict
                )
            except Exception as exc:
                comparison_age_evidence = {
                    "warnings": [f"comparison_age_study_failed: {exc}"],
                    "aging_features": {},
                }
                comparison_age_status = "degraded"
                logger.warning("Comparison age evidence failed: %s", exc)
            self._record_stage(
                doc_stage_telemetry,
                "comparison_age_estimation",
                comparison_age_start,
                status=comparison_age_status,
            )

            pri_demo = primary_face.demographics or {}
            comp_demo = comparison_face.demographics or {}
            primary_age_context = self._build_age_context(pri_demo, primary_study)
            comparison_age_context = self._build_age_context(comp_demo, comparison_age_evidence)

            primary_study.setdefault("aging_features", {})
            primary_study["aging_features"].update({
                "consensus_estimated_age": primary_age_context.get("estimated_age"),
                "consensus_confidence": primary_age_context.get("confidence"),
                "age_consistency": primary_age_context.get("consistency"),
                "detector_estimated_age": primary_age_context.get("detector_age"),
            })
            primary_study["age_context"] = primary_age_context
            primary_study.setdefault("findings", []).append(
                "Age Consensus: "
                f"detector={primary_age_context.get('detector_age', 'unknown')}, "
                f"study_range={primary_age_context.get('study_age_range', 'unknown')}, "
                f"resolved_age={primary_age_context.get('estimated_age', 'unknown')}, "
                f"policy={primary_age_context.get('variant_policy', 'disabled')}"
            )
            doc_row["age_context"] = {
                "primary": primary_age_context,
                "comparison": comparison_age_context,
            }

            reconstruction_guidance = primary_study.get("reconstruction_guidance", "")

            # --- Compare ---
            compare_start = time.perf_counter()
            match = self._compare_pair(primary_face, comparison_face)
            self._record_stage(
                doc_stage_telemetry,
                "pair_compare",
                compare_start,
                status="ok" if match.get("verified") is not None else "degraded",
                details={"verified": match.get("verified"), "agreement": match.get("agreement")},
            )
            doc_row["match"] = match
            source_img = comparison_face.source_path
            legacy_evidence_path = self._safe_evidence_path(applicant.role, doc_name)

            # --- Forensics (generates spectral_analysis.jpg + biometric_pulse.png) ---
            logger.info("Running forensics pipeline...")
            forensics_start = time.perf_counter()
            comparison_video_path = doc_file_path if _is_video_file(doc_file_path) else (
                comparison_video_paths[0] if comparison_video_paths else None
            )
            forensics = self._run_forensics(
                source_img,
                video_path=comparison_video_path,
                evidence_dir=evidence_dir,
            )
            self._record_stage(
                doc_stage_telemetry,
                "forensics",
                forensics_start,
                status="degraded" if forensics.get("warnings") else "ok",
            )
            doc_row["forensics"] = forensics

            # --- Document Intelligence (generates tamper_heatmap.jpg) ---
            logger.info("Running document intelligence pipeline...")
            document_start = time.perf_counter()
            document_intelligence = self._run_document_analysis(
                source_img, comparison_face.box, evidence_dir=evidence_dir
            )
            self._record_stage(
                doc_stage_telemetry,
                "document_intelligence",
                document_start,
                status="degraded" if document_intelligence.get("warnings") else "ok",
            )
            doc_row["document_intelligence"] = document_intelligence

            # --- Explainability ---
            gradcam_path = os.path.join(evidence_dir, "gradcam_overlay.png")
            explain_start = time.perf_counter()
            explain = self._generate_explainability(primary_face, comparison_face, gradcam_path)
            self._record_stage(
                doc_stage_telemetry,
                "explainability",
                explain_start,
                status="degraded" if explain.get("error") else "ok",
            )
            doc_row["explainability"] = explain

            # --- Stage 3.5: Advanced Biometric Analysis (7 modules) ---
            logger.info("Running Stage 3.5 — Advanced Biometric Analysis (7 modules)...")
            advanced_biometrics_start = time.perf_counter()
            adv_bio_primary = self._run_advanced_biometrics_single(
                primary_face.source_path, primary_box_dict,
                landmarks=primary_face.landmarks,
            )
            adv_bio_comparison = self._run_advanced_biometrics_single(
                comparison_face.source_path, comparison_box_dict,
                landmarks=comparison_face.landmarks,
            )
            face_match_pct = self._match_confidence_percent(match)
            adv_bio_pair = self._run_advanced_biometrics_pair(
                primary_face.source_path, primary_box_dict,
                comparison_face.source_path, comparison_box_dict,
                face_match_pct,
                primary_landmarks=primary_face.landmarks,
                comparison_landmarks=comparison_face.landmarks,
            )
            self._record_stage(
                doc_stage_telemetry,
                "advanced_biometrics",
                advanced_biometrics_start,
                status="degraded" if adv_bio_pair.get("warnings") else "ok",
                details={"pair_verdict": adv_bio_pair.get("final_verdict", adv_bio_pair.get("verdict"))},
            )
            doc_row["advanced_biometrics"] = {
                "primary": adv_bio_primary,
                "comparison": adv_bio_comparison,
                "pair_analysis": adv_bio_pair,
            }
            logger.info(
                "Advanced Biometrics — primary_threat=%s, comparison_threat=%s, pair_verdict=%s",
                adv_bio_primary.get("threat_level", "N/A"),
                adv_bio_comparison.get("threat_level", "N/A"),
                adv_bio_pair.get("final_verdict", adv_bio_pair.get("verdict", "N/A")),
            )

            # --- Reconstruction (generates reconstruction_hq.jpg) — ALWAYS runs ---
            logger.info("Running reconstruction pipeline (always active)...")
            recon_mode = "deocclusion"
            suspicious = (
                not bool(match.get("verified", False))
                or bool(forensics.get("frequency", {}).get("deepfake_suspected", False))
                or bool(document_intelligence.get("noiseprint", {}).get("suspected_splice", False))
            )
            if not suspicious:
                recon_mode = "age_progression"

            comp_age = comparison_age_context.get("estimated_age")
            comp_sex = comparison_age_context.get("sex") or comp_demo.get("sex")
            pri_age = primary_age_context.get("estimated_age")
            pri_sex = primary_age_context.get("sex") or pri_demo.get("sex")

            recon_save_path = os.path.join(evidence_dir, "reconstruction_hq.jpg")
            reconstruction_start = time.perf_counter()
            reconstruction = self._run_reconstruction(
                source_img,
                comparison_face.embedding,
                mode=recon_mode,
                save_path=recon_save_path,
                reconstruction_guidance=reconstruction_guidance,
                estimated_age=comp_age,
                sex=comp_sex,
                age_context=comparison_age_context,
            )
            self._record_stage(
                doc_stage_telemetry,
                "reconstruction_comparison",
                reconstruction_start,
                status="degraded" if reconstruction.get("warnings") else "ok",
                details={"mode": recon_mode},
            )
            doc_row["reconstruction"] = reconstruction

            # --- Also reconstruct primary image for comparison ---
            primary_recon_path = os.path.join(evidence_dir, "primary_reconstruction_hq.jpg")
            primary_reconstruction = {}
            primary_recon_start = time.perf_counter()
            primary_recon_status = "ok"
            try:
                primary_reconstruction = self._run_reconstruction(
                    primary_face.source_path,
                    primary_face.embedding,
                    mode=recon_mode,
                    save_path=primary_recon_path,
                    reconstruction_guidance=reconstruction_guidance,
                    estimated_age=pri_age,
                    sex=pri_sex,
                    age_context=primary_age_context,
                )
            except Exception:
                primary_recon_status = "degraded"
            self._record_stage(
                doc_stage_telemetry,
                "reconstruction_primary",
                primary_recon_start,
                status=primary_recon_status if not primary_reconstruction.get("warnings") else "degraded",
                details={"mode": recon_mode},
            )

            # --- Expression Suite (capture + transfer + animation) ---
            logger.info("Running expression suite (capture + transfer + animation)...")
            expression_suite_start = time.perf_counter()
            expression_suite = self.run_expression_suite(
                source_image_path=primary_face.source_path,
                expression_image_path=comparison_face.source_path,
                save_path=os.path.join(evidence_dir, "expression_transfer.jpg"),
                transfer_pose=False,
            )
            self._record_stage(
                doc_stage_telemetry,
                "expression_suite",
                expression_suite_start,
                status="degraded" if expression_suite.get("warnings") and not expression_suite.get("generated_image_path") else "ok",
                details={
                    "has_transfer": bool(expression_suite.get("generated_image_path")),
                    "has_animation": bool(expression_suite.get("animation_gif_path")),
                    "has_teaser": bool(expression_suite.get("teaser_gif_path")),
                },
            )
            doc_row["expression_suite"] = expression_suite

            # --- Reporting (Chain-of-Thought with deep image study) ---
            logger.info("Generating forensic report with deep image study...")
            reporting_start = time.perf_counter()
            report = self._run_reporting(
                applicant_id=applicant.role,
                biometrics=match,
                forensics=forensics,
                document=document_intelligence,
                reconstruction=reconstruction,
                primary_image_study=primary_study,
                advanced_biometrics=doc_row.get("advanced_biometrics", {}),
            )
            doc_row["report"] = report
            doc_row["primary_image_study"] = primary_study

            # --- Write FINAL_REPORT.md ---
            self._write_final_report(
                evidence_dir, report, match, primary_study,
                advanced_biometrics=doc_row.get("advanced_biometrics", {}),
            )
            self._record_stage(
                doc_stage_telemetry,
                "reporting",
                reporting_start,
                status="degraded" if report.get("warnings") else "ok",
                details={"verdict": report.get("verdict")},
            )

            # --- Generate Military Dashboard Evidence Card ---
            logger.info("Generating 1920x1080 evidence dashboard...")
            dashboard_start = time.perf_counter()
            dashboard_status = "ok"
            try:
                dashboard_path = self.visualizer.generate_evidence_card(
                    img1_path=primary_face.source_path,
                    img2_path=comparison_face.source_path,
                    face1_box=primary_box_dict,
                    face2_box=comparison_box_dict,
                    match_data=match,
                    applicant_id=applicant.role,
                    forensic_data=forensics,
                    compliance_data=document_intelligence,
                    deepfake_data=forensics.get("frequency", {}),
                    biometric_data=match,
                    advanced_biometrics=doc_row.get("advanced_biometrics", {}),
                    reconstruction_path=recon_save_path,
                    gradcam_path=gradcam_path,
                    evidence_dir=evidence_dir,
                )
                if dashboard_path:
                    doc_row["dashboard"] = dashboard_path
                    logger.info("Dashboard saved: %s", dashboard_path)
            except Exception as exc:
                dashboard_status = "degraded"
                logger.warning("Dashboard generation failed: %s", exc)
            self._record_stage(
                doc_stage_telemetry,
                "dashboard",
                dashboard_start,
                status=dashboard_status,
            )

            # --- 3D Forensic Cross-Validation (Multi-Signal) ---
            logger.info("Running multi-signal forensic cross-validation...")
            cross_validation_start = time.perf_counter()
            cross_validation_status = "ok"
            try:
                from src.forensics.anthropometry import ForensicAnthropometry
                from src.forensics.consistency_checker import ForensicConsistencyChecker
                import numpy as _np

                recon_3d_comp = reconstruction.get("forensic_3d", {})
                recon_3d_prim = primary_reconstruction.get("forensic_3d", {})

                # Cross-image anthropometric comparison
                anthro_cmp = None
                ap = recon_3d_prim.get("anthropometry", {})
                ac = recon_3d_comp.get("anthropometry", {})
                if ap.get("ratios") and ac.get("ratios"):
                    anthro_cmp = ForensicAnthropometry().compare(ap, ac)

                # Cross-image BFM identity coefficient comparison
                bfm_id_cmp = None
                sig_p = recon_3d_prim.get("coefficient_analysis", {}).get(
                    "identity_signature"
                )
                sig_c = recon_3d_comp.get("coefficient_analysis", {}).get(
                    "identity_signature"
                )
                if sig_p and sig_c:
                    _sp = _np.array(sig_p, dtype=_np.float64)
                    _sc = _np.array(sig_c, dtype=_np.float64)
                    _cos = float(
                        _np.dot(_sp, _sc)
                        / (_np.linalg.norm(_sp) * _np.linalg.norm(_sc) + 1e-8)
                    )
                    bfm_id_cmp = {
                        "match": _cos > 0.65,
                        "cosine_similarity": round(_cos, 4),
                        "verdict": (
                            "BFM_IDENTITY_MATCH"
                            if _cos > 0.65
                            else "BFM_IDENTITY_MISMATCH"
                        ),
                        "confidence": (
                            "HIGH" if _cos > 0.85
                            else "MODERATE" if _cos > 0.65
                            else "LOW"
                        ),
                    }

                # Multi-signal consistency analysis
                consistency = ForensicConsistencyChecker().analyze(
                    match_result=match,
                    forensics_result=forensics,
                    document_result=document_intelligence,
                    adv_biometrics=doc_row.get("advanced_biometrics", {}),
                    recon_primary=primary_reconstruction,
                    recon_comparison=reconstruction,
                    anthropometry_cmp=anthro_cmp,
                    bfm_identity_cmp=bfm_id_cmp,
                )

                doc_row["forensic_3d_cross_validation"] = {
                    "anthropometric_comparison": anthro_cmp,
                    "bfm_identity_comparison": bfm_id_cmp,
                    "consistency_analysis": consistency,
                    "primary_anthropometry": ap,
                    "comparison_anthropometry": ac,
                }
                logger.info(
                    "3D Cross-Validation: threat=%s, consistency=%.3f, "
                    "contradictions=%d, agreements=%d",
                    consistency.get("threat_level", "N/A"),
                    consistency.get("consistency_score", 0),
                    consistency.get("contradiction_count", 0),
                    consistency.get("agreement_count", 0),
                )
            except Exception as exc:
                cross_validation_status = "degraded"
                logger.warning("3D cross-validation failed: %s", exc)
            self._record_stage(
                doc_stage_telemetry,
                "forensic_cross_validation",
                cross_validation_start,
                status=cross_validation_status,
            )

            # --- Evidence Chain SHA-256 Integrity Manifest ---
            doc_row["evidence_integrity"] = {"status": "pending_final_manifest"}

            result["comparisons"].append(doc_row)

        # Backward-compatible top-level values
        first = next(
            (
                comparison
                for comparison in result["comparisons"]
                if isinstance(comparison, dict)
            ),
            None,
        )
        if first:
            first_match = first.get("match") or {}
            consistency = (
                first.get("forensic_3d_cross_validation", {})
                .get("consistency_analysis", {})
            )
            threat_level = str(consistency.get("threat_level", "UNKNOWN")).upper()
            contradiction_count = int(consistency.get("contradiction_count", 0) or 0)
            deepfake_flag = bool(first.get("forensics", {}).get("frequency", {}).get("deepfake_suspected", False))
            splice_flag = bool(first.get("document_intelligence", {}).get("noiseprint", {}).get("suspected_splice", False))
            pair_analysis = (
                first.get("advanced_biometrics", {})
                .get("pair_analysis", {})
                or {}
            )
            alteration_context = pair_analysis.get("identity_alteration_context", {}) or {}

            direct_match_confidence = self._match_confidence_percent(first_match)
            raw_confidence = direct_match_confidence
            adjusted_confidence = consistency.get("adjusted_confidence")
            if adjusted_confidence is not None:
                try:
                    raw_confidence = round(float(adjusted_confidence), 2)
                except Exception:
                    pass

            alteration_review_guard = bool(
                first_match.get("verified", False)
                and alteration_context.get("detected")
                and direct_match_confidence >= 80.0
                and not deepfake_flag
                and not splice_flag
                and threat_level == "HIGH"
                and contradiction_count <= 2
            )

            result["is_match"] = bool(
                first_match.get("verified", False)
                and not deepfake_flag
                and not splice_flag
                and (threat_level not in {"HIGH", "CRITICAL"} or alteration_review_guard)
            )

            if contradiction_count:
                result.setdefault("warnings", []).append(
                    f"consistency_checker_detected_{contradiction_count}_contradictions"
                )
            if deepfake_flag:
                result.setdefault("warnings", []).append("deepfake_signal_detected_in_comparison_asset")
            if splice_flag:
                result.setdefault("warnings", []).append("document_splice_signal_detected_in_comparison_asset")
            if threat_level in {"HIGH", "CRITICAL"} and not alteration_review_guard:
                result.setdefault("warnings", []).append(
                    f"forensic_threat_level_{threat_level.lower()}_overrode_public_match_decision"
                )
            if alteration_context.get("detected"):
                result.setdefault("warnings", []).append(
                    "appearance_alteration_context_detected_" + str(alteration_context.get("category", "review"))
                )
                first.setdefault("warnings", []).append(
                    "Appearance alteration context detected: "
                    + str(alteration_context.get("summary", "review stable identity signals"))
                )
            if alteration_review_guard:
                result.setdefault("warnings", []).append(
                    "high_confidence_match_preserved_with_alteration_review"
                )

            # ---- Cataract / Eye-Condition Aware Adjustment ----
            # If iris analysis detects cataracts or severe eye disease in
            # elderly subjects, recognition accuracy degrades.  Add a
            # warning and note the limitation rather than silently returning
            # a low-confidence mismatch.
            eye_condition_notes: List[str] = []
            for side_label, side_bio in [
                ("primary", first.get("advanced_biometrics", {}).get("primary", {})),
                ("comparison", first.get("advanced_biometrics", {}).get("comparison", {})),
            ]:
                if not side_bio:
                    continue
                iris_data = side_bio.get("iris", {})
                health = iris_data.get("health_indicators", {})
                cataract_prob = health.get("cataract_probability", 0.0)
                iris_clarity = health.get("iris_clarity", 1.0)
                arcus = iris_data.get("age_estimation", {}).get("arcus_senilis_detected", False)

                if cataract_prob > 0.4:
                    eye_condition_notes.append(
                        f"{side_label}_cataract_probability_{cataract_prob:.0%}"
                    )
                if iris_clarity < 0.4:
                    eye_condition_notes.append(
                        f"{side_label}_low_iris_clarity_{iris_clarity:.2f}"
                    )
                if arcus:
                    eye_condition_notes.append(f"{side_label}_arcus_senilis_detected")

            if eye_condition_notes:
                result.setdefault("warnings", []).extend(eye_condition_notes)
                result.setdefault("warnings", []).append(
                    "eye_condition_detected_recognition_may_be_degraded"
                )
                # Add a note to the comparison too
                first.setdefault("warnings", []).append(
                    "NOTE: Eye condition (cataract/opacity) detected. "
                    "Facial recognition confidence may be reduced for this subject. "
                    "Refer to age-regressed reconstruction for identity reference."
                )

            result["confidence"] = raw_confidence
        else:
            result["is_match"] = False
            result["confidence"] = 0.0

        run_metadata["completed_utc"] = self._utc_now()
        run_metadata["comparison_count"] = len(result.get("comparisons", []))
        run_metadata["runtime_ms"] = round((time.perf_counter() - run_start_perf) * 1000.0, 2)

        finalization_start = time.perf_counter()
        try:
            evidence_summary = self._finalize_evidence_package(evidence_dir, result)
            self._record_stage(
                run_metadata["stage_telemetry"],
                "finalize_evidence_package",
                finalization_start,
                details={
                    "casefile_path": evidence_summary.get("interactive_casefile", {}).get("html_path"),
                    "manifest_path": evidence_summary.get("manifest_path"),
                },
            )
        except Exception as exc:
            result.setdefault("warnings", []).append(f"evidence_package_finalization_failed: {exc}")
            self._record_stage(
                run_metadata["stage_telemetry"],
                "finalize_evidence_package",
                finalization_start,
                status="degraded",
                details={"error": str(exc)},
            )

        run_metadata["runtime_ms"] = round((time.perf_counter() - run_start_perf) * 1000.0, 2)

        logger.info(
            "Applicant %s processed — is_match=%s, confidence=%.2f, evidence_dir=%s",
            applicant.role, result["is_match"], result["confidence"], evidence_dir,
        )
        return to_builtin(result)

    async def process_applicant_async(self, applicant: Applicant) -> Dict[str, Any]:
        """Async wrapper — runs the synchronous pipeline in a thread."""
        return await asyncio.to_thread(self.process_applicant, applicant)

    def cleanup(self) -> None:
        try:
            self.expression_transfer.cleanup()
        except Exception:
            pass
        try:
            self.expression_suite.cleanup()
        except Exception:
            pass
        deep3d = getattr(self.reconstruction, "_deep3d", None)
        if deep3d is not None:
            try:
                deep3d.cleanup()
            except Exception:
                pass
        self.extractor.cleanup()
        self.analyzer.cleanup()


def _np_cosine(a: List[float], b: List[float]) -> float:
    import numpy as np

    va = np.asarray(a, dtype=np.float32)
    vb = np.asarray(b, dtype=np.float32)
    denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
    if denom <= 1e-8:
        return -1.0
    return float(np.dot(va, vb) / denom)
