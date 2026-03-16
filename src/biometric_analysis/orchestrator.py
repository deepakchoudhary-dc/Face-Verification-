"""
CA_MONK v4 — ADVANCED BIOMETRIC ORCHESTRATOR
=============================================
Stage 3.5 in the verification pipeline.

Wraps BiometricAnalysisSuite with:
- Per-module timing and error isolation
- Structured threat-level assessment
- Single-image and pair-comparison modes
- Engine-compatible Dict output for reporting

Pipeline position:
    Biometrics (Stage 2) → Forensics (Stage 3) → **Advanced Biometrics (Stage 3.5)**
    → Reconstruction (Stage 4) → Reporting (Stage 5)
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

logger = logging.getLogger("ca_monk.adv_biometrics")


class AdvancedBiometricOrchestrator:
    """
    Orchestrates all 7 biometric sub-analyzers with fault isolation
    and unified threat scoring.
    """

    # Module weights for unified threat scoring
    THREAT_WEIGHTS = {
        "tampering": 0.25,
        "morphing": 0.20,
        "makeup_disguise": 0.15,
        "iris": 0.10,
        "uniqueness": 0.10,
        "facial_markers": 0.10,
        "age_invariant": 0.10,
    }

    def __init__(self) -> None:
        from src.biometric_analysis import BiometricAnalysisSuite

        self.suite = BiometricAnalysisSuite()
        logger.info(
            "AdvancedBiometricOrchestrator initialized — 7 modules armed."
        )

    # ------------------------------------------------------------------
    # Single-image analysis
    # ------------------------------------------------------------------
    def analyze_single(
        self,
        image: np.ndarray,
        face_box: Dict[str, int],
        landmarks: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """
        Run the full 7-module biometric sweep on a single face crop.

        Returns a dictionary with per-module results, unified threat score,
        alerts, and timing information.
        """
        t0 = time.perf_counter()
        result: Dict[str, Any] = {
            "modules_run": [],
            "modules_failed": [],
            "alerts": [],
            "threat_score": 0.0,
            "threat_level": "LOW",
            "overall_confidence": 0.0,
            "age_invariant": {},
            "tampering": {},
            "makeup_disguise": {},
            "iris": {},
            "uniqueness": {},
            "facial_markers": {},
            "morphing": {},
            "timing_ms": 0.0,
        }

        if image is None:
            result["modules_failed"].append("ALL — null image")
            return result

        try:
            raw = self.suite.full_analysis(image, face_box, landmarks)
        except Exception as exc:
            logger.error("BiometricAnalysisSuite.full_analysis crashed: %s", exc)
            result["modules_failed"].append(f"suite_crash: {exc}")
            return result

        # Propagate per-module results
        for key in (
            "age_invariant", "tampering", "makeup_disguise",
            "iris", "uniqueness", "facial_markers", "morphing",
        ):
            mod_result = raw.get(key, {})
            result[key] = mod_result
            if isinstance(mod_result, dict) and mod_result and "error" not in mod_result:
                result["modules_run"].append(key)
            else:
                result["modules_failed"].append(key)

        # Propagate alerts
        result["alerts"] = raw.get("alerts", [])
        result["overall_confidence"] = raw.get("overall_confidence", 0.0)

        # Compute unified threat score
        threat = self._compute_threat_score(result)
        result["threat_score"] = threat
        result["threat_level"] = self._classify_threat(threat)

        elapsed = (time.perf_counter() - t0) * 1000
        result["timing_ms"] = round(elapsed, 1)
        logger.info(
            "Single-image biometric sweep complete — threat=%s (%.2f), %d modules, %.0fms",
            result["threat_level"], threat, len(result["modules_run"]), elapsed,
        )
        return result

    # ------------------------------------------------------------------
    # Pair comparison
    # ------------------------------------------------------------------
    def compare_pair(
        self,
        image1: np.ndarray,
        face_box1: Dict[str, int],
        image2: np.ndarray,
        face_box2: Dict[str, int],
        face_match_score: float,
        landmarks1: Optional[Dict] = None,
        landmarks2: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """
        Run full cross-image biometric comparison using the suite's
        compare_faces method, then augment with threat scoring.
        """
        t0 = time.perf_counter()

        try:
            raw = self.suite.compare_faces(
                image1, face_box1,
                image2, face_box2,
                face_match_score,
                landmarks1, landmarks2,
            )
        except Exception as exc:
            logger.error("BiometricAnalysisSuite.compare_faces crashed: %s", exc)
            return {
                "error": str(exc),
                "verdict": "MANUAL_REVIEW",
                "confidence": 0.0,
                "threat_level": "UNKNOWN",
            }

        # Augment with threat classification
        threat = 0.0
        if raw.get("doppelganger_analysis", {}).get("is_doppelganger"):
            threat += 0.30
        if raw.get("morphing_check", {}).get("is_morphed"):
            threat += 0.40
        verdict = raw.get("final_verdict", "MANUAL_REVIEW")
        if verdict == "REJECT":
            threat += 0.20

        raw["threat_score"] = round(min(threat, 1.0), 2)
        raw["threat_level"] = self._classify_threat(raw["threat_score"])

        elapsed = (time.perf_counter() - t0) * 1000
        raw["timing_ms"] = round(elapsed, 1)
        logger.info(
            "Pair biometric comparison — verdict=%s, threat=%s, %.0fms",
            verdict, raw["threat_level"], elapsed,
        )
        return raw

    # ------------------------------------------------------------------
    # Threat scoring
    # ------------------------------------------------------------------
    def _compute_threat_score(self, result: Dict[str, Any]) -> float:
        """
        Weighted threat score across all modules.
        Each module contributes a probability-like signal.
        """
        score = 0.0

        # Tampering
        tamp = result.get("tampering", {})
        if tamp.get("tampering_detected") or tamp.get("is_tampered"):
            score += self.THREAT_WEIGHTS["tampering"] * tamp.get("tampering_probability", 0.8)

        # Morphing
        morph = result.get("morphing", {})
        if morph.get("is_morphed"):
            score += self.THREAT_WEIGHTS["morphing"] * morph.get("morphing_probability", 0.8)

        # Makeup / Disguise
        makeup = result.get("makeup_disguise", {})
        if makeup.get("disguise_detected"):
            score += self.THREAT_WEIGHTS["makeup_disguise"] * makeup.get("disguise_probability", 0.7)

        # Iris spoofing
        iris = result.get("iris", {})
        spoof = iris.get("anti_spoofing", {})
        if spoof.get("contact_lens_detected") or not spoof.get("is_real_eye", True):
            score += self.THREAT_WEIGHTS["iris"]

        # Low uniqueness = might be generic/generated face
        uniq = result.get("uniqueness", {})
        uniq_score = uniq.get("uniqueness_score", 0.5)
        if uniq_score < 0.3:
            score += self.THREAT_WEIGHTS["uniqueness"] * (1.0 - uniq_score)

        return round(min(score, 1.0), 3)

    @staticmethod
    def _classify_threat(score: float) -> str:
        if score >= 0.60:
            return "CRITICAL"
        if score >= 0.40:
            return "HIGH"
        if score >= 0.20:
            return "MEDIUM"
        return "LOW"

    # ------------------------------------------------------------------
    # Utility: load image from path and extract face crop
    # ------------------------------------------------------------------
    @staticmethod
    def crop_face(image_path: str, face_box: Dict[str, int]) -> Optional[np.ndarray]:
        """Load image and crop to face region with 20% padding."""
        img = cv2.imread(image_path)
        if img is None:
            return None
        h, w = img.shape[:2]
        x = max(0, face_box["x"] - int(face_box["w"] * 0.1))
        y = max(0, face_box["y"] - int(face_box["h"] * 0.1))
        x2 = min(w, face_box["x"] + face_box["w"] + int(face_box["w"] * 0.1))
        y2 = min(h, face_box["y"] + face_box["h"] + int(face_box["h"] * 0.1))
        return img[y:y2, x:x2]
