from __future__ import annotations

import logging
import os
from typing import Optional

import cv2

from src.core.contracts import (
    ForensicsRequest,
    ForensicsResponse,
    FrequencyResult,
    RPPGResult,
)
from src.forensics.f3net_detector import FrequencyAwareDeepfakeDetector
from src.forensics.rppg_liveness import RPPGLivenessDetector

logger = logging.getLogger("ca_monk.forensics")


class ForensicsService:
    """
    Forensics pipeline:
      1. Frequency‐domain deepfake detection (F3-Net Lite DCT)
      2. rPPG liveness / heartbeat extraction
      3. Artifact generation: spectral_analysis.jpg + biometric_pulse.png
    """

    def __init__(self) -> None:
        self.deepfake = FrequencyAwareDeepfakeDetector()
        self.rppg = RPPGLivenessDetector()

    def analyze(
        self,
        req: ForensicsRequest,
        evidence_dir: Optional[str] = None,
    ) -> ForensicsResponse:
        warnings: list[str] = []

        image = cv2.imread(req.image_path)
        if image is None:
            warnings.append(f"image_not_readable: {req.image_path}")
            freq = FrequencyResult(
                deepfake_probability=0.0, deepfake_suspected=False, model_name="unavailable"
            )
        else:
            prob = self.deepfake.predict_probability(image)
            freq = FrequencyResult(
                deepfake_probability=float(prob),
                deepfake_suspected=bool(prob >= self.deepfake.threshold),
            )

            # Generate spectral ghost image for evidence
            if evidence_dir:
                try:
                    spectral_path = os.path.join(evidence_dir, "spectral_analysis.jpg")
                    self.deepfake.generate_spectral_heatmap(image, save_path=spectral_path)
                except Exception as exc:
                    warnings.append(f"spectral_heatmap_failed: {exc}")

        if req.video_path:
            est = self.rppg.estimate(req.video_path)
            variance = float(est.details.get("bpm_variance", 0.0))
            bpm_val = float(est.bpm) if est.bpm is not None else None
            spoof = bool(variance == 0.0 or (bpm_val is not None and bpm_val > 180.0))
            if spoof:
                warnings.append("rppg_spoof_pattern_detected")
            rppg_res = RPPGResult(
                is_live=bool(est.is_live and not spoof),
                bpm=bpm_val,
                confidence=float(est.confidence if not spoof else min(est.confidence, 0.2)),
                method="POS",
                signal_state="spoof" if spoof else "live" if est.is_live else "indeterminate",
                details=dict(est.details),
            )

            # Generate pulse graph for evidence
            if evidence_dir:
                try:
                    pulse_path = os.path.join(evidence_dir, "biometric_pulse.png")
                    self.rppg.generate_pulse_graph(est, save_path=pulse_path)
                except Exception as exc:
                    warnings.append(f"pulse_graph_failed: {exc}")
        else:
            warnings.append("video_not_provided_rppg_not_available")
            # Still generate a "no signal" pulse graph for completeness
            from src.forensics.liveness import RPPGEstimate
            no_signal = RPPGEstimate(False, None, 0.0, {"route": "no_video"}, bpm_series=[])
            rppg_res = RPPGResult(
                is_live=False,
                bpm=None,
                confidence=0.0,
                method="POS",
                signal_state="not_available",
                details={"reason": "missing_video", "availability": "not_available"},
            )
            if evidence_dir:
                try:
                    pulse_path = os.path.join(evidence_dir, "biometric_pulse.png")
                    self.rppg.generate_pulse_graph(no_signal, save_path=pulse_path)
                except Exception as exc:
                    warnings.append(f"pulse_graph_failed: {exc}")

        return ForensicsResponse(frequency=freq, rppg=rppg_res, warnings=warnings)
