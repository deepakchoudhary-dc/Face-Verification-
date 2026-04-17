from __future__ import annotations

import json
import logging
import os
import re
import urllib.request
from typing import Any, Dict, List, Tuple

from src.core.contracts import ReportRequest, ReportResponse
from src.core.serialization import to_builtin

logger = logging.getLogger("ca_monk.llm_analyst")


class LlamaForensicAnalyst:
    """
    Backward-compatible reporting analyst class.

    Despite the legacy class name, this implementation now uses a local
    Ollama model (default: qwen3:1.7b) for report generation.
    """

    SYSTEM_PROMPT = (
        "You are CA_MONK, a Tier-1 Military Forensic Biometric Intelligence Analyst.\n"
        "You receive telemetry from a multi-stage forensic pipeline and must produce a STRICT "
        "JSON verdict with ZERO hallucination.\n\n"
        "YOU MUST THINK BEFORE YOU SPEAK. Use <thinking> tags internally for your chain-of-thought "
        "reasoning, but the final JSON output must NOT contain <thinking> tags.\n\n"
        "=== MANDATORY 8-STEP ANALYSIS PROTOCOL ===\n\n"
        "Step 0 — PRIMARY IMAGE DEEP STUDY:\n"
        "  Analyze primary_image_study. Report: skin condition, detected marks/scars, occlusions,\n"
        "  facial symmetry score, aging features, wrinkle density, quality metrics (blur, noise,\n"
        "  resolution). State what the reconstruction pipeline was guided to fix.\n\n"
        "Step 1 — VISUAL TRIAGE:\n"
        "  Examine embedding_norm. <22 = LOW FIDELITY (evidence unreliable). >28 = HIGH FIDELITY.\n"
        "  State the exact norm value and fidelity tier.\n\n"
        "Step 2 — SPECTRAL SCAN:\n"
        "  Examine deepfake_probability and deepfake_suspected from F3-Net.\n"
        "  State exact probability. If >0.5, declare SPECTRAL ANOMALY.\n\n"
        "Step 3 — BIOLOGICAL SCAN:\n"
        "  Examine rPPG: signal_state, is_live, bpm, confidence. If signal_state=not_available,\n"
        "  state that video liveness evidence was unavailable. Only flag NON-LIVING ARTIFACT when\n"
        "  a liveness signal was actually attempted and failed.\n\n"
        "Step 4 — STRUCTURAL INTEGRITY:\n"
        "  Examine cosine_similarity, verified flag, NoisePrint splice detection.\n"
        "  Cosine <0.3 = STRUCTURAL FAILURE. State exact cosine.\n\n"
        "Step 5 — ADVANCED BIOMETRICS (Stage 3.5):\n"
        "  Examine advanced_biometrics payload. Report:\n"
        "  - Threat levels (primary, comparison)\n"
        "  - Tampering/morphing/disguise detection results\n"
        "  - Micro-seam boundary findings and seam box regions if present\n"
        "  - Doppelganger analysis verdict\n"
        "  - Kinship / bloodline similarity findings\n"
        "  - Iris spoofing indicators and sclera vascular AI-noise signals\n"
        "  - Uniqueness score and facial marker count\n"
        "  - Pair verdict and confidence\n"
        "  ANY critical threat level = mandatory FLAGGED verdict.\n\n"
        "Step 6 — RECONSTRUCTION ASSESSMENT:\n"
        "  Describe what Deep3D Forensic Pipeline (MediaPipe + CodeFormer) reconstructed.\n"
        "  Was identity preserved? What artifacts remain?\n\n"
        "Step 7 — RUTHLESS EXECUTIVE SUMMARY (3 paragraphs):\n"
        "  Para 1: What the data PROVES (cite exact numbers).\n"
        "  Para 2: What the data CANNOT prove (gaps, missing signals).\n"
        "  Para 3: Operational recommendation — CLEARED / FLAGGED / Inconclusive.\n"
        "  No hedging. No 'may' or 'might'. State facts or state unknowns.\n\n"
        "=== OUTPUT FORMAT ===\n"
        "Return ONLY valid JSON with these exact keys:\n"
        "{\n"
        '  "summary": "3-paragraph ruthless executive summary",\n'
        '  "verdict": "CLEARED|FLAGGED|Inconclusive|Conclusive Match|Fraud Attempt",\n'
        '  "confidence": 0.0-1.0,\n'
        '  "reasoning_steps": ["Step 0: ...", "Step 1: ...", ... "Step 7: ..."]\n'
        "}\n"
        "reasoning_steps must be a list of 8 strings (Steps 0-7).\n"
        "Include EXACT numeric values from telemetry in every step.\n"
        "NEVER invent data not present in the payload."
    )

    ALLOWED_VERDICTS = {
        "Conclusive Match",
        "Inconclusive",
        "Fraud Attempt",
        "CLEARED",
        "FLAGGED",
    }

    def __init__(self, model_name: str | None = None, base_url: str | None = None) -> None:
        self.model_name = model_name or os.getenv("OLLAMA_MODEL", "qwen3:1.7b")
        self.base_url = (base_url or os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")).rstrip("/")
        self.timeout_seconds = float(os.getenv("OLLAMA_TIMEOUT_SECONDS", "120"))
        self.num_ctx = int(os.getenv("OLLAMA_NUM_CTX", os.getenv("LLAMA_N_CTX", "4096")))
        self.temperature = float(os.getenv("OLLAMA_TEMPERATURE", "0.1"))
        self.healthcheck_enabled = os.getenv("OLLAMA_HEALTHCHECK_ENABLED", "1").strip().lower() not in {
            "0",
            "false",
            "no",
            "off",
        }
        self.healthcheck_timeout_seconds = float(
            os.getenv("OLLAMA_HEALTHCHECK_TIMEOUT_SECONDS", "2.5")
        )
        self.healthcheck_retries = max(
            1,
            int(os.getenv("OLLAMA_HEALTHCHECK_RETRIES", "1")),
        )
        self.llm_available = self._healthcheck()
        logger.info(
            "LlamaForensicAnalyst initialized with Ollama model=%s base_url=%s available=%s",
            self.model_name,
            self.base_url,
            self.llm_available,
        )

    def _healthcheck(self) -> bool:
        """Check Ollama reachability with lightweight retry."""
        import time

        if not self.healthcheck_enabled:
            logger.info("Ollama healthcheck disabled; using deterministic report fallback.")
            return False

        url = f"{self.base_url}/api/tags"
        max_retries = self.healthcheck_retries
        for attempt in range(max_retries):
            try:
                req = urllib.request.Request(url, method="GET")
                with urllib.request.urlopen(req, timeout=self.healthcheck_timeout_seconds) as resp:
                    payload = json.loads(resp.read().decode("utf-8"))
                models = payload.get("models", [])
                for model in models:
                    name = str(model.get("name", ""))
                    if name == self.model_name:
                        return True
                    if ":" in name and ":" in self.model_name:
                        if name.split(":", 1)[0] == self.model_name.split(":", 1)[0]:
                            return True
                logger.warning("Ollama is reachable, but model '%s' is not listed.", self.model_name)
                return False
            except Exception as exc:
                if attempt < max_retries - 1:
                    logger.debug(
                        "Ollama healthcheck attempt %d failed: %s. Retrying in 0.5s...",
                        attempt + 1,
                        exc,
                    )
                    time.sleep(0.5)
                else:
                    logger.warning("Ollama healthcheck failed after %d attempts: %s", max_retries, exc)
                    return False
        return False

    def _chat(self, prompt: str) -> str:
        url = f"{self.base_url}/api/chat"
        body = {
            "model": self.model_name,
            "stream": False,
            "messages": [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "options": {
                "temperature": self.temperature,
                "num_ctx": self.num_ctx,
            },
        }
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout_seconds) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        message = payload.get("message", {})
        text = str(message.get("content", "")).strip()
        if not text:
            raise ValueError("Empty response received from Ollama chat API.")
        return text

    def _fallback(self, payload: Dict[str, Any]) -> Tuple[str, float, List[str]]:
        """Deterministic rubric used when the Ollama model is unavailable."""
        bio = payload.get("biometrics", {})
        forensics = payload.get("forensics", {})
        doc = payload.get("document", {})
        study = payload.get("primary_image_study", {})
        adv_bio = payload.get("advanced_biometrics", {})

        verified = bool(bio.get("verified", False))
        cosine = float(bio.get("cosine_similarity", 0.0))
        norm = float(bio.get("embedding_norm", 25.0))
        rppg = forensics.get("rppg", {})
        signal_state = str(rppg.get("signal_state", "unknown"))
        live = None if signal_state == "not_available" else bool(rppg.get("is_live", False))
        bpm = rppg.get("bpm")
        deepfake = bool(forensics.get("frequency", {}).get("deepfake_suspected", False))
        df_prob = float(forensics.get("frequency", {}).get("deepfake_probability", 0.0))
        splice = bool(doc.get("noiseprint", {}).get("suspected_splice", False))

        if norm < 22.0:
            q_tier = "LOW FIDELITY"
        elif norm > 28.0:
            q_tier = "HIGH FIDELITY"
        else:
            q_tier = "STANDARD"

        # --- Step 0: Primary Image Deep Study ---
        study_findings = study.get("findings", [])
        issues = study.get("issues_detected", [])
        marks = study.get("marks_and_injuries", {})
        quality = study.get("quality", {})
        skin = study.get("skin_analysis", {})
        occlusions = study.get("occlusions", {})
        symmetry = study.get("symmetry", {})
        lighting = study.get("lighting", {})
        aging = study.get("aging_features", {})
        guidance = study.get("reconstruction_guidance", "")

        study_lines = ["Step 0 — PRIMARY IMAGE DEEP STUDY:"]
        if study_findings:
            for f in study_findings:
                study_lines.append(f"  - {f}")
        else:
            study_lines.append("  - No detailed study data available.")
        if issues:
            study_lines.append(f"  Issues detected: {', '.join(str(i) for i in issues)}")
        if marks.get("total_marks", 0) > 0:
            study_lines.append(
                f"  Facial marks: {marks.get('total_marks', 0)} total "
                f"(scars={marks.get('scar_like_regions', 0)}, "
                f"dark_spots={marks.get('dark_spots', 0)}, "
                f"red_spots={marks.get('red_spots', 0)})"
            )
        if quality.get("blur_score"):
            study_lines.append(
                f"  Quality: blur={quality.get('blur_score', 0):.1f}, "
                f"noise={quality.get('noise_level', 0):.1f}, "
                f"resolution={quality.get('resolution', 'unknown')}"
            )
        if symmetry.get("symmetry_score"):
            study_lines.append(f"  Symmetry: {symmetry.get('symmetry_score', 0):.2f}")
        if aging.get("estimated_age_range"):
            study_lines.append(f"  Estimated age range: {aging.get('estimated_age_range', 'unknown')}")

        step_study = "\n".join(study_lines)

        # --- Step 5: Advanced Biometrics ---
        adv_primary = adv_bio.get("primary", {})
        adv_comp = adv_bio.get("comparison", {})
        adv_pair = adv_bio.get("pair_analysis", {})
        primary_threat = adv_primary.get("threat_level", "N/A")
        comp_threat = adv_comp.get("threat_level", "N/A")
        pair_verdict = adv_pair.get("final_verdict", adv_pair.get("verdict", "N/A"))
        pair_conf = adv_pair.get("confidence", 0)
        tampered = adv_primary.get("tampering", {}).get("tampering_detected", False)
        morphed = adv_primary.get("morphing", {}).get("is_morphed", False)
        disguised = adv_primary.get("makeup_disguise", {}).get("disguise_detected", False)
        is_doppel = adv_pair.get("doppelganger_analysis", {}).get("is_doppelganger", False)
        seam = adv_primary.get("tampering", {}).get("micro_seam_analysis", {})
        seam_prob = float(seam.get("seam_probability", 0.0))
        seam_regions = len(seam.get("candidate_regions", []))
        kinship = adv_pair.get("kinship_analysis", {})
        kinship_prob = float(kinship.get("kinship_probability", 0.0))
        kinship_label = kinship.get("relationship_hypothesis", "not_indicated")
        sclera = adv_primary.get("iris", {}).get("sclera_analysis", {})
        sclera_ai = bool(sclera.get("deepfake_suspected", False))
        sclera_noise = float(sclera.get("ai_noise_probability", 0.0))

        adv_bio_lines = [
            f"Step 5 — ADVANCED BIOMETRICS:",
            f"  Primary threat level: {primary_threat} (score={adv_primary.get('threat_score', 0):.3f})",
            f"  Comparison threat level: {comp_threat} (score={adv_comp.get('threat_score', 0):.3f})",
            f"  Tampering: {'DETECTED' if tampered else 'CLEAR'}",
            f"  Micro-seam boundary: probability={seam_prob:.3f}, regions={seam_regions}",
            f"  Morphing: {'DETECTED' if morphed else 'CLEAR'}",
            f"  Disguise/Makeup: {'DETECTED' if disguised else 'CLEAR'}",
            f"  Doppelganger: {'SUSPECTED' if is_doppel else 'CLEAR'}",
            f"  Iris spoofing: {'SUSPECTED' if adv_primary.get('iris', {}).get('anti_spoofing', {}).get('contact_lens_detected') else 'CLEAR'}",
            f"  Sclera vascular AI noise: probability={sclera_noise:.3f}, suspected={sclera_ai}",
            f"  Kinship signal: {kinship_label}, probability={kinship_prob:.2f}%",
            f"  Uniqueness score: {adv_primary.get('uniqueness', {}).get('uniqueness_score', 0):.2f}",
            f"  Facial markers detected: {adv_primary.get('facial_markers', {}).get('markers_detected', 0)}",
            f"  Pair verdict: {pair_verdict}, confidence: {pair_conf:.2f}%",
        ]
        step_adv_bio = "\n".join(adv_bio_lines)

        # Check if any critical biometric threat
        critical_bio_threat = primary_threat == "CRITICAL" or comp_threat == "CRITICAL"

        steps = [
            step_study,
            f"Step 1 — VISUAL TRIAGE: MagFace norm={norm:.1f} -> {q_tier}. "
            f"Evidence {'reliable' if norm >= 22 else 'unreliable due to low quality capture'}.",
            f"Step 2 — SPECTRAL SCAN: F3-Net deepfake_probability={df_prob:.3f}, "
            f"deepfake_suspected={deepfake}. "
            f"{'SPECTRAL ANOMALY detected.' if deepfake else 'No frequency anomalies.'}",
            (
                f"Step 3 — BIOLOGICAL SCAN: rPPG state={signal_state}, is_live={live}, BPM={bpm}. "
                "Video liveness evidence unavailable for this case."
                if signal_state == "not_available"
                else f"Step 3 — BIOLOGICAL SCAN: rPPG state={signal_state}, is_live={live}, BPM={bpm}. "
                f"{'Physiological signal confirms living subject.' if live else 'NON-LIVING ARTIFACT — no pulse detected; possible spoof/print/mask.'}"
            ),
            f"Step 4 — STRUCTURAL INTEGRITY: AdaFace cosine={cosine:.3f}, verified={verified}, "
            f"splice_suspected={splice}. "
            f"{'Document-face coherence confirmed.' if verified and not splice else 'STRUCTURAL FAILURE — coherence issues detected.'}",
            step_adv_bio,
            f"Step 6 — RECONSTRUCTION ASSESSMENT: Guidance: {guidance[:200] if guidance else 'Standard reconstruction.'}",
        ]

        live_clear = live is not False
        if verified and live_clear and not deepfake and not splice and norm >= 22.0 and not critical_bio_threat:
            verdict = "CLEARED"
            confidence = max(0.80, min(0.98, (cosine + 1.0) / 2.0))
            steps.append(
                f"Step 7 — RUTHLESS EXECUTIVE SUMMARY: All pipeline stages passed. "
                f"Cosine={cosine:.3f}, live_state={signal_state}, no spectral anomalies, no splice, "
                f"biometric threat={primary_threat}. Subject CLEARED with confidence={confidence:.2f}."
            )
        elif deepfake or splice or live is False or norm < 20.0 or critical_bio_threat or tampered or morphed or sclera_ai:
            verdict = "FLAGGED"
            confidence = 0.85
            flags = []
            if deepfake:
                flags.append("deepfake")
            if splice:
                flags.append("splice")
            if live is False:
                flags.append("non-living")
            if norm < 20.0:
                flags.append("low-fidelity")
            if critical_bio_threat:
                flags.append(f"critical-bio-threat({primary_threat})")
            if tampered:
                flags.append("tampering")
            if morphed:
                flags.append("morphing-attack")
            if sclera_ai:
                flags.append("sclera-ai-noise")
            steps.append(
                f"Step 7 — RUTHLESS EXECUTIVE SUMMARY: FLAGGED. "
                f"Red flags: {', '.join(flags)}. "
                f"Biometric pair verdict: {pair_verdict}. "
                f"This subject requires immediate manual review. Confidence={confidence:.2f}."
            )
        else:
            verdict = "Inconclusive"
            confidence = 0.60
            steps.append(
                f"Step 7 — RUTHLESS EXECUTIVE SUMMARY: Insufficient evidence for definitive verdict. "
                f"Cosine={cosine:.3f}, bio_threat={primary_threat}, pair_verdict={pair_verdict}. "
                f"Recommend additional verification. Confidence={confidence:.2f}."
            )

        return verdict, float(confidence), steps

    def _extract_json(self, text: str) -> Dict[str, Any]:
        text = text.strip()
        if text.startswith("```"):
            text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
        try:
            return json.loads(text)
        except Exception:
            pass
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise ValueError("No JSON object found in LLM response.")
        return json.loads(match.group(0))

    def generate(self, req: ReportRequest) -> ReportResponse:
        payload = to_builtin(req.model_dump())
        fallback_verdict, fallback_conf, fallback_steps = self._fallback(payload)

        if not self.llm_available:
            logger.warning("Ollama model unavailable. Using deterministic fallback rubric.")
            return ReportResponse(
                summary=(
                    "CA_MONK FORENSIC DOSSIER - Generated via deterministic rubric. "
                    "Local Ollama model unavailable."
                ),
                verdict=fallback_verdict,
                confidence=fallback_conf,
                reasoning_steps=fallback_steps,
            )

        prompt = (
            "=== CA_MONK FORENSIC TELEMETRY PAYLOAD ===\n"
            f"{json.dumps(payload, indent=2, ensure_ascii=False)}\n\n"
            "Analyze the payload with the 8-step protocol and return strict JSON."
        )

        try:
            logger.info("Generating forensic report via Ollama model '%s'...", self.model_name)
            text = self._chat(prompt)
            parsed = self._extract_json(text)
            verdict = str(parsed.get("verdict", fallback_verdict))
            use_fallback_reasoning = False
            if verdict not in self.ALLOWED_VERDICTS:
                logger.warning("LLM returned invalid verdict '%s'. Falling back.", verdict)
                verdict = fallback_verdict
                use_fallback_reasoning = True
            confidence = float(parsed.get("confidence", fallback_conf))
            confidence = max(0.0, min(1.0, confidence))
            if use_fallback_reasoning:
                confidence = fallback_conf
            steps = parsed.get("reasoning_steps", fallback_steps)
            if use_fallback_reasoning:
                steps = fallback_steps
            if not isinstance(steps, list) or not steps:
                steps = fallback_steps
            logger.info("Report generated - verdict=%s confidence=%.2f", verdict, confidence)
            return ReportResponse(
                summary=str(parsed.get("summary", "")) or "CA_MONK forensic dossier generated.",
                verdict=verdict,
                confidence=confidence,
                reasoning_steps=[str(s) for s in steps],
            )
        except Exception as exc:
            logger.error("LLM generation failed: %s. Using deterministic fallback.", exc)
            return ReportResponse(
                summary=(
                    "CA_MONK FORENSIC DOSSIER - Deterministic fallback due to LLM failure. "
                    "Reasoning approximated from raw telemetry."
                ),
                verdict=fallback_verdict,
                confidence=fallback_conf,
                reasoning_steps=fallback_steps,
            )
