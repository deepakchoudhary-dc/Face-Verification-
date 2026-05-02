"""
MULTI-SIGNAL FORENSIC CONSISTENCY CHECKER — CA_MONK v6.1
====================================================
Cross-validates ALL forensic, biometric, and reconstruction
signals to detect contradictions, compute calibrated confidence,
and produce a unified threat assessment.

Analyzes agreement / disagreement between:
    1. Neural embedding match  vs  BFM identity match
    2. Neural embedding match  vs  Anthropometric compatibility
    3. F3-Net deepfake score   vs  NoisePrint splice detection
    4. Age-invariant match     vs  Neural match
    5. BFM coefficient auth    vs  all forensic signals
    6. Primary image quality   vs  reconstruction quality

Output:
    - Per-channel consistency scores
    - Contradiction list with severity
    - Calibrated overall confidence (adjusted from raw cosine)
    - Unified threat level (CLEAR / ELEVATED / HIGH / CRITICAL)

Author: CA_MONK Forensic Intelligence Unit
Version: 6.1.0
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger("ca_monk.consistency")

# ============================================================================
#  Signal weight configuration
# ============================================================================
_WEIGHTS = {
    "neural_vs_bfm":           0.20,
    "neural_vs_anthropometry": 0.15,
    "forensic_agreement":      0.20,
    "biometric_consistency":   0.15,
    "coefficient_authenticity": 0.15,
    "cross_modal_agreement":   0.15,
}


class ForensicConsistencyChecker:
    """
    Cross-validates all forensic signals and produces a unified
    consistency report with confidence adjustments.
    """

    def analyze(
        self,
        match_result: Dict[str, Any],
        forensics_result: Dict[str, Any],
        document_result: Optional[Dict[str, Any]],
        adv_biometrics: Dict[str, Any],
        recon_primary: Dict[str, Any],
        recon_comparison: Dict[str, Any],
        anthropometry_cmp: Optional[Dict[str, Any]] = None,
        bfm_identity_cmp: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Run full consistency analysis.

        Returns dict with: checks[], contradictions[], agreements[],
            consistency_score, adjusted_confidence, threat_level
        """
        checks: List[Dict[str, Any]] = []
        contradictions: List[str] = []
        agreements: List[str] = []

        # ---- 1. Neural vs BFM identity ----
        neural_match = bool(match_result.get("verified", False))
        neural_cos = float(match_result.get("cosine_similarity", 0.0))

        bfm_match = None
        bfm_cos = None
        if bfm_identity_cmp:
            bfm_match = bfm_identity_cmp.get("match")
            bfm_cos = bfm_identity_cmp.get("cosine_similarity")

        ch1 = self._check_neural_vs_bfm(neural_match, neural_cos, bfm_match, bfm_cos)
        checks.append(ch1)
        if ch1.get("contradiction"):
            contradictions.append(ch1["detail"])
        elif ch1.get("agreement"):
            agreements.append(ch1["detail"])

        # ---- 2. Neural vs Anthropometry ----
        ch2 = self._check_neural_vs_anthro(neural_match, anthropometry_cmp)
        checks.append(ch2)
        if ch2.get("contradiction"):
            contradictions.append(ch2["detail"])
        elif ch2.get("agreement"):
            agreements.append(ch2["detail"])

        # ---- 3. Forensic signal agreement ----
        ch3 = self._check_forensic_signals(forensics_result, document_result or {}, match_result)
        checks.append(ch3)
        if ch3.get("contradiction"):
            contradictions.append(ch3["detail"])
        elif ch3.get("agreement"):
            agreements.append(ch3["detail"])

        # ---- 4. Biometric consistency ----
        ch4 = self._check_biometric_consistency(adv_biometrics, neural_match)
        checks.append(ch4)
        if ch4.get("contradiction"):
            contradictions.append(ch4["detail"])
        elif ch4.get("agreement"):
            agreements.append(ch4["detail"])

        # ---- 5. Coefficient authenticity ----
        ch5_a = self._check_coefficient_auth(recon_primary, "primary")
        ch5_b = self._check_coefficient_auth(recon_comparison, "comparison")
        checks.extend([ch5_a, ch5_b])
        for ch in [ch5_a, ch5_b]:
            if ch.get("contradiction"):
                contradictions.append(ch["detail"])
            elif ch.get("agreement"):
                agreements.append(ch["detail"])

        # ---- 6. Cross-modal agreement ----
        ch6 = self._cross_modal(
            neural_match, bfm_match, anthropometry_cmp, adv_biometrics,
        )
        checks.append(ch6)
        if ch6.get("contradiction"):
            contradictions.append(ch6["detail"])
        elif ch6.get("agreement"):
            agreements.append(ch6["detail"])

        # ---- Compute unified scores ----
        scores = [c.get("score", 0.5) for c in checks if "score" in c]
        consistency_score = float(np.mean(scores)) if scores else 0.5

        # Adjust raw neural confidence based on multi-signal consistency
        raw_conf = (neural_cos + 1.0) * 50.0  # map [-1,1] → [0,100]
        n_contra = len(contradictions)
        n_agree = len(agreements)

        if n_contra == 0 and n_agree >= 3:
            adj = min(raw_conf * 1.15, 99.0)
        elif n_contra >= 3:
            adj = raw_conf * 0.6
        elif n_contra >= 1:
            adj = raw_conf * 0.85
        else:
            adj = raw_conf
        adjusted_confidence = round(float(np.clip(adj, 0, 99)), 2)

        # Threat level
        if n_contra >= 3 or consistency_score < 0.35:
            threat = "CRITICAL"
        elif n_contra >= 2 or consistency_score < 0.50:
            threat = "HIGH"
        elif n_contra >= 1 or consistency_score < 0.65:
            threat = "ELEVATED"
        else:
            threat = "CLEAR"

        return {
            "checks": checks,
            "contradictions": contradictions,
            "agreements": agreements,
            "contradiction_count": n_contra,
            "agreement_count": n_agree,
            "consistency_score": round(consistency_score, 4),
            "raw_confidence": round(raw_conf, 2),
            "adjusted_confidence": adjusted_confidence,
            "threat_level": threat,
        }

    # ------------------------------------------------------------------
    #  Individual check methods
    # ------------------------------------------------------------------

    @staticmethod
    def _check_neural_vs_bfm(
        n_match: bool, n_cos: float,
        b_match: Optional[bool], b_cos: Optional[float],
    ) -> Dict[str, Any]:
        name = "neural_vs_bfm_identity"
        if b_match is None:
            return {"name": name, "score": 0.5, "detail": "bfm_comparison_unavailable"}

        if n_match == b_match:
            return {
                "name": name, "score": 0.9, "agreement": True,
                "detail": f"Neural({n_match}) AGREES with BFM({b_match}) — "
                          f"cos_neural={n_cos:.3f}, cos_bfm={b_cos:.3f}",
            }
        return {
            "name": name, "score": 0.2, "contradiction": True,
            "detail": f"CONTRADICTION: Neural({n_match}) vs BFM({b_match}) — "
                      f"cos_neural={n_cos:.3f}, cos_bfm={b_cos:.3f}",
        }

    @staticmethod
    def _check_neural_vs_anthro(
        n_match: bool, anthro: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        name = "neural_vs_anthropometry"
        if not anthro:
            return {"name": name, "score": 0.5, "detail": "anthropometry_unavailable"}

        a_compat = anthro.get("compatible", None)
        a_score = anthro.get("score", 0.0)

        if a_compat is None:
            return {"name": name, "score": 0.5, "detail": "anthropometry_incomplete"}

        if n_match == a_compat:
            return {
                "name": name, "score": 0.85, "agreement": True,
                "detail": f"Neural({n_match}) AGREES with anthropometry "
                          f"(score={a_score:.3f})",
            }
        return {
            "name": name, "score": 0.25, "contradiction": True,
            "detail": f"CONTRADICTION: Neural({n_match}) vs "
                      f"anthropometry({a_compat}, score={a_score:.3f})",
        }

    @staticmethod
    def _check_forensic_signals(
        forensics: Dict[str, Any], document: Dict[str, Any], match: Dict[str, Any],
    ) -> Dict[str, Any]:
        name = "forensic_signal_agreement"

        deepfake = forensics.get("frequency", {}).get("deepfake_suspected", False)
        splice = document.get("noiseprint", {}).get("suspected_splice", False)
        verified = match.get("verified", False)

        # If verified=True but deepfake or splice detected → contradiction
        if verified and (deepfake or splice):
            detail = (
                f"CONTRADICTION: verified={verified} but "
                f"deepfake_suspected={deepfake}, splice_detected={splice}"
            )
            return {"name": name, "score": 0.15, "contradiction": True, "detail": detail}

        if not verified and (deepfake or splice):
            return {
                "name": name, "score": 0.8, "agreement": True,
                "detail": "Forensic signals consistent with non-match — "
                         f"deepfake={deepfake}, splice={splice}",
            }

        return {
            "name": name, "score": 0.7, "agreement": True,
            "detail": f"No forensic anomalies — deepfake={deepfake}, splice={splice}",
        }

    @staticmethod
    def _check_biometric_consistency(
        adv_bio: Dict[str, Any], neural_match: bool,
    ) -> Dict[str, Any]:
        name = "biometric_consistency"
        pair = adv_bio.get("pair_analysis", {})
        verdict = pair.get("final_verdict", pair.get("verdict", "UNKNOWN"))
        verdict_class = ForensicConsistencyChecker._normalize_pair_verdict(verdict)

        # Check if biometric pair verdict aligns with neural match
        bio_positive = verdict_class == "positive"
        bio_negative = verdict_class == "negative"

        if neural_match and bio_positive:
            return {
                "name": name, "score": 0.9, "agreement": True,
                "detail": f"Neural match AGREES with biometric verdict={verdict}",
            }
        if not neural_match and bio_negative:
            return {
                "name": name, "score": 0.85, "agreement": True,
                "detail": f"Neural non-match consistent with verdict={verdict}",
            }
        if neural_match and bio_negative:
            return {
                "name": name, "score": 0.2, "contradiction": True,
                "detail": f"CONTRADICTION: Neural match but biometric verdict={verdict}",
            }
        if not neural_match and bio_positive:
            return {
                "name": name, "score": 0.3, "contradiction": True,
                "detail": f"CONTRADICTION: Neural non-match but biometric verdict={verdict}",
            }

        return {"name": name, "score": 0.5, "detail": f"Biometric verdict={verdict}"}

    @staticmethod
    def _check_coefficient_auth(
        recon: Dict[str, Any], label: str,
    ) -> Dict[str, Any]:
        name = f"coefficient_authenticity_{label}"
        f3d = recon.get("forensic_3d", {})
        coeff = f3d.get("coefficient_analysis", {})

        if not coeff:
            return {"name": name, "score": 0.5, "detail": f"{label}_no_coefficient_data"}

        auth = coeff.get("authenticity_score", 1.0)
        outlier = coeff.get("is_statistical_outlier", False)

        if outlier:
            return {
                "name": name, "score": 0.2, "contradiction": True,
                "detail": f"{label} face is statistical outlier "
                          f"(auth={auth:.2f}) — possible synthetic/morphed",
            }
        return {
            "name": name, "score": 0.8, "agreement": True,
            "detail": f"{label} face statistically normal (auth={auth:.2f})",
        }

    @staticmethod
    def _cross_modal(
        neural: bool,
        bfm: Optional[bool],
        anthro: Optional[Dict[str, Any]],
        bio: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Count how many independent channels agree."""
        name = "cross_modal_agreement"
        votes: List[bool] = [neural]

        if bfm is not None:
            votes.append(bfm)
        if anthro and anthro.get("compatible") is not None:
            votes.append(anthro["compatible"])

        pair = bio.get("pair_analysis", {})
        v = pair.get("final_verdict", pair.get("verdict", ""))
        verdict_class = ForensicConsistencyChecker._normalize_pair_verdict(v)
        if verdict_class == "positive":
            votes.append(True)
        elif verdict_class == "negative":
            votes.append(False)

        if len(votes) < 2:
            return {"name": name, "score": 0.5, "detail": "insufficient_channels"}

        n_true = sum(votes)
        n_false = len(votes) - n_true
        majority = n_true > n_false
        agreement_pct = max(n_true, n_false) / len(votes)
        score = float(agreement_pct)

        unanimous = (n_true == len(votes)) or (n_false == len(votes))
        if unanimous:
            return {
                "name": name, "score": 0.95, "agreement": True,
                "detail": f"UNANIMOUS: {len(votes)}/{len(votes)} channels agree "
                          f"({'MATCH' if majority else 'MISMATCH'})",
            }
        if agreement_pct >= 0.75:
            return {
                "name": name, "score": 0.7, "agreement": True,
                "detail": f"Strong majority: {max(n_true,n_false)}/{len(votes)} "
                          f"channels agree",
            }
        return {
            "name": name, "score": 0.3, "contradiction": True,
            "detail": f"Split vote: {n_true} match vs {n_false} mismatch "
                      f"across {len(votes)} channels",
        }

    @staticmethod
    def _normalize_pair_verdict(verdict: str) -> str:
        verdict = str(verdict or "").strip().upper()
        if verdict in {
            "VERIFIED",
            "LIKELY_MATCH",
            "LIKELY_MATCH_WITH_ALTERATION_REVIEW",
            "VERIFIED_WITH_ALTERATION_REVIEW",
            "SAME_PERSON_WITH_ALTERATION_REVIEW",
            "CONFIRMED_SAME_PERSON",
            "SAME_PERSON",
            "LIKELY_SAME",
        }:
            return "positive"
        if verdict in {
            "REJECT",
            "CONFIRMED_DIFFERENT",
            "LIKELY_DIFFERENT",
            "LIKELY_DOPPELGANGER",
        }:
            return "negative"
        return "neutral"
