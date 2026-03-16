"""
BFM COEFFICIENT SPACE FORENSICS — CA_MONK v6.1
====================================================
Statistical analysis of Basel Face Model reconstruction
coefficients for synthetic face detection and cross-image
identity verification.

The 257 BFM coefficients encode:
    Identity   (80) : face shape eigenspace     — WHO the person is
    Expression (64) : expression PCA basis       — WHAT emotion is shown
    Texture    (80) : skin albedo eigenspace     — skin color / pattern
    Rotation    (3) : Euler angles (rad)         — head pose
    Illumination(27): Spherical Harmonics (3×9)  — lighting environment
    Translation (3) : tx, ty, tz                 — face position

Real human faces follow known statistical distributions in
this coefficient space.  Synthetic, morphed, or manipulated
faces produce anomalous coefficient patterns detectable via:

    1. Magnitude analysis  (L2 norm per subspace)
    2. Per-coefficient Z-scores (eigenvalue decay assumption)
    3. Expression feasibility  (energy concentration check)
    4. Illumination physical plausibility  (SH band analysis)
    5. Identity coefficient comparison (POSE-INVARIANT matching,
       completely independent of neural embeddings)

Author: CA_MONK Forensic Intelligence Unit
Version: 6.1.0
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import numpy as np

logger = logging.getLogger("ca_monk.coeff_forensics")

# ============================================================================
#  Statistical priors  (approx BFM eigenvalue spectrum)
# ============================================================================
_ID_STD = 1.0
_EXP_STD = 0.5
_TEX_STD = 1.0
_Z_THRESH = 3.0
_IDENTITY_MATCH_THRESH = 0.65


class CoefficientForensics:
    """
    Forensic analysis of BFM reconstruction coefficients.

    Capabilities:
        1. Statistical anomaly detection → synthetic / morphed face flags
        2. Pose-invariant identity comparison via identity coefficients
        3. Expression feasibility scoring
        4. Illumination plausibility audit
        5. Per-face authenticity score (0–1)
    """

    # ------------------------------------------------------------------
    #  Single-face analysis
    # ------------------------------------------------------------------
    def analyze(
        self,
        coefficients: np.ndarray,
        coeff_dict: Optional[Dict[str, np.ndarray]] = None,
    ) -> Dict[str, Any]:
        """
        Analyze one face's 257 BFM coefficients for forensic anomalies.
        """
        coefficients = np.asarray(coefficients).flatten()
        if coefficients.shape[0] != 257:
            return {"error": f"expected 257 coefficients, got {coefficients.shape[0]}"}

        if coeff_dict is None:
            coeff_dict = self._split(coefficients)
        else:
            coeff_dict = {k: np.asarray(v).flatten() for k, v in coeff_dict.items()}

        anomalies: list[str] = []
        results: Dict[str, Any] = {}

        # ---- 1. Magnitude analysis ----
        id_c, exp_c, tex_c = coeff_dict["id"], coeff_dict["exp"], coeff_dict["tex"]
        id_mag = float(np.linalg.norm(id_c))
        exp_mag = float(np.linalg.norm(exp_c))
        tex_mag = float(np.linalg.norm(tex_c))

        id_exp_mag = np.sqrt(80) * _ID_STD
        exp_exp_mag = np.sqrt(64) * _EXP_STD
        tex_exp_mag = np.sqrt(80) * _TEX_STD

        id_z = (id_mag - id_exp_mag) / (id_exp_mag * 0.3 + 1e-8)
        exp_z = (exp_mag - exp_exp_mag) / (exp_exp_mag * 0.3 + 1e-8)
        tex_z = (tex_mag - tex_exp_mag) / (tex_exp_mag * 0.3 + 1e-8)

        results["magnitudes"] = {
            "identity": round(id_mag, 4),
            "expression": round(exp_mag, 4),
            "texture": round(tex_mag, 4),
            "identity_zscore": round(id_z, 2),
            "expression_zscore": round(exp_z, 2),
            "texture_zscore": round(tex_z, 2),
        }
        for tag, zv in [("identity", id_z), ("expression", exp_z), ("texture", tex_z)]:
            if abs(zv) > _Z_THRESH:
                anomalies.append(f"{tag}_magnitude_outlier: z={zv:.2f}")

        # ---- 2. Per-coefficient Z-scores (eigenvalue decay) ----
        id_stds = np.array([_ID_STD * np.exp(-0.02 * i) for i in range(80)])
        exp_stds = np.array([_EXP_STD * np.exp(-0.03 * i) for i in range(64)])
        tex_stds = np.array([_TEX_STD * np.exp(-0.02 * i) for i in range(80)])

        id_out = int(np.sum(np.abs(id_c) / (id_stds + 1e-8) > _Z_THRESH))
        exp_out = int(np.sum(np.abs(exp_c) / (exp_stds + 1e-8) > _Z_THRESH))
        tex_out = int(np.sum(np.abs(tex_c) / (tex_stds + 1e-8) > _Z_THRESH))
        total_out = id_out + exp_out + tex_out

        results["outlier_coefficients"] = {
            "identity": id_out,
            "expression": exp_out,
            "texture": tex_out,
            "total": total_out,
        }
        if id_out > 10:
            anomalies.append(f"excessive_identity_outliers: {id_out}/80")
        if tex_out > 10:
            anomalies.append(f"excessive_texture_outliers: {tex_out}/80")

        # ---- 3. Expression feasibility ----
        results["expression_analysis"] = self._analyze_expression(exp_c)
        if not results["expression_analysis"].get("feasible", True):
            anomalies.append("infeasible_expression_pattern")

        # ---- 4. Illumination plausibility ----
        gamma = coeff_dict.get("gamma", np.zeros(27))
        results["illumination"] = self._analyze_illumination(gamma)
        if not results["illumination"].get("plausible", True):
            anomalies.append("implausible_illumination")

        # ---- 5. Pose ----
        ang = coeff_dict.get("angle", np.zeros(3))
        results["pose"] = {
            "yaw_deg": round(float(np.degrees(ang[1])) if len(ang) > 1 else 0.0, 2),
            "pitch_deg": round(float(np.degrees(ang[0])) if len(ang) > 0 else 0.0, 2),
            "roll_deg": round(float(np.degrees(ang[2])) if len(ang) > 2 else 0.0, 2),
            "is_frontal": bool(all(abs(np.degrees(a)) < 30 for a in ang[:3])),
        }

        # ---- 6. Authenticity score ----
        outlier_ratio = total_out / 224.0
        mag_penalty = max(0.0, (abs(id_z) + abs(tex_z)) / 2.0 - 1.0) * 0.2
        auth = float(np.clip(1.0 - outlier_ratio * 2.0 - mag_penalty, 0, 1))
        results["authenticity_score"] = round(auth, 4)
        results["anomalies"] = anomalies
        results["is_statistical_outlier"] = len(anomalies) > 2 or auth < 0.5

        # ---- 7. Identity signature (for cross-image comparison) ----
        id_norm = id_c / (np.linalg.norm(id_c) + 1e-8)
        results["identity_signature"] = id_norm.tolist()

        return results

    # ------------------------------------------------------------------
    #  Expression feasibility
    # ------------------------------------------------------------------
    @staticmethod
    def _analyze_expression(exp: np.ndarray) -> Dict[str, Any]:
        mag = float(np.linalg.norm(exp))
        mx = float(np.max(np.abs(exp)))
        sorted_abs = np.sort(np.abs(exp))[::-1]
        top5_e = float(np.sum(sorted_abs[:5] ** 2))
        total_e = float(np.sum(sorted_abs ** 2) + 1e-8)
        conc = top5_e / total_e

        feasible = mag < 20.0 and mx < 8.0
        if mag < 2.0:
            intensity = "neutral"
        elif mag < 5.0:
            intensity = "mild"
        elif mag < 10.0:
            intensity = "moderate"
        else:
            intensity = "extreme"

        return {
            "magnitude": round(mag, 4),
            "max_coefficient": round(mx, 4),
            "energy_concentration_top5": round(conc, 4),
            "feasible": feasible,
            "intensity": intensity,
        }

    # ------------------------------------------------------------------
    #  Illumination plausibility (SH band analysis)
    # ------------------------------------------------------------------
    @staticmethod
    def _analyze_illumination(gamma: np.ndarray) -> Dict[str, Any]:
        gamma = np.asarray(gamma).flatten()
        if gamma.size < 27:
            return {"plausible": True, "note": "insufficient_sh_coefficients"}

        g3x9 = gamma[:27].reshape(3, 9)
        dc_energy = float(np.mean(np.abs(g3x9[:, 0])))
        band_energies = [float(np.mean(np.abs(g3x9[:, i]))) for i in range(9)]

        # RGB channel correlation (real light is correlated across R/G/B)
        corrs = []
        for a, b in [(0, 1), (1, 2), (0, 2)]:
            c = np.corrcoef(g3x9[a], g3x9[b])[0, 1]
            if not np.isnan(c):
                corrs.append(c)
        rgb_corr = float(np.mean(corrs)) if corrs else 0.0

        plausible = dc_energy > 0.1 and rgb_corr > -0.5
        direction = "directional" if band_energies[1] > dc_energy * 0.5 else "ambient"

        return {
            "dc_energy": round(dc_energy, 4),
            "band_energies": [round(e, 4) for e in band_energies],
            "rgb_sh_correlation": round(rgb_corr, 4),
            "dominant_direction": direction,
            "plausible": plausible,
        }

    # ------------------------------------------------------------------
    #  Cross-image identity comparison (pose-invariant, NO neural net)
    # ------------------------------------------------------------------
    def compare_identity(
        self, coeffs_a: np.ndarray, coeffs_b: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Compare two faces using ONLY their 80-dim BFM identity
        coefficients.  This is completely independent of neural
        face embeddings and provides a physics-based verification
        channel.
        """
        coeffs_a = np.asarray(coeffs_a).flatten()
        coeffs_b = np.asarray(coeffs_b).flatten()
        id_a = coeffs_a[:80]
        id_b = coeffs_b[:80]

        na, nb = np.linalg.norm(id_a), np.linalg.norm(id_b)
        if na < 1e-8 or nb < 1e-8:
            return {"match": False, "score": 0.0, "reason": "degenerate"}

        cos = float(np.dot(id_a, id_b) / (na * nb))
        l2 = float(np.linalg.norm(id_a / na - id_b / nb))

        # Top differing shape components
        diffs = np.abs(id_a - id_b)
        top5 = np.argsort(-diffs)[:5].tolist()

        match = cos > _IDENTITY_MATCH_THRESH
        if cos > 0.85:
            conf = "HIGH"
        elif cos > 0.65:
            conf = "MODERATE"
        else:
            conf = "LOW"

        return {
            "match": bool(match),
            "cosine_similarity": round(cos, 4),
            "l2_distance": round(l2, 4),
            "confidence": conf,
            "top_differing_components": top5,
            "verdict": "BFM_IDENTITY_MATCH" if match else "BFM_IDENTITY_MISMATCH",
        }

    # ------------------------------------------------------------------
    #  Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _split(c: np.ndarray) -> Dict[str, np.ndarray]:
        return {
            "id": c[:80],
            "exp": c[80:144],
            "tex": c[144:224],
            "angle": c[224:227],
            "gamma": c[227:254],
            "trans": c[254:],
        }
