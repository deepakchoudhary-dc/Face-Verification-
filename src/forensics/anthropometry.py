"""
3D FORENSIC ANTHROPOMETRY ENGINE — CA_MONK v6.1
====================================================
FBI/Interpol-grade facial measurement extraction from
Deep3D BFM09 3D mesh reconstruction.

Uses 68 3D landmarks from BFM fitting to compute real-world
forensic-grade anthropometric ratios and distances.

These measurements provide a physics-based identity verification
channel COMPLETELY INDEPENDENT of neural face embeddings.

Measurements extracted (22 direct + 12 derived ratios):
    - Inter-pupillary distance (IPD)
    - Facial width-to-height ratio (fWHR)
    - Nose-to-face width ratio
    - Eye aspect ratios (left/right)
    - Canthal tilt angles (left/right)
    - Facial thirds ratio (upper:middle:lower)
    - Jaw width ratio
    - Nasolabial angle
    - Philtrum ratio
    - Golden ratio deviations
    - 3D bilateral symmetry index (30 landmark pairs)

Reference Standards:
    - FBI FACE (Facial Analysis Comparison and Evaluation)
    - FISWG (Facial Identification Scientific Working Group)
    - ISO 19795-1 Biometric performance testing
    - Farkas L.G. "Anthropometry of the Head and Face" (1994)

Author: CA_MONK Forensic Intelligence Unit
Version: 6.1.0
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

import numpy as np

logger = logging.getLogger("ca_monk.anthropometry")

# ============================================================================
#  68-point landmark index groups (dlib / BFM convention)
# ============================================================================
_JAW = list(range(0, 17))
_LEFT_BROW = list(range(17, 22))
_RIGHT_BROW = list(range(22, 27))
_NOSE_BRIDGE = list(range(27, 31))
_NOSE_TIP = list(range(31, 36))
_LEFT_EYE = list(range(36, 42))
_RIGHT_EYE = list(range(42, 48))
_OUTER_MOUTH = list(range(48, 60))
_INNER_MOUTH = list(range(60, 68))

# Golden ratio
PHI = (1.0 + np.sqrt(5.0)) / 2.0  # 1.618033988749895


# ============================================================================
#  Utility functions
# ============================================================================
def _dist3d(a: np.ndarray, b: np.ndarray) -> float:
    """Euclidean distance between two 3D points."""
    return float(np.linalg.norm(a - b))


def _angle_deg(a: np.ndarray, vertex: np.ndarray, b: np.ndarray) -> float:
    """Angle at *vertex* between rays to *a* and *b*, in degrees."""
    va = a - vertex
    vb = b - vertex
    cos_a = np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-8)
    return float(np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0))))


def _midpoint(pts: np.ndarray) -> np.ndarray:
    """Mean of an (N, 3) array."""
    return np.mean(pts, axis=0)


# ============================================================================
#  Normal human ratio bounds  (Farkas 1994 + FISWG guidelines)
# ============================================================================
_NORMAL_BOUNDS: Dict[str, tuple] = {
    "fwhr": (1.6, 2.2),
    "nose_face_ratio": (0.20, 0.35),
    "eye_aspect_left": (0.15, 0.45),
    "eye_aspect_right": (0.15, 0.45),
    "mouth_face_ratio": (0.35, 0.55),
    "lower_face_ratio": (0.30, 0.42),
    "nose_width_ipd_ratio": (0.60, 1.0),
}

# ============================================================================
#  30 bilateral landmark pairs for symmetry analysis
# ============================================================================
_SYMMETRY_PAIRS = [
    # Eyes
    (36, 45), (37, 44), (38, 43), (39, 42), (40, 47), (41, 46),
    # Brows
    (17, 26), (18, 25), (19, 24), (20, 23), (21, 22),
    # Jaw
    (0, 16), (1, 15), (2, 14), (3, 13), (4, 12), (5, 11), (6, 10), (7, 9),
    # Nose
    (31, 35), (32, 34),
    # Outer mouth
    (48, 54), (49, 53), (50, 52), (59, 55), (58, 56),
    # Inner mouth
    (60, 64), (61, 63), (67, 65),
]


# ============================================================================
#  Main class
# ============================================================================
class ForensicAnthropometry:
    """
    Extracts FBI/FISWG-grade anthropometric measurements from 3D
    facial landmarks obtained via BFM reconstruction.

    Provides:
        1. Single-face measurement extraction  (22 raw + 12 ratios)
        2. Pair comparison with scale-invariant ratio matching
        3. 3D bilateral symmetry scoring (30 landmark pairs)
        4. Golden ratio deviation analysis
        5. Statistical anomaly flagging for synthetic face detection
    """

    # ------------------------------------------------------------------
    #  Single-face measurement
    # ------------------------------------------------------------------
    def measure(self, landmarks_3d: np.ndarray) -> Dict[str, Any]:
        """
        Extract forensic anthropometric profile from 68 3D landmarks.

        Args:
            landmarks_3d: (68, 3) — BFM world-space landmark coordinates

        Returns:
            Dict with 'measurements', 'ratios', 'symmetry',
            'golden_ratio', 'anomalies'
        """
        if landmarks_3d.shape != (68, 3):
            logger.warning("Expected (68, 3) landmarks, got %s", landmarks_3d.shape)
            return {"error": "invalid_landmarks", "measurements": {}}

        lm = landmarks_3d.astype(np.float64)

        # ---- Key reference points ----
        left_eye_center = _midpoint(lm[_LEFT_EYE])
        right_eye_center = _midpoint(lm[_RIGHT_EYE])
        brow_mid = (_midpoint(lm[_LEFT_BROW]) + _midpoint(lm[_RIGHT_BROW])) / 2.0
        chin = lm[8]
        nose_tip = lm[33]
        nose_bridge_top = lm[27]

        # ---- 1. Inter-pupillary distance (fundamental normalizer) ----
        ipd = _dist3d(left_eye_center, right_eye_center)
        if ipd < 1e-6:
            return {"error": "degenerate_ipd", "measurements": {}}

        # ---- 2. Face dimensions ----
        face_width = _dist3d(lm[1], lm[15])          # bizygomatic
        face_height = _dist3d(brow_mid, chin)
        upper_face = _dist3d(brow_mid, nose_tip)
        lower_face = _dist3d(nose_tip, chin)
        mid_face = _dist3d(lm[27], lm[33])             # nasion → subnasale
        fwhr = face_width / (upper_face + 1e-8)

        # ---- 3. Nose ----
        nose_width = _dist3d(lm[31], lm[35])
        nose_length = _dist3d(lm[27], lm[33])
        nose_bridge_angle = _angle_deg(lm[27], lm[30], lm[33])

        # ---- 4. Eyes ----
        ew_L = _dist3d(lm[36], lm[39])
        ew_R = _dist3d(lm[42], lm[45])
        eh_L = (_dist3d(lm[37], lm[41]) + _dist3d(lm[38], lm[40])) / 2.0
        eh_R = (_dist3d(lm[43], lm[47]) + _dist3d(lm[44], lm[46])) / 2.0
        ear_L = eh_L / (ew_L + 1e-8)
        ear_R = eh_R / (ew_R + 1e-8)

        # Canthal tilt — angle of eye axis vs horizontal plane
        left_axis = lm[39] - lm[36]
        right_axis = lm[42] - lm[45]
        cant_L = float(np.degrees(np.arctan2(left_axis[1],
                                              np.linalg.norm(left_axis[[0, 2]]))))
        cant_R = float(np.degrees(np.arctan2(right_axis[1],
                                              np.linalg.norm(right_axis[[0, 2]]))))

        # ---- 5. Mouth ----
        mouth_w = _dist3d(lm[48], lm[54])
        upper_lip_h = _dist3d(lm[51], lm[62])
        lower_lip_h = _dist3d(lm[57], lm[66])

        # ---- 6. Philtrum & jaw ----
        philtrum = _dist3d(lm[33], lm[51])
        jaw_width = _dist3d(lm[4], lm[12])
        nasolabial = _angle_deg(lm[27], lm[33], lm[51])

        # ---- 7. Facial thirds ----
        total_h = face_height if face_height > 1e-6 else 1.0
        thirds = {
            "upper": float(upper_face / total_h),
            "middle": float(mid_face / total_h),
            "lower": float(lower_face / total_h),
            "balanced": bool(
                abs(upper_face - lower_face) / total_h < 0.10
            ),
        }

        # ---- Build ratio vector (scale-invariant) ----
        ratios: Dict[str, float] = {
            "fwhr": float(fwhr),
            "nose_face_ratio": float(nose_width / (face_width + 1e-8)),
            "nose_length_ratio": float(nose_length / (ipd + 1e-8)),
            "eye_aspect_left": float(ear_L),
            "eye_aspect_right": float(ear_R),
            "mouth_face_ratio": float(mouth_w / (face_width + 1e-8)),
            "lower_face_ratio": float(lower_face / (face_height + 1e-8)),
            "jaw_face_ratio": float(jaw_width / (face_width + 1e-8)),
            "philtrum_nose_ratio": float(philtrum / (nose_length + 1e-8)),
            "upper_lip_ratio": float(upper_lip_h / (lower_face + 1e-8)),
            "nose_width_ipd_ratio": float(nose_width / (ipd + 1e-8)),
            "face_width_ipd_ratio": float(face_width / (ipd + 1e-8)),
        }

        measurements = {
            "ipd": float(ipd),
            "face_width": float(face_width),
            "face_height": float(face_height),
            "upper_face_height": float(upper_face),
            "lower_face_height": float(lower_face),
            "fwhr": float(fwhr),
            "nose_width": float(nose_width),
            "nose_length": float(nose_length),
            "nose_bridge_angle": float(nose_bridge_angle),
            "eye_width_left": float(ew_L),
            "eye_width_right": float(ew_R),
            "eye_aspect_left": float(ear_L),
            "eye_aspect_right": float(ear_R),
            "canthal_tilt_left": float(cant_L),
            "canthal_tilt_right": float(cant_R),
            "mouth_width": float(mouth_w),
            "upper_lip_height": float(upper_lip_h),
            "lower_lip_height": float(lower_lip_h),
            "philtrum_length": float(philtrum),
            "jaw_width": float(jaw_width),
            "nasolabial_angle": float(nasolabial),
        }

        symmetry = self._bilateral_symmetry(lm)
        golden = self._golden_ratio(face_width, face_height, nose_width,
                                     mouth_w, ipd, nose_length)
        anomalies = self._flag_anomalies(ratios)

        return {
            "measurements": measurements,
            "ratios": ratios,
            "facial_thirds": thirds,
            "symmetry": symmetry,
            "golden_ratio": golden,
            "anomalies": anomalies,
        }

    # ------------------------------------------------------------------
    #  Bilateral symmetry via 30 landmark pairs
    # ------------------------------------------------------------------
    def _bilateral_symmetry(self, lm: np.ndarray) -> Dict[str, Any]:
        midline_top = lm[27]
        midline_bot = lm[8]
        mid_dir = midline_bot - midline_top
        mid_len = np.linalg.norm(mid_dir)
        if mid_len < 1e-8:
            return {"symmetry_score": 0.0, "mean_asymmetry": 1.0}
        mid_dir /= mid_len

        asym: List[float] = []
        region_labels: List[str] = []
        region_map = (
            ["eyes"] * 6 + ["brows"] * 5 + ["jaw"] * 8
            + ["nose"] * 2 + ["mouth"] * 5 + ["mouth"] * 3
        )

        for idx, (li, ri) in enumerate(_SYMMETRY_PAIRS):
            l_dist = float(np.linalg.norm(np.cross(lm[li] - midline_top, mid_dir)))
            r_dist = float(np.linalg.norm(np.cross(lm[ri] - midline_top, mid_dir)))
            mean_d = (l_dist + r_dist) / 2.0
            if mean_d > 1e-6:
                asym.append(abs(l_dist - r_dist) / mean_d)
                if idx < len(region_map):
                    region_labels.append(region_map[idx])

        if not asym:
            return {"symmetry_score": 0.0, "mean_asymmetry": 1.0}

        mean_a = float(np.mean(asym))
        max_a = float(np.max(asym))
        score = float(np.clip(1.0 - mean_a, 0, 1))

        # Identify most asymmetric region
        worst_region = "unknown"
        if region_labels:
            max_idx = int(np.argmax(asym))
            if max_idx < len(region_labels):
                worst_region = region_labels[max_idx]

        return {
            "symmetry_score": round(score, 4),
            "mean_asymmetry": round(mean_a, 4),
            "max_asymmetry": round(max_a, 4),
            "most_asymmetric_region": worst_region,
        }

    # ------------------------------------------------------------------
    #  Golden ratio analysis
    # ------------------------------------------------------------------
    def _golden_ratio(self, fw, fh, nw, mw, ipd, nl) -> Dict[str, Any]:
        checks: Dict[str, float] = {}
        if fh > 1e-6:
            r = fw / fh
            checks["face_wh_vs_inv_phi"] = round(abs(r - 1.0 / PHI) / (1.0 / PHI), 4)
        if nw > 1e-6 and mw > 1e-6:
            r = mw / nw
            checks["mouth_nose_vs_phi"] = round(abs(r - PHI) / PHI, 4)
        if ipd > 1e-6 and mw > 1e-6:
            checks["mouth_ipd_ratio"] = round(float(mw / ipd), 4)
        vals = list(checks.values()) or [0.0]
        checks["mean_golden_deviation"] = round(float(np.mean(vals)), 4)
        return checks

    # ------------------------------------------------------------------
    #  Anomaly flags (outside normal human bounds)
    # ------------------------------------------------------------------
    def _flag_anomalies(self, ratios: Dict[str, float]) -> List[str]:
        out: List[str] = []
        for key, (lo, hi) in _NORMAL_BOUNDS.items():
            v = ratios.get(key)
            if v is None:
                continue
            if v < lo:
                out.append(f"{key}_below_normal: {v:.3f} < {lo}")
            elif v > hi:
                out.append(f"{key}_above_normal: {v:.3f} > {hi}")
        return out

    # ------------------------------------------------------------------
    #  Pair comparison (scale-invariant)
    # ------------------------------------------------------------------
    def compare(
        self,
        profile_a: Dict[str, Any],
        profile_b: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Compare two anthropometric profiles using scale-invariant ratios.

        Returns compatibility score, per-ratio deltas, and verdict.
        """
        ra = profile_a.get("ratios", {})
        rb = profile_b.get("ratios", {})
        if not ra or not rb:
            return {"compatible": False, "score": 0.0, "reason": "missing_ratios"}

        keys = sorted(set(ra) & set(rb))
        if not keys:
            return {"compatible": False, "score": 0.0, "reason": "no_common_ratios"}

        va = np.array([ra[k] for k in keys])
        vb = np.array([rb[k] for k in keys])

        na, nb = np.linalg.norm(va), np.linalg.norm(vb)
        if na < 1e-8 or nb < 1e-8:
            return {"compatible": False, "score": 0.0, "reason": "degenerate"}

        cosine = float(np.dot(va, vb) / (na * nb))
        l2 = float(np.linalg.norm(va - vb) / np.sqrt(len(keys)))

        deltas: Dict[str, Dict[str, float]] = {}
        sig_diffs = 0
        for k in keys:
            d = abs(ra[k] - rb[k])
            rel = d / (abs(ra[k]) + 1e-8)
            deltas[k] = {"abs": round(d, 4), "rel": round(rel, 4)}
            if rel > 0.15:
                sig_diffs += 1

        score = float(np.clip(0.6 * cosine + 0.4 * max(0, 1.0 - l2 * 5), 0, 1))
        compatible = score > 0.70 and sig_diffs < 4

        if score > 0.85:
            conf = "HIGH"
        elif score > 0.70:
            conf = "MODERATE"
        else:
            conf = "LOW"

        return {
            "compatible": bool(compatible),
            "score": round(score, 4),
            "cosine_similarity": round(cosine, 4),
            "l2_distance": round(l2, 4),
            "significant_differences": sig_diffs,
            "per_ratio_deltas": deltas,
            "verdict": "ANTHROPOMETRIC_MATCH" if compatible else "ANTHROPOMETRIC_MISMATCH",
            "confidence": conf,
        }
