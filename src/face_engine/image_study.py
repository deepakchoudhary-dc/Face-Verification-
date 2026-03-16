"""
PRIMARY IMAGE DEEP STUDY MODULE
================================
Performs a detailed forensic analysis of the primary face image to detect:
- Injury marks, scars, bruises
- Skin conditions (acne, discoloration, burns)
- Occlusions (glasses, masks, bandages, hair)
- Image quality issues (blur, noise, exposure)
- Facial asymmetry or deformations
- Lighting and shadow analysis
- Age-related features (wrinkles, spots)

The study output is used to:
1. Inform the Chain-of-Thought analysis in FINAL_REPORT.md
2. Guide CodeFormer ONNX + SD 1.5 reconstruction with specific instructions
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger("ca_monk.image_study")


class PrimaryImageStudy:
    """
    Deep forensic study of a primary face image.
    Produces a structured report of facial features, anomalies,
    and reconstruction guidance for downstream modules.
    """

    def __init__(self) -> None:
        self._face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self._eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_eye.xml"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def analyze(self, image_path: str, face_box: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Run full deep study on the primary image.
        Returns structured findings + reconstruction_guidance string.
        """
        image = cv2.imread(image_path)
        if image is None:
            return self._empty_study("image_not_readable")

        # Extract face region
        face_crop, face_region = self._extract_face(image, face_box)
        if face_crop is None or face_crop.size == 0:
            return self._empty_study("no_face_detected")

        # Run all analysis modules
        quality = self._analyze_quality(face_crop)
        skin = self._analyze_skin(face_crop)
        marks = self._detect_marks_and_injuries(face_crop)
        occlusion = self._detect_occlusions(face_crop)
        symmetry = self._analyze_symmetry(face_crop)
        lighting = self._analyze_lighting(face_crop)
        aging = self._analyze_aging_features(face_crop)
        color_analysis = self._analyze_color_distribution(face_crop)

        # Build findings list
        findings: List[str] = []
        issues_detected: List[str] = []

        # Quality findings
        findings.append(
            f"Image Quality: blur_score={quality['blur_score']:.1f}, "
            f"noise_level={quality['noise_level']:.1f}, "
            f"exposure={'overexposed' if quality['overexposed'] else 'underexposed' if quality['underexposed'] else 'normal'}"
        )
        if quality["blur_score"] < 60.0:
            issues_detected.append("blurry_image")
        if quality["noise_level"] > 30.0:
            issues_detected.append("noisy_image")
        if quality["overexposed"]:
            issues_detected.append("overexposed")
        if quality["underexposed"]:
            issues_detected.append("underexposed")

        # Skin findings
        findings.append(
            f"Skin Analysis: uniformity={skin['uniformity']:.2f}, "
            f"redness_ratio={skin['redness_ratio']:.3f}, "
            f"texture_roughness={skin['texture_roughness']:.1f}"
        )
        if skin["redness_ratio"] > 0.15:
            issues_detected.append("high_skin_redness")
        if skin["texture_roughness"] > 40.0:
            issues_detected.append("rough_skin_texture")

        # Marks and injuries
        if marks["total_marks"] > 0:
            findings.append(
                f"Marks/Injuries Detected: {marks['total_marks']} regions "
                f"(dark_spots={marks['dark_spots']}, red_spots={marks['red_spots']}, "
                f"scar_like={marks['scar_like_regions']})"
            )
            if marks["dark_spots"] > 0:
                issues_detected.append(f"dark_spots_{marks['dark_spots']}")
            if marks["red_spots"] > 0:
                issues_detected.append(f"red_spots_{marks['red_spots']}")
            if marks["scar_like_regions"] > 0:
                issues_detected.append(f"scar_like_regions_{marks['scar_like_regions']}")
        else:
            findings.append("Marks/Injuries: None detected — clean facial surface")

        # Occlusion findings
        if occlusion["any_occlusion"]:
            occl_parts = []
            if occlusion["glasses_detected"]:
                occl_parts.append("glasses")
            if occlusion["hair_occlusion"]:
                occl_parts.append("hair_over_face")
            if occlusion["partial_face"]:
                occl_parts.append("partial_face_visible")
            findings.append(f"Occlusions Detected: {', '.join(occl_parts)}")
            issues_detected.extend(occl_parts)
        else:
            findings.append("Occlusions: None — full face visible")

        # Symmetry
        findings.append(
            f"Facial Symmetry: score={symmetry['symmetry_score']:.2f} "
            f"({'symmetric' if symmetry['symmetry_score'] > 0.85 else 'asymmetric'})"
        )
        if symmetry["symmetry_score"] < 0.75:
            issues_detected.append("facial_asymmetry")

        # Lighting
        findings.append(
            f"Lighting: mean_brightness={lighting['mean_brightness']:.0f}, "
            f"shadow_ratio={lighting['shadow_ratio']:.2f}, "
            f"direction={'{'}{lighting['light_direction']}{'}'}"
        )
        if lighting["shadow_ratio"] > 0.4:
            issues_detected.append("heavy_shadows")
        if lighting["uneven"]:
            issues_detected.append("uneven_lighting")

        # Aging
        findings.append(
            f"Aging Features: wrinkle_density={aging['wrinkle_density']:.2f}, "
            f"estimated_age_range={aging['estimated_age_range']}"
        )

        # Color
        findings.append(
            f"Skin Color Profile: dominant_tone={color_analysis['dominant_tone']}, "
            f"variance={color_analysis['color_variance']:.1f}"
        )

        # Build reconstruction guidance
        guidance = self._build_reconstruction_guidance(
            quality, skin, marks, occlusion, symmetry, lighting, aging, issues_detected
        )

        study = {
            "image_path": image_path,
            "face_detected": True,
            "quality": quality,
            "skin_analysis": skin,
            "marks_and_injuries": marks,
            "occlusions": occlusion,
            "symmetry": symmetry,
            "lighting": lighting,
            "aging_features": aging,
            "color_analysis": color_analysis,
            "findings": findings,
            "issues_detected": issues_detected,
            "reconstruction_guidance": guidance,
        }

        logger.info(
            "Primary image study complete: %d findings, %d issues, guidance_length=%d",
            len(findings), len(issues_detected), len(guidance),
        )
        return study

    def estimate_age_evidence(self, image_path: str, face_box: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Lightweight age-oriented evidence extraction used to decide whether
        age progression/regression should run at all.
        """
        image = cv2.imread(image_path)
        if image is None:
            return {"image_path": image_path, "face_detected": False, "aging_features": {}}

        face_crop, _ = self._extract_face(image, face_box)
        if face_crop is None or face_crop.size == 0:
            return {"image_path": image_path, "face_detected": False, "aging_features": {}}

        quality = self._analyze_quality(face_crop)
        skin = self._analyze_skin(face_crop)
        marks = self._detect_marks_and_injuries(face_crop)
        lighting = self._analyze_lighting(face_crop)
        aging = self._analyze_aging_features(
            face_crop,
            skin=skin,
            marks=marks,
            quality=quality,
            lighting=lighting,
        )
        return {
            "image_path": image_path,
            "face_detected": True,
            "quality": quality,
            "skin_analysis": skin,
            "marks_and_injuries": marks,
            "lighting": lighting,
            "aging_features": aging,
        }

    # ------------------------------------------------------------------
    # Face extraction
    # ------------------------------------------------------------------
    def _extract_face(
        self, image: np.ndarray, face_box: Optional[Dict]
    ) -> Tuple[Optional[np.ndarray], Optional[Tuple[int, int, int, int]]]:
        h, w = image.shape[:2]
        if face_box:
            x = max(0, int(face_box.get("x", 0)))
            y = max(0, int(face_box.get("y", 0)))
            fw = min(int(face_box.get("w", w)), w - x)
            fh = min(int(face_box.get("h", h)), h - y)
            if fw > 10 and fh > 10:
                return image[y : y + fh, x : x + fw], (x, y, fw, fh)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self._face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
        if len(faces) > 0:
            x, y, fw, fh = max(faces, key=lambda f: f[2] * f[3])
            pad = int(max(fw, fh) * 0.15)
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(w, x + fw + pad)
            y2 = min(h, y + fh + pad)
            return image[y1:y2, x1:x2], (x1, y1, x2 - x1, y2 - y1)

        # Fallback: use center 70% of image
        cx, cy = w // 2, h // 2
        rw, rh = int(w * 0.35), int(h * 0.35)
        return image[cy - rh : cy + rh, cx - rw : cx + rw], (cx - rw, cy - rh, rw * 2, rh * 2)

    # ------------------------------------------------------------------
    # Quality analysis
    # ------------------------------------------------------------------
    def _analyze_quality(self, face: np.ndarray) -> Dict[str, Any]:
        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        noise_level = float(np.std(cv2.GaussianBlur(gray, (3, 3), 0).astype(float) - gray.astype(float)))
        mean_val = float(np.mean(gray))
        overexposed = float(np.mean(gray > 240)) > 0.15
        underexposed = float(np.mean(gray < 30)) > 0.20
        return {
            "blur_score": blur_score,
            "noise_level": noise_level,
            "mean_brightness": mean_val,
            "overexposed": overexposed,
            "underexposed": underexposed,
            "resolution": f"{face.shape[1]}x{face.shape[0]}",
        }

    # ------------------------------------------------------------------
    # Skin analysis
    # ------------------------------------------------------------------
    def _analyze_skin(self, face: np.ndarray) -> Dict[str, Any]:
        hsv = cv2.cvtColor(face, cv2.COLOR_BGR2HSV)
        # Skin detection in HSV space
        lower = np.array([0, 20, 70], dtype=np.uint8)
        upper = np.array([20, 255, 255], dtype=np.uint8)
        skin_mask = cv2.inRange(hsv, lower, upper)

        skin_pixels = face[skin_mask > 0]
        if len(skin_pixels) == 0:
            return {"uniformity": 0.0, "redness_ratio": 0.0, "texture_roughness": 0.0}

        # Color uniformity
        std_per_channel = np.std(skin_pixels.astype(float), axis=0)
        uniformity = 1.0 - min(float(np.mean(std_per_channel)) / 60.0, 1.0)

        # Redness (high red channel relative to others)
        b, g, r = np.mean(skin_pixels, axis=0)
        redness_ratio = float(r / (b + g + r + 1e-6))

        # Texture roughness via Laplacian on skin region
        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        skin_gray = cv2.bitwise_and(gray, gray, mask=skin_mask)
        texture_roughness = float(cv2.Laplacian(skin_gray, cv2.CV_64F).var())

        return {
            "uniformity": uniformity,
            "redness_ratio": redness_ratio,
            "texture_roughness": texture_roughness,
        }

    # ------------------------------------------------------------------
    # Marks, scars, injuries detection
    # ------------------------------------------------------------------
    def _detect_marks_and_injuries(self, face: np.ndarray) -> Dict[str, Any]:
        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(face, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(face, cv2.COLOR_BGR2LAB)
        h, w = gray.shape

        # Dark spots (scars, moles, birthmarks)
        blur = cv2.GaussianBlur(gray, (15, 15), 0)
        diff = cv2.absdiff(gray, blur)
        _, dark_thresh = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY)
        dark_contours, _ = cv2.findContours(dark_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_area = (h * w) * 0.001
        dark_spots = len([c for c in dark_contours if cv2.contourArea(c) > min_area])

        # Red spots (bruises, inflammation, rash)
        red_mask = cv2.inRange(hsv, np.array([0, 80, 80]), np.array([10, 255, 255]))
        red_mask2 = cv2.inRange(hsv, np.array([170, 80, 80]), np.array([180, 255, 255]))
        red_combined = cv2.bitwise_or(red_mask, red_mask2)
        red_contours, _ = cv2.findContours(red_combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        red_spots = len([c for c in red_contours if cv2.contourArea(c) > min_area])

        # Scar-like regions (linear high-contrast features)
        edges = cv2.Canny(gray, 80, 160)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
        dilated = cv2.dilate(edges, kernel, iterations=1)
        scar_contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        scar_like = 0
        for c in scar_contours:
            area = cv2.contourArea(c)
            if area < min_area:
                continue
            rect = cv2.minAreaRect(c)
            ww, hh = rect[1]
            if ww > 0 and hh > 0:
                aspect = max(ww, hh) / (min(ww, hh) + 1e-6)
                if aspect > 3.0:  # elongated = scar-like
                    scar_like += 1

        total = dark_spots + red_spots + scar_like
        return {
            "total_marks": total,
            "dark_spots": dark_spots,
            "red_spots": red_spots,
            "scar_like_regions": scar_like,
        }

    # ------------------------------------------------------------------
    # Occlusion detection
    # ------------------------------------------------------------------
    def _detect_occlusions(self, face: np.ndarray) -> Dict[str, Any]:
        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        # Glasses detection via eye cascade + edge density in eye region
        eyes = self._eye_cascade.detectMultiScale(gray, 1.1, 5, minSize=(20, 20))
        glasses_detected = False
        if len(eyes) >= 2:
            for ex, ey, ew, eh in eyes:
                eye_region = gray[ey : ey + eh, ex : ex + ew]
                edge_density = float(cv2.Canny(eye_region, 50, 150).mean())
                if edge_density > 40:
                    glasses_detected = True
                    break

        # Hair occlusion (dark region in upper face)
        upper_face = gray[0 : h // 3, :]
        dark_ratio = float(np.mean(upper_face < 40))
        hair_occlusion = dark_ratio > 0.30

        # Partial face (check if significant portion is uniform/missing)
        left_half = gray[:, : w // 2]
        right_half = gray[:, w // 2 :]
        left_var = float(np.var(left_half))
        right_var = float(np.var(right_half))
        partial_face = (left_var < 100 or right_var < 100) and abs(left_var - right_var) > 500

        return {
            "any_occlusion": glasses_detected or hair_occlusion or partial_face,
            "glasses_detected": glasses_detected,
            "hair_occlusion": hair_occlusion,
            "partial_face": partial_face,
        }

    # ------------------------------------------------------------------
    # Symmetry analysis
    # ------------------------------------------------------------------
    def _analyze_symmetry(self, face: np.ndarray) -> Dict[str, Any]:
        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        left = gray[:, : w // 2]
        right = cv2.flip(gray[:, w // 2 :], 1)
        # Resize to match if needed
        min_w = min(left.shape[1], right.shape[1])
        left = left[:, :min_w]
        right = right[:, :min_w]
        diff = np.abs(left.astype(float) - right.astype(float))
        score = 1.0 - min(float(np.mean(diff)) / 50.0, 1.0)
        return {"symmetry_score": max(0.0, score)}

    # ------------------------------------------------------------------
    # Lighting analysis
    # ------------------------------------------------------------------
    def _analyze_lighting(self, face: np.ndarray) -> Dict[str, Any]:
        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY).astype(float)
        h, w = gray.shape
        mean_brightness = float(np.mean(gray))
        shadow_ratio = float(np.mean(gray < 60))

        # Determine lighting direction
        left_mean = float(np.mean(gray[:, : w // 2]))
        right_mean = float(np.mean(gray[:, w // 2 :]))
        top_mean = float(np.mean(gray[: h // 2, :]))
        bottom_mean = float(np.mean(gray[h // 2 :, :]))

        direction = "frontal"
        if abs(left_mean - right_mean) > 15:
            direction = "left" if left_mean > right_mean else "right"
        if abs(top_mean - bottom_mean) > 15:
            vert = "top" if top_mean > bottom_mean else "bottom"
            direction = f"{direction}_{vert}" if direction != "frontal" else vert

        uneven = abs(left_mean - right_mean) > 25 or abs(top_mean - bottom_mean) > 25
        return {
            "mean_brightness": mean_brightness,
            "shadow_ratio": shadow_ratio,
            "light_direction": direction,
            "uneven": uneven,
        }

    # ------------------------------------------------------------------
    # Aging features
    # ------------------------------------------------------------------
    def _region_mean(
        self,
        image: np.ndarray,
        y0: float,
        y1: float,
        x0: float,
        x1: float,
    ) -> float:
        h, w = image.shape[:2]
        yy0 = max(0, min(h - 1, int(round(h * y0))))
        yy1 = max(yy0 + 1, min(h, int(round(h * y1))))
        xx0 = max(0, min(w - 1, int(round(w * x0))))
        xx1 = max(xx0 + 1, min(w, int(round(w * x1))))
        region = image[yy0:yy1, xx0:xx1]
        if region.size == 0:
            return 0.0
        return float(np.mean(region))

    def _analyze_aging_features(
        self,
        face: np.ndarray,
        skin: Optional[Dict[str, Any]] = None,
        marks: Optional[Dict[str, Any]] = None,
        quality: Optional[Dict[str, Any]] = None,
        lighting: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

        blur = cv2.GaussianBlur(gray, (7, 7), 0)
        high_freq = cv2.absdiff(gray, blur)
        wrinkle_density = float(np.mean(high_freq)) / 20.0

        grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = cv2.magnitude(grad_x, grad_y)
        gradient_score = float(np.mean(grad_mag)) / 36.0

        gabor_accum = np.zeros_like(gray, dtype=np.float32)
        for theta in (0.0, np.pi / 4.0, np.pi / 2.0, 3.0 * np.pi / 4.0):
            kernel = cv2.getGaborKernel(
                (17, 17),
                3.0,
                theta,
                8.0,
                0.45,
                0.0,
                ktype=cv2.CV_32F,
            )
            response = cv2.filter2D(gray.astype(np.float32), cv2.CV_32F, kernel)
            gabor_accum += np.abs(response)
        wrinkle_response = float(np.mean(gabor_accum)) / 150.0

        regional_scores = {
            "forehead": self._region_mean(high_freq, 0.05, 0.28, 0.20, 0.80) / 18.0,
            "glabella": self._region_mean(grad_mag, 0.20, 0.38, 0.38, 0.62) / 32.0,
            "crows_feet_left": self._region_mean(high_freq, 0.24, 0.50, 0.02, 0.22) / 18.0,
            "crows_feet_right": self._region_mean(high_freq, 0.24, 0.50, 0.78, 0.98) / 18.0,
            "under_eye": self._region_mean(grad_mag, 0.30, 0.48, 0.18, 0.82) / 34.0,
            "nasolabial": self._region_mean(grad_mag, 0.42, 0.78, 0.20, 0.80) / 38.0,
            "marionette": self._region_mean(high_freq, 0.58, 0.88, 0.28, 0.72) / 20.0,
        }
        regional_wrinkle_score = float(np.mean(list(regional_scores.values())))

        skin = skin or {}
        marks = marks or {}
        quality = quality or {}
        lighting = lighting or {}

        texture_score = min(float(skin.get("texture_roughness", 0.0)) / 900.0, 4.0)
        pigmentation_score = min(
            (
                float(marks.get("dark_spots", 0)) * 0.35
                + float(marks.get("scar_like_regions", 0)) * 0.45
                + float(marks.get("total_marks", 0)) * 0.08
            ),
            4.0,
        )
        uniformity = float(skin.get("uniformity", 1.0))

        composite_age_score = (
            wrinkle_density * 0.18
            + gradient_score * 0.16
            + wrinkle_response * 0.14
            + regional_wrinkle_score * 0.28
            + texture_score * 0.16
            + pigmentation_score * 0.10
            + (1.0 - min(uniformity, 1.0)) * 0.04
        )

        estimated_age = max(18.0, min(88.0, 18.0 + composite_age_score * 18.0))
        if estimated_age < 30:
            age_range = "18-30"
            age_min, age_max = 18, 30
        elif estimated_age < 50:
            age_range = "30-50"
            age_min, age_max = 30, 50
        else:
            age_range = "50+"
            age_min, age_max = 50, 80

        blur_score = float(quality.get("blur_score", 0.0))
        mean_brightness = float(quality.get("mean_brightness", float(np.mean(gray))))
        shadow_ratio = float(lighting.get("shadow_ratio", 0.0))
        lighting_penalty = 0.12 if lighting.get("uneven") else 0.0
        if shadow_ratio > 0.40:
            lighting_penalty += 0.06
        if mean_brightness < 55.0 or mean_brightness > 190.0:
            lighting_penalty += 0.05

        blur_bonus = min(blur_score / 500.0, 0.14)
        signal_spread = float(
            np.std(
                [
                    wrinkle_density,
                    gradient_score,
                    wrinkle_response,
                    regional_wrinkle_score,
                    texture_score,
                    pigmentation_score,
                ]
            )
        )
        confidence = 0.26 + blur_bonus + min(composite_age_score / 10.0, 0.12)
        confidence -= min(signal_spread / 2.5, 0.12)
        confidence -= lighting_penalty
        confidence = max(0.18, min(confidence, 0.58))
        midpoint = int(round((age_min + age_max) / 2))

        return {
            "wrinkle_density": wrinkle_density,
            "gradient_score": gradient_score,
            "wrinkle_response": wrinkle_response,
            "regional_wrinkle_score": regional_wrinkle_score,
            "texture_score": texture_score,
            "pigmentation_score": pigmentation_score,
            "composite_age_score": composite_age_score,
            "estimated_age": int(round(estimated_age)),
            "estimated_age_range": age_range,
            "range_min": age_min,
            "range_max": age_max,
            "range_midpoint": midpoint,
            "confidence": confidence,
            "signal_scores": regional_scores,
            "method": "multi_signal_age_heuristic",
        }

    # ------------------------------------------------------------------
    # Color distribution
    # ------------------------------------------------------------------
    def _analyze_color_distribution(self, face: np.ndarray) -> Dict[str, Any]:
        hsv = cv2.cvtColor(face, cv2.COLOR_BGR2HSV)
        mean_h = float(np.mean(hsv[:, :, 0]))
        mean_s = float(np.mean(hsv[:, :, 1]))
        variance = float(np.var(hsv[:, :, 1]))

        if mean_h < 15 and mean_s > 40:
            tone = "warm/fair"
        elif mean_h < 25:
            tone = "medium"
        else:
            tone = "dark/deep"

        return {
            "dominant_tone": tone,
            "mean_hue": mean_h,
            "mean_saturation": mean_s,
            "color_variance": variance,
        }

    # ------------------------------------------------------------------
    # Reconstruction guidance builder
    # ------------------------------------------------------------------
    def _build_reconstruction_guidance(
        self,
        quality: Dict,
        skin: Dict,
        marks: Dict,
        occlusion: Dict,
        symmetry: Dict,
        lighting: Dict,
        aging: Dict,
        issues: List[str],
    ) -> str:
        """
        Build natural-language guidance for the reconstruction pipeline.
        This is fed to SD 1.5 as prompt modifiers and to CodeFormer as priority flags.
        """
        parts: List[str] = []

        # Base prompt
        parts.append(
            "Reconstruct a clean, high-fidelity forensic portrait preserving identity features."
        )

        # Quality-driven
        if quality["blur_score"] < 60:
            parts.append("Input is blurry — prioritize sharpening and detail recovery.")
        if quality["noise_level"] > 30:
            parts.append("High noise detected — apply denoising before enhancement.")
        if quality["overexposed"]:
            parts.append("Image is overexposed — recover highlight details, normalize exposure.")
        if quality["underexposed"]:
            parts.append("Image is underexposed — brighten shadows, recover dark regions.")

        # Marks and injuries
        if marks["total_marks"] > 0:
            if marks["scar_like_regions"] > 0:
                parts.append(
                    f"Scar-like marks detected ({marks['scar_like_regions']} regions) — "
                    "preserve scars as identity features, do not remove them."
                )
            if marks["dark_spots"] > 0:
                parts.append(
                    f"Dark spots/marks detected ({marks['dark_spots']}) — "
                    "preserve as identifying features."
                )
            if marks["red_spots"] > 0:
                parts.append(
                    f"Red/inflamed areas detected ({marks['red_spots']}) — "
                    "these may be temporary injuries. Normalize skin tone gently."
                )

        # Occlusions
        if occlusion["glasses_detected"]:
            parts.append("Glasses detected — reconstruct eye area behind glasses.")
        if occlusion["hair_occlusion"]:
            parts.append("Hair occluding face — reconstruct forehead/temple region.")
        if occlusion["partial_face"]:
            parts.append("Partial face visible — reconstruct missing facial half using symmetry.")

        # Lighting
        if lighting["uneven"]:
            parts.append(
                f"Uneven lighting (direction: {lighting['light_direction']}) — "
                "normalize to frontal studio lighting."
            )
        if lighting["shadow_ratio"] > 0.4:
            parts.append("Heavy shadows — lift shadow regions to reveal facial detail.")

        # Symmetry
        if symmetry["symmetry_score"] < 0.75:
            parts.append(
                "Significant facial asymmetry detected — preserve natural asymmetry, "
                "do not artificially symmetrize."
            )

        # Aging
        parts.append(
            f"Subject appears {aging['estimated_age_range']} range — "
            "maintain age-appropriate skin texture and features."
        )

        return " ".join(parts)

    # ------------------------------------------------------------------
    # Empty study fallback
    # ------------------------------------------------------------------
    def _empty_study(self, reason: str) -> Dict[str, Any]:
        return {
            "image_path": "",
            "face_detected": False,
            "quality": {},
            "skin_analysis": {},
            "marks_and_injuries": {"total_marks": 0, "dark_spots": 0, "red_spots": 0, "scar_like_regions": 0},
            "occlusions": {"any_occlusion": False},
            "symmetry": {"symmetry_score": 0.0},
            "lighting": {},
            "aging_features": {},
            "color_analysis": {},
            "findings": [f"Study could not be completed: {reason}"],
            "issues_detected": [reason],
            "reconstruction_guidance": "Standard reconstruction — no detailed study available.",
        }
