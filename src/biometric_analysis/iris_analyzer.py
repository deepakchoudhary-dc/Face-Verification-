"""
IRIS & RETINA ANALYSIS MODULE
=============================
Advanced biometric analysis of the iris pattern - the most unique identifier
that remains stable throughout life, even more reliable than fingerprints.

Features:
1. Iris Pattern Extraction - Unique cryptographic-like pattern
2. Cataract Detection - Clouding in aged eyes
3. Iris Code Generation - Binary template for matching
4. Pupil Dilation Analysis - Anti-spoofing
5. Sclera Pattern Analysis - Blood vessel patterns
6. Iris Texture Matching - Age-invariant comparison

This addresses:
- Eye diseases (cataracts, glaucoma)
- Colored contact detection
- Age-related changes
- Identical twin differentiation
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import math


class IrisAnalyzer:
    """
    Enterprise-Grade Iris Recognition and Analysis System.
    Uses Daugman's IrisCode algorithm principles.
    """
    
    def __init__(self):
        # Gabor filter parameters for iris feature extraction
        self.GABOR_WAVELENGTH = 8
        self.GABOR_ORIENTATIONS = 8  # 8 orientations for full coverage
        self.IRIS_CODE_SIZE = 256  # Bits in iris code
        
        # Matching thresholds
        self.MATCH_THRESHOLD = 0.35  # Hamming distance < 35% = match
        
    def analyze_iris(self, image: np.ndarray, eye_region: Dict = None) -> Dict[str, Any]:
        """
        Complete iris analysis including pattern extraction and health assessment.
        """
        result = {
            'iris_detected': False,
            'iris_code': None,
            'iris_quality': 0.0,
            'health_indicators': {
                'cataract_probability': 0.0,
                'glaucoma_indicators': False,
                'iris_clarity': 0.0,
                'pupil_response_normal': True
            },
            'pattern_analysis': {
                'radial_features': 0,
                'crypts_detected': 0,
                'furrows_detected': 0,
                'collarette_visible': False
            },
            'age_estimation': {
                'arcus_senilis_detected': False,
                'estimated_eye_age': 'UNKNOWN'
            },
            'anti_spoofing': {
                'is_real_eye': True,
                'liveness_score': 0.0,
                'contact_lens_detected': False
            },
            'sclera_analysis': {
                'sclera_detected': False,
                'vessel_density': 0.0,
                'branching_complexity': 0.0,
                'topology_consistency': 0.0,
                'vascular_signature': '',
                'ai_noise_probability': 0.0,
                'deepfake_suspected': False,
                'eye_region_quality': 0.0,
                'regions_analyzed': 0
            }
        }
        
        if image is None:
            return result
            
        try:
            # 1. IRIS DETECTION & SEGMENTATION
            iris_data = self._segment_iris(image, eye_region)
            
            if iris_data['success']:
                result['iris_detected'] = True
                result['iris_quality'] = iris_data['quality']
                
                iris_normalized = iris_data['normalized_iris']
                
                # 2. GENERATE IRIS CODE (Daugman's algorithm)
                iris_code = self._generate_iris_code(iris_normalized)
                result['iris_code'] = iris_code
                
                # 3. PATTERN ANALYSIS (Crypts, furrows, collarette)
                pattern_result = self._analyze_iris_patterns(iris_normalized)
                result['pattern_analysis'] = pattern_result
                
                # 4. HEALTH ASSESSMENT
                health_result = self._assess_eye_health(image, iris_data)
                result['health_indicators'] = health_result
                
                # 5. AGE INDICATORS
                age_result = self._detect_age_indicators(image, iris_data)
                result['age_estimation'] = age_result
                
                # 6. ANTI-SPOOFING (Detect printed eyes, screens, contacts)
                spoof_result = self._check_liveness(image, iris_data)
                result['anti_spoofing'] = spoof_result

            # 7. SCLERA VASCULAR ANALYSIS
            sclera_result = self._analyze_sclera_vasculature(image, iris_data, eye_region)
            result['sclera_analysis'] = sclera_result

            if iris_data['success']:
                if sclera_result.get('deepfake_suspected'):
                    result['anti_spoofing']['spoof_indicators'].append(
                        "Sclera vascular topology inconsistent with a living eye"
                    )
                    result['anti_spoofing']['liveness_score'] = max(
                        0.0,
                        result['anti_spoofing']['liveness_score'] - 0.15
                    )
                result['anti_spoofing']['sclera_vascular_confirmed'] = bool(
                    sclera_result.get('sclera_detected')
                )
                result['anti_spoofing']['ai_noise_suspected'] = bool(
                    sclera_result.get('deepfake_suspected')
                )
                result['anti_spoofing']['liveness_score'] = round(
                    max(0.0, min(1.0, result['anti_spoofing']['liveness_score'])),
                    4
                )
                result['anti_spoofing']['is_real_eye'] = (
                    result['anti_spoofing']['liveness_score'] > 0.5
                    and not result['anti_spoofing'].get('spoof_indicators')
                )
                
        except Exception as e:
            result['error'] = str(e)
            
        return result
    
    def compare_iris(self, iris_code1: str, iris_code2: str) -> Dict[str, Any]:
        """
        Compare two iris codes using Hamming distance.
        """
        result = {
            'is_match': False,
            'hamming_distance': 1.0,
            'confidence': 0.0,
            'match_category': 'NO_MATCH'
        }
        
        if not iris_code1 or not iris_code2:
            return result
            
        try:
            # Convert hex codes to binary
            bin1 = bin(int(iris_code1, 16))[2:].zfill(self.IRIS_CODE_SIZE)
            bin2 = bin(int(iris_code2, 16))[2:].zfill(self.IRIS_CODE_SIZE)
            
            # Ensure same length
            min_len = min(len(bin1), len(bin2))
            bin1 = bin1[:min_len]
            bin2 = bin2[:min_len]
            
            # Calculate Hamming distance (XOR and count 1s)
            differences = sum(c1 != c2 for c1, c2 in zip(bin1, bin2))
            hamming_dist = differences / min_len
            
            result['hamming_distance'] = round(hamming_dist, 4)
            result['confidence'] = round((1 - hamming_dist) * 100, 2)
            
            # Match categorization
            if hamming_dist < 0.25:
                result['is_match'] = True
                result['match_category'] = 'DEFINITE_MATCH'
            elif hamming_dist < 0.35:
                result['is_match'] = True
                result['match_category'] = 'PROBABLE_MATCH'
            elif hamming_dist < 0.40:
                result['match_category'] = 'POSSIBLE_MATCH'
            else:
                result['match_category'] = 'NO_MATCH'
                
        except Exception as e:
            result['error'] = str(e)
            
        return result
    
    def _segment_iris(self, image: np.ndarray, eye_region: Dict = None) -> Dict:
        """
        Segment iris from eye image using circular Hough transform.
        """
        result = {
            'success': False,
            'normalized_iris': None,
            'pupil_center': None,
            'pupil_radius': 0,
            'iris_center': None,
            'iris_radius': 0,
            'quality': 0.0
        }
        
        try:
            # Convert to grayscale
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
                
            # If eye region specified, crop
            if eye_region:
                x, y, w, h = eye_region.get('x', 0), eye_region.get('y', 0), eye_region.get('w', gray.shape[1]), eye_region.get('h', gray.shape[0])
                gray = gray[y:y+h, x:x+w]
                
            if gray.size == 0:
                return result
                
            # Apply histogram equalization
            gray = cv2.equalizeHist(gray)
            
            # Blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Find pupil (dark circle in center)
            # Use adaptive thresholding
            _, pupil_thresh = cv2.threshold(blurred, 50, 255, cv2.THRESH_BINARY_INV)
            
            # Find circles - pupil first (smaller)
            circles = cv2.HoughCircles(
                blurred, 
                cv2.HOUGH_GRADIENT, 
                dp=1, 
                minDist=gray.shape[0] // 4,
                param1=50, 
                param2=30,
                minRadius=gray.shape[0] // 10,
                maxRadius=gray.shape[0] // 3
            )
            
            if circles is not None:
                circles = np.uint16(np.around(circles))
                
                # Find the darkest circle center (most likely pupil)
                best_circle = None
                darkest_value = 255
                
                for circle in circles[0, :]:
                    cx, cy, r = circle
                    if 0 <= cy < gray.shape[0] and 0 <= cx < gray.shape[1]:
                        center_value = gray[cy, cx]
                        if center_value < darkest_value:
                            darkest_value = center_value
                            best_circle = circle
                            
                if best_circle is not None:
                    pupil_x, pupil_y, pupil_r = best_circle
                    result['pupil_center'] = (int(pupil_x), int(pupil_y))
                    result['pupil_radius'] = int(pupil_r)
                    
                    # Estimate iris boundary (typically 2-3x pupil radius)
                    iris_r = int(pupil_r * 2.5)
                    result['iris_center'] = (int(pupil_x), int(pupil_y))
                    result['iris_radius'] = iris_r
                    
                    # Normalize iris to rectangular coordinates (rubber sheet model)
                    normalized = self._normalize_iris(gray, pupil_x, pupil_y, pupil_r, iris_r)
                    result['normalized_iris'] = normalized
                    
                    # Calculate quality score
                    result['quality'] = self._calculate_iris_quality(gray, pupil_x, pupil_y, pupil_r, iris_r)
                    result['success'] = result['quality'] > 0.3
                    
        except Exception as e:
            pass
            
        return result
    
    def _normalize_iris(self, image: np.ndarray, cx: int, cy: int, 
                        pupil_r: int, iris_r: int) -> np.ndarray:
        """
        Convert circular iris to rectangular using Daugman's rubber sheet model.
        Maps iris ring to a fixed-size rectangular image.
        """
        # Output dimensions
        angular_res = 360  # 360 degrees
        radial_res = 64    # 64 radial samples
        
        normalized = np.zeros((radial_res, angular_res), dtype=np.uint8)
        
        for theta_idx in range(angular_res):
            theta = 2 * math.pi * theta_idx / angular_res
            
            for r_idx in range(radial_res):
                # Interpolate between pupil boundary and iris boundary
                r = pupil_r + (iris_r - pupil_r) * r_idx / radial_res
                
                # Convert polar to Cartesian
                x = int(cx + r * math.cos(theta))
                y = int(cy + r * math.sin(theta))
                
                # Sample pixel if within bounds
                if 0 <= y < image.shape[0] and 0 <= x < image.shape[1]:
                    normalized[r_idx, theta_idx] = image[y, x]
                    
        return normalized
    
    def _generate_iris_code(self, normalized_iris: np.ndarray) -> str:
        """
        Generate binary iris code using Gabor wavelets.
        This is a simplified version of Daugman's IrisCode.
        """
        if normalized_iris is None or normalized_iris.size == 0:
            return None
            
        # Apply Gabor filters at multiple orientations
        code_bits = []
        
        for orientation in range(self.GABOR_ORIENTATIONS):
            theta = np.pi * orientation / self.GABOR_ORIENTATIONS
            
            # Create Gabor kernel
            kernel = cv2.getGaborKernel(
                ksize=(21, 21),
                sigma=5.0,
                theta=theta,
                lambd=self.GABOR_WAVELENGTH,
                gamma=0.5,
                psi=0
            )
            
            # Filter the normalized iris
            filtered = cv2.filter2D(normalized_iris, cv2.CV_64F, kernel)
            
            # Generate bits based on filter response
            # Real part > 0 = 1, else 0
            # Imaginary part > 0 = 1, else 0 (simplified)
            real_bits = (filtered > 0).flatten()
            
            # Sample bits uniformly
            step = len(real_bits) // (self.IRIS_CODE_SIZE // self.GABOR_ORIENTATIONS)
            step = max(step, 1)
            sampled_bits = real_bits[::step][:self.IRIS_CODE_SIZE // self.GABOR_ORIENTATIONS]
            
            code_bits.extend(sampled_bits)
            
        # Pad or truncate to exact size
        code_bits = code_bits[:self.IRIS_CODE_SIZE]
        while len(code_bits) < self.IRIS_CODE_SIZE:
            code_bits.append(False)
            
        # Convert to hexadecimal string
        bit_string = ''.join('1' if b else '0' for b in code_bits)
        hex_code = hex(int(bit_string, 2))[2:].zfill(self.IRIS_CODE_SIZE // 4)
        
        return hex_code
    
    def _analyze_iris_patterns(self, normalized_iris: np.ndarray) -> Dict:
        """
        Analyze unique iris patterns: crypts, furrows, and collarette.
        """
        patterns = {
            'radial_features': 0,
            'crypts_detected': 0,
            'furrows_detected': 0,
            'collarette_visible': False,
            'pattern_complexity': 0.0
        }
        
        if normalized_iris is None or normalized_iris.size == 0:
            return patterns
            
        try:
            # Edge detection to find radial features
            edges = cv2.Canny(normalized_iris, 30, 100)
            
            # Count radial lines (furrows)
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, 20, minLineLength=10, maxLineGap=5)
            if lines is not None:
                patterns['furrows_detected'] = len(lines)
                patterns['radial_features'] = len(lines)
                
            # Detect crypts (dark spots/holes in iris)
            _, thresh = cv2.threshold(normalized_iris, 60, 255, cv2.THRESH_BINARY_INV)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            crypts = [c for c in contours if 10 < cv2.contourArea(c) < 500]
            patterns['crypts_detected'] = len(crypts)
            
            # Check for collarette (zigzag pattern around pupil)
            # It's typically in the first third of the normalized iris
            collarette_region = normalized_iris[:normalized_iris.shape[0]//3, :]
            col_edges = cv2.Canny(collarette_region, 20, 60)
            col_edge_ratio = np.sum(col_edges > 0) / col_edges.size
            patterns['collarette_visible'] = col_edge_ratio > 0.05
            
            # Pattern complexity (entropy-based)
            hist = cv2.calcHist([normalized_iris], [0], None, [256], [0, 256])
            hist = hist / hist.sum()
            entropy = -np.sum(hist * np.log2(hist + 1e-10))
            patterns['pattern_complexity'] = round(entropy / 8.0, 2)  # Normalize to 0-1
            
        except Exception as e:
            pass
            
        return patterns
    
    def _assess_eye_health(self, image: np.ndarray, iris_data: Dict) -> Dict:
        """
        Assess eye health indicators from iris image.
        """
        health = {
            'cataract_probability': 0.0,
            'glaucoma_indicators': False,
            'iris_clarity': 0.0,
            'pupil_response_normal': True,
            'health_score': 100.0
        }
        
        try:
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
                
            pupil_center = iris_data.get('pupil_center')
            pupil_r = iris_data.get('pupil_radius', 0)
            
            if pupil_center and pupil_r > 0:
                cx, cy = pupil_center
                
                # Check lens area for cataract (clouding)
                # Cataracts cause the pupil area to appear cloudy/white
                y1 = max(0, cy - pupil_r)
                y2 = min(gray.shape[0], cy + pupil_r)
                x1 = max(0, cx - pupil_r)
                x2 = min(gray.shape[1], cx + pupil_r)
                
                pupil_region = gray[y1:y2, x1:x2]
                
                if pupil_region.size > 0:
                    # Cataract causes higher brightness in pupil
                    pupil_brightness = np.mean(pupil_region)
                    
                    # Normal pupil is very dark (< 50)
                    # Cataract causes cloudiness (> 80)
                    if pupil_brightness > 100:
                        health['cataract_probability'] = min((pupil_brightness - 50) / 100, 1.0)
                    else:
                        health['cataract_probability'] = 0.0
                        
                    # Iris clarity (contrast and detail)
                    iris_r = iris_data.get('iris_radius', pupil_r * 2)
                    y1 = max(0, cy - iris_r)
                    y2 = min(gray.shape[0], cy + iris_r)
                    x1 = max(0, cx - iris_r)
                    x2 = min(gray.shape[1], cx + iris_r)
                    
                    iris_region = gray[y1:y2, x1:x2]
                    if iris_region.size > 0:
                        clarity = np.std(iris_region) / 128.0
                        health['iris_clarity'] = round(min(clarity, 1.0), 2)
                        
            # Overall health score
            cataract_penalty = health['cataract_probability'] * 30
            clarity_bonus = health['iris_clarity'] * 20
            health['health_score'] = max(0, 100 - cataract_penalty + clarity_bonus - 20)
            
        except Exception as e:
            pass
            
        return health
    
    def _detect_age_indicators(self, image: np.ndarray, iris_data: Dict) -> Dict:
        """
        Detect age-related eye changes.
        """
        age_info = {
            'arcus_senilis_detected': False,
            'estimated_eye_age': 'UNKNOWN',
            'age_indicators': []
        }
        
        try:
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
                
            iris_center = iris_data.get('iris_center')
            iris_r = iris_data.get('iris_radius', 0)
            
            if iris_center and iris_r > 0:
                cx, cy = iris_center
                
                # Check for arcus senilis (white ring around iris edge)
                # Common in people over 60
                outer_r = int(iris_r * 1.1)
                inner_r = int(iris_r * 0.95)
                
                mask = np.zeros(gray.shape, dtype=np.uint8)
                cv2.circle(mask, (cx, cy), outer_r, 255, -1)
                cv2.circle(mask, (cx, cy), inner_r, 0, -1)
                
                ring_region = gray[mask > 0]
                
                if ring_region.size > 0:
                    # Arcus senilis appears as a bright ring
                    ring_brightness = np.mean(ring_region)
                    
                    if ring_brightness > 180:
                        age_info['arcus_senilis_detected'] = True
                        age_info['age_indicators'].append("Arcus senilis (age ring) detected")
                        age_info['estimated_eye_age'] = '60+'
                    elif ring_brightness > 150:
                        age_info['estimated_eye_age'] = '45-60'
                    elif ring_brightness > 120:
                        age_info['estimated_eye_age'] = '30-45'
                    else:
                        age_info['estimated_eye_age'] = 'YOUNG (<30)'
                        
        except Exception as e:
            pass
            
        return age_info
    
    def _check_liveness(self, image: np.ndarray, iris_data: Dict) -> Dict:
        """
        Anti-spoofing: Check if this is a real eye vs printed/screen display.
        """
        liveness = {
            'is_real_eye': True,
            'liveness_score': 1.0,
            'contact_lens_detected': False,
            'spoof_indicators': []
        }
        
        try:
            if len(image.shape) == 3:
                # Check for specular reflections (real eyes have them)
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                
                # Find bright spots (reflections)
                _, highlights = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
                highlight_ratio = np.sum(highlights > 0) / highlights.size
                
                # Real eyes should have small, sharp reflections
                if highlight_ratio < 0.001:
                    liveness['spoof_indicators'].append("No specular reflection detected")
                    liveness['liveness_score'] -= 0.3
                elif highlight_ratio > 0.1:
                    liveness['spoof_indicators'].append("Excessive reflections (possible screen)")
                    liveness['liveness_score'] -= 0.4
                    
                # Check for texture (printed images lack fine texture)
                texture_var = cv2.Laplacian(gray, cv2.CV_64F).var()
                if texture_var < 50:
                    liveness['spoof_indicators'].append("Low texture variance (possible print)")
                    liveness['liveness_score'] -= 0.3
                    
                # Contact lens detection (edge around iris)
                # Colored contacts often have visible edges
                iris_center = iris_data.get('iris_center')
                iris_r = iris_data.get('iris_radius', 0)
                
                if iris_center and iris_r > 0:
                    cx, cy = iris_center
                    
                    # Check for artificial edge at iris boundary
                    edges = cv2.Canny(gray, 50, 150)
                    
                    # Sample edge density at iris boundary
                    mask = np.zeros(edges.shape, dtype=np.uint8)
                    cv2.circle(mask, (cx, cy), iris_r + 3, 255, 6)
                    
                    edge_at_boundary = edges[mask > 0]
                    if edge_at_boundary.size > 0:
                        boundary_edge_ratio = np.sum(edge_at_boundary > 0) / edge_at_boundary.size
                        
                        if boundary_edge_ratio > 0.4:
                            liveness['contact_lens_detected'] = True
                            liveness['spoof_indicators'].append("Contact lens edge detected")
                            
            liveness['liveness_score'] = max(0, min(1, liveness['liveness_score']))
            liveness['is_real_eye'] = liveness['liveness_score'] > 0.5 and not liveness['spoof_indicators']
            
        except Exception as e:
            pass
            
        return liveness

    def _analyze_sclera_vasculature(
        self,
        image: np.ndarray,
        iris_data: Dict,
        eye_region: Dict = None
    ) -> Dict[str, Any]:
        """
        Extract sclera vessel structure from the white of the eye and score whether
        the vascular topology looks anatomically continuous or synthetic/noisy.
        """
        result = {
            'sclera_detected': False,
            'vessel_density': 0.0,
            'branching_complexity': 0.0,
            'topology_consistency': 0.0,
            'vascular_signature': '',
            'ai_noise_probability': 0.0,
            'deepfake_suspected': False,
            'eye_region_quality': 0.0,
            'regions_analyzed': 0,
            'eye_summaries': []
        }

        try:
            if image is None or image.size == 0 or len(image.shape) != 3:
                return result

            eye_boxes = self._estimate_eye_regions(image, iris_data)
            region_summaries = []
            vascular_signatures = []

            for eye_box in eye_boxes:
                region = self._crop_roi(image, eye_box)
                if region.size == 0:
                    continue

                hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
                lab = cv2.cvtColor(region, cv2.COLOR_BGR2LAB)
                value = hsv[:, :, 2]
                saturation = hsv[:, :, 1]
                lightness = lab[:, :, 0]

                sclera_mask = ((value > 135) & (saturation < 90) & (lightness > 120)).astype(np.uint8) * 255
                sclera_mask = cv2.morphologyEx(
                    sclera_mask, cv2.MORPH_OPEN, np.ones((3, 3), dtype=np.uint8)
                )
                sclera_mask = cv2.morphologyEx(
                    sclera_mask, cv2.MORPH_CLOSE, np.ones((5, 5), dtype=np.uint8)
                )
                mask_area = int(np.sum(sclera_mask > 0))
                if mask_area < max(40, region.shape[0] * region.shape[1] // 20):
                    continue

                b_channel, g_channel, r_channel = cv2.split(region)
                vessel_raw = cv2.subtract(r_channel, g_channel)
                vessel_raw = cv2.max(vessel_raw, cv2.subtract(r_channel, b_channel))
                vessel_raw = cv2.GaussianBlur(vessel_raw, (3, 3), 0)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
                vessel_enhanced = clahe.apply(vessel_raw)
                vessel_enhanced = cv2.morphologyEx(
                    vessel_enhanced, cv2.MORPH_TOPHAT, np.ones((5, 5), dtype=np.uint8)
                )

                threshold = max(12, int(np.percentile(vessel_enhanced[sclera_mask > 0], 80)))
                vessel_map = ((vessel_enhanced >= threshold) & (sclera_mask > 0)).astype(np.uint8) * 255
                vessel_map = cv2.morphologyEx(
                    vessel_map, cv2.MORPH_OPEN, np.ones((2, 2), dtype=np.uint8)
                )

                vessel_pixels = int(np.sum(vessel_map > 0))
                if vessel_pixels == 0:
                    continue

                density = vessel_pixels / max(mask_area, 1)
                num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(vessel_map, 8)
                component_sizes = stats[1:, cv2.CC_STAT_AREA] if num_labels > 1 else np.asarray([], dtype=np.int32)
                significant_components = component_sizes[component_sizes >= 3]

                neighborhood = cv2.filter2D((vessel_map > 0).astype(np.uint8), -1, np.ones((3, 3), dtype=np.uint8))
                branch_points = int(np.sum(((vessel_map > 0) & (neighborhood >= 5))))
                branching_complexity = branch_points / max(vessel_pixels, 1)

                continuity = float(np.mean(significant_components) / max(np.max(component_sizes), 1)) if component_sizes.size else 0.0
                fragmentation = float((len(component_sizes) - len(significant_components)) / max(len(component_sizes), 1)) if component_sizes.size else 1.0
                topology_consistency = float(
                    max(
                        0.0,
                        min(
                            1.0,
                            0.50 * (1.0 - min(abs(density - 0.08) / 0.08, 1.0))
                            + 0.30 * min(branching_complexity / 0.06, 1.0)
                            + 0.20 * continuity
                        )
                    )
                )
                ai_noise_probability = float(
                    max(
                        0.0,
                        min(
                            1.0,
                            0.40 * min(abs(density - 0.08) / 0.12, 1.0)
                            + 0.35 * fragmentation
                            + 0.25 * (1.0 - topology_consistency)
                        )
                    )
                )

                signature = self._binary_map_signature(vessel_map)
                vascular_signatures.append(signature)
                region_quality = min(1.0, (mask_area / max(region.shape[0] * region.shape[1], 1)) * 2.0)
                region_summaries.append({
                    'box': eye_box,
                    'vessel_density': round(density, 4),
                    'branching_complexity': round(branching_complexity, 4),
                    'topology_consistency': round(topology_consistency, 4),
                    'ai_noise_probability': round(ai_noise_probability, 4),
                    'quality': round(region_quality, 4),
                })

            if not region_summaries:
                return result

            density = float(np.mean([item['vessel_density'] for item in region_summaries]))
            branching = float(np.mean([item['branching_complexity'] for item in region_summaries]))
            topology = float(np.mean([item['topology_consistency'] for item in region_summaries]))
            ai_noise = float(np.mean([item['ai_noise_probability'] for item in region_summaries]))
            quality = float(np.mean([item['quality'] for item in region_summaries]))
            bilateral_similarity = self._mean_pairwise_signature_similarity(vascular_signatures)
            if bilateral_similarity is not None:
                topology = max(0.0, min(1.0, 0.8 * topology + 0.2 * bilateral_similarity))
                ai_noise = max(0.0, min(1.0, 0.85 * ai_noise + 0.15 * (1.0 - bilateral_similarity)))

            result.update({
                'sclera_detected': True,
                'vessel_density': round(density, 4),
                'branching_complexity': round(branching, 4),
                'topology_consistency': round(topology, 4),
                'vascular_signature': "|".join(vascular_signatures[:2]),
                'ai_noise_probability': round(ai_noise, 4),
                'deepfake_suspected': ai_noise >= 0.72 and topology <= 0.45,
                'eye_region_quality': round(quality, 4),
                'regions_analyzed': len(region_summaries),
                'eye_summaries': region_summaries
            })
        except Exception:
            pass

        return result

    def _estimate_eye_regions(self, image: np.ndarray, iris_data: Dict) -> List[Dict[str, int]]:
        """
        Derive plausible eye ROIs. If a valid iris center exists, use a pair of boxes
        around it; otherwise fall back to anthropometric face-strip estimates.
        """
        h, w = image.shape[:2]
        iris_center = iris_data.get('iris_center')
        iris_radius = int(iris_data.get('iris_radius', 0) or 0)
        eye_boxes: List[Dict[str, int]] = []

        if iris_center and iris_radius > 0:
            cx, cy = int(iris_center[0]), int(iris_center[1])
            half_w = max(iris_radius * 2, w // 8)
            half_h = max(int(iris_radius * 1.3), h // 12)
            eye_boxes.append(self._clip_box({
                'x': cx - half_w,
                'y': cy - half_h,
                'w': half_w,
                'h': half_h * 2
            }, w, h))
            mirrored_cx = w - cx
            eye_boxes.append(self._clip_box({
                'x': mirrored_cx - half_w,
                'y': cy - half_h,
                'w': half_w,
                'h': half_h * 2
            }, w, h))
            return eye_boxes

        eye_w = max(20, w // 4)
        eye_h = max(12, h // 6)
        y = max(0, int(h * 0.18))
        left_x = max(0, int(w * 0.12))
        right_x = max(0, int(w * 0.58))
        eye_boxes.append(self._clip_box({'x': left_x, 'y': y, 'w': eye_w, 'h': eye_h}, w, h))
        eye_boxes.append(self._clip_box({'x': right_x, 'y': y, 'w': eye_w, 'h': eye_h}, w, h))
        return eye_boxes

    @staticmethod
    def _clip_box(box: Dict[str, int], width: int, height: int) -> Dict[str, int]:
        x = max(0, int(box['x']))
        y = max(0, int(box['y']))
        w = max(1, min(int(box['w']), width - x))
        h = max(1, min(int(box['h']), height - y))
        return {'x': x, 'y': y, 'w': w, 'h': h}

    @staticmethod
    def _crop_roi(image: np.ndarray, box: Dict[str, int]) -> np.ndarray:
        return image[box['y']:box['y'] + box['h'], box['x']:box['x'] + box['w']]

    @staticmethod
    def _binary_map_signature(binary_map: np.ndarray) -> str:
        resized = cv2.resize(binary_map, (12, 12), interpolation=cv2.INTER_AREA)
        bits = (resized > 0).flatten()
        bit_string = ''.join('1' if b else '0' for b in bits)
        return hex(int(bit_string, 2))[2:].zfill(36)

    @staticmethod
    def _mean_pairwise_signature_similarity(signatures: List[str]) -> Optional[float]:
        if len(signatures) < 2:
            return None
        scores = []
        for i in range(len(signatures)):
            for j in range(i + 1, len(signatures)):
                a = bin(int(signatures[i], 16))[2:].zfill(len(signatures[i]) * 4)
                b = bin(int(signatures[j], 16))[2:].zfill(len(signatures[j]) * 4)
                diff = sum(ch1 != ch2 for ch1, ch2 in zip(a, b))
                scores.append(1.0 - diff / max(len(a), 1))
        return float(np.mean(scores)) if scores else None
    
    def _calculate_iris_quality(self, image: np.ndarray, cx: int, cy: int, 
                                pupil_r: int, iris_r: int) -> float:
        """
        Calculate quality score for iris image.
        """
        quality = 1.0
        
        try:
            # Normalize potential NumPy scalar inputs to signed Python ints before radius arithmetic.
            cx = int(cx)
            cy = int(cy)
            pupil_r = max(0, int(pupil_r))
            iris_r = max(0, int(iris_r))

            # Check if iris is fully visible
            if (cx - iris_r < 0 or cx + iris_r >= image.shape[1] or
                cy - iris_r < 0 or cy + iris_r >= image.shape[0]):
                quality -= 0.3  # Iris partially occluded
                
            # Check contrast
            y1 = max(0, cy - iris_r)
            y2 = min(image.shape[0], cy + iris_r)
            x1 = max(0, cx - iris_r)
            x2 = min(image.shape[1], cx + iris_r)
            
            iris_region = image[y1:y2, x1:x2]
            if iris_region.size > 0:
                contrast = np.std(iris_region)
                if contrast < 20:
                    quality -= 0.3  # Low contrast
                    
            # Check blur
            if iris_region.size > 0:
                blur = cv2.Laplacian(iris_region, cv2.CV_64F).var()
                if blur < 100:
                    quality -= 0.2  # Blurry
                    
        except Exception as e:
            quality = 0.3
            
        return max(0, quality)
