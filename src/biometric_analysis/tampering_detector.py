"""
FACIAL TAMPERING DETECTION MODULE
=================================
Advanced detection of photo manipulation where someone has:
- Transplanted facial features (nose, eyes, mouth) from another person
- Digitally altered facial proportions
- Spliced different photos together
- Used AI face-swap tools

Techniques:
1. Feature Boundary Analysis - Detect unnatural transitions
2. Lighting Consistency Check - Shadows should match across features
3. Skin Texture Continuity - Texture should be uniform
4. Color Histogram Matching - Skin tones should be consistent
5. Noise Pattern Analysis - Different cameras have different noise signatures
6. JPEG Ghost Detection - Multiple compression artifacts
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from scipy import ndimage
from scipy.stats import entropy


class TamperingDetector:
    """
    Military-Grade Facial Tampering Detection System.
    Detects when facial features have been transplanted or modified.
    """
    
    def __init__(self):
        self.SUSPICIOUS_THRESHOLD = 0.65  # 65% tampering probability
        
    def analyze_tampering(self, image: np.ndarray, face_box: Dict, landmarks: Dict = None) -> Dict[str, Any]:
        """
        Comprehensive tampering analysis on a facial image.
        """
        result = {
            'tampering_detected': False,
            'tampering_probability': 0.0,
            'feature_analysis': {},
            'boundary_anomalies': [],
            'lighting_inconsistencies': [],
            'texture_anomalies': [],
            'splicing_indicators': [],
            'micro_seam_analysis': {
                'seam_detected': False,
                'seam_probability': 0.0,
                'edge_zone_scores': {},
                'candidate_regions': [],
                'highlight_box': None,
                'highlight_box_normalized': None,
                'summary': 'No micro-seam anomalies detected'
            },
            'risk_level': 'LOW',
            'detailed_report': ''
        }
        
        if image is None:
            return result
            
        try:
            # 1. FEATURE BOUNDARY ANALYSIS
            boundary_score, boundary_details = self._analyze_feature_boundaries(image, face_box, landmarks)
            result['feature_analysis']['boundary_integrity'] = {
                'score': round(1 - boundary_score, 2),  # Higher = more natural
                'anomalies': boundary_details
            }
            if boundary_details:
                result['boundary_anomalies'] = boundary_details
            
            # 2. LIGHTING CONSISTENCY
            lighting_score, lighting_issues = self._check_lighting_consistency(image, face_box, landmarks)
            result['feature_analysis']['lighting_consistency'] = {
                'score': round(lighting_score, 2),
                'issues': lighting_issues
            }
            if lighting_issues:
                result['lighting_inconsistencies'] = lighting_issues
            
            # 3. SKIN TEXTURE CONTINUITY
            texture_score, texture_anomalies = self._analyze_texture_continuity(image, face_box)
            result['feature_analysis']['texture_continuity'] = {
                'score': round(texture_score, 2),
                'anomalies': texture_anomalies
            }
            if texture_anomalies:
                result['texture_anomalies'] = texture_anomalies
            
            # 4. COLOR HISTOGRAM ANALYSIS (Skin tone consistency)
            color_score, color_issues = self._analyze_color_consistency(image, face_box, landmarks)
            result['feature_analysis']['color_consistency'] = {
                'score': round(color_score, 2),
                'issues': color_issues
            }
            
            # 5. NOISE PATTERN ANALYSIS (Camera fingerprint)
            noise_score, noise_anomalies = self._analyze_noise_patterns(image, face_box)
            result['feature_analysis']['noise_consistency'] = {
                'score': round(noise_score, 2),
                'anomalies': noise_anomalies
            }
            
            # 6. JPEG GHOST DETECTION (Multiple saves/edits)
            ghost_score, ghost_regions = self._detect_jpeg_ghosts(image)
            result['feature_analysis']['compression_integrity'] = {
                'score': round(1 - ghost_score, 2),
                'ghost_regions_detected': len(ghost_regions) > 0
            }
            if ghost_regions:
                result['splicing_indicators'].extend(ghost_regions)
            
            # 7. EDGE DISCONTINUITY DETECTION
            edge_score, edge_issues = self._detect_edge_discontinuities(image, face_box, landmarks)
            result['feature_analysis']['edge_continuity'] = {
                'score': round(edge_score, 2),
                'discontinuities': edge_issues
            }

            # 8. MICRO-SEAM BOUNDARY ANALYSIS
            seam_result = self._analyze_micro_seam_boundaries(image, face_box, landmarks)
            result['micro_seam_analysis'] = seam_result
            result['feature_analysis']['micro_seam_boundary'] = {
                'score': round(1 - seam_result.get('seam_probability', 0.0), 2),
                'seam_detected': seam_result.get('seam_detected', False),
                'candidate_regions': seam_result.get('candidate_regions', []),
                'highlight_box': seam_result.get('highlight_box')
            }
            if seam_result.get('candidate_regions'):
                result['splicing_indicators'].append(
                    f"Micro-seam scan flagged {len(seam_result['candidate_regions'])} boundary region(s)"
                )
            if seam_result.get('seam_detected'):
                result['boundary_anomalies'].append(seam_result.get('summary', 'Micro-seam anomaly detected'))
            
            # CALCULATE OVERALL TAMPERING PROBABILITY
            scores = [
                1 - boundary_score,  # Invert (lower boundary score = better)
                lighting_score,
                texture_score,
                color_score,
                noise_score,
                1 - ghost_score,  # Invert
                edge_score,
                max(0.0, 1 - seam_result.get('seam_probability', 0.0))
            ]
            
            # Weighted average (micro-seam signal is additive and intentionally conservative)
            weights = [0.19, 0.2, 0.15, 0.15, 0.1, 0.1, 0.08, 0.03]
            overall_integrity = sum(s * w for s, w in zip(scores, weights))
            tampering_probability = 1 - overall_integrity
            
            result['tampering_probability'] = round(tampering_probability * 100, 2)
            result['tampering_detected'] = tampering_probability >= self.SUSPICIOUS_THRESHOLD
            
            # Risk Level Assessment
            if tampering_probability >= 0.8:
                result['risk_level'] = 'CRITICAL'
            elif tampering_probability >= 0.65:
                result['risk_level'] = 'HIGH'
            elif tampering_probability >= 0.4:
                result['risk_level'] = 'MEDIUM'
            else:
                result['risk_level'] = 'LOW'
                
            # Generate detailed report
            result['detailed_report'] = self._generate_report(result)
            
        except Exception as e:
            result['detailed_report'] = f"Analysis error: {str(e)}"
            
        return result
    
    def _analyze_feature_boundaries(self, image: np.ndarray, face_box: Dict, 
                                     landmarks: Dict = None) -> Tuple[float, List[str]]:
        """
        Analyze boundaries around facial features for unnatural transitions.
        Transplanted features often have visible seams or unnatural edges.
        """
        anomalies = []
        anomaly_score = 0.0
        
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Define regions to check (approximate if no landmarks)
            x, y, w, h = face_box.get('x', 0), face_box.get('y', 0), face_box.get('w', 100), face_box.get('h', 100)
            
            # Nose region (center of face)
            nose_x = x + w // 3
            nose_y = y + h // 3
            nose_w = w // 3
            nose_h = h // 3
            
            # Extract nose region
            nose_region = gray[nose_y:nose_y+nose_h, nose_x:nose_x+nose_w]
            
            if nose_region.size > 0:
                # Check for unnatural edges around nose
                edges = cv2.Canny(nose_region, 50, 150)
                edge_density = np.sum(edges > 0) / edges.size
                
                # Too many sharp edges inside nose area = possible transplant boundary
                if edge_density > 0.15:
                    anomalies.append("Nose region: Unusual edge density detected (possible splicing)")
                    anomaly_score += 0.3
                    
            # Check eye regions
            eye_y = y + h // 4
            eye_h = h // 4
            
            # Left eye
            left_eye_region = gray[eye_y:eye_y+eye_h, x:x+w//2]
            # Right eye
            right_eye_region = gray[eye_y:eye_y+eye_h, x+w//2:x+w]
            
            if left_eye_region.size > 0 and right_eye_region.size > 0:
                # Check for asymmetric edge patterns (different source images)
                left_edges = cv2.Canny(left_eye_region, 50, 150)
                right_edges = cv2.Canny(right_eye_region, 50, 150)
                
                left_density = np.sum(left_edges > 0) / left_edges.size
                right_density = np.sum(right_edges > 0) / right_edges.size
                
                # Eyes should have similar edge characteristics
                if abs(left_density - right_density) > 0.1:
                    anomalies.append("Eye regions: Asymmetric edge patterns (possible feature swap)")
                    anomaly_score += 0.25
                    
            # Check mouth region
            mouth_y = y + 2 * h // 3
            mouth_h = h // 3
            mouth_region = gray[mouth_y:mouth_y+mouth_h, nose_x:nose_x+nose_w]
            
            if mouth_region.size > 0:
                # Check for rectangular boundaries (copy-paste indicator)
                edges = cv2.Canny(mouth_region, 30, 100)
                lines = cv2.HoughLinesP(edges, 1, np.pi/180, 20, minLineLength=15, maxLineGap=5)
                
                if lines is not None and len(lines) > 10:
                    # Too many straight lines around mouth = possible paste
                    anomalies.append("Mouth region: Excessive linear patterns (possible copy-paste)")
                    anomaly_score += 0.2
                    
        except Exception as e:
            anomalies.append(f"Boundary analysis partial: {str(e)}")
            
        return min(anomaly_score, 1.0), anomalies
    
    def _check_lighting_consistency(self, image: np.ndarray, face_box: Dict,
                                     landmarks: Dict = None) -> Tuple[float, List[str]]:
        """
        Check if lighting/shadows are consistent across facial features.
        Transplanted features often have mismatched lighting directions.
        """
        issues = []
        consistency_score = 1.0
        
        try:
            # Convert to LAB for better lighting analysis
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l_channel = lab[:, :, 0]
            
            x, y, w, h = face_box.get('x', 0), face_box.get('y', 0), face_box.get('w', 100), face_box.get('h', 100)
            
            # Divide face into quadrants
            face_region = l_channel[y:y+h, x:x+w]
            
            if face_region.size == 0:
                return consistency_score, issues
                
            mid_x = w // 2
            mid_y = h // 2
            
            # Quadrant luminance
            q1 = face_region[:mid_y, :mid_x]  # Top-left
            q2 = face_region[:mid_y, mid_x:]  # Top-right
            q3 = face_region[mid_y:, :mid_x]  # Bottom-left
            q4 = face_region[mid_y:, mid_x:]  # Bottom-right
            
            means = []
            for q in [q1, q2, q3, q4]:
                if q.size > 0:
                    means.append(np.mean(q))
                    
            if len(means) >= 4:
                # Check lighting gradient consistency
                # Natural lighting has smooth gradients
                left_right_diff = abs(means[0] - means[1]) + abs(means[2] - means[3])
                top_bottom_diff = abs(means[0] - means[2]) + abs(means[1] - means[3])
                
                # Diagonal check (should be consistent)
                diag1 = abs(means[0] - means[3])
                diag2 = abs(means[1] - means[2])
                
                if abs(diag1 - diag2) > 30:
                    issues.append("Lighting gradient: Inconsistent diagonal illumination")
                    consistency_score -= 0.3
                    
                # Check for abrupt lighting changes (sign of editing)
                if left_right_diff > 50 or top_bottom_diff > 50:
                    issues.append("Lighting: Abrupt luminance transitions detected")
                    consistency_score -= 0.2
                    
            # Shadow direction analysis
            # Compute gradient to find shadow direction
            grad_x = cv2.Sobel(face_region, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(face_region, cv2.CV_64F, 0, 1, ksize=3)
            
            # Shadow should have consistent direction across face
            # High variance in gradient direction = inconsistent lighting
            angles = np.arctan2(grad_y, grad_x)
            angle_var = np.var(angles)
            
            if angle_var > 2.0:  # High variance threshold
                issues.append("Shadow direction: Inconsistent across facial regions")
                consistency_score -= 0.2
                
        except Exception as e:
            issues.append(f"Lighting analysis partial: {str(e)}")
            
        return max(consistency_score, 0.0), issues
    
    def _analyze_texture_continuity(self, image: np.ndarray, face_box: Dict) -> Tuple[float, List[str]]:
        """
        Analyze skin texture continuity.
        Pasted features often have different texture/noise characteristics.
        """
        anomalies = []
        continuity_score = 1.0
        
        try:
            x, y, w, h = face_box.get('x', 0), face_box.get('y', 0), face_box.get('w', 100), face_box.get('h', 100)
            
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            face_gray = gray[y:y+h, x:x+w]
            
            if face_gray.size == 0:
                return continuity_score, anomalies
            
            # Compute Local Binary Pattern (LBP) for texture
            # Simplified LBP using Laplacian variance
            regions = []
            region_h = h // 3
            region_w = w // 3
            
            for i in range(3):
                for j in range(3):
                    ry = i * region_h
                    rx = j * region_w
                    region = face_gray[ry:ry+region_h, rx:rx+region_w]
                    if region.size > 0:
                        # Texture measure: Laplacian variance
                        lap_var = cv2.Laplacian(region, cv2.CV_64F).var()
                        regions.append(lap_var)
                        
            if len(regions) >= 9:
                # All regions should have similar texture
                mean_texture = np.mean(regions)
                std_texture = np.std(regions)
                
                # High standard deviation = inconsistent texture
                cv_texture = std_texture / mean_texture if mean_texture > 0 else 0
                
                if cv_texture > 0.5:
                    anomalies.append("Texture: High variability across facial regions")
                    continuity_score -= 0.3
                    
                # Check for outlier regions (potential paste)
                for i, r in enumerate(regions):
                    if mean_texture > 0 and abs(r - mean_texture) / mean_texture > 0.8:
                        region_name = ['TL', 'TM', 'TR', 'ML', 'MM', 'MR', 'BL', 'BM', 'BR'][i]
                        anomalies.append(f"Texture outlier in region {region_name}")
                        continuity_score -= 0.1
                        
        except Exception as e:
            anomalies.append(f"Texture analysis partial: {str(e)}")
            
        return max(continuity_score, 0.0), anomalies
    
    def _analyze_color_consistency(self, image: np.ndarray, face_box: Dict,
                                    landmarks: Dict = None) -> Tuple[float, List[str]]:
        """
        Analyze skin color consistency across facial regions.
        Pasted features may have different color temperatures.
        """
        issues = []
        consistency_score = 1.0
        
        try:
            x, y, w, h = face_box.get('x', 0), face_box.get('y', 0), face_box.get('w', 100), face_box.get('h', 100)
            
            # Convert to HSV for better skin analysis
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            face_hsv = hsv[y:y+h, x:x+w]
            
            if face_hsv.size == 0:
                return consistency_score, issues
                
            # Sample different facial regions
            # Forehead, cheeks, nose, chin
            regions_hue = []
            
            # Forehead
            forehead = face_hsv[:h//4, w//4:3*w//4]
            if forehead.size > 0:
                regions_hue.append(np.median(forehead[:, :, 0]))
                
            # Left cheek
            left_cheek = face_hsv[h//3:2*h//3, :w//3]
            if left_cheek.size > 0:
                regions_hue.append(np.median(left_cheek[:, :, 0]))
                
            # Right cheek
            right_cheek = face_hsv[h//3:2*h//3, 2*w//3:]
            if right_cheek.size > 0:
                regions_hue.append(np.median(right_cheek[:, :, 0]))
                
            # Nose
            nose = face_hsv[h//3:2*h//3, w//3:2*w//3]
            if nose.size > 0:
                regions_hue.append(np.median(nose[:, :, 0]))
                
            if len(regions_hue) >= 3:
                # All skin regions should have similar hue
                hue_std = np.std(regions_hue)
                
                if hue_std > 15:  # Significant hue variation
                    issues.append("Skin tone: Inconsistent hue across facial regions")
                    consistency_score -= 0.3
                    
                # Check for dramatic differences between adjacent regions
                if len(regions_hue) >= 4:
                    cheek_diff = abs(regions_hue[1] - regions_hue[2])  # Left vs Right cheek
                    if cheek_diff > 20:
                        issues.append("Color mismatch: Left and right cheeks have different tones")
                        consistency_score -= 0.2
                        
        except Exception as e:
            issues.append(f"Color analysis partial: {str(e)}")
            
        return max(consistency_score, 0.0), issues
    
    def _analyze_noise_patterns(self, image: np.ndarray, face_box: Dict) -> Tuple[float, List[str]]:
        """
        Analyze camera noise patterns.
        Different source images have different noise signatures.
        """
        anomalies = []
        consistency_score = 1.0
        
        try:
            x, y, w, h = face_box.get('x', 0), face_box.get('y', 0), face_box.get('w', 100), face_box.get('h', 100)
            
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float64)
            face_gray = gray[y:y+h, x:x+w]
            
            if face_gray.size == 0:
                return consistency_score, anomalies
                
            # Extract noise using high-pass filter
            blurred = cv2.GaussianBlur(face_gray, (5, 5), 0)
            noise = face_gray - blurred
            
            # Analyze noise in different regions
            noise_vars = []
            region_h = h // 3
            region_w = w // 3
            
            for i in range(3):
                for j in range(3):
                    ry = i * region_h
                    rx = j * region_w
                    region_noise = noise[ry:ry+region_h, rx:rx+region_w]
                    if region_noise.size > 0:
                        noise_vars.append(np.var(region_noise))
                        
            if len(noise_vars) >= 9:
                # Noise should be uniform across face from same camera
                noise_cv = np.std(noise_vars) / np.mean(noise_vars) if np.mean(noise_vars) > 0 else 0
                
                if noise_cv > 0.6:
                    anomalies.append("Noise pattern: Inconsistent across regions (possible multi-source)")
                    consistency_score -= 0.3
                    
        except Exception as e:
            anomalies.append(f"Noise analysis partial: {str(e)}")
            
        return max(consistency_score, 0.0), anomalies
    
    def _detect_jpeg_ghosts(self, image: np.ndarray) -> Tuple[float, List[str]]:
        """
        Detect JPEG ghosts - artifacts from multiple compressions.
        Edited regions are often saved multiple times at different quality levels.
        """
        ghost_regions = []
        ghost_score = 0.0
        
        try:
            # Resave at different quality levels and check for inconsistencies
            quality_levels = [60, 75, 90]
            differences = []
            
            for quality in quality_levels:
                # Encode and decode
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
                _, encoded = cv2.imencode('.jpg', image, encode_param)
                decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
                
                if decoded is not None:
                    diff = cv2.absdiff(image, decoded)
                    diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
                    differences.append(diff_gray)
                    
            if len(differences) >= 2:
                # Look for regions that behave differently at different quality levels
                # Edited regions show up as "ghosts"
                combined_diff = np.zeros_like(differences[0], dtype=np.float64)
                
                for diff in differences:
                    combined_diff += diff.astype(np.float64)
                    
                combined_diff /= len(differences)
                
                # Threshold to find anomalous regions
                _, thresh = cv2.threshold(combined_diff.astype(np.uint8), 30, 255, cv2.THRESH_BINARY)
                
                # Find contours of ghost regions
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                significant_ghosts = [c for c in contours if cv2.contourArea(c) > 500]
                
                if significant_ghosts:
                    ghost_score = min(len(significant_ghosts) * 0.15, 1.0)
                    ghost_regions.append(f"Detected {len(significant_ghosts)} potential edited regions")
                    
        except Exception as e:
            ghost_regions.append(f"JPEG ghost detection partial: {str(e)}")
            
        return ghost_score, ghost_regions
    
    def _detect_edge_discontinuities(self, image: np.ndarray, face_box: Dict,
                                      landmarks: Dict = None) -> Tuple[float, List[str]]:
        """
        Detect edge discontinuities that indicate splicing.
        """
        issues = []
        continuity_score = 1.0
        
        try:
            x, y, w, h = face_box.get('x', 0), face_box.get('y', 0), face_box.get('w', 100), face_box.get('h', 100)
            
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            face_gray = gray[y:y+h, x:x+w]
            
            if face_gray.size == 0:
                return continuity_score, issues
                
            # Detect edges
            edges = cv2.Canny(face_gray, 50, 150)
            
            # Look for unnaturally straight edges (copy-paste boundaries)
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=30, 
                                    minLineLength=w//4, maxLineGap=10)
            
            if lines is not None:
                # Count horizontal and vertical lines (unnatural in face)
                h_lines = 0
                v_lines = 0
                
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    angle = abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
                    
                    if angle < 10 or angle > 170:  # Horizontal
                        h_lines += 1
                    elif 80 < angle < 100:  # Vertical
                        v_lines += 1
                        
                if h_lines > 3 or v_lines > 3:
                    issues.append(f"Unnatural edges: {h_lines} horizontal, {v_lines} vertical lines detected")
                    continuity_score -= 0.3
                    
        except Exception as e:
            issues.append(f"Edge analysis partial: {str(e)}")
            
        return max(continuity_score, 0.0), issues

    def _resolve_local_face_box(self, image: np.ndarray, face_box: Optional[Dict]) -> Dict[str, int]:
        """
        Resolve a usable face box inside the current image. If the provided box is
        out-of-frame, fall back to the full crop because advanced biometrics often
        runs on a pre-cropped face image.
        """
        h, w = image.shape[:2]
        if not face_box:
            return {'x': 0, 'y': 0, 'w': w, 'h': h}

        x = int(face_box.get('x', 0))
        y = int(face_box.get('y', 0))
        bw = int(face_box.get('w', w))
        bh = int(face_box.get('h', h))

        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(w, x1 + max(1, bw))
        y2 = min(h, y1 + max(1, bh))

        if x2 - x1 < max(16, w // 5) or y2 - y1 < max(16, h // 5):
            return {'x': 0, 'y': 0, 'w': w, 'h': h}
        return {'x': x1, 'y': y1, 'w': x2 - x1, 'h': y2 - y1}

    @staticmethod
    def _fractional_box(face_box: Dict[str, int], x0: float, y0: float, x1: float, y1: float) -> Dict[str, int]:
        fx = face_box['x'] + int(round(face_box['w'] * x0))
        fy = face_box['y'] + int(round(face_box['h'] * y0))
        fw = max(4, int(round(face_box['w'] * max(0.01, x1 - x0))))
        fh = max(4, int(round(face_box['h'] * max(0.01, y1 - y0))))
        return {'x': fx, 'y': fy, 'w': fw, 'h': fh}

    @staticmethod
    def _crop_box(image: np.ndarray, box: Dict[str, int]) -> np.ndarray:
        x1 = max(0, box['x'])
        y1 = max(0, box['y'])
        x2 = min(image.shape[1], x1 + max(1, box['w']))
        y2 = min(image.shape[0], y1 + max(1, box['h']))
        if x2 <= x1 or y2 <= y1:
            return np.empty((0, 0), dtype=image.dtype)
        return image[y1:y2, x1:x2]

    @staticmethod
    def _normalized_box(box: Dict[str, int], face_box: Dict[str, int]) -> Dict[str, float]:
        fw = max(face_box['w'], 1)
        fh = max(face_box['h'], 1)
        return {
            'x': round((box['x'] - face_box['x']) / fw, 4),
            'y': round((box['y'] - face_box['y']) / fh, 4),
            'w': round(box['w'] / fw, 4),
            'h': round(box['h'] / fh, 4),
        }

    def _analyze_micro_seam_boundaries(
        self,
        image: np.ndarray,
        face_box: Dict,
        landmarks: Dict = None
    ) -> Dict[str, Any]:
        """
        Scan high-risk seam corridors where a morphed T-zone is commonly stitched
        onto a legitimate head shell. The detector looks for abrupt changes in
        Laplacian energy, residual noise variance, and 8x8 blockiness.
        """
        result = {
            'seam_detected': False,
            'seam_probability': 0.0,
            'edge_zone_scores': {},
            'candidate_regions': [],
            'highlight_box': None,
            'highlight_box_normalized': None,
            'summary': 'No micro-seam anomalies detected'
        }

        try:
            local_face = self._resolve_local_face_box(image, face_box)
            face = self._crop_box(image, local_face)
            if face.size == 0:
                return result

            gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (3, 3), 0)
            lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
            lap_abs = np.abs(lap)
            residual = gray.astype(np.float32) - cv2.GaussianBlur(gray, (7, 7), 0).astype(np.float32)

            zones = {
                'hairline': self._fractional_box(local_face, 0.18, 0.04, 0.82, 0.22),
                'left_cheek_edge': self._fractional_box(local_face, 0.14, 0.28, 0.34, 0.78),
                'right_cheek_edge': self._fractional_box(local_face, 0.66, 0.28, 0.86, 0.78),
                'jawline': self._fractional_box(local_face, 0.22, 0.68, 0.78, 0.92),
            }

            candidate_regions: List[Dict[str, Any]] = []
            zone_scores: Dict[str, float] = {}

            for zone_name, zone_box in zones.items():
                local_zone = {
                    'x': zone_box['x'] - local_face['x'],
                    'y': zone_box['y'] - local_face['y'],
                    'w': zone_box['w'],
                    'h': zone_box['h'],
                }
                zone_img = self._crop_box(gray, local_zone)
                zone_lap = self._crop_box(lap_abs, local_zone)
                zone_residual = self._crop_box(residual, local_zone)
                if zone_img.size == 0 or zone_lap.size == 0 or zone_residual.size == 0:
                    zone_scores[zone_name] = 0.0
                    continue

                vertical_split = zone_name in {'left_cheek_edge', 'right_cheek_edge'}
                if vertical_split:
                    split = max(4, zone_img.shape[1] // 2)
                    region_a = zone_img[:, :split]
                    region_b = zone_img[:, split:]
                    lap_a = zone_lap[:, :split]
                    lap_b = zone_lap[:, split:]
                    noise_a = zone_residual[:, :split]
                    noise_b = zone_residual[:, split:]
                    seam_strip = zone_lap[:, max(0, split - 2):min(zone_lap.shape[1], split + 2)]
                else:
                    split = max(4, zone_img.shape[0] // 2)
                    region_a = zone_img[:split, :]
                    region_b = zone_img[split:, :]
                    lap_a = zone_lap[:split, :]
                    lap_b = zone_lap[split:, :]
                    noise_a = zone_residual[:split, :]
                    noise_b = zone_residual[split:, :]
                    seam_strip = zone_lap[max(0, split - 2):min(zone_lap.shape[0], split + 2), :]

                if region_a.size == 0 or region_b.size == 0:
                    zone_scores[zone_name] = 0.0
                    continue

                lap_shift = abs(np.log1p(float(np.var(lap_a))) - np.log1p(float(np.var(lap_b))))
                noise_shift = abs(np.log1p(float(np.var(noise_a))) - np.log1p(float(np.var(noise_b))))
                edge_density = float(np.mean(seam_strip > np.percentile(zone_lap, 80)))
                blockiness = self._measure_blockiness(zone_img)
                zone_score = min(
                    1.0,
                    0.36 * min(lap_shift / 0.45, 1.0)
                    + 0.29 * min(noise_shift / 0.5, 1.0)
                    + 0.20 * min(edge_density / 0.18, 1.0)
                    + 0.15 * min(blockiness / 12.0, 1.0)
                )
                zone_score = float(round(zone_score, 4))
                zone_scores[zone_name] = zone_score

                if zone_score >= 0.52:
                    candidate = {
                        'zone': zone_name,
                        'confidence': round(zone_score * 100, 2),
                        'box': zone_box,
                        'box_normalized': self._normalized_box(zone_box, local_face),
                        'signals': {
                            'laplacian_shift': round(float(lap_shift), 4),
                            'noise_shift': round(float(noise_shift), 4),
                            'edge_density': round(float(edge_density), 4),
                            'blockiness': round(float(blockiness), 4),
                        }
                    }
                    candidate_regions.append(candidate)

            if zone_scores:
                ordered_scores = sorted(zone_scores.values(), reverse=True)
                dominant = ordered_scores[0]
                support = float(np.mean(ordered_scores[:2])) if len(ordered_scores) > 1 else dominant
                seam_probability = min(1.0, 0.65 * dominant + 0.35 * support)
            else:
                seam_probability = 0.0

            candidate_regions.sort(key=lambda item: item['confidence'], reverse=True)
            highlight_box = candidate_regions[0]['box'] if candidate_regions else None
            highlight_box_normalized = candidate_regions[0]['box_normalized'] if candidate_regions else None

            if seam_probability >= 0.62 and candidate_regions:
                seam_detected = True
                top_zones = ", ".join(region['zone'] for region in candidate_regions[:2])
                summary = f"Microscopic seam signature concentrated along {top_zones}"
            elif candidate_regions:
                seam_detected = False
                summary = "Boundary micro-texture drift detected but below hard seam threshold"
            else:
                seam_detected = False
                summary = "No micro-seam anomalies detected"

            result.update({
                'seam_detected': seam_detected,
                'seam_probability': round(seam_probability, 4),
                'edge_zone_scores': {k: round(v, 4) for k, v in zone_scores.items()},
                'candidate_regions': candidate_regions,
                'highlight_box': highlight_box,
                'highlight_box_normalized': highlight_box_normalized,
                'summary': summary,
            })
        except Exception as e:
            result['summary'] = f"Micro-seam analysis partial: {str(e)}"

        return result

    @staticmethod
    def _measure_blockiness(region: np.ndarray) -> float:
        """
        Approximate JPEG blocking energy by measuring step discontinuities on 8-pixel
        boundaries relative to the local gradient floor.
        """
        if region.size == 0:
            return 0.0
        region = region.astype(np.float32)
        vertical = []
        horizontal = []
        for idx in range(8, region.shape[1], 8):
            vertical.append(float(np.mean(np.abs(region[:, idx] - region[:, idx - 1]))))
        for idx in range(8, region.shape[0], 8):
            horizontal.append(float(np.mean(np.abs(region[idx, :] - region[idx - 1, :]))))

        boundary_energy = 0.0
        if vertical:
            boundary_energy += float(np.mean(vertical))
        if horizontal:
            boundary_energy += float(np.mean(horizontal))

        grad_x = np.mean(np.abs(np.diff(region, axis=1))) if region.shape[1] > 1 else 0.0
        grad_y = np.mean(np.abs(np.diff(region, axis=0))) if region.shape[0] > 1 else 0.0
        baseline = max(1.0, float(grad_x + grad_y))
        return boundary_energy / baseline
    
    def _generate_report(self, result: Dict) -> str:
        """Generate a human-readable tampering analysis report."""
        lines = []
        lines.append("=" * 50)
        lines.append("FACIAL TAMPERING ANALYSIS REPORT")
        lines.append("=" * 50)
        lines.append(f"Risk Level: {result['risk_level']}")
        lines.append(f"Tampering Probability: {result['tampering_probability']}%")
        lines.append(f"Detection Status: {'[WARNING] TAMPERING DETECTED' if result['tampering_detected'] else '[OK] No tampering detected'}")
        lines.append("")
        
        if result['boundary_anomalies']:
            lines.append("BOUNDARY ANOMALIES:")
            for a in result['boundary_anomalies']:
                lines.append(f"  • {a}")
                
        if result['lighting_inconsistencies']:
            lines.append("LIGHTING ISSUES:")
            for l in result['lighting_inconsistencies']:
                lines.append(f"  • {l}")
                
        if result['texture_anomalies']:
            lines.append("TEXTURE ANOMALIES:")
            for t in result['texture_anomalies']:
                lines.append(f"  • {t}")
                
        if result['splicing_indicators']:
            lines.append("SPLICING INDICATORS:")
            for s in result['splicing_indicators']:
                lines.append(f"  • {s}")
                
        lines.append("=" * 50)
        return "\n".join(lines)
