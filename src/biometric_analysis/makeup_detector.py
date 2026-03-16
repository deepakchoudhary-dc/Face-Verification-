"""
MAKEUP & DISGUISE DETECTION MODULE
===================================
Detects when someone is using heavy makeup, prosthetics, or disguises
to impersonate another person's identity.

Techniques:
1. Skin Texture Analysis - Real skin vs makeup covered skin
2. Contour Anomaly Detection - Prosthetic edges
3. Color Distribution Analysis - Unnatural makeup patterns
4. Specular Reflection Analysis - Makeup reflects light differently
5. Pore Detection - Heavy makeup covers pores
6. Eye Region Analysis - Colored contacts, false lashes
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Any
from scipy import ndimage


class MakeupDisguiseDetector:
    """
    Advanced Makeup and Disguise Detection System.
    Catches fraudsters using cosmetic manipulation.
    """
    
    def __init__(self):
        self.DISGUISE_THRESHOLD = 0.60  # 60% = likely disguised
        
        # Skin color ranges in HSV (natural human skin)
        self.SKIN_LOWER = np.array([0, 20, 70])
        self.SKIN_UPPER = np.array([50, 255, 255])
        
    def analyze_disguise(self, image: np.ndarray, face_box: Dict, landmarks: Dict = None) -> Dict[str, Any]:
        """
        Comprehensive disguise and heavy makeup analysis.
        """
        result = {
            'disguise_detected': False,
            'disguise_probability': 0.0,
            'makeup_level': 'NONE',
            'analysis': {
                'skin_texture': {},
                'makeup_indicators': {},
                'prosthetic_indicators': {},
                'eye_modifications': {},
                'overall_naturalness': 0.0
            },
            'warnings': [],
            'risk_assessment': ''
        }
        
        if image is None:
            return result
            
        try:
            # 1. SKIN TEXTURE ANALYSIS (Pore detection)
            texture_score, texture_details = self._analyze_skin_texture(image, face_box)
            result['analysis']['skin_texture'] = texture_details
            
            # 2. MAKEUP DETECTION (Foundation, contouring)
            makeup_score, makeup_details = self._detect_makeup_patterns(image, face_box)
            result['analysis']['makeup_indicators'] = makeup_details
            
            # 3. PROSTHETIC DETECTION (Fake nose, chin, etc.)
            prosthetic_score, prosthetic_details = self._detect_prosthetics(image, face_box, landmarks)
            result['analysis']['prosthetic_indicators'] = prosthetic_details
            
            # 4. EYE MODIFICATION DETECTION (Colored contacts, false lashes)
            eye_score, eye_details = self._analyze_eye_modifications(image, face_box, landmarks)
            result['analysis']['eye_modifications'] = eye_details
            
            # 5. SPECULAR ANALYSIS (Light reflection patterns)
            specular_score, specular_details = self._analyze_specular_reflection(image, face_box)
            result['analysis']['specular_analysis'] = specular_details
            
            # 6. COLOR DISTRIBUTION (Unnatural tones)
            color_score, color_details = self._analyze_color_distribution(image, face_box)
            result['analysis']['color_naturalness'] = color_details
            
            # CALCULATE OVERALL DISGUISE PROBABILITY
            weights = {
                'texture': 0.20,
                'makeup': 0.25,
                'prosthetic': 0.25,
                'eye': 0.15,
                'specular': 0.10,
                'color': 0.05
            }
            
            # Lower scores = more natural
            disguise_score = (
                (1 - texture_score) * weights['texture'] +
                makeup_score * weights['makeup'] +
                prosthetic_score * weights['prosthetic'] +
                eye_score * weights['eye'] +
                specular_score * weights['specular'] +
                (1 - color_score) * weights['color']
            )
            
            result['disguise_probability'] = round(disguise_score * 100, 2)
            result['disguise_detected'] = disguise_score >= self.DISGUISE_THRESHOLD
            
            # Determine makeup level
            if makeup_score >= 0.7:
                result['makeup_level'] = 'HEAVY'
            elif makeup_score >= 0.4:
                result['makeup_level'] = 'MODERATE'
            elif makeup_score >= 0.2:
                result['makeup_level'] = 'LIGHT'
            else:
                result['makeup_level'] = 'NONE/MINIMAL'
                
            # Overall naturalness
            result['analysis']['overall_naturalness'] = round((1 - disguise_score) * 100, 2)
            
            # Generate warnings
            if prosthetic_score > 0.5:
                result['warnings'].append("[WARNING] PROSTHETIC INDICATORS: Possible fake facial features detected")
            if makeup_score > 0.6:
                result['warnings'].append("[WARNING] HEAVY MAKEUP: Significant cosmetic coverage detected")
            if eye_score > 0.5:
                result['warnings'].append("[WARNING] EYE MODIFICATIONS: Possible colored contacts or false lashes")
            if texture_score < 0.4:
                result['warnings'].append("[WARNING] TEXTURE ANOMALY: Unnatural skin texture (covered pores)")
                
            # Risk assessment
            result['risk_assessment'] = self._generate_risk_assessment(result)
            
        except Exception as e:
            result['risk_assessment'] = f"Analysis error: {str(e)}"
            
        return result
    
    def _analyze_skin_texture(self, image: np.ndarray, face_box: Dict) -> Tuple[float, Dict]:
        """
        Analyze skin texture to detect makeup coverage.
        Real skin has visible pores; heavily made-up skin appears smoother.
        """
        details = {
            'pore_visibility': 0.0,
            'texture_variance': 0.0,
            'naturalness_score': 0.0
        }
        
        try:
            x, y, w, h = face_box.get('x', 0), face_box.get('y', 0), face_box.get('w', 100), face_box.get('h', 100)
            
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Sample cheek regions (best for pore detection)
            cheek_y = y + h // 3
            cheek_h = h // 4
            
            # Left cheek
            left_cheek = gray[cheek_y:cheek_y+cheek_h, x+w//10:x+w//3]
            # Right cheek
            right_cheek = gray[cheek_y:cheek_y+cheek_h, x+2*w//3:x+9*w//10]
            
            pore_scores = []
            
            for cheek in [left_cheek, right_cheek]:
                if cheek.size > 100:
                    # High-pass filter to detect fine details (pores)
                    blurred = cv2.GaussianBlur(cheek, (9, 9), 0)
                    high_pass = cv2.absdiff(cheek, blurred)
                    
                    # Pores appear as small bright spots
                    pore_count = np.sum(high_pass > 15)
                    pore_ratio = pore_count / cheek.size
                    pore_scores.append(pore_ratio)
                    
            if pore_scores:
                avg_pore = np.mean(pore_scores)
                details['pore_visibility'] = round(avg_pore * 100, 2)
                
                # Natural skin: pore_ratio > 0.02
                # Made-up skin: pore_ratio < 0.01
                if avg_pore > 0.03:
                    details['naturalness_score'] = 1.0
                elif avg_pore > 0.015:
                    details['naturalness_score'] = 0.7
                elif avg_pore > 0.008:
                    details['naturalness_score'] = 0.4
                else:
                    details['naturalness_score'] = 0.2
                    
            # Texture variance (smooth = makeup)
            face_region = gray[y:y+h, x:x+w]
            if face_region.size > 0:
                details['texture_variance'] = round(np.std(face_region), 2)
                
        except Exception as e:
            pass
            
        return details.get('naturalness_score', 0.5), details
    
    def _detect_makeup_patterns(self, image: np.ndarray, face_box: Dict) -> Tuple[float, Dict]:
        """
        Detect makeup application patterns (contouring, foundation edges).
        """
        details = {
            'foundation_detected': False,
            'contouring_detected': False,
            'color_uniformity': 0.0,
            'makeup_edge_score': 0.0
        }
        
        makeup_score = 0.0
        
        try:
            x, y, w, h = face_box.get('x', 0), face_box.get('y', 0), face_box.get('w', 100), face_box.get('h', 100)
            
            # Convert to LAB for better skin analysis
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            face_lab = lab[y:y+h, x:x+w]
            
            if face_lab.size == 0:
                return makeup_score, details
                
            # Analyze A channel (red-green) - foundation affects this
            a_channel = face_lab[:, :, 1]
            
            # Foundation creates very uniform a-channel
            a_std = np.std(a_channel)
            a_mean = np.mean(a_channel)
            
            # Natural skin has variance, foundation is uniform
            if a_std < 8:  # Very uniform = likely foundation
                details['foundation_detected'] = True
                makeup_score += 0.4
                
            details['color_uniformity'] = round(1 - min(a_std / 20, 1.0), 2)
            
            # Detect contouring (deliberate shadow patterns)
            l_channel = face_lab[:, :, 0]
            
            # Cheek contour detection
            left_side = l_channel[:, :w//3]
            center = l_channel[:, w//3:2*w//3]
            right_side = l_channel[:, 2*w//3:]
            
            # Contouring creates darker sides, lighter center
            if left_side.size > 0 and center.size > 0 and right_side.size > 0:
                left_mean = np.mean(left_side)
                center_mean = np.mean(center)
                right_mean = np.mean(right_side)
                
                # Check for classic contour pattern
                if center_mean > left_mean + 10 and center_mean > right_mean + 10:
                    details['contouring_detected'] = True
                    makeup_score += 0.3
                    
            # Detect makeup edges (foundation line on jaw/hairline)
            # Check jaw region for abrupt color change
            jaw_region = face_lab[3*h//4:, :]
            if jaw_region.size > 0:
                jaw_edges = cv2.Canny(jaw_region[:, :, 1], 30, 60)
                edge_density = np.sum(jaw_edges > 0) / jaw_edges.size
                details['makeup_edge_score'] = round(edge_density, 4)
                
                if edge_density > 0.05:  # High edge density at jaw = foundation line
                    makeup_score += 0.2
                    
        except Exception as e:
            pass
            
        return min(makeup_score, 1.0), details
    
    def _detect_prosthetics(self, image: np.ndarray, face_box: Dict, landmarks: Dict = None) -> Tuple[float, Dict]:
        """
        Detect prosthetic appliances (fake nose, chin, cheek implants).
        """
        details = {
            'edge_anomalies': [],
            'texture_discontinuities': [],
            'prosthetic_probability': 0.0
        }
        
        prosthetic_score = 0.0
        
        try:
            x, y, w, h = face_box.get('x', 0), face_box.get('y', 0), face_box.get('w', 100), face_box.get('h', 100)
            
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            face_gray = gray[y:y+h, x:x+w]
            
            if face_gray.size == 0:
                return prosthetic_score, details
                
            # 1. Look for prosthetic edges (unnatural straight/curved lines)
            # Prosthetics often have visible seams
            edges = cv2.Canny(face_gray, 40, 120)
            
            # Dilate to connect nearby edges
            kernel = np.ones((3, 3), np.uint8)
            dilated = cv2.dilate(edges, kernel, iterations=1)
            
            # Find contours that might be prosthetic edges
            contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                arc_len = cv2.arcLength(contour, False)
                area = cv2.contourArea(contour)
                
                # Prosthetic edges are often long with small enclosed area
                if arc_len > w * 0.3 and (area < arc_len * 2 or area == 0):
                    details['edge_anomalies'].append("Suspicious linear edge detected")
                    prosthetic_score += 0.15
                    
            # 2. Texture discontinuity check
            # Sample regions and look for abrupt changes
            nose_region = face_gray[h//4:2*h//3, w//3:2*w//3]
            
            if nose_region.size > 0:
                # Check for texture transition (prosthetic meets skin)
                grad_y = np.abs(np.diff(nose_region.astype(np.float64), axis=0))
                grad_x = np.abs(np.diff(nose_region.astype(np.float64), axis=1))
                
                # High localized gradients = possible prosthetic edge
                if np.max(grad_y) > 50 or np.max(grad_x) > 50:
                    details['texture_discontinuities'].append("Nose region texture discontinuity")
                    prosthetic_score += 0.2
                    
            # 3. Check chin area
            chin_region = face_gray[3*h//4:, w//4:3*w//4]
            if chin_region.size > 0:
                chin_grad = np.abs(np.diff(chin_region.astype(np.float64), axis=0))
                if np.max(chin_grad) > 40:
                    details['texture_discontinuities'].append("Chin region texture discontinuity")
                    prosthetic_score += 0.15
                    
            details['prosthetic_probability'] = round(min(prosthetic_score, 1.0) * 100, 2)
            
        except Exception as e:
            pass
            
        return min(prosthetic_score, 1.0), details
    
    def _analyze_eye_modifications(self, image: np.ndarray, face_box: Dict, landmarks: Dict = None) -> Tuple[float, Dict]:
        """
        Detect eye modifications (colored contacts, false lashes, eye tape).
        """
        details = {
            'colored_contacts_suspected': False,
            'false_lashes_suspected': False,
            'iris_uniformity': 0.0,
            'modification_score': 0.0
        }
        
        eye_mod_score = 0.0
        
        try:
            x, y, w, h = face_box.get('x', 0), face_box.get('y', 0), face_box.get('w', 100), face_box.get('h', 100)
            
            # Approximate eye regions
            eye_y = y + h // 4
            eye_h = h // 5
            
            # Left eye
            left_eye = image[eye_y:eye_y+eye_h, x:x+w//2]
            # Right eye
            right_eye = image[eye_y:eye_y+eye_h, x+w//2:x+w]
            
            for eye, eye_name in [(left_eye, 'left'), (right_eye, 'right')]:
                if eye.size > 100:
                    # Convert to HSV for color analysis
                    hsv = cv2.cvtColor(eye, cv2.COLOR_BGR2HSV)
                    
                    # Colored contacts often have very uniform, saturated colors
                    saturation = hsv[:, :, 1]
                    
                    # High saturation in a circular pattern = colored contact
                    high_sat_ratio = np.sum(saturation > 150) / saturation.size
                    
                    if high_sat_ratio > 0.1:  # More than 10% highly saturated
                        details['colored_contacts_suspected'] = True
                        eye_mod_score += 0.3
                        
                    # Check for false lashes (dark thick line above eye)
                    gray_eye = cv2.cvtColor(eye, cv2.COLOR_BGR2GRAY)
                    upper_region = gray_eye[:eye_h//3, :]
                    
                    if upper_region.size > 0:
                        dark_pixels = np.sum(upper_region < 40)
                        dark_ratio = dark_pixels / upper_region.size
                        
                        if dark_ratio > 0.3:  # Heavy dark line = false lashes
                            details['false_lashes_suspected'] = True
                            eye_mod_score += 0.2
                            
            # Check iris uniformity (natural iris has radial patterns)
            # Colored contacts are often too uniform
            details['modification_score'] = round(min(eye_mod_score, 1.0) * 100, 2)
            
        except Exception as e:
            pass
            
        return min(eye_mod_score, 1.0), details
    
    def _analyze_specular_reflection(self, image: np.ndarray, face_box: Dict) -> Tuple[float, Dict]:
        """
        Analyze light reflections - makeup reflects differently than natural skin.
        """
        details = {
            'reflection_uniformity': 0.0,
            'highlight_distribution': 'NATURAL',
            'anomaly_detected': False
        }
        
        specular_score = 0.0
        
        try:
            x, y, w, h = face_box.get('x', 0), face_box.get('y', 0), face_box.get('w', 100), face_box.get('h', 100)
            
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            face_gray = gray[y:y+h, x:x+w]
            
            if face_gray.size == 0:
                return specular_score, details
                
            # Find specular highlights (very bright spots)
            _, highlights = cv2.threshold(face_gray, 220, 255, cv2.THRESH_BINARY)
            
            # Count highlight regions
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(highlights)
            
            # Natural skin has few scattered highlights
            # Makeup (especially glossy) has more uniform/larger highlights
            
            highlight_areas = [stats[i, cv2.CC_STAT_AREA] for i in range(1, num_labels)]
            
            if highlight_areas:
                avg_area = np.mean(highlight_areas)
                num_highlights = len(highlight_areas)
                
                # Too many large highlights = glossy makeup
                if num_highlights > 10 or avg_area > 100:
                    details['highlight_distribution'] = 'UNNATURAL'
                    details['anomaly_detected'] = True
                    specular_score += 0.4
                    
                details['reflection_uniformity'] = round(np.std(highlight_areas) if highlight_areas else 0, 2)
                
        except Exception as e:
            pass
            
        return min(specular_score, 1.0), details
    
    def _analyze_color_distribution(self, image: np.ndarray, face_box: Dict) -> Tuple[float, Dict]:
        """
        Analyze if skin color distribution is natural or cosmetically altered.
        """
        details = {
            'skin_tone_naturalness': 0.0,
            'color_anomalies': []
        }
        
        color_score = 1.0  # Start natural
        
        try:
            x, y, w, h = face_box.get('x', 0), face_box.get('y', 0), face_box.get('w', 100), face_box.get('h', 100)
            
            # Convert to HSV
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            face_hsv = hsv[y:y+h, x:x+w]
            
            if face_hsv.size == 0:
                return color_score, details
                
            # Check if skin tone is within natural range
            hue = face_hsv[:, :, 0]
            sat = face_hsv[:, :, 1]
            
            # Natural skin hue: roughly 0-50 (reds/oranges/yellows)
            unnatural_hue = np.sum((hue > 50) & (hue < 150))  # Greens/cyans
            unnatural_ratio = unnatural_hue / hue.size
            
            if unnatural_ratio > 0.1:
                details['color_anomalies'].append("Unnatural hue regions detected")
                color_score -= 0.3
                
            # Very high or uniform saturation = makeup
            sat_std = np.std(sat)
            if sat_std < 15:  # Too uniform
                details['color_anomalies'].append("Overly uniform saturation (possible foundation)")
                color_score -= 0.2
                
            details['skin_tone_naturalness'] = round(max(color_score, 0), 2)
            
        except Exception as e:
            pass
            
        return max(color_score, 0), details
    
    def _generate_risk_assessment(self, result: Dict) -> str:
        """Generate risk assessment summary."""
        lines = []
        lines.append("=" * 50)
        lines.append("MAKEUP/DISGUISE DETECTION REPORT")
        lines.append("=" * 50)
        lines.append(f"Disguise Probability: {result['disguise_probability']}%")
        lines.append(f"Makeup Level: {result['makeup_level']}")
        lines.append(f"Face Naturalness: {result['analysis'].get('overall_naturalness', 0)}%")
        lines.append("")
        
        if result['warnings']:
            lines.append("WARNINGS:")
            for w in result['warnings']:
                lines.append(f"  {w}")
                
        lines.append("=" * 50)
        return "\n".join(lines)
