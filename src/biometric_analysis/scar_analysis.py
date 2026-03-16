"""
SCAR, WRINKLE, AND INJURY ANALYSIS MODULE
==========================================
Permanent facial markers that serve as unique biometric identifiers.

This module detects and maps:
1. Scars - Surgical scars, injury scars, acne scars
2. Wrinkles - Expression lines, age-related creases
3. Birthmarks - Port wine stains, moles, pigmentation patterns
4. Facial injuries - Burns, reconstructive surgery signs
5. Permanent deformities - Broken nose, cauliflower ear, asymmetries

These markers are:
- Extremely difficult to fake or replicate
- Persist across age and weight changes
- Unique identifiers even for identical twins
- Cannot be easily hidden with makeup

This is CRITICAL for:
- Long-term identity verification
- Criminal identification
- Fraud detection where subject has had facial surgery
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Any
from enum import Enum


class MarkerType(Enum):
    SCAR = "SCAR"
    WRINKLE = "WRINKLE"
    BIRTHMARK = "BIRTHMARK"
    INJURY = "INJURY"
    SURGERY_SIGN = "SURGERY_SIGN"
    PIGMENTATION = "PIGMENTATION"


class ScarAndInjuryAnalyzer:
    """
    Comprehensive facial marker detection and analysis.
    Maps permanent facial features as unique identifiers.
    """
    
    def __init__(self):
        # Regions of face for detailed analysis
        self.FACIAL_REGIONS = [
            'forehead', 'left_eye', 'right_eye', 'nose_bridge',
            'left_cheek', 'right_cheek', 'nose_tip', 'upper_lip',
            'lower_lip', 'chin', 'left_jaw', 'right_jaw'
        ]
        
    def analyze_facial_markers(self, image: np.ndarray, face_box: Dict, 
                                landmarks: Dict = None) -> Dict[str, Any]:
        """
        Comprehensive analysis of permanent facial markers.
        """
        result = {
            'markers_detected': 0,
            'marker_map': [],
            'scar_analysis': {},
            'wrinkle_map': {},
            'birthmark_detection': {},
            'injury_signs': [],
            'surgery_indicators': [],
            'unique_marker_signature': '',
            'marker_confidence': 0.0
        }
        
        if image is None:
            return result
            
        try:
            # 1. SCAR DETECTION
            scars = self._detect_scars(image, face_box)
            result['scar_analysis'] = scars
            
            # 2. WRINKLE MAPPING
            wrinkles = self._map_wrinkles(image, face_box, landmarks)
            result['wrinkle_map'] = wrinkles
            
            # 3. BIRTHMARK/PIGMENTATION DETECTION
            birthmarks = self._detect_birthmarks(image, face_box)
            result['birthmark_detection'] = birthmarks
            
            # 4. INJURY SIGNS
            injuries = self._detect_injury_signs(image, face_box)
            result['injury_signs'] = injuries
            
            # 5. SURGICAL SIGNS
            surgery = self._detect_surgery_signs(image, face_box, landmarks)
            result['surgery_indicators'] = surgery
            
            # COMPILE MARKER MAP
            marker_map = []
            
            # Add scars to map
            for scar in scars.get('scars', []):
                marker_map.append({
                    'type': MarkerType.SCAR.value,
                    'location': scar['location'],
                    'confidence': scar['confidence'],
                    'description': scar.get('description', '')
                })
                
            # Add birthmarks
            for bm in birthmarks.get('birthmarks', []):
                marker_map.append({
                    'type': MarkerType.BIRTHMARK.value,
                    'location': bm['location'],
                    'confidence': bm['confidence'],
                    'description': f"Size: {bm.get('size', 'unknown')}"
                })
                
            # Add injury signs
            for inj in injuries:
                marker_map.append({
                    'type': MarkerType.INJURY.value,
                    'location': inj['location'],
                    'confidence': inj['confidence'],
                    'description': inj.get('type', '')
                })
                
            result['marker_map'] = marker_map
            result['markers_detected'] = len(marker_map)
            
            # CREATE UNIQUE SIGNATURE
            signature = self._create_marker_signature(marker_map, wrinkles)
            result['unique_marker_signature'] = signature
            
            # Calculate confidence
            if result['markers_detected'] > 0:
                avg_conf = sum(m['confidence'] for m in marker_map) / len(marker_map)
                result['marker_confidence'] = round(avg_conf * 100, 2)
                
        except Exception as e:
            result['error'] = str(e)
            
        return result
    
    def _detect_scars(self, image: np.ndarray, face_box: Dict) -> Dict:
        """
        Detect scar tissue on face.
        Scars have different texture than surrounding skin.
        """
        scars = {
            'scars_detected': False,
            'scar_count': 0,
            'scars': [],
            'scar_severity': 'none'
        }
        
        try:
            x, y, w, h = face_box.get('x', 0), face_box.get('y', 0), face_box.get('w', 100), face_box.get('h', 100)
            
            face = image[y:y+h, x:x+w]
            if face.size == 0:
                return scars
                
            gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            
            # Scars often have different texture - detect using multiple methods
            
            # Method 1: Edge detection for linear scars
            edges = cv2.Canny(gray, 30, 100)
            
            # Method 2: Texture analysis using LBP-like approach
            # Calculate local standard deviation
            kernel_size = 5
            mean = cv2.blur(gray.astype(float), (kernel_size, kernel_size))
            sqr_mean = cv2.blur(gray.astype(float)**2, (kernel_size, kernel_size))
            std_dev = np.sqrt(np.maximum(sqr_mean - mean**2, 0))
            
            # Scars have lower texture variance than normal skin
            smooth_regions = std_dev < 10
            
            # Method 3: Color analysis - scars are often lighter or darker
            hsv = cv2.cvtColor(face, cv2.COLOR_BGR2HSV)
            saturation = hsv[:, :, 1]
            value = hsv[:, :, 2]
            
            # Scars often have low saturation (less color)
            desaturated = saturation < 30
            
            # Combine indicators
            scar_indicators = np.logical_and(smooth_regions, desaturated)
            
            # Find connected components (potential scars)
            scar_indicators = scar_indicators.astype(np.uint8) * 255
            contours, _ = cv2.findContours(scar_indicators, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                
                # Scars are elongated, not circular
                if area > 50:
                    x_c, y_c, w_c, h_c = cv2.boundingRect(contour)
                    aspect = max(w_c, h_c) / (min(w_c, h_c) + 1)
                    
                    if aspect > 1.5:  # Elongated shape (scar-like)
                        # Calculate position relative to face
                        center_x = (x_c + w_c/2) / w
                        center_y = (y_c + h_c/2) / h
                        
                        region = self._get_facial_region(center_x, center_y)
                        
                        scar_info = {
                            'location': {
                                'relative_x': round(center_x, 3),
                                'relative_y': round(center_y, 3),
                                'region': region
                            },
                            'size': {
                                'width': round(w_c / w, 3),
                                'height': round(h_c / h, 3)
                            },
                            'shape': 'linear' if aspect > 3 else 'irregular',
                            'confidence': min(0.5 + area / 1000, 0.95),
                            'description': f'{region} area, {"linear" if aspect > 3 else "irregular"} pattern'
                        }
                        scars['scars'].append(scar_info)
                        
            scars['scar_count'] = len(scars['scars'])
            scars['scars_detected'] = scars['scar_count'] > 0
            
            if scars['scar_count'] > 3:
                scars['scar_severity'] = 'high'
            elif scars['scar_count'] > 0:
                scars['scar_severity'] = 'moderate' if scars['scar_count'] > 1 else 'low'
                
        except Exception as e:
            pass
            
        return scars
    
    def _map_wrinkles(self, image: np.ndarray, face_box: Dict, landmarks: Dict = None) -> Dict:
        """
        Detailed mapping of facial wrinkles and creases.
        """
        wrinkles = {
            'wrinkle_map': {},
            'total_wrinkle_score': 0.0,
            'primary_wrinkles': [],
            'age_indicator': ''
        }
        
        try:
            x, y, w, h = face_box.get('x', 0), face_box.get('y', 0), face_box.get('w', 100), face_box.get('h', 100)
            
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            face = gray[y:y+h, x:x+w]
            
            if face.size == 0:
                return wrinkles
                
            # ANALYZE EACH WRINKLE ZONE
            
            # 1. FOREHEAD - Horizontal worry lines
            forehead = face[:h//4, w//6:5*w//6]
            forehead_wrinkles = self._detect_horizontal_lines(forehead, 'forehead')
            wrinkles['wrinkle_map']['forehead'] = forehead_wrinkles
            
            # 2. GLABELLA - "11" lines between eyebrows
            glabella = face[h//6:h//3, w//3:2*w//3]
            glabella_wrinkles = self._detect_vertical_lines(glabella, 'glabella')
            wrinkles['wrinkle_map']['glabella'] = glabella_wrinkles
            
            # 3. CROW'S FEET - Corner of eyes
            left_eye_corner = face[h//5:h//2.5, :w//5]
            right_eye_corner = face[h//5:h//2.5, 4*w//5:]
            
            left_crows = self._detect_radial_lines(left_eye_corner, 'left_crows_feet')
            right_crows = self._detect_radial_lines(right_eye_corner, 'right_crows_feet')
            
            wrinkles['wrinkle_map']['left_crows_feet'] = left_crows
            wrinkles['wrinkle_map']['right_crows_feet'] = right_crows
            
            # 4. NASOLABIAL FOLDS - Nose to mouth lines
            left_nasolabial = face[h//2:3*h//4, w//6:w//3]
            right_nasolabial = face[h//2:3*h//4, 2*w//3:5*w//6]
            
            wrinkles['wrinkle_map']['left_nasolabial'] = self._detect_diagonal_lines(left_nasolabial, 'left_nasolabial')
            wrinkles['wrinkle_map']['right_nasolabial'] = self._detect_diagonal_lines(right_nasolabial, 'right_nasolabial')
            
            # 5. MARIONETTE LINES - Mouth corners to chin
            left_marionette = face[3*h//4:, w//4:w//2]
            right_marionette = face[3*h//4:, w//2:3*w//4]
            
            wrinkles['wrinkle_map']['left_marionette'] = self._detect_vertical_lines(left_marionette, 'left_marionette')
            wrinkles['wrinkle_map']['right_marionette'] = self._detect_vertical_lines(right_marionette, 'right_marionette')
            
            # 6. PERIORAL - Lip lines
            lip_area = face[2*h//3:3*h//4, w//3:2*w//3]
            wrinkles['wrinkle_map']['perioral'] = self._detect_vertical_lines(lip_area, 'perioral')
            
            # Calculate total score
            total_score = 0
            for region, data in wrinkles['wrinkle_map'].items():
                total_score += data.get('score', 0)
                if data.get('score', 0) > 0.3:
                    wrinkles['primary_wrinkles'].append({
                        'region': region,
                        'score': data.get('score', 0),
                        'pattern': data.get('pattern', 'undefined')
                    })
                    
            wrinkles['total_wrinkle_score'] = round(total_score / len(wrinkles['wrinkle_map']), 2)
            
            # Estimate age indicator
            if wrinkles['total_wrinkle_score'] > 0.5:
                wrinkles['age_indicator'] = 'mature (50+)'
            elif wrinkles['total_wrinkle_score'] > 0.3:
                wrinkles['age_indicator'] = 'middle-aged (35-50)'
            elif wrinkles['total_wrinkle_score'] > 0.1:
                wrinkles['age_indicator'] = 'young adult (25-35)'
            else:
                wrinkles['age_indicator'] = 'youthful (<25)'
                
        except Exception as e:
            pass
            
        return wrinkles
    
    def _detect_horizontal_lines(self, region: np.ndarray, name: str) -> Dict:
        """Detect horizontal lines (wrinkles) in region."""
        result = {'score': 0.0, 'line_count': 0, 'pattern': 'none'}
        
        if region.size == 0:
            return result
            
        try:
            edges = cv2.Canny(region, 20, 60)
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, 10, 
                                    minLineLength=region.shape[1]//4, maxLineGap=5)
            
            if lines is not None:
                h_lines = [l for l in lines if abs(l[0][1] - l[0][3]) < 5]
                result['line_count'] = len(h_lines)
                result['score'] = min(len(h_lines) * 0.15, 1.0)
                
                if len(h_lines) > 4:
                    result['pattern'] = 'heavy'
                elif len(h_lines) > 2:
                    result['pattern'] = 'moderate'
                elif len(h_lines) > 0:
                    result['pattern'] = 'light'
                    
        except:
            pass
            
        return result
    
    def _detect_vertical_lines(self, region: np.ndarray, name: str) -> Dict:
        """Detect vertical lines in region."""
        result = {'score': 0.0, 'line_count': 0, 'pattern': 'none'}
        
        if region.size == 0:
            return result
            
        try:
            edges = cv2.Canny(region, 20, 60)
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, 10, 
                                    minLineLength=region.shape[0]//4, maxLineGap=5)
            
            if lines is not None:
                v_lines = [l for l in lines if abs(l[0][0] - l[0][2]) < 5]
                result['line_count'] = len(v_lines)
                result['score'] = min(len(v_lines) * 0.2, 1.0)
                
                if len(v_lines) > 3:
                    result['pattern'] = 'heavy'
                elif len(v_lines) > 1:
                    result['pattern'] = 'moderate'
                elif len(v_lines) > 0:
                    result['pattern'] = 'light'
                    
        except:
            pass
            
        return result
    
    def _detect_radial_lines(self, region: np.ndarray, name: str) -> Dict:
        """Detect radial lines (crow's feet pattern)."""
        result = {'score': 0.0, 'line_count': 0, 'pattern': 'none'}
        
        if region.size == 0:
            return result
            
        try:
            edges = cv2.Canny(region, 15, 50)
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, 8, 
                                    minLineLength=region.shape[1]//5, maxLineGap=3)
            
            if lines is not None:
                # Radial lines have varying angles
                angles = []
                for l in lines:
                    angle = np.arctan2(l[0][3] - l[0][1], l[0][2] - l[0][0])
                    angles.append(abs(angle * 180 / np.pi))
                    
                # Count lines that spread from corner (varying angles)
                radial_lines = len([a for a in angles if 10 < a < 80])
                
                result['line_count'] = radial_lines
                result['score'] = min(radial_lines * 0.2, 1.0)
                
                if radial_lines > 4:
                    result['pattern'] = 'heavy'
                elif radial_lines > 2:
                    result['pattern'] = 'moderate'
                elif radial_lines > 0:
                    result['pattern'] = 'light'
                    
        except:
            pass
            
        return result
    
    def _detect_diagonal_lines(self, region: np.ndarray, name: str) -> Dict:
        """Detect diagonal lines (nasolabial folds)."""
        result = {'score': 0.0, 'depth': 'none', 'pattern': 'none'}
        
        if region.size == 0:
            return result
            
        try:
            edges = cv2.Canny(region, 25, 75)
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, 10, 
                                    minLineLength=region.shape[0]//3, maxLineGap=5)
            
            if lines is not None:
                diagonal_lines = []
                for l in lines:
                    angle = abs(np.arctan2(l[0][3] - l[0][1], l[0][2] - l[0][0]) * 180 / np.pi)
                    if 20 < angle < 70:
                        diagonal_lines.append(l)
                        
                if diagonal_lines:
                    # Measure depth by edge intensity
                    max_intensity = np.max(edges)
                    result['score'] = min(len(diagonal_lines) * 0.3 + max_intensity/255 * 0.3, 1.0)
                    
                    if result['score'] > 0.6:
                        result['depth'] = 'deep'
                        result['pattern'] = 'prominent'
                    elif result['score'] > 0.3:
                        result['depth'] = 'moderate'
                        result['pattern'] = 'visible'
                    else:
                        result['depth'] = 'shallow'
                        result['pattern'] = 'faint'
                        
        except:
            pass
            
        return result
    
    def _detect_birthmarks(self, image: np.ndarray, face_box: Dict) -> Dict:
        """
        Detect birthmarks, port wine stains, and pigmentation anomalies.
        """
        birthmarks = {
            'birthmarks_detected': False,
            'birthmarks': [],
            'pigmentation_anomalies': []
        }
        
        try:
            x, y, w, h = face_box.get('x', 0), face_box.get('y', 0), face_box.get('w', 100), face_box.get('h', 100)
            
            face = image[y:y+h, x:x+w]
            if face.size == 0:
                return birthmarks
                
            # Convert to different color spaces for analysis
            hsv = cv2.cvtColor(face, cv2.COLOR_BGR2HSV)
            lab = cv2.cvtColor(face, cv2.COLOR_BGR2LAB)
            
            # DETECT PORT WINE STAINS (reddish patches)
            lower_red = np.array([0, 50, 50])
            upper_red = np.array([15, 255, 255])
            
            red_mask = cv2.inRange(hsv, lower_red, upper_red)
            
            # DETECT BROWN SPOTS (birthmarks/moles)
            lower_brown = np.array([10, 50, 50])
            upper_brown = np.array([25, 255, 180])
            
            brown_mask = cv2.inRange(hsv, lower_brown, upper_brown)
            
            # DETECT DEPIGMENTATION (vitiligo-like)
            # Low saturation, high value
            l_channel = lab[:, :, 0]
            high_brightness = l_channel > 200
            low_saturation = hsv[:, :, 1] < 30
            
            depigment_mask = np.logical_and(high_brightness, low_saturation).astype(np.uint8) * 255
            
            # Analyze each mask for significant patches
            for mask, bm_type in [(red_mask, 'port_wine_stain'), 
                                   (brown_mask, 'brown_birthmark'),
                                   (depigment_mask, 'depigmentation')]:
                                   
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                for contour in contours:
                    area = cv2.contourArea(contour)
                    
                    if area > 100:  # Significant size
                        x_c, y_c, w_c, h_c = cv2.boundingRect(contour)
                        
                        # Calculate relative position
                        center_x = (x_c + w_c/2) / w
                        center_y = (y_c + h_c/2) / h
                        
                        region = self._get_facial_region(center_x, center_y)
                        
                        birthmark_info = {
                            'type': bm_type,
                            'location': {
                                'relative_x': round(center_x, 3),
                                'relative_y': round(center_y, 3),
                                'region': region
                            },
                            'size': 'large' if area > 500 else 'medium' if area > 200 else 'small',
                            'area': round(area / (w * h) * 100, 2),  # % of face
                            'confidence': min(0.6 + area / 2000, 0.95)
                        }
                        
                        birthmarks['birthmarks'].append(birthmark_info)
                        
            birthmarks['birthmarks_detected'] = len(birthmarks['birthmarks']) > 0
            
        except Exception as e:
            pass
            
        return birthmarks
    
    def _detect_injury_signs(self, image: np.ndarray, face_box: Dict) -> List[Dict]:
        """
        Detect signs of past facial injuries.
        """
        injuries = []
        
        try:
            x, y, w, h = face_box.get('x', 0), face_box.get('y', 0), face_box.get('w', 100), face_box.get('h', 100)
            
            face = image[y:y+h, x:x+w]
            gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            
            # DETECT NOSE DEVIATION (broken nose)
            nose_region = gray[h//3:2*h//3, w//3:2*w//3]
            if nose_region.size > 0:
                # Check for asymmetry in nose region
                mid = nose_region.shape[1] // 2
                left = nose_region[:, :mid]
                right = np.fliplr(nose_region[:, mid:])
                
                min_w = min(left.shape[1], right.shape[1])
                left = left[:, :min_w]
                right = right[:, :min_w]
                
                if left.shape == right.shape:
                    diff = cv2.absdiff(left, right)
                    asymmetry = np.mean(diff)
                    
                    if asymmetry > 25:  # Significant asymmetry
                        injuries.append({
                            'type': 'possible_broken_nose',
                            'location': {
                                'relative_x': 0.5,
                                'relative_y': 0.5,
                                'region': 'nose'
                            },
                            'severity': 'moderate' if asymmetry > 40 else 'mild',
                            'confidence': min(asymmetry / 50, 0.9)
                        })
                        
            # DETECT CAULIFLOWER EAR indicators (check edge regions)
            # This would be more visible in side profiles
            
            # DETECT BURN SCARS (unusual smooth texture patterns)
            # Already partially covered in scar detection
            
        except Exception as e:
            pass
            
        return injuries
    
    def _detect_surgery_signs(self, image: np.ndarray, face_box: Dict, landmarks: Dict = None) -> List[Dict]:
        """
        Detect signs of cosmetic/reconstructive surgery.
        """
        surgery_signs = []
        
        try:
            x, y, w, h = face_box.get('x', 0), face_box.get('y', 0), face_box.get('w', 100), face_box.get('h', 100)
            
            face = image[y:y+h, x:x+w]
            gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            
            # RHINOPLASTY INDICATORS
            # Unusually straight nose bridge, thin nostrils
            nose_bridge = gray[h//4:h//2, 2*w//5:3*w//5]
            if nose_bridge.size > 0:
                edges = cv2.Canny(nose_bridge, 20, 60)
                lines = cv2.HoughLinesP(edges, 1, np.pi/180, 10, 
                                        minLineLength=nose_bridge.shape[0]//2, maxLineGap=3)
                
                if lines is not None:
                    # Check for unusually straight vertical lines
                    straight_lines = [l for l in lines if abs(l[0][0] - l[0][2]) < 3]
                    if len(straight_lines) > 2:
                        surgery_signs.append({
                            'type': 'possible_rhinoplasty',
                            'indicator': 'unusually_straight_nose_bridge',
                            'confidence': 0.4
                        })
                        
            # FACELIFT INDICATORS
            # Unusually tight skin near hairline/ears
            # This is difficult to detect reliably from frontal photos
            
            # EYELID SURGERY INDICATORS
            # Absence of natural eyelid crease variations
            
        except Exception as e:
            pass
            
        return surgery_signs
    
    def _get_facial_region(self, rel_x: float, rel_y: float) -> str:
        """
        Determine facial region from relative coordinates.
        """
        if rel_y < 0.25:
            return 'forehead'
        elif rel_y < 0.45:
            if rel_x < 0.35:
                return 'left_eye'
            elif rel_x > 0.65:
                return 'right_eye'
            else:
                return 'nose_bridge'
        elif rel_y < 0.65:
            if rel_x < 0.35:
                return 'left_cheek'
            elif rel_x > 0.65:
                return 'right_cheek'
            else:
                return 'nose_tip'
        elif rel_y < 0.80:
            if rel_x < 0.35:
                return 'left_mouth'
            elif rel_x > 0.65:
                return 'right_mouth'
            else:
                return 'lips'
        else:
            if rel_x < 0.35:
                return 'left_jaw'
            elif rel_x > 0.65:
                return 'right_jaw'
            else:
                return 'chin'
                
    def _create_marker_signature(self, marker_map: List[Dict], wrinkle_map: Dict) -> str:
        """
        Create unique signature from all detected markers.
        """
        signature_parts = []
        
        # Sort markers by position for consistent signature
        # Use .get() with defaults to avoid KeyError on malformed location dicts
        sorted_markers = sorted(
            marker_map,
            key=lambda m: (
                m.get('location', {}).get('relative_y', 0.5),
                m.get('location', {}).get('relative_x', 0.5),
            ),
        )
        
        for marker in sorted_markers:
            loc = marker.get('location', {})
            rx = loc.get('relative_x', 0.5)
            ry = loc.get('relative_y', 0.5)
            part = f"{marker['type'][0]}{int(rx*10)}{int(ry*10)}"
            signature_parts.append(part)
            
        # Add wrinkle pattern code
        wrinkle_code = ""
        for region in ['forehead', 'glabella', 'left_crows_feet', 'right_crows_feet']:
            data = wrinkle_map.get('wrinkle_map', {}).get(region, {})
            if data.get('score', 0) > 0.3:
                wrinkle_code += region[0].upper()
                
        if wrinkle_code:
            signature_parts.append(f"W{wrinkle_code}")
            
        return "-".join(signature_parts) if signature_parts else "NO_MARKERS"
    
    def compare_markers(self, markers1: Dict, markers2: Dict) -> Dict[str, Any]:
        """
        Compare marker maps between two faces.
        """
        result = {
            'match_score': 0.0,
            'matching_markers': [],
            'unmatched_markers': [],
            'verdict': ''
        }
        
        try:
            map1 = markers1.get('marker_map', [])
            map2 = markers2.get('marker_map', [])
            
            matched = 0
            tolerance = 0.15  # Position tolerance
            
            for m1 in map1:
                for m2 in map2:
                    if m1['type'] == m2['type']:
                        loc1 = m1.get('location', {})
                        loc2 = m2.get('location', {})
                        dist = np.sqrt(
                            (loc1.get('relative_x', 0.5) - loc2.get('relative_x', 0.5))**2 +
                            (loc1.get('relative_y', 0.5) - loc2.get('relative_y', 0.5))**2
                        )
                        if dist < tolerance:
                            matched += 1
                            result['matching_markers'].append({
                                'type': m1['type'],
                                'region': m1['location'].get('region', 'unknown')
                            })
                            break
                            
            total_markers = max(len(map1), len(map2), 1)
            result['match_score'] = round(matched / total_markers * 100, 2)
            
            if result['match_score'] > 70:
                result['verdict'] = 'MARKERS_MATCH - Strong identity correlation'
            elif result['match_score'] > 40:
                result['verdict'] = 'PARTIAL_MATCH - Some markers correspond'
            else:
                result['verdict'] = 'MARKERS_DIFFER - Unique markers do not match'
                
        except Exception as e:
            result['error'] = str(e)
            
        return result
