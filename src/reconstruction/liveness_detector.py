"""
MICRO-EXPRESSION & LIVENESS DETECTION ENGINE
==============================================
Advanced analysis for:
1. Micro-expression detection (genuine vs fake emotions)
2. Liveness detection (anti-spoofing)
3. Photo vs real person detection
4. Video replay attack detection
5. 3D mask detection
6. Deep fake indicators

Critical for KYC:
- Detect if person is being coerced
- Verify real person vs photo/video
- Detect presentation attacks
- Fraud prevention

Techniques:
- Optical flow analysis
- Texture analysis (LBP, frequency)
- Reflection analysis
- Motion patterns
- Moiré pattern detection

Standards:
- ISO 30107-3 (Biometric Presentation Attack Detection)

Author: Anti-Fraud Security Research
Version: 1.0.0
"""

import cv2
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from scipy import ndimage, fft


class LivenessDetector:
    """
    Detects if a face is from a live person or a spoof attack.
    """
    
    def __init__(self):
        # Thresholds
        self.TEXTURE_THRESHOLD = 0.5  # LBP variance threshold
        self.FREQUENCY_THRESHOLD = 0.4  # High-frequency content threshold
        self.MOIRE_THRESHOLD = 0.3  # Moiré pattern detection
        
    def detect_liveness(self, face: np.ndarray) -> Dict[str, Any]:
        """
        Complete liveness detection pipeline.
        
        Returns:
            Liveness score and attack type if detected
        """
        result = {
            'is_live': True,
            'liveness_score': 100.0,
            'attack_detected': False,
            'attack_type': None,
            'analysis': {
                'texture_analysis': {},
                'frequency_analysis': {},
                'reflection_analysis': {},
                'quality_analysis': {}
            },
            'confidence': 0.0
        }
        
        if face is None or face.size == 0:
            result['is_live'] = False
            result['liveness_score'] = 0.0
            return result
            
        try:
            scores = []
            
            # 1. TEXTURE ANALYSIS (photo detection)
            texture_result = self._analyze_texture(face)
            result['analysis']['texture_analysis'] = texture_result
            scores.append(texture_result.get('liveness_indicator', 0.5))
            
            if texture_result.get('is_print', False):
                result['attack_detected'] = True
                result['attack_type'] = 'printed_photo'
                
            # 2. FREQUENCY ANALYSIS (screen detection)
            freq_result = self._analyze_frequency(face)
            result['analysis']['frequency_analysis'] = freq_result
            scores.append(freq_result.get('liveness_indicator', 0.5))
            
            if freq_result.get('is_screen', False):
                result['attack_detected'] = True
                result['attack_type'] = 'screen_display'
                
            # 3. REFLECTION ANALYSIS (glass/plastic detection)
            reflection_result = self._analyze_reflections(face)
            result['analysis']['reflection_analysis'] = reflection_result
            scores.append(reflection_result.get('liveness_indicator', 0.5))
            
            if reflection_result.get('has_fake_reflections', False):
                result['attack_detected'] = True
                result['attack_type'] = 'photo_with_coating'
                
            # 4. QUALITY ANALYSIS (blur, JPEG artifacts)
            quality_result = self._analyze_quality(face)
            result['analysis']['quality_analysis'] = quality_result
            scores.append(quality_result.get('liveness_indicator', 0.5))
            
            # 5. 3D STRUCTURE CHECK
            structure_result = self._check_3d_structure(face)
            scores.append(structure_result.get('liveness_indicator', 0.5))
            
            if structure_result.get('is_flat', False):
                result['attack_detected'] = True
                result['attack_type'] = 'flat_surface'
                
            # Calculate overall score
            avg_score = np.mean(scores)
            result['liveness_score'] = round(avg_score * 100, 2)
            # Require at least 2 independent attack indicators for spoof verdict
            attack_count = sum([
                texture_result.get('is_print', False),
                freq_result.get('is_screen', False),
                reflection_result.get('has_fake_reflections', False),
                structure_result.get('is_flat', False)
            ])
            result['is_live'] = avg_score > 0.45 or attack_count < 2
            result['confidence'] = round(abs(avg_score - 0.5) * 200, 2)  # Distance from threshold
            
        except Exception as e:
            result['error'] = str(e)
            result['is_live'] = False
            
        return result
    
    def _analyze_texture(self, face: np.ndarray) -> Dict:
        """
        Analyze texture for print detection.
        Real skin has specific texture, printed photos have different patterns.
        """
        result = {
            'is_print': False,
            'liveness_indicator': 0.5,
            'texture_variance': 0.0,
            'lbp_histogram': None
        }
        
        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        
        # Local Binary Pattern analysis
        lbp = self._compute_lbp(gray)
        
        # LBP histogram
        hist, _ = np.histogram(lbp.ravel(), bins=256, range=(0, 256))
        hist = hist.astype(float) / (hist.sum() + 1e-10)
        
        # Real skin has more uniform LBP distribution
        # Prints have peaks at certain values
        hist_variance = np.var(hist)
        hist_peaks = np.sum(hist > 0.05)  # Count significant peaks
        
        result['texture_variance'] = round(hist_variance * 1000, 4)
        
        # Too many peaks or too low variance suggests print
        if hist_peaks > 10 or hist_variance < 0.0001:
            result['is_print'] = True
            result['liveness_indicator'] = 0.2
        else:
            result['liveness_indicator'] = min(0.3 + hist_variance * 500, 1.0)
            
        return result
    
    def _compute_lbp(self, gray: np.ndarray) -> np.ndarray:
        """Compute Local Binary Pattern."""
        h, w = gray.shape
        lbp = np.zeros_like(gray)
        
        # 3x3 neighborhood
        for i in range(1, h - 1):
            for j in range(1, w - 1):
                center = gray[i, j]
                code = 0
                
                neighbors = [
                    gray[i-1, j-1], gray[i-1, j], gray[i-1, j+1],
                    gray[i, j+1], gray[i+1, j+1], gray[i+1, j],
                    gray[i+1, j-1], gray[i, j-1]
                ]
                
                for k, neighbor in enumerate(neighbors):
                    if neighbor >= center:
                        code |= (1 << k)
                        
                lbp[i, j] = code
                
        return lbp
    
    def _analyze_frequency(self, face: np.ndarray) -> Dict:
        """
        Analyze frequency domain for screen/display detection.
        Screens have specific frequency patterns (Moiré, refresh rate artifacts).
        """
        result = {
            'is_screen': False,
            'liveness_indicator': 0.5,
            'high_freq_ratio': 0.0,
            'has_moire': False
        }
        
        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        
        # FFT analysis
        f_transform = np.fft.fft2(gray.astype(float))
        f_shift = np.fft.fftshift(f_transform)
        magnitude = np.abs(f_shift)
        
        # Log magnitude for visualization
        magnitude_log = np.log(magnitude + 1)
        
        h, w = gray.shape
        center_y, center_x = h // 2, w // 2
        
        # Calculate energy in different frequency bands
        # Low frequency (center)
        low_mask = np.zeros((h, w))
        cv2.circle(low_mask, (center_x, center_y), min(h, w) // 8, 1, -1)
        low_energy = np.sum(magnitude * low_mask)
        
        # High frequency (edges)
        high_mask = np.ones((h, w)) - low_mask
        cv2.circle(high_mask, (center_x, center_y), min(h, w) // 4, 0, -1)
        high_energy = np.sum(magnitude * high_mask)
        
        total_energy = low_energy + high_energy + 1e-10
        high_freq_ratio = high_energy / total_energy
        
        result['high_freq_ratio'] = round(high_freq_ratio, 4)
        
        # Moiré pattern detection (periodic peaks in frequency domain)
        # Look for peaks away from center
        threshold = np.mean(magnitude_log) + 2 * np.std(magnitude_log)
        peaks = magnitude_log > threshold
        
        # Exclude center
        peaks[center_y-10:center_y+10, center_x-10:center_x+10] = False
        
        num_peaks = np.sum(peaks)
        
        if num_peaks > 50:  # Many periodic peaks = Moiré
            result['has_moire'] = True
            result['is_screen'] = True
            result['liveness_indicator'] = 0.2
        elif high_freq_ratio < 0.1:  # Too little high frequency = blurry/screen
            result['is_screen'] = True
            result['liveness_indicator'] = 0.3
        else:
            result['liveness_indicator'] = 0.5 + high_freq_ratio
            
        return result
    
    def _analyze_reflections(self, face: np.ndarray) -> Dict:
        """
        Analyze reflections for plastic/glass coating detection.
        Photos behind glass have specific reflection patterns.
        """
        result = {
            'has_fake_reflections': False,
            'liveness_indicator': 0.5,
            'specular_count': 0,
            'reflection_pattern': 'normal'
        }
        
        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(face, cv2.COLOR_BGR2HSV)
        
        # Detect specular highlights
        specular_mask = (hsv[:, :, 2] > 240) & (hsv[:, :, 1] < 30)
        
        # Find specular regions
        contours, _ = cv2.findContours(specular_mask.astype(np.uint8),
                                        cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        result['specular_count'] = len(contours)
        
        # Analyze specular shapes
        rectangular_speculars = 0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 300:  # Only consider substantial specular regions
                # Check if rectangular (glass reflection)
                x, y, w, h = cv2.boundingRect(cnt)
                rect_area = w * h
                if area / (rect_area + 1e-10) > 0.6:  # Close to rectangular
                    rectangular_speculars += 1
                    
        if rectangular_speculars > 5:
            result['has_fake_reflections'] = True
            result['reflection_pattern'] = 'glass_like'
            result['liveness_indicator'] = 0.2
        elif result['specular_count'] > 50:  # Extremely excessive highlights
            result['has_fake_reflections'] = True
            result['reflection_pattern'] = 'excessive'
            result['liveness_indicator'] = 0.3
        else:
            result['liveness_indicator'] = 0.7
            
        return result
    
    def _analyze_quality(self, face: np.ndarray) -> Dict:
        """
        Analyze image quality for compression artifacts.
        Photos of photos have double compression artifacts.
        """
        result = {
            'liveness_indicator': 0.5,
            'jpeg_artifacts': False,
            'blur_level': 0.0,
            'noise_level': 0.0
        }
        
        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        
        # Blur detection (Laplacian variance)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        result['blur_level'] = round(laplacian_var, 2)
        
        # JPEG artifact detection
        # Look for 8x8 block boundaries
        h, w = gray.shape
        
        # Difference at 8-pixel boundaries
        boundary_diff = 0
        non_boundary_diff = 0
        
        for i in range(7, h - 1, 8):
            boundary_diff += np.sum(np.abs(gray[i, :].astype(float) - gray[i+1, :].astype(float)))
        for i in range(1, h - 1):
            if i % 8 != 7:
                non_boundary_diff += np.sum(np.abs(gray[i, :].astype(float) - gray[i+1, :].astype(float)))
                
        if non_boundary_diff > 0:
            block_ratio = boundary_diff / (non_boundary_diff / 7 + 1e-10)
            if block_ratio > 1.5:  # Boundaries more visible than interior
                result['jpeg_artifacts'] = True
                result['liveness_indicator'] = 0.3
            else:
                result['liveness_indicator'] = 0.6
        else:
            result['liveness_indicator'] = 0.5
            
        # Very blurry images are suspicious
        if laplacian_var < 50:
            result['liveness_indicator'] *= 0.7
            
        # Estimate noise
        noise = self._estimate_noise(gray)
        result['noise_level'] = round(noise, 2)
        
        return result
    
    def _estimate_noise(self, gray: np.ndarray) -> float:
        """Estimate noise level."""
        blur = cv2.GaussianBlur(gray.astype(float), (5, 5), 0)
        noise = gray.astype(float) - blur
        return np.std(noise)
    
    def _check_3d_structure(self, face: np.ndarray) -> Dict:
        """
        Check for 3D structure hints.
        Real faces have depth, flat surfaces don't.
        """
        result = {
            'is_flat': False,
            'liveness_indicator': 0.5,
            'depth_indicators': 0.0
        }
        
        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        
        # Gradient analysis (3D objects have consistent gradient patterns)
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        # Magnitude and direction
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        direction = np.arctan2(grad_y, grad_x)
        
        # Real faces have specific gradient patterns (nose ridge, eye sockets)
        # Analyze gradient consistency in facial regions
        h, w = gray.shape
        
        # Nose region should have strong vertical gradients
        nose_region = magnitude[h//3:2*h//3, w//3:2*w//3]
        nose_strength = np.mean(nose_region)
        
        # Eye regions should have significant gradients
        left_eye = magnitude[h//4:h//2, w//4:w//2]
        right_eye = magnitude[h//4:h//2, w//2:3*w//4]
        eye_strength = (np.mean(left_eye) + np.mean(right_eye)) / 2
        
        depth_indicator = (nose_strength + eye_strength) / 2
        result['depth_indicators'] = round(depth_indicator, 2)
        
        # Very uniform gradients suggest flat surface
        gradient_std = np.std(magnitude)
        
        if gradient_std < 5 and depth_indicator < 10:
            result['is_flat'] = True
            result['liveness_indicator'] = 0.2
        else:
            result['liveness_indicator'] = min(0.5 + depth_indicator / 100, 1.0)
            
        return result


class MicroExpressionAnalyzer:
    """
    Analyzes subtle facial expressions that indicate true emotions.
    """
    
    def __init__(self):
        # Expression indicators
        self.EXPRESSION_REGIONS = {
            'forehead': {'y': (0, 0.25), 'indicators': ['surprise', 'worry']},
            'eyebrows': {'y': (0.15, 0.35), 'indicators': ['anger', 'surprise']},
            'eyes': {'y': (0.25, 0.50), 'indicators': ['fear', 'happiness', 'sadness']},
            'nose': {'y': (0.35, 0.60), 'indicators': ['disgust']},
            'mouth': {'y': (0.55, 0.85), 'indicators': ['happiness', 'sadness', 'contempt']}
        }
        
    def analyze_expression(self, face: np.ndarray) -> Dict[str, Any]:
        """
        Analyze facial micro-expressions.
        """
        result = {
            'dominant_expression': 'neutral',
            'expression_scores': {},
            'asymmetry_detected': False,
            'potential_deception_indicators': [],
            'confidence': 0.0
        }
        
        if face is None or face.size == 0:
            return result
            
        try:
            gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape
            
            # Analyze each facial region
            region_activities = {}
            
            for region_name, params in self.EXPRESSION_REGIONS.items():
                y1 = int(params['y'][0] * h)
                y2 = int(params['y'][1] * h)
                region = gray[y1:y2, :]
                
                # Calculate activity (texture, edges)
                edges = cv2.Canny(region, 50, 150)
                activity = np.sum(edges > 0) / edges.size
                
                region_activities[region_name] = round(activity, 4)
                
            # Check for asymmetry (indicator of fake expressions)
            left_half = gray[:, :w//2]
            right_half = np.fliplr(gray[:, w//2:])
            
            min_w = min(left_half.shape[1], right_half.shape[1])
            left_half = left_half[:, :min_w]
            right_half = right_half[:, :min_w]
            
            asymmetry = np.mean(cv2.absdiff(left_half, right_half))
            
            if asymmetry > 20:
                result['asymmetry_detected'] = True
                result['potential_deception_indicators'].append('facial_asymmetry')
                
            # Simple expression scoring based on region activity
            expressions = {
                'surprise': region_activities.get('forehead', 0) * 2 + region_activities.get('eyebrows', 0),
                'anger': region_activities.get('eyebrows', 0) * 2,
                'happiness': region_activities.get('mouth', 0) * 2 + region_activities.get('eyes', 0),
                'sadness': region_activities.get('mouth', 0) + region_activities.get('eyes', 0) * 0.5,
                'disgust': region_activities.get('nose', 0) * 2 + region_activities.get('mouth', 0),
                'fear': region_activities.get('eyes', 0) * 2 + region_activities.get('eyebrows', 0),
                'neutral': 1 - max(region_activities.values()) if region_activities else 0
            }
            
            # Normalize
            total = sum(expressions.values()) + 1e-10
            for exp in expressions:
                expressions[exp] = round(expressions[exp] / total * 100, 2)
                
            result['expression_scores'] = expressions
            result['dominant_expression'] = max(expressions, key=expressions.get)
            result['confidence'] = expressions[result['dominant_expression']]
            
        except Exception as e:
            result['error'] = str(e)
            
        return result
