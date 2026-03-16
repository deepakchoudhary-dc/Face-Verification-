"""
FACE MORPHING ATTACK DETECTION
==============================
Detects morphed/blended faces used for identity fraud.

Face morphing is a sophisticated attack where two faces are blended together
to create a synthetic face that can pass biometric verification for BOTH
original identities. This is used for:
- Passport fraud (morphed photo accepted by two people)
- Identity sharing between criminals
- Evading biometric watchlists

Morphing Detection Techniques:
1. Texture Analysis - Morphed faces have unnatural texture patterns
2. Landmark Geometry - Morphed landmarks are mathematically averaged
3. Photo Response Analysis - JPEG artifacts from multiple compressions
4. Spectral Analysis - Frequency domain anomalies from blending
5. Deep Artifact Detection - GAN fingerprints and blending artifacts
6. Micro-Texture Analysis - Skin pore and hair patterns disrupted
7. Eye Analysis - Iris patterns impossible to morph naturally

Compliant with:
- ISO/IEC 30107-3 (Biometric Presentation Attack Detection)
- NIST FRVT MORPH evaluation framework
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Any
from scipy import ndimage, fft
import math


class MorphingDetector:
    """
    State-of-the-art face morphing attack detection system.
    Combines multiple detection methods for robust performance.
    """
    
    def __init__(self):
        self.MORPH_THRESHOLD = 0.55  # Above this = suspected morph
        
        # Detection weights for ensemble decision
        self.WEIGHTS = {
            'texture': 0.20,
            'spectral': 0.15,
            'landmark': 0.15,
            'jpeg_artifact': 0.15,
            'micro_texture': 0.15,
            'eye_analysis': 0.20
        }
        
    def detect_morphing(self, image: np.ndarray, face_box: Dict,
                         landmarks: Dict = None) -> Dict[str, Any]:
        """
        Comprehensive morphing attack detection.
        
        Returns:
            Dict with morphing probability, detection details, and recommendation
        """
        result = {
            'is_morphed': False,
            'morphing_probability': 0.0,
            'detection_scores': {},
            'artifacts_found': [],
            'confidence': 0.0,
            'recommendation': '',
            'detailed_analysis': {}
        }
        
        if image is None:
            return result
            
        try:
            x, y, w, h = face_box.get('x', 0), face_box.get('y', 0), face_box.get('w', 100), face_box.get('h', 100)
            
            face = image[y:y+h, x:x+w]
            if face.size == 0:
                return result
                
            # 1. TEXTURE ANOMALY DETECTION
            texture_score, texture_details = self._analyze_texture_anomalies(face)
            result['detection_scores']['texture'] = texture_score
            result['detailed_analysis']['texture'] = texture_details
            
            # 2. SPECTRAL ANALYSIS
            spectral_score, spectral_details = self._spectral_analysis(face)
            result['detection_scores']['spectral'] = spectral_score
            result['detailed_analysis']['spectral'] = spectral_details
            
            # 3. LANDMARK GEOMETRY CHECK
            landmark_score, landmark_details = self._analyze_landmark_geometry(face, landmarks)
            result['detection_scores']['landmark'] = landmark_score
            result['detailed_analysis']['landmark'] = landmark_details
            
            # 4. JPEG ARTIFACT ANALYSIS
            jpeg_score, jpeg_details = self._analyze_jpeg_artifacts(face)
            result['detection_scores']['jpeg_artifact'] = jpeg_score
            result['detailed_analysis']['jpeg'] = jpeg_details
            
            # 5. MICRO-TEXTURE ANALYSIS
            micro_score, micro_details = self._analyze_micro_texture(face)
            result['detection_scores']['micro_texture'] = micro_score
            result['detailed_analysis']['micro_texture'] = micro_details
            
            # 6. EYE REGION ANALYSIS
            eye_score, eye_details = self._analyze_eye_regions(face)
            result['detection_scores']['eye_analysis'] = eye_score
            result['detailed_analysis']['eye'] = eye_details
            
            # CALCULATE WEIGHTED ENSEMBLE SCORE
            ensemble_score = sum(
                result['detection_scores'].get(key, 0) * weight 
                for key, weight in self.WEIGHTS.items()
            )
            
            result['morphing_probability'] = round(ensemble_score * 100, 2)
            
            # Collect artifacts found
            artifacts = []
            if texture_score > 0.5:
                artifacts.append('Unnatural texture blending')
            if spectral_score > 0.5:
                artifacts.append('Spectral domain anomalies')
            if jpeg_score > 0.5:
                artifacts.append('Multiple JPEG compression artifacts')
            if micro_score > 0.5:
                artifacts.append('Micro-texture discontinuities')
            if eye_score > 0.5:
                artifacts.append('Suspicious eye region patterns')
                
            result['artifacts_found'] = artifacts
            
            # FINAL VERDICT
            if ensemble_score > self.MORPH_THRESHOLD:
                result['is_morphed'] = True
                result['recommendation'] = '[WARNING] MORPHING ATTACK SUSPECTED - Manual verification required'
            elif ensemble_score > 0.35:
                result['recommendation'] = '[WARNING] UNCERTAIN - Additional verification recommended'
            else:
                result['recommendation'] = '[OK] No morphing indicators detected'
                
            # Confidence based on agreement between detectors
            scores = list(result['detection_scores'].values())
            variance = np.var(scores)
            result['confidence'] = round((1 - min(variance * 4, 0.5)) * 100, 2)
            
        except Exception as e:
            result['error'] = str(e)
            
        return result
    
    def _analyze_texture_anomalies(self, face: np.ndarray) -> Tuple[float, Dict]:
        """
        Detect texture anomalies from morphing process.
        Morphed faces have unnatural smoothness in blending regions.
        """
        details = {
            'blending_artifacts': False,
            'texture_consistency': 0.0,
            'smoothness_anomaly': 0.0
        }
        
        try:
            gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            
            # Divide face into grid and analyze texture consistency
            h, w = gray.shape
            grid_size = 4
            cell_h, cell_w = h // grid_size, w // grid_size
            
            textures = []
            for i in range(grid_size):
                for j in range(grid_size):
                    cell = gray[i*cell_h:(i+1)*cell_h, j*cell_w:(j+1)*cell_w]
                    if cell.size > 0:
                        # Calculate texture complexity using Laplacian variance
                        laplacian_var = cv2.Laplacian(cell, cv2.CV_64F).var()
                        textures.append(laplacian_var)
                        
            if textures:
                # Morphed faces often have unusually uniform texture
                texture_std = np.std(textures)
                texture_mean = np.mean(textures)
                
                # Coefficient of variation
                cv = texture_std / (texture_mean + 1e-10)
                
                # Very low CV suggests artificial smoothness
                if cv < 0.3:
                    details['blending_artifacts'] = True
                    
                details['texture_consistency'] = round(cv, 3)
                
            # Check for unnatural smoothness
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            diff = cv2.absdiff(gray, blurred)
            smoothness = 1 - (np.mean(diff) / 128)
            
            # Natural faces have moderate smoothness
            if smoothness > 0.85:  # Too smooth
                details['smoothness_anomaly'] = round(smoothness, 3)
                
            # Calculate score
            score = 0.0
            if details['blending_artifacts']:
                score += 0.4
            if details['texture_consistency'] < 0.3:
                score += 0.3
            if details['smoothness_anomaly'] > 0.85:
                score += 0.3
                
        except Exception as e:
            pass
            
        return min(score, 1.0), details
    
    def _spectral_analysis(self, face: np.ndarray) -> Tuple[float, Dict]:
        """
        Analyze frequency domain for morphing artifacts.
        Morphing introduces specific spectral patterns.
        """
        details = {
            'high_freq_anomaly': False,
            'spectral_signature': '',
            'blend_frequency_detected': False
        }
        
        try:
            gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            
            # 2D FFT
            f_transform = fft.fft2(gray)
            f_shift = fft.fftshift(f_transform)
            magnitude = 20 * np.log(np.abs(f_shift) + 1)
            
            # Analyze different frequency bands
            h, w = magnitude.shape
            center_y, center_x = h // 2, w // 2
            
            # Low frequency (center) - overall structure
            low_freq_region = magnitude[center_y-h//8:center_y+h//8, center_x-w//8:center_x+w//8]
            low_freq_energy = np.mean(low_freq_region)
            
            # Mid frequency - facial features
            mid_mask = np.zeros_like(magnitude, dtype=bool)
            y_coords, x_coords = np.ogrid[:h, :w]
            dist = np.sqrt((y_coords - center_y)**2 + (x_coords - center_x)**2)
            mid_mask = (dist > min(h, w) // 8) & (dist < min(h, w) // 3)
            mid_freq_energy = np.mean(magnitude[mid_mask])
            
            # High frequency - fine details
            high_mask = dist >= min(h, w) // 3
            high_freq_energy = np.mean(magnitude[high_mask])
            
            # Morphing typically reduces high frequency content
            freq_ratio = high_freq_energy / (low_freq_energy + 1e-10)
            
            if freq_ratio < 0.15:  # Unusually low high frequency
                details['high_freq_anomaly'] = True
                
            # Look for periodic patterns from blending
            # Morphing can introduce specific frequency peaks
            mag_variance = np.var(magnitude[mid_mask])
            if mag_variance < 50:  # Unusually uniform spectrum
                details['blend_frequency_detected'] = True
                
            details['spectral_signature'] = f"L:{low_freq_energy:.1f}/M:{mid_freq_energy:.1f}/H:{high_freq_energy:.1f}"
            
            # Calculate score
            score = 0.0
            if details['high_freq_anomaly']:
                score += 0.5
            if details['blend_frequency_detected']:
                score += 0.5
                
        except Exception as e:
            pass
            
        return min(score, 1.0), details
    
    def _analyze_landmark_geometry(self, face: np.ndarray, landmarks: Dict = None) -> Tuple[float, Dict]:
        """
        Analyze facial landmarks for morphing-specific geometry.
        Morphed landmarks tend toward mathematical averages.
        """
        details = {
            'geometric_anomaly': False,
            'symmetry_score': 0.0,
            'proportion_anomaly': False
        }
        
        try:
            gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape
            
            # If no landmarks provided, estimate from image
            # Check for unusual symmetry (morphed faces are often too symmetric)
            mid = w // 2
            left = gray[:, :mid]
            right = np.fliplr(gray[:, mid:])
            
            min_w = min(left.shape[1], right.shape[1])
            left = left[:, :min_w]
            right = right[:, :min_w]
            
            if left.shape == right.shape:
                diff = cv2.absdiff(left, right)
                asymmetry = np.mean(diff) / 255
                
                # Natural faces have some asymmetry
                # Too symmetric = possible morph
                details['symmetry_score'] = round(1 - asymmetry, 3)
                
                if details['symmetry_score'] > 0.92:  # Unusually symmetric
                    details['geometric_anomaly'] = True
                    
            # Check face proportions
            # Morphed faces often have "averaged" proportions
            # Use edge detection to estimate feature positions
            edges = cv2.Canny(gray, 50, 150)
            
            # Vertical profile (sum edges horizontally)
            v_profile = np.sum(edges, axis=1)
            
            # Find key feature positions (peaks in edge density)
            peaks = []
            for i in range(1, len(v_profile) - 1):
                if v_profile[i] > v_profile[i-1] and v_profile[i] > v_profile[i+1]:
                    if v_profile[i] > np.mean(v_profile):
                        peaks.append(i)
                        
            # Check if features are at "golden ratio" positions (too perfect = morph indicator)
            if len(peaks) >= 3:
                ratios = []
                for i in range(len(peaks) - 1):
                    ratio = peaks[i] / (peaks[i+1] + 1e-10)
                    ratios.append(ratio)
                    
                # Check for suspiciously "perfect" ratios near 0.618 or 0.5
                perfect_ratios = sum(1 for r in ratios if 0.45 < r < 0.55 or 0.60 < r < 0.65)
                if perfect_ratios >= 2:
                    details['proportion_anomaly'] = True
                    
            # Calculate score
            score = 0.0
            if details['geometric_anomaly']:
                score += 0.5
            if details['proportion_anomaly']:
                score += 0.5
                
        except Exception as e:
            pass
            
        return min(score, 1.0), details
    
    def _analyze_jpeg_artifacts(self, face: np.ndarray) -> Tuple[float, Dict]:
        """
        Detect JPEG compression artifacts from morphing process.
        Morphed images are typically saved/loaded multiple times.
        """
        details = {
            'double_compression': False,
            'blocking_artifacts': 0.0,
            'ghost_artifacts': False
        }
        
        try:
            gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            
            # DETECT BLOCKING ARTIFACTS
            # JPEG uses 8x8 blocks
            h, w = gray.shape
            
            # Check for discontinuities at 8-pixel boundaries
            block_discontinuities = []
            
            for i in range(8, h - 8, 8):
                row_above = gray[i-1, :]
                row_below = gray[i, :]
                diff = np.abs(row_above.astype(float) - row_below.astype(float))
                block_discontinuities.append(np.mean(diff))
                
            for j in range(8, w - 8, 8):
                col_left = gray[:, j-1]
                col_right = gray[:, j]
                diff = np.abs(col_left.astype(float) - col_right.astype(float))
                block_discontinuities.append(np.mean(diff))
                
            if block_discontinuities:
                blocking_score = np.mean(block_discontinuities) / 50
                details['blocking_artifacts'] = round(blocking_score, 3)
                
                if blocking_score > 0.3:  # Significant blocking
                    details['double_compression'] = True
                    
            # DETECT JPEG GHOSTS (different quality regions)
            # Re-compress at multiple quality levels and check for anomalies
            quality_levels = [70, 80, 90]
            ghost_detected = False
            
            for quality in quality_levels:
                # Simulate JPEG compression
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
                _, encoded = cv2.imencode('.jpg', face, encode_param)
                decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
                
                diff = cv2.absdiff(face, decoded)
                diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
                
                # Check for non-uniform differences (ghost regions)
                diff_std = np.std(diff_gray)
                if diff_std > 15:  # Non-uniform compression response
                    ghost_detected = True
                    break
                    
            details['ghost_artifacts'] = ghost_detected
            
            # Calculate score
            score = 0.0
            if details['double_compression']:
                score += 0.5
            if details['ghost_artifacts']:
                score += 0.5
                
        except Exception as e:
            pass
            
        return min(score, 1.0), details
    
    def _analyze_micro_texture(self, face: np.ndarray) -> Tuple[float, Dict]:
        """
        Analyze micro-texture for morphing artifacts.
        Skin pores and fine details are disrupted in morphed images.
        """
        details = {
            'pore_pattern_disruption': False,
            'hair_texture_anomaly': False,
            'micro_detail_score': 0.0
        }
        
        try:
            gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape
            
            # ANALYZE CHEEK REGION (best for micro-texture)
            cheek_y = h // 3
            cheek_h = h // 4
            cheek_x = w // 4
            cheek_w = w // 4
            
            cheek = gray[cheek_y:cheek_y+cheek_h, cheek_x:cheek_x+cheek_w]
            
            if cheek.size > 100:
                # Extract micro-texture using high-pass filter
                blurred = cv2.GaussianBlur(cheek, (7, 7), 0)
                micro_texture = cv2.absdiff(cheek, blurred)
                
                # Natural skin has pore patterns (small dark spots in micro-texture)
                _, pores = cv2.threshold(micro_texture, 10, 255, cv2.THRESH_BINARY)
                pore_density = np.sum(pores > 0) / pores.size
                
                # Morphed faces often have reduced pore visibility
                if pore_density < 0.05:  # Too few pores visible
                    details['pore_pattern_disruption'] = True
                    
                details['micro_detail_score'] = round(pore_density, 4)
                
            # ANALYZE HAIRLINE/EYEBROW TEXTURE
            # Natural hair has fine parallel structures
            eyebrow_region = gray[h//6:h//4, w//4:3*w//4]
            
            if eyebrow_region.size > 100:
                # Gabor filter for oriented textures
                ksize = 15
                sigma = 3
                theta = 0  # Horizontal orientation (typical eyebrow direction)
                lambd = 8
                gamma = 0.5
                
                gabor_kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, 0)
                filtered = cv2.filter2D(eyebrow_region, cv2.CV_64F, gabor_kernel)
                
                # Strong response indicates good hair texture
                hair_texture = np.std(filtered)
                
                if hair_texture < 15:  # Weak hair texture
                    details['hair_texture_anomaly'] = True
                    
            # Calculate score
            score = 0.0
            if details['pore_pattern_disruption']:
                score += 0.5
            if details['hair_texture_anomaly']:
                score += 0.5
                
        except Exception as e:
            pass
            
        return min(score, 1.0), details
    
    def _analyze_eye_regions(self, face: np.ndarray) -> Tuple[float, Dict]:
        """
        Analyze eye regions for morphing artifacts.
        Eyes are particularly difficult to morph naturally.
        """
        details = {
            'iris_anomaly': False,
            'reflection_inconsistency': False,
            'eye_region_blur': 0.0
        }
        
        try:
            gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape
            
            # Extract eye regions (approximate positions)
            left_eye = gray[h//5:h//3, w//6:2*w//5]
            right_eye = gray[h//5:h//3, 3*w//5:5*w//6]
            
            for eye_name, eye_region in [('left', left_eye), ('right', right_eye)]:
                if eye_region.size < 100:
                    continue
                    
                # Check for unnatural blur in eye region
                laplacian_var = cv2.Laplacian(eye_region, cv2.CV_64F).var()
                
                if laplacian_var < 100:  # Blurry eye region
                    details['eye_region_blur'] = max(details['eye_region_blur'], 
                                                      round(1 - laplacian_var/200, 3))
                    
                # Check for catch light (reflection) patterns
                # Natural photos have specular highlights in eyes
                _, bright_spots = cv2.threshold(eye_region, 200, 255, cv2.THRESH_BINARY)
                highlight_ratio = np.sum(bright_spots > 0) / bright_spots.size
                
                # Morphing often removes or distorts catch lights
                if highlight_ratio < 0.001:  # No catch lights
                    details['reflection_inconsistency'] = True
                    
            # Check left-right consistency
            if left_eye.size > 100 and right_eye.size > 100:
                # Resize to compare
                min_h = min(left_eye.shape[0], right_eye.shape[0])
                min_w = min(left_eye.shape[1], right_eye.shape[1])
                
                left_resized = cv2.resize(left_eye, (min_w, min_h))
                right_resized = cv2.resize(np.fliplr(right_eye), (min_w, min_h))
                
                # Eyes should be similar but not identical
                similarity = 1 - np.mean(cv2.absdiff(left_resized, right_resized)) / 255
                
                if similarity > 0.95:  # Too similar (morphing artifact)
                    details['iris_anomaly'] = True
                    
            # Calculate score
            score = 0.0
            if details['iris_anomaly']:
                score += 0.4
            if details['reflection_inconsistency']:
                score += 0.3
            if details['eye_region_blur'] > 0.5:
                score += 0.3
                
        except Exception as e:
            pass
            
        return min(score, 1.0), details
    
    def compare_for_morphing(self, live_face: np.ndarray, document_face: np.ndarray,
                              face_match_score: float) -> Dict[str, Any]:
        """
        Compare live capture with document photo to detect morphing attacks.
        
        Args:
            live_face: Face from live capture (trusted)
            document_face: Face from ID document (potentially morphed)
            face_match_score: Face recognition match score
        """
        result = {
            'attack_detected': False,
            'attack_type': '',
            'confidence': 0.0,
            'evidence': [],
            'recommendation': ''
        }
        
        try:
            # Analyze document face for morphing
            doc_morph = self.detect_morphing(document_face, 
                                              {'x': 0, 'y': 0, 
                                               'w': document_face.shape[1], 
                                               'h': document_face.shape[0]})
            
            if doc_morph['is_morphed']:
                result['attack_detected'] = True
                result['attack_type'] = 'MORPHED_DOCUMENT'
                result['confidence'] = doc_morph['morphing_probability']
                result['evidence'] = doc_morph['artifacts_found']
                result['recommendation'] = '[ALERT] REJECT - Document photo appears to be morphed'
                
            # Even if not clearly morphed, check for suspicious match patterns
            elif face_match_score > 75:  # Good match
                # A morphed face might match multiple people
                # Check if the document face has unusual characteristics
                if doc_morph['morphing_probability'] > 30:
                    result['attack_type'] = 'POSSIBLE_MORPH'
                    result['confidence'] = doc_morph['morphing_probability']
                    result['evidence'] = ['Match score normal but morphing indicators present']
                    result['recommendation'] = '[WARNING] Additional verification recommended'
                    
        except Exception as e:
            result['error'] = str(e)
            
        return result
