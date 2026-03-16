"""
LIGHTING NORMALIZATION ENGINE
==============================
Corrects lighting issues while PRESERVING natural skin color and texture.

Key principle: Only fix what's broken. Don't over-process.
"""

import cv2
import numpy as np
from typing import Dict, Any


class LightingNormalizer:
    """
    Gentle lighting correction that preserves skin tones.
    
    Only applies corrections when actual lighting issues are detected.
    Never washes out colors. Never flattens contrast unnecessarily.
    """
    
    def normalize_lighting(self, face: np.ndarray) -> Dict[str, Any]:
        """
        Analyze and correct lighting issues.
        Only applies corrections if needed.
        """
        result = {
            'normalized': face.copy() if face is not None else None,
            'success': False,
            'lighting_analysis': {},
            'corrections_applied': []
        }
        
        if face is None or face.size == 0:
            return result
            
        try:
            img = face.copy()
            if img.dtype != np.uint8:
                if img.max() <= 1.0:
                    img = (img * 255).astype(np.uint8)
                else:
                    img = img.astype(np.uint8)
            
            # Analyze lighting conditions
            analysis = self._analyze_lighting(img)
            result['lighting_analysis'] = analysis
            
            # Only correct if there are actual issues
            if analysis.get('is_underexposed'):
                img = self._fix_underexposure(img)
                result['corrections_applied'].append('underexposure_correction')
                
            if analysis.get('is_overexposed'):
                img = self._fix_overexposure(img)
                result['corrections_applied'].append('overexposure_correction')
                
            if analysis.get('has_strong_shadows'):
                img = self._reduce_shadows(img)
                result['corrections_applied'].append('shadow_reduction')
                
            if analysis.get('has_specular_highlights'):
                img = self._reduce_highlights(img)
                result['corrections_applied'].append('highlight_reduction')
            
            if analysis.get('color_cast_detected'):
                img = self._correct_white_balance(img)
                result['corrections_applied'].append('white_balance')
            
            result['normalized'] = img
            result['success'] = True
            
        except Exception as e:
            result['error'] = str(e)
            result['normalized'] = face.copy()
            
        return result
    
    def _analyze_lighting(self, img: np.ndarray) -> Dict[str, Any]:
        """Analyze lighting conditions without modifying the image."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        
        mean_brightness = float(np.mean(gray))
        std_brightness = float(np.std(gray))
        
        # Check for shadows (left-right brightness imbalance)
        h, w = gray.shape
        left_mean = float(np.mean(gray[:, :w//2]))
        right_mean = float(np.mean(gray[:, w//2:]))
        shadow_ratio = abs(left_mean - right_mean) / max(mean_brightness, 1)
        
        # Check for specular highlights (very bright spots)
        highlight_ratio = float(np.sum(gray > 240)) / gray.size
        
        # Check for color cast
        b, g, r = cv2.split(img) if len(img.shape) == 3 else (gray, gray, gray)
        color_cast = max(
            abs(float(np.mean(b)) - float(np.mean(g))),
            abs(float(np.mean(r)) - float(np.mean(g)))
        )
        
        return {
            'mean_brightness': mean_brightness,
            'brightness_std': std_brightness,
            'is_underexposed': mean_brightness < 70,
            'is_overexposed': mean_brightness > 200,
            'has_strong_shadows': shadow_ratio > 0.25,
            'shadow_ratio': shadow_ratio,
            'has_specular_highlights': highlight_ratio > 0.02,
            'highlight_ratio': highlight_ratio,
            'color_cast_detected': color_cast > 25,
            'color_cast_strength': color_cast
        }
    
    def _fix_underexposure(self, img: np.ndarray) -> np.ndarray:
        """Fix underexposed images by lifting shadows while preserving highlights."""
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l_channel = lab[:, :, 0].astype(np.float32)
        
        # Gamma correction - lifts shadows without blowing highlights
        mean_l = np.mean(l_channel)
        gamma = min(max(0.5, 128.0 / max(mean_l, 1)), 2.0)
        l_corrected = np.power(l_channel / 255.0, 1.0 / gamma) * 255.0
        
        # Blend to avoid over-correction
        lab[:, :, 0] = np.clip(l_corrected * 0.7 + l_channel * 0.3, 0, 255).astype(np.uint8)
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    def _fix_overexposure(self, img: np.ndarray) -> np.ndarray:
        """Fix overexposed images."""
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l_channel = lab[:, :, 0].astype(np.float32)
        
        gamma = max(1.2, min(np.mean(l_channel) / 128.0, 2.5))
        l_corrected = np.power(l_channel / 255.0, gamma) * 255.0
        
        lab[:, :, 0] = np.clip(l_corrected * 0.6 + l_channel * 0.4, 0, 255).astype(np.uint8)
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    def _reduce_shadows(self, img: np.ndarray) -> np.ndarray:
        """Reduce strong shadows while preserving overall look."""
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l_channel = lab[:, :, 0].astype(np.float32)
        
        # Large blur to estimate illumination
        blur = cv2.GaussianBlur(l_channel, (0, 0), 30)
        
        # Only lift dark areas toward local average
        shadow_mask = (l_channel < blur * 0.8).astype(np.float32)
        shadow_mask = cv2.GaussianBlur(shadow_mask, (15, 15), 5)
        
        # Lift shadows gently
        lift = (blur - l_channel) * shadow_mask * 0.3
        l_corrected = l_channel + lift
        
        lab[:, :, 0] = np.clip(l_corrected, 0, 255).astype(np.uint8)
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    def _reduce_highlights(self, img: np.ndarray) -> np.ndarray:
        """Reduce specular highlights."""
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l_channel = lab[:, :, 0].astype(np.float32)
        
        highlight_mask = (l_channel > 220).astype(np.float32)
        highlight_mask = cv2.GaussianBlur(highlight_mask, (7, 7), 2)
        
        pull = (l_channel - 200) * highlight_mask * 0.4
        l_corrected = l_channel - pull
        
        lab[:, :, 0] = np.clip(l_corrected, 0, 255).astype(np.uint8)
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    def _correct_white_balance(self, img: np.ndarray) -> np.ndarray:
        """Simple gray-world white balance correction."""
        result = img.astype(np.float32)
        avg_b, avg_g, avg_r = [np.mean(result[:,:,i]) for i in range(3)]
        avg_all = (avg_b + avg_g + avg_r) / 3
        
        if avg_b > 0: result[:,:,0] *= avg_all / avg_b
        if avg_g > 0: result[:,:,1] *= avg_all / avg_g
        if avg_r > 0: result[:,:,2] *= avg_all / avg_r
        
        # Blend 50% with original to avoid over-correction
        blended = result * 0.5 + img.astype(np.float32) * 0.5
        return np.clip(blended, 0, 255).astype(np.uint8)


class ReflectanceExtractor:
    """Extract illumination-invariant reflectance."""
    
    def extract_reflectance(self, face: np.ndarray) -> Dict[str, Any]:
        result = {'reflectance': None, 'illumination': None}
        if face is None or face.size == 0:
            return result
        try:
            gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY).astype(np.float32) + 1
            log_img = np.log(gray)
            illumination = cv2.GaussianBlur(log_img, (0, 0), 20)
            reflectance = log_img - illumination
            reflectance = cv2.normalize(reflectance, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            result['reflectance'] = reflectance
            result['illumination'] = cv2.normalize(illumination, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        except Exception:
            pass
        return result
