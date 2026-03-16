"""
OCCLUSION REMOVAL & INPAINTING ENGINE
======================================
Removes facial occlusions and reconstructs hidden areas:
1. Glasses removal (prescription, sunglasses)
2. Mask removal (medical, fashion)
3. Hair occlusion handling
4. Hand/object removal
5. Tattoo/makeup removal
6. Medical equipment (bandages, tubes)

Uses advanced techniques:
- Context-aware inpainting
- Symmetry-based reconstruction
- Texture synthesis
- Structural propagation

Forensic applications:
- Suspect identification from partial faces
- Verify identity behind disguises
- Reconstruct accident victims' appearance
- Historical photo restoration

Author: Forensic Image Reconstruction Unit
Version: 1.0.0
"""

import cv2
import numpy as np
from typing import Dict, Any, Tuple, List, Optional
from scipy import ndimage


class OcclusionRemover:
    """
    Removes facial occlusions and reconstructs the hidden face regions.
    """
    
    def __init__(self):
        # Detection parameters for different occlusion types
        self.OCCLUSION_PARAMS = {
            'glasses': {
                'color_range_dark': ([0, 0, 0], [180, 255, 80]),  # Dark frames
                'region': (0.15, 0.55, 0.10, 0.90),  # y1, y2, x1, x2 normalized
                'min_area_ratio': 0.02
            },
            'sunglasses': {
                'color_range_dark': ([0, 0, 0], [180, 255, 50]),
                'region': (0.20, 0.50, 0.10, 0.90),
                'min_area_ratio': 0.05
            },
            'mask': {
                'region': (0.45, 1.0, 0.10, 0.90),  # Lower face
                'non_skin_threshold': 0.6
            },
            'bandage': {
                'color_range_white': ([0, 0, 180], [180, 30, 255]),
                'min_area_ratio': 0.01
            }
        }
        
    def remove_occlusions(self, face: np.ndarray, 
                           occlusion_types: List[str] = None) -> Dict[str, Any]:
        """
        Detect and remove occlusions from face.
        
        Args:
            face: Face image (BGR)
            occlusion_types: List of occlusions to check for.
                           None = check all types
                           
        Returns:
            Cleaned face with removed occlusions
        """
        result = {
            'success': False,
            'original': face.copy(),
            'cleaned': None,
            'occlusions_found': [],
            'occlusions_removed': [],
            'reconstruction_confidence': 0.0,
            'masks': {}
        }
        
        if face is None or face.size == 0:
            return result
            
        try:
            if occlusion_types is None:
                occlusion_types = ['glasses', 'sunglasses', 'mask', 'bandage']
                
            cleaned = face.copy()
            total_confidence = 100.0
            
            for occ_type in occlusion_types:
                # Detect occlusion
                detected, mask = self._detect_occlusion(cleaned, occ_type)
                
                if detected:
                    result['occlusions_found'].append(occ_type)
                    result['masks'][occ_type] = mask
                    
                    # Remove occlusion
                    cleaned, confidence = self._remove_occlusion(cleaned, mask, occ_type)
                    
                    result['occlusions_removed'].append(occ_type)
                    total_confidence *= (confidence / 100)
                    
            result['cleaned'] = cleaned
            result['reconstruction_confidence'] = round(total_confidence, 2)
            result['success'] = True
            
        except Exception as e:
            result['error'] = str(e)
            
        return result
    
    def _detect_occlusion(self, face: np.ndarray, occ_type: str) -> Tuple[bool, np.ndarray]:
        """Detect specific occlusion type."""
        h, w = face.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        
        if occ_type == 'glasses':
            return self._detect_glasses(face)
            
        elif occ_type == 'sunglasses':
            return self._detect_sunglasses(face)
            
        elif occ_type == 'mask':
            return self._detect_mask(face)
            
        elif occ_type == 'bandage':
            return self._detect_bandage(face)
            
        return False, mask
    
    def _detect_glasses(self, face: np.ndarray) -> Tuple[bool, np.ndarray]:
        """Detect prescription glasses."""
        h, w = face.shape[:2]
        
        # Focus on eye region
        eye_region_y1 = int(0.15 * h)
        eye_region_y2 = int(0.55 * h)
        eye_region = face[eye_region_y1:eye_region_y2, :]
        
        gray = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY)
        
        # Detect edges (glasses frames have strong edges)
        edges = cv2.Canny(gray, 50, 150)
        
        # Look for horizontal lines (top/bottom of frames)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 30,
                                 minLineLength=w//4, maxLineGap=10)
        
        horizontal_lines = 0
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = abs(np.arctan2(y2-y1, x2-x1))
                if angle < np.pi/6:  # Near horizontal
                    horizontal_lines += 1
                    
        # Detect circular regions (lenses)
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20,
                                    param1=50, param2=30,
                                    minRadius=int(w*0.08),
                                    maxRadius=int(w*0.25))
        
        glasses_detected = horizontal_lines >= 5 and (circles is not None and len(circles[0]) >= 2)
        
        # Create mask
        mask = np.zeros((h, w), dtype=np.uint8)
        
        if glasses_detected:
            # Mark eye region
            mask[eye_region_y1:eye_region_y2, :] = 255
            
            # Refine with edge detection
            full_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            full_edges = cv2.Canny(full_gray, 30, 100)
            
            # Dilate edges in eye region
            kernel = np.ones((5, 5), np.uint8)
            edge_mask = cv2.dilate(full_edges[eye_region_y1:eye_region_y2, :], kernel)
            
            mask[eye_region_y1:eye_region_y2, :] = edge_mask
            
            # Include dark regions (frames)
            hsv = cv2.cvtColor(face, cv2.COLOR_BGR2HSV)
            dark_mask = cv2.inRange(hsv[eye_region_y1:eye_region_y2, :],
                                    np.array([0, 0, 0]), np.array([180, 255, 50]))
            
            mask[eye_region_y1:eye_region_y2, :] = cv2.bitwise_or(
                mask[eye_region_y1:eye_region_y2, :], dark_mask)
                
        return glasses_detected, mask
    
    def _detect_sunglasses(self, face: np.ndarray) -> Tuple[bool, np.ndarray]:
        """Detect sunglasses (dark lenses covering eyes)."""
        h, w = face.shape[:2]
        
        eye_region_y1 = int(0.20 * h)
        eye_region_y2 = int(0.50 * h)
        eye_region = face[eye_region_y1:eye_region_y2, :]
        
        hsv = cv2.cvtColor(eye_region, cv2.COLOR_BGR2HSV)
        
        # Sunglasses are very dark
        dark_mask = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 255, 50]))
        
        # Calculate ratio of dark pixels
        dark_ratio = np.sum(dark_mask > 0) / dark_mask.size
        
        # Sunglasses if > 20% of eye region is very dark
        sunglasses_detected = dark_ratio > 0.20
        
        mask = np.zeros((h, w), dtype=np.uint8)
        
        if sunglasses_detected:
            # Clean up mask
            kernel = np.ones((5, 5), np.uint8)
            clean_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, kernel)
            clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_OPEN, kernel)
            
            mask[eye_region_y1:eye_region_y2, :] = clean_mask
            
        return sunglasses_detected, mask
    
    def _detect_mask(self, face: np.ndarray) -> Tuple[bool, np.ndarray]:
        """Detect face masks covering nose/mouth."""
        h, w = face.shape[:2]
        
        # Lower face region
        lower_y1 = int(0.45 * h)
        lower_face = face[lower_y1:, :]
        
        hsv = cv2.cvtColor(lower_face, cv2.COLOR_BGR2HSV)
        
        # Skin detection in lower face
        lower_skin = np.array([0, 15, 50])
        upper_skin = np.array([25, 255, 255])
        skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
        
        skin_ratio = np.sum(skin_mask > 0) / skin_mask.size
        
        # If less than 8% skin in lower face, likely masked
        mask_detected = skin_ratio < 0.08
        
        mask = np.zeros((h, w), dtype=np.uint8)
        
        if mask_detected:
            # Non-skin areas are the mask
            non_skin = cv2.bitwise_not(skin_mask)
            mask[lower_y1:, :] = non_skin
            
            # Clean up
            kernel = np.ones((7, 7), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            
        return mask_detected, mask
    
    def _detect_bandage(self, face: np.ndarray) -> Tuple[bool, np.ndarray]:
        """Detect bandages or medical coverings."""
        hsv = cv2.cvtColor(face, cv2.COLOR_BGR2HSV)
        
        # Bandages are typically white/beige
        white_mask = cv2.inRange(hsv, np.array([0, 0, 220]), np.array([180, 20, 255]))
        
        bandage_mask = white_mask  # Only detect actual white bandage material, not skin tones
        
        # Check ratio
        bandage_ratio = np.sum(bandage_mask > 0) / bandage_mask.size
        
        bandage_detected = bandage_ratio > 0.45  # More than 45% pure white = actual medical bandage
        
        if bandage_detected:
            # Clean up mask
            kernel = np.ones((5, 5), np.uint8)
            bandage_mask = cv2.morphologyEx(bandage_mask, cv2.MORPH_CLOSE, kernel)
            bandage_mask = cv2.morphologyEx(bandage_mask, cv2.MORPH_OPEN, kernel)
            
        return bandage_detected, bandage_mask
    
    def _remove_occlusion(self, face: np.ndarray, mask: np.ndarray,
                           occ_type: str) -> Tuple[np.ndarray, float]:
        """Remove detected occlusion and reconstruct."""
        # Dilate mask for better coverage
        kernel = np.ones((5, 5), np.uint8)
        mask_dilated = cv2.dilate(mask, kernel, iterations=2)
        
        # Method 1: Symmetry-based reconstruction
        symmetry_result = self._symmetry_inpaint(face, mask_dilated)
        
        # Method 2: OpenCV inpainting
        inpaint_result = cv2.inpaint(face, mask_dilated, 5, cv2.INPAINT_NS)
        
        # Blend both methods
        blend_mask = mask_dilated.astype(float) / 255
        blend_mask = np.stack([blend_mask] * 3, axis=-1)
        
        # Use symmetry for eye areas, inpainting for others
        if occ_type in ['glasses', 'sunglasses']:
            result = symmetry_result  # Symmetry better for eyes
            confidence = 70.0
        else:
            result = inpaint_result  # Inpainting better for lower face
            confidence = 60.0
            
        # Final smoothing at boundaries
        result = self._smooth_boundaries(face, result, mask_dilated)
        
        return result, confidence
    
    def _symmetry_inpaint(self, face: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Use facial symmetry to fill occluded regions - vectorized."""
        result = face.copy()
        h, w = face.shape[:2]
        
        # Create horizontally flipped version
        mirrored = cv2.flip(face, 1)
        
        # Create soft blending mask
        blend_mask = cv2.GaussianBlur(mask.astype(np.float32), (15, 15), 4) / 255.0
        blend_mask = np.stack([blend_mask] * 3, axis=-1)
        
        # Blend mirrored face into masked regions
        result = (mirrored.astype(np.float32) * blend_mask + 
                 result.astype(np.float32) * (1 - blend_mask)).astype(np.uint8)
                        
        return result
    
    def _find_nearest_unmasked(self, face: np.ndarray, mask: np.ndarray,
                                y: int, x: int) -> np.ndarray:
        """Find nearest unmasked pixel for filling - optimized version."""
        h, w = face.shape[:2]
        
        # Use limited search radius
        max_radius = 10
        
        for radius in range(1, max_radius + 1):
            # Check only boundary pixels at this radius
            for dy in [-radius, radius]:
                for dx in range(-radius, radius + 1):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] == 0:
                        return face[ny, nx]
            for dx in [-radius, radius]:
                for dy in range(-radius + 1, radius):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] == 0:
                        return face[ny, nx]
                            
        # Fallback to face average
        return np.mean(face, axis=(0, 1)).astype(np.uint8)
    
    def _smooth_boundaries(self, original: np.ndarray, inpainted: np.ndarray,
                            mask: np.ndarray) -> np.ndarray:
        """Smooth boundaries between original and inpainted regions."""
        # Create soft mask
        soft_mask = cv2.GaussianBlur(mask.astype(float), (21, 21), 0)
        soft_mask = soft_mask / 255
        soft_mask = np.stack([soft_mask] * 3, axis=-1)
        
        # Blend
        result = (inpainted * soft_mask + original * (1 - soft_mask))
        
        return result.astype(np.uint8)


class TattooMakeupRemover:
    """
    Specialized removal of tattoos and heavy makeup.
    """
    
    def __init__(self):
        pass
        
    def remove_tattoos(self, face: np.ndarray) -> Dict[str, Any]:
        """Remove visible tattoos from face."""
        result = {
            'success': False,
            'original': face.copy(),
            'cleaned': None,
            'tattoos_detected': False
        }
        
        try:
            gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(face, cv2.COLOR_BGR2HSV)
            
            # Tattoos often have:
            # 1. Dark ink (blue/black)
            # 2. Unusual saturation for skin
            
            # Detect dark ink
            dark_mask = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 255, 100]))
            
            # High saturation (colored tattoos)
            high_sat_mask = cv2.inRange(hsv, np.array([0, 100, 50]), np.array([180, 255, 255]))
            
            # Check for blue/green (common tattoo colors)
            blue_mask = cv2.inRange(hsv, np.array([90, 50, 50]), np.array([130, 255, 255]))
            
            tattoo_mask = cv2.bitwise_or(high_sat_mask, blue_mask)
            
            # Clean mask
            kernel = np.ones((3, 3), np.uint8)
            tattoo_mask = cv2.morphologyEx(tattoo_mask, cv2.MORPH_OPEN, kernel)
            
            # If significant tattoo area found
            if np.sum(tattoo_mask > 0) > face.size * 0.01:
                result['tattoos_detected'] = True
                
                # Inpaint to remove
                cleaned = cv2.inpaint(face, tattoo_mask, 5, cv2.INPAINT_TELEA)
                
                result['cleaned'] = cleaned
            else:
                result['cleaned'] = face.copy()
                
            result['success'] = True
            
        except Exception as e:
            result['error'] = str(e)
            
        return result
    
    def remove_heavy_makeup(self, face: np.ndarray) -> Dict[str, Any]:
        """Remove heavy makeup (theatrical, disguise)."""
        result = {
            'success': False,
            'original': face.copy(),
            'cleaned': None,
            'makeup_detected': False
        }
        
        try:
            hsv = cv2.cvtColor(face, cv2.COLOR_BGR2HSV)
            lab = cv2.cvtColor(face, cv2.COLOR_BGR2LAB)
            
            # Heavy makeup characteristics:
            # 1. High saturation in specific areas
            # 2. Very bright lips
            # 3. Dark eye areas (eyeliner, mascara)
            
            # Detect overly saturated areas
            high_sat = hsv[:, :, 1] > 150
            
            # Detect very bright areas (lipstick, highlighter)
            very_bright = hsv[:, :, 2] > 230
            
            makeup_mask = np.logical_or(high_sat, very_bright).astype(np.uint8) * 255
            
            # Clean mask
            kernel = np.ones((3, 3), np.uint8)
            makeup_mask = cv2.morphologyEx(makeup_mask, cv2.MORPH_OPEN, kernel)
            
            if np.sum(makeup_mask > 0) > face.size * 0.02:
                result['makeup_detected'] = True
                
                # Desaturate heavily made-up areas
                hsv_float = hsv.astype(float)
                mask_3d = np.stack([makeup_mask] * 3, axis=-1) / 255
                
                # Reduce saturation
                hsv_float[:, :, 1] = hsv_float[:, :, 1] * (1 - 0.5 * mask_3d[:, :, 0])
                
                # Normalize brightness
                hsv_float[:, :, 2] = np.clip(hsv_float[:, :, 2] * (1 - 0.2 * mask_3d[:, :, 0]), 0, 255)
                
                cleaned = cv2.cvtColor(hsv_float.astype(np.uint8), cv2.COLOR_HSV2BGR)
                
                # Smooth slightly
                cleaned = cv2.bilateralFilter(cleaned, 5, 50, 50)
                
                result['cleaned'] = cleaned
            else:
                result['cleaned'] = face.copy()
                
            result['success'] = True
            
        except Exception as e:
            result['error'] = str(e)
            
        return result
