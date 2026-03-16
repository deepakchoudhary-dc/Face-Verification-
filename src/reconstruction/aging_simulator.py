"""
FACIAL AGING & DE-AGING SIMULATION ENGINE
==========================================
Simulates how a face ages or de-ages by:
1. Modeling biological aging patterns
2. Adding/removing wrinkles
3. Adjusting skin texture
4. Modifying facial structure (sagging, volume loss)
5. Simulating hair changes

Used in forensics for:
- Age-progressed images of missing children
- Identifying criminals after years
- Verifying identity across decades
- Cold case investigations

Scientific basis:
- Skin elasticity degradation models
- Bone structure changes with age
- Fat distribution patterns
- Wrinkle formation physics

Author: Forensic Age Simulation Lab
Version: 1.0.0
"""

import cv2
import numpy as np
from typing import Dict, Any, Tuple, Optional
from scipy import ndimage


class AgingSimulator:
    """
    Simulates facial aging and de-aging for forensic identification.
    """
    
    def __init__(self):
        # Aging characteristics by decade
        self.AGING_PROFILES = {
            'wrinkle_progression': {
                20: 0.0,   # Base - no wrinkles
                30: 0.1,   # First forehead lines
                40: 0.25,  # Crow's feet, deeper forehead
                50: 0.45,  # Nasolabial folds, under-eye
                60: 0.65,  # Full face wrinkles
                70: 0.85,  # Deep wrinkles, skin folding
                80: 1.0    # Maximum aging
            },
            'skin_texture': {
                20: 1.0,   # Smooth, even
                30: 0.95,  # Very slight texture
                40: 0.85,  # Some roughness
                50: 0.70,  # Visible pores, texture
                60: 0.55,  # Rougher texture
                70: 0.40,  # Significant texture
                80: 0.30   # Very textured
            },
            'skin_elasticity': {
                20: 1.0,   # Perfect elasticity
                30: 0.95,
                40: 0.85,
                50: 0.70,
                60: 0.55,
                70: 0.40,
                80: 0.25
            },
            'volume_loss': {
                20: 0.0,   # Full volume
                30: 0.05,
                40: 0.15,
                50: 0.30,
                60: 0.50,
                70: 0.70,
                80: 0.85
            }
        }
        
        # Wrinkle patterns
        self.WRINKLE_REGIONS = {
            'forehead': {'y_range': (0, 0.25), 'direction': 'horizontal', 'intensity': 1.0},
            'glabella': {'y_range': (0.15, 0.30), 'x_range': (0.35, 0.65), 'direction': 'vertical', 'intensity': 0.8},
            'crow_feet_l': {'y_range': (0.25, 0.45), 'x_range': (0, 0.25), 'direction': 'radial', 'intensity': 0.9},
            'crow_feet_r': {'y_range': (0.25, 0.45), 'x_range': (0.75, 1.0), 'direction': 'radial', 'intensity': 0.9},
            'nasolabial_l': {'y_range': (0.45, 0.75), 'x_range': (0.2, 0.4), 'direction': 'diagonal', 'intensity': 1.0},
            'nasolabial_r': {'y_range': (0.45, 0.75), 'x_range': (0.6, 0.8), 'direction': 'diagonal', 'intensity': 1.0},
            'under_eye_l': {'y_range': (0.35, 0.45), 'x_range': (0.15, 0.40), 'direction': 'horizontal', 'intensity': 0.7},
            'under_eye_r': {'y_range': (0.35, 0.45), 'x_range': (0.60, 0.85), 'direction': 'horizontal', 'intensity': 0.7},
            'marionette_l': {'y_range': (0.70, 0.90), 'x_range': (0.2, 0.4), 'direction': 'vertical', 'intensity': 0.8},
            'marionette_r': {'y_range': (0.70, 0.90), 'x_range': (0.6, 0.8), 'direction': 'vertical', 'intensity': 0.8},
            'lip_lines': {'y_range': (0.60, 0.75), 'x_range': (0.30, 0.70), 'direction': 'vertical', 'intensity': 0.6}
        }
        
    def simulate_aging(self, face: np.ndarray, current_age: int, 
                        target_age: int) -> Dict[str, Any]:
        """
        Age or de-age a face to target age.
        
        Args:
            face: Face image (BGR)
            current_age: Estimated current age
            target_age: Desired output age
            
        Returns:
            Aged/de-aged face with transformation details
        """
        result = {
            'success': False,
            'original': face.copy(),
            'simulated': None,
            'current_age': current_age,
            'target_age': target_age,
            'transformations': [],
            'confidence': 0.0
        }
        
        if face is None or face.size == 0:
            return result
            
        try:
            age_diff = target_age - current_age
            
            if age_diff > 0:
                # AGING
                simulated = self._apply_aging(face, current_age, target_age)
                result['transformations'].append('Wrinkle Addition')
                result['transformations'].append('Skin Texture Aging')
                result['transformations'].append('Volume Loss Simulation')
            else:
                # DE-AGING
                simulated = self._apply_deaging(face, current_age, target_age)
                result['transformations'].append('Wrinkle Removal')
                result['transformations'].append('Skin Smoothing')
                result['transformations'].append('Volume Restoration')
                
            result['simulated'] = simulated
            result['success'] = True
            
            # Confidence based on age difference
            # More accurate for smaller age differences
            age_diff_abs = abs(age_diff)
            if age_diff_abs <= 10:
                result['confidence'] = 90.0
            elif age_diff_abs <= 20:
                result['confidence'] = 75.0
            elif age_diff_abs <= 30:
                result['confidence'] = 60.0
            else:
                result['confidence'] = 45.0
                
        except Exception as e:
            result['error'] = str(e)
            
        return result
    
    def _apply_aging(self, face: np.ndarray, current_age: int, 
                      target_age: int) -> np.ndarray:
        """Apply aging effects to face.
        
        Uses properly scaled effects for 512px 3D renders.
        """
        aged = face.copy()
        h, w = face.shape[:2]
        scale = min(h, w)
        
        # Get aging parameters
        current_profile = self._get_age_profile(current_age)
        target_profile = self._get_age_profile(target_age)
        
        # Calculate deltas
        wrinkle_delta = target_profile['wrinkle'] - current_profile['wrinkle']
        texture_delta = current_profile['texture'] - target_profile['texture']
        volume_delta = target_profile['volume'] - current_profile['volume']
        
        # Overall strength = max of all deltas, minimum 0.3
        strength = max(wrinkle_delta, texture_delta, volume_delta, 0.3)
        
        # 1. GEOMETRIC AGING (sagging, hollowing) — scale-aware 
        aged = self._age_geometry(aged, strength)
        
        # 2. ADD WRINKLES (works on renders since we ADD texture)
        if wrinkle_delta > 0:
            aged = self._add_wrinkles(aged, max(wrinkle_delta, 0.3))
            
        # 3. DEGRADE SKIN TEXTURE
        if texture_delta > 0:
            aged = self._degrade_skin_texture(aged, max(texture_delta, 0.25))
        
        # 4. COLOR AGING (desaturation, yellowing, age spots)
        aged = self._age_color(aged, strength, target_age)
        
        return aged
    
    def _age_geometry(self, face: np.ndarray, strength: float) -> np.ndarray:
        """Forward-aging geometric warp: sagging jowls, hollow cheeks, drooping brow.
        Scale-aware: produces 15-30px peak displacement on 512px images."""
        h, w = face.shape[:2]
        scale = min(h, w)
        
        y_grid, x_grid = np.mgrid[0:h, 0:w].astype(np.float32)
        y_norm = y_grid / h
        x_norm = x_grid / w
        
        disp_x = np.zeros((h, w), dtype=np.float32)
        disp_y = np.zeros((h, w), dtype=np.float32)
        
        # --- Jowl sagging: downward pull in lower face ---
        jowl_y = np.clip((y_norm - 0.58) / 0.42, 0, 1)
        jowl_center = np.exp(-((x_norm - 0.5)**2 / 0.12))
        disp_y += strength * scale * 0.06 * jowl_y * jowl_center  # ~15px peak
        
        # --- Cheek hollowing: pull inward ---
        for cx, sign in [(0.32, +1), (0.68, -1)]:  # pull toward center
            cheek = np.exp(-((y_norm - 0.48)**2 / 0.025 + (x_norm - cx)**2 / 0.012))
            disp_x += sign * strength * scale * 0.025 * cheek
        
        # --- Under-eye sag: droop downward ---
        for cx in [0.35, 0.65]:
            eye_bag = np.exp(-((y_norm - 0.38)**2 / 0.006 + (x_norm - cx)**2 / 0.01))
            disp_y += strength * scale * 0.02 * eye_bag  # ~5px droop
        
        # --- Brow droop ---
        brow = np.exp(-((y_norm - 0.20)**2 / 0.006 + (x_norm - 0.5)**2 / 0.08))
        disp_y += strength * scale * 0.015 * brow
        
        # --- Temple hollowing ---
        for cx, sign in [(0.15, +1), (0.85, -1)]:
            temple = np.exp(-((y_norm - 0.22)**2 / 0.01 + (x_norm - cx)**2 / 0.008))
            disp_x += sign * strength * scale * 0.015 * temple
        
        # --- Nasolabial fold deepening ---
        for cx, sign in [(0.35, +1), (0.65, -1)]:
            naso = np.exp(-((y_norm - 0.58)**2 / 0.01 + (x_norm - cx)**2 / 0.004))
            disp_x += sign * strength * scale * 0.012 * naso
            disp_y += strength * scale * 0.008 * naso
        
        map_x = (x_grid + disp_x).astype(np.float32)
        map_y = (y_grid + disp_y).astype(np.float32)
        
        result = cv2.remap(face, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        
        # Add subtle shadow in hollowed areas (not too much — was causing darkening)
        shadow = np.exp(-((y_norm - 0.45)**2 / 0.03)) * np.maximum(0, 0.3 - np.abs(x_norm - 0.5))
        shadow_effect = 1.0 - shadow * strength * 0.15  # reduced from 0.35
        shadow_effect = np.stack([shadow_effect] * 3, axis=-1)
        result = (result.astype(np.float32) * shadow_effect).clip(0, 255).astype(np.uint8)
        
        return result
    
    def _age_color(self, face: np.ndarray, strength: float, target_age: int) -> np.ndarray:
        """Aging color: desaturation, yellowing, age spots, contrast loss.
        NO arbitrary brightness changes."""
        # Skin mask
        hsv = cv2.cvtColor(face, cv2.COLOR_BGR2HSV)
        skin_mask = ((hsv[:, :, 0] < 25) & (hsv[:, :, 1] > 15) & (hsv[:, :, 2] > 40)).astype(np.float32)
        skin_mask = cv2.GaussianBlur(skin_mask, (15, 15), 5)
        
        # --- Desaturation (old skin loses color) ---
        hsv_f = hsv.astype(np.float32)
        desat = 1.0 - strength * 0.35  # up to 35% less saturated
        old_s = hsv_f[:, :, 1].copy()
        hsv_f[:, :, 1] = np.clip(old_s * desat * skin_mask + old_s * (1 - skin_mask), 0, 255)
        result = cv2.cvtColor(hsv_f.astype(np.uint8), cv2.COLOR_HSV2BGR)
        
        # --- Yellow/sallow shift via LAB ---
        lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB).astype(np.float32)
        # b+ = more yellow, a- = less red (sallow skin)
        lab[:, :, 2] = np.clip(lab[:, :, 2] + strength * 5.0 * skin_mask, 0, 255)  # yellow
        lab[:, :, 1] = np.clip(lab[:, :, 1] - strength * 2.0 * skin_mask, 0, 255)  # less pink
        # L channel untouched
        
        # --- Add color noise/uneven pigmentation ---
        h, w = face.shape[:2]
        color_noise = np.random.RandomState(42).normal(0, strength * 4.0, (h//4, w//4)).astype(np.float32)
        color_noise = cv2.resize(color_noise, (w, h), interpolation=cv2.INTER_LINEAR)
        color_noise = cv2.GaussianBlur(color_noise, (11, 11), 3)
        lab[:, :, 1] = np.clip(lab[:, :, 1] + color_noise * skin_mask, 0, 255)
        lab[:, :, 2] = np.clip(lab[:, :, 2] + color_noise * 0.7 * skin_mask, 0, 255)
        
        result = cv2.cvtColor(lab.clip(0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
        
        # --- Age spots ---
        result = self._add_age_spots(result, target_age)
        
        return result
    
    def _apply_deaging(self, face: np.ndarray, current_age: int,
                        target_age: int) -> np.ndarray:
        """Apply de-aging effects to face.
        
        Key design: these images are 3D renders, NOT photographs.
        3D renders have minimal wrinkles already, so wrinkle-detection
        approaches fail. Instead we use:
        1. Aggressive geometric warping (face lift, cheek fill) — MOST VISIBLE
        2. Direct skin smoothing (blanket HF reduction) — visible
        3. Color vitality (saturation + pink warmth) — visible
        """
        deaged = face.copy()
        
        # Get profiles — used to compute age_delta as a 0-1 scale factor
        current_profile = self._get_age_profile(current_age)
        target_profile = self._get_age_profile(target_age)
        
        # age_factor: 0 = no change, ~0.35 = 20yr de-age from 76, ~0.55 = 30yr
        age_factor = current_profile['wrinkle'] - target_profile['wrinkle']
        vol_factor = current_profile['volume'] - target_profile['volume']
        
        # Keep de-aging identity-safe. Stronger deltas should still remain
        # conservative enough that the face stays recognizable.
        strength = min(max(age_factor * 0.85, vol_factor * 0.75, 0.18), 0.45)
        
        # 1. GEOMETRIC RESHAPE (biggest visible impact)
        deaged = self._deage_geometry(deaged, strength)
        
        # 2. SKIN SMOOTHING (blanket high-frequency detail removal)
        deaged = self._deage_skin(deaged, strength)
        
        # 3. COLOR VITALITY (saturation + warmth, NO brightness)
        deaged = self._deage_color(deaged, strength, target_age)
        
        return deaged
    
    def _get_age_profile(self, age: int) -> Dict:
        """Interpolate aging parameters for given age."""
        profile = {}
        
        for key, values in self.AGING_PROFILES.items():
            ages = sorted(values.keys())
            
            # Find bracketing ages
            lower_age = max([a for a in ages if a <= age], default=ages[0])
            upper_age = min([a for a in ages if a >= age], default=ages[-1])

            # Normalize key: 'wrinkle_progression' -> 'wrinkle',
            #   'skin_texture' -> 'texture', 'skin_elasticity' -> 'elasticity',
            #   'volume_loss' -> 'volume'
            short_key = (
                key.replace('_progression', '')
                .replace('skin_', '')
                .replace('_loss', '')
            )
            
            if lower_age == upper_age:
                profile[short_key] = values[lower_age]
            else:
                # Linear interpolation
                t = (age - lower_age) / (upper_age - lower_age)
                value = values[lower_age] + t * (values[upper_age] - values[lower_age])
                profile[short_key] = value
                
        return profile
    
    def _add_wrinkles(self, face: np.ndarray, intensity: float) -> np.ndarray:
        """Add realistic wrinkles using edge-guided texture synthesis."""
        result = face.copy()
        h, w = face.shape[:2]
        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        
        # Detect existing skin structure with Gabor filters at wrinkle angles
        wrinkle_composite = np.zeros((h, w), dtype=np.float32)
        
        for theta in [0, np.pi/6, np.pi/4, np.pi/3, np.pi/2, 2*np.pi/3, 5*np.pi/6]:
            kern = cv2.getGaborKernel((21, 21), 3.0, theta, 8.0, 0.5, 0, ktype=cv2.CV_32F)
            response = cv2.filter2D(gray.astype(np.float32), cv2.CV_32F, kern)
            wrinkle_composite += np.abs(response)
        
        # Normalize
        wrinkle_composite = cv2.normalize(wrinkle_composite, None, 0, 1, cv2.NORM_MINMAX)
        
        # Region-based wrinkle masks with smooth gradients
        region_mask = np.zeros((h, w), dtype=np.float32)
        
        for region_name, params in self.WRINKLE_REGIONS.items():
            y1 = int(params['y_range'][0] * h)
            y2 = int(params['y_range'][1] * h)
            x1 = int(params.get('x_range', (0, 1))[0] * w)
            x2 = int(params.get('x_range', (0, 1))[1] * w)
            
            # Create smooth gradient mask for this region
            mask = np.zeros((h, w), dtype=np.float32)
            mask[y1:y2, x1:x2] = params['intensity']
            mask = cv2.GaussianBlur(mask, (31, 31), 10)
            region_mask = np.maximum(region_mask, mask)
        
        # Combine: wrinkle pattern follows skin structure in wrinkle-prone regions
        wrinkle_layer = wrinkle_composite * region_mask * intensity
        
        # Apply prominent darkening along wrinkle lines (2x stronger)
        wrinkle_dark = 1.0 - wrinkle_layer * 0.50
        wrinkle_dark = np.stack([wrinkle_dark] * 3, axis=-1)
        result = (result.astype(np.float32) * wrinkle_dark).clip(0, 255).astype(np.uint8)
        
        # Add strong wrinkle texture via noise in wrinkle regions
        noise = np.random.normal(0, 5.0 * intensity, (h, w)).astype(np.float32)  # 2x noise
        noise *= region_mask
        noise = cv2.GaussianBlur(noise, (3, 3), 0.8)
        
        for c in range(3):
            result[:, :, c] = np.clip(result[:, :, c].astype(np.float32) + noise, 0, 255).astype(np.uint8)
        
        return result
    
    def _generate_horizontal_wrinkles(self, w: int, h: int, intensity: float) -> np.ndarray:
        """Legacy placeholder - wrinkles now use Gabor-based synthesis."""
        return np.zeros((h, w), dtype=np.float32)
    
    def _generate_vertical_wrinkles(self, w: int, h: int, intensity: float) -> np.ndarray:
        """Legacy placeholder."""
        return np.zeros((h, w), dtype=np.float32)
    
    def _generate_radial_wrinkles(self, w: int, h: int, intensity: float) -> np.ndarray:
        """Legacy placeholder."""
        return np.zeros((h, w), dtype=np.float32)
    
    def _generate_diagonal_wrinkles(self, w: int, h: int, intensity: float) -> np.ndarray:
        """Legacy placeholder."""
        return np.zeros((h, w), dtype=np.float32)
    
    def _degrade_skin_texture(self, face: np.ndarray, intensity: float) -> np.ndarray:
        """Realistic skin texture aging - enlarged pores, roughness, translucency loss."""
        h, w = face.shape[:2]
        result = face.copy()
        
        # Create multi-frequency texture noise (pore enlargement) - 2x stronger
        # Fine texture (pores)
        fine_noise = np.random.normal(0, 4.0 * intensity, (h, w)).astype(np.float32)
        fine_noise = cv2.GaussianBlur(fine_noise, (3, 3), 0.5)
        
        # Medium texture (skin roughness)
        medium_noise = np.random.normal(0, 7.0 * intensity, (h // 2, w // 2)).astype(np.float32)
        medium_noise = cv2.resize(medium_noise, (w, h), interpolation=cv2.INTER_LINEAR)
        medium_noise = cv2.GaussianBlur(medium_noise, (5, 5), 1.5)
        
        # Coarse texture (large-scale unevenness) - NEW
        coarse_noise = np.random.normal(0, 5.0 * intensity, (h // 4, w // 4)).astype(np.float32)
        coarse_noise = cv2.resize(coarse_noise, (w, h), interpolation=cv2.INTER_LINEAR)
        coarse_noise = cv2.GaussianBlur(coarse_noise, (9, 9), 3)
        
        combined = fine_noise * 0.4 + medium_noise * 0.35 + coarse_noise * 0.25
        
        # Apply only in skin-colored regions (avoid eyes/hair)
        hsv = cv2.cvtColor(face, cv2.COLOR_BGR2HSV)
        skin_mask = ((hsv[:, :, 0] < 25) & (hsv[:, :, 1] > 20) & (hsv[:, :, 2] > 40)).astype(np.float32)
        skin_mask = cv2.GaussianBlur(skin_mask, (15, 15), 5)
        combined *= skin_mask
        
        for c in range(3):
            result[:, :, c] = np.clip(result[:, :, c].astype(np.float32) + combined, 0, 255).astype(np.uint8)
        
        return result
    
    def _simulate_volume_loss(self, face: np.ndarray, intensity: float) -> np.ndarray:
        """Simulate aggressive facial volume loss: hollow cheeks, sagging jowls, temple recession."""
        h, w = face.shape[:2]
        
        # Create coordinate grids
        y_grid, x_grid = np.mgrid[0:h, 0:w].astype(np.float32)
        y_norm = y_grid / h
        x_norm = x_grid / w
        
        # Strong cheek hollowing: pull inward in mid-face (3x stronger)
        cheek_mask = np.exp(-((y_norm - 0.5)**2 / 0.04 + (x_norm - 0.5)**2 / 0.08))
        hollow_x = intensity * 14 * (x_norm - 0.5) * cheek_mask  # Was 4, now 14
        
        # Strong jowl sagging: downward pull in lower face (3x stronger)
        jowl_mask = np.maximum(0, (y_norm - 0.60)) * np.exp(-((x_norm - 0.5)**2 / 0.07))
        sag_y = intensity * 16 * jowl_mask  # Was 5, now 16
        
        # Temple recession: visible inward pull at temples (was effectively 0)
        temple_l = np.exp(-((y_norm - 0.2)**2 / 0.02 + (x_norm - 0.15)**2 / 0.01))
        temple_r = np.exp(-((y_norm - 0.2)**2 / 0.02 + (x_norm - 0.85)**2 / 0.01))
        temple_x = intensity * 6 * (temple_l * (x_norm - 0.15) + temple_r * (x_norm - 0.85))
        temple_y = intensity * 4 * (temple_l + temple_r)  # Slight downward
        
        # Nasolabial fold deepening: pull skin around nose-mouth 
        naso_l = np.exp(-((y_norm - 0.55)**2 / 0.02 + (x_norm - 0.3)**2 / 0.008))
        naso_r = np.exp(-((y_norm - 0.55)**2 / 0.02 + (x_norm - 0.7)**2 / 0.008))
        sag_y += intensity * 5 * (naso_l + naso_r)
        
        map_y = (y_grid + sag_y + temple_y).astype(np.float32)
        map_x = (x_grid + hollow_x + temple_x).astype(np.float32)
        
        result = cv2.remap(face, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        
        # Strong shadow in hollow areas for 3D depth (3x stronger)
        shadow = np.exp(-((y_norm - 0.45)**2 / 0.03)) * np.maximum(0, 0.3 - np.abs(x_norm - 0.5))
        shadow_effect = 1.0 - shadow * intensity * 0.45  # Was 0.15, now 0.45
        shadow_effect = np.stack([shadow_effect] * 3, axis=-1)
        result = (result.astype(np.float32) * shadow_effect).clip(0, 255).astype(np.uint8)
        
        return result
    
    def _add_age_spots(self, face: np.ndarray, target_age: int) -> np.ndarray:
        """Add realistic age spots (solar lentigines) - visible brown spots."""
        if target_age < 45:
            return face
            
        result = face.copy()
        h, w = face.shape[:2]
        
        # Create skin mask to only add spots on skin
        hsv = cv2.cvtColor(face, cv2.COLOR_BGR2HSV)
        skin_mask = ((hsv[:, :, 0] < 25) & (hsv[:, :, 1] > 15) & (hsv[:, :, 2] > 40)).astype(np.float32)
        
        # More spots, scaling faster with age (2x more)
        num_spots = int((target_age - 45) / 1.5)  # Was /3, now /1.5
        np.random.seed(42)  # Deterministic for reproducibility
        
        spot_layer = np.zeros((h, w), dtype=np.float32)
        
        for _ in range(num_spots):
            x = np.random.randint(w // 5, 4 * w // 5)
            y = np.random.randint(h // 5, 4 * h // 5)
            
            # Only place on skin
            if skin_mask[min(y, h-1), min(x, w-1)] < 0.5:
                continue
            
            # Larger spots with more visible edges
            radius = np.random.randint(max(3, h // 50), max(6, h // 20))  # Bigger
            
            # Create spot with Gaussian falloff
            y_grid, x_grid = np.ogrid[-radius*2:radius*2+1, -radius*2:radius*2+1]
            aspect = np.random.uniform(0.7, 1.3)
            dist = np.sqrt((x_grid * aspect)**2 + y_grid**2).astype(np.float32)
            spot = np.exp(-dist**2 / (2 * (radius * 0.6)**2))
            spot *= np.random.uniform(0.25, 0.50)  # 2x intensity
            
            # Place on layer
            sy = max(0, y - radius*2)
            sx = max(0, x - radius*2)
            ey = min(h, y + radius*2 + 1)
            ex = min(w, x + radius*2 + 1)
            
            spot_cropped = spot[:ey-sy, :ex-sx]
            spot_layer[sy:ey, sx:ex] = np.maximum(spot_layer[sy:ey, sx:ex], spot_cropped)
        
        # Apply spots as darkening with warm brown tone
        spot_layer *= skin_mask
        spot_3ch = np.stack([spot_layer] * 3, axis=-1)
        
        # Brown tint: darken blue channel more, red less
        tint = np.array([1.0, 0.85, 0.7])  # BGR - blue darkens most
        spot_effect = 1.0 - spot_3ch * tint
        
        result = (result.astype(np.float32) * spot_effect).clip(0, 255).astype(np.uint8)
        
        return result
    
    def _overall_aging_adjustments(self, face: np.ndarray, target_age: int) -> np.ndarray:
        """Apply overall aging color/contrast adjustments - visible skin aging effects."""
        # Convert to LAB for better color manipulation
        lab = cv2.cvtColor(face, cv2.COLOR_BGR2LAB)
        
        # 1. Noticeable desaturation with age (2x stronger)
        hsv = cv2.cvtColor(face, cv2.COLOR_BGR2HSV)
        desaturation = max(0, (target_age - 30) / 60)  # Steeper curve, starts earlier
        desaturation = min(desaturation, 0.40)  # Cap at 40% desaturation
        hsv[:, :, 1] = (hsv[:, :, 1] * (1 - desaturation)).astype(np.uint8)
        result = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        # 2. Warmth shift for older skin (more visible yellow undertone)
        if target_age > 45:
            warm_factor = min((target_age - 45) / 50, 0.30)  # 2x stronger, starts earlier
            result = result.astype(np.float32)
            result[:, :, 2] = np.clip(result[:, :, 2] * (1 + warm_factor * 0.8), 0, 255)  # Red
            result[:, :, 1] = np.clip(result[:, :, 1] * (1 + warm_factor * 0.5), 0, 255)  # Green (yellow)
            result = result.astype(np.uint8)
        
        # 3. Contrast reduction for aged skin (more uniform, duller)
        if target_age > 55:
            contrast_reduction = min((target_age - 55) / 40, 0.25)  # Stronger
            mean = np.mean(result, axis=(0, 1), keepdims=True)
            result = (result.astype(np.float32) * (1 - contrast_reduction) + mean * contrast_reduction)
            result = result.clip(0, 255).astype(np.uint8)
        
        # 4. Slight darkness / reduced luminance for very old
        if target_age > 70:
            dark_factor = min((target_age - 70) / 30, 0.12)
            result = (result.astype(np.float32) * (1 - dark_factor)).clip(0, 255).astype(np.uint8)
        
        return result
    
    # =========================================================================
    #  DE-AGING METHODS (3 consolidated, properly scaled for 512px renders)
    # =========================================================================
    
    def _deage_geometry(self, face: np.ndarray, strength: float) -> np.ndarray:
        """Massive geometric face-lift: jowl lift, cheek fill, brow raise, jaw tighten.
        
        For a 512px image we need 15-30px peak displacement to be visible.
        strength=0.35 should give ~15px peak, strength=0.55 should give ~25px.
        """
        h, w = face.shape[:2]
        scale = min(h, w)  # pixel-scale factor
        
        y_grid, x_grid = np.mgrid[0:h, 0:w].astype(np.float32)
        y_norm = y_grid / h
        x_norm = x_grid / w
        
        disp_x = np.zeros((h, w), dtype=np.float32)
        disp_y = np.zeros((h, w), dtype=np.float32)
        
        # --- 1. JOWL LIFT (most visible) ---
        # Strong upward pull in lower 40% of face, centered horizontally
        # Wider gaussian so it affects most of the lower face
        jowl_y = np.clip((y_norm - 0.58) / 0.42, 0, 1)  # ramp from 0 at y=0.58 to 1 at bottom
        jowl_center = np.exp(-((x_norm - 0.5)**2 / 0.12))  # wide center band
        jowl_lift = jowl_y * jowl_center
        disp_y += -strength * scale * 0.045 * jowl_lift
        
        # --- 2. CHEEK FILL (outward push from center) ---
        # Two cheek blobs that push outward
        for cx, sign in [(0.32, -1), (0.68, +1)]:
            cheek = np.exp(-((y_norm - 0.48)**2 / 0.025 + (x_norm - cx)**2 / 0.012))
            disp_x += sign * strength * scale * 0.018 * cheek
        
        # --- 3. UNDER-EYE LIFT ---
        for cx in [0.35, 0.65]:
            eye_bag = np.exp(-((y_norm - 0.38)**2 / 0.006 + (x_norm - cx)**2 / 0.01))
            disp_y += -strength * scale * 0.015 * eye_bag
        
        # --- 4. BROW LIFT ---
        brow = np.exp(-((y_norm - 0.20)**2 / 0.006 + (x_norm - 0.5)**2 / 0.08))
        disp_y += -strength * scale * 0.012 * brow
        
        # --- 5. JAW TIGHTENING (pull sides of jaw inward for sharper jawline) ---
        jaw_y = np.exp(-((y_norm - 0.75)**2 / 0.015))
        for cx, sign in [(0.20, +1), (0.80, -1)]:
            jaw_x = np.exp(-((x_norm - cx)**2 / 0.01))
            disp_x += sign * strength * scale * 0.012 * jaw_y * jaw_x
        
        # --- 6. NASOLABIAL FOLD FILL ---
        for cx, sign in [(0.35, -1), (0.65, +1)]:
            naso = np.exp(-((y_norm - 0.58)**2 / 0.01 + (x_norm - cx)**2 / 0.004))
            disp_x += sign * strength * scale * 0.010 * naso
            disp_y += -strength * scale * 0.006 * naso
        
        # Apply warp
        map_x = (x_grid + disp_x).astype(np.float32)
        map_y = (y_grid + disp_y).astype(np.float32)
        
        result = cv2.remap(face, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        return result
    
    def _deage_skin(self, face: np.ndarray, strength: float) -> np.ndarray:
        """Smooth skin via BLANKET high-frequency detail reduction.
        
        Does NOT try to detect wrinkles (useless on 3D renders).
        Instead: two-scale frequency separation with direct reduction.
        Preserves mean luminance exactly (no brightening).
        """
        h, w = face.shape[:2]
        face_f = face.astype(np.float32)
        
        # Skin mask: protect eyes, eyebrows, lips, hair
        hsv = cv2.cvtColor(face, cv2.COLOR_BGR2HSV)
        skin_mask = ((hsv[:, :, 0] < 25) & (hsv[:, :, 1] > 15) & (hsv[:, :, 2] > 40)).astype(np.float32)
        skin_mask = cv2.GaussianBlur(skin_mask, (15, 15), 5)
        
        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        edge_mask = cv2.Canny(gray, 60, 140).astype(np.float32) / 255.0
        edge_mask = cv2.GaussianBlur(edge_mask, (9, 9), 2)
        skin_smooth_mask = np.clip(skin_mask * (1.0 - edge_mask * 0.55), 0.0, 1.0)

        # --- Scale 1: Medium freq (wrinkles, creases) ---
        r1 = max(7, int(min(h, w) * 0.035)) | 1
        base1 = cv2.GaussianBlur(face_f, (r1, r1), r1 / 2.5)
        detail1 = face_f - base1
        
        reduction1 = min(0.18 + strength * 0.35, 0.45)
        mask1 = np.stack([skin_smooth_mask * reduction1] * 3, axis=-1)
        new_detail1 = detail1 * (1.0 - mask1)
        
        # --- Scale 2: Fine freq (pores, micro-texture) ---
        r2 = max(3, int(min(h, w) * 0.012)) | 1
        base2_of_detail = cv2.GaussianBlur(new_detail1, (r2, r2), r2 / 2.5)
        fine_detail = new_detail1 - base2_of_detail
        
        reduction2 = min(0.08 + strength * 0.18, 0.22)
        mask2 = np.stack([skin_smooth_mask * reduction2] * 3, axis=-1)
        new_fine = fine_detail * (1.0 - mask2)
        
        # Reconstruct
        result = base1 + base2_of_detail + new_fine

        # Re-inject a small amount of true micro-detail so de-aging does not
        # collapse into plastic skin.
        micro_base = cv2.GaussianBlur(face_f, (0, 0), 1.0)
        micro_detail = face_f - micro_base
        reinject = np.stack([skin_smooth_mask * (0.10 + (0.18 - strength * 0.12))] * 3, axis=-1)
        result += micro_detail * reinject
        
        # Ensure NO brightness shift: force mean luminance to match original
        orig_mean = np.mean(face_f, axis=(0, 1), keepdims=True)
        result_mean = np.mean(result, axis=(0, 1), keepdims=True)
        result = result - (result_mean - orig_mean)  # exact correction
        
        return result.clip(0, 255).astype(np.uint8)
    
    def _deage_color(self, face: np.ndarray, strength: float, target_age: int) -> np.ndarray:
        """Youthful color: richer saturation + warmer skin tone + even chrominance.
        
        ZERO brightness modification. Only changes color/saturation.
        """
        # Skin mask for targeted changes
        hsv = cv2.cvtColor(face, cv2.COLOR_BGR2HSV)
        skin_mask = ((hsv[:, :, 0] < 25) & (hsv[:, :, 1] > 15) & (hsv[:, :, 2] > 40)).astype(np.float32)
        skin_mask = cv2.GaussianBlur(skin_mask, (15, 15), 5)
        
        # --- 1. BOOST SATURATION on skin (richer, healthier color) ---
        hsv_f = hsv.astype(np.float32)
        sat_boost = 1.0 + strength * 0.18
        old_s = hsv_f[:, :, 1].copy()
        hsv_f[:, :, 1] = np.clip(old_s * sat_boost * skin_mask + old_s * (1 - skin_mask), 0, 255)
        # V channel UNTOUCHED
        result = cv2.cvtColor(hsv_f.astype(np.uint8), cv2.COLOR_HSV2BGR)
        
        # --- 2. WARM SKIN TONE via LAB a/b channels ---
        lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB).astype(np.float32)
        
        # a+ = more red/pink (healthy young skin)  
        warmth_a = strength * 2.5 * skin_mask
        lab[:, :, 1] = np.clip(lab[:, :, 1] + warmth_a, 0, 255)
        
        # b+ = slightly warmer/golden (vs blue/gray old skin)
        warmth_b = strength * 1.5 * skin_mask
        lab[:, :, 2] = np.clip(lab[:, :, 2] + warmth_b, 0, 255)
        
        # --- 3. EVEN CHROMINANCE (reduce color splotchiness, age spots) ---
        # Only on a/b channels, NOT L (luminance stays unchanged)
        a_ch = lab[:, :, 1]
        b_ch = lab[:, :, 2]
        a_smooth = cv2.GaussianBlur(a_ch, (21, 21), 7)
        b_smooth = cv2.GaussianBlur(b_ch, (21, 21), 7)
        
        even_strength = min(strength * 0.25, 0.18)
        mask_even = skin_mask * even_strength
        lab[:, :, 1] = a_ch * (1 - mask_even) + a_smooth * mask_even
        lab[:, :, 2] = b_ch * (1 - mask_even) + b_smooth * mask_even
        
        # L channel COMPLETELY UNTOUCHED
        result = cv2.cvtColor(lab.clip(0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
        return result
    
    # Legacy wrappers (keep for backward compat, redirect to new methods)
    def _remove_wrinkles(self, face: np.ndarray, intensity: float) -> np.ndarray:
        return self._deage_skin(face, intensity)

    def _smooth_skin(self, face: np.ndarray, intensity: float) -> np.ndarray:
        return face  # already done in _deage_skin

    def _restore_volume(self, face: np.ndarray, intensity: float) -> np.ndarray:
        return self._deage_geometry(face, intensity)

    def _even_skin_tone(self, face: np.ndarray) -> np.ndarray:
        return face  # handled in _deage_color

    def _add_youthful_glow(self, face: np.ndarray, target_age: int) -> np.ndarray:
        return face  # handled in _deage_color
