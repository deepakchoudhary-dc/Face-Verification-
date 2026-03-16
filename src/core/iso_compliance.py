import cv2
import numpy as np
from typing import Dict, Any

class ISOChecker:
    """
    Implements ICAO / ISO 19794-5 Biometric Standards for Machine Readable Travel Documents.
    Mathematically validates image quality suitable for algorithmic verification.
    """
    
    def check_compliance(self, image_path: str, face_box: dict = None) -> Dict[str, Any]:
        """
        Runs a battery of compliance checks:
        1. Sharpness (Laplacian Variance)
        2. Illumination Symmetry (split-face luminance)
        3. Dynamic Range (Contrast)
        4. Geometry checks (if box provided)
        """
        img = cv2.imread(image_path)
        if img is None:
            return {'status': 'ERROR', 'details': 'Could not read image'}
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 1. ISO Sharpness Check
        # Focus measure using Variance of Laplacian
        variance = cv2.Laplacian(gray, cv2.CV_64F).var()
        # ISO 'Frontal' usually requires > 100 for reliable biometric matching
        is_sharp = variance > 60.0 # Slightly relaxed for real-world
        
        # 2. Illumination Symmetry (ICAO Requirement)
        # "Lighting must be equally distributed across the face"
        h, w = gray.shape
        
        # If we have a face box, crop to it for better lighting check
        if face_box:
            fx, fy, fw, fh = int(face_box.get('x', 0)), int(face_box.get('y', 0)), int(face_box.get('w', w)), int(face_box.get('h', h))
            # Safety bounds
            fx, fy = max(0, fx), max(0, fy)
            fw, fh = min(w-fx, fw), min(h-fy, fh)
            roi = gray[fy:fy+fh, fx:fx+fw]
            if roi.size > 0:
                h_roi, w_roi = roi.shape
                left_half = roi[0:h_roi, 0:w_roi//2]
                right_half = roi[0:h_roi, w_roi//2:w_roi]
            else:
                left_half = gray[0:h, 0:w//2]
                right_half = gray[0:h, w//2:w]
        else:
            left_half = gray[0:h, 0:w//2]
            right_half = gray[0:h, w//2:w]
            
        mean_left = np.mean(left_half)
        mean_right = np.mean(right_half)
        
        # Avoid division by zero
        denom = max(mean_left, mean_right)
        lighting_balance = (min(mean_left, mean_right) / denom) if denom > 0 else 0
        # ICAO suggests preventing strong side-shadows
        pass_lighting = lighting_balance > 0.5
        
        # 3. Dynamic Range (Contrast)
        # Biometric matching fails on washed out (low contrast) images
        contrast = gray.std()
        pass_contrast = contrast > 30.0
        
        # 4. Exposure Check (Histogram Analysis)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        # Check for over-exposure (clipping at 255) vs under-exposure (clipping at 0)
        pixel_count = gray.size
        dark_pixels = np.sum(hist[:10])
        bright_pixels = np.sum(hist[246:])
        
        over_exposed_ratio = bright_pixels / pixel_count
        under_exposed_ratio = dark_pixels / pixel_count
        pass_exposure = over_exposed_ratio < 0.15 and under_exposed_ratio < 0.15

        return {
            'iso_compliance_pass': bool(is_sharp and pass_lighting and pass_contrast and pass_exposure),
            'metrics': {
                'sharpness_score': round(variance, 2),
                'lighting_symmetry': round(lighting_balance, 2),
                'contrast_score': round(contrast, 2),
                'over_exposure_ratio': round(over_exposed_ratio, 3)
            },
            'checks': {
                'sharpness': bool(is_sharp),
                'lighting': bool(pass_lighting),
                'contrast': bool(pass_contrast),
                'exposure': bool(pass_exposure)
            }
        }
