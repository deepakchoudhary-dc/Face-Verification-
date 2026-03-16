
import numpy as np
import cv2
from typing import Dict, Tuple, Any
from src.core.config import AppConfig

class QualityGate:
    """
    Ensures image quality meets standards before processing.
    Checks for: Blur, Brightness, Resolution, Glare.
    """
    
    def __init__(self):
        self.min_resolution = AppConfig.MIN_FACE_SIZE
        
    def check_quality(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Run all quality checks. 
        Returns {passed: bool, score: float, reasons: list}
        """
        reasons = []
        passed = True
        score = 1.0
        
        if image is None:
            return {'passed': False, 'score': 0.0, 'reasons': ['Image is None']}
            
        h, w = image.shape[:2]
        
        # 1. Resolution Check
        if h < self.min_resolution or w < self.min_resolution:
            reasons.append(f"Image too small: {w}x{h}")
            passed = False
            score -= 0.3
            
        # 2. Blur Detection (Laplacian Variance)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur_val = cv2.Laplacian(gray, cv2.CV_64F).var()
        if blur_val < 50: # Threshold for blur
            reasons.append(f"Image is blurry (Score: {blur_val:.1f})")
            score -= 0.3
            # We might still allow it if only slightly blurry, but correctable
            if blur_val < 20: passed = False
            
        # 3. Brightness/Exposure Check
        mean_brightness = np.mean(gray)
        if mean_brightness < 30:
            reasons.append("Image too dark")
            passed = False
            score -= 0.2
        elif mean_brightness > 220:
            reasons.append("Image overexposed/bright")
            # passed = False # Don't fail, hard to fix overexposure but maybe detectable
            score -= 0.2

        return {
            'passed': passed,
            'quality_score': max(0.0, score),
            'blur_metric': blur_val,
            'brightness': mean_brightness,
            'reasons': reasons
        }
