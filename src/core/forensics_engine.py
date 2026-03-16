import cv2
import numpy as np
import os
from typing import Tuple, Dict

class ForensicsEngine:
    """
    Advanced Digital Forensics for manipulating and analyzing potential forgeries.
    Implements Error Level Analysis (ELA) and Structural Analysis.
    """
    
    def __init__(self):
        pass

    def analyze_image(self, image_path: str) -> Dict:
        """
        Run full forensic analysis on an image.
        Returns metrics and paths to generated forensic maps.
        """
        if not os.path.exists(image_path):
            return {'error': 'File not found'}

        # Load image
        original = cv2.imread(image_path)
        if original is None:
            return {'error': 'Could not read image'}

        # 1. Error Level Analysis (ELA)
        ela_map, tamper_score = self._compute_ela(image_path, original)
        
        # 2. Lighting/Structure Analysis (Glare/Blur)
        gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Glare detection (simple thresholding on saturation/value)
        hsv = cv2.cvtColor(original, cv2.COLOR_BGR2HSV)
        _, _, v = cv2.split(hsv)
        # Ratio of very bright pixels
        glare_ratio = np.sum(v > 230) / v.size

        return {
            'ela_map': ela_map,
            'tamper_score': tamper_score, # 0-1, higher is more suspicious
            'sharpness': blur_score, # < 100 usually blurry
            'glare_ratio': glare_ratio, # > 0.05 implies flash/glare
            'is_suspicious': tamper_score > 0.15 or blur_score < 50
        }

    def _compute_ela(self, path: str, original: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Computes Error Level Analysis to detect potential photoshop interpolation.
        """
        try:
            # 1. Save at 90% quality to temp
            temp_path = path + ".ela.jpg"
            cv2.imwrite(temp_path, original, [cv2.IMWRITE_JPEG_QUALITY, 90])
            
            # 2. Read back
            resaved = cv2.imread(temp_path)
            os.remove(temp_path)
            
            if resaved is None:
                return original, 0.0

            # 3. Calculate absolute difference
            diff = cv2.absdiff(original, resaved)
            
            # 4. Enhance the difference (scale up extreme values)
            # Find max diff to normalize? Or just strict scaling
            # For visualization, we amplify the noise
            scale = 10
            ela_image = cv2.scaleAdd(diff, scale, np.zeros_like(diff))
            
            # 5. Calculate "Tamper Score"
            # Logic: Uniform compression noise = good. 
            # High variance in noise clusters = bad (pasting).
            gray_ela = cv2.cvtColor(ela_image, cv2.COLOR_BGR2GRAY)
            
            # We look for clusters of high intensity
            _, thresh = cv2.threshold(gray_ela, 50, 255, cv2.THRESH_BINARY)
            non_zero = cv2.countNonZero(thresh)
            score = non_zero / gray_ela.size
            
            return ela_image, score
            
        except Exception as e:
            print(f"ELA Failed: {e}")
            return original, 0.0
