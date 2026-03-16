
import cv2
import numpy as np

class ImageEnhancer:
    """
    Standardized image enhancement pipeline for face recognition pre-processing.
    """
    
    @staticmethod
    def apply_clahe(image: np.ndarray, clip_limit=3.0, tile_size=(8,8)) -> np.ndarray:
        """
        Contrast Limited Adaptive Histogram Equalization.
        Best for fixing lighting variations on faces.
        """
        try:
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
            cl = clahe.apply(l)
            
            limg = cv2.merge((cl, a, b))
            return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        except Exception:
            return image

    @staticmethod
    def adjust_gamma(image: np.ndarray, gamma=1.0) -> np.ndarray:
        """
        Non-linear brightness adjustment provided for dark images.
        """
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255
            for i in np.arange(0, 256)]).astype("uint8")
        return cv2.LUT(image, table)
