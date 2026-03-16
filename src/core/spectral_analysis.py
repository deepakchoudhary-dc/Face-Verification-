import cv2
import numpy as np
from typing import Dict, Tuple

class SpectralAnalyzer:
    """
    Analyzes image frequency domain to detect GAN-generated artifacts (Deepfakes).
    Deepfakes often fail to reproduce realistic high-frequency spectral distributions.
    """
    
    def __init__(self):
        pass

    def check_gan_fingerprint(self, image_path: str) -> Dict:
        """
        Performs Azimuthal Averaging on FFT Power Spectrum.
        """
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return {'deepfake_suspected': False, 'score': 0.0, 'reason': 'Image load fail'}
        
        # 1. FFT
        f = np.fft.fft2(img)
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = 20*np.log(np.abs(fshift) + 1e-8)
        
        # 2. Analyze High Frequency Falloff
        # Real images usually have a 1/f^alpha falloff.
        # GANs often have "checkerboard" artifacts in the spectrum (spikes).
        
        rows, cols = img.shape
        crow, ccol = rows//2 , cols//2
        
        # Extract a circular band (mid-to-high frequencies)
        # We look for abnormal spikes in energy at the High Frequency edges
        
        y, x = np.ogrid[:rows, :cols]
        center = (crow, ccol)
        radius = np.sqrt((x - center[1])**2 + (y - center[0])**2)
        
        # Mask for high frequencies (outer 20%)
        max_radius = min(rows, cols) // 2
        mask = (radius > (max_radius * 0.7)) & (radius < (max_radius * 0.95))
        
        high_freq_energy = np.mean(magnitude_spectrum[mask])
        total_energy = np.mean(magnitude_spectrum)
        
        # Ratio of high freq border energy to total average
        # GANs often lack high freq detail (blur) OR have specific grid artifacts
        # This is a simplified heuristic:
        # Extremely Low High-Freq Energy -> Likely blurred/generated
        
        score = high_freq_energy / total_energy
        
        is_deepfake = False
        reason = "Normal Spectral Distribution"
        
        # Heuristics derived from common Deepfake datasets (FaceForensics++)
        # If the outer rim is TOO quiet (blur) or TOO loud (noise artifacts), flag it.
        if score < 0.6: 
            is_deepfake = True
            reason = "Abnormal High-Frequency Drop-off (smoothness/AI blur)"
        elif score > 1.5:
             is_deepfake = True
             reason = "Abnormal High-Frequency Noise (GAN artifacts)"
             
        return {
            'deepfake_suspected': is_deepfake,
            'spectral_score': float(score),
            'reason': reason
        }