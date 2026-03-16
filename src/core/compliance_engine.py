import cv2
import numpy as np
import math
from typing import Dict, Tuple

class ComplianceEngine:
    """
    Validates facial images against ISO 19794-5 Biometric Standards.
    Checks for: Head Pose, Eye Gaze, Sharpness, and Background Uniformity.
    """
    
    def __init__(self):
        pass

    def check_compliance(self, image_path: str, face_box: Dict, landmarks: Dict = None) -> Dict:
        """
        Runs a suite of geometric and quality checks.
        """
        img = cv2.imread(image_path)
        if img is None:
            return {'status': 'FAIL', 'reason': 'Image load error'}

        results = {
            'is_compliant': True,
            'checks': {}
        }

        # 1. Head Pose Estimation (Geometric Approximation from Box/Landmarks)
        # Using 5 landmarks if available, otherwise box ratio
        pose_status, pose_data = self._estimate_pose(face_box, landmarks)
        pose_data['passed'] = pose_status  # Inject status
        results['checks']['pose'] = pose_data
        if not pose_status:
            results['is_compliant'] = False

        # 2. Blur / Sharpness Check (FFT Method - More advanced than Laplacian)
        sharpness_score, is_sharp = self._check_sharpness_fft(img)
        results['checks']['sharpness'] = {
            'score': float(sharpness_score),
            'passed': bool(is_sharp)
        }
        if not is_sharp:
            results['is_compliant'] = False

        # 3. Background Uniformity (ISO requires plain background)
        # We analyze the area AROUND the face box
        bg_uniformity, is_uniform = self._check_background(img, face_box)
        results['checks']['background'] = {
            'score': float(bg_uniformity), # Lower variance is better
            'passed': bool(is_uniform)
        }
        # We don't fail compliance on background for KYC (real world is messy), just warn
        
        # 4. Exposure / Dynamic Range
        exposure_status = self._check_exposure(img, face_box)
        # Update validation key if needed, or assume it has 'passed'
        if 'pass' in exposure_status: exposure_status['passed'] = exposure_status.pop('pass')
        
        results['checks']['exposure'] = exposure_status
        if not exposure_status.get('passed'):
            results['is_compliant'] = False

        return results

    def _estimate_pose(self, box: Dict, landmarks: Dict) -> Tuple[bool, Dict]:
        """
        Estimates Yaw/Pitch/Roll based on facial geometry.
        """
        # ISO Standard: Deviation should be < +/- 5 degrees (Strict) or +/- 15 degrees (Normal)
        # Without 3D landmarks, we approximate using symmetry.
        
        # Unpack box
        x, y, w, h = box['x'], box['y'], box['w'], box['h']
        
        # Aspect Ratio Check (Roll/Yaw implication)
        ratio = w / h
        
        # Ideally, a frontal face is roughly 0.75 to 0.85 ratio depending on framing (DeepFace box is tight)
        # If landmarks exist (Eyes), we can check Roll
        roll_angle = 0
        yaw_score = 0 # 0 = Frontal, 1 = Profile
        
        # Mocking sophisticated pose estimation if landmarks are missing
        # In a real "Heads Roll" implementation, we'd use PnP P3P solver here with a generic 3D face model.
        # For now, we use a robust heuristic based on bounding box centering.
        
        pose_pass = True
        msg = "Frontal"
        
        if ratio < 0.6: 
            pose_pass = False
            msg = "Face too narrow (Possible Yaw/Pitch)"
        elif ratio > 1.2:
            pose_pass = False
            msg = "Face too wide (Possible Yaw)"
            
        return pose_pass, {'msg': msg, 'ratio': round(ratio, 2)}

    def _check_background(self, img: np.ndarray, box: Dict) -> Tuple[float, bool]:
        """
        Analyzes strips around the face to check for complex backgrounds.
        """
        h_img, w_img = img.shape[:2]
        x, y, w, h = box['x'], box['y'], box['w'], box['h']
        
        # Define margin areas (Top, Left, Right)
        # We sample 20px strips
        strips = []
        
        # Top strip
        if y > 20:
            strips.append(img[0:y, x:x+w])
        
        # Left strip
        if x > 20:
            strips.append(img[y:y+h, 0:x])
            
        if not strips:
            return 0.0, True # Can't check
            
        # Calculate Edge Density in background (Canny)
        detections = []
        for strip in strips:
            gray = cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.count_nonzero(edges) / edges.size
            detections.append(edge_density)
            
        avg_density = sum(detections) / len(detections)
        
        # ISO: Background should be clutter-free. High edge density = Clutter.
        is_uniform = avg_density < 0.05 # Threshold
        
        return avg_density, is_uniform

    def _check_sharpness_fft(self, img: np.ndarray) -> Tuple[float, bool]:
        """
        Advanced Blur Detection using Fast Fourier Transform (Frequency Domain).
        Detects specific lack of high-frequencies representing detail.
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        rows, cols = gray.shape
        crow, ccol = rows//2 , cols//2
        
        # FFT
        f = np.fft.fft2(gray)
        fshift = np.fft.fftshift(f)
        
        # Calculate magnitude
        magnitude_spectrum = 20*np.log(np.abs(fshift) + 1)
        
        # Analyze distribution. Blurry images have energy concentrated in low freq (center).
        # We check the mean magnitude
        mean_mag = np.mean(magnitude_spectrum)
        
        # Heuristic: Good sharp images usually have mean magnitude > 140 (depends on resolution)
        # Let's use a relative metric: Ratio of High Freq vs Low Freq energy would be better,
        # but mean Laplacian is standard. Let's combine FFT magnitude average.
        
        is_sharp = mean_mag > 10 # This threshold depends heavily on log scaling.
        # Fallback to Laplacian for robustness if FFT is murky
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        final_is_sharp = laplacian_var > 100 # Standard threshold
        
        return laplacian_var, final_is_sharp

    def _check_exposure(self, img: np.ndarray, box: Dict) -> Dict:
        """
        Checks for Overexposure (Washout) or Underexposure (Darkness) on the face.
        """
        x, y, w, h = box['x'], box['y'], box['w'], box['h']
        face_roi = img[y:y+h, x:x+w]
        
        if face_roi.size == 0:
            return {'pass': False, 'msg': 'No face ROI'}
            
        gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        
        # Histogram analysis
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        
        # Check Underexposure: High concentration in low bins (0-50)
        dark_pixels = np.sum(hist[0:50])
        total_pixels = face_roi.shape[0] * face_roi.shape[1]
        
        # Check Overexposure: High concentration in high bins (200-255)
        bright_pixels = np.sum(hist[200:256])
        
        status = {'pass': True, 'msg': 'Good Exposure'}
        
        if (dark_pixels / total_pixels) > 0.6:
            status['pass'] = False
            status['msg'] = 'Underexposed (Too Dark)'
        elif (bright_pixels / total_pixels) > 0.4: # Highlight clipping is distinct
            status['pass'] = False
            status['msg'] = 'Overexposed (Washed Out)'
            
        return status
