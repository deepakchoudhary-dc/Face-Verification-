"""
3D FACE GEOMETRY ESTIMATION ENGINE
===================================
Estimates 3D facial structure from 2D images:
1. Depth map estimation
2. 3D landmark positioning
3. Surface normal estimation
4. Facial plane angles
5. Shape-from-shading analysis
6. Multi-view synthesis

Why 3D analysis matters:
- 2D photos can be spoofed easily
- 3D structure is unique to individual
- Harder to fake depth information
- Better for cross-pose matching
- Detects 2D mask attacks

Techniques:
- Monocular depth estimation
- Shape-from-shading
- Facial geometry priors
- Statistical face models (3DMM)

Author: 3D Vision Research Lab
Version: 1.0.0
"""

import cv2
import numpy as np
from typing import Dict, Any, Tuple, List, Optional
from scipy import ndimage


class FaceGeometryEstimator:
    """
    Estimates 3D facial geometry from single 2D image.
    """
    
    def __init__(self):
        # Average face 3D model parameters (simplified)
        self.FACE_3D_TEMPLATE = {
            'nose_depth': 30,  # mm protrusion
            'eye_depth': -15,  # mm inset
            'cheek_curve': 40,  # mm radius
            'forehead_curve': 80,  # mm radius
            'chin_protrusion': 20  # mm
        }
        
        # Standard facial landmarks for 3D estimation
        self.LANDMARK_3D = {
            'nose_tip': (0, 0, 30),
            'nose_bridge': (0, -15, 15),
            'left_eye': (-30, -20, -10),
            'right_eye': (30, -20, -10),
            'left_ear': (-70, -10, -20),
            'right_ear': (70, -10, -20),
            'chin': (0, 50, 10),
            'forehead': (0, -50, 5)
        }
        
    def estimate_3d_geometry(self, face: np.ndarray) -> Dict[str, Any]:
        """
        Estimate 3D facial geometry from 2D image.
        
        Returns:
            Depth map, normals, and 3D measurements
        """
        result = {
            'success': False,
            'original': face.copy(),
            'depth_map': None,
            'normal_map': None,
            'face_angle': {'yaw': 0, 'pitch': 0, 'roll': 0},
            'measurements_3d': {},
            '3d_score': 0.0
        }
        
        if face is None or face.size == 0:
            return result
            
        try:
            # 1. ESTIMATE DEPTH MAP
            depth_map = self._estimate_depth(face)
            result['depth_map'] = depth_map
            
            # 2. COMPUTE SURFACE NORMALS
            normal_map = self._compute_normals(depth_map)
            result['normal_map'] = normal_map
            
            # 3. ESTIMATE FACE POSE
            pose = self._estimate_face_pose(face, depth_map)
            result['face_angle'] = pose
            
            # 4. EXTRACT 3D MEASUREMENTS
            measurements = self._extract_3d_measurements(face, depth_map)
            result['measurements_3d'] = measurements
            
            # 5. CALCULATE 3D CONFIDENCE SCORE
            score = self._calculate_3d_score(depth_map, normal_map)
            result['3d_score'] = score
            
            result['success'] = True
            
        except Exception as e:
            result['error'] = str(e)
            
        return result
    
    def _estimate_depth(self, face: np.ndarray) -> np.ndarray:
        """
        Estimate depth map using shape-from-shading and priors.
        """
        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY).astype(float)
        h, w = gray.shape
        
        # Normalize brightness
        gray = (gray - gray.min()) / (gray.max() - gray.min() + 1e-10)
        
        # 1. Initial depth from brightness (Lambertian assumption)
        # Brighter = closer (for front lighting)
        depth_brightness = gray.copy()
        
        # 2. Add face geometry prior
        depth_prior = self._create_face_depth_prior(h, w)
        
        # 3. Gradient-based refinement
        # Compute image gradients
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        # Integrate gradients for surface (simplified)
        depth_gradient = self._integrate_gradients(grad_x, grad_y)
        
        # 4. Combine methods
        depth_combined = (
            0.3 * depth_brightness +
            0.4 * depth_prior +
            0.3 * depth_gradient
        )
        
        # Normalize to 0-255 range
        depth_combined = (depth_combined - depth_combined.min()) / \
                        (depth_combined.max() - depth_combined.min() + 1e-10)
        depth_combined = (depth_combined * 255).astype(np.uint8)
        
        return depth_combined
    
    def _create_face_depth_prior(self, h: int, w: int) -> np.ndarray:
        """
        Create depth prior based on typical face shape.
        """
        depth = np.zeros((h, w), dtype=float)
        
        # Face center coordinates
        cy, cx = h // 2, w // 2
        
        # Create ellipsoid shape for face
        y_coords, x_coords = np.ogrid[:h, :w]
        
        # Base face shape (ellipsoid)
        a = w // 2  # horizontal radius
        b = h // 2  # vertical radius
        
        # Distance from center normalized
        dist = ((x_coords - cx) ** 2 / (a ** 2 + 1e-10) + 
                (y_coords - cy) ** 2 / (b ** 2 + 1e-10))
        
        # Ellipsoid depth
        depth = np.maximum(0, 1 - dist)
        depth = np.sqrt(depth)  # Sphere-like falloff
        
        # Add nose protrusion
        nose_x, nose_y = cx, int(cy * 1.1)  # Slightly below center
        nose_mask = np.exp(-((x_coords - nose_x) ** 2 + (y_coords - nose_y) ** 2) / (w * 0.1) ** 2)
        depth += nose_mask * 0.3
        
        # Eye sockets (depressions)
        left_eye_x, left_eye_y = int(cx - w * 0.2), int(cy * 0.8)
        right_eye_x, right_eye_y = int(cx + w * 0.2), int(cy * 0.8)
        
        left_eye_mask = np.exp(-((x_coords - left_eye_x) ** 2 + (y_coords - left_eye_y) ** 2) / (w * 0.08) ** 2)
        right_eye_mask = np.exp(-((x_coords - right_eye_x) ** 2 + (y_coords - right_eye_y) ** 2) / (w * 0.08) ** 2)
        
        depth -= (left_eye_mask + right_eye_mask) * 0.15
        
        # Normalize
        depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-10)
        
        return depth
    
    def _integrate_gradients(self, grad_x: np.ndarray, grad_y: np.ndarray) -> np.ndarray:
        """
        Integrate gradient field to get depth (simplified Poisson solver).
        """
        h, w = grad_x.shape
        
        # Simple cumulative sum approach
        depth_x = np.cumsum(grad_x, axis=1)
        depth_y = np.cumsum(grad_y, axis=0)
        
        # Average both integrations
        depth = (depth_x + depth_y) / 2
        
        # Normalize
        depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-10)
        
        return depth
    
    def _compute_normals(self, depth_map: np.ndarray) -> np.ndarray:
        """
        Compute surface normal map from depth map.
        """
        # Convert to float
        depth = depth_map.astype(float) / 255
        
        # Compute gradients
        grad_x = cv2.Sobel(depth, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(depth, cv2.CV_64F, 0, 1, ksize=3)
        
        # Normal = (-dz/dx, -dz/dy, 1) normalized
        h, w = depth.shape
        normals = np.zeros((h, w, 3), dtype=float)
        
        normals[:, :, 0] = -grad_x
        normals[:, :, 1] = -grad_y
        normals[:, :, 2] = 1
        
        # Normalize
        magnitude = np.sqrt(np.sum(normals ** 2, axis=2, keepdims=True))
        normals = normals / (magnitude + 1e-10)
        
        # Convert to 0-255 RGB for visualization
        normal_rgb = ((normals + 1) / 2 * 255).astype(np.uint8)
        
        return normal_rgb
    
    def _estimate_face_pose(self, face: np.ndarray, depth_map: np.ndarray) -> Dict:
        """
        Estimate face pose angles from image and depth.
        """
        pose = {'yaw': 0.0, 'pitch': 0.0, 'roll': 0.0}
        
        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        
        # YAW estimation from horizontal asymmetry
        left_half = gray[:, :w//2]
        right_half = gray[:, w//2:]
        
        left_brightness = np.mean(left_half)
        right_brightness = np.mean(right_half)
        
        # Brighter side is facing camera more
        brightness_diff = (right_brightness - left_brightness) / 128
        pose['yaw'] = round(brightness_diff * 45, 2)  # Scale to degrees
        
        # PITCH estimation from vertical distribution
        top_half = gray[:h//2, :]
        bottom_half = gray[h//2:, :]
        
        top_brightness = np.mean(top_half)
        bottom_brightness = np.mean(bottom_half)
        
        # If looking down, forehead is brighter; if up, chin is brighter
        vert_diff = (top_brightness - bottom_brightness) / 128
        pose['pitch'] = round(vert_diff * 30, 2)
        
        # ROLL estimation from eye level difference
        # Detect edges to find eye line
        edges = cv2.Canny(gray, 50, 150)
        eye_region = edges[h//4:h//2, :]
        
        # Find centroid of edge pixels in left and right halves
        left_edges = eye_region[:, :w//2]
        right_edges = eye_region[:, w//2:]
        
        left_y = np.mean(np.where(left_edges > 0)[0]) if np.any(left_edges > 0) else h//8
        right_y = np.mean(np.where(right_edges > 0)[0]) if np.any(right_edges > 0) else h//8
        
        # Roll from eye level difference
        eye_diff = (right_y - left_y) / (h // 4)
        pose['roll'] = round(np.arctan(eye_diff) * 180 / np.pi, 2)
        
        return pose
    
    def _extract_3d_measurements(self, face: np.ndarray, depth_map: np.ndarray) -> Dict:
        """
        Extract 3D facial measurements.
        """
        measurements = {
            'face_width': 0.0,
            'face_height': 0.0,
            'nose_protrusion': 0.0,
            'eye_depth': 0.0,
            'chin_protrusion': 0.0,
            'forehead_curve': 0.0
        }
        
        h, w = face.shape[:2]
        depth = depth_map.astype(float) / 255
        
        # Face dimensions (in pixels, relative)
        measurements['face_width'] = round(w, 2)
        measurements['face_height'] = round(h, 2)
        
        # Nose protrusion (max depth in nose region)
        nose_region = depth[int(0.35*h):int(0.65*h), int(0.35*w):int(0.65*w)]
        if nose_region.size > 0:
            measurements['nose_protrusion'] = round(np.max(nose_region) * 100, 2)
            
        # Eye depth (depth in eye regions)
        left_eye = depth[int(0.25*h):int(0.45*h), int(0.15*w):int(0.35*w)]
        right_eye = depth[int(0.25*h):int(0.45*h), int(0.65*w):int(0.85*w)]
        
        if left_eye.size > 0 and right_eye.size > 0:
            eye_depth_val = (np.mean(left_eye) + np.mean(right_eye)) / 2
            measurements['eye_depth'] = round(eye_depth_val * 100, 2)
            
        # Chin protrusion
        chin_region = depth[int(0.8*h):, int(0.3*w):int(0.7*w)]
        if chin_region.size > 0:
            measurements['chin_protrusion'] = round(np.max(chin_region) * 100, 2)
            
        # Forehead curve (variance in forehead depth)
        forehead = depth[:int(0.25*h), int(0.2*w):int(0.8*w)]
        if forehead.size > 0:
            measurements['forehead_curve'] = round(np.std(forehead) * 100, 2)
            
        return measurements
    
    def _calculate_3d_score(self, depth_map: np.ndarray, normal_map: np.ndarray) -> float:
        """
        Calculate confidence score for 3D estimation.
        Higher score = more 3D structure detected.
        """
        depth = depth_map.astype(float) / 255
        
        # Depth variation (real faces have significant depth variation)
        depth_var = np.var(depth)
        
        # Normal map consistency
        normals = normal_map.astype(float) / 255 * 2 - 1
        normal_consistency = np.mean(np.std(normals, axis=(0, 1)))
        
        # Score combines both
        score = (depth_var * 100 + normal_consistency * 50) / 2
        score = min(100, max(0, score))
        
        return round(score, 2)
    
    def compare_3d_geometry(self, face1: np.ndarray, face2: np.ndarray) -> Dict[str, Any]:
        """
        Compare 3D geometry of two faces.
        """
        result = {
            'match_score': 0.0,
            'depth_similarity': 0.0,
            'pose_difference': {},
            'measurement_similarity': 0.0
        }
        
        try:
            # Get 3D info for both faces
            geo1 = self.estimate_3d_geometry(face1)
            geo2 = self.estimate_3d_geometry(face2)
            
            if not geo1['success'] or not geo2['success']:
                return result
                
            # Compare depth maps (normalized correlation)
            depth1 = geo1['depth_map'].astype(float) / 255
            depth2 = cv2.resize(geo2['depth_map'], (depth1.shape[1], depth1.shape[0])).astype(float) / 255
            
            correlation = np.corrcoef(depth1.ravel(), depth2.ravel())[0, 1]
            result['depth_similarity'] = round(max(0, correlation) * 100, 2)
            
            # Compare poses
            pose_diff = {
                'yaw': abs(geo1['face_angle']['yaw'] - geo2['face_angle']['yaw']),
                'pitch': abs(geo1['face_angle']['pitch'] - geo2['face_angle']['pitch']),
                'roll': abs(geo1['face_angle']['roll'] - geo2['face_angle']['roll'])
            }
            result['pose_difference'] = pose_diff
            
            # Compare measurements
            m1 = geo1['measurements_3d']
            m2 = geo2['measurements_3d']
            
            common_keys = set(m1.keys()) & set(m2.keys())
            if common_keys:
                similarities = []
                for key in common_keys:
                    if m1[key] > 0 and m2[key] > 0:
                        sim = 1 - abs(m1[key] - m2[key]) / max(m1[key], m2[key])
                        similarities.append(max(0, sim))
                        
                result['measurement_similarity'] = round(np.mean(similarities) * 100, 2)
                
            # Overall match score
            result['match_score'] = round(
                0.5 * result['depth_similarity'] +
                0.3 * result['measurement_similarity'] +
                0.2 * (100 - sum(pose_diff.values()) / 3),
                2
            )
            
        except Exception as e:
            result['error'] = str(e)
            
        return result
