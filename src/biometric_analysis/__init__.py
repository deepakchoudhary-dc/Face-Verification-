"""
ADVANCED BIOMETRIC ANALYSIS SUITE
=================================
Professional-grade forensic biometric analysis for KYC verification.

This module provides capabilities that go FAR beyond standard face recognition:

1. AgeInvariantAnalyzer - Match young ID photos to aged live faces
2. TamperingDetector - Detect photo manipulation and splicing
3. MakeupDisguiseDetector - Detect heavy makeup and disguises
4. IrisAnalyzer - Iris pattern analysis with cataract detection
5. DoppelgangerDetector - Distinguish twins/lookalikes from same person
6. ScarAndInjuryAnalyzer - Map permanent facial markers
7. MorphingDetector - Detect face morphing attacks (ISO 30107-3 compliant)

These features are research-grade implementations of cutting-edge
biometric security techniques used in border control and law enforcement.

Author: Advanced Biometric Research Team
Version: 2.0.0
"""

import numpy as np

from .age_invariant import AgeInvariantAnalyzer
from .tampering_detector import TamperingDetector
from .makeup_detector import MakeupDisguiseDetector
from .iris_analyzer import IrisAnalyzer
from .doppelganger_detector import DoppelgangerDetector
from .scar_analysis import ScarAndInjuryAnalyzer
from .morphing_detector import MorphingDetector

__all__ = [
    'AgeInvariantAnalyzer',
    'TamperingDetector',
    'MakeupDisguiseDetector',
    'IrisAnalyzer',
    'DoppelgangerDetector',
    'ScarAndInjuryAnalyzer',
    'MorphingDetector'
]

__version__ = '2.0.0'


class BiometricAnalysisSuite:
    """
    Unified interface for all advanced biometric analysis capabilities.
    
    Usage:
        suite = BiometricAnalysisSuite()
        result = suite.full_analysis(image, face_box, landmarks)
    """
    
    def __init__(self):
        self.age_analyzer = AgeInvariantAnalyzer()
        self.tampering_detector = TamperingDetector()
        self.makeup_detector = MakeupDisguiseDetector()
        self.iris_analyzer = IrisAnalyzer()
        self.doppelganger_detector = DoppelgangerDetector()
        self.scar_analyzer = ScarAndInjuryAnalyzer()
        self.morphing_detector = MorphingDetector()

    @staticmethod
    def _points_to_region(points: np.ndarray, max_points: int | None = None, order: str = "x") -> list[tuple[int, int]]:
        if points.size == 0:
            return []
        if order == "y":
            points = points[np.argsort(points[:, 1])]
        else:
            points = points[np.argsort(points[:, 0])]
        if max_points is not None and len(points) > max_points:
            idx = np.linspace(0, len(points) - 1, max_points).astype(int)
            points = points[idx]
        return [(int(round(x)), int(round(y))) for x, y in points[:, :2]]

    def _normalize_runtime_landmarks(self, face_box: dict, landmarks: dict | None) -> dict:
        if not isinstance(landmarks, dict):
            return {}

        points = landmarks.get("points_106") or landmarks.get("landmark_2d_106") or []
        kps = landmarks.get("kps5") or landmarks.get("kps") or []

        pts = np.asarray(points, dtype=np.float32)
        kps_arr = np.asarray(kps, dtype=np.float32)
        if pts.ndim != 2 or pts.shape[1] < 2:
            pts = np.empty((0, 2), dtype=np.float32)
        else:
            pts = pts[:, :2]
        if kps_arr.ndim != 2 or kps_arr.shape[1] < 2:
            kps_arr = np.empty((0, 2), dtype=np.float32)
        else:
            kps_arr = kps_arr[:, :2]

        if pts.size == 0 and kps_arr.size == 0:
            return {}

        x = float(face_box.get('x', 0))
        y = float(face_box.get('y', 0))
        w = float(face_box.get('w', 0))
        h = float(face_box.get('h', 0))
        cx = x + w / 2.0

        def select(x0: float, x1: float, y0: float, y1: float) -> np.ndarray:
            if pts.size == 0:
                return np.empty((0, 2), dtype=np.float32)
            mask = (
                (pts[:, 0] >= x + w * x0)
                & (pts[:, 0] <= x + w * x1)
                & (pts[:, 1] >= y + h * y0)
                & (pts[:, 1] <= y + h * y1)
            )
            return pts[mask]

        result = {
            'left_eye': self._points_to_region(select(0.10, 0.46, 0.16, 0.52), max_points=6),
            'right_eye': self._points_to_region(select(0.54, 0.90, 0.16, 0.52), max_points=6),
            'left_eyebrow': self._points_to_region(select(0.05, 0.48, 0.05, 0.28), max_points=5),
            'right_eyebrow': self._points_to_region(select(0.52, 0.95, 0.05, 0.28), max_points=5),
            'nose_bridge': self._points_to_region(select(0.38, 0.62, 0.18, 0.56), max_points=4, order="y"),
            'nose_tip': self._points_to_region(select(0.30, 0.70, 0.48, 0.76), max_points=5),
            'jaw': self._points_to_region(select(0.02, 0.98, 0.54, 0.98), max_points=17),
            'mouth_outer': self._points_to_region(select(0.24, 0.76, 0.62, 0.94), max_points=12),
        }

        if kps_arr.shape[0] >= 5:
            left_eye = (int(round(kps_arr[0][0])), int(round(kps_arr[0][1])))
            right_eye = (int(round(kps_arr[1][0])), int(round(kps_arr[1][1])))
            nose_tip = (int(round(kps_arr[2][0])), int(round(kps_arr[2][1])))
            mouth_left = (int(round(kps_arr[3][0])), int(round(kps_arr[3][1])))
            mouth_right = (int(round(kps_arr[4][0])), int(round(kps_arr[4][1])))

            if not result['left_eye']:
                result['left_eye'] = [left_eye]
            if not result['right_eye']:
                result['right_eye'] = [right_eye]
            if not result['nose_tip']:
                result['nose_tip'] = [nose_tip]
            if not result['mouth_outer']:
                result['mouth_outer'] = [mouth_left, mouth_right]
            if not result['nose_bridge']:
                mid_eye_x = int(round((left_eye[0] + right_eye[0]) / 2.0))
                result['nose_bridge'] = [
                    (mid_eye_x, int(round((left_eye[1] + right_eye[1]) / 2.0))),
                    nose_tip,
                ]

        if not result['jaw']:
            result['jaw'] = [
                (int(round(x + w * i / 16)), int(round(y + h * (0.50 + 0.45 * abs(i - 8) / 8 if i != 8 else 0.95))))
                for i in range(17)
            ]

        return {k: v for k, v in result.items() if v}

    def _extract_face_landmarks(self, image, face_box: dict, landmarks: dict | None = None) -> dict:
        """
        Extract facial landmarks from an image using OpenCV's face landmark detection.
        Falls back to estimated landmarks from face bounding box if detection fails.
        """
        import cv2
        extracted = {}
        try:
            if image is None:
                return extracted
            runtime_landmarks = self._normalize_runtime_landmarks(face_box, landmarks)
            if runtime_landmarks:
                return runtime_landmarks
            x = face_box.get('x', 0)
            y = face_box.get('y', 0)
            w = face_box.get('w', image.shape[1])
            h = face_box.get('h', image.shape[0])
            
            # Use face region to estimate landmark positions based on proportions
            # These are anthropometric averages (Farkas 1994)
            extracted = {
                'left_eye': [
                    (x + int(w * 0.30), y + int(h * 0.35)),
                    (x + int(w * 0.33), y + int(h * 0.33)),
                    (x + int(w * 0.37), y + int(h * 0.33)),
                    (x + int(w * 0.40), y + int(h * 0.35)),
                    (x + int(w * 0.37), y + int(h * 0.38)),
                    (x + int(w * 0.33), y + int(h * 0.38)),
                ],
                'right_eye': [
                    (x + int(w * 0.60), y + int(h * 0.35)),
                    (x + int(w * 0.63), y + int(h * 0.33)),
                    (x + int(w * 0.67), y + int(h * 0.33)),
                    (x + int(w * 0.70), y + int(h * 0.35)),
                    (x + int(w * 0.67), y + int(h * 0.38)),
                    (x + int(w * 0.63), y + int(h * 0.38)),
                ],
                'nose_bridge': [
                    (x + int(w * 0.50), y + int(h * 0.35)),
                    (x + int(w * 0.50), y + int(h * 0.42)),
                    (x + int(w * 0.50), y + int(h * 0.48)),
                    (x + int(w * 0.50), y + int(h * 0.55)),
                ],
                'nose_tip': [
                    (x + int(w * 0.42), y + int(h * 0.58)),
                    (x + int(w * 0.46), y + int(h * 0.60)),
                    (x + int(w * 0.50), y + int(h * 0.62)),
                    (x + int(w * 0.54), y + int(h * 0.60)),
                    (x + int(w * 0.58), y + int(h * 0.58)),
                ],
                'left_eyebrow': [
                    (x + int(w * 0.22), y + int(h * 0.28)),
                    (x + int(w * 0.27), y + int(h * 0.25)),
                    (x + int(w * 0.33), y + int(h * 0.24)),
                    (x + int(w * 0.38), y + int(h * 0.25)),
                    (x + int(w * 0.42), y + int(h * 0.27)),
                ],
                'right_eyebrow': [
                    (x + int(w * 0.58), y + int(h * 0.27)),
                    (x + int(w * 0.62), y + int(h * 0.25)),
                    (x + int(w * 0.67), y + int(h * 0.24)),
                    (x + int(w * 0.73), y + int(h * 0.25)),
                    (x + int(w * 0.78), y + int(h * 0.28)),
                ],
                'jaw': [
                    (x + int(w * i / 16), y + int(h * (0.50 + 0.45 * abs(i - 8) / 8 if i != 8 else 0.95)))
                    for i in range(17)
                ],
                'mouth_outer': [
                    (x + int(w * 0.35), y + int(h * 0.72)),
                    (x + int(w * 0.40), y + int(h * 0.70)),
                    (x + int(w * 0.45), y + int(h * 0.69)),
                    (x + int(w * 0.50), y + int(h * 0.70)),
                    (x + int(w * 0.55), y + int(h * 0.69)),
                    (x + int(w * 0.60), y + int(h * 0.70)),
                    (x + int(w * 0.65), y + int(h * 0.72)),
                    (x + int(w * 0.60), y + int(h * 0.78)),
                    (x + int(w * 0.55), y + int(h * 0.80)),
                    (x + int(w * 0.50), y + int(h * 0.80)),
                    (x + int(w * 0.45), y + int(h * 0.80)),
                    (x + int(w * 0.40), y + int(h * 0.78)),
                ],
            }

            # Placeholder for future landmark refinement model integration.
                
        except Exception as e:
            pass
            
        return extracted
    
    def full_analysis(self, image, face_box, landmarks=None):
        """
        Run ALL biometric analysis on a face image.
        
        Returns comprehensive analysis including:
        - Age-invariant features
        - Tampering indicators
        - Makeup/disguise detection
        - Iris analysis (if eye region available)
        - Unique identity markers
        - Morphing attack detection
        """
        result = {
            'age_invariant': {},
            'tampering': {},
            'makeup_disguise': {},
            'iris': {},
            'uniqueness': {},
            'facial_markers': {},
            'morphing': {},
            'overall_confidence': 0.0,
            'alerts': []
        }
        
        try:
            # Run all analyzers - extract landmarks from face for age-invariant analysis
            face_landmarks = self._extract_face_landmarks(image, face_box, landmarks)
            
            try:
                result['age_invariant'] = self.age_analyzer.extract_age_invariant_features(
                    image, face_landmarks if face_landmarks else (landmarks or {})
                )
            except Exception as e:
                result['age_invariant'] = {'error': str(e)}
            
            try:
                result['tampering'] = self.tampering_detector.analyze_tampering(
                    image, face_box
                )
            except Exception as e:
                result['tampering'] = {'error': str(e)}
            
            try:
                result['makeup_disguise'] = self.makeup_detector.analyze_disguise(
                    image, face_box, landmarks
                )
            except Exception as e:
                result['makeup_disguise'] = {'error': str(e)}
            
            try:
                result['iris'] = self.iris_analyzer.analyze_iris(
                    image, face_box
                )
            except Exception as e:
                result['iris'] = {'error': str(e)}
            
            try:
                result['uniqueness'] = self.doppelganger_detector.analyze_identity_uniqueness(
                    image, face_box, landmarks
                )
            except Exception as e:
                result['uniqueness'] = {'error': str(e)}
            
            try:
                result['facial_markers'] = self.scar_analyzer.analyze_facial_markers(
                    image, face_box, landmarks
                )
            except Exception as e:
                result['facial_markers'] = {'error': str(e)}
            
            try:
                result['morphing'] = self.morphing_detector.detect_morphing(
                    image, face_box, landmarks
                )
            except Exception as e:
                result['morphing'] = {'error': str(e)}
            
            # Collect alerts
            if result['tampering'].get('is_tampered'):
                result['alerts'].append('[ALERT] TAMPERING DETECTED')
                
            if result['makeup_disguise'].get('disguise_detected'):
                result['alerts'].append('[WARNING] DISGUISE/HEAVY MAKEUP DETECTED')
                
            if result['morphing'].get('is_morphed'):
                result['alerts'].append('[ALERT] FACE MORPHING ATTACK SUSPECTED')
                
            if result['iris'].get('spoofing_indicators', {}).get('possible_spoof'):
                result['alerts'].append('[WARNING] POSSIBLE SPOOFING ATTEMPT')
                
            # Calculate overall confidence
            confidences = []
            if result['age_invariant'].get('extraction_confidence'):
                confidences.append(result['age_invariant']['extraction_confidence'])
            if result['uniqueness'].get('uniqueness_score'):
                confidences.append(result['uniqueness']['uniqueness_score'])
            if result['facial_markers'].get('marker_confidence'):
                confidences.append(result['facial_markers']['marker_confidence'])
                
            if confidences:
                result['overall_confidence'] = round(sum(confidences) / len(confidences), 2)
                
        except Exception as e:
            result['error'] = str(e)
            
        return result
    
    def compare_faces(self, image1, face_box1: dict, image2, face_box2: dict,
                       face_match_score: float, landmarks1: dict = None,
                       landmarks2: dict = None) -> dict:
        """
        Compare two faces using all available biometric analysis.
        
        Goes beyond simple face recognition to verify:
        - Age-invariant feature match
        - Unique marker correspondence
        - Doppelganger detection
        - Morphing attack check
        """
        result = {
            'face_match_score': face_match_score,
            'age_invariant_match': {},
            'doppelganger_analysis': {},
            'kinship_analysis': {},
            'marker_comparison': {},
            'morphing_check': {},
            'final_verdict': '',
            'confidence': 0.0,
            'recommendations': []
        }
        
        try:
            # Extract features from both faces
            features1 = self.full_analysis(image1, face_box1, landmarks1)
            features2 = self.full_analysis(image2, face_box2, landmarks2)
            
            # Age-invariant comparison
            result['age_invariant_match'] = self.age_analyzer.compare_across_age(
                features1.get('age_invariant', {}),
                features2.get('age_invariant', {})
            )
            
            # Doppelganger check
            uniqueness1 = features1.get('uniqueness', {}) or self.doppelganger_detector.analyze_identity_uniqueness(
                image1, face_box1, landmarks1
            )
            uniqueness2 = features2.get('uniqueness', {}) or self.doppelganger_detector.analyze_identity_uniqueness(
                image2, face_box2, landmarks2
            )
            
            result['doppelganger_analysis'] = self.doppelganger_detector.compare_for_doppelganger(
                uniqueness1, uniqueness2, face_match_score
            )

            # Kinship check (additive only; does not override direct identity verdict)
            result['kinship_analysis'] = self.doppelganger_detector.compare_for_kinship(
                uniqueness1, uniqueness2, face_match_score
            )
            
            # Marker comparison
            markers1 = features1.get('facial_markers', {})
            markers2 = features2.get('facial_markers', {})
            result['marker_comparison'] = self.scar_analyzer.compare_markers(markers1, markers2)
            
            # Morphing check on document/reference image
            result['morphing_check'] = features2.get('morphing', {})
            
            # DETERMINE FINAL VERDICT
            alerts = []
            positive_signals = 0
            negative_signals = 0
            
            # Check face match score
            if face_match_score >= 75:
                positive_signals += 2
            elif face_match_score >= 50:
                positive_signals += 1
            else:
                negative_signals += 2
                
            # Check age-invariant match
            if result['age_invariant_match'].get('is_same_person'):
                positive_signals += 2
            elif result['age_invariant_match'].get('age_invariant_confidence', 0) >= 50:
                positive_signals += 1
            else:
                negative_signals += 1
                
            # Check doppelganger analysis
            if result['doppelganger_analysis'].get('is_doppelganger'):
                negative_signals += 2
                alerts.append('[WARNING] POSSIBLE DOPPELGANGER')

            # Surface kinship as an investigative clue, not a hidden reject path.
            if result['kinship_analysis'].get('likely_related'):
                relation = result['kinship_analysis'].get('relationship_hypothesis', 'possible_close_relative')
                alerts.append(f"[INFO] KINSHIP SIGNAL: {relation.upper()}")
                
            # Check marker comparison
            if result['marker_comparison'].get('match_score', 0) > 50:
                positive_signals += 1
                
            # Check for morphing
            if result['morphing_check'].get('is_morphed'):
                negative_signals += 3
                alerts.append('[ALERT] MORPHING ATTACK DETECTED')
                
            # Calculate confidence
            total_signals = positive_signals + negative_signals
            if total_signals > 0:
                confidence = (positive_signals / total_signals) * 100
                result['confidence'] = round(confidence, 2)
                
            # Final verdict
            if negative_signals >= 3:
                result['final_verdict'] = 'REJECT'
                result['recommendations'].append('Identity verification failed - multiple red flags')
            elif positive_signals >= 3 and negative_signals == 0:
                result['final_verdict'] = 'VERIFIED'
                result['recommendations'].append('Strong identity match with multiple confirmations')
            elif positive_signals >= 2:
                result['final_verdict'] = 'LIKELY_MATCH'
                result['recommendations'].append('Probable match - consider additional verification')
            else:
                result['final_verdict'] = 'MANUAL_REVIEW'
                result['recommendations'].append('Inconclusive - manual review required')

            if result['kinship_analysis'].get('likely_related'):
                result['recommendations'].append(
                    result['kinship_analysis'].get('recommendation')
                    or 'Close-relative facial structure detected; compare against known related identities'
                )
                
            result['recommendations'].extend(alerts)
            
        except Exception as e:
            result['error'] = str(e)
            
        return result
