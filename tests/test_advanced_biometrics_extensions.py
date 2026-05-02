import unittest

import cv2
import numpy as np

from src.biometric_analysis import BiometricAnalysisSuite
from src.biometric_analysis.doppelganger_detector import DoppelgangerDetector
from src.biometric_analysis.iris_analyzer import IrisAnalyzer
from src.biometric_analysis.tampering_detector import TamperingDetector
from src.reporting.llm_analyst import LlamaForensicAnalyst


class AdvancedBiometricExtensionTests(unittest.TestCase):
    @staticmethod
    def _make_seam_image() -> np.ndarray:
        rng = np.random.default_rng(42)
        base = np.full((160, 160, 3), 128, dtype=np.int16)
        base += rng.integers(-4, 5, size=base.shape)
        image = np.clip(base, 0, 255).astype(np.uint8)

        center = image[20:140, 40:120].astype(np.int16)
        center = np.clip(center + rng.integers(-35, 36, size=center.shape), 0, 255).astype(np.uint8)
        image[20:140, 40:120] = center
        cv2.rectangle(image, (40, 20), (120, 140), (205, 205, 205), 2)
        return image

    @staticmethod
    def _make_eye_image() -> np.ndarray:
        image = np.full((120, 120, 3), 235, dtype=np.uint8)
        cv2.circle(image, (60, 60), 22, (120, 120, 120), -1)
        cv2.circle(image, (60, 60), 9, (15, 15, 15), -1)
        cv2.line(image, (18, 55), (42, 45), (30, 30, 210), 2)
        cv2.line(image, (78, 52), (102, 62), (30, 30, 210), 2)
        cv2.line(image, (20, 67), (40, 74), (40, 40, 180), 1)
        cv2.line(image, (80, 70), (98, 76), (40, 40, 180), 1)
        return image

    @staticmethod
    def _make_face_image(horizontal_shift: int = 0) -> np.ndarray:
        image = np.full((180, 140, 3), 160, dtype=np.uint8)
        cv2.ellipse(image, (70, 90), (50, 70), 0, 0, 360, (150, 150, 150), -1)
        cv2.circle(image, (50 + horizontal_shift, 70), 7, (40, 40, 40), -1)
        cv2.circle(image, (90 + horizontal_shift, 70), 7, (40, 40, 40), -1)
        cv2.ellipse(image, (70, 100), (10, 18), 0, 0, 360, (100, 100, 100), -1)
        cv2.ellipse(image, (70, 130), (20, 8), 0, 0, 180, (70, 70, 70), 2)
        cv2.line(image, (28, 88), (18, 115), (80, 80, 80), 2)
        return image

    def test_micro_seam_analysis_returns_candidate_regions_and_box(self):
        detector = TamperingDetector()
        image = self._make_seam_image()

        result = detector.analyze_tampering(image, {'x': 0, 'y': 0, 'w': 160, 'h': 160})
        seam = result['micro_seam_analysis']

        self.assertTrue(seam['seam_detected'])
        self.assertGreater(seam['seam_probability'], 0.6)
        self.assertTrue(seam['candidate_regions'])
        self.assertIsNotNone(seam['highlight_box'])
        self.assertIsNotNone(seam['highlight_box_normalized'])

    def test_sclera_vascular_analysis_extracts_signature(self):
        analyzer = IrisAnalyzer()
        eye = self._make_eye_image()

        sclera = analyzer._analyze_sclera_vasculature(
            eye,
            {'iris_center': (60, 60), 'iris_radius': 22}
        )
        result = analyzer.analyze_iris(eye)

        self.assertTrue(sclera['sclera_detected'])
        self.assertGreater(sclera['regions_analyzed'], 0)
        self.assertTrue(sclera['vascular_signature'])
        self.assertIn('sclera_analysis', result)

    def test_kinship_analysis_surfaces_phenotypic_similarity(self):
        detector = DoppelgangerDetector()
        face1 = self._make_face_image(0)
        face2 = self._make_face_image(1)
        face_box = {'x': 0, 'y': 0, 'w': 140, 'h': 180}

        features1 = detector.analyze_identity_uniqueness(face1, face_box)
        features2 = detector.analyze_identity_uniqueness(face2, face_box)
        kinship = detector.compare_for_kinship(features1, features2, 58.0)

        self.assertIn('phenotypic_signature', features1)
        self.assertTrue(features1['phenotypic_signature']['signature_vector'])
        self.assertGreater(kinship['kinship_probability'], 70.0)
        self.assertNotEqual(kinship['relationship_hypothesis'], 'not_indicated')

    def test_compare_faces_exposes_kinship_analysis(self):
        suite = BiometricAnalysisSuite()
        face1 = self._make_face_image(0)
        face2 = self._make_face_image(1)
        face_box = {'x': 0, 'y': 0, 'w': 140, 'h': 180}

        result = suite.compare_faces(face1, face_box, face2, face_box, 58.0)

        self.assertIn('kinship_analysis', result)
        self.assertIn('relationship_hypothesis', result['kinship_analysis'])

    def test_strong_match_with_alteration_context_is_not_blunt_reject(self):
        suite = BiometricAnalysisSuite()
        face1 = self._make_face_image(0)
        face2 = self._make_face_image(1)
        face_box = {'x': 0, 'y': 0, 'w': 140, 'h': 180}

        base_features = {
            'age_invariant': {},
            'uniqueness': {'uniqueness_score': 0.8},
            'facial_markers': {'marker_map': [], 'markers_detected': 0},
            'makeup_disguise': {'disguise_detected': False, 'disguise_probability': 0.0, 'makeup_level': 'NONE/MINIMAL'},
            'iris': {'health_indicators': {'cataract_probability': 0.0, 'iris_clarity': 0.8}},
            'tampering': {},
            'morphing': {'is_morphed': False, 'morphing_probability': 0.0},
        }
        altered_features = {
            **base_features,
            'facial_markers': {
                'marker_map': [],
                'markers_detected': 1,
                'scar_analysis': {'scar_count': 1},
                'injury_signs': [],
                'surgery_indicators': [{'type': 'possible_rhinoplasty'}],
            },
            'makeup_disguise': {
                'disguise_detected': True,
                'disguise_probability': 68.0,
                'makeup_level': 'HEAVY',
                'analysis': {'prosthetic_indicators': {'prosthetic_probability': 42.0}},
            },
            'iris': {'health_indicators': {'cataract_probability': 0.45, 'iris_clarity': 0.35}},
            'morphing': {'is_morphed': True, 'morphing_probability': 62.0},
        }

        features_by_id = {id(face1): base_features, id(face2): altered_features}
        suite.full_analysis = lambda image, *_args, **_kwargs: features_by_id[id(image)]
        suite.age_analyzer.compare_across_age = lambda *_args, **_kwargs: {
            'is_same_person': False,
            'age_invariant_confidence': 35.0,
        }
        suite.doppelganger_detector.compare_for_doppelganger = lambda *_args, **_kwargs: {
            'is_doppelganger': False,
        }
        suite.doppelganger_detector.compare_for_kinship = lambda *_args, **_kwargs: {
            'likely_related': False,
        }
        suite.scar_analyzer.compare_markers = lambda *_args, **_kwargs: {
            'match_score': 0.0,
            'verdict': 'MARKERS_DIFFER',
        }

        result = suite.compare_faces(face1, face_box, face2, face_box, 88.0)

        self.assertEqual(result['final_verdict'], 'LIKELY_MATCH_WITH_ALTERATION_REVIEW')
        self.assertTrue(result['identity_alteration_context']['detected'])
        self.assertIn('comparison_surgery_indicator', result['identity_alteration_context']['factors'])

    def test_report_fallback_mentions_new_advanced_biometric_signals(self):
        analyst = LlamaForensicAnalyst.__new__(LlamaForensicAnalyst)
        verdict, confidence, steps = analyst._fallback(
            {
                "biometrics": {"verified": False, "cosine_similarity": 0.31, "embedding_norm": 26.0},
                "forensics": {
                    "frequency": {"deepfake_suspected": False, "deepfake_probability": 0.1},
                    "rppg": {"signal_state": "not_available", "is_live": False, "bpm": None},
                },
                "document": {"noiseprint": {"suspected_splice": False}},
                "primary_image_study": {},
                "advanced_biometrics": {
                    "primary": {
                        "threat_level": "MEDIUM",
                        "threat_score": 0.33,
                        "tampering": {
                            "tampering_detected": False,
                            "micro_seam_analysis": {"seam_probability": 0.74, "candidate_regions": [1, 2]},
                        },
                        "morphing": {"is_morphed": False},
                        "makeup_disguise": {"disguise_detected": False},
                        "iris": {
                            "anti_spoofing": {"contact_lens_detected": False},
                            "sclera_analysis": {"deepfake_suspected": True, "ai_noise_probability": 0.81},
                        },
                        "uniqueness": {"uniqueness_score": 61.0},
                        "facial_markers": {"markers_detected": 2},
                    },
                    "comparison": {"threat_level": "LOW", "threat_score": 0.1},
                    "pair_analysis": {
                        "final_verdict": "MANUAL_REVIEW",
                        "confidence": 62.0,
                        "doppelganger_analysis": {"is_doppelganger": False},
                        "kinship_analysis": {
                            "relationship_hypothesis": "possible_sibling",
                            "kinship_probability": 84.2,
                        },
                    },
                },
            }
        )

        step5 = next(step for step in steps if step.startswith("Step 5"))
        self.assertEqual(verdict, "FLAGGED")
        self.assertEqual(confidence, 0.85)
        self.assertIn("Micro-seam boundary", step5)
        self.assertIn("Sclera vascular AI noise", step5)
        self.assertIn("Kinship signal", step5)


if __name__ == "__main__":
    unittest.main()
