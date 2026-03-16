import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from src.core.engine import VerificationEngine
from src.core.serialization import to_builtin
from src.core.contracts import ForensicsRequest, ReportRequest
from src.forensics.consistency_checker import ForensicConsistencyChecker
from src.forensics.service import ForensicsService
from src.biometric_analysis.iris_analyzer import IrisAnalyzer
from src.reporting.llm_analyst import LlamaForensicAnalyst


class _StubDeepfake:
    threshold = 0.5

    def predict_probability(self, image):
        return 0.1

    def generate_spectral_heatmap(self, image, save_path=None):
        return None


class _StubRPPG:
    def generate_pulse_graph(self, estimate, save_path=None):
        return None


class TrustCoreTests(unittest.TestCase):
    def test_to_builtin_normalizes_numpy_scalars_arrays_and_paths(self):
        payload = {
            "flag": np.bool_(True),
            "score": np.float32(0.75),
            "vector": np.asarray([1, 2, 3], dtype=np.int16),
            "path": Path("evidence/final.json"),
        }

        built = to_builtin(payload)

        self.assertIsInstance(built["flag"], bool)
        self.assertTrue(built["flag"])
        self.assertIsInstance(built["score"], float)
        self.assertEqual(built["vector"], [1, 2, 3])
        self.assertEqual(built["path"], str(Path("evidence/final.json")))

    def test_consistency_checker_uses_document_splice_signal(self):
        checker = ForensicConsistencyChecker()

        result = checker.analyze(
            match_result={"verified": True, "cosine_similarity": 0.82},
            forensics_result={"frequency": {"deepfake_suspected": False}},
            document_result={"noiseprint": {"suspected_splice": True}},
            adv_biometrics={"pair_analysis": {"final_verdict": "VERIFIED"}},
            recon_primary={},
            recon_comparison={},
        )

        self.assertGreaterEqual(result["contradiction_count"], 1)
        self.assertTrue(
            any("splice_detected=True" in detail for detail in result["contradictions"])
        )

    def test_match_confidence_percent_prefers_confidence_then_fusion_then_cosine(self):
        self.assertEqual(
            VerificationEngine._match_confidence_percent({"confidence": 0.834}),
            83.4,
        )
        self.assertEqual(
            VerificationEngine._match_confidence_percent({"fusion_score": 0.61}),
            61.0,
        )
        self.assertEqual(
            VerificationEngine._match_confidence_percent({"cosine_similarity": 0.2}),
            60.0,
        )

    def test_forensics_missing_video_is_not_reported_as_spoof(self):
        service = ForensicsService.__new__(ForensicsService)
        service.deepfake = _StubDeepfake()
        service.rppg = _StubRPPG()

        with patch("src.forensics.service.cv2.imread", return_value=np.full((32, 32, 3), 127, dtype=np.uint8)):
            result = service.analyze(ForensicsRequest(image_path="comparison.jpg", video_path=None))

        self.assertEqual(result.rppg.signal_state, "not_available")
        self.assertFalse(result.rppg.is_live)
        self.assertIn("video_not_provided_rppg_not_available", result.warnings)

    def test_report_fallback_does_not_call_missing_video_non_living(self):
        analyst = LlamaForensicAnalyst.__new__(LlamaForensicAnalyst)
        verdict, confidence, steps = analyst._fallback(
            {
                "biometrics": {"verified": False, "cosine_similarity": 0.2, "embedding_norm": 26.0},
                "forensics": {
                    "frequency": {"deepfake_suspected": False, "deepfake_probability": 0.1},
                    "rppg": {"signal_state": "not_available", "is_live": False, "bpm": None},
                },
                "document": {"noiseprint": {"suspected_splice": True}},
                "advanced_biometrics": {"primary": {}, "comparison": {}, "pair_analysis": {"final_verdict": "REJECT"}},
                "primary_image_study": {},
            }
        )

        self.assertEqual(verdict, "FLAGGED")
        self.assertEqual(confidence, 0.85)
        self.assertTrue(any("Video liveness evidence unavailable" in step for step in steps))
        self.assertFalse(any("non-living" in step.lower() for step in steps))

    def test_invalid_llm_verdict_uses_fallback_reasoning(self):
        analyst = LlamaForensicAnalyst.__new__(LlamaForensicAnalyst)
        analyst.llm_available = True
        analyst.model_name = "stub"
        analyst._chat = lambda prompt: (
            '{"summary":"bad summary","verdict":"REJECT","confidence":0.99,'
            '"reasoning_steps":["wrong reasoning"]}'
        )

        req = ReportRequest(
            applicant_id="subject",
            biometrics={"verified": False, "cosine_similarity": 0.2, "embedding_norm": 26.0},
            forensics={
                "frequency": {"deepfake_suspected": False, "deepfake_probability": 0.1},
                "rppg": {"signal_state": "not_available", "is_live": False, "bpm": None},
            },
            document={"noiseprint": {"suspected_splice": True}},
            reconstruction={},
            primary_image_study={},
            advanced_biometrics={"primary": {}, "comparison": {}, "pair_analysis": {"final_verdict": "REJECT"}},
        )

        result = analyst.generate(req)

        self.assertEqual(result.verdict, "FLAGGED")
        self.assertEqual(result.confidence, 0.85)
        self.assertTrue(any("Video liveness evidence unavailable" in step for step in result.reasoning_steps))
        self.assertFalse(any("non-living" in step.lower() for step in result.reasoning_steps))

    def test_iris_quality_accepts_numpy_unsigned_coordinates(self):
        analyzer = IrisAnalyzer()
        image = np.zeros((20, 20), dtype=np.uint8)

        quality = analyzer._calculate_iris_quality(
            image=image,
            cx=np.uint16(4),
            cy=np.uint16(4),
            pupil_r=np.uint16(2),
            iris_r=np.uint16(6),
        )

        self.assertIsInstance(quality, float)


if __name__ == "__main__":
    unittest.main()
