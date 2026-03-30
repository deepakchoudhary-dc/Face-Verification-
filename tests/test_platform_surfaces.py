import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from src.api.job_store import InMemoryJobStore
from src.api.main import app
from src.core.benchmarking import BenchmarkHarness
from src.core.data_structures import Applicant
from src.core.evidence_integrity import EvidenceIntegrity
from src.face_engine.liveness import LivenessDetector
from src.reporting.interactive_casefile import InteractiveCasefileBuilder


class InteractiveCasefileTests(unittest.TestCase):
    def test_casefile_builder_writes_html_and_json(self):
        builder = InteractiveCasefileBuilder()
        with tempfile.TemporaryDirectory() as tmp:
            evidence_dir = Path(tmp)
            (evidence_dir / "reconstruction_hq_mesh.obj").write_text(
                "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n",
                encoding="utf-8",
            )
            (evidence_dir / "reconstruction_hq.jpg").write_bytes(b"fake")
            (evidence_dir / "expression_animation.gif").write_bytes(b"GIF89a")

            result = {
                "role": "subject",
                "is_match": False,
                "confidence": 41.2,
                "warnings": ["splice_detected"],
                "run_metadata": {"run_id": "abc123", "stage_telemetry": []},
                "runtime_capabilities": {"face_analyzer": {"liveness": {"backend": "advanced_heuristic_cpu_pad"}}},
                "comparisons": [
                    {
                        "filename": "comparison.png",
                        "match": {
                            "confidence": 0.2,
                            "model_scores": {},
                            "calibration_features": {"calibrated_confidence": 0.34},
                            "risk_flags": ["spoof_detected_comparison"],
                        },
                        "report": {"verdict": "FLAGGED"},
                        "forensics": {"frequency": {"deepfake_suspected": False}, "rppg": {"signal_state": "not_available"}},
                        "document_intelligence": {"noiseprint": {"suspected_splice": True}},
                        "advanced_biometrics": {"pair_analysis": {"final_verdict": "REJECT"}},
                        "forensic_3d_cross_validation": {"consistency_analysis": {"threat_level": "HIGH"}},
                        "face_evidence": {
                            "primary": {"model_name": "adaface", "embedding_norm": 24.0, "quality": "reliable", "liveness": {"score": 0.82, "signal_state": "live", "attack_type": None, "backend": "advanced_heuristic_cpu_pad"}},
                            "comparison": {"model_name": "adaface", "embedding_norm": 23.0, "quality": "reliable", "liveness": {"score": 0.19, "signal_state": "spoof", "attack_type": "replay_attack", "backend": "advanced_heuristic_cpu_pad"}},
                        },
                        "expression_suite": {
                            "generated_image_path": "expression_transfer.jpg",
                            "animation_gif_path": "expression_animation.gif",
                            "teaser_gif_path": "expression_teaser.gif",
                        },
                        "stage_telemetry": [{"stage": "forensics", "duration_ms": 12.5}],
                    }
                ],
            }

            meta = builder.build(str(evidence_dir), result)

            html_text = Path(meta["html_path"]).read_text(encoding="utf-8")
            data = json.loads(Path(meta["data_path"]).read_text(encoding="utf-8"))
            self.assertIn("Evidence Weight Ledger", html_text)
            self.assertIn("Runtime Capabilities", html_text)
            self.assertIn("Face Evidence", html_text)
            self.assertIn("PAD Score", html_text)
            self.assertIn("PAD Attack", html_text)
            self.assertIn("Expression Suite", html_text)
            self.assertIn("mesh-viewer-0", html_text)
            self.assertEqual(data["applicant_role"], "subject")
            self.assertEqual(len(data["comparisons"]), 1)
            self.assertEqual(data["artifacts"][0]["kind"], "image")
            self.assertEqual(data["comparisons"][0]["expression_suite"]["animation_gif_path"], "expression_animation.gif")
            self.assertTrue(any(item["source"] == "face_pad:comparison" for item in data["comparisons"][0]["ledger"]))


class BenchmarkHarnessTests(unittest.TestCase):
    def test_compute_metrics(self):
        metrics = BenchmarkHarness._compute_metrics(
            [
                {"expected_match": True, "observed_match": True, "runtime_ms": 100.0, "confidence": 90.0},
                {"expected_match": True, "observed_match": False, "runtime_ms": 120.0, "confidence": 40.0},
                {"expected_match": False, "observed_match": False, "runtime_ms": 80.0, "confidence": 30.0},
                {"expected_match": False, "observed_match": True, "runtime_ms": 70.0, "confidence": 60.0},
            ]
        )
        self.assertEqual(metrics["true_positive"], 1)
        self.assertEqual(metrics["false_negative"], 1)
        self.assertEqual(metrics["false_positive"], 1)
        self.assertEqual(metrics["true_negative"], 1)
        self.assertEqual(metrics["accuracy"], 0.5)
        self.assertIn("brier_score", metrics)

    def test_threshold_sweep_and_recommendation(self):
        rows = [
            {"expected_match": True, "observed_match": True, "confidence": 92.0, "runtime_ms": 10.0},
            {"expected_match": True, "observed_match": True, "confidence": 81.0, "runtime_ms": 10.0},
            {"expected_match": False, "observed_match": False, "confidence": 22.0, "runtime_ms": 10.0},
            {"expected_match": False, "observed_match": False, "confidence": 31.0, "runtime_ms": 10.0},
        ]
        sweep = BenchmarkHarness._compute_threshold_sweep(rows, {"thresholds": [0.3, 0.5, 0.7]})
        recommendation = BenchmarkHarness._recommend_threshold(sweep, {"target_far_max": 0.0})
        self.assertEqual(len(sweep), 3)
        self.assertEqual(recommendation["threshold"], 0.5)
        self.assertEqual(recommendation["reason"], "best_f1_under_far_constraint")


class LivenessDetectorTests(unittest.TestCase):
    def test_capabilities_report_fallback_backend_when_model_missing(self):
        detector = LivenessDetector(model_path="models/does_not_exist.onnx")
        caps = detector.capabilities()
        self.assertEqual(caps["backend"], "advanced_heuristic_cpu_pad")
        self.assertFalse(caps["model_loaded"])
        self.assertEqual(caps["analysis_version"], "still_pad_v2")


class JobStoreTests(unittest.IsolatedAsyncioTestCase):
    async def test_job_store_completes_job(self):
        async def runner(applicant: Applicant):
            return {"role": applicant.role, "is_match": False, "confidence": 12.5}

        store = InMemoryJobStore(runner)
        job = await store.submit(Applicant(role="job_subject"))

        for _ in range(20):
            state = await store.get(job["job_id"])
            if state and state["status"] == "completed":
                break
            await __import__("asyncio").sleep(0.01)

        result = await store.get_result(job["job_id"])
        self.assertEqual(result["role"], "job_subject")
        self.assertFalse(result["is_match"])
        self.assertEqual(result["confidence"], 12.5)


class ApiSurfaceTests(unittest.TestCase):
    def test_capabilities_endpoint_uses_engine_runtime_capabilities(self):
        class _StubEngine:
            def runtime_capabilities(self):
                return {
                    "face_analyzer": {"liveness": {"backend": "advanced_heuristic_cpu_pad"}},
                    "expression_transfer": {"available": True, "backend": "Deep3DExpressionTransferService"},
                    "expression_suite": {"available": True, "backend": "Deep3DExpressionSuiteService"},
                }

        with patch("src.api.main.get_engine", return_value=_StubEngine()):
            response = TestClient(app).get("/capabilities")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json()["runtime_capabilities"]["face_analyzer"]["liveness"]["backend"],
            "advanced_heuristic_cpu_pad",
        )
        self.assertTrue(response.json()["runtime_capabilities"]["expression_transfer"]["available"])
        self.assertTrue(response.json()["runtime_capabilities"]["expression_suite"]["available"])

    def test_expression_transfer_capabilities_endpoint_uses_engine_runtime_capabilities(self):
        class _StubService:
            def capabilities(self):
                return {
                    "available": True,
                    "backend": "Deep3DExpressionTransferService",
                }

        with patch("src.api.main.get_expression_transfer_service", return_value=_StubService()):
            response = TestClient(app).get("/expression-transfer/capabilities")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json()["expression_transfer"]["backend"],
            "Deep3DExpressionTransferService",
        )

    def test_expression_transfer_endpoint_delegates_to_engine(self):
        class _StubService:
            def generate(self, req):
                return {
                    "generated_image_path": req.evidence_save_path or "evidence_cards/expression_transfer/result.jpg",
                    "metadata": {
                        "source_image_path": req.source_image_path,
                        "expression_image_path": req.expression_image_path,
                        "transfer_pose": req.transfer_pose,
                    },
                }

        with patch("src.api.main.get_expression_transfer_service", return_value=_StubService()):
            response = TestClient(app).post(
                "/expression-transfer",
                json={
                    "source_image_path": "source.jpg",
                    "expression_image_path": "expression.jpg",
                    "save_path": "out.jpg",
                    "transfer_pose": True,
                },
            )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["generated_image_path"], "out.jpg")
        self.assertTrue(body["metadata"]["transfer_pose"])

    def test_expression_suite_capabilities_endpoint_uses_suite_service(self):
        class _StubService:
            def capabilities(self):
                return {
                    "available": True,
                    "backend": "Deep3DExpressionSuiteService",
                    "available_presets": {"laughing": {"description": "test"}},
                }

        with patch("src.api.main.get_expression_suite_service", return_value=_StubService()):
            response = TestClient(app).get("/expression-suite/capabilities")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json()["expression_suite"]["backend"],
            "Deep3DExpressionSuiteService",
        )
        self.assertIn("laughing", response.json()["expression_suite"]["available_presets"])

    def test_expression_suite_endpoint_returns_animation_payload(self):
        class _StubService:
            def generate(self, req):
                return {
                    "generated_image_path": req.evidence_save_path or "evidence_cards/expression_suite/result.jpg",
                    "animation_gif_path": "evidence_cards/expression_suite/result_animation.gif",
                    "teaser_gif_path": "evidence_cards/expression_suite/result_teaser.gif",
                    "turntable_gif_path": "evidence_cards/expression_suite/result_turntable.gif",
                    "preset_gallery_image_path": "evidence_cards/expression_suite/result_gallery.jpg",
                    "metadata": {
                        "source_image_path": req.source_image_path,
                        "expression_image_path": req.expression_image_path,
                        "transfer_pose": req.transfer_pose,
                        "expression_preset": req.expression_preset,
                        "target_yaw_deg": req.target_yaw_deg,
                    },
                }

        with patch("src.api.main.get_expression_suite_service", return_value=_StubService()):
            response = TestClient(app).post(
                "/expression-suite",
                json={
                    "source_image_path": "source.jpg",
                    "expression_image_path": "expression.jpg",
                    "save_path": "suite.jpg",
                    "transfer_pose": False,
                    "expression_preset": "laughing",
                    "target_yaw_deg": 32.0,
                },
            )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["generated_image_path"], "suite.jpg")
        self.assertEqual(body["animation_gif_path"], "evidence_cards/expression_suite/result_animation.gif")
        self.assertEqual(body["turntable_gif_path"], "evidence_cards/expression_suite/result_turntable.gif")
        self.assertEqual(body["preset_gallery_image_path"], "evidence_cards/expression_suite/result_gallery.jpg")
        self.assertEqual(body["metadata"]["expression_preset"], "laughing")
        self.assertEqual(body["metadata"]["target_yaw_deg"], 32.0)

    def test_evidence_verify_endpoint_returns_manifest_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            evidence_dir = Path(tmp)
            (evidence_dir / "artifact.txt").write_text("evidence", encoding="utf-8")
            EvidenceIntegrity().create_manifest(str(evidence_dir))

            response = TestClient(app).post(
                "/evidence/verify",
                json={"evidence_dir": str(evidence_dir)},
            )

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["verification"]["valid"])


if __name__ == "__main__":
    unittest.main()
