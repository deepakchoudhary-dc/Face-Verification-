from __future__ import annotations

import os
from typing import Any, Dict

from src.core.contracts import (
    BiometricsRequest,
    BiometricsResponse,
    DocumentRequest,
    ForensicsRequest,
    PairMatchRequest,
    ReconstructionRequest,
    ReportRequest,
)
from src.core.model_swapper import ModelSwapper
from src.document_processor.analysis import DocumentAnalysisService
from src.face_engine.analyzer import FaceAnalyzer
from src.forensics.service import ForensicsService as LocalForensicsService
from src.reconstruction.generative import OpenVINOForensicReconstructor
from src.reporting.llm_analyst import LlamaForensicAnalyst


def _ray_available() -> bool:
    try:
        import ray  # noqa: F401
        from ray import serve  # noqa: F401

        return True
    except Exception:
        return False


if _ray_available():
    from ray import serve

    @serve.deployment(
        name="BiometricsService",
        max_ongoing_requests=6,
        ray_actor_options={"num_cpus": 2},
    )
    class BiometricsService:
        def __init__(self) -> None:
            self.analyzer: FaceAnalyzer | None = None

        def _get_analyzer(self) -> FaceAnalyzer:
            if self.analyzer is None:
                self.analyzer = FaceAnalyzer()
            return self.analyzer

        async def extract(self, payload: Dict[str, Any]) -> Dict[str, Any]:
            req = BiometricsRequest.model_validate(payload)
            warnings: list[str] = []
            try:
                analyzer = self._get_analyzer()
                if req.calibrate_from_dir:
                    analyzer.calibrate_quality_threshold(req.calibrate_from_dir)
                faces = analyzer.get_face_embeddings(req.image_path)
                face_models = [
                    c for row in faces if (c := analyzer.to_contract(row)) is not None
                ]
            except Exception as exc:
                face_models = []
                warnings.append(f"biometrics_extract_failed: {exc}")
            out = BiometricsResponse(
                faces=face_models,
                quality_threshold=(self.analyzer.quality_threshold if self.analyzer else 20.0),
                warnings=warnings,
            )
            return out.model_dump()

        async def compare(self, payload: Dict[str, Any]) -> Dict[str, Any]:
            try:
                analyzer = self._get_analyzer()
                req = PairMatchRequest.model_validate(payload)
                return analyzer.compare_pair(req.primary, req.comparison).model_dump()
            except Exception as exc:
                return {
                    "cosine_similarity": -1.0,
                    "verified": False,
                    "threshold": 0.35,
                    "quality_gate_passed": False,
                    "rationale": f"biometrics_compare_failed: {exc}",
                }

        async def explain(self, payload: Dict[str, Any]) -> Dict[str, Any]:
            try:
                analyzer = self._get_analyzer()
                return analyzer.explain_similarity(
                    image_a_path=str(payload["image_a"]),
                    box_a=payload["box_a"],
                    image_b_path=str(payload["image_b"]),
                    box_b=payload["box_b"],
                    save_path=payload.get("save_path"),
                )
            except Exception as exc:
                return {"error": f"explainability_failed: {exc}"}

    @serve.deployment(
        name="ForensicsService",
        max_ongoing_requests=8,
        ray_actor_options={"num_cpus": 2},
    )
    class ForensicsService:
        def __init__(self) -> None:
            self.service = LocalForensicsService()

        async def analyze(self, payload: Dict[str, Any]) -> Dict[str, Any]:
            try:
                req = ForensicsRequest.model_validate(payload)
                return self.service.analyze(req).model_dump()
            except Exception as exc:
                return {
                    "frequency": {
                        "deepfake_probability": 0.0,
                        "deepfake_suspected": False,
                        "model_name": "analysis_failed",
                    },
                    "rppg": {
                        "is_live": False,
                        "bpm": None,
                        "confidence": 0.0,
                        "method": "POS",
                        "signal_state": "unknown",
                        "details": {"reason": f"forensics_failed: {exc}"},
                    },
                    "warnings": [f"forensics_failed: {exc}"],
                }

    @serve.deployment(
        name="DocumentService",
        max_ongoing_requests=6,
        ray_actor_options={"num_cpus": 2},
    )
    class DocumentService:
        def __init__(self) -> None:
            self.service: DocumentAnalysisService | None = None

        def _get_service(self) -> DocumentAnalysisService:
            if self.service is None:
                self.service = DocumentAnalysisService()
            return self.service

        async def parse(self, payload: Dict[str, Any]) -> Dict[str, Any]:
            try:
                service = self._get_service()
                req = DocumentRequest.model_validate(payload)
                return service.analyze(req.image_path, req.face_box).model_dump()
            except Exception as exc:
                return {"fields": {}, "noiseprint": None, "warnings": [f"document_failed: {exc}"]}

    @serve.deployment(
        name="ReconstructionService",
        max_ongoing_requests=1,
        ray_actor_options={"num_cpus": 4},
    )
    class ReconstructionService:
        def __init__(self) -> None:
            keep_loaded = os.getenv("CA_MONK_KEEP_HEAVY_MODELS", "0") == "1"
            self.swapper = ModelSwapper(
                loader=lambda: OpenVINOForensicReconstructor(),
                keep_loaded=keep_loaded,
            )

        async def reconstruct(self, payload: Dict[str, Any]) -> Dict[str, Any]:
            try:
                req = ReconstructionRequest.model_validate(payload)
                with self.swapper.session() as service:
                    return service.generate(req).model_dump()
            except Exception as exc:
                return {
                    "generated_image_path": None,
                    "warnings": [f"reconstruction_failed: {exc}"],
                }

    @serve.deployment(
        name="ReportingService",
        max_ongoing_requests=1,
        ray_actor_options={"num_cpus": 4},
    )
    class ReportingService:
        def __init__(self) -> None:
            keep_loaded = os.getenv("CA_MONK_KEEP_HEAVY_MODELS", "0") == "1"
            self.swapper = ModelSwapper(
                loader=lambda: LlamaForensicAnalyst(),
                keep_loaded=keep_loaded,
            )

        async def report(self, payload: Dict[str, Any]) -> Dict[str, Any]:
            try:
                req = ReportRequest.model_validate(payload)
                with self.swapper.session() as service:
                    return service.generate(req).model_dump()
            except Exception as exc:
                return {
                    "summary": "Analysis Failed",
                    "verdict": "Inconclusive",
                    "confidence": 0.0,
                    "reasoning_steps": [f"reporting_failed: {exc}"],
                }

    @serve.deployment(
        name="ServiceMesh",
        max_ongoing_requests=16,
        ray_actor_options={"num_cpus": 1},
    )
    class ServiceMesh:
        def __init__(self, biometrics, forensics, documents, generative, reporting) -> None:
            self.biometrics = biometrics
            self.forensics = forensics
            self.documents = documents
            self.generative = generative
            self.reporting = reporting

        async def extract(self, payload: Dict[str, Any]) -> Dict[str, Any]:
            return await self.biometrics.extract.remote(payload)

        async def compare(self, payload: Dict[str, Any]) -> Dict[str, Any]:
            return await self.biometrics.compare.remote(payload)

        async def explain(self, payload: Dict[str, Any]) -> Dict[str, Any]:
            return await self.biometrics.explain.remote(payload)

        async def analyze(self, payload: Dict[str, Any]) -> Dict[str, Any]:
            return await self.forensics.analyze.remote(payload)

        async def parse(self, payload: Dict[str, Any]) -> Dict[str, Any]:
            return await self.documents.parse.remote(payload)

        async def reconstruct(self, payload: Dict[str, Any]) -> Dict[str, Any]:
            return await self.generative.reconstruct.remote(payload)

        async def report(self, payload: Dict[str, Any]) -> Dict[str, Any]:
            return await self.reporting.report.remote(payload)


def initialize_serve_handles() -> Dict[str, Any]:
    if not _ray_available():
        raise RuntimeError("Ray Serve is not available in this environment.")

    import ray
    from ray import serve

    if not ray.is_initialized():
        configured = os.getenv("CA_MONK_RAY_CPUS")
        cpus = int(configured) if configured else max(1, os.cpu_count() or 1)
        ray.init(num_cpus=cpus, ignore_reinit_error=True, log_to_driver=False)

    serve.start(detached=False)

    app_name = os.getenv("CA_MONK_SERVE_APP", "ca_monk_app")
    mesh_app = ServiceMesh.bind(
        BiometricsService.bind(),
        ForensicsService.bind(),
        DocumentService.bind(),
        ReconstructionService.bind(),
        ReportingService.bind(),
    )
    mesh_handle = serve.run(mesh_app, name=app_name, route_prefix="/")

    return {
        "biometrics": mesh_handle,
        "forensics": mesh_handle,
        "documents": mesh_handle,
        "generative": mesh_handle,
        "reporting": mesh_handle,
    }
