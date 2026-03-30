from __future__ import annotations

from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.api.job_store import InMemoryJobStore
from src.core.contracts import ExpressionTransferRequest
from src.core.evidence_integrity import EvidenceIntegrity
from src.core.data_structures import Applicant, Document
from src.core.engine import VerificationEngine
from src.core.serialization import to_builtin
from src.reconstruction.expression_transfer import Deep3DExpressionTransferService
from src.reconstruction.expression_suite import Deep3DExpressionSuiteService


app = FastAPI(title="CA_Monk Forensic API", version="3.2")
_ENGINE: VerificationEngine | None = None
_JOB_STORE: InMemoryJobStore | None = None
_EXPRESSION_TRANSFER: Deep3DExpressionTransferService | None = None
_EXPRESSION_SUITE: Deep3DExpressionSuiteService | None = None


def get_engine() -> VerificationEngine:
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = VerificationEngine()
    return _ENGINE


def get_job_store() -> InMemoryJobStore:
    global _JOB_STORE
    if _JOB_STORE is None:
        _JOB_STORE = InMemoryJobStore(lambda applicant: get_engine().process_applicant_async(applicant))
    return _JOB_STORE


def get_expression_transfer_service() -> Deep3DExpressionTransferService:
    global _EXPRESSION_TRANSFER
    if _EXPRESSION_TRANSFER is None:
        _EXPRESSION_TRANSFER = Deep3DExpressionTransferService()
    return _EXPRESSION_TRANSFER


def get_expression_suite_service() -> Deep3DExpressionSuiteService:
    global _EXPRESSION_SUITE
    if _EXPRESSION_SUITE is None:
        _EXPRESSION_SUITE = Deep3DExpressionSuiteService()
    return _EXPRESSION_SUITE


class DocumentPayload(BaseModel):
    file_path: str
    doc_class: str = "unknown"
    original_filename: Optional[str] = None
    s3_url: Optional[str] = None


class ApplicantPayload(BaseModel):
    role: str
    primary_docs: List[DocumentPayload] = Field(default_factory=list)
    comparison_docs: List[DocumentPayload] = Field(default_factory=list)


class EvidenceVerifyPayload(BaseModel):
    evidence_dir: str


class ExpressionTransferPayload(BaseModel):
    source_image_path: str
    expression_image_path: str
    save_path: Optional[str] = None
    transfer_pose: bool = False
    expression_preset: Optional[str] = None
    expression_strength: float = 1.0
    target_yaw_deg: Optional[float] = None
    target_pitch_deg: Optional[float] = None
    target_roll_deg: Optional[float] = None
    blink_strength: float = 0.15
    animation_mode: str = "cinematic"
    animation_frames: int = 18
    sideview_angle_deg: float = 30.0


def _build_applicant(payload: ApplicantPayload) -> Applicant:
    return Applicant(
        role=payload.role,
        primary_docs=[
            Document(
                file_path=d.file_path,
                doc_class=d.doc_class,
                original_filename=d.original_filename,
                s3_url=d.s3_url,
            )
            for d in payload.primary_docs
        ],
        comparison_docs=[
            Document(
                file_path=d.file_path,
                doc_class=d.doc_class,
                original_filename=d.original_filename,
                s3_url=d.s3_url,
            )
            for d in payload.comparison_docs
        ],
    )


@app.get("/health")
async def health() -> dict:
    engine = get_engine()
    jobs = get_job_store()
    job_rows = await jobs.list_jobs()
    return {
        "status": "ok",
        "ray_enabled": bool(getattr(engine, "handles", None)),
        "pipeline_mode": "linear",
        "jobs": {
            "queued": sum(1 for row in job_rows if row.get("status") == "queued"),
            "running": sum(1 for row in job_rows if row.get("status") == "running"),
            "completed": sum(1 for row in job_rows if row.get("status") == "completed"),
            "failed": sum(1 for row in job_rows if row.get("status") == "failed"),
        },
    }


@app.get("/capabilities")
async def capabilities() -> dict:
    engine = get_engine()
    return {
        "status": "ok",
        "runtime_capabilities": to_builtin(engine.runtime_capabilities()),
    }


@app.post("/process")
async def process(payload: ApplicantPayload) -> dict:
    engine = get_engine()
    applicant = _build_applicant(payload)
    result = await engine.process_applicant_async(applicant)
    return to_builtin(result)


@app.get("/expression-transfer/capabilities")
async def expression_transfer_capabilities() -> dict:
    return {
        "status": "ok",
        "expression_transfer": to_builtin(get_expression_transfer_service().capabilities()),
    }


@app.post("/expression-transfer")
async def expression_transfer(payload: ExpressionTransferPayload) -> dict:
    result = get_expression_transfer_service().generate(
        ExpressionTransferRequest(
            source_image_path=payload.source_image_path,
            expression_image_path=payload.expression_image_path,
            evidence_save_path=payload.save_path,
            transfer_pose=payload.transfer_pose,
            expression_preset=payload.expression_preset,
            expression_strength=payload.expression_strength,
            target_yaw_deg=payload.target_yaw_deg,
            target_pitch_deg=payload.target_pitch_deg,
            target_roll_deg=payload.target_roll_deg,
            blink_strength=payload.blink_strength,
            animation_mode=payload.animation_mode,
            animation_frames=payload.animation_frames,
            sideview_angle_deg=payload.sideview_angle_deg,
        )
    )
    return to_builtin(result)


@app.get("/expression-suite/capabilities")
async def expression_suite_capabilities() -> dict:
    return {
        "status": "ok",
        "expression_suite": to_builtin(get_expression_suite_service().capabilities()),
    }


@app.post("/expression-suite")
async def expression_suite(payload: ExpressionTransferPayload) -> dict:
    result = get_expression_suite_service().generate(
        ExpressionTransferRequest(
            source_image_path=payload.source_image_path,
            expression_image_path=payload.expression_image_path,
            evidence_save_path=payload.save_path,
            transfer_pose=payload.transfer_pose,
            expression_preset=payload.expression_preset,
            expression_strength=payload.expression_strength,
            target_yaw_deg=payload.target_yaw_deg,
            target_pitch_deg=payload.target_pitch_deg,
            target_roll_deg=payload.target_roll_deg,
            blink_strength=payload.blink_strength,
            animation_mode=payload.animation_mode,
            animation_frames=payload.animation_frames,
            sideview_angle_deg=payload.sideview_angle_deg,
        )
    )
    return to_builtin(result)


@app.get("/jobs")
async def list_jobs() -> dict:
    return {"jobs": await get_job_store().list_jobs()}


@app.post("/jobs")
async def create_job(payload: ApplicantPayload) -> dict:
    applicant = _build_applicant(payload)
    job = await get_job_store().submit(applicant)
    return {
        "job": job,
        "status_url": f"/jobs/{job['job_id']}",
        "result_url": f"/jobs/{job['job_id']}/result",
    }


@app.get("/jobs/{job_id}")
async def get_job(job_id: str) -> dict:
    job = await get_job_store().get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job_not_found")
    return {"job": job}


@app.get("/jobs/{job_id}/result")
async def get_job_result(job_id: str) -> dict:
    result = await get_job_store().get_result(job_id)
    if result is None:
        raise HTTPException(status_code=404, detail="job_not_found")
    if result.get("status") in {"queued", "running", "failed"}:
        raise HTTPException(status_code=409, detail=result)
    return to_builtin(result)


@app.post("/evidence/verify")
async def verify_evidence(payload: EvidenceVerifyPayload) -> dict:
    verification = EvidenceIntegrity().verify_manifest(payload.evidence_dir)
    if verification.get("error") == "manifest_not_found":
        raise HTTPException(status_code=404, detail="manifest_not_found")
    return {
        "evidence_dir": payload.evidence_dir,
        "verification": to_builtin(verification),
    }


@app.on_event("shutdown")
async def shutdown() -> None:
    global _ENGINE, _JOB_STORE, _EXPRESSION_TRANSFER, _EXPRESSION_SUITE
    if _ENGINE is not None:
        _ENGINE.cleanup()
        _ENGINE = None
    if _EXPRESSION_TRANSFER is not None:
        _EXPRESSION_TRANSFER.cleanup()
        _EXPRESSION_TRANSFER = None
    if _EXPRESSION_SUITE is not None:
        _EXPRESSION_SUITE.cleanup()
        _EXPRESSION_SUITE = None
    _JOB_STORE = None
