from __future__ import annotations

import asyncio
import uuid
from datetime import datetime
from typing import Any, Awaitable, Callable, Dict, List, Optional

from src.core.data_structures import Applicant
from src.core.serialization import to_builtin


def _utc_now() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


class InMemoryJobStore:
    """Queue-like in-memory job tracking for long-running CPU pipeline requests."""

    def __init__(self, runner: Callable[[Applicant], Awaitable[Dict[str, Any]]]) -> None:
        self._runner = runner
        self._jobs: Dict[str, Dict[str, Any]] = {}
        self._lock = asyncio.Lock()
        self._runner_lock = asyncio.Lock()

    async def submit(self, applicant: Applicant) -> Dict[str, Any]:
        job_id = uuid.uuid4().hex
        record = {
            "job_id": job_id,
            "status": "queued",
            "created_utc": _utc_now(),
            "started_utc": None,
            "completed_utc": None,
            "applicant_role": applicant.role,
            "error": None,
            "result": None,
        }
        async with self._lock:
            self._jobs[job_id] = record
        asyncio.create_task(self._run(job_id, applicant))
        return self._public_record(record)

    async def list_jobs(self) -> List[Dict[str, Any]]:
        async with self._lock:
            records = list(self._jobs.values())
        records.sort(key=lambda item: item.get("created_utc", ""), reverse=True)
        return [self._public_record(record) for record in records]

    async def get(self, job_id: str) -> Optional[Dict[str, Any]]:
        async with self._lock:
            record = self._jobs.get(job_id)
        return self._public_record(record) if record else None

    async def get_result(self, job_id: str) -> Optional[Dict[str, Any]]:
        async with self._lock:
            record = self._jobs.get(job_id)
        if not record:
            return None
        if record.get("status") != "completed":
            return self._public_record(record)
        return to_builtin(record.get("result"))

    async def _run(self, job_id: str, applicant: Applicant) -> None:
        async with self._lock:
            record = self._jobs[job_id]
            record["status"] = "running"
            record["started_utc"] = _utc_now()

        async with self._runner_lock:
            try:
                result = await self._runner(applicant)
            except Exception as exc:
                async with self._lock:
                    record = self._jobs[job_id]
                    record["status"] = "failed"
                    record["completed_utc"] = _utc_now()
                    record["error"] = str(exc)
            else:
                async with self._lock:
                    record = self._jobs[job_id]
                    record["status"] = "completed"
                    record["completed_utc"] = _utc_now()
                    record["result"] = to_builtin(result)

    @staticmethod
    def _public_record(record: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if record is None:
            return None
        result = record.get("result") or {}
        return {
            "job_id": record.get("job_id"),
            "status": record.get("status"),
            "created_utc": record.get("created_utc"),
            "started_utc": record.get("started_utc"),
            "completed_utc": record.get("completed_utc"),
            "applicant_role": record.get("applicant_role"),
            "error": record.get("error"),
            "summary": {
                "is_match": result.get("is_match"),
                "confidence": result.get("confidence"),
                "evidence_dir": result.get("evidence_dir"),
                "interactive_casefile": result.get("interactive_casefile", {}),
            } if result else None,
        }
