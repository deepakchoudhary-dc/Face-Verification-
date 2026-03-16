from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.core.data_structures import Applicant, Document
from src.core.engine import VerificationEngine
from src.core.serialization import to_builtin


class BenchmarkHarness:
    """Run a manifest of golden cases and compute operational verification metrics."""

    def __init__(self, engine: Optional[VerificationEngine] = None) -> None:
        self.engine = engine
        self._owns_engine = engine is None

    def run_manifest(
        self,
        manifest_path: str,
        output_dir: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> Dict[str, Any]:
        manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        cases = list(manifest.get("cases", []) or [])
        if limit is not None:
            cases = cases[: max(0, int(limit))]

        engine = self.engine or VerificationEngine()
        results: List[Dict[str, Any]] = []

        try:
            for case in cases:
                applicant = self._applicant_from_case(case)
                started = time.perf_counter()
                result = to_builtin(engine.process_applicant(applicant))
                runtime_ms = round((time.perf_counter() - started) * 1000.0, 2)
                observed_match = bool(result.get("is_match", False))
                expected_match = bool(case.get("expected_match", False))
                first_comparison = ((result.get("comparisons", []) or [{}])[0] or {})

                case_row = {
                    "case_id": case.get("id", applicant.role),
                    "expected_match": expected_match,
                    "observed_match": observed_match,
                    "passed": observed_match == expected_match,
                    "confidence": float(result.get("confidence", 0.0) or 0.0),
                    "normalized_score": self._normalized_score(result.get("confidence", 0.0)),
                    "runtime_ms": runtime_ms,
                    "evidence_dir": result.get("evidence_dir"),
                    "warnings": list(result.get("warnings", []) or []),
                    "tags": list(case.get("tags", []) or []),
                    "top_report_verdict": str(((first_comparison.get("report", {}) or {}).get("verdict", "")) or ""),
                    "top_threat_level": str(
                        (
                            (
                                (first_comparison.get("forensic_3d_cross_validation", {}) or {})
                                .get("consistency_analysis", {})
                                or {}
                            ).get("threat_level", "")
                        )
                        or ""
                    ),
                    "top_match": to_builtin(first_comparison.get("match", {}) or {}),
                    "result": result,
                }
                results.append(case_row)

            metrics = self._compute_metrics(results)
            threshold_sweep = self._compute_threshold_sweep(
                results,
                manifest.get("threshold_sweep", {}) or {},
            )
            payload = {
                "manifest_path": str(Path(manifest_path).resolve()),
                "manifest_name": manifest.get("name", Path(manifest_path).stem),
                "case_count": len(results),
                "metrics": metrics,
                "confidence_distribution": self._confidence_distribution(results),
                "threshold_sweep": threshold_sweep,
                "recommended_threshold": self._recommend_threshold(
                    threshold_sweep,
                    manifest.get("threshold_sweep", {}) or {},
                ),
                "cases": results,
            }

            if output_dir:
                self._write_outputs(output_dir, payload)
            return payload
        finally:
            if self._owns_engine:
                engine.cleanup()

    @staticmethod
    def _applicant_from_case(case: Dict[str, Any]) -> Applicant:
        def build_docs(rows: List[Dict[str, Any]]) -> List[Document]:
            docs: List[Document] = []
            for row in rows:
                docs.append(
                    Document(
                        file_path=str(row["file_path"]),
                        doc_class=str(row.get("doc_class", "unknown")),
                        original_filename=row.get("original_filename"),
                        s3_url=row.get("s3_url"),
                    )
                )
            return docs

        return Applicant(
            role=str(case.get("role", case.get("id", "benchmark_case"))),
            primary_docs=build_docs(list(case.get("primary_docs", []) or [])),
            comparison_docs=build_docs(list(case.get("comparison_docs", []) or [])),
        )

    @staticmethod
    def _normalized_score(confidence: Any) -> float:
        raw = float(confidence or 0.0)
        return float(raw / 100.0) if raw > 1.0 else raw

    @classmethod
    def _compute_metrics(cls, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        tp = sum(1 for row in results if row["expected_match"] and row["observed_match"])
        tn = sum(1 for row in results if (not row["expected_match"]) and (not row["observed_match"]))
        fp = sum(1 for row in results if (not row["expected_match"]) and row["observed_match"])
        fn = sum(1 for row in results if row["expected_match"] and (not row["observed_match"]))
        total = max(1, len(results))
        positives = max(1, sum(1 for row in results if row["expected_match"]))
        negatives = max(1, sum(1 for row in results if not row["expected_match"]))

        precision_den = max(1, tp + fp)
        recall_den = max(1, tp + fn)
        precision = tp / precision_den
        recall = tp / recall_den
        f1_den = max(1e-8, precision + recall)

        positive_scores = [
            cls._normalized_score(row.get("confidence", 0.0))
            for row in results
            if row["expected_match"]
        ]
        negative_scores = [
            cls._normalized_score(row.get("confidence", 0.0))
            for row in results
            if not row["expected_match"]
        ]
        brier_score = sum(
            (cls._normalized_score(row.get("confidence", 0.0)) - (1.0 if row["expected_match"] else 0.0)) ** 2
            for row in results
        ) / total

        return {
            "true_positive": tp,
            "true_negative": tn,
            "false_positive": fp,
            "false_negative": fn,
            "accuracy": round((tp + tn) / total, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round((2 * precision * recall) / f1_den, 4),
            "false_accept_rate": round(fp / negatives, 4),
            "false_reject_rate": round(fn / positives, 4),
            "brier_score": round(brier_score, 4),
            "average_runtime_ms": round(
                sum(float(row.get("runtime_ms", 0.0) or 0.0) for row in results) / total,
                2,
            ),
            "average_confidence": round(
                sum(float(row.get("confidence", 0.0) or 0.0) for row in results) / total,
                2,
            ),
            "positive_average_score": round(sum(positive_scores) / max(len(positive_scores), 1), 4),
            "negative_average_score": round(sum(negative_scores) / max(len(negative_scores), 1), 4),
        }

    @classmethod
    def _compute_threshold_sweep(
        cls,
        results: List[Dict[str, Any]],
        config: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        if not results:
            return []

        thresholds = list(config.get("thresholds", []) or [])
        if not thresholds:
            start = float(config.get("start", 0.35) or 0.35)
            stop = float(config.get("stop", 0.95) or 0.95)
            step = float(config.get("step", 0.05) or 0.05)
            thresholds = []
            cursor = start
            while cursor <= stop + 1e-8:
                thresholds.append(round(cursor, 4))
                cursor += step

        rows: List[Dict[str, Any]] = []
        for threshold in thresholds:
            projected = []
            for row in results:
                score = cls._normalized_score(row.get("confidence", 0.0))
                projected.append(
                    {
                        "expected_match": bool(row["expected_match"]),
                        "observed_match": bool(score >= float(threshold)),
                        "confidence": score,
                        "runtime_ms": float(row.get("runtime_ms", 0.0) or 0.0),
                    }
                )
            metrics = cls._compute_metrics(projected)
            rows.append(
                {
                    "threshold": round(float(threshold), 4),
                    "accuracy": metrics["accuracy"],
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "f1_score": metrics["f1_score"],
                    "false_accept_rate": metrics["false_accept_rate"],
                    "false_reject_rate": metrics["false_reject_rate"],
                }
            )
        return rows

    @staticmethod
    def _recommend_threshold(
        threshold_sweep: List[Dict[str, Any]],
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        if not threshold_sweep:
            return {"threshold": None, "reason": "no_sweep_data"}

        target_far = float(config.get("target_far_max", 0.05) or 0.05)
        eligible = [
            row for row in threshold_sweep
            if float(row["false_accept_rate"]) <= target_far
        ]
        pool = eligible or threshold_sweep
        best = max(
            pool,
            key=lambda row: (
                float(row["f1_score"]),
                float(row["accuracy"]),
                -float(row["false_accept_rate"]),
            ),
        )
        return {
            "threshold": best.get("threshold"),
            "reason": "best_f1_under_far_constraint" if eligible else "best_f1_without_far_constraint",
            "target_far_max": target_far,
            "metrics": {
                "accuracy": best.get("accuracy"),
                "precision": best.get("precision"),
                "recall": best.get("recall"),
                "f1_score": best.get("f1_score"),
                "false_accept_rate": best.get("false_accept_rate"),
                "false_reject_rate": best.get("false_reject_rate"),
            },
        }

    @classmethod
    def _confidence_distribution(cls, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        positive = [
            cls._normalized_score(row.get("confidence", 0.0))
            for row in results
            if row["expected_match"]
        ]
        negative = [
            cls._normalized_score(row.get("confidence", 0.0))
            for row in results
            if not row["expected_match"]
        ]

        def summarize(values: List[float]) -> Dict[str, Any]:
            if not values:
                return {"count": 0, "min": None, "max": None, "mean": None}
            return {
                "count": len(values),
                "min": round(min(values), 4),
                "max": round(max(values), 4),
                "mean": round(sum(values) / len(values), 4),
            }

        return {
            "positive": summarize(positive),
            "negative": summarize(negative),
            "separation_gap": round(
                (sum(positive) / max(len(positive), 1)) - (sum(negative) / max(len(negative), 1)),
                4,
            ),
        }

    def _write_outputs(self, output_dir: str, payload: Dict[str, Any]) -> None:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "benchmark_results.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        (out_dir / "benchmark_report.md").write_text(
            self._render_markdown(payload),
            encoding="utf-8",
        )
        (out_dir / "benchmark_scores.csv").write_text(
            self._render_scores_csv(payload),
            encoding="utf-8",
        )

    @staticmethod
    def _render_markdown(payload: Dict[str, Any]) -> str:
        metrics = payload.get("metrics", {}) or {}
        recommendation = payload.get("recommended_threshold", {}) or {}
        distribution = payload.get("confidence_distribution", {}) or {}
        lines = [
            "# CA_Monk Benchmark Report",
            "",
            f"- Manifest: `{payload.get('manifest_name', 'unknown')}`",
            f"- Cases: `{payload.get('case_count', 0)}`",
            f"- Accuracy: `{metrics.get('accuracy', 0):.4f}`",
            f"- Precision: `{metrics.get('precision', 0):.4f}`",
            f"- Recall: `{metrics.get('recall', 0):.4f}`",
            f"- F1: `{metrics.get('f1_score', 0):.4f}`",
            f"- FAR: `{metrics.get('false_accept_rate', 0):.4f}`",
            f"- FRR: `{metrics.get('false_reject_rate', 0):.4f}`",
            f"- Brier: `{metrics.get('brier_score', 0):.4f}`",
            f"- Avg runtime: `{metrics.get('average_runtime_ms', 0):.2f} ms`",
            "",
            "## Operating Point",
            "",
            f"- Recommended threshold: `{recommendation.get('threshold')}`",
            f"- Reason: `{recommendation.get('reason')}`",
            f"- Target FAR max: `{recommendation.get('target_far_max')}`",
            "",
            "## Confidence Distribution",
            "",
            f"- Positive mean score: `{((distribution.get('positive', {}) or {}).get('mean'))}`",
            f"- Negative mean score: `{((distribution.get('negative', {}) or {}).get('mean'))}`",
            f"- Separation gap: `{distribution.get('separation_gap')}`",
            "",
            "## Threshold Sweep",
            "",
            "| Threshold | Accuracy | Precision | Recall | F1 | FAR | FRR |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for row in payload.get("threshold_sweep", [])[:12]:
            lines.append(
                f"| `{float(row.get('threshold', 0.0) or 0.0):.2f}` | `{float(row.get('accuracy', 0.0) or 0.0):.4f}` | "
                f"`{float(row.get('precision', 0.0) or 0.0):.4f}` | `{float(row.get('recall', 0.0) or 0.0):.4f}` | "
                f"`{float(row.get('f1_score', 0.0) or 0.0):.4f}` | `{float(row.get('false_accept_rate', 0.0) or 0.0):.4f}` | "
                f"`{float(row.get('false_reject_rate', 0.0) or 0.0):.4f}` |"
            )

        lines.extend(
            [
                "",
                "## Case Results",
                "",
                "| Case | Expected | Observed | Confidence | Score | Runtime ms | Verdict | Threat | Pass |",
                "| --- | --- | --- | ---: | ---: | ---: | --- | --- | --- |",
            ]
        )
        for row in payload.get("cases", []):
            lines.append(
                f"| `{row.get('case_id')}` | `{row.get('expected_match')}` | `{row.get('observed_match')}` | "
                f"`{float(row.get('confidence', 0.0) or 0.0):.2f}` | "
                f"`{float(row.get('normalized_score', 0.0) or 0.0):.4f}` | "
                f"`{float(row.get('runtime_ms', 0.0) or 0.0):.2f}` | "
                f"`{row.get('top_report_verdict')}` | `{row.get('top_threat_level')}` | `{row.get('passed')}` |"
            )
        lines.append("")
        return "\n".join(lines)

    @staticmethod
    def _render_scores_csv(payload: Dict[str, Any]) -> str:
        lines = [
            "case_id,expected_match,observed_match,confidence,normalized_score,runtime_ms,top_report_verdict,top_threat_level,passed",
        ]
        for row in payload.get("cases", []):
            values = [
                row.get("case_id"),
                row.get("expected_match"),
                row.get("observed_match"),
                row.get("confidence"),
                row.get("normalized_score"),
                row.get("runtime_ms"),
                row.get("top_report_verdict"),
                row.get("top_threat_level"),
                row.get("passed"),
            ]
            escaped = []
            for item in values:
                text = str(item).replace('"', '""')
                escaped.append(f'"{text}"')
            lines.append(",".join(escaped))
        return "\n".join(lines) + "\n"
