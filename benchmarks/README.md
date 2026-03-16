# Benchmark Manifests

Each manifest describes a set of golden cases for the CPU-first verification pipeline.

Schema:

```json
{
  "name": "benchmark_name",
  "version": "1.0",
  "threshold_sweep": {
    "start": 0.35,
    "stop": 0.95,
    "step": 0.05,
    "target_far_max": 0.05
  },
  "cases": [
    {
      "id": "case_id",
      "role": "role_for_engine_run",
      "expected_match": true,
      "tags": ["positive", "baseline"],
      "primary_docs": [{"file_path": "path", "doc_class": "photo"}],
      "comparison_docs": [{"file_path": "path", "doc_class": "photo"}]
    }
  ]
}
```

Run:

```powershell
.\.venv\Scripts\python.exe .\scripts\benchmark_suite.py .\benchmarks\sample_manifest.json
```

Outputs:

- `benchmark_results.json`: full per-case payloads plus metrics
- `benchmark_report.md`: human-readable operating-point summary
- `benchmark_scores.csv`: flat score export for external analysis
