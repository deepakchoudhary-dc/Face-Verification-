# CA_Monk

CA_Monk is a CPU-first forensic face verification pipeline built for evidence-driven review, not just single-score matching. It combines face matching, still-image PAD, document forensics, 3D reconstruction, reporting, and evidence packaging into one local workflow.

## What It Does

- Extracts images from applicant and comparison documents.
- Detects faces and generates CPU-first embeddings with InsightFace plus AdaFace ONNX when available.
- Runs passive still-image PAD with explicit backend reporting:
  - `onnx_pad+heuristics` when a local PAD model exists
  - `heuristic_cpu_pad` otherwise
- Runs deepfake and rPPG forensics.
- Runs advanced biometric analysis, reconstruction, and report generation.
- Produces a tamper-evident evidence package with:
  - `FINAL_REPORT.md`
  - `EVIDENCE_MANIFEST.json`
  - `INTERACTIVE_CASEFILE.html`
  - dashboard and reconstruction artifacts

## Current Runtime Principles

- CPU-first by default. GPU is optional and opt-in.
- No silent fallback claims. Runtime outputs expose which model path and backend actually ran.
- Quality thresholds are calibrated only from an explicit external directory, never from bundled demo data.
- Missing video is reported as `not_available` for rPPG, not as spoof.

## Core Pipeline

1. Document ingestion and image extraction
2. Primary/comparison biometric extraction
3. Best-pair selection
4. Multi-signal face matching with calibrated confidence
5. Forensics and document intelligence
6. Advanced biometrics
7. Reconstruction
8. Reporting
9. Interactive casefile generation
10. Evidence manifest creation and verification

## Main Surfaces

### Synchronous API

- `POST /process`
- `GET /health`
- `GET /capabilities`
- `POST /evidence/verify`

### Async Job API

- `POST /jobs`
- `GET /jobs`
- `GET /jobs/{job_id}`
- `GET /jobs/{job_id}/result`

### Benchmark Harness

Run a manifest-driven benchmark:

```powershell
.\.venv\Scripts\python.exe .\scripts\benchmark_suite.py .\benchmarks\sample_manifest.json
```

Outputs:

- `benchmark_results.json`
- `benchmark_report.md`
- `benchmark_scores.csv`

The harness now exports threshold sweeps, recommended operating points, confidence separation, and Brier score in addition to simple accuracy.

## Important Files

- `src/core/engine.py`: main orchestration
- `src/face_engine/analyzer.py`: embeddings, PAD, calibrated pair scoring
- `src/face_engine/liveness.py`: pluggable CPU PAD backend
- `src/reporting/interactive_casefile.py`: local HTML review surface
- `src/core/benchmarking.py`: benchmark and threshold analysis
- `src/api/main.py`: FastAPI entrypoint

## Model Notes

Expected local models already used by the repo:

- `models/adaface_ir101_webface12m.onnx`
- `models/codeformer.onnx`
- Deep3D assets under `models/deep3d/`

Optional still-image PAD model:

- Set `CA_MONK_PAD_ONNX_PATH` to a local ONNX PAD model.
- If no PAD ONNX file is present, CA_Monk falls back to heuristic CPU PAD and reports that explicitly.

Useful environment variables:

- `CA_MONK_CALIBRATION_DIR`
- `CA_MONK_PAD_ONNX_PATH`
- `CA_MONK_PAD_LIVE_INDEX`
- `CA_MONK_PAD_ATTACK_LABELS`
- `CA_MONK_ALLOW_GPU`
- `CA_MONK_ORT_THREADS`
- `MAGFACE_NORM_THRESHOLD`

## Running Locally

Install dependencies:

```powershell
pip install -r requirements.txt
```

Run the integration script:

```powershell
.\.venv\Scripts\python.exe .\run_test.py
```

Run the API:

```powershell
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

Example request:

```json
{
  "role": "applicant",
  "primary_docs": [
    {
      "file_path": "path/to/id.jpg",
      "doc_class": "photo"
    }
  ],
  "comparison_docs": [
    {
      "file_path": "path/to/selfie.jpg",
      "doc_class": "photo"
    }
  ]
}
```

## Evidence Package

Each run writes a package under `forensic_output/` containing the generated artifacts for that applicant. The package includes an integrity manifest and an interactive casefile intended for investigator review.

The casefile exposes:

- run metadata and stage telemetry
- artifact catalog
- comparison timeline
- evidence-weight ledger
- contradiction and agreement signals
- face evidence and PAD backend details
- calibrated match trace
- local OBJ wireframe viewer for the reconstruction mesh

## Accuracy and Trust

CA_Monk is designed for human-reviewed forensic workflows. It is not a substitute for a formally validated production biometric decision engine on its own. The repo now includes the benchmark harness needed to measure operating points, but final thresholds and claims should be based on real validation datasets, not sample/demo cases.
