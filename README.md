# CA_Monk

CA_Monk is a CPU-first forensic face verification pipeline built for evidence-driven review, not just single-score matching. It combines face matching, still-image PAD, document forensics, 3D reconstruction, reporting, and evidence packaging into one local workflow.

## What It Does

- Extracts images from applicant and comparison documents.
- Detects faces and generates CPU-first embeddings with InsightFace plus AdaFace ONNX when available.
- Runs passive still-image PAD with explicit backend reporting:
  - `onnx_pad+advanced_heuristics` when a local PAD model exists
  - `advanced_heuristic_cpu_pad` otherwise
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

### Expression Transfer API

- `GET /expression-transfer/capabilities`
- `POST /expression-transfer`

### Expression Suite API

- `GET /expression-suite/capabilities`
- `POST /expression-suite`

Example request:

```json
{
  "source_image_path": "path/to/source.jpg",
  "expression_image_path": "path/to/expression.jpg",
  "transfer_pose": false
}
```

This uses the existing Deep3D/BFM coefficients already present in CA_Monk to
transfer facial expression from one image onto another while leaving the main
`/process` pipeline unchanged. It writes:

- composited transfer image
- pure rendered transfer image
- preview strip
- depth, geometry, normal, and side-view renders
- OBJ mesh for MeshLab

The full expression suite extends that with DECA-style capture and animation
artifacts inspired by DECA's published demo surface:

- source expression capture JSON and visual card
- donor expression capture JSON and visual card
- directed preset gallery for review and manual choice
- interpolation animation GIF and keyframe sheet
- profile swing GIF and keyframe sheet
- 360 turntable GIF and keyframe sheet
- suite manifest JSON for downstream review

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
- `src/face_engine/liveness.py`: context-aware still-image PAD backend
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

`run_test.py` now also runs the full expression suite as part of the normal
comparison flow. Alongside the existing forensic outputs, each comparison can
emit:

- expression transfer overlay and render
- source and donor capture cards
- animation GIF
- teaser repose GIF
- OBJ mesh
- suite JSON manifest

Run the API:

```powershell
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

Run expression transfer directly from the repo:

```powershell
.\.venv\Scripts\python.exe .\scripts\run_expression_transfer.py --source path\to\source.jpg --expression path\to\expression.jpg
```

Run the full expression suite directly from the repo:

```powershell
.\.venv\Scripts\python.exe .\scripts\run_expression_suite.py --source path\to\source.jpg --expression path\to\expression.jpg
```

Example with a directed emotion and side view:

```powershell
.\.venv\Scripts\python.exe .\scripts\run_expression_suite.py --source path\to\source.jpg --expression path\to\expression.jpg --preset laughing --yaw 32 --roll 4 --strength 1.15
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

## Note On DECA

The requested DECA repository was not vendored into this codebase. DECA's
upstream license is non-commercial research only and is not a clean drop-in fit
for this repo. Instead, CA_Monk now exposes the requested expression-transfer
workflow natively through the already-integrated Deep3D/BFM stack, so the
current working pipeline stays intact and no external DECA dependency is
required for this feature.

## Accuracy and Trust

CA_Monk is designed for human-reviewed forensic workflows. It is not a substitute for a formally validated production biometric decision engine on its own. The repo now includes the benchmark harness needed to measure operating points, but final thresholds and claims should be based on real validation datasets, not sample/demo cases.
