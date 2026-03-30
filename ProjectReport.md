## Quick Run Commands

Run all commands from `E:\CA_Monk`.

Your old command still works for the full sample pipeline:

```powershell
.\.venv\Scripts\python.exe run_test.py
```

### 1. If `.venv` does not exist yet

```powershell
python -m venv .venv
```

### 2. Install Python packages

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 3. Optional: install/download local helper models if anything is missing

```powershell
.\.venv\Scripts\python.exe install_deps.py
.\.venv\Scripts\python.exe .\scripts\setup_deep3d.py
```

### 4. Run the complete bundled sample pipeline

```powershell
.\.venv\Scripts\python.exe run_test.py
```

Outputs:

- `output.json`
- `forensic_output\...`
- interactive casefile HTML
- reconstruction mesh and dashboard artifacts
- expression transfer overlay/render
- expression capture cards and JSON summaries
- directed preset gallery for review and manual choice
- expression interpolation GIF and keyframes
- profile swing GIF and keyframes
- 360 turntable GIF and keyframes
- suite manifest JSON

### 5. Start the API server

```powershell
.\.venv\Scripts\python.exe -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000
```

Useful URLs after starting:

- `http://127.0.0.1:8000/docs`
- `http://127.0.0.1:8000/health`
- `http://127.0.0.1:8000/capabilities`
- `http://127.0.0.1:8000/expression-transfer/capabilities`
- `http://127.0.0.1:8000/expression-suite/capabilities`

### 6. Run the main verification pipeline through the API

```powershell
Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8000/process" -ContentType "application/json" -Body '{"role":"test_subject","primary_docs":[{"file_path":"test_data\\applicant\\primary\\image.png","doc_class":"photo"}],"comparison_docs":[{"file_path":"test_data\\applicant\\compare_with\\image.png","doc_class":"photo"}]}'
```

### 7. Run expression transfer directly from the terminal

Example with bundled sample images:

```powershell
.\.venv\Scripts\python.exe .\scripts\run_expression_transfer.py --source vendor\Deep3DFaceRecon_pytorch\datasets\examples\000002.jpg --expression vendor\Deep3DFaceRecon_pytorch\datasets\examples\000006.jpg
```

If you also want to copy the target head pose:

```powershell
.\.venv\Scripts\python.exe .\scripts\run_expression_transfer.py --source vendor\Deep3DFaceRecon_pytorch\datasets\examples\000002.jpg --expression vendor\Deep3DFaceRecon_pytorch\datasets\examples\000006.jpg --transfer-pose
```

### 8. Run expression transfer through the API

```powershell
Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8000/expression-transfer" -ContentType "application/json" -Body '{"source_image_path":"vendor\\Deep3DFaceRecon_pytorch\\datasets\\examples\\000002.jpg","expression_image_path":"vendor\\Deep3DFaceRecon_pytorch\\datasets\\examples\\000006.jpg","transfer_pose":false}'
```

### 9. Run the full expression suite directly from terminal

```powershell
.\.venv\Scripts\python.exe .\scripts\run_expression_suite.py --source vendor\Deep3DFaceRecon_pytorch\datasets\examples\000002.jpg --expression vendor\Deep3DFaceRecon_pytorch\datasets\examples\000006.jpg
```

With pose transfer too:

```powershell
.\.venv\Scripts\python.exe .\scripts\run_expression_suite.py --source vendor\Deep3DFaceRecon_pytorch\datasets\examples\000002.jpg --expression vendor\Deep3DFaceRecon_pytorch\datasets\examples\000006.jpg --transfer-pose
```

If you want a more directed result, for example laughing in side view:

```powershell
.\.venv\Scripts\python.exe .\scripts\run_expression_suite.py --source vendor\Deep3DFaceRecon_pytorch\datasets\examples\000002.jpg --expression vendor\Deep3DFaceRecon_pytorch\datasets\examples\000006.jpg --preset laughing --yaw 32 --roll 4 --strength 1.15
```

If you want a frightened side-view style:

```powershell
.\.venv\Scripts\python.exe .\scripts\run_expression_suite.py --source vendor\Deep3DFaceRecon_pytorch\datasets\examples\000002.jpg --expression vendor\Deep3DFaceRecon_pytorch\datasets\examples\000006.jpg --preset frightened --yaw 28 --pitch 3 --strength 1.05
```

### 10. Run the full expression suite through the API

```powershell
Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8000/expression-suite" -ContentType "application/json" -Body '{"source_image_path":"vendor\\Deep3DFaceRecon_pytorch\\datasets\\examples\\000002.jpg","expression_image_path":"vendor\\Deep3DFaceRecon_pytorch\\datasets\\examples\\000006.jpg","transfer_pose":false}'
```

Directed API example:

```powershell
Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8000/expression-suite" -ContentType "application/json" -Body '{"source_image_path":"vendor\\Deep3DFaceRecon_pytorch\\datasets\\examples\\000002.jpg","expression_image_path":"vendor\\Deep3DFaceRecon_pytorch\\datasets\\examples\\000006.jpg","expression_preset":"laughing","target_yaw_deg":32,"target_roll_deg":4,"expression_strength":1.15,"transfer_pose":false}'
```

### 11. Stop the API server

Press `Ctrl+C`.

## How To Run This Project

> Status note
>
> Treat this file as a historical and supplemental project report.
> For the live CPU-first runtime, current API endpoints, benchmark workflow, and evidence package outputs, use `README.md` as the source of truth.

This repository currently has two practical run modes:

1. Start the FastAPI server.
2. Run the built-in sample integration flow against `test_data/applicant`.

### 1. Open the project root

Run all commands from `E:\CA_Monk`.

### 2. Create or reuse the virtual environment

```powershell
python -m venv .venv
```

If `.venv` already exists, you can reuse it.

### 3. Install dependencies

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 4. Start the API server

```powershell
.\.venv\Scripts\python.exe -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000
```

Then open:

- `http://127.0.0.1:8000/docs`
- `http://127.0.0.1:8000/health`

Note: the first request can be slow because models are loaded lazily.

### 5. Run the sample codebase pipeline directly

This uses the bundled sample applicant images in `test_data/applicant` and writes results to `output.json` plus a new folder under `forensic_output/`.

```powershell
.\.venv\Scripts\python.exe run_test.py
```

Expected outputs:

- `output.json`
- `forensic_output\evidence_package_<role>_<timestamp>\`
- dashboard and reconstruction artifacts inside that evidence package

### 6. Optional quick checks

```powershell
.\.venv\Scripts\python.exe quick_smoke.py
.\.venv\Scripts\python.exe test_deep3d.py
.\.venv\Scripts\python.exe test_deep3d_real.py
```

### 7. Stop the API server

Press `Ctrl+C` in the terminal running Uvicorn.

### Important current entrypoint note

There is no root-level `main.py` in the current workspace. The active runnable entrypoints are:

- `src.api.main:app` for the API
- `run_test.py` for the sample end-to-end integration run









# CA_MONK - Comprehensive Project Report

## Military-Grade Biometric Intelligence Engine

> **Version:** 6.0.0 - Deep3D Gouraud Renderer + Dual-Branch Forensics
> **Codename:** GOD-TIER Biometric Intelligence
> **Platform:** CPU-first, ONNX-optimized, laptop-safe
> **Target Hardware:** Intel i7/i9, RTX 3060/4060 (GPU optional)
> **Report Date:** February 28, 2026
> **Classification:** INTERNAL TECHNICAL DOCUMENTATION

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [System Architecture](#2-system-architecture)
3. [Full Pipeline Flow](#3-full-pipeline-flow)
4. [Module Inventory](#4-module-inventory--what-we-use)
5. [Reconstruction Engine - Deep3D v6.0](#5-reconstruction-engine--deep3d-v60)
6. [What Was Removed - SD 1.5 Retirement](#6-what-was-removed--sd-15-retirement)
7. [Biometric Analysis Suite](#7-biometric-analysis-suite)
8. [Face Engine](#8-face-engine)
9. [Forensics Engine](#9-forensics-engine)
10. [Document Intelligence](#10-document-intelligence)
11. [Reporting and LLM Integration](#11-reporting--llm-integration)
12. [Configuration and Contracts](#12-configuration--contracts)
13. [Models and Weights Inventory](#13-models--weights-inventory)
14. [Dependencies](#14-dependencies--full-breakdown)
15. [Evidence Package - Deliverables](#15-evidence-package--deliverables)
16. [Performance Benchmarks](#16-performance-benchmarks)
17. [File Inventory](#17-file-inventory--complete-source-tree)
18. [Test Infrastructure](#18-test-infrastructure)
19. [API and Deployment](#19-api--deployment)
20. [Version History](#20-version-history)
21. [Known Limitations and Future Work](#21-known-limitations--future-work)

---

## 1. Executive Summary

**CA_MONK** is a military-grade biometric intelligence engine designed for forensic face verification, identity authentication, and fraud detection. It operates entirely on CPU - no GPU required - making it deployable on standard laptops and field hardware.

### What CA_MONK Does

Given a **primary face image** and a **comparison face image**, the engine:

1. **Detects and aligns** faces using RetinaFace + InsightFace
2. **Extracts embeddings** using a Council of Models (ArcFace, Facenet512, GhostFaceNet)
3. **Performs forensic analysis** - deepfake detection (F3-Net), spectral analysis (FFT), error-level analysis (ELA)
4. **Runs advanced biometrics** - 7-module suite: age-invariant matching, iris analysis, morphing detection, makeup detection, tampering detection, scar/marker analysis, doppelganger detection
5. **Reconstructs faces** using the Deep3D Forensic Pipeline - ResNet50 -> BFM09 3D morphable model -> Gouraud-shaded CPU triangle rasterization -> multi-view renders (geometry, depth, normals, side-view) -> .obj mesh export
6. **Generates an LLM-powered forensic dossier** via local Ollama (qwen3:1.7b)
7. **Produces a military-style evidence dashboard** (1920x1080) with all forensic artifacts
8. **Outputs a structured JSON result** with match verdict, confidence score, and reasoning chain

### v6.0 Key Changes

| Change | Details |
|--------|---------|
| **CPUMeshRenderer rewrite** | 2-pass triangle rasterization (cv2.fillConvexPoly painter's + vectorized barycentric interpolation) replaces v5.1 KDTree+IDW scatter. Gouraud shading, 2x SSAA anti-aliasing, 512 native rendering. |
| **CW winding fix** | BFM mesh uses clockwise winding. Previous cross > 0 filter culled 93.7% of triangles (0.6% face coverage). Fixed to cross < 0 -> 40.2% coverage. |
| **Multi-view renders** | Geometry (Lambertian gray), depth (INFERNO colormap), normal map (RGB), side-view (30 deg yaw), all saved as companion evidence files. |
| **NoisePrint dual-branch** | Added Scharr edge residual branch alongside PRNU bilateral filter. Fused scoring: 0.45 x variance + 0.3 x correlation + 0.25 x edge_ratio. Splice ratio annotation with auto SPLICED_PHOTO detection when face/bg noise > 1.5x. |
| **Dashboard reconstruction panel** | Shows geometry/depth/sideview thumbnails alongside main Deep3D render. 2-column terminal data readout. |
| **Scar analysis fix** | Fixed relative_y KeyError - injury dicts now include proper coordinates, all consumers use defensive .get() with defaults. |

---

## 2. System Architecture

### High-Level Architecture

```
CA_MONK v6 Pipeline Flow

  Input              Biometrics          Forensics         Reconstruction
  +------+          +----------+       +----------+      +------------+
  |JSON/ |--------> |InsightFace|----->|F3-Net    |----->|Deep3D      |
  |Folder|          |ArcFace   |      |rPPG      |      |ResNet50    |
  |S3    |          |AdaFace   |      |ELA/FFT   |      |BFM09 3DMM  |
  +------+          +----------+      +----------+      |Gouraud CPU |
       |                 |                 |              +------------+
       v                 v                 v                    |
  +------+          +----------+       +----------+      +----------+
  |Doc   |          |7-Module  |       |Spectral  |      |Evidence  |
  |Intel |          |Biometric |       |Analysis  |      |Package   |
  |(Donut|          |Suite     |       |NoisePrint|      |Dashboard |
  |OCR)  |          |          |       |Dual-Brch |      |Report    |
  +------+          +----------+       +----------+      +----------+

  LLM Reporting: Ollama (qwen3:1.7b) -> 8-Step Chain-of-Thought Dossier
```

### Package Structure

```
src/                          # 61 Python files, ~14,349 lines total
+-- api/                      # FastAPI REST endpoints (2 files, 59 lines)
+-- biometric_analysis/       # 7-module advanced biometric suite (9 files, 4,684 lines)
+-- core/                     # Engine, config, contracts, quality gate (14 files, 1,877 lines)
+-- document_processor/       # Donut OCR + NoisePrint forgery detection (4 files, 404 lines)
+-- face_engine/              # Face detection, recognition, enhancement, visualization (9 files, 1,957 lines)
+-- forensics/                # F3-Net deepfake, rPPG liveness, spectral (6 files, 393 lines)
+-- input_handlers/           # JSON, folder, S3 input adapters (3 files, 176 lines)
+-- reconstruction/           # Deep3D forensic reconstruction pipeline (11 files, 4,443 lines)
+-- reporting/                # LLM-powered forensic report generation (3 files, 356 lines)
```

### Design Principles

| Principle | Implementation |
|-----------|---------------|
| **CPU-first** | All models run on CPU via ONNX Runtime / PyTorch / OpenCV. No CUDA required. |
| **Lazy Loading** | Heavy modules loaded via @property decorators - only when first accessed. |
| **Graceful Degradation** | Every module wrapped in try/except - pipeline continues if a module fails. |
| **Evidence Chain** | Every processing step saved as a forensic artifact for auditability. |
| **Contract-Driven** | All inter-module communication via Pydantic models (17 contracts). |
| **Identity-Preserving** | Reconstruction NEVER hallucinates features - deterministic BFM coefficient regression. |
| **Multi-View Output** | 3D reconstruction generates textured render, geometry, depth, normals, side-view, and .obj mesh. |

---

## 3. Full Pipeline Flow

The VerificationEngine.process_applicant() method executes the following stages sequentially:

### Stage 1: Input Ingestion
- Input handler (JSON/folder/S3) parses documents
- Images loaded and validated (format, size, quality)

### Stage 2: Face Detection and Alignment
- **RetinaFace** backend detects faces (min confidence 0.90)
- **Rotation handler** tries 0/90/180/270 deg if no face found
- Face cropped and aligned to canonical position

### Stage 3: Embedding Extraction - Council of Models
- **ArcFace** (ONNX, cosine threshold 0.68)
- **Facenet512** (ONNX, cosine threshold 0.40)
- **GhostFaceNet** (ONNX, cosine threshold 0.65)
- Consensus strategy: strict_majority (2 of 3 must agree)

### Stage 4: Forensic Analysis
- **F3-Net Lite DCT** - deepfake probability via frequency analysis
- **rPPG Liveness** - heartbeat extraction from video/image (pyVHR)
- **ELA** - Error Level Analysis for splice detection
- **Spectral Analysis** - FFT high-pass filtering for ghost image detection

### Stage 5: Document Intelligence
- **Donut OCR** - transformer-based document parsing (no traditional OCR)
- **NoisePrint Dual-Branch** - PRNU bilateral noise + Scharr edge residual for image provenance and splice detection

### Stage 6: Advanced Biometric Analysis (7 modules)
- Age-Invariant matching, Iris analysis, Tampering detection, Makeup/Disguise, Morphing detection, Scar/Marker analysis, Doppelganger detection
- Pair analysis: compares biometric signatures between primary and comparison

### Stage 7: Face Reconstruction - Deep3D Forensic Pipeline
- 9-stage pipeline (see Section 5 for full breakdown)
- Generates: textured face render (512x512), geometry render, depth map, normal map, side-view, .obj mesh, evidence chain strip, feathered overlay

### Stage 8: LLM Forensic Report
- Ollama (qwen3:1.7b) generates chain-of-thought analysis
- Mandatory 8-Step Analysis Protocol (Steps 0-7)
- Produces verdict: CLEARED | FLAGGED | Inconclusive | Conclusive Match | Fraud Attempt

### Stage 9: Dashboard and Evidence Package
- Military-style composite dashboard (1920x1080 JPEG, dark theme)
- Evidence directory with 20+ forensic artifacts
- Structured JSON output (output.json)

---

## 4. Module Inventory - What We Use

### Active Technologies and Libraries

| Technology | Purpose | Model/Version | Size |
|-----------|---------|---------------|------|
| **InsightFace** (buffalo_l) | Face detection + alignment + embedding | buffalo_l | ~300MB |
| **ArcFace** | Face recognition embedding | IR-101, WebFace12M | ONNX |
| **Facenet512** | Face recognition embedding | InceptionResNet | Via DeepFace |
| **GhostFaceNet** | Face recognition embedding | Ghost module | Via DeepFace |
| **AdaFace** | Additional recognition model | IR-101, WebFace12M ONNX | ~250MB |
| **RetinaFace** | Face detection backend | MobileNet | Via DeepFace |
| **Deep3DFaceRecon** | 3D face reconstruction (ResNet50 + BFM09) | epoch_20.pth | 289MB |
| **CodeFormer** | Neural face restoration | ONNX | ~400MB |
| **F3-Net Lite DCT** | Deepfake detection | Frequency analysis | Lightweight |
| **pyVHR** | rPPG liveness (heartbeat) | Signal processing | CPU |
| **Donut** | Document OCR | Transformer | Via transformers |
| **NoisePrint** | Camera noise fingerprint (dual-branch: PRNU + Scharr) | Custom CNN | Built-in |
| **Ollama** | LLM forensic reporting | qwen3:1.7b | Local |
| **ONNX Runtime** | Cross-platform inference | >=1.18 | CPU/GPU |
| **OpenCV** | Image processing backbone + triangle rasterization | >=4.8 | Universal |
| **NumPy** | Vectorized barycentric interpolation + array computation | >=1.24 | Core |
| **SciPy** | .mat model loading + interpolation | >=1.11 | Core |
| **PyTorch** | Deep3D ResNet50 backbone + F3-Net + transformers | >=2.2 | Required |
| **Pillow** | Image I/O | >=10.0 | Core |
| **Pydantic** | Contract validation (17 models) | >=2.7 | Core |
| **Matplotlib** | Pulse graphs, heatmaps | >=3.7 | Visualization |
| **FastAPI** | REST API (optional) | >=0.110 | Deployment |

### Custom OpenCV/NumPy Modules (No External DNN)

| Module | File | Lines | Purpose |
|--------|------|------:|---------|
| **CPUMeshRenderer** | reconstruction/deep3d_recon.py | 889 | 2-pass Gouraud triangle rasterizer (cv2.fillConvexPoly + vectorized barycentric) |
| **ForensicFaceReconstructor** | reconstruction/face_reconstructor.py | 798 | Injury/scar detection + symmetry inpainting |
| **SuperResolutionEngine** | reconstruction/super_resolution.py | 201 | Lanczos upscale + unsharp mask + CLAHE |
| **OcclusionRemover** | reconstruction/occlusion_remover.py | 477 | Glasses/mask/bandage removal via symmetry |
| **LightingNormalizer** | reconstruction/lighting_normalizer.py | 199 | Shadow/highlight correction, white balance |
| **FaceGeometryEstimator** | reconstruction/geometry_3d.py | 405 | Shape-from-shading depth + surface normals |
| **AgingSimulator** | reconstruction/aging_simulator.py | 567 | Age progression/regression synthesis |
| **LivenessDetector** | reconstruction/liveness_detector.py | 493 | Reconstruction-level liveness checks |
| **OpenVINOForensicReconstructor** | reconstruction/generative.py | 390 | 9-stage pipeline orchestrator |
| **NoisePrintAnalyzer** | document_processor/noiseprint.py | 210 | Dual-branch (PRNU + Scharr) forgery detection |
| **ForensicVisualizer** | face_engine/visualizer.py | 805 | 1920x1080 military dashboard generator |

---

## 5. Reconstruction Engine - Deep3D v6.0

### Architecture Overview

v6.0 uses **Microsoft's Deep3DFaceReconstruction** (sicxu/Deep3DFaceRecon_pytorch) with a completely rewritten CPU renderer. The ResNet50 backbone regresses **257 BFM coefficients**, decoded into a full 35,709-vertex 3D face mesh with per-vertex color, then rendered via a **2-pass Gouraud-shaded triangle rasterizer** with 2x SSAA.

### 9-Stage Reconstruction Pipeline

```
Input Image (any resolution)
    |
    v
Stage 1: Load and Validate
    |
    v
Stage 2: Deep3D Reconstruction (CORE ENGINE)
  - InsightFace 5-point Keypoints -> face detection + 5 landmarks
  - Least-squares 224x224 alignment -> similarity transform to BFM space
  - ResNet50 Forward Pass -> 24M params, ~100ms on CPU
  - 257 BFM Coefficients: 80 id + 64 exp + 80 tex + 3 rot + 27 SH + 3 trans
  - BFM09 Parametric Reconstruction -> 35,709 vertices, 70,789 triangles
  - Spherical Harmonics lighting -> 3 bands x 9 coefficients x RGB
  - CPU Triangle Rasterizer (v6.0 REWRITTEN):
      Pass 1: cv2.fillConvexPoly -> per-pixel triangle ID map (painter's sort)
      Pass 2: Vectorized barycentric -> Gouraud-shaded per-pixel color + depth + normals
      Pass 3: 2x SSAA downsample -> cv2.INTER_AREA anti-aliasing
      Winding: CW (cross < 0 = front-facing)
  - OUTPUTS: textured render (512x512), geometry, depth, normal map, side-view, overlay, .obj mesh, 68 landmarks
    |
    v
Stage 3: Upscale -> 512x512 (Lanczos interpolation)
    |
    v
Stage 4: Occlusion Removal (Glasses, masks, bandages - symmetry-based inpainting)
    |
    v
Stage 5: Lighting Normalization (Shadow/highlight correction, white balance)
    |
    v
Stage 6: Forensic Reconstruction (Injury/scar aware inpainting)
    |
    v
Stage 7: Super-Resolution (Lanczos + unsharp + CLAHE)
    |
    v
Stage 8: CodeFormer ONNX (Neural face restoration - optional)
    |
    v
Stage 9: Evidence Chain + Mesh Save
  -> render + geometry + depth + normals + side-view + .obj mesh
  -> multi-stage evidence chain strip
```

### Core Classes (deep3d_recon.py - 889 lines)

#### _ResNet50 (ResNet50 backbone)
- Standard ResNet50 with _Bottleneck blocks [3, 4, 6, 3]
- use_last_fc=False - outputs (B, 2048, 1, 1) feature vector
- Kaiming normal initialization

#### ReconNetWrapper (Coefficient regression network)
- Wraps _ResNet50 backbone
- **7 parallel conv1x1 heads** regress 257 BFM coefficients:
  - Identity: 80, Expression: 64, Texture: 80
  - Rotation angles: 3, SH illumination: 27, Translation: 3 (tx, ty, tz)
- **24,034,625 parameters**, ~160MB
- Pretrained: models/deep3d/checkpoints/epoch_20.pth (289MB)

#### ParametricFaceModel (Basel Face Model 2009)
- **35,709 vertices**, 70,789 triangles, 68 keypoints
- Disentangled reconstruction: shape = mean_shape + id_base @ id_coeff + exp_base @ exp_coeff
- Spherical Harmonics illumination (27 coefficients: 3 bands x 9 x RGB)
- Perspective projection: focal=1015, center=112, camera_distance=10
- Methods: split_coeff, compute_shape, compute_texture, compute_norm, compute_color, compute_rotation, reconstruct
- Loaded from models/deep3d/BFM/BFM_model_front.mat (127MB)

#### CPUMeshRenderer (v6.0 - Production triangle rasterizer)
- **Replaces nvdiffrast** (GPU-only renderer from original repo)
- **Replaces v5.1 KDTree+IDW** (scatter interpolation - poor quality)
- **2-pass Gouraud-shaded triangle rasterization:**
  - **Pass 1:** Painter's algorithm - triangles sorted by average Z depth (back-to-front). Each triangle rasterized via cv2.fillConvexPoly into a per-pixel triangle-ID buffer. This leverages OpenCV's C++ scanline rasterizer for speed.
  - **Pass 2:** Fully-vectorized NumPy barycentric coordinate interpolation. For every pixel, computes barycentric weights from triangle vertices, then interpolates: vertex colors (Gouraud shading), depth values, and surface normals. Zero Python loops.
  - **Pass 3:** 2x SSAA (SuperSample Anti-Aliasing) - renders at 2x resolution, then downsamples via cv2.INTER_AREA for smooth triangle edges.
- **Winding order:** CW (clockwise) - BFM standard. Front-facing test: cross < 0 in screen space.
- **Outputs per render call:**
  - rendered - Gouraud-shaded textured face (BGR, 512x512)
  - geometry - Lambertian gray shading (gray, 512x512)
  - depth - raw Z-buffer values
  - depth_colored - INFERNO colormap depth map (BGR)
  - normal_map - RGB-encoded surface normals
  - mask - binary face silhouette
- **render_rotated()** - side-view at arbitrary yaw angle (default 30 deg)
- **Performance:** ~350ms per render on CPU (i7-13700H)

#### Deep3DFaceReconstructor (Top-level reconstruction engine)
- Lazy initialization - ResNet50, BFM, InsightFace loaded on first reconstruct() call
- **Methods:**
  - reconstruct(image_bgr) -> dict with: rendered face (224 and 512), geometry render, depth map, normal map, side-view, overlay, .obj mesh data, 68 landmarks, 257 coefficients, aligned input
  - save_obj(result, path) -> exports .obj mesh with per-vertex colors
  - cleanup() -> releases all model memory
- **Performance:** ~8.4s first call (includes InsightFace model load), ~2.4s cached

### OpenVINOForensicReconstructor (generative.py - 390 lines)
- Main 9-stage pipeline orchestrator
- **Lazy-loaded** sub-modules via @property decorators:
  - deep3d -> Deep3DFaceReconstructor
  - face_reconstructor -> ForensicFaceReconstructor
  - occlusion_remover -> OcclusionRemover
  - lighting_normalizer -> LightingNormalizer
  - super_resolution -> SuperResolutionEngine
  - codeformer -> CodeFormerONNXRestorer (optional)
- **Saves per face:** reconstruction_hq.jpg, _chain.jpg, _depth.jpg, _geometry.jpg, _sideview.jpg, _normals.jpg, _mesh.obj
- Contract: ReconstructionRequest -> ReconstructionResponse

### Key Design Decisions (v6.0)

1. **True 3DMM Fitting**: Actual Basel Face Model (BFM09) with ResNet50-regressed coefficients. Disentangled shape/expression/texture/illumination.
2. **Real 3D Mesh**: 35,709-vertex mesh with per-vertex color, exportable as .obj. Depth maps from actual 3D geometry.
3. **Production CPU Renderer**: 2-pass Gouraud with 2x SSAA. Correct CW winding order for BFM meshes.
4. **Identity Preservation**: ResNet50 trained with ArcFace loss ensures reconstructed face preserves identity.
5. **Multi-View Output**: Single forward pass generates front, geometry, depth, normals; second pass generates side-view.
6. **Graceful Degradation**: If Deep3D models are missing, pipeline falls back to direct resize.

---

## 6. What Was Removed - SD 1.5 Retirement

### Removed Components

| Component | Size | Purpose | Reason for Removal |
|-----------|------|---------|-------------------|
| **Stable Diffusion 1.5 Realistic Vision v6.0** | ~2.5GB | Face reconstruction via img2img | 3-5 min/image on CPU, identity hallucination |
| **diffusers** library | ~500MB installed | SD pipeline orchestration | No longer needed |
| **accelerate** library | ~50MB installed | Model offloading | No longer needed |
| **UNet** (SD 1.5) | ~1.7GB | Denoising backbone | Replaced by Deep3D |
| **VAE ft-mse** | ~335MB | Latent-to-pixel decoder | Not needed |
| **Text Encoder** (CLIP) | ~490MB | Prompt conditioning | Not needed |
| **IP-Adapter FaceID** | ~100MB+ | Identity conditioning for SD | Not needed |

### Performance Impact of Removal

| Metric | SD 1.5 (v4) | Deep3D v5.0 | Deep3D v5.1 | **Deep3D v6.0** |
|--------|-------------|-------------|-------------|----------------|
| **Time/Image** | 180-300s | 5-6s | ~500ms cached | **~2.4s cached** |
| **Peak RAM** | ~3.0 GB | ~500 MB | ~600 MB | **~600 MB** |
| **Model Size** | ~2.5 GB | ~3.7 MB | ~467 MB | **~467 MB** |
| **Render Quality** | SD hallucination | MediaPipe proxy | KDTree scatter | **Gouraud + 2x SSAA** |
| **Identity** | Poor | Good | Good | **Excellent (ArcFace)** |
| **Face Coverage** | N/A | N/A | 0.6% (winding bug) | **40.2% (CW fixed)** |

### Legacy Artifacts (Still on Disk)

- models/sd15_rv6/ - SD 1.5 Realistic Vision weights (~2.5GB) - **can be safely deleted**
- models/vae_ft_mse/ - Fine-tuned MSE VAE weights (~335MB) - **can be safely deleted**

---

## 7. Biometric Analysis Suite

The AdvancedBiometricOrchestrator (src/biometric_analysis/orchestrator.py, 213 lines) runs 7 specialized biometric modules on each face image. All modules are wrapped in try/except for graceful degradation.

### Module Breakdown

#### 1. Age-Invariant Analysis (age_invariant.py - 392 lines)
- **Purpose**: Identifies structural bone landmarks that remain stable across aging
- **Output**: Bone signature hash, stability score (0-100)
- **Pair Analysis**: Compares bone signatures between images, estimates age gap
- **Technology**: OpenCV landmark geometry + hash distance

#### 2. Tampering Detection (tampering_detector.py - 592 lines)
- **Purpose**: Detects image manipulation (splicing, copy-move, inpainting)
- **Output**: Tampering probability (%), risk level (LOW/MEDIUM/HIGH/CRITICAL)
- **Technology**: Error Level Analysis (ELA), noise consistency checks

#### 3. Makeup and Disguise Detection (makeup_detector.py - 511 lines)
- **Purpose**: Identifies cosmetic manipulation, colored contacts, prosthetics
- **Output**: Makeup probability (%), naturalness score, specific detections
- **Technology**: Color histogram analysis, skin texture uniformity

#### 4. Iris Analysis (iris_analyzer.py - 591 lines)
- **Purpose**: Iris pattern extraction, quality assessment, health scoring
- **Output**: 144 radial features, quality score, health score, estimated eye age
- **Anti-Spoof**: Checks for specular reflection (real vs. printed eye)
- **Technology**: Gabor filter bank, radial feature extraction

#### 5. Uniqueness and Doppelganger Detection (doppelganger_detector.py - 671 lines)
- **Purpose**: Identifies unique facial markers (moles, asymmetry, proportions)
- **Output**: Uniqueness score, mole count, asymmetry index
- **Pair Analysis**: Anti-doppelganger confidence based on marker pattern matching
- **Technology**: Connected component analysis, bilateral symmetry metrics

#### 6. Morphing Detection (morphing_detector.py - 609 lines)
- **Purpose**: Detects face morphing attacks (blended identities)
- **Output**: Morphing probability (%), confidence score
- **Technology**: Texture analysis, frequency-domain artifacts

#### 7. Scar and Facial Marker Analysis (scar_analysis.py - 743 lines)
- **Purpose**: Identifies permanent markers - scars, birthmarks, injuries
- **Output**: Marker list with type, location (relative_x, relative_y), and forensic significance
- **v6.0 Fix**: relative_y KeyError resolved - injury dicts now include proper relative_x/relative_y coordinates, all consumer methods use defensive .get('relative_y', 0.5) access

### Biometric Orchestrator (__init__.py - 362 lines)
- Wraps all 7 modules with try/except safety
- Computes per-image analysis + pair comparison
- Produces threat level aggregation

### Pair Verdict System

| Verdict | Meaning |
|---------|---------|
| CONFIRMED_SAME_PERSON | High confidence match across biometric dimensions |
| LIKELY_MATCH | Moderate confidence, some biometric alignment |
| INCONCLUSIVE | Insufficient evidence for determination |
| LIKELY_DIFFERENT | Biometric patterns suggest different individuals |
| CONFIRMED_DIFFERENT | Strong evidence of different identity |

---

## 8. Face Engine

### Core Components

#### FaceAnalyzer (face_engine/analyzer.py - 282 lines)
- Face detection via DeepFace (RetinaFace backend)
- Multi-model embedding extraction (Council of Models)
- Face quality assessment (blur, exposure, occlusion)

#### Enhancement (face_engine/enhancement.py - 33 lines)
- Pre-processing: histogram equalization, denoising
- Contrast-limited adaptive histogram equalization (CLAHE)

#### Liveness Detection (face_engine/liveness.py - 32 lines)
- Texture-based anti-spoofing (LBP analysis)
- Moire pattern detection (print attack)

#### Rotation Handler (face_engine/rotation_handler.py - 63 lines)
- Tests 0/90/180/270 deg orientations if face detection fails
- Returns best orientation with highest detection confidence

#### Recognition (face_engine/recognition.py - 93 lines)
- Cosine/Euclidean distance computation
- Threshold-based verification per model
- Consensus aggregation (strict_majority)

#### Image Study (face_engine/image_study.py - 489 lines)
- Primary image deep analysis - face quality, marks, lighting, pose
- Reports: blur score, noise level, exposure, skin uniformity, redness ratio, texture roughness
- Detects: dark spots, red spots, scar-like regions, asymmetry, occlusions, uneven lighting

#### Restoration (face_engine/restoration.py - 59 lines)
- **CodeFormerONNXRestorer**: ONNX-based neural face restoration (~400MB model)
- Used as final stage of reconstruction pipeline (optional)

#### Siamese GradCAM (face_engine/siamese_gradcam.py - 101 lines)
- Explainability overlay showing which facial regions contributed to match/mismatch decision
- Per-region contribution scoring (mouth, eyes, nose, forehead, etc.)

#### ForensicVisualizer (face_engine/visualizer.py - 805 lines)
- **1920x1080** military-grade evidence dashboard generator
- Dark theme (#0a0a0a) with neon accent system
- **Dashboard panels:**
  - Header: CLASSIFIED banner with role/timestamp
  - Left: Input images (primary + comparison) + face crops
  - Center: GradCAM overlay + forensic signals
  - Right: Reconstruction panel with 200x200 main render + 140x140 thumbnails (geometry, depth, side-view)
  - Bottom: 7-module biometric analysis grid
  - Footer: Final verdict stamp (CLEARED / FLAGGED)
- **v6.0 upgrade:** Reconstruction thumbnails for geometry/depth/sideview, 2-column terminal data readout, Deep3D + BFM RECONSTRUCTION label

---

## 9. Forensics Engine

### F3-Net Deepfake Detection (forensics/f3net_detector.py - 102 lines)
- Frequency-domain deepfake detector using DCT (Discrete Cosine Transform)
- Lightweight variant (f3net_lite_dct)
- Output: deepfake probability (0.0-1.0), threshold at 0.60

### rPPG Liveness (forensics/rppg_liveness.py - 2 lines)
- Remote photoplethysmography - extracts heartbeat from face color changes
- Uses pyVHR library for signal processing
- Output: is_live, BPM, confidence, method

### Forensic Liveness (forensics/liveness.py - 169 lines)
- Extended liveness analysis module
- Multi-signal fusion for liveness determination

### Frequency Extractor (forensics/frequency_extractor.py - 21 lines)
- FFT-based high-pass filtering for hidden patterns
- Ghost image detection in frequency domain

### Spectral Analysis (core/spectral_analysis.py - 69 lines)
- FFT high-pass ghost image heatmap generation
- Outputs: spectral_analysis.jpg

### Forensics Service (forensics/service.py - 89 lines)
- Orchestrates F3-Net + rPPG + frequency analysis
- Contract: ForensicsRequest -> ForensicsResponse

---

## 10. Document Intelligence

### Donut OCR Parser (document_processor/donut_parser.py - 70 lines)
- Transformer-based document understanding (no traditional OCR pipeline)
- Extracts structured fields from identity documents

### NoisePrint Dual-Branch Forgery Detection (document_processor/noiseprint.py - 210 lines)
- **Branch 1 - PRNU Residual**: Bilateral filter noise extraction -> face vs. background variance discrepancy
- **Branch 2 - Scharr Edge Residual** (NEW in v6.0): cv2.Scharr high-pass edge magnitude -> splice boundary detection
- **Fusion**: 0.45 x variance_discrepancy + 0.3 x correlation_gap + 0.25 x edge_ratio
- **Tamper Heatmap**: Fused ELA (60%) + Scharr (40%) per-pixel map, Gaussian blur smoothing, INFERNO colormap
- **Splice Detection**: Auto-annotates face/background noise ratio; triggers SPLICED_PHOTO flag when ratio > 1.5x
- **Methods:**
  - _residual() - PRNU bilateral filter noise residual
  - _scharr_residual() - Scharr high-pass edge residual
  - _split_regions() - Split face ROI vs background regions
  - _correlation() - Cross-correlation between face/background noise
  - compute_ela() - Error Level Analysis (JPEG compression inconsistency)
  - generate_tamper_heatmap() - Dual-branch pixel-level tampering heatmap
  - analyze() - Full NoisePrint analysis returning NoisePrintResult

### Document Analysis (document_processor/analysis.py - 46 lines)
- Higher-level document analysis combining OCR + forgery detection

### Document Extractor (document_processor/extractor.py - 78 lines)
- Main document processing orchestrator
- Contract: DocumentRequest -> DocumentResponse

---

## 11. Reporting and LLM Integration

### LLM Analyst (reporting/llm_analyst.py - 348 lines)
- **Class**: LlamaForensicAnalyst
- Connects to local **Ollama** server (default: http://127.0.0.1:11434)
- Model: **qwen3:1.7b** (lightweight, CPU-friendly, 1.7B parameters)
- Configurable via environment variables:
  - OLLAMA_MODEL - model name (default: qwen3:1.7b)
  - OLLAMA_BASE_URL - server URL
  - OLLAMA_TIMEOUT_SECONDS - timeout (default: 120)
  - OLLAMA_NUM_CTX - context window (default: 4096)
  - OLLAMA_TEMPERATURE - temperature (default: 0.1)

### 8-Step Mandatory Analysis Protocol

| Step | Name | Analysis |
|------|------|----------|
| **Step 0** | PRIMARY IMAGE DEEP STUDY | Skin condition, marks/scars, occlusions, symmetry, aging, quality metrics |
| **Step 1** | VISUAL TRIAGE | Embedding norm fidelity tier (<22 LOW, >28 HIGH) |
| **Step 2** | SPECTRAL SCAN | F3-Net deepfake probability, spectral anomaly check |
| **Step 3** | BIOLOGICAL SCAN | rPPG liveness, BPM, confidence |
| **Step 4** | STRUCTURAL INTEGRITY | Cosine similarity, NoisePrint splice detection |
| **Step 5** | ADVANCED BIOMETRICS | Threat levels, tampering, morphing, doppelganger, iris, markers |
| **Step 6** | RECONSTRUCTION ASSESSMENT | Deep3D pipeline output, identity preservation |
| **Step 7** | RUTHLESS EXECUTIVE SUMMARY | 3 paragraphs: proven facts, unknowns, operational recommendation |

### System Prompt Features
- Mandatory thinking tags for chain-of-thought reasoning
- Zero-hallucination policy - must cite exact numeric values
- Strict JSON output format with 5 allowed verdicts
- Temperature 0.1 for deterministic output

### Report Verdicts

| Verdict | Description |
|---------|-------------|
| Conclusive Match | High-confidence identity match confirmed |
| CLEARED | Subject cleared, no forensic concerns |
| Inconclusive | Cannot determine with available evidence |
| FLAGGED | Suspicious indicators requiring human review |
| Fraud Attempt | Evidence of deliberate identity fraud |

### Llama Reporter (reporting/llama_reporter.py - 9 lines)
- Alternative/extended reporting module entry point

---

## 12. Configuration and Contracts

### AppConfig (core/config.py - 44 lines)

| Setting | Value | Description |
|---------|-------|-------------|
| DETECTOR_BACKEND | retinaface | Face detection model |
| MIN_FACE_CONFIDENCE | 0.90 | Minimum detection confidence |
| MIN_FACE_SIZE | 30px | Minimum face pixel size |
| MAX_FACE_AREA_RATIO | 0.85 | Maximum face-to-image ratio |
| RECOGNITION_MODELS | ArcFace, Facenet512, GhostFaceNet | Council of Models |
| CONSENSUS_STRATEGY | strict_majority | 2 of 3 models must agree |
| ENABLE_ENHANCEMENT | True | Pre-processing enabled |
| ENABLE_ROTATION_CHECK | True | Auto-rotation detection |
| ROTATION_ANGLES | [0, 90, 180, 270] | Angles to test |
| ENABLE_ELA | True | Error Level Analysis |
| ENABLE_SPECTRAL | True | FFT spectral analysis |
| DEEPFAKE_THRESHOLD | 0.60 | F3-Net threshold |

### Match Thresholds

| Model | Cosine Distance Threshold |
|-------|--------------------------|
| ArcFace | 0.68 |
| Facenet512 | 0.40 |
| GhostFaceNet | 0.65 |
| Default | 0.60 |

### Pydantic Contracts (core/contracts.py - 164 lines, 17 models)

| Contract | Direction | Purpose |
|----------|-----------|---------|
| FaceBox | Internal | Bounding box (x, y, w, h) |
| EmbeddingResult | Bio -> Engine | Embedding + quality + confidence |
| PairMatchRequest | Engine -> Match | Primary + comparison embeddings |
| PairMatchResult | Match -> Engine | Similarity, verified, threshold, rationale |
| BiometricsRequest | Engine -> Bio | Image path + calibration |
| BiometricsResponse | Bio -> Engine | Embeddings + quality + warnings |
| ForensicsRequest | Engine -> Forensics | Image + video path |
| FrequencyResult | F3Net -> Engine | Deepfake probability |
| RPPGResult | rPPG -> Engine | Liveness, BPM, confidence |
| ForensicsResponse | Forensics -> Engine | Frequency + rPPG + warnings |
| DocumentRequest | Engine -> Doc | Image path + face box |
| NoisePrintResult | NoisePrint -> Engine | Noise variance + splice flag |
| DocumentResponse | Doc -> Engine | OCR fields + noiseprint |
| ReconstructionRequest | Engine -> Recon | Image + mode + guidance |
| ReconstructionResponse | Recon -> Engine | Generated path + warnings |
| ReportRequest | Engine -> LLM | Full analysis payload |
| ReportResponse | LLM -> Engine | Verdict + confidence + reasoning |

---

## 13. Models and Weights Inventory

### Active Models

| Model | Path | Format | Size | Used By |
|-------|------|--------|------|---------|
| Deep3D ResNet50 | models/deep3d/checkpoints/epoch_20.pth | PyTorch | 289 MB | 3D face reconstruction (coefficient regression) |
| BFM09 (Basel Face Model) | models/deep3d/BFM/BFM_model_front.mat | MATLAB | 127 MB | Parametric 3D face model |
| BFM Expression Basis | models/deep3d/BFM/Exp_Pca.bin | Binary | 51 MB | Expression PCA basis |
| AdaFace IR-101 | models/adaface_ir101_webface12m.onnx | ONNX | ~250 MB | Face recognition |
| CodeFormer | models/codeformer.onnx | ONNX | ~400 MB | Neural face restoration (optional) |
| InsightFace buffalo_l | Auto-downloaded to ~/.insightface/models/buffalo_l/ | ONNX | ~300 MB | Face detection + alignment |

### Legacy Models (No Longer Used)

| Model | Path | Size | Status |
|-------|------|------|--------|
| SD 1.5 Realistic Vision v6 | models/sd15_rv6/ | ~2.5 GB | **UNUSED** - safe to delete |
| VAE ft-mse | models/vae_ft_mse/ | ~335 MB | **UNUSED** - safe to delete |
| MediaPipe Face Landmarker | models/face_landmarker.task | 3.7 MB | **UNUSED** - replaced by Deep3D BFM landmarks |

### Model Download

```bash
# Download all Deep3D models (epoch_20.pth, BFM_model_front.mat, Exp_Pca.bin)
python setup_deep3d.py

# InsightFace buffalo_l - auto-downloaded on first use
# CodeFormer ONNX - must be present at models/codeformer.onnx (optional)
```

---

## 14. Dependencies - Full Breakdown

### Production Dependencies (requirements.txt - 60 lines)

| Package | Version | Purpose | Critical? |
|---------|---------|---------|-----------|
| numpy | >=1.24 | Array computation + vectorized barycentric interpolation | Yes |
| scipy | >=1.11 | BFM .mat file loading + interpolation | Yes |
| pandas | >=1.5 | Data handling | No |
| pydantic | >=2.7 | Contract validation (17 models) | Yes |
| python-dotenv | >=1.0 | Environment config | No |
| opencv-python | >=4.8 | Image processing + triangle rasterization (fillConvexPoly) | Yes |
| Pillow | >=10.0 | Image I/O | Yes |
| pdf2image | >=1.16 | PDF document extraction | No |
| boto3 | >=1.28 | S3 input handler | No |
| openpyxl | >=3.1 | Excel data handling | No |
| onnxruntime | >=1.18 | ONNX model inference (CodeFormer, AdaFace) | Yes |
| openvino | >=2025.0 | Model optimization | No |
| openvino-genai | >=2025.0 | OpenVINO generation | No |
| optimum-intel | >=1.18 | Intel optimized models | No |
| insightface | >=0.7.3 | Face detection + 5-point alignment for Deep3D | Yes |
| transformers | >=4.44 | Donut OCR, model loading | Yes |
| trimesh | >=4.0 | 3D mesh export (.obj) | No |
| gdown | >=5.0 | Google Drive model downloads | No |
| kornia | >=0.7 | Image augmentation/alignment | No |
| pyVHR | >=2.0 | rPPG liveness detection | No |
| mediapipe | >=0.10 | Legacy (optional) | No |
| matplotlib | >=3.7 | Visualization (pulse, heatmaps) | No |
| fastapi | >=0.110 | REST API | No |
| uvicorn | >=0.30 | ASGI server | No |
| torch | >=2.2 | Deep3D ResNet50 + BFM reconstruction + F3-Net | Yes |

### Removed Dependencies

| Package | Was Version | Removed In | Reason |
|---------|------------|------------|--------|
| diffusers | >=0.25 | v5.0 | SD 1.5 pipeline no longer used |
| accelerate | >=0.25 | v5.0 | Model offloading no longer needed |

### External Services

| Service | Default URL | Required? | Purpose |
|---------|------------|-----------|---------|
| Ollama | http://127.0.0.1:11434 | Optional | LLM forensic reporting (qwen3:1.7b) |

---

## 15. Evidence Package - Deliverables

### Per-Run Output (20 artifacts)

Each pipeline run generates a structured evidence package:

```
forensic_output/evidence_package_{role}_{timestamp}/
+-- FINAL_REPORT.md                              # LLM-generated forensic dossier
+-- FINAL_DASHBOARD_{role}.jpg                   # Military-style composite dashboard (1920x1080)
|
+-- reconstruction_hq.jpg                        # Deep3D reconstructed face - comparison (512x512)
+-- reconstruction_hq_chain.jpg                  # Multi-stage evidence chain strip
+-- reconstruction_hq_depth.jpg                  # 3D depth map (INFERNO colormap)
+-- reconstruction_hq_geometry.jpg               # Lambertian geometry render (gray)
+-- reconstruction_hq_normals.jpg                # RGB-encoded surface normal map
+-- reconstruction_hq_sideview.jpg               # 30 deg side-view render
+-- reconstruction_hq_mesh.obj                   # 35K-vertex .obj mesh (~3.3 MB)
|
+-- primary_reconstruction_hq.jpg                # Deep3D reconstructed face - primary
+-- primary_reconstruction_hq_chain.jpg          # Primary evidence chain
+-- primary_reconstruction_hq_depth.jpg          # Primary depth map
+-- primary_reconstruction_hq_geometry.jpg       # Primary geometry render
+-- primary_reconstruction_hq_normals.jpg        # Primary normal map
+-- primary_reconstruction_hq_sideview.jpg       # Primary side-view
+-- primary_reconstruction_hq_mesh.obj           # Primary .obj mesh (~3.3 MB)
|
+-- spectral_analysis.jpg                        # FFT high-pass ghost image heatmap
+-- tamper_heatmap.jpg                           # Dual-branch (ELA + Scharr) tampering heatmap
+-- biometric_pulse.png                          # rPPG liveness pulse graph
+-- gradcam_overlay.png                          # Siamese occlusion CAM explainability
```

### Dashboard Output

```
evidence_cards/DASHBOARD_{role}_{timestamp}.jpg   # ~380 KB, 1920x1080
```

### JSON Output

Structured results written to output.json containing:
- Match verdict (is_match: boolean)
- Confidence score (0-100%)
- Per-model match results (ArcFace, Facenet512, GhostFaceNet)
- Forensic analysis results (F3-Net, ELA, NoisePrint)
- Advanced biometric analysis (7 modules + pair verdict)
- Reconstruction pipeline warnings/stages
- Primary image deep study (quality metrics, marks, lighting)
- Explainability data (contributing regions)
- LLM report (verdict, confidence, reasoning steps)

---

## 16. Performance Benchmarks

### Last Test Run (v6.0, February 28, 2026)

| Metric | Value |
|--------|-------|
| **Deep3D Reconstruction (first call)** | ~8.4s (includes InsightFace model load) |
| **Deep3D Reconstruction (cached)** | ~2.4s per face |
| **ResNet50 Inference** | ~100ms |
| **BFM Reconstruction** | ~35ms |
| **CPU Triangle Rasterization (512x512, 2x SSAA)** | ~350ms |
| **3D Mesh Vertices** | 35,709 |
| **3D Mesh Triangles** | 70,789 (CW winding) |
| **Face Coverage** | 40.2% of 512x512 canvas |
| **BFM Coefficients** | 257 (80 id + 64 exp + 80 tex + 3 rot + 27 SH + 3 trans) |
| **Landmark Projection** | 68 keypoints |
| **Output .obj Mesh** | ~3.3 MB |
| **Render Outputs Per Face** | 7 (rendered, geometry, depth, normals, sideview, overlay, chain) |
| **Peak RAM** | ~600 MB (ResNet50 + BFM + InsightFace + CodeFormer) |
| **Total Pipeline (full run)** | 13 stages, ~30s including all biometrics |
| **Biometric Suite (7 modules)** | ~549ms total |
| **Total Evidence Artifacts** | 20 files per run |

### Version Comparison

| Metric | SD 1.5 (v4) | MediaPipe (v5.0) | KDTree+IDW (v5.1) | **Triangle Raster (v6.0)** |
|--------|-------------|------------------|-------------------|-----------------------|
| **Time/Image** | 180-300s | 5-6s | ~500ms | **~2.4s** (6 renders) |
| **Peak RAM** | ~3.0 GB | ~500 MB | ~600 MB | **~600 MB** |
| **Model Size** | ~2.5 GB | ~3.7 MB | ~467 MB | **~467 MB** |
| **Render Quality** | SD hallucination | Flat depth proxy | Blocky scatter | **Gouraud + 2x SSAA** |
| **Face Coverage** | N/A | N/A | 0.6% | **40.2%** |
| **3D Mesh** | None | None | 35K vert .obj | **35K vert .obj** |
| **Multi-View** | None | None | None | **Geometry + Depth + Normal + Side** |
| **Identity** | Poor (stochastic) | Good | Good | **Excellent (ArcFace)** |
| **Evidence Artifacts** | 1 | ~8 | ~10 | **20** |

### Test Results (Last Run - Feb 28, 2026)

| Field | Value |
|-------|-------|
| **Role** | test_subject |
| **Primary Faces Found** | 1 |
| **is_match** | false |
| **Confidence** | 53.61% |
| **Cosine Similarity** | 0.072 |
| **Verification Threshold** | 0.35 |
| **Biometric Pair Verdict** | LIKELY_MATCH (60% confidence) |
| **Threat Level** | MEDIUM |
| **Biometric Threat Score** | 0.1 (LOW) |
| **Age-Invariant Match** | is_same_person=true, confidence=99.01% |
| **Estimated Age Gap** | 10 years |
| **Doppelganger Flag** | LIKELY_DOPPELGANGER (96.39%) |
| **Iris Anti-Spoof** | is_real_eye=false (no specular reflection) |
| **Morphing** | Not morphed (probability=12%, confidence=92%) |
| **Tampering** | Not detected (probability=25.5%, risk=LOW) |
| **Makeup** | Not detected (probability=28%, possible colored contacts) |

---

## 17. File Inventory - Complete Source Tree

### Source Files (61 Python files, ~14,349 lines across 9 packages)

#### src/api/ - REST API (2 files, 59 lines)
| File | Lines | Purpose |
|------|------:|---------|
| __init__.py | 2 | Package init |
| main.py | 57 | FastAPI application (health + /process endpoint) |

#### src/biometric_analysis/ - Advanced Biometrics (9 files, 4,684 lines)
| File | Lines | Purpose |
|------|------:|---------|
| __init__.py | 362 | Module wrappers with try/except safety |
| orchestrator.py | 213 | 7-module biometric orchestrator + pair verdict |
| age_invariant.py | 392 | Bone structure stability analysis |
| doppelganger_detector.py | 671 | Uniqueness + anti-doppelganger detection |
| iris_analyzer.py | 591 | Iris pattern + health + anti-spoof |
| makeup_detector.py | 511 | Cosmetic manipulation detection |
| morphing_detector.py | 609 | Face morph attack detection |
| scar_analysis.py | 743 | Scar, birthmark, injury marker extraction (v6.0 fixed) |
| tampering_detector.py | 592 | Image manipulation detection |

#### src/core/ - Engine Core (14 files, 1,877 lines)
| File | Lines | Purpose |
|------|------:|---------|
| engine.py | 654 | Main VerificationEngine - 9-stage pipeline orchestrator |
| config.py | 44 | AppConfig with environment-driven settings |
| contracts.py | 164 | 17 Pydantic contract models |
| data_structures.py | 15 | Applicant + Document dataclasses |
| quality_gate.py | 59 | Face quality gate (blur, exposure, occlusion) |
| interfaces.py | 33 | Abstract base classes |
| iso_compliance.py | 90 | ISO/IEC 19795 compliance checks |
| compliance_engine.py | 188 | Compliance reporting |
| forensics_engine.py | 81 | Forensic analysis engine wrapper |
| spectral_analysis.py | 69 | FFT ghost image detection |
| model_manager.py | 196 | Model lifecycle management |
| model_swapper.py | 31 | Hot model swapping |
| gpu_guard.py | 24 | GPU memory guard (CPU fallback) |
| serve_deployments.py | 229 | Deployment configuration |

#### src/document_processor/ - Document Intelligence (4 files, 404 lines)
| File | Lines | Purpose |
|------|------:|---------|
| extractor.py | 78 | Main document processing orchestrator |
| donut_parser.py | 70 | Transformer-based OCR |
| noiseprint.py | 210 | Dual-branch (PRNU + Scharr) forgery detection (v6.0 enhanced) |
| analysis.py | 46 | High-level document analysis |

#### src/face_engine/ - Face Processing (9 files, 1,957 lines)
| File | Lines | Purpose |
|------|------:|---------|
| analyzer.py | 282 | Face detection + embedding extraction |
| enhancement.py | 33 | Image pre-processing (CLAHE, denoising) |
| image_study.py | 489 | Primary image deep analysis (quality, marks, lighting) |
| liveness.py | 32 | Anti-spoofing checks (LBP, moire) |
| recognition.py | 93 | Cosine distance + consensus matching |
| restoration.py | 59 | CodeFormer ONNX restorer |
| rotation_handler.py | 63 | Auto-rotation detection (0/90/180/270 deg) |
| siamese_gradcam.py | 101 | Explainability overlay (Siamese Occlusion CAM) |
| visualizer.py | 805 | Military dashboard generator (1920x1080) (v6.0 upgraded) |

#### src/forensics/ - Forensic Analysis (6 files, 393 lines)
| File | Lines | Purpose |
|------|------:|---------|
| __init__.py | 10 | Package init |
| f3net_detector.py | 102 | F3-Net deepfake detection (DCT frequency) |
| frequency_extractor.py | 21 | FFT frequency analysis |
| liveness.py | 169 | Extended forensic liveness |
| rppg_liveness.py | 2 | Remote PPG heartbeat extraction (pyVHR) |
| service.py | 89 | Forensics service orchestrator |

#### src/input_handlers/ - Input Adapters (3 files, 176 lines)
| File | Lines | Purpose |
|------|------:|---------|
| base_handler.py | 10 | Abstract input handler |
| folder_handler.py | 50 | Local folder processor |
| json_handler.py | 116 | JSON manifest processor |

#### src/reconstruction/ - Face Reconstruction (11 files, 4,443 lines)
| File | Lines | Purpose |
|------|------:|---------|
| __init__.py | 17 | Exports: OpenVINOForensicReconstructor, Deep3DFaceReconstructor |
| deep3d_recon.py | 889 | Deep3D engine: ResNet50 + BFM09 + Gouraud CPU rasterizer (v6.0 rewritten) |
| generative.py | 390 | 9-stage pipeline orchestrator + multi-view output |
| sdxl_reconstructor.py | 7 | Backward-compat alias (extends OpenVINOForensicReconstructor) |
| face_reconstructor.py | 798 | ForensicFaceReconstructor (injury/scar aware inpainting) |
| super_resolution.py | 201 | Lanczos + unsharp + CLAHE enhancement |
| occlusion_remover.py | 477 | Glasses/mask/bandage removal (symmetry-based) |
| lighting_normalizer.py | 199 | Shadow/highlight correction + white balance |
| geometry_3d.py | 405 | Shape-from-shading depth + surface normals |
| aging_simulator.py | 567 | Age progression/regression synthesis |
| liveness_detector.py | 493 | Reconstruction-level liveness checks |

#### src/reporting/ - Report Generation (3 files, 356 lines)
| File | Lines | Purpose |
|------|------:|---------|
| __init__.py | 3 | Package init |
| llm_analyst.py | 348 | Ollama LLM forensic analyst (qwen3:1.7b, 8-step protocol) |
| llama_reporter.py | 5 | Extended report generator entry point |

### Root Files

| File | Purpose |
|------|---------|
| main.py | Application entry point |
| run_test.py | Integration test runner (~70 lines) |
| output.json | Latest pipeline output |
| requirements.txt | Python dependencies (60 lines) |
| install_deps.py | Model download helper |
| README.md | Project documentation (554 lines) |
| ProjectReport.md | This comprehensive report |
| .env.example | Environment variable template |
| .env | Active environment configuration |
| .gitignore | Git ignore rules |

### Scripts

| File | Purpose |
|------|---------|
| setup_deep3d.py | Download + verify all Deep3D model files (epoch_20.pth, BFM, Exp_Pca) |
| test_deep3d.py | Component-level test: BFM load -> ResNet50 forward -> CPU render |
| test_deep3d_real.py | Real image test: full reconstruction pipeline on test photo |
| test_render_quality.py | Render quality assessment: save all outputs, report face coverage |
| quick_smoke.py | Quick smoke test: load -> reconstruct -> print shapes/timing |

### Vendor

```
Deep3DFaceRecon_pytorch/        # Official Sicxu PyTorch reimplementation (reference, at project root)
```

---

## 18. Test Infrastructure

### Integration Test (run_test.py)

The integration test:
1. Loads primary and comparison images from test_data/applicant/
2. Creates an Applicant object with primary_docs and comparison_docs
3. Instantiates VerificationEngine
4. Calls process_applicant(applicant)
5. Writes results to output.json
6. Logs dashboard path and match result

### Test Data

```
test_data/applicant/
+-- primary/
|   +-- image.png          # Primary face image (276x361)
+-- compare_with/
    +-- image.png          # Comparison face image
```

### Running Tests

```bash
# Full integration test
python run_test.py

# Component-level Deep3D test (no full pipeline)
python test_deep3d.py

# Real image Deep3D test (single face reconstruction)
python test_deep3d_real.py

# Render quality check (verifies face coverage + saves all outputs)
python test_render_quality.py

# Quick smoke test (minimal load + reconstruct)
python quick_smoke.py
```

---

## 19. API and Deployment

### FastAPI Endpoint (src/api/main.py - 75 lines)

| Endpoint | Method | Purpose |
|----------|--------|---------|
| /health | GET | Returns {status: "healthy", ray_enabled: false} |
| /process | POST | Accepts ApplicantPayload, runs full pipeline, returns JSON result |

- Title: CA_Monk Forensic API, Version: 3.0
- Lazy-initialized singleton VerificationEngine
- Shutdown hook cleans up engine resources

### Resource Requirements

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| **CPU** | 4 cores | 8 cores (i7/i9) |
| **RAM** | 8 GB | 16 GB |
| **Disk** | 2 GB (without legacy models) | 5 GB |
| **GPU** | Not required | Optional (ONNX Runtime GPU) |
| **Network** | Ollama only | First run: model downloads (~467 MB Deep3D) |

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| OLLAMA_MODEL | qwen3:1.7b | LLM model name |
| OLLAMA_BASE_URL | http://127.0.0.1:11434 | Ollama server URL |
| OLLAMA_TIMEOUT_SECONDS | 120 | LLM timeout |
| OLLAMA_NUM_CTX | 4096 | LLM context window |
| OLLAMA_TEMPERATURE | 0.1 | LLM temperature |

---

## 20. Version History

| Version | Date | Key Changes |
|---------|------|-------------|
| **v4.0** | Feb 2025 | SD 1.5 Realistic Vision reconstruction, 7-module biometric suite |
| **v5.0** | Feb 2025 | SD 1.5 REMOVED. Replaced with Deep3D (MediaPipe 478 landmarks + OpenCV). ~20x faster. |
| **v5.1** | Feb 2026 | True Deep3DFaceReconstruction: ResNet50 + BFM09 + KDTree+IDW CPU renderer. 35K-vertex .obj mesh export. ~500ms cached. |
| **v6.0** | Feb 28, 2026 | **CPUMeshRenderer rewrite**: 2-pass Gouraud triangle rasterization (cv2.fillConvexPoly + barycentric + 2x SSAA). CW winding fix (0.6% -> 40.2% coverage). Multi-view renders (geometry, depth, normals, side-view). NoisePrint dual-branch (Scharr edge). Dashboard reconstruction thumbnails. Scar analysis relative_y fix. |

### v6.0 Bug Fixes

| Bug | Impact | Fix |
|-----|--------|-----|
| **CW winding order** | 93.7% of triangles culled (face coverage 0.6%) | Changed cross > 0 to cross < 0 in CPUMeshRenderer |
| **relative_y KeyError** | scar_analysis.py crashed on injury markers (masked by except) | Added relative_x/relative_y to injury dicts + defensive .get() |

### v6.0 Enhancements

| Enhancement | Details |
|-------------|---------|
| **2-pass Gouraud rasterizer** | cv2.fillConvexPoly triangle-ID map -> vectorized barycentric interpolation -> 2x SSAA anti-aliasing |
| **Multi-view renders** | Geometry (gray Lambertian) + depth (INFERNO) + normal map (RGB) + side-view (30 deg yaw) |
| **NoisePrint dual-branch** | Added Scharr edge residual alongside PRNU bilateral filter. Fused scoring with splice ratio annotation. |
| **Dashboard reconstruction panel** | Geometry/depth/sideview thumbnails (140x140) alongside main render (200x200) |
| **2-column data readout** | Match telemetry + forensic signals (left) / biometric data + compliance (right) |

---

## 21. Known Limitations and Future Work

### Current Limitations

| Issue | Impact | Status |
|-------|--------|--------|
| iris_r overflow warning | RuntimeWarning in iris_analyzer.py (line 571/578) | Non-crash - cosmetic |
| CodeFormer ONNX often skipped | Neural restoration not applied | Pipeline works without it |
| Iris anti-spoof false negative | is_real_eye=false on real photos (no specular reflection) | Known limitation of specular check |
| models/sd15_rv6/ on disk | ~2.5 GB wasted disk space | Safe to delete manually |
| Single-image rPPG liveness | Less accurate than video-based | Use video input when available |
| Deep3D first-call latency | ~8.4s on first call (InsightFace init) | ~2.4s on subsequent calls |

### Future Improvements

1. **Delete legacy SD 1.5 model files** - reclaim ~2.5 GB disk space
2. **INT8 quantize ONNX models** - faster inference for AdaFace, CodeFormer
3. **Add video-based rPPG** - multi-frame heartbeat for stronger liveness signal
4. **Reduce SSAA for side-views** - 1x SSAA instead of 2x to speed up secondary renders
5. **Batch processing** - process multiple applicants in parallel
6. **Docker container** - Dockerfile with all dependencies pre-installed
7. **CI/CD pipeline** - automated testing on push
8. **Confidence calibration** - calibrate match thresholds against larger dataset
9. **Cast iris_r to int64** - fix overflow warning in iris_analyzer.py
10. **Separate CodeFormer download script** - automated download for optional ONNX model

---

## Appendix A: Evidence Package History

| # | Package | Date | Pipeline Version |
|---|---------|------|-----------------|
| 1 | evidence_package_test_subject_20260219_162422 | Feb 19 16:24 | v4 (SD 1.5) |
| 2 | evidence_package_test_subject_20260223_102257 | Feb 23 10:22 | v5.0 (Deep3D MediaPipe) |
| 3 | evidence_package_test_subject_20260223_110148 | Feb 23 11:01 | v5.0 (Deep3D MediaPipe) |
| 4 | evidence_package_test_subject_20260223_110642 | Feb 23 11:06 | v5.0 (Deep3D MediaPipe fixed) |
| 5 | evidence_package_test_subject_20260223_121649 | Feb 23 12:16 | v5.1 (Deep3D BFM + KDTree) |
| 6 | evidence_package_test_subject_20260228_084215 | Feb 28 08:42 | v6.0 (Gouraud rasterizer) |
| 7 | evidence_package_test_subject_20260228_090132 | Feb 28 09:01 | **v6.0 (final - 20 artifacts)** |

## Appendix B: Explainability - Top Contributing Regions (Last Run)

| Region | Contribution Score |
|--------|--------------------|
| Mouth | 0.645 |
| Eyes | 0.638 |
| Nose | 0.593 |
| Method | Siamese Occlusion CAM |

## Appendix C: Primary Image Deep Study (Last Run)

| Metric | Value |
|--------|-------|
| Blur Score | 392.0 |
| Noise Level | 4.4 |
| Exposure | normal |
| Skin Uniformity | 0.39 |
| Redness Ratio | 0.471 |
| Texture Roughness | 851.6 |
| Facial Symmetry | 0.14 (asymmetric) |
| Mean Brightness | 83 |
| Shadow Ratio | 0.35 |
| Lighting Direction | right |
| Wrinkle Density | 0.24 |
| Estimated Age Range | 18-30 |
| Marks Detected | 14 (dark_spots=4, red_spots=1, scar_like=9) |
| Resolution | 276x361 |

## Appendix D: BFM Technical Constants

| Parameter | Value | Description |
|-----------|-------|-------------|
| Vertices | 35,709 | Face region subset of BFM09 (53,490 total) |
| Triangles | 70,789 | CW winding order |
| Keypoints | 68 | Projected from 3D landmarks |
| ID Coefficients | 80 | Shape eigenspace |
| Expression Coefficients | 64 | Expression PCA basis |
| Texture Coefficients | 80 | Albedo eigenspace |
| Rotation | 3 | Euler angles (x, y, z) |
| SH Illumination | 27 | 3 bands x 9 x RGB |
| Translation | 3 | tx, ty, tz |
| Focal Length | 1015 | Perspective projection |
| Center | 112 | Image center (224/2) |
| Camera Distance | 10.0 | Fixed camera-to-face distance |
| Z Near / Z Far | 5.0 / 15.0 | Depth clipping planes |
| Field of View | ~12.59 deg | Derived from focal/sensor |
| SSAA Factor | 2x | SuperSample Anti-Aliasing |

---

*End of Report - CA_MONK v6.0.0 Deep3D Gouraud Renderer + Dual-Branch Forensics*
