from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

from src.core.serialization import to_builtin


class InteractiveCasefileBuilder:
    """Generate an interactive HTML casefile and its backing JSON payload."""

    HTML_FILENAME = "INTERACTIVE_CASEFILE.html"
    DATA_FILENAME = "CASEFILE_DATA.json"

    def build(self, evidence_dir: str, result: Dict[str, Any]) -> Dict[str, Any]:
        evidence_path = Path(evidence_dir)
        evidence_path.mkdir(parents=True, exist_ok=True)

        payload = self._build_payload(evidence_path, result)
        data_path = evidence_path / self.DATA_FILENAME
        data_path.write_text(
            json.dumps(to_builtin(payload), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        html_path = evidence_path / self.HTML_FILENAME
        html_path.write_text(self._render_html(payload), encoding="utf-8")

        return {
            "html_path": str(html_path),
            "data_path": str(data_path),
            "comparison_count": len(payload.get("comparisons", [])),
        }

    def _build_payload(self, evidence_path: Path, result: Dict[str, Any]) -> Dict[str, Any]:
        comparisons = [self._comparison_payload(evidence_path, item) for item in result.get("comparisons", []) if isinstance(item, dict)]
        artifacts = self._artifact_catalog(evidence_path)
        run_metadata = to_builtin(result.get("run_metadata", {}) or {})
        return {
            "applicant_role": result.get("role", "unknown"),
            "is_match": bool(result.get("is_match", False)),
            "confidence": float(result.get("confidence", 0.0) or 0.0),
            "warnings": list(result.get("warnings", []) or []),
            "run_metadata": run_metadata,
            "runtime_capabilities": to_builtin(result.get("runtime_capabilities", {}) or run_metadata.get("runtime_capabilities", {}) or {}),
            "evidence_integrity": to_builtin(result.get("evidence_integrity", {}) or {}),
            "artifacts": artifacts,
            "comparisons": comparisons,
        }

    def _comparison_payload(self, evidence_path: Path, comparison: Dict[str, Any]) -> Dict[str, Any]:
        match = comparison.get("match", {}) or {}
        report = comparison.get("report", {}) or {}
        forensics = comparison.get("forensics", {}) or {}
        document = comparison.get("document_intelligence", {}) or {}
        advanced = comparison.get("advanced_biometrics", {}) or {}
        cross_validation = comparison.get("forensic_3d_cross_validation", {}) or {}
        consistency = cross_validation.get("consistency_analysis", {}) or {}
        expression_suite = comparison.get("expression_suite", {}) or {}
        face_evidence = comparison.get("face_evidence", {}) or {}
        ledger = self._evidence_ledger(match, forensics, document, advanced, consistency, face_evidence)

        mesh_text = ""
        mesh_file = evidence_path / "reconstruction_hq_mesh.obj"
        if mesh_file.exists():
            try:
                mesh_text = mesh_file.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                mesh_text = ""

        localized_expression_suite = {
            key: self._localize_artifact_path(evidence_path, value) if key.endswith("_path") else to_builtin(value)
            for key, value in expression_suite.items()
        }

        return {
            "filename": comparison.get("filename", "unknown"),
            "match": to_builtin(match),
            "report": to_builtin(report),
            "forensics": to_builtin(forensics),
            "document_intelligence": to_builtin(document),
            "advanced_biometrics": to_builtin(advanced),
            "cross_validation": to_builtin(cross_validation),
            "stage_telemetry": to_builtin(comparison.get("stage_telemetry", []) or []),
            "warnings": list(comparison.get("warnings", []) or []),
            "dashboard": self._localize_artifact_path(evidence_path, comparison.get("dashboard")),
            "reconstruction_path": self._localize_artifact_path(
                evidence_path,
                (comparison.get("reconstruction", {}) or {}).get("generated_image_path"),
            ),
            "expression_suite": to_builtin(localized_expression_suite),
            "report_verdict": report.get("verdict", "UNKNOWN"),
            "threat_level": consistency.get("threat_level", "UNKNOWN"),
            "contradictions": list(consistency.get("contradictions", []) or []),
            "agreements": list(consistency.get("agreements", []) or []),
            "face_evidence": to_builtin(comparison.get("face_evidence", {}) or {}),
            "ledger": ledger,
            "mesh_text": mesh_text,
        }

    def _artifact_catalog(self, evidence_path: Path) -> List[Dict[str, Any]]:
        image_exts = {".png", ".jpg", ".jpeg", ".webp", ".gif"}
        artifacts: List[Dict[str, Any]] = []
        for path in sorted(p for p in evidence_path.iterdir() if p.is_file()):
            suffix = path.suffix.lower()
            artifacts.append(
                {
                    "name": path.name,
                    "path": path.name,
                    "kind": "image" if suffix in image_exts else "mesh" if suffix == ".obj" else "document",
                    "size_bytes": path.stat().st_size,
                }
            )
        return artifacts

    @staticmethod
    def _localize_artifact_path(evidence_path: Path, path: Any) -> Any:
        if not isinstance(path, str) or not path:
            return path
        candidate = evidence_path / Path(path).name
        return candidate.name if candidate.exists() else path

    def _evidence_ledger(
        self,
        match: Dict[str, Any],
        forensics: Dict[str, Any],
        document: Dict[str, Any],
        advanced: Dict[str, Any],
        consistency: Dict[str, Any],
        face_evidence: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        entries: List[Dict[str, Any]] = []
        model_scores = match.get("model_scores", {}) or {}
        for model_name, model_data in model_scores.items():
            passed = bool(model_data.get("passed", False))
            entries.append(
                {
                    "source": f"biometric:{model_name}",
                    "direction": "support" if passed else "risk",
                    "impact": round(float(model_data.get("weight", 0.0) or 0.0), 4),
                    "observation": (
                        f"cosine={float(model_data.get('cosine_similarity', 0.0)):.4f}, "
                        f"threshold={float(model_data.get('threshold', 0.0)):.4f}, "
                        f"passed={passed}"
                    ),
                }
            )

        deepfake = bool((forensics.get("frequency", {}) or {}).get("deepfake_suspected", False))
        splice = bool((document.get("noiseprint", {}) or {}).get("suspected_splice", False))
        rppg = forensics.get("rppg", {}) or {}
        signal_state = str(rppg.get("signal_state", "unknown"))
        entries.append(
            {
                "source": "forensics:deepfake",
                "direction": "risk" if deepfake else "support",
                "impact": 0.2,
                "observation": (
                    f"deepfake_probability={float((forensics.get('frequency', {}) or {}).get('deepfake_probability', 0.0)):.4f}, "
                    f"suspected={deepfake}"
                ),
            }
        )
        entries.append(
            {
                "source": "forensics:rppg",
                "direction": "neutral" if signal_state == "not_available" else "risk" if signal_state == "spoof" else "support",
                "impact": 0.15,
                "observation": f"signal_state={signal_state}, is_live={rppg.get('is_live')}, bpm={rppg.get('bpm')}",
            }
        )
        entries.append(
            {
                "source": "document:noiseprint",
                "direction": "risk" if splice else "support",
                "impact": 0.2,
                "observation": f"suspected_splice={splice}",
            }
        )

        pair = advanced.get("pair_analysis", {}) or {}
        verdict = str(pair.get("final_verdict", pair.get("verdict", "UNKNOWN")))
        entries.append(
            {
                "source": "advanced_biometrics",
                "direction": "risk" if verdict in {"REJECT", "LIKELY_DOPPELGANGER"} else "support",
                "impact": 0.2,
                "observation": f"pair_verdict={verdict}, confidence={pair.get('confidence', 0)}",
            }
        )
        alteration = pair.get("identity_alteration_context", {}) or {}
        if alteration.get("detected"):
            entries.append(
                {
                    "source": "advanced_biometrics:alteration_context",
                    "direction": "neutral",
                    "impact": 0.12,
                    "observation": (
                        f"category={alteration.get('category')}, "
                        f"factors={alteration.get('factors')}, "
                        f"face_match_score={alteration.get('face_match_score')}"
                    ),
                }
            )
        for role in ("primary", "comparison"):
            side = advanced.get(role, {}) or {}
            makeup = side.get("makeup_disguise", {}) or {}
            iris = side.get("iris", {}) or {}
            markers = side.get("facial_markers", {}) or {}
            tamper = side.get("tampering", {}) or {}
            morph = side.get("morphing", {}) or {}
            seam = tamper.get("micro_seam_analysis", {}) or {}
            entries.append(
                {
                    "source": f"advanced_biometrics:{role}",
                    "direction": "risk" if (
                        makeup.get("disguise_detected")
                        or morph.get("is_morphed")
                        or tamper.get("tampering_detected")
                        or seam.get("seam_detected")
                        or (iris.get("sclera_analysis", {}) or {}).get("deepfake_suspected")
                    ) else "support",
                    "impact": 0.12,
                    "observation": (
                        f"makeup={makeup.get('makeup_level')}/{makeup.get('disguise_probability')}%, "
                        f"markers={markers.get('markers_detected')}, "
                        f"cataract={(iris.get('health_indicators', {}) or {}).get('cataract_probability')}, "
                        f"seam={seam.get('seam_probability')}, "
                        f"morph={morph.get('morphing_probability')}"
                    ),
                }
            )
        for role in ("primary", "comparison"):
            payload = (face_evidence.get(role, {}) or {}).get("liveness", {}) or {}
            state = str(payload.get("signal_state", "unknown"))
            score = float(payload.get("score", 0.0) or 0.0)
            entries.append(
                {
                    "source": f"face_pad:{role}",
                    "direction": "risk" if state == "spoof" else "neutral" if state == "indeterminate" else "support",
                    "impact": 0.16,
                    "observation": (
                        f"state={state}, score={score:.4f}, "
                        f"attack_type={payload.get('attack_type')}, backend={payload.get('backend')}, "
                        f"indicators={payload.get('attack_indicators')}"
                    ),
                }
            )
        calibration = match.get("calibration_features", {}) or {}
        if calibration:
            entries.append(
                {
                    "source": "match:calibration",
                    "direction": "support" if float(calibration.get("calibrated_confidence", 0.0) or 0.0) >= 0.57 else "risk",
                    "impact": 0.18,
                    "observation": (
                        f"quality={calibration.get('quality_factor')}, "
                        f"liveness={calibration.get('liveness_factor')}, "
                        f"age_gap={calibration.get('age_gap_factor')}, "
                        f"calibrated_confidence={calibration.get('calibrated_confidence')}"
                    ),
                }
            )
        for risk in match.get("risk_flags", []) or []:
            entries.append(
                {
                    "source": "match:risk_flag",
                    "direction": "risk",
                    "impact": 0.14,
                    "observation": str(risk),
                }
            )

        for detail in consistency.get("contradictions", []) or []:
            entries.append(
                {
                    "source": "consistency",
                    "direction": "risk",
                    "impact": 0.25,
                    "observation": str(detail),
                }
            )
        for detail in consistency.get("agreements", []) or []:
            entries.append(
                {
                    "source": "consistency",
                    "direction": "support",
                    "impact": 0.1,
                    "observation": str(detail),
                }
            )
        return entries

    def _render_html(self, payload: Dict[str, Any]) -> str:
        comparisons_html = "\n".join(
            self._render_comparison(index, comparison)
            for index, comparison in enumerate(payload.get("comparisons", []))
        )

        warnings_html = "".join(
            f"<li>{html.escape(str(item))}</li>"
            for item in payload.get("warnings", [])
        ) or "<li>No top-level warnings recorded.</li>"

        artifact_cards = "\n".join(
            self._render_artifact_card(artifact)
            for artifact in payload.get("artifacts", [])
        )

        run_metadata_json = html.escape(json.dumps(payload.get("run_metadata", {}), indent=2, ensure_ascii=False))
        capabilities_json = html.escape(json.dumps(payload.get("runtime_capabilities", {}), indent=2, ensure_ascii=False))
        evidence_json = html.escape(json.dumps(payload.get("evidence_integrity", {}), indent=2, ensure_ascii=False))
        payload_json = html.escape(json.dumps(payload, indent=2, ensure_ascii=False))
        run_timeline = self._render_timeline((payload.get("run_metadata", {}) or {}).get("stage_telemetry", []))

        return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>CA_Monk Interactive Casefile</title>
  <style>
    :root {{
      --bg: #09111b;
      --panel: #122131;
      --panel-soft: #182b40;
      --border: #2a425c;
      --text: #f3f7fb;
      --muted: #9bb2c8;
      --good: #6ee7b7;
      --warn: #fbbf24;
      --bad: #fb7185;
      --accent: #7dd3fc;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Segoe UI", Tahoma, sans-serif;
      background:
        radial-gradient(circle at top right, rgba(125, 211, 252, 0.12), transparent 28%),
        linear-gradient(160deg, #07111a 0%, #0b1724 46%, #09131e 100%);
      color: var(--text);
    }}
    a {{ color: var(--accent); text-decoration: none; }}
    .shell {{
      max-width: 1440px;
      margin: 0 auto;
      padding: 24px;
      display: grid;
      gap: 18px;
    }}
    .hero, .panel {{
      background: rgba(18, 33, 49, 0.92);
      border: 1px solid var(--border);
      border-radius: 18px;
      box-shadow: 0 24px 60px rgba(0, 0, 0, 0.28);
    }}
    .hero {{
      padding: 24px;
      display: grid;
      grid-template-columns: 2fr 1fr;
      gap: 16px;
    }}
    .hero h1 {{
      margin: 0 0 6px;
      font-size: 32px;
      letter-spacing: 0.03em;
    }}
    .muted {{ color: var(--muted); }}
    .metric-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 12px;
      margin-top: 16px;
    }}
    .metric {{
      background: rgba(255, 255, 255, 0.03);
      border: 1px solid rgba(255, 255, 255, 0.07);
      border-radius: 14px;
      padding: 14px;
    }}
    .metric .label {{ color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: 0.08em; }}
    .metric .value {{ margin-top: 6px; font-size: 28px; font-weight: 700; }}
    .badge {{
      display: inline-flex;
      align-items: center;
      gap: 8px;
      border-radius: 999px;
      padding: 8px 12px;
      background: rgba(255, 255, 255, 0.05);
      border: 1px solid rgba(255, 255, 255, 0.09);
      margin-right: 8px;
      margin-bottom: 8px;
    }}
    .badge.good {{ color: var(--good); border-color: rgba(110, 231, 183, 0.25); }}
    .badge.bad {{ color: var(--bad); border-color: rgba(251, 113, 133, 0.25); }}
    .badge.warn {{ color: var(--warn); border-color: rgba(251, 191, 36, 0.25); }}
    .layout {{
      display: grid;
      grid-template-columns: 1.1fr 0.9fr;
      gap: 18px;
    }}
    .panel {{
      padding: 20px;
    }}
    .panel h2 {{
      margin: 0 0 14px;
      font-size: 19px;
      letter-spacing: 0.03em;
    }}
    .warning-list, .plain-list {{
      margin: 0;
      padding-left: 18px;
      color: var(--muted);
    }}
    .artifact-grid, .stage-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 12px;
    }}
    .artifact {{
      background: rgba(255, 255, 255, 0.03);
      border: 1px solid rgba(255, 255, 255, 0.08);
      border-radius: 14px;
      padding: 12px;
      min-height: 160px;
    }}
    .artifact img {{
      width: 100%;
      border-radius: 10px;
      border: 1px solid rgba(255, 255, 255, 0.08);
      margin-top: 10px;
      object-fit: cover;
      max-height: 180px;
    }}
    .comparison {{
      display: grid;
      gap: 16px;
      margin-top: 16px;
      padding-top: 16px;
      border-top: 1px solid rgba(255, 255, 255, 0.09);
    }}
    .subgrid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 12px;
    }}
    .fact {{
      background: var(--panel-soft);
      border: 1px solid rgba(255, 255, 255, 0.08);
      border-radius: 14px;
      padding: 12px;
    }}
    .fact .k {{ color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: 0.08em; }}
    .fact .v {{ font-size: 24px; font-weight: 700; margin-top: 8px; }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 14px;
    }}
    th, td {{
      padding: 10px 8px;
      border-bottom: 1px solid rgba(255, 255, 255, 0.08);
      text-align: left;
      vertical-align: top;
    }}
    th {{ color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: 0.08em; }}
    .support {{ color: var(--good); }}
    .risk {{ color: var(--bad); }}
    .neutral {{ color: var(--warn); }}
    .timeline {{
      display: grid;
      gap: 10px;
    }}
    .timeline-row {{
      display: grid;
      grid-template-columns: 160px 1fr 78px;
      gap: 10px;
      align-items: center;
    }}
    .bar {{
      height: 12px;
      border-radius: 999px;
      background: rgba(255, 255, 255, 0.07);
      overflow: hidden;
    }}
    .bar > span {{
      display: block;
      height: 100%;
      border-radius: inherit;
      background: linear-gradient(90deg, #38bdf8, #22d3ee);
    }}
    .viewer-wrap {{
      display: grid;
      grid-template-columns: 1.1fr 0.9fr;
      gap: 12px;
      align-items: start;
    }}
    canvas.viewer {{
      width: 100%;
      min-height: 360px;
      background: linear-gradient(180deg, #0d1824 0%, #101e2f 100%);
      border-radius: 16px;
      border: 1px solid rgba(255, 255, 255, 0.08);
    }}
    pre {{
      margin: 0;
      white-space: pre-wrap;
      word-break: break-word;
      font-size: 12px;
      color: #d7e4f3;
      background: #0b1622;
      border-radius: 12px;
      padding: 12px;
      border: 1px solid rgba(255, 255, 255, 0.06);
      max-height: 420px;
      overflow: auto;
    }}
    details {{
      background: rgba(255, 255, 255, 0.03);
      border: 1px solid rgba(255, 255, 255, 0.08);
      border-radius: 12px;
      padding: 10px 12px;
    }}
    summary {{
      cursor: pointer;
      color: var(--accent);
      font-weight: 600;
    }}
    @media (max-width: 980px) {{
      .hero, .layout, .viewer-wrap {{
        grid-template-columns: 1fr;
      }}
      .timeline-row {{
        grid-template-columns: 1fr;
      }}
    }}
  </style>
</head>
<body>
  <div class="shell">
    <section class="hero">
      <div>
        <div class="badge {'good' if payload.get('is_match') else 'bad'}">{'MATCH ACCEPTED' if payload.get('is_match') else 'MATCH BLOCKED'}</div>
        <div class="badge warn">CPU-FIRST FORENSIC RUN</div>
        <h1>CA_Monk Interactive Casefile</h1>
        <p class="muted">Evidence-centric review surface for the generated package. This casefile is local, self-contained, and built from the actual pipeline payload.</p>
        <div class="metric-grid">
          <div class="metric">
            <div class="label">Applicant Role</div>
            <div class="value">{html.escape(str(payload.get('applicant_role', 'unknown')))}</div>
          </div>
          <div class="metric">
            <div class="label">Confidence</div>
            <div class="value">{float(payload.get('confidence', 0.0) or 0.0):.2f}%</div>
          </div>
          <div class="metric">
            <div class="label">Comparisons</div>
            <div class="value">{len(payload.get('comparisons', []))}</div>
          </div>
        </div>
      </div>
      <div class="panel">
        <h2>Run Metadata</h2>
        <pre>{run_metadata_json}</pre>
      </div>
    </section>

    <div class="layout">
      <section class="panel">
        <h2>Warnings</h2>
        <ul class="warning-list">{warnings_html}</ul>
      </section>
      <section class="panel">
        <h2>Artifact Catalog</h2>
        <div class="artifact-grid">{artifact_cards}</div>
      </section>
    </div>

    <div class="layout">
      <section class="panel">
        <h2>Run Timeline</h2>
        <div class="timeline">{run_timeline}</div>
      </section>
      <section class="panel">
        <h2>Runtime Capabilities</h2>
        <pre>{capabilities_json}</pre>
      </section>
    </div>

    <div class="layout">
      <section class="panel">
        <h2>Evidence Integrity</h2>
        <pre>{evidence_json}</pre>
      </section>
      <section class="panel">
        <h2>Review Notes</h2>
        <ul class="plain-list">
          <li>The casefile is rendered from the exact pipeline payload stored for this run.</li>
          <li>PAD backend, providers, and calibrated confidence are exposed so reviewers can see which evidence was real versus fallback.</li>
          <li>Manifest verification is available when the final package summary includes it.</li>
        </ul>
      </section>
    </div>

    <section class="panel">
      <h2>Comparison Review</h2>
      {comparisons_html}
    </section>

    <section class="panel">
      <h2>Raw Payload</h2>
      <details>
        <summary>Expand full casefile JSON</summary>
        <pre>{payload_json}</pre>
      </details>
    </section>
  </div>
  <script>
    function parseObj(text) {{
      const vertices = [];
      const faces = [];
      for (const raw of text.split(/\\r?\\n/)) {{
        const line = raw.trim();
        if (!line) continue;
        if (line.startsWith('v ')) {{
          const [, x, y, z] = line.split(/\\s+/);
          vertices.push([parseFloat(x), parseFloat(y), parseFloat(z)]);
        }} else if (line.startsWith('f ')) {{
          const parts = line.split(/\\s+/).slice(1).map(item => parseInt(item.split('/')[0], 10) - 1);
          if (parts.length >= 3) faces.push(parts);
        }}
      }}
      const edges = new Set();
      for (const face of faces) {{
        for (let i = 0; i < face.length; i++) {{
          const a = face[i];
          const b = face[(i + 1) % face.length];
          const key = a < b ? `${{a}}:${{b}}` : `${{b}}:${{a}}`;
          edges.add(key);
        }}
      }}
      return {{
        vertices,
        edges: Array.from(edges).map(item => item.split(':').map(v => parseInt(v, 10)))
      }};
    }}

    function mountViewer(canvasId, dataId) {{
      const canvas = document.getElementById(canvasId);
      const dataNode = document.getElementById(dataId);
      if (!canvas || !dataNode) return;
      const objText = dataNode.textContent || '';
      if (!objText.trim()) {{
        const ctx = canvas.getContext('2d');
        canvas.width = canvas.clientWidth;
        canvas.height = 360;
        ctx.fillStyle = '#9bb2c8';
        ctx.font = '16px Segoe UI';
        ctx.fillText('No mesh data available for this comparison.', 20, 40);
        return;
      }}

      const model = parseObj(objText);
      const ctx = canvas.getContext('2d');
      let angleY = 0.35;
      let angleX = -0.15;
      let dragging = false;
      let lastX = 0;
      let lastY = 0;

      function resize() {{
        canvas.width = canvas.clientWidth;
        canvas.height = 360;
      }}
      resize();
      window.addEventListener('resize', resize);

      canvas.addEventListener('pointerdown', (event) => {{
        dragging = true;
        lastX = event.clientX;
        lastY = event.clientY;
      }});
      window.addEventListener('pointerup', () => dragging = false);
      window.addEventListener('pointermove', (event) => {{
        if (!dragging) return;
        angleY += (event.clientX - lastX) * 0.01;
        angleX += (event.clientY - lastY) * 0.01;
        lastX = event.clientX;
        lastY = event.clientY;
      }});

      function rotate(point) {{
        const [x0, y0, z0] = point;
        const cosY = Math.cos(angleY);
        const sinY = Math.sin(angleY);
        const cosX = Math.cos(angleX);
        const sinX = Math.sin(angleX);

        const x1 = x0 * cosY + z0 * sinY;
        const z1 = -x0 * sinY + z0 * cosY;
        const y1 = y0 * cosX - z1 * sinX;
        const z2 = y0 * sinX + z1 * cosX;
        return [x1, y1, z2];
      }}

      function render() {{
        if (!dragging) angleY += 0.003;
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.fillStyle = '#101a26';
        ctx.fillRect(0, 0, canvas.width, canvas.height);

        const rotated = model.vertices.map(rotate);
        const scale = Math.min(canvas.width, canvas.height) * 0.8;
        const projected = rotated.map(([x, y, z]) => {{
          const depth = 2.4 / (2.8 + z);
          return [
            canvas.width * 0.5 + x * scale * depth,
            canvas.height * 0.52 - y * scale * depth,
            z
          ];
        }});

        ctx.lineWidth = 1.1;
        ctx.strokeStyle = '#7dd3fc';
        for (const [a, b] of model.edges) {{
          const pa = projected[a];
          const pb = projected[b];
          if (!pa || !pb) continue;
          ctx.beginPath();
          ctx.moveTo(pa[0], pa[1]);
          ctx.lineTo(pb[0], pb[1]);
          ctx.stroke();
        }}

        requestAnimationFrame(render);
      }}

      render();
    }}

    {''.join(f"mountViewer('mesh-viewer-{i}', 'mesh-data-{i}');" for i, _ in enumerate(payload.get('comparisons', [])))}
  </script>
</body>
</html>"""

    def _render_artifact_card(self, artifact: Dict[str, Any]) -> str:
        name = html.escape(str(artifact.get("name", "artifact")))
        path = html.escape(str(artifact.get("path", "")))
        kind = html.escape(str(artifact.get("kind", "document")))
        size_kb = (float(artifact.get("size_bytes", 0) or 0.0) / 1024.0)
        preview = f'<img src="{path}" alt="{name}">' if kind == "image" else ""
        return (
            f'<article class="artifact">'
            f'<div class="muted">{kind.upper()}</div>'
            f'<div><a href="{path}">{name}</a></div>'
            f'<div class="muted">{size_kb:.1f} KB</div>'
            f"{preview}"
            f"</article>"
        )

    def _preview_card(self, title: str, path: str | None) -> str:
        if not path:
            return ""
        safe_title = html.escape(title)
        safe_path = html.escape(str(path))
        lower = str(path).lower()
        preview = f'<img src="{safe_path}" alt="{safe_title}">' if lower.endswith((".png", ".jpg", ".jpeg", ".webp", ".gif")) else ""
        return (
            f'<article class="artifact">'
            f'<div class="muted">EXPRESSION SUITE</div>'
            f'<div><a href="{safe_path}">{safe_title}</a></div>'
            f"{preview}"
            f"</article>"
        )

    def _render_comparison(self, index: int, comparison: Dict[str, Any]) -> str:
        match = comparison.get("match", {}) or {}
        report = comparison.get("report", {}) or {}
        consistency = (comparison.get("cross_validation", {}) or {}).get("consistency_analysis", {}) or {}
        face_evidence = comparison.get("face_evidence", {}) or {}
        primary_face = face_evidence.get("primary", {}) or {}
        comparison_face = face_evidence.get("comparison", {}) or {}
        primary_liveness = primary_face.get("liveness", {}) or {}
        comparison_liveness = comparison_face.get("liveness", {}) or {}
        calibration = match.get("calibration_features", {}) or {}
        risk_flags = "".join(
            f"<li>{html.escape(str(item))}</li>"
            for item in match.get("risk_flags", []) or []
        ) or "<li>No explicit risk flags recorded.</li>"
        stage_rows = self._render_timeline(comparison.get("stage_telemetry", []))
        ledger_rows = "".join(
            "<tr>"
            f"<td>{html.escape(str(item.get('source', 'unknown')))}</td>"
            f"<td class=\"{html.escape(str(item.get('direction', 'neutral')))}\">{html.escape(str(item.get('direction', 'neutral')).upper())}</td>"
            f"<td>{float(item.get('impact', 0.0) or 0.0):.3f}</td>"
            f"<td>{html.escape(str(item.get('observation', '')))}</td>"
            "</tr>"
            for item in comparison.get("ledger", [])
        )
        contradictions = "".join(
            f"<li>{html.escape(str(item))}</li>"
            for item in comparison.get("contradictions", [])
        ) or "<li>No contradictions recorded.</li>"
        agreements = "".join(
            f"<li>{html.escape(str(item))}</li>"
            for item in comparison.get("agreements", [])
        ) or "<li>No agreements recorded.</li>"
        mesh_text = html.escape(str(comparison.get("mesh_text", "")))
        expression_suite = comparison.get("expression_suite", {}) or {}
        expression_suite_cards = "".join(
            [
                self._preview_card("Expression Transfer Overlay", expression_suite.get("generated_image_path")),
                self._preview_card("Expression Rendered", expression_suite.get("rendered_image_path")),
                self._preview_card("Source Capture Card", expression_suite.get("source_capture_image_path")),
                self._preview_card("Expression Capture Card", expression_suite.get("expression_capture_image_path")),
                self._preview_card("Preset Gallery", expression_suite.get("preset_gallery_image_path")),
                self._preview_card("Expression Animation", expression_suite.get("animation_gif_path")),
                self._preview_card("Animation Keyframes", expression_suite.get("animation_keyframes_path")),
                self._preview_card("Profile Swing", expression_suite.get("teaser_gif_path")),
                self._preview_card("Profile Swing Keyframes", expression_suite.get("teaser_keyframes_path")),
                self._preview_card("Turntable 360", expression_suite.get("turntable_gif_path")),
                self._preview_card("Turntable Keyframes", expression_suite.get("turntable_keyframes_path")),
            ]
        ) or '<div class="muted">Expression suite artifacts were not generated for this comparison.</div>'
        advanced_feature_rows = self._advanced_feature_rows(comparison.get("advanced_biometrics", {}) or {})
        raw_json = html.escape(json.dumps(comparison, indent=2, ensure_ascii=False))

        return f"""
        <div class="comparison">
          <div class="subgrid">
            <div class="fact">
              <div class="k">Comparison File</div>
              <div class="v">{html.escape(str(comparison.get('filename', 'unknown')))}</div>
            </div>
            <div class="fact">
              <div class="k">Report Verdict</div>
              <div class="v">{html.escape(str(comparison.get('report_verdict', 'UNKNOWN')))}</div>
            </div>
            <div class="fact">
              <div class="k">Threat Level</div>
              <div class="v">{html.escape(str(comparison.get('threat_level', 'UNKNOWN')))}</div>
            </div>
            <div class="fact">
              <div class="k">Fusion Confidence</div>
              <div class="v">{float(match.get('confidence', 0.0) or 0.0) * 100.0:.2f}%</div>
            </div>
          </div>

          <div class="subgrid">
            <div class="panel">
              <h2>Stage Timeline</h2>
              <div class="timeline">{stage_rows}</div>
            </div>
            <div class="panel">
              <h2>Consistency Signals</h2>
              <div class="badge {'bad' if (comparison.get('contradictions') or []) else 'good'}">Contradictions: {len(comparison.get('contradictions', []))}</div>
              <div class="badge {'good' if (comparison.get('agreements') or []) else 'warn'}">Agreements: {len(comparison.get('agreements', []))}</div>
              <h3>Contradictions</h3>
              <ul class="plain-list">{contradictions}</ul>
              <h3>Agreements</h3>
              <ul class="plain-list">{agreements}</ul>
            </div>
          </div>

          <div class="subgrid">
            <div class="panel">
              <h2>Face Evidence</h2>
              <table>
                <thead>
                  <tr>
                    <th>Subject</th>
                    <th>Model</th>
                    <th>Norm</th>
                    <th>Quality</th>
                    <th>PAD Score</th>
                    <th>PAD State</th>
                    <th>PAD Attack</th>
                    <th>PAD Backend</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td>Primary</td>
                    <td>{html.escape(str(primary_face.get('model_name', 'unknown')))}</td>
                    <td>{float(primary_face.get('embedding_norm', 0.0) or 0.0):.2f}</td>
                    <td>{html.escape(str(primary_face.get('quality', 'unknown')))}</td>
                    <td>{float(primary_liveness.get('score', 0.0) or 0.0):.4f}</td>
                    <td>{html.escape(str(primary_liveness.get('signal_state', 'unknown')))}</td>
                    <td>{html.escape(str(primary_liveness.get('attack_type', 'none')))}</td>
                    <td>{html.escape(str(primary_liveness.get('backend', 'unknown')))}</td>
                  </tr>
                  <tr>
                    <td>Comparison</td>
                    <td>{html.escape(str(comparison_face.get('model_name', 'unknown')))}</td>
                    <td>{float(comparison_face.get('embedding_norm', 0.0) or 0.0):.2f}</td>
                    <td>{html.escape(str(comparison_face.get('quality', 'unknown')))}</td>
                    <td>{float(comparison_liveness.get('score', 0.0) or 0.0):.4f}</td>
                    <td>{html.escape(str(comparison_liveness.get('signal_state', 'unknown')))}</td>
                    <td>{html.escape(str(comparison_liveness.get('attack_type', 'none')))}</td>
                    <td>{html.escape(str(comparison_liveness.get('backend', 'unknown')))}</td>
                  </tr>
                </tbody>
              </table>
            </div>
            <div class="panel">
              <h2>Calibration Trace</h2>
              <table>
                <thead>
                  <tr>
                    <th>Feature</th>
                    <th>Value</th>
                  </tr>
                </thead>
                <tbody>
                  <tr><td>quality_factor</td><td>{html.escape(str(calibration.get('quality_factor', 'n/a')))}</td></tr>
                  <tr><td>liveness_factor</td><td>{html.escape(str(calibration.get('liveness_factor', 'n/a')))}</td></tr>
                  <tr><td>age_gap_factor</td><td>{html.escape(str(calibration.get('age_gap_factor', 'n/a')))}</td></tr>
                  <tr><td>detector_factor</td><td>{html.escape(str(calibration.get('detector_factor', 'n/a')))}</td></tr>
                  <tr><td>calibrated_confidence</td><td>{html.escape(str(calibration.get('calibrated_confidence', 'n/a')))}</td></tr>
                </tbody>
              </table>
              <h3>Risk Flags</h3>
              <ul class="plain-list">{risk_flags}</ul>
            </div>
          </div>

          <div class="panel">
            <h2>Advanced Feature Matrix</h2>
            <table>
              <thead>
                <tr>
                  <th>Feature</th>
                  <th>Primary</th>
                  <th>Comparison</th>
                  <th>Pair / Review Meaning</th>
                </tr>
              </thead>
              <tbody>{advanced_feature_rows}</tbody>
            </table>
          </div>

          <div class="panel">
            <h2>Evidence Weight Ledger</h2>
            <table>
              <thead>
                <tr>
                  <th>Source</th>
                  <th>Direction</th>
                  <th>Impact</th>
                  <th>Observation</th>
                </tr>
              </thead>
              <tbody>{ledger_rows}</tbody>
            </table>
          </div>

          <div class="panel">
            <h2>Expression Suite</h2>
            <div class="artifact-grid">{expression_suite_cards}</div>
          </div>

          <div class="panel">
            <h2>3D Mesh Viewer</h2>
            <div class="viewer-wrap">
              <canvas id="mesh-viewer-{index}" class="viewer"></canvas>
              <div>
                <p class="muted">Drag to rotate the wireframe. This viewer is built from the local OBJ mesh already stored in the evidence package.</p>
                <div class="badge warn">OBJ source: reconstruction_hq_mesh.obj</div>
                <details>
                  <summary>Expand mesh-backed comparison JSON</summary>
                  <pre>{raw_json}</pre>
                </details>
              </div>
            </div>
            <script id="mesh-data-{index}" type="text/plain">{mesh_text}</script>
          </div>
        </div>
        """

    def _advanced_feature_rows(self, advanced: Dict[str, Any]) -> str:
        primary = advanced.get("primary", {}) or {}
        comparison = advanced.get("comparison", {}) or {}
        pair = advanced.get("pair_analysis", {}) or {}

        def fmt(value: Any) -> str:
            if isinstance(value, float):
                return f"{value:.3f}"
            return html.escape(str(value))

        def side_summary(side: Dict[str, Any]) -> Dict[str, str]:
            makeup = side.get("makeup_disguise", {}) or {}
            markers = side.get("facial_markers", {}) or {}
            iris = side.get("iris", {}) or {}
            tamper = side.get("tampering", {}) or {}
            morph = side.get("morphing", {}) or {}
            age = side.get("age_invariant", {}) or {}
            unique = side.get("uniqueness", {}) or {}

            scar_count = ((markers.get("scar_analysis", {}) or {}).get("scar_count", 0))
            injury_count = len(markers.get("injury_signs", []) or [])
            surgery_count = len(markers.get("surgery_indicators", []) or [])
            health = iris.get("health_indicators", {}) or {}
            sclera = iris.get("sclera_analysis", {}) or {}
            anti = iris.get("anti_spoofing", {}) or {}
            seam = tamper.get("micro_seam_analysis", {}) or {}

            return {
                "Makeup / disguise": (
                    f"level={makeup.get('makeup_level', 'N/A')}, "
                    f"probability={float(makeup.get('disguise_probability', 0.0) or 0.0):.1f}%, "
                    f"detected={bool(makeup.get('disguise_detected', False))}"
                ),
                "Scars / injury / surgery": (
                    f"markers={markers.get('markers_detected', 0)}, scars={scar_count}, "
                    f"injuries={injury_count}, surgery_indicators={surgery_count}"
                ),
                "Iris / cataract / sclera": (
                    f"cataract={float(health.get('cataract_probability', 0.0) or 0.0):.2f}, "
                    f"clarity={float(health.get('iris_clarity', 0.0) or 0.0):.2f}, "
                    f"contact_lens={bool(anti.get('contact_lens_detected', False))}, "
                    f"sclera_ai={float(sclera.get('ai_noise_probability', 0.0) or 0.0):.2f}"
                ),
                "Tamper / seam / morph": (
                    f"tamper={bool(tamper.get('tampering_detected') or tamper.get('is_tampered'))}, "
                    f"seam_probability={float(seam.get('seam_probability', 0.0) or 0.0):.2f}, "
                    f"morph_probability={float(morph.get('morphing_probability', 0.0) or 0.0):.1f}%, "
                    f"is_morphed={bool(morph.get('is_morphed', False))}"
                ),
                "Age / uniqueness": (
                    f"age_confidence={fmt(age.get('extraction_confidence', 0.0))}, "
                    f"uniqueness={fmt(unique.get('uniqueness_score', 0.0))}"
                ),
            }

        primary_summary = side_summary(primary)
        comparison_summary = side_summary(comparison)
        marker_pair = pair.get("marker_comparison", {}) or {}
        doppel = pair.get("doppelganger_analysis", {}) or {}
        kinship = pair.get("kinship_analysis", {}) or {}
        alteration = pair.get("identity_alteration_context", {}) or {}
        morphing_check = pair.get("morphing_check", {}) or {}

        pair_meaning = {
            "Makeup / disguise": html.escape(str(alteration.get("summary", "review makeup/disguise effects"))),
            "Scars / injury / surgery": html.escape(str(marker_pair.get("verdict", "marker comparison unavailable"))),
            "Iris / cataract / sclera": "Eye-condition signals can degrade direct iris evidence; use stable face and marker signals together.",
            "Tamper / seam / morph": html.escape(str(morphing_check.get("recommendation", "review morph/tamper risks"))),
            "Age / uniqueness": (
                f"doppelganger={bool(doppel.get('is_doppelganger', False))}; "
                f"kinship={float(kinship.get('kinship_probability', 0.0) or 0.0):.1f}%"
            ),
            "Altered appearance context": html.escape(str(alteration.get("reviewer_note", ""))),
        }

        rows: List[str] = []
        for feature in ["Makeup / disguise", "Scars / injury / surgery", "Iris / cataract / sclera", "Tamper / seam / morph", "Age / uniqueness"]:
            rows.append(
                "<tr>"
                f"<td>{html.escape(feature)}</td>"
                f"<td>{html.escape(primary_summary.get(feature, 'n/a'))}</td>"
                f"<td>{html.escape(comparison_summary.get(feature, 'n/a'))}</td>"
                f"<td>{pair_meaning.get(feature, '')}</td>"
                "</tr>"
            )

        rows.append(
            "<tr>"
            "<td>Altered appearance context</td>"
            f"<td colspan=\"2\">detected={bool(alteration.get('detected', False))}; category={html.escape(str(alteration.get('category', 'none')))}; factors={html.escape(', '.join(alteration.get('factors', []) or []))}</td>"
            f"<td>{pair_meaning['Altered appearance context']}</td>"
            "</tr>"
        )
        return "".join(rows)

    def _render_timeline(self, stages: Iterable[Dict[str, Any]]) -> str:
        stages = list(stages)
        max_duration = max((float(stage.get("duration_ms", 0.0) or 0.0) for stage in stages), default=1.0)
        rows: List[str] = []
        for stage in stages:
            duration = float(stage.get("duration_ms", 0.0) or 0.0)
            width = 0.0 if max_duration <= 0.0 else (duration / max_duration) * 100.0
            rows.append(
                f'<div class="timeline-row">'
                f'<div>{html.escape(str(stage.get("stage", "unknown")))}</div>'
                f'<div class="bar"><span style="width:{width:.2f}%"></span></div>'
                f'<div>{duration:.2f} ms</div>'
                f'</div>'
            )
        if not rows:
            rows.append('<div class="muted">No stage telemetry recorded.</div>')
        return "\n".join(rows)
