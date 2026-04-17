"""
CA_MONK v4 — MILITARY-GRADE FORENSIC EVIDENCE DASHBOARD
========================================================
1920×1080 dark-themed intelligence dashboard with:
- CLASSIFIED header with operation ID
- Left panel: Input images with face bounding boxes
- Center: GradCAM overlay, NoisePrint overlay, spectral analysis
- Right: Reconstruction display + terminal-style data readout
- Bottom: Advanced biometric status grid + threat assessment
- Neon green/red accent system for PASS/FAIL states
"""

import cv2
import numpy as np
import os
from datetime import datetime
from typing import Any, Dict, Optional, Tuple


# --- Color Palette (BGR) ---
BLACK = (0, 0, 0)
BG_DARK = (15, 15, 15)
PANEL_BG = (25, 25, 25)
PANEL_BORDER = (45, 45, 45)
NEON_GREEN = (0, 255, 80)
NEON_RED = (0, 0, 255)
NEON_AMBER = (0, 180, 255)
NEON_CYAN = (255, 220, 0)
NEON_BLUE = (255, 140, 0)
WHITE = (255, 255, 255)
GRAY_TEXT = (180, 180, 180)
DIM_TEXT = (120, 120, 120)
HEADER_GOLD = (0, 200, 255)
CRITICAL_RED = (50, 50, 255)
SEPARATOR = (50, 50, 50)


class ForensicVisualizer:
    """Military-Grade 1920x1080 Evidence Dashboard Generator."""

    CANVAS_W = 1920
    CANVAS_H = 1080

    def __init__(self, output_dir: str = "evidence_cards") -> None:
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------
    def generate_evidence_card(
        self,
        img1_path: str,
        img2_path: str,
        face1_box: dict,
        face2_box: dict,
        match_data: dict,
        applicant_id: str,
        forensic_data: dict = None,
        compliance_data: dict = None,
        deepfake_data: dict = None,
        biometric_data: dict = None,
        advanced_biometrics: dict = None,
        reconstruction_path: str = None,
        gradcam_path: str = None,
        evidence_dir: str = None,
    ) -> Optional[str]:
        """Generate the full 1920x1080 classified intelligence dashboard."""

        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)
        if img1 is None or img2 is None:
            return None

        canvas = np.full((self.CANVAS_H, self.CANVAS_W, 3), BG_DARK, dtype=np.uint8)

        # === HEADER: CLASSIFIED BANNER ===
        self._draw_header(canvas, applicant_id, match_data)

        # === LEFT PANEL: Input Images (x=20, y=90) ===
        self._draw_input_panel(canvas, img1, img2, face1_box, face2_box, match_data, advanced_biometrics)

        # === CENTER PANEL: Forensic Overlays (x=520, y=90) ===
        self._draw_forensic_panel(
            canvas, forensic_data, deepfake_data,
            gradcam_path, evidence_dir,
        )

        # === RIGHT PANEL: Reconstruction + Terminal Data (x=1120, y=90) ===
        self._draw_recon_panel(
            canvas, reconstruction_path, match_data,
            compliance_data, forensic_data,
        )

        # === BOTTOM PANEL: Advanced Biometrics Grid (y=700) ===
        self._draw_biometrics_grid(canvas, advanced_biometrics, biometric_data)

        # === FOOTER: Verdict Stamp ===
        self._draw_footer(canvas, match_data, advanced_biometrics)

        # === Save ===
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_id = "".join(c for c in applicant_id if c.isalnum() or c in "_-")
        filename = f"DASHBOARD_{safe_id}_{ts}.jpg"
        full_path = os.path.join(self.output_dir, filename)
        cv2.imwrite(full_path, canvas, [cv2.IMWRITE_JPEG_QUALITY, 95])

        # Also save to evidence_dir if provided
        if evidence_dir and os.path.isdir(evidence_dir):
            ev_path = os.path.join(evidence_dir, f"FINAL_DASHBOARD_{safe_id}.jpg")
            cv2.imwrite(ev_path, canvas, [cv2.IMWRITE_JPEG_QUALITY, 95])

        return full_path

    # ------------------------------------------------------------------
    # HEADER
    # ------------------------------------------------------------------
    def _draw_header(self, canvas: np.ndarray, applicant_id: str, match_data: dict) -> None:
        # Top bar background
        cv2.rectangle(canvas, (0, 0), (self.CANVAS_W, 80), (10, 10, 10), -1)
        # Gold border line
        cv2.line(canvas, (0, 80), (self.CANVAS_W, 80), HEADER_GOLD, 2)

        # CLASSIFIED label
        cv2.putText(
            canvas, "[ CLASSIFIED ]", (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, NEON_RED, 2, cv2.LINE_AA,
        )
        cv2.putText(
            canvas, "CA_MONK v4 // FORENSIC INTELLIGENCE DASHBOARD", (200, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 2, cv2.LINE_AA,
        )

        # Operation info line
        ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
        verified = match_data.get("verified", False)
        status_color = NEON_GREEN if verified else NEON_RED
        status_text = "VERIFIED" if verified else "REJECTED"

        cv2.putText(
            canvas, f"SUBJECT: {applicant_id}", (20, 60),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, GRAY_TEXT, 1, cv2.LINE_AA,
        )
        cv2.putText(
            canvas, f"TIMESTAMP: {ts}", (400, 60),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, DIM_TEXT, 1, cv2.LINE_AA,
        )
        cv2.putText(
            canvas, f"STATUS: {status_text}", (800, 60),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 2, cv2.LINE_AA,
        )
        # Cosine score
        cosine = match_data.get("cosine_similarity", 0.0)
        cv2.putText(
            canvas, f"COSINE: {cosine:.4f}", (1050, 60),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, NEON_CYAN, 1, cv2.LINE_AA,
        )

        # Classification stamp (top right)
        cv2.putText(
            canvas, "TIER-1 BIOMETRIC INTELLIGENCE", (1500, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, NEON_AMBER, 1, cv2.LINE_AA,
        )
        cv2.putText(
            canvas, "AUTHORIZED PERSONNEL ONLY", (1540, 60),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, DIM_TEXT, 1, cv2.LINE_AA,
        )

    # ------------------------------------------------------------------
    # LEFT PANEL: Input images (480px wide)
    # ------------------------------------------------------------------
    def _draw_input_panel(
        self, canvas: np.ndarray,
        img1: np.ndarray, img2: np.ndarray,
        face1_box: dict, face2_box: dict,
        match_data: dict,
        advanced_biometrics: dict = None,
    ) -> None:
        panel_x, panel_y = 10, 90
        panel_w, panel_h = 490, 590

        # Panel background
        cv2.rectangle(
            canvas,
            (panel_x, panel_y),
            (panel_x + panel_w, panel_y + panel_h),
            PANEL_BG, -1,
        )
        self._draw_bracket(canvas, panel_x, panel_y, panel_w, panel_h)

        # Panel label
        cv2.putText(
            canvas, "INPUT IMAGES", (panel_x + 15, panel_y + 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, HEADER_GOLD, 1, cv2.LINE_AA,
        )

        # Image dimensions
        img_w = (panel_w - 30) // 2  # ~230px each
        img_h = 260

        # Primary
        img1_disp, r1 = self._resize_to_fit(img1, img_w, img_h)
        ix1 = panel_x + 10
        iy = panel_y + 40
        canvas[iy:iy + img1_disp.shape[0], ix1:ix1 + img1_disp.shape[1]] = img1_disp

        # Draw face box on primary
        verified = match_data.get("verified", False)
        box_color = NEON_GREEN if verified else NEON_RED
        if face1_box:
            self._draw_face_box(canvas, face1_box, r1, ix1, iy, box_color, "PRIMARY")

        # Comparison
        img2_disp, r2 = self._resize_to_fit(img2, img_w, img_h)
        ix2 = panel_x + 10 + img_w + 10
        canvas[iy:iy + img2_disp.shape[0], ix2:ix2 + img2_disp.shape[1]] = img2_disp

        if face2_box:
            self._draw_face_box(canvas, face2_box, r2, ix2, iy, box_color, "COMPARE")

        # Similarity line between faces
        if face1_box and face2_box:
            c1x = ix1 + img1_disp.shape[1] // 2
            c1y = iy + img1_disp.shape[0] // 2
            c2x = ix2 + img2_disp.shape[1] // 2
            c2y = iy + img2_disp.shape[0] // 2
            cv2.line(canvas, (c1x, c1y), (c2x, c2y), box_color, 1, cv2.LINE_AA)
            mx = (c1x + c2x) // 2
            my = (c1y + c2y) // 2
            cosine = match_data.get("cosine_similarity", 0)
            pct = round((cosine + 1.0) * 50.0) if cosine else 0
            cv2.circle(canvas, (mx, my), 20, BLACK, -1)
            cv2.circle(canvas, (mx, my), 20, box_color, 1, cv2.LINE_AA)
            cv2.putText(
                canvas, f"{pct}%", (mx - 14, my + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, box_color, 1, cv2.LINE_AA,
            )

        # Face crop close-ups
        crop_y = iy + max(img1_disp.shape[0], img2_disp.shape[0]) + 15
        cv2.putText(
            canvas, "FACE CROPS", (panel_x + 15, crop_y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, DIM_TEXT, 1, cv2.LINE_AA,
        )
        crop_y += 10
        crop_size = 120

        for label, img, box, cx in [
            ("PRI", img1, face1_box, ix1),
            ("CMP", img2, face2_box, ix2),
        ]:
            if box:
                crop = self._safe_crop(img, box)
                if crop is not None:
                    crop_disp = cv2.resize(crop, (crop_size, crop_size))
                    cy_end = min(crop_y + crop_size, canvas.shape[0])
                    cx_end = min(cx + crop_size, canvas.shape[1])
                    canvas[crop_y:cy_end, cx:cx_end] = crop_disp[:cy_end - crop_y, :cx_end - cx]
                    cv2.rectangle(canvas, (cx, crop_y), (cx + crop_size, crop_y + crop_size), box_color, 1)
                    seam_data = ((advanced_biometrics or {}).get("primary" if label == "PRI" else "comparison", {})
                                 .get("tampering", {}).get("micro_seam_analysis", {}))
                    self._draw_micro_seam_box(canvas, cx, crop_y, crop_size, seam_data)
                    cv2.putText(
                        canvas, label, (cx + 3, crop_y + crop_size - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, NEON_CYAN, 1,
                    )

    # ------------------------------------------------------------------
    # CENTER PANEL: Forensic overlays (580px wide)
    # ------------------------------------------------------------------
    def _draw_forensic_panel(
        self, canvas: np.ndarray,
        forensic_data: dict = None,
        deepfake_data: dict = None,
        gradcam_path: str = None,
        evidence_dir: str = None,
    ) -> None:
        panel_x, panel_y = 510, 90
        panel_w, panel_h = 590, 590

        cv2.rectangle(
            canvas,
            (panel_x, panel_y),
            (panel_x + panel_w, panel_y + panel_h),
            PANEL_BG, -1,
        )
        self._draw_bracket(canvas, panel_x, panel_y, panel_w, panel_h)

        cv2.putText(
            canvas, "FORENSIC ANALYSIS", (panel_x + 15, panel_y + 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, HEADER_GOLD, 1, cv2.LINE_AA,
        )

        current_y = panel_y + 45
        overlay_w = (panel_w - 30) // 2  # ~280px each
        overlay_h = 180

        # GradCAM overlay
        cv2.putText(
            canvas, "GradCAM OVERLAY", (panel_x + 10, current_y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, NEON_CYAN, 1, cv2.LINE_AA,
        )
        current_y += 8

        gradcam_img = None
        if gradcam_path and os.path.exists(gradcam_path):
            gradcam_img = cv2.imread(gradcam_path)
        if gradcam_img is not None:
            gd, _ = self._resize_to_fit(gradcam_img, overlay_w, overlay_h)
            canvas[current_y:current_y + gd.shape[0], panel_x + 10:panel_x + 10 + gd.shape[1]] = gd
        else:
            self._draw_placeholder(canvas, panel_x + 10, current_y, overlay_w, overlay_h, "NO GRADCAM")

        # ELA / Spectral heatmap
        cv2.putText(
            canvas, "ELA / SPECTRAL", (panel_x + overlay_w + 20, current_y - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, NEON_CYAN, 1, cv2.LINE_AA,
        )

        ela_placed = False
        if forensic_data and forensic_data.get("ela_map") is not None:
            ela_raw = forensic_data["ela_map"]
            ela_gray = cv2.cvtColor(ela_raw, cv2.COLOR_BGR2GRAY) if len(ela_raw.shape) == 3 else ela_raw
            ela_heatmap = cv2.applyColorMap(ela_gray, cv2.COLORMAP_JET)
            ed, _ = self._resize_to_fit(ela_heatmap, overlay_w, overlay_h)
            ex = panel_x + overlay_w + 20
            canvas[current_y:current_y + ed.shape[0], ex:ex + ed.shape[1]] = ed
            ela_placed = True

        # Try spectral_analysis.jpg from evidence_dir
        if not ela_placed and evidence_dir:
            spectral_path = os.path.join(evidence_dir, "spectral_analysis.jpg")
            if os.path.exists(spectral_path):
                spec_img = cv2.imread(spectral_path)
                if spec_img is not None:
                    sd, _ = self._resize_to_fit(spec_img, overlay_w, overlay_h)
                    ex = panel_x + overlay_w + 20
                    canvas[current_y:current_y + sd.shape[0], ex:ex + sd.shape[1]] = sd
                    ela_placed = True

        if not ela_placed:
            self._draw_placeholder(canvas, panel_x + overlay_w + 20, current_y, overlay_w, overlay_h, "NO SPECTRAL")

        current_y += overlay_h + 15

        # Tamper heatmap
        cv2.putText(
            canvas, "TAMPER HEATMAP", (panel_x + 10, current_y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, NEON_CYAN, 1, cv2.LINE_AA,
        )
        current_y += 8

        tamper_placed = False
        if evidence_dir:
            tamper_path = os.path.join(evidence_dir, "tamper_heatmap.jpg")
            if os.path.exists(tamper_path):
                tmp_img = cv2.imread(tamper_path)
                if tmp_img is not None:
                    td, _ = self._resize_to_fit(tmp_img, overlay_w, overlay_h)
                    canvas[current_y:current_y + td.shape[0], panel_x + 10:panel_x + 10 + td.shape[1]] = td
                    tamper_placed = True

        if not tamper_placed:
            self._draw_placeholder(canvas, panel_x + 10, current_y, overlay_w, overlay_h, "NO TAMPER MAP")

        # NoisePrint / rPPG pulse
        cv2.putText(
            canvas, "rPPG PULSE", (panel_x + overlay_w + 20, current_y - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, NEON_CYAN, 1, cv2.LINE_AA,
        )
        np_placed = False
        if evidence_dir:
            np_path = os.path.join(evidence_dir, "biometric_pulse.png")
            if os.path.exists(np_path):
                np_img = cv2.imread(np_path)
                if np_img is not None:
                    nd, _ = self._resize_to_fit(np_img, overlay_w, overlay_h)
                    ex = panel_x + overlay_w + 20
                    canvas[current_y:current_y + nd.shape[0], ex:ex + nd.shape[1]] = nd
                    np_placed = True

        if not np_placed:
            self._draw_placeholder(canvas, panel_x + overlay_w + 20, current_y, overlay_w, overlay_h, "NO PULSE")

        current_y += overlay_h + 15

        # Deepfake / Spectral status bar
        if deepfake_data:
            is_fake = deepfake_data.get("is_fake", False)
            conf = deepfake_data.get("confidence", 0)
            df_color = NEON_RED if is_fake else NEON_GREEN
            df_text = "DEEPFAKE DETECTED" if is_fake else "AUTHENTIC SIGNAL"
            cv2.putText(
                canvas, df_text, (panel_x + 10, current_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, df_color, 2, cv2.LINE_AA,
            )
            cv2.putText(
                canvas, f"Spectral Confidence: {conf:.2f}%", (panel_x + 280, current_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, GRAY_TEXT, 1, cv2.LINE_AA,
            )
        current_y += 20

        # Forensic score bars
        if forensic_data:
            tamper_score = forensic_data.get("tamper_score", 0)
            t_color = NEON_RED if forensic_data.get("is_suspicious") else NEON_GREEN
            self._draw_bar(canvas, panel_x + 10, current_y, 260, "TAMPER", tamper_score, t_color)

    # ------------------------------------------------------------------
    # RIGHT PANEL: Reconstruction + data readout (790px wide)
    # ------------------------------------------------------------------
    def _draw_recon_panel(
        self, canvas: np.ndarray,
        reconstruction_path: str = None,
        match_data: dict = None,
        compliance_data: dict = None,
        forensic_data: dict = None,
    ) -> None:
        panel_x, panel_y = 1110, 90
        panel_w, panel_h = 800, 590

        cv2.rectangle(
            canvas,
            (panel_x, panel_y),
            (panel_x + panel_w, panel_y + panel_h),
            PANEL_BG, -1,
        )
        self._draw_bracket(canvas, panel_x, panel_y, panel_w, panel_h)

        cv2.putText(
            canvas, "RECONSTRUCTION & DATA", (panel_x + 15, panel_y + 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, HEADER_GOLD, 1, cv2.LINE_AA,
        )

        # Reconstruction image — main render
        recon_y = panel_y + 40
        recon_w, recon_h = 200, 200
        thumb_w, thumb_h = 140, 140

        if reconstruction_path and os.path.exists(reconstruction_path):
            recon_img = cv2.imread(reconstruction_path)
            if recon_img is not None:
                rd, _ = self._resize_to_fit(recon_img, recon_w, recon_h)
                canvas[recon_y:recon_y + rd.shape[0], panel_x + 10:panel_x + 10 + rd.shape[1]] = rd
                cv2.rectangle(
                    canvas,
                    (panel_x + 10, recon_y),
                    (panel_x + 10 + rd.shape[1], recon_y + rd.shape[0]),
                    NEON_CYAN, 1,
                )
                cv2.putText(
                    canvas, "Deep3D + BFM RECONSTRUCTION",
                    (panel_x + 10, recon_y + rd.shape[0] + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, DIM_TEXT, 1, cv2.LINE_AA,
                )

            # Load companion renders (geometry, depth, sideview) if they exist
            base = reconstruction_path
            geo_path = base.replace(".jpg", "_geometry.jpg").replace(".png", "_geometry.png")
            depth_path = base.replace(".jpg", "_depth.jpg").replace(".png", "_depth.png")
            side_path = base.replace(".jpg", "_sideview.jpg").replace(".png", "_sideview.png")

            thumb_x_start = panel_x + 10 + recon_w + 15
            for i, (tp, tlabel) in enumerate([
                (geo_path, "GEOMETRY"),
                (depth_path, "DEPTH"),
                (side_path, "SIDE VIEW"),
            ]):
                if os.path.exists(tp):
                    timg = cv2.imread(tp)
                    if timg is not None:
                        td, _ = self._resize_to_fit(timg, thumb_w, thumb_h)
                        tx = thumb_x_start + i * (thumb_w + 8)
                        ty = recon_y
                        if tx + td.shape[1] <= panel_x + panel_w - 5 and ty + td.shape[0] <= panel_y + panel_h - 30:
                            canvas[ty:ty + td.shape[0], tx:tx + td.shape[1]] = td
                            cv2.rectangle(canvas, (tx, ty), (tx + td.shape[1], ty + td.shape[0]), NEON_BLUE, 1)
                            cv2.putText(
                                canvas, tlabel, (tx + 2, ty + td.shape[0] + 12),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.28, DIM_TEXT, 1, cv2.LINE_AA,
                            )
        else:
            self._draw_placeholder(canvas, panel_x + 10, recon_y, recon_w, recon_h, "NO RECON")

        # Terminal-style data readout (below reconstruction renders — 2-column layout)
        col1_x = panel_x + 15
        col2_x = panel_x + 410
        data_y = recon_y + max(recon_h, thumb_h) + 30
        line_h = 16

        # ---- Column 1: Match data + Forensic signals ----
        cv2.putText(
            canvas, "> MATCH TELEMETRY", (col1_x, data_y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.42, NEON_GREEN, 1, cv2.LINE_AA,
        )
        dy1 = data_y + line_h + 4
        cv2.line(canvas, (col1_x, dy1), (col1_x + 370, dy1), SEPARATOR, 1)
        dy1 += 8

        if match_data:
            calibration = match_data.get("calibration_features", {}) or {}
            entries = [
                ("COSINE SIM", f"{match_data.get('cosine_similarity', 0):.4f}"),
                ("FUSION SCORE", f"{match_data.get('fusion_score', 0):.4f}"),
                ("CONFIDENCE", f"{match_data.get('confidence', 0):.4f}"),
                ("CALIBRATED", f"{float(calibration.get('calibrated_confidence', match_data.get('confidence', 0))):.4f}"),
                ("VERIFIED", str(match_data.get("verified", False))),
                ("THRESHOLD", f"{match_data.get('threshold', 0):.3f}"),
                ("QUALITY GATE", str(match_data.get("quality_gate_passed", "N/A"))),
                ("AGREEMENT", str(match_data.get("agreement", "N/A")).upper()),
            ]
            for label, val in entries:
                val_color = NEON_GREEN if "True" in val or float_safe(val) > 0.5 else NEON_AMBER
                cv2.putText(
                    canvas, f"{label}:", (col1_x, dy1),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, DIM_TEXT, 1, cv2.LINE_AA,
                )
                cv2.putText(
                    canvas, val, (col1_x + 130, dy1),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, val_color, 1, cv2.LINE_AA,
                )
                dy1 += line_h
            for risk_flag in list(match_data.get("risk_flags", []) or [])[:3]:
                cv2.putText(
                    canvas, "RISK:", (col1_x, dy1),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.33, DIM_TEXT, 1, cv2.LINE_AA,
                )
                cv2.putText(
                    canvas, str(risk_flag)[:34], (col1_x + 130, dy1),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.33, NEON_RED, 1, cv2.LINE_AA,
                )
                dy1 += line_h - 1

        dy1 += 4
        cv2.line(canvas, (col1_x, dy1), (col1_x + 370, dy1), SEPARATOR, 1)
        dy1 += 8

        if forensic_data:
            freq = forensic_data.get("frequency", {})
            rppg = forensic_data.get("rppg", {})
            cv2.putText(
                canvas, "> FORENSIC SIGNALS", (col1_x, dy1),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, NEON_CYAN, 1, cv2.LINE_AA,
            )
            dy1 += line_h + 4

            df_prob = freq.get("deepfake_probability", 0)
            df_susp = freq.get("deepfake_suspected", False)
            live = rppg.get("is_live", False)
            bpm = rppg.get("bpm")
            signal_state = str(rppg.get("signal_state", "unknown"))

            for label, val, is_bad in [
                ("DEEPFAKE PROB", f"{df_prob:.3f}", df_prob > 0.5),
                ("DEEPFAKE FLAG", str(df_susp), df_susp),
                ("rPPG STATE", signal_state.upper(), signal_state == "spoof"),
                ("rPPG LIVE", "N/A" if signal_state == "not_available" else str(live), signal_state == "spoof" or (signal_state != "not_available" and not live)),
                ("BPM", str(bpm) if bpm else "NULL", signal_state != "not_available" and bpm is None),
            ]:
                c = NEON_RED if is_bad else NEON_GREEN
                if signal_state == "not_available" and label in {"rPPG STATE", "rPPG LIVE", "BPM"}:
                    c = NEON_AMBER
                cv2.putText(canvas, f"  {label}:", (col1_x, dy1), cv2.FONT_HERSHEY_SIMPLEX, 0.33, DIM_TEXT, 1, cv2.LINE_AA)
                cv2.putText(canvas, val, (col1_x + 140, dy1), cv2.FONT_HERSHEY_SIMPLEX, 0.33, c, 1, cv2.LINE_AA)
                dy1 += line_h - 1

        # ---- Column 2: ISO Compliance ----
        cv2.putText(
            canvas, "> ISO 19794-5 COMPLIANCE", (col2_x, data_y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.42, NEON_CYAN, 1, cv2.LINE_AA,
        )
        dy2 = data_y + line_h + 4
        cv2.line(canvas, (col2_x, dy2), (col2_x + 370, dy2), SEPARATOR, 1)
        dy2 += 8

        if compliance_data:
            checks = compliance_data.get("checks", compliance_data)
            for key, res in checks.items():
                if key in ("status", "is_compliant") or not isinstance(res, dict):
                    continue
                passed = res.get("passed", False)
                sc = res.get("score", 0)
                c = NEON_GREEN if passed else NEON_RED
                label_short = key.upper()[:20]
                cv2.putText(
                    canvas, f"  {label_short}", (col2_x, dy2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.33, c, 1, cv2.LINE_AA,
                )
                cv2.putText(
                    canvas, f"{'PASS' if passed else 'FAIL'} ({sc:.2f})" if isinstance(sc, float) else f"{'PASS' if passed else 'FAIL'}",
                    (col2_x + 190, dy2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.33, c, 1, cv2.LINE_AA,
                )
                dy2 += line_h - 1
                if dy2 > panel_y + panel_h - 20:
                    break

    # ------------------------------------------------------------------
    # BOTTOM PANEL: Advanced Biometrics Grid (full width, y=690)
    # ------------------------------------------------------------------
    def _draw_biometrics_grid(
        self, canvas: np.ndarray,
        advanced_biometrics: dict = None,
        biometric_data: dict = None,
    ) -> None:
        panel_x, panel_y = 10, 690
        panel_w, panel_h = 1900, 320

        cv2.rectangle(canvas, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), PANEL_BG, -1)
        self._draw_bracket(canvas, panel_x, panel_y, panel_w, panel_h)

        cv2.putText(
            canvas, "ADVANCED BIOMETRIC ANALYSIS GRID // STAGE 3.5",
            (panel_x + 15, panel_y + 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, HEADER_GOLD, 1, cv2.LINE_AA,
        )

        if not advanced_biometrics:
            cv2.putText(
                canvas, "[ NO ADVANCED BIOMETRIC DATA AVAILABLE ]",
                (panel_x + 600, panel_y + 160),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, DIM_TEXT, 1, cv2.LINE_AA,
            )
            return

        # Extract data
        primary = advanced_biometrics.get("primary", {})
        comparison = advanced_biometrics.get("comparison", {})
        pair = advanced_biometrics.get("pair_analysis", {})

        # Module status grid (7 columns)
        modules = [
            ("TAMPERING", "tampering", "tampering_detected"),
            ("MORPHING", "morphing", "is_morphed"),
            ("DISGUISE", "makeup_disguise", "disguise_detected"),
            ("IRIS SPOOF", "iris", None),
            ("UNIQUENESS", "uniqueness", None),
            ("MARKERS", "facial_markers", None),
            ("AGE-INV", "age_invariant", None),
        ]

        col_w = (panel_w - 40) // 7
        grid_y = panel_y + 45

        for i, (label, key, flag_key) in enumerate(modules):
            cx = panel_x + 20 + i * col_w

            # Module header
            cv2.putText(
                canvas, label, (cx, grid_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, WHITE, 1, cv2.LINE_AA,
            )

            # Primary + Comparison status
            pri_data = primary.get(key, {})
            comp_data = comparison.get(key, {})

            for row_label, data, row_y_offset in [("PRI", pri_data, 25), ("CMP", comp_data, 45)]:
                ry = grid_y + row_y_offset

                if flag_key and isinstance(data, dict):
                    detected = data.get(flag_key, False)
                    c = NEON_RED if detected else NEON_GREEN
                    text = "DETECTED" if detected else "CLEAR"
                elif key == "iris":
                    spoof = data.get("anti_spoofing", {}) if isinstance(data, dict) else {}
                    sclera = data.get("sclera_analysis", {}) if isinstance(data, dict) else {}
                    lens = spoof.get("contact_lens_detected", False)
                    sclera_ai = sclera.get("deepfake_suspected", False)
                    c = NEON_RED if lens or sclera_ai else NEON_GREEN
                    if sclera_ai:
                        text = "SCLERA AI"
                    else:
                        text = "LENS DET" if lens else "CLEAR"
                elif key == "uniqueness":
                    score = data.get("uniqueness_score", 0) if isinstance(data, dict) else 0
                    c = NEON_GREEN if score >= 0.5 else NEON_AMBER if score >= 0.3 else NEON_RED
                    text = f"{score:.2f}"
                elif key == "facial_markers":
                    count = data.get("markers_detected", 0) if isinstance(data, dict) else 0
                    c = NEON_CYAN
                    text = f"{count} found"
                elif key == "age_invariant":
                    conf = data.get("extraction_confidence", 0) if isinstance(data, dict) else 0
                    c = NEON_GREEN if conf >= 0.5 else NEON_AMBER
                    text = f"{conf:.2f}"
                else:
                    c = DIM_TEXT
                    text = "N/A"

                cv2.putText(
                    canvas, f"{row_label}: {text}", (cx, ry),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, c, 1, cv2.LINE_AA,
                )

        # Threat level bars
        threat_y = grid_y + 75
        cv2.line(canvas, (panel_x + 20, threat_y), (panel_x + panel_w - 20, threat_y), SEPARATOR, 1)
        threat_y += 15

        cv2.putText(
            canvas, "THREAT ASSESSMENT", (panel_x + 20, threat_y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, NEON_AMBER, 1, cv2.LINE_AA,
        )
        threat_y += 10

        for label, bio_data in [("PRIMARY", primary), ("COMPARISON", comparison)]:
            threat_score = bio_data.get("threat_score", 0)
            threat_level = bio_data.get("threat_level", "N/A")
            t_color = self._threat_color(threat_level)

            self._draw_bar(canvas, panel_x + 20, threat_y, 400, f"{label} THREAT", threat_score, t_color)
            cv2.putText(
                canvas, threat_level, (panel_x + 570, threat_y + 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, t_color, 1, cv2.LINE_AA,
            )
            threat_y += 25

        # Pair analysis summary
        pair_y = threat_y + 5
        pair_verdict = pair.get("final_verdict", pair.get("verdict", "N/A"))
        pair_conf = pair.get("confidence", 0)
        is_doppel = pair.get("doppelganger_analysis", {}).get("is_doppelganger", False)
        kinship = pair.get("kinship_analysis", {})

        cv2.putText(
            canvas, "PAIR VERDICT:", (panel_x + 20, pair_y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1, cv2.LINE_AA,
        )

        vcolor = NEON_GREEN if pair_verdict == "VERIFIED" else NEON_RED if pair_verdict == "REJECT" else NEON_AMBER
        cv2.putText(
            canvas, f"{pair_verdict} ({pair_conf:.1f}%)", (panel_x + 180, pair_y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, vcolor, 2, cv2.LINE_AA,
        )

        # Doppelganger status
        cv2.putText(
            canvas, "DOPPELGANGER:", (panel_x + 500, pair_y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, DIM_TEXT, 1, cv2.LINE_AA,
        )
        d_col = NEON_RED if is_doppel else NEON_GREEN
        cv2.putText(
            canvas, "SUSPECTED" if is_doppel else "CLEAR", (panel_x + 660, pair_y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, d_col, 1, cv2.LINE_AA,
        )

        kinship_y = pair_y + 22
        kinship_prob = kinship.get("kinship_probability", 0)
        kinship_label = str(kinship.get("relationship_hypothesis", "not_indicated")).upper()
        kin_col = NEON_AMBER if kinship.get("likely_related") else DIM_TEXT
        cv2.putText(
            canvas, "KINSHIP:", (panel_x + 500, kinship_y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.42, DIM_TEXT, 1, cv2.LINE_AA,
        )
        cv2.putText(
            canvas, f"{kinship_label} ({kinship_prob:.1f}%)", (panel_x + 620, kinship_y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, kin_col, 1, cv2.LINE_AA,
        )

        # Alerts
        alerts = primary.get("alerts", []) + comparison.get("alerts", [])
        if alerts:
            alert_y = pair_y + 45
            cv2.putText(
                canvas, "ALERTS:", (panel_x + 20, alert_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, NEON_RED, 1, cv2.LINE_AA,
            )
            for a in alerts[:4]:
                alert_y += 16
                cv2.putText(
                    canvas, f"  ! {a}", (panel_x + 20, alert_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, NEON_RED, 1, cv2.LINE_AA,
                )

        # Recommendations
        recs = pair.get("recommendations", [])
        if recs:
            rec_x = panel_x + 900
            rec_y = grid_y + 75 + 15
            cv2.putText(
                canvas, "RECOMMENDATIONS:", (rec_x, rec_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, NEON_CYAN, 1, cv2.LINE_AA,
            )
            for r in recs[:6]:
                rec_y += 18
                r_text = r[:70] + "..." if len(r) > 70 else r
                cv2.putText(
                    canvas, f"> {r_text}", (rec_x, rec_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, GRAY_TEXT, 1, cv2.LINE_AA,
                )

    # ------------------------------------------------------------------
    # FOOTER: Verdict stamp
    # ------------------------------------------------------------------
    def _draw_footer(
        self, canvas: np.ndarray,
        match_data: dict,
        advanced_biometrics: dict = None,
    ) -> None:
        footer_y = self.CANVAS_H - 60
        cv2.line(canvas, (0, footer_y - 5), (self.CANVAS_W, footer_y - 5), HEADER_GOLD, 2)

        verified = match_data.get("verified", False)
        pair = (advanced_biometrics or {}).get("pair_analysis", {})
        bio_verdict = pair.get("final_verdict", pair.get("verdict", ""))

        # Determine overall decision
        if verified and bio_verdict not in ("REJECT",):
            decision = "IDENTITY VERIFIED"
            d_color = NEON_GREEN
        elif bio_verdict == "REJECT":
            decision = "IDENTITY REJECTED -- BIOMETRIC THREAT"
            d_color = NEON_RED
        else:
            decision = "IDENTITY REJECTED"
            d_color = NEON_RED

        # Stamp rectangle
        cv2.rectangle(canvas, (20, footer_y - 2), (550, footer_y + 40), d_color, -1)
        cv2.putText(
            canvas, decision, (30, footer_y + 28),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, BLACK, 2, cv2.LINE_AA,
        )

        # Ensemble confidence
        ensemble_conf = float(match_data.get("confidence", 0.0) or 0.0) * 100.0
        cv2.putText(
            canvas, f"ENSEMBLE CONFIDENCE: {ensemble_conf:.2f}%", (580, footer_y + 28),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 1, cv2.LINE_AA,
        )

        # Bio verdict
        if bio_verdict:
            bio_color = NEON_GREEN if bio_verdict == "VERIFIED" else NEON_RED if bio_verdict == "REJECT" else NEON_AMBER
            cv2.putText(
                canvas, f"BIO: {bio_verdict}", (1000, footer_y + 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, bio_color, 1, cv2.LINE_AA,
            )

        # Platform tag
        cv2.putText(
            canvas, "CA_MONK v4 // Tier-1 Biometric Intelligence Platform",
            (1350, footer_y + 28),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, DIM_TEXT, 1, cv2.LINE_AA,
        )

    # ------------------------------------------------------------------
    # Drawing utilities
    # ------------------------------------------------------------------
    @staticmethod
    def _resize_to_fit(
        img: np.ndarray, max_w: int, max_h: int,
    ) -> Tuple[np.ndarray, float]:
        """Resize image to fit within max_w x max_h, preserving aspect ratio."""
        h, w = img.shape[:2]
        ratio = min(max_w / w, max_h / h)
        new_w = int(w * ratio)
        new_h = int(h * ratio)
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return resized, ratio

    @staticmethod
    def _draw_bracket(
        canvas: np.ndarray, x: int, y: int, w: int, h: int, size: int = 15,
    ) -> None:
        """Draw corner brackets (military HUD style)."""
        color = NEON_GREEN
        t = 1
        # Top-left
        cv2.line(canvas, (x, y), (x + size, y), color, t)
        cv2.line(canvas, (x, y), (x, y + size), color, t)
        # Top-right
        cv2.line(canvas, (x + w, y), (x + w - size, y), color, t)
        cv2.line(canvas, (x + w, y), (x + w, y + size), color, t)
        # Bottom-left
        cv2.line(canvas, (x, y + h), (x + size, y + h), color, t)
        cv2.line(canvas, (x, y + h), (x, y + h - size), color, t)
        # Bottom-right
        cv2.line(canvas, (x + w, y + h), (x + w - size, y + h), color, t)
        cv2.line(canvas, (x + w, y + h), (x + w, y + h - size), color, t)

    @staticmethod
    def _draw_placeholder(
        canvas: np.ndarray, x: int, y: int, w: int, h: int, text: str,
    ) -> None:
        cv2.rectangle(canvas, (x, y), (x + w, y + h), PANEL_BORDER, 1)
        cv2.putText(
            canvas, text, (x + w // 4, y + h // 2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, DIM_TEXT, 1, cv2.LINE_AA,
        )

    @staticmethod
    def _draw_bar(
        canvas: np.ndarray, x: int, y: int, w: int,
        label: str, value: float, color: Tuple[int, int, int],
    ) -> None:
        """Draw a horizontal progress bar."""
        bar_h = 14
        cv2.putText(canvas, label, (x, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.33, DIM_TEXT, 1, cv2.LINE_AA)
        bar_x = x + 130
        cv2.rectangle(canvas, (bar_x, y), (bar_x + w, y + bar_h), PANEL_BORDER, 1)
        fill_w = int(w * min(value, 1.0))
        if fill_w > 0:
            cv2.rectangle(canvas, (bar_x, y), (bar_x + fill_w, y + bar_h), color, -1)
        cv2.putText(
            canvas, f"{value:.2f}", (bar_x + w + 5, y + 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.3, GRAY_TEXT, 1, cv2.LINE_AA,
        )

    @staticmethod
    def _draw_face_box(
        canvas: np.ndarray, box: dict, ratio: float,
        offset_x: int, offset_y: int,
        color: Tuple[int, int, int], label: str,
    ) -> None:
        x = int(box["x"] * ratio) + offset_x
        y = int(box["y"] * ratio) + offset_y
        w = int(box["w"] * ratio)
        h = int(box["h"] * ratio)
        cv2.rectangle(canvas, (x, y), (x + w, y + h), color, 2)
        cv2.putText(
            canvas, label, (x, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA,
        )

    @staticmethod
    def _draw_micro_seam_box(
        canvas: np.ndarray,
        offset_x: int,
        offset_y: int,
        crop_size: int,
        seam_data: dict,
    ) -> None:
        normalized = seam_data.get("highlight_box_normalized") if isinstance(seam_data, dict) else None
        if not normalized:
            return

        x = offset_x + int(normalized.get("x", 0) * crop_size)
        y = offset_y + int(normalized.get("y", 0) * crop_size)
        w = max(8, int(normalized.get("w", 0.2) * crop_size))
        h = max(8, int(normalized.get("h", 0.2) * crop_size))

        for pad, color, thickness in [(4, (0, 0, 120), 1), (2, (0, 0, 180), 1), (0, NEON_RED, 2)]:
            cv2.rectangle(
                canvas,
                (max(0, x - pad), max(0, y - pad)),
                (min(canvas.shape[1] - 1, x + w + pad), min(canvas.shape[0] - 1, y + h + pad)),
                color,
                thickness,
            )
        cv2.putText(
            canvas, "SEAM", (x, max(offset_y + 10, y - 4)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.3, NEON_RED, 1, cv2.LINE_AA,
        )

    @staticmethod
    def _safe_crop(img: np.ndarray, box: dict) -> Optional[np.ndarray]:
        """Safely crop a face region from an image."""
        h, w = img.shape[:2]
        x1 = max(0, box.get("x", 0))
        y1 = max(0, box.get("y", 0))
        x2 = min(w, x1 + box.get("w", 0))
        y2 = min(h, y1 + box.get("h", 0))
        if x2 <= x1 or y2 <= y1:
            return None
        return img[y1:y2, x1:x2]

    @staticmethod
    def _threat_color(level: str) -> Tuple[int, int, int]:
        if level == "CRITICAL":
            return CRITICAL_RED
        if level == "HIGH":
            return NEON_RED
        if level == "MEDIUM":
            return NEON_AMBER
        return NEON_GREEN


def float_safe(val: str) -> float:
    """Try to parse a float, return 0.0 on failure."""
    try:
        return float(val)
    except (ValueError, TypeError):
        return 0.0
