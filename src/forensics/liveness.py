from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import cv2
import numpy as np

logger = logging.getLogger("ca_monk.rppg_liveness")


@dataclass
class RPPGEstimate:
    is_live: bool
    bpm: Optional[float]
    confidence: float
    details: Dict[str, float]
    bpm_series: List[float] = None  # Windowed BPM values for pulse graph

    def __post_init__(self):
        if self.bpm_series is None:
            self.bpm_series = []


class RPPGLivenessDetector:
    """
    CPU-focused rPPG module with pyVHR POS primary route.
    Generates a pulse graph image for evidence cards.
    """

    def __init__(self, min_bpm: float = 42.0, max_bpm: float = 180.0) -> None:
        self.min_bpm = min_bpm
        self.max_bpm = max_bpm

    def _face_signal(self, video_path: str) -> tuple[List[float], float]:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return [], 0.0

        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        detector = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        signal: List[float] = []

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
            if len(faces) == 0:
                continue
            x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
            crop = frame[y : y + h, x : x + w]
            if crop.size == 0:
                continue
            signal.append(float(np.mean(crop[:, :, 1].astype(np.float32))))
        cap.release()
        return signal, fps

    def _dominant_bpm(self, signal: np.ndarray, fps: float) -> Optional[float]:
        if fps <= 0.0 or signal.size < 90:
            return None
        sig = signal - np.mean(signal)
        freqs = np.fft.rfftfreq(sig.size, d=1.0 / fps)
        power = np.abs(np.fft.rfft(sig)) ** 2
        band = (freqs >= self.min_bpm / 60.0) & (freqs <= self.max_bpm / 60.0)
        if not np.any(band):
            return None
        idx = int(np.argmax(power[band]))
        return float(freqs[band][idx] * 60.0)

    def _bpm_series(self, signal: np.ndarray, fps: float) -> List[float]:
        if fps <= 0.0:
            return []
        window = max(int(6 * fps), 1)
        step = max(int(1 * fps), 1)
        series: List[float] = []
        if signal.size < window:
            bpm = self._dominant_bpm(signal, fps)
            return [bpm] if bpm is not None else []
        for start in range(0, signal.size - window + 1, step):
            bpm = self._dominant_bpm(signal[start : start + window], fps)
            if bpm is not None:
                series.append(float(bpm))
        return series

    def _evaluate(self, bpm_series: List[float], route: str) -> RPPGEstimate:
        if not bpm_series:
            return RPPGEstimate(False, None, 0.0, {"route": route, "bpm_variance": 0.0}, bpm_series=[])

        bpm = float(np.mean(bpm_series))
        variance = float(np.var(np.asarray(bpm_series, dtype=np.float32)))
        spoof = variance == 0.0 or bpm > 180.0
        in_range = self.min_bpm <= bpm <= self.max_bpm
        is_live = bool(in_range and not spoof)
        confidence = 0.85 if is_live else 0.2
        return RPPGEstimate(
            is_live=is_live,
            bpm=bpm,
            confidence=confidence,
            details={"route": route, "bpm_variance": variance},
            bpm_series=bpm_series,
        )

    def estimate(self, video_path: str) -> RPPGEstimate:
        try:
            from pyVHR.methods.base import method_factory
            from pyVHR.utils.printutils import get_fps

            pos = method_factory("POS")
            try:
                signal, _ = pos(video_path, cuda=False)
            except TypeError:
                signal, _ = pos(video_path)
            fps = float(get_fps(video_path))
            signal_np = np.asarray(signal).squeeze().astype(np.float32)
            series = self._bpm_series(signal_np, fps)
            return self._evaluate(series, route="pyvhr_pos")
        except Exception:
            signal, fps = self._face_signal(video_path)
            if not signal:
                return RPPGEstimate(False, None, 0.0, {"route": "fallback", "bpm_variance": 0.0}, bpm_series=[])
            series = self._bpm_series(np.asarray(signal, dtype=np.float32), fps)
            return self._evaluate(series, route="fallback_fft")

    # ------------------------------------------------------------------
    # Pulse Graph Generator (for evidence card)
    # ------------------------------------------------------------------
    def generate_pulse_graph(
        self,
        estimate: RPPGEstimate,
        save_path: Optional[str] = None,
    ) -> Optional[str]:
        """
        Plot the heart rate (BPM) time-series graph.
        A flat line indicates Fake/Mask.
        Returns path to saved PNG or None.
        """
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not available — cannot generate pulse graph.")
            return None

        fig, ax = plt.subplots(figsize=(8, 3), dpi=120)
        fig.patch.set_facecolor("#1a1a2e")
        ax.set_facecolor("#16213e")

        series = estimate.bpm_series if estimate.bpm_series else []

        if series:
            t = list(range(len(series)))
            ax.plot(t, series, color="#00ff88", linewidth=1.8, label="BPM")
            ax.fill_between(t, series, alpha=0.15, color="#00ff88")

            # Healthy range band
            ax.axhspan(self.min_bpm, self.max_bpm, alpha=0.08, color="#00ff88")
            ax.axhline(y=float(np.mean(series)), color="#ff6b6b", linewidth=1, linestyle="--",
                        label=f"Mean: {np.mean(series):.0f} BPM")
        else:
            # Flat line = no pulse detected
            ax.axhline(y=0, color="#ff0000", linewidth=2, label="NO PULSE DETECTED")
            ax.text(0.5, 0.5, "FLAT LINE — FAKE / MASK / STILL IMAGE",
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=14, color="#ff4444", fontweight="bold")

        verdict = "LIVE ✓" if estimate.is_live else "DEAD / SPOOF ✗"
        verdict_color = "#00ff88" if estimate.is_live else "#ff4444"

        ax.set_title(f"rPPG BIOLOGICAL SCAN  —  {verdict}",
                     color=verdict_color, fontsize=11, fontweight="bold", pad=10)
        ax.set_xlabel("Time Window", color="#aaaaaa", fontsize=9)
        ax.set_ylabel("BPM", color="#aaaaaa", fontsize=9)
        ax.tick_params(colors="#888888", labelsize=8)
        ax.legend(loc="upper right", fontsize=8, facecolor="#1a1a2e", edgecolor="#333",
                  labelcolor="#cccccc")

        for spine in ax.spines.values():
            spine.set_color("#333333")

        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            plt.savefig(save_path, dpi=120, bbox_inches="tight",
                        facecolor=fig.get_facecolor(), edgecolor="none")
            logger.info("Pulse graph saved → %s", save_path)
            plt.close(fig)
            return save_path

        plt.close(fig)
        return None

