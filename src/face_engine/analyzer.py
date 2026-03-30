from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from src.core.contracts import EmbeddingResult, FaceBox, PairMatchResult
from src.core.gpu_guard import clear_cuda_cache
from src.face_engine.liveness import LivenessDetector
from src.face_engine.recognition import AdaFaceRecognizer
from src.face_engine.rotation_handler import RotationHandler
from src.face_engine.restoration import CodeFormerONNXRestorer
from src.face_engine.siamese_gradcam import SiameseGradCAM

logger = logging.getLogger("ca_monk.face_analyzer")

_MODEL_THRESHOLDS = {
    "adaface": 0.36,
    "buffalo_l": 0.33,
    "haar_fallback": 0.80,
}

_MODEL_WEIGHTS = {
    "adaface": 0.62,
    "buffalo_l": 0.38,
    "haar_fallback": 0.10,
}


def _is_image_file(path: str) -> bool:
    ext = Path(path).suffix.lower()
    return ext in {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}


def _aligned_crop_112(img: np.ndarray, face) -> np.ndarray:
    """
    Use InsightFace face_align for strict 112×112 aligned crop.
    Falls back to bounding-box crop + resize if landmarks unavailable.
    """
    try:
        from insightface.utils import face_align
        landmarks = getattr(face, "kps", None)
        if landmarks is not None and len(landmarks) >= 5:
            return face_align.norm_crop(img, landmarks, image_size=112)
    except Exception:
        pass
    # Fallback: simple bbox crop
    bbox = np.asarray(getattr(face, "bbox", [0, 0, 0, 0]), dtype=np.int32).tolist()
    x1, y1, x2, y2 = bbox if len(bbox) == 4 else [0, 0, 0, 0]
    crop = img[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
    if crop.size == 0:
        return crop
    return cv2.resize(crop, (112, 112), interpolation=cv2.INTER_AREA)


def _to_float_points(points: Any) -> list[list[float]]:
    arr = np.asarray(points, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] < 2:
        return []
    return [[float(x), float(y)] for x, y in arr[:, :2]]


class FaceAnalyzer:
    """
    InsightFace + ONNX Runtime face pipeline.

    Pipeline:
      1. Rotation correction (optional, via Haar detector)
      2. CodeFormer ONNX pre-restoration for blurry inputs
      3. InsightFace buffalo_l detection / alignment
      4. AdaFace ONNX embedding (or InsightFace fallback)
      5. MagFace quality scoring (norm < 22 = Low Fidelity, > 28 = Enrollment Quality)
      6. Context-aware still-image PAD / liveness check
    """

    def __init__(
        self,
        model_pack: str = "buffalo_l",
        det_size: Tuple[int, int] = (640, 640),
        embedding_quality_threshold: Optional[float] = None,
        enable_rotation: bool = True,
    ) -> None:
        self.model_pack = model_pack
        self.det_size = det_size
        self.quality_threshold = float(
            embedding_quality_threshold
            if embedding_quality_threshold is not None
            else os.getenv("EMBEDDING_NORM_THRESHOLD", "20.0")
        )
        self.liveness_detector = LivenessDetector()
        self.rotation_handler = RotationHandler() if enable_rotation else None
        self.recognizer = AdaFaceRecognizer(magface_norm_threshold=20.0)
        self.restorer = CodeFormerONNXRestorer()
        self.gradcam = SiameseGradCAM()
        self._fallback_detector = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self._app = self._init_insightface()

    def runtime_capabilities(self) -> Dict[str, Any]:
        return {
            "insightface_loaded": self._app is not None,
            "model_pack": self.model_pack,
            "det_size": list(self.det_size),
            "quality_threshold": float(self.quality_threshold),
            "recognizer": self.recognizer.capabilities(),
            "liveness": self.liveness_detector.capabilities(),
            "restoration": {
                "model_path": self.restorer.model_path,
                "loaded": self.restorer.session is not None,
                "blur_threshold": float(self.restorer.blur_threshold),
            },
        }

    def _init_insightface(self) -> Any:
        try:
            from insightface.app import FaceAnalysis

            app = FaceAnalysis(name=self.model_pack, providers=["CPUExecutionProvider"])
            app.prepare(ctx_id=-1, det_size=self.det_size)
            return app
        except Exception:
            return None

    def calibrate_quality_threshold(self, calibration_dir: str) -> float:
        norms: List[float] = []
        base = Path(calibration_dir)
        if not base.exists():
            return self.quality_threshold

        for p in base.rglob("*"):
            if not p.is_file() or not _is_image_file(str(p)):
                continue
            faces = self.get_face_embeddings(str(p))
            for face in faces:
                norms.append(float(face["embedding_norm"]))

        if norms:
            percentile = float(os.getenv("EMBEDDING_NORM_CALIB_PERCENTILE", "5"))
            percentile = max(0.0, min(100.0, percentile))
            calibrated = float(np.percentile(np.array(norms, dtype=np.float32), percentile))
            self.quality_threshold = max(1e-6, calibrated)
        return self.quality_threshold

    def assess_quality(self, embedding_norm: float) -> str:
        if float(embedding_norm) < 20.0:
            return "unidentifiable_noise"
        return (
            "reliable"
            if float(embedding_norm) >= float(self.quality_threshold)
            else "low_quality_unreliable"
        )

    def get_face_embeddings(self, img_path: str) -> List[Dict[str, Any]]:
        img = cv2.imread(img_path)
        if img is None:
            return []

        rotation_angle = 0
        processed = img
        if self.rotation_handler is not None:
            processed, rotation_angle = self.rotation_handler.find_best_rotation_array(img)
        if self.restorer.should_restore(processed):
            restored, restored_ok, restore_metrics = self.restorer.restore_if_beneficial(processed)
            if restored_ok:
                processed = restored
                logger.debug(
                    "Analyzer restoration accepted: sharpness_gain=%.3f mean_shift=%.2f saturation_shift=%.2f",
                    float(restore_metrics.get("sharpness_gain", 0.0)),
                    float(restore_metrics.get("mean_shift", 0.0)),
                    float(restore_metrics.get("saturation_shift", 0.0)),
                )

        try:
            if self._app is None:
                gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
                detections = self._fallback_detector.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
                )
                faces = []
                for (x, y, w, h) in detections:
                    crop = processed[y : y + h, x : x + w]
                    if crop.size == 0:
                        continue
                    v = cv2.resize(crop, (56, 56), interpolation=cv2.INTER_AREA)
                    v = cv2.cvtColor(v, cv2.COLOR_BGR2GRAY).astype(np.float32).flatten()
                    v = v / (np.linalg.norm(v) + 1e-8)

                    class _F:
                        pass

                    f = _F()
                    f.embedding = v
                    f.bbox = [x, y, x + w, y + h]
                    f.det_score = 0.5
                    faces.append(f)
            else:
                faces = self._app.get(processed)
        except Exception:
            clear_cuda_cache()
            return []

        out: List[Dict[str, Any]] = []
        for face in faces:
            bbox = np.asarray(getattr(face, "bbox", [0, 0, 0, 0]), dtype=np.int32).tolist()
            x1, y1, x2, y2 = bbox if len(bbox) == 4 else [0, 0, 0, 0]
            # Strict 112×112 aligned crop via InsightFace face_align
            aligned_crop = _aligned_crop_112(processed, face)
            crop = processed[max(0, y1) : max(0, y2), max(0, x1) : max(0, x2)]

            # Use aligned crop for embedding
            emb_input = aligned_crop if aligned_crop.size > 0 else crop
            ada_emb, ada_norm = self.recognizer.embedding(emb_input)
            insight_embedding = getattr(face, "embedding", None)
            embeddings: Dict[str, List[float]] = {}
            if ada_emb is not None and ada_norm is not None:
                emb_vec = np.asarray(ada_emb, dtype=np.float32)
                emb_norm = float(ada_norm)
                quality = self.recognizer.quality_label(emb_norm)
                model_name = "adaface"
                embeddings["adaface"] = emb_vec.tolist()
            else:
                if insight_embedding is None:
                    continue
                emb_vec = np.asarray(insight_embedding, dtype=np.float32)
                emb_norm = float(np.linalg.norm(emb_vec))
                quality = self.assess_quality(emb_norm)
                model_name = "buffalo_l"

            if insight_embedding is not None:
                embeddings["buffalo_l"] = np.asarray(insight_embedding, dtype=np.float32).tolist()
            if not embeddings:
                embeddings[model_name] = emb_vec.tolist()

            liveness = self.liveness_detector.check_liveness(
                crop if crop.size else processed,
                frame_img=processed,
                face_box={"x": int(x1), "y": int(y1), "w": int(x2 - x1), "h": int(y2 - y1)},
            )

            # MagFace quality description
            magface_desc = self.recognizer.quality_description(emb_norm)

            # Demographics from InsightFace genderage model
            det_age = int(getattr(face, "age", -1))
            det_sex_raw = getattr(face, "sex", None)
            # InsightFace: sex='M'/'F' or gender=0/1
            if det_sex_raw is None:
                det_gender_code = getattr(face, "gender", None)
                det_sex_raw = "M" if det_gender_code == 1 else ("F" if det_gender_code == 0 else None)
            det_sex = str(det_sex_raw) if det_sex_raw is not None else "unknown"

            out.append(
                {
                    "embedding": emb_vec.tolist(),
                    "embeddings": embeddings,
                    "embedding_norm": emb_norm,
                    "quality": {
                        "label": quality,
                        "magface": magface_desc,
                        "threshold": self.quality_threshold,
                        "passed": quality == "reliable",
                    },
                    "box": {"x": int(x1), "y": int(y1), "w": int(x2 - x1), "h": int(y2 - y1)},
                    "confidence": float(getattr(face, "det_score", 0.0)),
                    "rotation_angle": rotation_angle,
                    "liveness": liveness,
                    "liveness_score": float(liveness.get("score", 0.0)),
                    "source_path": img_path,
                    "quality_score": float(getattr(face, "det_score", 0.0)),
                    "model_name": model_name,
                    "landmarks": {
                        "kps5": _to_float_points(getattr(face, "kps", [])),
                        "points_106": _to_float_points(getattr(face, "landmark_2d_106", [])),
                    },
                    "demographics": {
                        "estimated_age": det_age if det_age >= 0 else None,
                        "sex": det_sex,
                        "model": "insightface_genderage",
                    },
                    "liveness_runtime": self.liveness_detector.capabilities(),
                }
            )
        return out

    def _extract_vector(self, emb: Any) -> np.ndarray:
        if isinstance(emb, dict):
            if "buffalo_l" in emb:
                return np.asarray(emb["buffalo_l"], dtype=np.float32)
            if emb:
                first = next(iter(emb.values()))
                return np.asarray(first, dtype=np.float32)
        return np.asarray(emb, dtype=np.float32)

    def verify_embeddings(self, emb1_dict: Any, emb2_dict: Any) -> Dict[str, Any]:
        v1 = self._extract_vector(emb1_dict)
        v2 = self._extract_vector(emb2_dict)
        n1 = float(np.linalg.norm(v1))
        n2 = float(np.linalg.norm(v2))
        denom = max(n1 * n2, 1e-8)
        cosine_similarity = float(np.dot(v1, v2) / denom)
        distance = float(1.0 - cosine_similarity)
        quality_ok = n1 >= 20.0 and n2 >= 20.0
        verified = bool(cosine_similarity >= 0.35 and quality_ok)
        return {
            "verified": verified,
            "distance": distance,
            "score": max(0.0, min(100.0, (cosine_similarity + 1.0) * 50.0)),
            "similarity": cosine_similarity,
            "model_details": {
                "buffalo_l": {"distance": distance, "verified": verified},
            },
            "vote_count": "1/1",
        }

    def _pair_model_scores(
        self,
        primary: EmbeddingResult,
        comparison: EmbeddingResult,
    ) -> Tuple[Dict[str, Dict[str, Any]], float, float, str]:
        primary_embeddings = dict(primary.embeddings or {})
        comparison_embeddings = dict(comparison.embeddings or {})

        if not primary_embeddings and primary.embedding:
            primary_embeddings[primary.model_name or "primary"] = list(primary.embedding)
        if not comparison_embeddings and comparison.embedding:
            comparison_embeddings[comparison.model_name or "comparison"] = list(comparison.embedding)

        common = [name for name in primary_embeddings if name in comparison_embeddings]
        if not common:
            return {}, 0.0, 0.0, "none"

        model_scores: Dict[str, Dict[str, Any]] = {}
        weighted_cosine = 0.0
        weighted_prob = 0.0
        total_weight = 0.0
        pass_count = 0

        for model_name in common:
            v1 = np.asarray(primary_embeddings[model_name], dtype=np.float32)
            v2 = np.asarray(comparison_embeddings[model_name], dtype=np.float32)
            denom = float(np.linalg.norm(v1) * np.linalg.norm(v2)) + 1e-8
            cosine = float(np.dot(v1, v2) / denom)
            threshold = float(_MODEL_THRESHOLDS.get(model_name, 0.35))
            base_weight = float(_MODEL_WEIGHTS.get(model_name, 0.20))
            prob = float(1.0 / (1.0 + np.exp(-(cosine - threshold) * 14.0)))
            passed = bool(cosine >= threshold)
            if passed:
                pass_count += 1

            weight = base_weight
            if primary.quality != "reliable" or comparison.quality != "reliable":
                weight *= 0.75

            weighted_cosine += cosine * weight
            weighted_prob += prob * weight
            total_weight += weight
            model_scores[model_name] = {
                "cosine_similarity": round(cosine, 4),
                "threshold": threshold,
                "passed": passed,
                "weight": round(weight, 4),
                "probability": round(prob, 4),
            }

        if total_weight <= 1e-8:
            return model_scores, 0.0, 0.0, "none"

        fused_cosine = float(weighted_cosine / total_weight)
        fused_prob = float(weighted_prob / total_weight)
        if pass_count == len(common):
            agreement = "strong"
        elif pass_count > 0:
            agreement = "mixed"
        else:
            agreement = "none"
        return model_scores, fused_cosine, fused_prob, agreement

    def compare_pair(
        self,
        primary: EmbeddingResult,
        comparison: EmbeddingResult,
        cosine_threshold: float = 0.35,
    ) -> PairMatchResult:
        model_scores, cosine, fusion_score, agreement = self._pair_model_scores(primary, comparison)
        if not model_scores:
            v1 = np.asarray(primary.embedding, dtype=np.float32)
            v2 = np.asarray(comparison.embedding, dtype=np.float32)
            n1 = float(np.linalg.norm(v1))
            n2 = float(np.linalg.norm(v2))
            denom = max(n1 * n2, 1e-8)
            cosine = float(np.dot(v1, v2) / denom)
            fusion_score = float(1.0 / (1.0 + np.exp(-(cosine - cosine_threshold) * 14.0)))
            agreement = "single"

        quality_ok = (
            primary.quality == "reliable"
            and comparison.quality == "reliable"
            and primary.embedding_norm >= 20.0
            and comparison.embedding_norm >= 20.0
        )
        any_model_passed = any(bool(item.get("passed", False)) for item in model_scores.values()) if model_scores else bool(cosine >= cosine_threshold)
        quality_factor = self._quality_factor(primary, comparison)
        liveness_factor, liveness_flags = self._liveness_factor(primary, comparison)
        age_gap_factor = self._age_gap_factor(primary, comparison)
        detector_factor = float(
            np.clip(
                (primary.detector_confidence + comparison.detector_confidence) / 2.0,
                0.0,
                1.0,
            )
        )
        calibrated_confidence = float(
            np.clip(
                fusion_score * 0.58
                + max(0.0, cosine) * 0.16
                + quality_factor * 0.12
                + liveness_factor * 0.09
                + age_gap_factor * 0.05,
                0.0,
                1.0,
            )
        )
        risk_flags = list(liveness_flags)
        if agreement == "mixed":
            risk_flags.append("mixed_model_agreement")
        if detector_factor < 0.45:
            risk_flags.append("weak_detector_confidence")

        liveness_block = any(flag.startswith("spoof_") for flag in risk_flags)
        verified = bool(
            quality_ok
            and fusion_score >= 0.55
            and any_model_passed
            and not liveness_block
            and calibrated_confidence >= 0.57
        )

        if verified:
            rationale = "fused multi-model match accepted after quality and PAD calibration"
        elif liveness_block:
            rationale = "rejected due to presentation-attack indicators"
        elif not quality_ok:
            rationale = "rejected due to low quality embedding norm"
        elif not any_model_passed:
            rationale = "rejected because no model cleared its acceptance threshold"
        else:
            rationale = "rejected due to low calibrated confidence"

        decision_trace = [
            f"fused_cosine={cosine:.4f}",
            f"fusion_score={fusion_score:.4f}",
            f"quality_factor={quality_factor:.4f}",
            f"liveness_factor={liveness_factor:.4f}",
            f"age_gap_factor={age_gap_factor:.4f}",
            f"detector_factor={detector_factor:.4f}",
            f"calibrated_confidence={calibrated_confidence:.4f}",
        ]

        return PairMatchResult(
            cosine_similarity=cosine,
            verified=verified,
            threshold=cosine_threshold,
            quality_gate_passed=quality_ok,
            rationale=rationale,
            fusion_score=round(fusion_score, 4),
            confidence=round(calibrated_confidence, 4),
            agreement=agreement,
            model_scores=model_scores,
            evidence_weights={k: float(v.get("weight", 0.0)) for k, v in model_scores.items()},
            calibration_features={
                "quality_factor": round(quality_factor, 4),
                "liveness_factor": round(liveness_factor, 4),
                "age_gap_factor": round(age_gap_factor, 4),
                "detector_factor": round(detector_factor, 4),
                "calibrated_confidence": round(calibrated_confidence, 4),
            },
            risk_flags=sorted(set(risk_flags)),
            decision_trace=decision_trace,
        )

    def explain_similarity(
        self,
        image_a_path: str,
        box_a: FaceBox | Dict[str, Any],
        image_b_path: str,
        box_b: FaceBox | Dict[str, Any],
        save_path: str | None = None,
    ) -> Dict[str, Any]:
        if isinstance(box_a, dict):
            box_a = FaceBox.model_validate(box_a)
        if isinstance(box_b, dict):
            box_b = FaceBox.model_validate(box_b)
        return self.gradcam.explain(image_a_path, box_a, image_b_path, box_b, save_path=save_path)

    def to_contract(self, row: Dict[str, Any]) -> Optional[EmbeddingResult]:
        embeddings = row.get("embeddings", {}) or {}
        emb = (
            embeddings.get("adaface")
            or embeddings.get("buffalo_l")
            or row.get("embedding")
        )
        box = row.get("box", {})
        if not emb or not box:
            return None
        return EmbeddingResult(
            embedding=[float(x) for x in emb],
            embeddings={
                str(name): [float(x) for x in vec]
                for name, vec in embeddings.items()
                if vec
            },
            embedding_norm=float(row.get("embedding_norm", 0.0)),
            quality=row.get("quality", {}).get("label", "low_quality_unreliable"),
            detector_confidence=float(row.get("confidence", 0.0)),
            box=FaceBox(
                x=int(box.get("x", 0)),
                y=int(box.get("y", 0)),
                w=int(box.get("w", 0)),
                h=int(box.get("h", 0)),
            ),
            source_path=str(row.get("source_path", "")),
            model_name=str(row.get("model_name", self.model_pack)),
            demographics=row.get("demographics", {}),
            landmarks=row.get("landmarks", {}) or {},
            liveness=row.get("liveness", {}) or {},
        )

    def cleanup(self) -> None:
        clear_cuda_cache()

    @staticmethod
    def _quality_factor(primary: EmbeddingResult, comparison: EmbeddingResult) -> float:
        norms = [float(primary.embedding_norm), float(comparison.embedding_norm)]
        min_norm = min(norms)
        mean_norm = sum(norms) / 2.0
        norm_factor = float(np.clip((min_norm - 18.0) / 12.0, 0.0, 1.0))
        stability_factor = float(np.clip((mean_norm - 20.0) / 10.0, 0.0, 1.0))
        return float(np.clip(norm_factor * 0.65 + stability_factor * 0.35, 0.0, 1.0))

    @staticmethod
    def _liveness_factor(
        primary: EmbeddingResult,
        comparison: EmbeddingResult,
    ) -> Tuple[float, List[str]]:
        flags: List[str] = []
        scores: List[float] = []

        for role, payload in (("primary", primary.liveness), ("comparison", comparison.liveness)):
            state = str((payload or {}).get("signal_state", "unknown"))
            score = float((payload or {}).get("score", 0.5) or 0.5)
            if state == "live":
                scores.append(max(0.8, score))
            elif state == "indeterminate":
                scores.append(max(0.45, min(score, 0.7)))
                flags.append(f"pad_indeterminate_{role}")
            elif state == "spoof":
                scores.append(min(score, 0.15))
                flags.append(f"spoof_detected_{role}")
            else:
                scores.append(0.5)
                flags.append(f"pad_unknown_{role}")

        return float(sum(scores) / max(len(scores), 1)), flags

    @staticmethod
    def _age_gap_factor(primary: EmbeddingResult, comparison: EmbeddingResult) -> float:
        p_age = (primary.demographics or {}).get("estimated_age")
        c_age = (comparison.demographics or {}).get("estimated_age")
        if p_age is None or c_age is None:
            return 0.8
        try:
            gap = abs(float(p_age) - float(c_age))
        except Exception:
            return 0.8
        if gap <= 10.0:
            return 1.0
        if gap <= 20.0:
            return 0.92
        if gap <= 35.0:
            return 0.84
        return 0.76
