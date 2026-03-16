from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Union, get_args, get_origin

try:
    from pydantic import BaseModel, Field
except Exception:
    # Lightweight fallback so code can run even before dependencies are installed.
    class _FieldSpec:
        def __init__(self, default: Any = None, default_factory: Any = None):
            self.default = default
            self.default_factory = default_factory

    def Field(default: Any = None, default_factory: Any = None):
        return _FieldSpec(default=default, default_factory=default_factory)

    def _coerce(value: Any, annotation: Any) -> Any:
        if value is None:
            return None
        origin = get_origin(annotation)
        args = get_args(annotation)

        if origin in (list, List):
            item_t = args[0] if args else Any
            return [_coerce(v, item_t) for v in value]
        if origin in (dict, Dict):
            value_t = args[1] if len(args) > 1 else Any
            return {k: _coerce(v, value_t) for k, v in value.items()}
        if origin is Union:
            non_none = [a for a in args if a is not type(None)]
            target = non_none[0] if non_none else Any
            return _coerce(value, target)
        if str(origin).endswith("Literal"):
            return value
        if isinstance(annotation, type) and issubclass(annotation, BaseModel):
            return annotation.model_validate(value)
        return value

    class BaseModel:
        __field_specs__: Dict[str, _FieldSpec] = {}

        def __init_subclass__(cls) -> None:
            specs: Dict[str, _FieldSpec] = {}
            for name, val in cls.__dict__.items():
                if isinstance(val, _FieldSpec):
                    specs[name] = val
            cls.__field_specs__ = specs

        def __init__(self, **data: Any) -> None:
            annotations = getattr(type(self), "__annotations__", {})
            for name, annotation in annotations.items():
                if name in data:
                    raw = data[name]
                elif name in self.__field_specs__:
                    spec = self.__field_specs__[name]
                    raw = spec.default_factory() if spec.default_factory is not None else spec.default
                elif hasattr(type(self), name):
                    raw = getattr(type(self), name)
                else:
                    raw = None
                setattr(self, name, _coerce(raw, annotation))

        @classmethod
        def model_validate(cls, obj: Any):
            if isinstance(obj, cls):
                return obj
            if isinstance(obj, dict):
                return cls(**obj)
            raise TypeError(f"Cannot validate object into {cls.__name__}: {type(obj)}")

        def model_dump(self) -> Dict[str, Any]:
            def dump(v: Any) -> Any:
                if isinstance(v, BaseModel):
                    return v.model_dump()
                if isinstance(v, list):
                    return [dump(i) for i in v]
                if isinstance(v, dict):
                    return {k: dump(i) for k, i in v.items()}
                return v

            out: Dict[str, Any] = {}
            annotations = getattr(type(self), "__annotations__", {})
            for name in annotations:
                out[name] = dump(getattr(self, name))
            return out


class FaceBox(BaseModel):
    x: int
    y: int
    w: int
    h: int


class EmbeddingResult(BaseModel):
    embedding: List[float]
    embeddings: Dict[str, List[float]] = Field(default_factory=dict)
    embedding_norm: float
    quality: Literal["reliable", "low_quality_unreliable", "unidentifiable_noise"]
    detector_confidence: float = 0.0
    box: FaceBox
    source_path: str
    model_name: str = "buffalo_l"
    demographics: Dict[str, Any] = Field(default_factory=dict)
    landmarks: Dict[str, Any] = Field(default_factory=dict)
    liveness: Dict[str, Any] = Field(default_factory=dict)


class PairMatchRequest(BaseModel):
    primary: EmbeddingResult
    comparison: EmbeddingResult


class PairMatchResult(BaseModel):
    cosine_similarity: float
    verified: bool
    threshold: float
    quality_gate_passed: bool
    rationale: str
    fusion_score: float = 0.0
    confidence: float = 0.0
    agreement: str = "none"
    model_scores: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    evidence_weights: Dict[str, float] = Field(default_factory=dict)
    calibration_features: Dict[str, Any] = Field(default_factory=dict)
    risk_flags: List[str] = Field(default_factory=list)
    decision_trace: List[str] = Field(default_factory=list)


class BiometricsRequest(BaseModel):
    image_path: str
    calibrate_from_dir: Optional[str] = None


class BiometricsResponse(BaseModel):
    faces: List[EmbeddingResult] = Field(default_factory=list)
    quality_threshold: float
    warnings: List[str] = Field(default_factory=list)


class ForensicsRequest(BaseModel):
    image_path: str
    video_path: Optional[str] = None


class FrequencyResult(BaseModel):
    deepfake_probability: float
    deepfake_suspected: bool
    model_name: str = "f3net_lite_dct"


class RPPGResult(BaseModel):
    is_live: bool
    bpm: Optional[float] = None
    confidence: float = 0.0
    method: str = "POS"
    signal_state: str = "unknown"
    details: Dict[str, Any] = Field(default_factory=dict)


class ForensicsResponse(BaseModel):
    frequency: FrequencyResult
    rppg: RPPGResult
    warnings: List[str] = Field(default_factory=list)


class DocumentRequest(BaseModel):
    image_path: str
    face_box: Optional[FaceBox] = None


class NoisePrintResult(BaseModel):
    face_noise_variance: float
    background_noise_variance: float
    variance_discrepancy: float
    suspected_splice: bool


class DocumentResponse(BaseModel):
    fields: Dict[str, Any] = Field(default_factory=dict)
    noiseprint: Optional[NoisePrintResult] = None
    warnings: List[str] = Field(default_factory=list)


class ReconstructionRequest(BaseModel):
    image_path: str
    face_embedding: Optional[List[float]] = None
    mode: Literal["deocclusion", "age_progression"] = "deocclusion"
    prompt: Optional[str] = None
    evidence_save_path: Optional[str] = None
    reconstruction_guidance: Optional[str] = None
    estimated_age: Optional[int] = None
    sex: Optional[str] = None
    age_context: Dict[str, Any] = Field(default_factory=dict)


class ReconstructionResponse(BaseModel):
    generated_image_path: Optional[str] = None
    warnings: List[str] = Field(default_factory=list)
    forensic_3d: Dict[str, Any] = Field(default_factory=dict)
    age_simulation: Dict[str, Any] = Field(default_factory=dict)


class ReportRequest(BaseModel):
    applicant_id: str
    biometrics: Dict[str, Any]
    forensics: Dict[str, Any]
    document: Dict[str, Any]
    reconstruction: Dict[str, Any]
    primary_image_study: Dict[str, Any] = Field(default_factory=dict)
    advanced_biometrics: Dict[str, Any] = Field(default_factory=dict)


class ReportResponse(BaseModel):
    summary: str
    verdict: Literal[
        "Conclusive Match",
        "Inconclusive",
        "Fraud Attempt",
        "CLEARED",
        "FLAGGED",
        "verified",
        "review_required",
        "rejected",
    ]
    confidence: float
    reasoning_steps: List[str] = Field(default_factory=list)
