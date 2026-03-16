
import os
from typing import List, Dict

class AppConfig:
    """
    Centralized configuration for the application.
    Loads from environment variables or defaults.
    """
    
    # Face Detection Settings
    DETECTOR_BACKEND = os.getenv("DETECTOR_BACKEND", "retinaface")
    MIN_FACE_CONFIDENCE = float(os.getenv("MIN_FACE_CONFIDENCE", "0.90"))
    MIN_FACE_SIZE = int(os.getenv("MIN_FACE_SIZE", "30"))
    MAX_FACE_AREA_RATIO = float(os.getenv("MAX_FACE_AREA_RATIO", "0.85"))
    
    # Recognition Models - "Council of Models"
    # We use multiple models for ensemble decision making
    RECOGNITION_MODELS: List[str] = [
        "ArcFace", 
        "Facenet512", 
        "GhostFaceNet"
    ]
    
    # Thresholds
    # Each model has different distance metrics (usually Cosine or Euclidean)
    # These are illustrative defaults for Cosine distance
    MATCH_THRESHOLDS: Dict[str, float] = {
        "ArcFace": 0.68,
        "Facenet512": 0.40,
        "GhostFaceNet": 0.65,
        "default": 0.60
    }
    
    # Preprocessing
    ENABLE_ENHANCEMENT = True
    ENABLE_ROTATION_CHECK = True
    ROTATION_ANGLES = [0, 90, 180, 270] # Reduced set for speed, expand if needed
    
    # Forensics
    ENABLE_ELA = True
    ENABLE_SPECTRAL = True
    DEEPFAKE_THRESHOLD = 0.60
    
    # Similarity Strategy
    CONSENSUS_STRATEGY = "strict_majority" # simple_majority, strict_majority, unanimous
