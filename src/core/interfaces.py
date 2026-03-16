
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import numpy as np

class IImageLoader(ABC):
    @abstractmethod
    def load(self, source: Any) -> np.ndarray:
        pass

class IPreProcessor(ABC):
    @abstractmethod
    def process(self, image: np.ndarray) -> np.ndarray:
        pass

class IFaceDetector(ABC):
    @abstractmethod
    def detect(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Returns a list of detected face objects with:
        - box: {x, y, w, h}
        - landmarks: dict
        - confidence: float
        """
        pass

class IFaceEmbedder(ABC):
    @abstractmethod
    def get_embedding(self, face_image: np.ndarray) -> List[float]:
        pass

class ILivenessDetector(ABC):
    @abstractmethod
    def check_liveness(self, face_image: np.ndarray) -> float:
        pass

class IForensicAnalyzer(ABC):
    @abstractmethod
    def analyze(self, image_path: str) -> Dict[str, Any]:
        pass
