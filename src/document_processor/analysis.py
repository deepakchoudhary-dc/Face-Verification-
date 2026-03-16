from __future__ import annotations

import logging
import os
from typing import Optional

from src.core.contracts import DocumentResponse, FaceBox
from src.document_processor.donut_parser import DonutDocumentParser
from src.document_processor.noiseprint import NoisePrintAnalyzer

logger = logging.getLogger("ca_monk.document_analysis")


class DocumentAnalysisService:
    """
    Visual document understanding + forgery checks.
    Generates tamper_heatmap.jpg when evidence_dir is provided.
    """

    def __init__(self) -> None:
        self.donut = DonutDocumentParser()
        self.noiseprint = NoisePrintAnalyzer()

    def analyze(
        self,
        image_path: str,
        face_box: Optional[FaceBox] = None,
        evidence_dir: Optional[str] = None,
    ) -> DocumentResponse:
        warnings: list[str] = []
        fields = {}
        noise = None

        try:
            fields = self.donut.parse_to_json(image_path)
            if "warning" in fields:
                warnings.append(str(fields.get("warning")))
        except Exception as exc:
            warnings.append(f"donut_failed: {exc}")
            fields = {"warning": "donut_failed"}

        try:
            noise = self.noiseprint.analyze(image_path, face_box)
        except Exception as exc:
            warnings.append(f"noiseprint_failed: {exc}")

        # Generate ELA tamper heatmap for evidence card
        if evidence_dir:
            try:
                heatmap_path = os.path.join(evidence_dir, "tamper_heatmap.jpg")
                self.noiseprint.generate_tamper_heatmap(
                    image_path, face_box=face_box, save_path=heatmap_path,
                )
            except Exception as exc:
                warnings.append(f"tamper_heatmap_failed: {exc}")

        return DocumentResponse(fields=fields, noiseprint=noise, warnings=warnings)

