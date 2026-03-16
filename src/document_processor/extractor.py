from __future__ import annotations

import os
import tempfile
import zipfile
from pathlib import Path
from typing import Dict, Generator, Optional

from pdf2image import convert_from_path

from src.core.contracts import DocumentResponse, FaceBox
from src.document_processor.analysis import DocumentAnalysisService


class DocumentExtractor:
    """
    Document intake + image extraction + intelligent parsing.
    Keeps temporary outputs in OS temp dir and supports cleanup.
    """

    def __init__(self, temp_dir: Optional[str] = None):
        self.temp_dir = temp_dir or os.path.join(tempfile.gettempdir(), "ca_monk_extracts")
        os.makedirs(self.temp_dir, exist_ok=True)
        self.analysis = DocumentAnalysisService()
        self._ephemeral_paths: list[str] = []

    def _remember(self, path: str) -> str:
        self._ephemeral_paths.append(path)
        return path

    def extract_images(self, file_path: str) -> Generator[str, None, None]:
        if not os.path.exists(file_path):
            return

        ext = Path(file_path).suffix.lower()
        if ext in {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}:
            yield file_path
            return
        if ext == ".pdf":
            yield from self._extract_from_pdf(file_path)
            return
        if ext in {".xlsx", ".xlsm"}:
            yield from self._extract_from_excel(file_path)
            return

    def _extract_from_pdf(self, file_path: str) -> Generator[str, None, None]:
        try:
            images = convert_from_path(file_path, dpi=250)
            base_name = Path(file_path).stem
            for idx, img in enumerate(images):
                out = os.path.join(self.temp_dir, f"{base_name}_page_{idx+1}.jpg")
                img.save(out, "JPEG", quality=95)
                yield self._remember(out)
        except Exception:
            return

    def _extract_from_excel(self, file_path: str) -> Generator[str, None, None]:
        try:
            base_name = Path(file_path).stem
            with zipfile.ZipFile(file_path, "r") as zf:
                media = [m for m in zf.namelist() if m.startswith("xl/media/")]
                for m in media:
                    ext = Path(m).suffix.lower()
                    if ext not in {".png", ".jpg", ".jpeg"}:
                        continue
                    out = os.path.join(self.temp_dir, f"{base_name}_{Path(m).name}")
                    with open(out, "wb") as f:
                        f.write(zf.read(m))
                    yield self._remember(out)
        except Exception:
            return

    def parse_id_document(self, image_path: str, face_box: Optional[FaceBox] = None) -> DocumentResponse:
        return self.analysis.analyze(image_path, face_box)

    def parse_id_document_with_evidence(
        self,
        image_path: str,
        face_box: Optional[FaceBox] = None,
        evidence_dir: Optional[str] = None,
    ) -> DocumentResponse:
        """Parse document with evidence artifact generation (tamper_heatmap.jpg)."""
        return self.analysis.analyze(image_path, face_box, evidence_dir=evidence_dir)

    def cleanup(self) -> None:
        for path in self._ephemeral_paths:
            try:
                if os.path.isfile(path):
                    os.remove(path)
            except Exception:
                pass
        self._ephemeral_paths.clear()
