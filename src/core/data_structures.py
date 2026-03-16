from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class Document:
    file_path: str
    doc_class: str
    original_filename: Optional[str] = None
    s3_url: Optional[str] = None

@dataclass
class Applicant:
    role: str  # e.g., "Applicant", "CoApplicant1"
    primary_docs: List[Document] = field(default_factory=list)
    comparison_docs: List[Document] = field(default_factory=list)

    def __repr__(self):
        return f"Applicant(role={self.role}, primary={len(self.primary_docs)}, comparison={len(self.comparison_docs)})"
