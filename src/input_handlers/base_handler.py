from abc import ABC, abstractmethod
from typing import List, Generator
from src.core.data_structures import Applicant

class BaseInputHandler(ABC):
    @abstractmethod
    def get_applicants(self) -> Generator[Applicant, None, None]:
        """
        Yields Applicant objects containing processed document lists.
        """
        pass
