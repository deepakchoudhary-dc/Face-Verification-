import os
from typing import Generator
from src.input_handlers.base_handler import BaseInputHandler
from src.core.data_structures import Applicant, Document

class FolderInputHandler(BaseInputHandler):
    def __init__(self, root_dir: str):
        self.root_dir = root_dir

    def get_applicants(self) -> Generator[Applicant, None, None]:
        # Structure: root/applicant/primary/file.ext
        #            root/applicant/compare_with/file.ext
        #            root/co-applicant/primary/...
        
        # Iterate over top-level directories (roles)
        if not os.path.exists(self.root_dir):
            print(f"Error: Input directory {self.root_dir} does not exist.")
            return

        for role_name in os.listdir(self.root_dir):
            role_path = os.path.join(self.root_dir, role_name)
            if not os.path.isdir(role_path):
                continue

            # Normalize role name for consistency if needed, or keep folder name
            # User said: "applicant and co-applicant subfolders"
            # We can just use the folder name as the role.
            
            applicant = Applicant(role=role_name)
            
            # Check for 'primary' and 'compare_with' subfolders
            primary_dir = os.path.join(role_path, 'primary')
            compare_dir = os.path.join(role_path, 'compare_with')

            # Process Primary Docs
            if os.path.exists(primary_dir):
                for fname in os.listdir(primary_dir):
                    fpath = os.path.join(primary_dir, fname)
                    if os.path.isfile(fpath):
                        applicant.primary_docs.append(Document(
                            file_path=fpath,
                            doc_class='primary', # Generic class for folder input
                            original_filename=fname
                        ))

            # Process Comparison Docs
            if os.path.exists(compare_dir):
                for fname in os.listdir(compare_dir):
                    fpath = os.path.join(compare_dir, fname)
                    if os.path.isfile(fpath):
                        applicant.comparison_docs.append(Document(
                            file_path=fpath,
                            doc_class='compare_with', # Generic class
                            original_filename=fname
                        ))

            yield applicant
