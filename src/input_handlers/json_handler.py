import json
import os
import boto3
from typing import Generator, Dict, List
from urllib.parse import urlparse
from dotenv import load_dotenv

from src.input_handlers.base_handler import BaseInputHandler
from src.core.data_structures import Applicant, Document

load_dotenv()

class JsonInputHandler(BaseInputHandler):
    def __init__(self, json_path: str):
        self.json_path = json_path
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        
        self.s3_client = self._init_s3_client()
        # Create dataset folder structure
        self.base_dir = os.path.dirname(json_path)
        self.dataset_dir = os.path.join(self.base_dir, "dataset")
        os.makedirs(self.dataset_dir, exist_ok=True)

    def _init_s3_client(self):
        return boto3.client(
            's3',
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
            region_name=os.getenv('AWS_REGION', 'ap-south-1') # Defaulting to ap-south-1 as seen in .env
        )

    def _download_file(self, s3_url: str, role: str, category: str) -> str:
        """
        Download file from S3 to organized dataset folder
        Args:
            s3_url: S3 URL of the file
            role: Role name (e.g., 'applicant', 'co-applicant')
            category: 'primary' or 'compare_with'
        """
        parsed = urlparse(s3_url)
        bucket = parsed.netloc
        key = parsed.path.lstrip('/')
        filename = os.path.basename(key)
        
        # Create organized folder structure
        role_folder = role.lower().replace(' ', '-')
        target_dir = os.path.join(self.dataset_dir, role_folder, category)
        os.makedirs(target_dir, exist_ok=True)
        
        local_path = os.path.join(target_dir, filename)

        if not os.path.exists(local_path):
            print(f"Downloading {s3_url} to {local_path}...")
            try:
                self.s3_client.download_file(bucket, key, local_path)
            except Exception as e:
                print(f"Error downloading {s3_url}: {e}")
                return None
        return local_path

    def get_applicants(self) -> Generator[Applicant, None, None]:
        # 1. Parse Comparison Matrix
        # Map role -> {primary: class_name, compare_with: [class_names]}
        role_config = {}
        for matrix in self.data.get('comparison_matrix', []):
            role = matrix.get('role')
            role_config[role] = {
                'primary': matrix.get('primary'),
                'compare_with': set(matrix.get('compare_with', []))
            }

        # 2. Group Documents by Role (Applicant, CoApplicant1, etc.)
        for person_data in self.data.get('applicants', []):
            # Determine the role for this person
            person_key = person_data.get('key', '').lower()
            
            matched_role = None
            if person_key == 'applicant':
                matched_role = "Applicant"
            elif person_key.startswith('co_applicant_'):
                # Handle "co_applicant_1" or "co_applicant_1_name"
                parts = person_key.split('_')
                if len(parts) >= 3 and parts[2].isdigit():
                     matched_role = f"CoApplicant{parts[2]}"
            
            # If we can't match strictly, skip
            if matched_role not in role_config:
                print(f"Warning: Could not map person key '{person_key}' to a configured role in comparison_matrix.")
                continue

            config = role_config[matched_role]
            primary_class = config['primary']
            compare_classes = config['compare_with']

            applicant = Applicant(role=matched_role)

            for doc_data in person_data.get('documents', []):
                doc_class = doc_data.get('document_class')
                s3_path = doc_data.get('file_path')
                
                # Determine category for organized folder structure
                category = None
                if doc_class == primary_class:
                    category = 'primary'
                elif doc_class in compare_classes:
                    category = 'compare_with'
                else:
                    # Skip documents not in primary or compare_with
                    continue
                
                # Download file to organized folder
                local_path = self._download_file(s3_path, matched_role, category)
                if not local_path:
                    continue

                doc = Document(
                    file_path=local_path,
                    doc_class=doc_class,
                    original_filename=doc_data.get('original_filename'),
                    s3_url=s3_path
                )

                if doc_class == primary_class:
                    applicant.primary_docs.append(doc)
                elif doc_class in compare_classes:
                    applicant.comparison_docs.append(doc)
            
            yield applicant
