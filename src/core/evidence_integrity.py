"""
FORENSIC EVIDENCE CHAIN INTEGRITY — CA_MONK v6.1
====================================================
SHA-256 hash-chain for tamper-evident evidence packages.

Every artifact (image, mesh, report, dashboard) in an evidence
package is individually hashed.  The manifest itself is then
hash-chained to produce a single root hash.

Verification is deterministic: re-hash any artifact and compare
against the manifest.  If *any* file has been altered, the root
hash will not match.

Compliant with:
    - NIST SP 800-92 (Guide to Computer Security Log Management)
    - Digital Forensics Research Workshop (DFRWS) evidence standards
    - ISO/IEC 27037 (Digital evidence collection)

Author: CA_MONK Forensic Intelligence Unit
Version: 6.1.0
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger("ca_monk.evidence_integrity")


class EvidenceIntegrity:
    """
    Creates and verifies SHA-256 hash manifests for evidence packages.
    """

    MANIFEST_FILENAME = "EVIDENCE_MANIFEST.json"
    HASH_ALGORITHM = "sha256"

    # ------------------------------------------------------------------
    #  Create manifest
    # ------------------------------------------------------------------
    def create_manifest(self, evidence_dir: str) -> Dict[str, Any]:
        """
        Walk *evidence_dir*, hash every file, produce a chained manifest.

        Returns the manifest dict (also written to EVIDENCE_MANIFEST.json).
        """
        if not os.path.isdir(evidence_dir):
            logger.warning("Evidence directory not found: %s", evidence_dir)
            return {"error": "directory_not_found"}

        entries: List[Dict[str, str]] = []
        chain_hash = hashlib.sha256(b"CA_MONK_EVIDENCE_CHAIN_v6.1")

        for root, _dirs, files in os.walk(evidence_dir):
            for fname in sorted(files):
                if fname == self.MANIFEST_FILENAME:
                    continue
                fpath = os.path.join(root, fname)
                file_hash = self._hash_file(fpath)
                rel_path = os.path.relpath(fpath, evidence_dir)
                size = os.path.getsize(fpath)

                entries.append({
                    "file": rel_path.replace("\\", "/"),
                    "sha256": file_hash,
                    "size_bytes": size,
                })

                # Chain: H(prev_chain || file_hash)
                chain_hash.update(file_hash.encode("utf-8"))

        root_hash = chain_hash.hexdigest()

        manifest: Dict[str, Any] = {
            "ca_monk_version": "6.1.0",
            "manifest_version": "1.0",
            "algorithm": self.HASH_ALGORITHM,
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "evidence_directory": os.path.basename(evidence_dir),
            "total_files": len(entries),
            "entries": entries,
            "chain_root_hash": root_hash,
        }

        # Write manifest
        manifest_path = os.path.join(evidence_dir, self.MANIFEST_FILENAME)
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)

        logger.info(
            "Evidence manifest: %d files, root_hash=%s → %s",
            len(entries), root_hash[:16], manifest_path,
        )
        return manifest

    # ------------------------------------------------------------------
    #  Verify manifest
    # ------------------------------------------------------------------
    def verify_manifest(self, evidence_dir: str) -> Dict[str, Any]:
        """
        Re-hash every file and compare against the stored manifest.

        Returns dict with 'valid', 'tampered_files', 'missing_files'.
        """
        manifest_path = os.path.join(evidence_dir, self.MANIFEST_FILENAME)
        if not os.path.isfile(manifest_path):
            return {"valid": False, "error": "manifest_not_found"}

        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        entries = manifest.get("entries", [])
        stored_root = manifest.get("chain_root_hash", "")

        tampered: List[str] = []
        missing: List[str] = []
        chain_hash = hashlib.sha256(b"CA_MONK_EVIDENCE_CHAIN_v6.1")

        for entry in entries:
            rel = entry["file"]
            expected = entry["sha256"]
            fpath = os.path.join(evidence_dir, rel)

            if not os.path.isfile(fpath):
                missing.append(rel)
                chain_hash.update(expected.encode("utf-8"))
                continue

            actual = self._hash_file(fpath)
            chain_hash.update(actual.encode("utf-8"))

            if actual != expected:
                tampered.append(rel)

        computed_root = chain_hash.hexdigest()
        root_valid = (computed_root == stored_root) and not tampered and not missing

        return {
            "valid": root_valid,
            "stored_root_hash": stored_root,
            "computed_root_hash": computed_root,
            "total_files": len(entries),
            "tampered_files": tampered,
            "missing_files": missing,
            "tampered_count": len(tampered),
            "missing_count": len(missing),
        }

    # ------------------------------------------------------------------
    #  File hashing
    # ------------------------------------------------------------------
    @staticmethod
    def _hash_file(path: str, block_size: int = 65536) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            while True:
                chunk = f.read(block_size)
                if not chunk:
                    break
                h.update(chunk)
        return h.hexdigest()
