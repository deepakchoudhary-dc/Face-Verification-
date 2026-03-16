"""
CA_MONK v4 — Standalone Integration Test Runner
Processes the test_data/applicant folder through the full pipeline.
"""

import json
import logging
import sys
import os

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("ca_monk.test")


def main():
    from src.core.data_structures import Applicant, Document
    from src.core.engine import VerificationEngine
    from src.core.serialization import to_builtin

    primary_path = os.path.join("test_data", "applicant", "primary", "image.png")
    comparison_path = os.path.join("test_data", "applicant", "compare_with", "image.png")

    if not os.path.exists(primary_path):
        logger.error("Primary image not found: %s", primary_path)
        sys.exit(1)
    if not os.path.exists(comparison_path):
        logger.error("Comparison image not found: %s", comparison_path)
        sys.exit(1)

    applicant = Applicant(
        role="test_subject",
        primary_docs=[Document(file_path=primary_path, doc_class="photo")],
        comparison_docs=[Document(file_path=comparison_path, doc_class="photo")],
    )

    logger.info("=" * 70)
    logger.info("CA_MONK v4 — INTEGRATION TEST")
    logger.info("Primary:    %s", primary_path)
    logger.info("Comparison: %s", comparison_path)
    logger.info("=" * 70)

    engine = VerificationEngine()
    result = to_builtin(engine.process_applicant(applicant))

    # Write output
    out_path = "output.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=str)

    logger.info("=" * 70)
    logger.info("RESULT: is_match=%s, confidence=%.2f%%", result.get("is_match"), result.get("confidence", 0))
    logger.info("Output written to %s", out_path)

    # Check for dashboard
    for comp in result.get("comparisons", []):
        dash = comp.get("dashboard")
        if dash:
            logger.info("Dashboard saved: %s", dash)
    casefile = result.get("interactive_casefile", {}) or {}
    if casefile.get("html_path"):
        logger.info("Interactive casefile: %s", casefile["html_path"])

    logger.info("=" * 70)
    engine.cleanup()


if __name__ == "__main__":
    main()
