from __future__ import annotations

import argparse
import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.core.contracts import ExpressionTransferRequest
from src.reconstruction.expression_transfer import Deep3DExpressionTransferService


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run Deep3D-based expression transfer without changing the main verification pipeline."
    )
    parser.add_argument("--source", required=True, help="Path to the identity/source face image.")
    parser.add_argument(
        "--expression",
        required=True,
        help="Path to the image whose expression should be transferred.",
    )
    parser.add_argument(
        "--save",
        default=None,
        help="Optional output path for the composited transfer image.",
    )
    parser.add_argument(
        "--transfer-pose",
        action="store_true",
        help="Also copy the expression image's global head pose.",
    )
    args = parser.parse_args()

    service = Deep3DExpressionTransferService()
    try:
        result = service.generate(
            ExpressionTransferRequest(
                source_image_path=args.source,
                expression_image_path=args.expression,
                evidence_save_path=args.save,
                transfer_pose=args.transfer_pose,
            )
        )
        payload = result.model_dump() if hasattr(result, "model_dump") else result
        print(json.dumps(payload, indent=2))
        return 0 if payload.get("generated_image_path") else 1
    finally:
        service.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
