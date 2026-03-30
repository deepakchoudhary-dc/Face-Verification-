from __future__ import annotations

import argparse
import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.core.contracts import ExpressionTransferRequest
from src.reconstruction.expression_suite import Deep3DExpressionSuiteService


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run DECA-style expression capture, transfer, and animation on top of the local Deep3D stack."
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
    parser.add_argument(
        "--preset",
        default=None,
        help="Optional directed preset such as donor_expression, laughing, frightened, sideview_left, sideview_right.",
    )
    parser.add_argument(
        "--strength",
        type=float,
        default=1.0,
        help="Expression intensity multiplier. Default: 1.0",
    )
    parser.add_argument("--yaw", type=float, default=None, help="Optional target yaw in degrees.")
    parser.add_argument("--pitch", type=float, default=None, help="Optional target pitch in degrees.")
    parser.add_argument("--roll", type=float, default=None, help="Optional target roll in degrees.")
    parser.add_argument(
        "--blink-strength",
        type=float,
        default=0.15,
        help="Blink amplitude for the cinematic animation. Default: 0.15",
    )
    parser.add_argument(
        "--animation-mode",
        default="cinematic",
        help="Animation mode. Use cinematic or linear. Default: cinematic",
    )
    parser.add_argument(
        "--animation-frames",
        type=int,
        default=18,
        help="Base frame count before the return loop is added. Default: 18",
    )
    parser.add_argument(
        "--sideview-angle",
        type=float,
        default=30.0,
        help="Fallback side-view render angle in degrees. Default: 30",
    )
    args = parser.parse_args()

    service = Deep3DExpressionSuiteService()
    try:
        result = service.generate(
            ExpressionTransferRequest(
                source_image_path=args.source,
                expression_image_path=args.expression,
                evidence_save_path=args.save,
                transfer_pose=args.transfer_pose,
                expression_preset=args.preset,
                expression_strength=args.strength,
                target_yaw_deg=args.yaw,
                target_pitch_deg=args.pitch,
                target_roll_deg=args.roll,
                blink_strength=args.blink_strength,
                animation_mode=args.animation_mode,
                animation_frames=args.animation_frames,
                sideview_angle_deg=args.sideview_angle,
            )
        )
        payload = result.model_dump() if hasattr(result, "model_dump") else result
        print(json.dumps(payload, indent=2))
        return 0 if payload.get("generated_image_path") else 1
    finally:
        service.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
