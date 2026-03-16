from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.benchmarking import BenchmarkHarness


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CA_Monk benchmark manifest.")
    parser.add_argument("manifest", help="Path to benchmark manifest JSON.")
    parser.add_argument(
        "--output-dir",
        default=os.path.join("forensic_output", "benchmark_runs"),
        help="Directory for benchmark JSON/Markdown outputs.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional case limit.")
    args = parser.parse_args()

    payload = BenchmarkHarness().run_manifest(
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        limit=args.limit,
    )
    print(json.dumps(payload.get("metrics", {}), indent=2))


if __name__ == "__main__":
    main()
