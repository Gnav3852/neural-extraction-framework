#!/usr/bin/env python3
"""
Universal experiment runner: runs benchmark → evaluation → add_semantic_f1 in order
and stores all results under ExpRes/<name>/ (e.g. ExpRes/ExpOne).

Uses GSoC25/NEF/Dataset for test, ground_truth, and ontologies.

Usage:
  python run_experiment.py --name ExpOne
  python run_experiment.py -n MyRun
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
EXPRES_DIR = SCRIPT_DIR / "ExpRes"


def main():
    parser = argparse.ArgumentParser(
        description="Run full NEF experiment: benchmark → eval → semantic F1.",
        epilog="Example: python run_experiment.py --name ExpOne",
    )
    parser.add_argument(
        "--name", "-n",
        type=str,
        required=True,
        metavar="FOLDER",
        help="Experiment folder name (e.g. ExpOne). Results go to ExpRes/<name>/.",
    )
    parser.add_argument(
        "--skip-benchmark",
        action="store_true",
        help="Skip benchmark step (use existing ont_*_nef_responses.jsonl in results dir)",
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Skip evaluation step",
    )
    parser.add_argument(
        "--skip-semantic",
        action="store_true",
        help="Skip add_semantic_f1 step",
    )
    parser.add_argument(
        "--semantic",
        action="store_true",
        help="Enable semantic matching when generating comparison tables (slower; needs google-genai, numpy, scikit-learn)",
    )
    args = parser.parse_args()

    exp_dir = EXPRES_DIR / args.name
    exp_dir.mkdir(parents=True, exist_ok=True)
    print("=" * 80)
    print(f"📁 Experiment directory: {exp_dir.absolute()}")
    print("=" * 80)

    # 1) Benchmark
    if not args.skip_benchmark:
        print("\n[1/3] Running benchmark_nef_text2kg.py ...")
        ret = subprocess.run(
            [
                sys.executable,
                str(SCRIPT_DIR / "benchmark_nef_text2kg.py"),
                "--output-dir", str(exp_dir),
            ],
            cwd=str(SCRIPT_DIR),
        )
        if ret.returncode != 0:
            print(f"❌ Benchmark failed with exit code {ret.returncode}")
            return ret.returncode
        print("✅ Benchmark done.\n")
    else:
        print("\n[1/3] Skipping benchmark (--skip-benchmark).\n")

    # 2) Evaluation (run_text2kg_evaluation.py --results-dir <exp_dir>)
    if not args.skip_eval:
        print("[2/3] Running run_text2kg_evaluation.py ...")
        env = None
        if args.semantic:
            env = os.environ.copy()
            env["USE_SEMANTIC_MATCHING"] = "true"
        ret = subprocess.run(
            [
                sys.executable,
                str(SCRIPT_DIR / "run_text2kg_evaluation.py"),
                "--results-dir", str(exp_dir),
            ],
            cwd=str(SCRIPT_DIR),
            env=env,
        )
        if ret.returncode != 0:
            print(f"❌ Evaluation failed with exit code {ret.returncode}")
            return ret.returncode
        print("✅ Evaluation done.\n")
    else:
        print("[2/3] Skipping evaluation (--skip-eval).\n")

    # 3) Add semantic F1
    if not args.skip_semantic:
        print("[3/3] Running add_semantic_f1.py ...")
        ret = subprocess.run(
            [
                sys.executable,
                str(SCRIPT_DIR / "add_semantic_f1.py"),
                "--results-dir", str(exp_dir),
            ],
            cwd=str(SCRIPT_DIR),
        )
        if ret.returncode != 0:
            print(f"❌ add_semantic_f1 failed with exit code {ret.returncode}")
            return ret.returncode
        print("✅ Semantic F1 done.\n")
    else:
        print("[3/3] Skipping add_semantic_f1 (--skip-semantic).\n")

    print("=" * 80)
    print("✅ Experiment complete. Results in:")
    print(f"   {exp_dir.absolute()}")
    print("   - ont_*_nef_responses.jsonl")
    print("   - eval_metrics/avg_eval_results.jsonl, ont_*_eval_results.jsonl")
    print("   - Tables/triple_comparison_*.csv")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    sys.exit(main())
