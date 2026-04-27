#!/usr/bin/env python3
"""
Universal experiment runner: runs benchmark → evaluation → add_semantic_f1 in order
and stores all results under ExpRes/<name>/ (e.g. ExpRes/ExpOne).

Uses GSoC25/NEF/Dataset for test, ground_truth, and ontologies.

Usage:
  # Default (Gemini reasoner)
  python run_experiment.py --name ExpOne

  # OpenRouter (e.g. Qwen 2.5 72B Instruct)
  export OPENROUTER_API_KEY=your_key
  python run_experiment.py -n Qwen --reasoner openrouter --reasoner-model qwen/qwen-2.5-72b-instruct

  # OpenAI direct (e.g. GPT-4o-mini)
  export OPENAI_API_KEY=your_key
  python run_experiment.py -n OpenAI_4omini --reasoner openai --reasoner-model gpt-4o-mini

  # 6-shot ICL with GPT-4o
  export OPENAI_API_KEY=your_key
  python run_experiment.py -n OpenAI_4o_6shot --reasoner openai --reasoner-model gpt-4o --shots 6 --semantic

  # 6-shot + surface-form fallback (Lever 5: sentence-verbatim output when
  # Redis grounding picks a zero-overlap URI; pairs well with the AWH-style
  # redirect bug fix in NEF.RedisEntityLinking.lookup)
  python run_experiment.py -n OpenAI_4o_6shot_sf --reasoner openai --reasoner-model gpt-4o \\
      --shots 6 --surface-fallback --semantic
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

    # Forwarded to benchmark_nef_text2kg.py
    parser.add_argument(
        "--reasoner",
        choices=["gemini", "openrouter", "openai"],
        default=None,
        help="Reasoning backend to forward to the benchmark step (gemini default, openrouter, or openai).",
    )
    parser.add_argument(
        "--reasoner-model",
        type=str,
        default=None,
        help="Model id for --reasoner=openrouter or --reasoner=openai (e.g. qwen/qwen-2.5-72b-instruct, gpt-4o-mini).",
    )
    parser.add_argument(
        "--openrouter-api-key",
        type=str,
        default=None,
        help="OpenRouter API key (or set OPENROUTER_API_KEY). Required when --reasoner=openrouter.",
    )
    parser.add_argument(
        "--openai-api-key",
        type=str,
        default=None,
        help="OpenAI API key (or set OPENAI_API_KEY). Required when --reasoner=openai.",
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default=None,
        help="Gemini model used when --reasoner=gemini (e.g. gemini-2.5-flash).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Sampling temperature forwarded to the benchmark.",
    )
    parser.add_argument(
        "--shots",
        type=int,
        default=None,
        help="Number of in-context exemplars to inject during extraction "
             "(0 = zero-shot). Forwarded to benchmark_nef_text2kg.py.",
    )
    parser.add_argument(
        "--train-dir",
        type=str,
        default=None,
        help="Override directory holding ont_<id>_train.jsonl files (forwarded).",
    )
    parser.add_argument(
        "--similars-dir",
        type=str,
        default=None,
        help="Override directory holding <id>_test_train_similarity.json files (forwarded).",
    )
    parser.add_argument(
        "--surface-fallback",
        action="store_true",
        help="Lever 5: emit the original LLM mention text when the resolved "
             "URI shares zero tokens with the mention (forwarded to benchmark).",
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
        bench_cmd = [
            sys.executable,
            str(SCRIPT_DIR / "benchmark_nef_text2kg.py"),
            "--output-dir", str(exp_dir),
        ]
        if args.reasoner:
            bench_cmd += ["--reasoner", args.reasoner]
        if args.reasoner_model:
            bench_cmd += ["--reasoner-model", args.reasoner_model]
        if args.openrouter_api_key:
            bench_cmd += ["--openrouter-api-key", args.openrouter_api_key]
        if args.openai_api_key:
            bench_cmd += ["--openai-api-key", args.openai_api_key]
        if args.llm_model:
            bench_cmd += ["--llm-model", args.llm_model]
        if args.temperature is not None:
            bench_cmd += ["--temperature", str(args.temperature)]
        if args.shots is not None:
            bench_cmd += ["--shots", str(args.shots)]
        if args.train_dir:
            bench_cmd += ["--train-dir", args.train_dir]
        if args.similars_dir:
            bench_cmd += ["--similars-dir", args.similars_dir]
        if args.surface_fallback:
            bench_cmd += ["--surface-fallback"]
        ret = subprocess.run(
            bench_cmd,
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
