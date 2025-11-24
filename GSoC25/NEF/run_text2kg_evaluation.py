#!/usr/bin/env python3
"""
Helper script to run Text2KGBench evaluation on NEF results.

This script runs the Text2KGBench evaluation script with the NEF config.
It should be run after benchmark_nef_text2kg.py has generated the results.
"""

import os
import sys
import subprocess
from pathlib import Path

# Paths
SCRIPT_DIR = Path(__file__).parent
TEXT2KG_ROOT = SCRIPT_DIR / "Text2KGBench" / "Text2KGBench-main"
EVAL_DIR = TEXT2KG_ROOT / "src" / "evaluation"
CONFIG_FILE = SCRIPT_DIR / "nef_text2kg_eval_config.json"


def main():
    """Run the Text2KGBench evaluation."""
    
    # Check if evaluation directory exists
    if not EVAL_DIR.exists():
        print(f"❌ ERROR: Text2KGBench evaluation directory not found: {EVAL_DIR}")
        print(f"   Please ensure Text2KGBench is located at: {TEXT2KG_ROOT}")
        return 1
    
    # Check if config file exists
    if not CONFIG_FILE.exists():
        print(f"❌ ERROR: Config file not found: {CONFIG_FILE}")
        print(f"   Please ensure nef_text2kg_eval_config.json exists in: {SCRIPT_DIR}")
        return 1
    
    # Check if run_eval.py exists
    eval_script = EVAL_DIR / "run_eval.py"
    if not eval_script.exists():
        print(f"❌ ERROR: Evaluation script not found: {eval_script}")
        return 1
    
    # Convert paths to relative paths from evaluation directory
    config_rel = os.path.relpath(CONFIG_FILE, EVAL_DIR)
    
    print("="*80)
    print("🔍 Running Text2KGBench Evaluation on NEF Results")
    print("="*80)
    print(f"Evaluation directory: {EVAL_DIR}")
    print(f"Config file: {CONFIG_FILE}")
    print(f"Config (relative): {config_rel}")
    print("="*80)
    print()
    
    # Change to evaluation directory and run
    try:
        os.chdir(EVAL_DIR)
        cmd = [sys.executable, "run_eval.py", "--eval_config_path", config_rel]
        print(f"Running: {' '.join(cmd)}")
        print()
        
        result = subprocess.run(cmd, check=True)
        
        print()
        print("="*80)
        print("✅ Evaluation completed successfully!")
        print("="*80)
        print(f"Results saved to: {SCRIPT_DIR / 'nef_text2kg_results' / 'eval_metrics'}")
        print("="*80)
        
        return 0
        
    except subprocess.CalledProcessError as e:
        print()
        print("="*80)
        print(f"❌ ERROR: Evaluation failed with exit code {e.returncode}")
        print("="*80)
        return e.returncode
    except Exception as e:
        print()
        print("="*80)
        print(f"❌ ERROR: {e}")
        print("="*80)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

