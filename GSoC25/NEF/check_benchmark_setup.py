#!/usr/bin/env python3
"""
Quick setup verification script for Text2KGBench benchmarking.

Checks if all required files and directories are in place.
"""

import os
import sys
from pathlib import Path

# Colors for output
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
RESET = "\033[0m"

def check_mark(ok: bool) -> str:
    return f"{GREEN}✓{RESET}" if ok else f"{RED}✗{RESET}"

def print_header(text: str):
    print(f"\n{BLUE}{'='*80}{RESET}")
    print(f"{BLUE}{text}{RESET}")
    print(f"{BLUE}{'='*80}{RESET}")

def check_file(path: Path, description: str) -> bool:
    """Check if a file exists."""
    exists = path.exists()
    status = check_mark(exists)
    print(f"  {status} {description}")
    if exists:
        print(f"      {path}")
    else:
        print(f"      {RED}Missing: {path}{RESET}")
    return exists

def check_dir(path: Path, description: str) -> bool:
    """Check if a directory exists."""
    exists = path.exists() and path.is_dir()
    status = check_mark(exists)
    print(f"  {status} {description}")
    if exists:
        print(f"      {path}")
        # Count files if it's a data directory
        if "test" in str(path) or "ground_truth" in str(path):
            files = list(path.glob("*.jsonl"))
            print(f"      {YELLOW}Found {len(files)} test files{RESET}")
    else:
        print(f"      {RED}Missing: {path}{RESET}")
    return exists

def main():
    """Run all checks."""
    script_dir = Path(__file__).parent
    
    print_header("🔍 Text2KGBench Benchmarking Setup Check")
    
    all_ok = True
    
    # Check NEF files
    print(f"\n{YELLOW}NEF Scripts:{RESET}")
    nef_scripts = [
        ("benchmark_nef_text2kg.py", "Main benchmarking script"),
        ("run_text2kg_evaluation.py", "Evaluation helper script"),
        ("nef_text2kg_eval_config.json", "Evaluation config file"),
        ("NEF.py", "NEF pipeline module"),
    ]
    
    for filename, desc in nef_scripts:
        path = script_dir / filename
        if not check_file(path, desc):
            all_ok = False
    
    # Check Text2KGBench structure
    print(f"\n{YELLOW}Text2KGBench Structure:{RESET}")
    text2kg_root = script_dir / "Text2KGBench" / "Text2KGBench-main"
    
    if not check_dir(text2kg_root, "Text2KGBench root directory"):
        all_ok = False
        print(f"\n{RED}⚠️  Text2KGBench not found. Please ensure it's located at:{RESET}")
        print(f"   {text2kg_root}")
    else:
        # Check data directories
        dbpedia_data = text2kg_root / "data" / "dbpedia_webnlg"
        if check_dir(dbpedia_data, "DBpedia data directory"):
            check_dir(dbpedia_data / "test", "Test data directory")
            check_dir(dbpedia_data / "ground_truth", "Ground truth directory")
            check_dir(dbpedia_data / "ontologies", "Ontologies directory")
        
        # Check evaluation script
        eval_dir = text2kg_root / "src" / "evaluation"
        if check_dir(eval_dir, "Evaluation script directory"):
            check_file(eval_dir / "run_eval.py", "Evaluation script")
    
    # Check NEF dependencies
    print(f"\n{YELLOW}NEF Dependencies:{RESET}")
    
    # Check embeddings
    emb_paths = [
        script_dir / "embeddings.npy",
        script_dir.parent / "embeddings.npy",
    ]
    emb_found = any(p.exists() for p in emb_paths)
    status = check_mark(emb_found)
    print(f"  {status} Embeddings file (embeddings.npy)")
    if emb_found:
        found_path = next(p for p in emb_paths if p.exists())
        print(f"      {found_path}")
    else:
        print(f"      {YELLOW}Not found in common locations{RESET}")
        print(f"      {YELLOW}You can specify with --embeddings flag{RESET}")
        all_ok = False
    
    # Check predicates
    pred_paths = [
        script_dir / "predicates.csv",
        script_dir.parent / "predicates.csv",
    ]
    pred_found = any(p.exists() for p in pred_paths)
    status = check_mark(pred_found)
    print(f"  {status} Predicates file (predicates.csv)")
    if pred_found:
        found_path = next(p for p in pred_paths if p.exists())
        print(f"      {found_path}")
    else:
        print(f"      {YELLOW}Not found in common locations{RESET}")
        print(f"      {YELLOW}You can specify with --predicates flag{RESET}")
        all_ok = False
    
    # Check environment variables
    print(f"\n{YELLOW}Environment Variables:{RESET}")
    env_vars = {
        "GEMINI_API_KEY": "Gemini API key",
        "NEF_REDIS_HOST": "Redis host (optional, default: localhost)",
        "NEF_REDIS_PORT": "Redis port (optional, default: 6379)",
    }
    
    for var, desc in env_vars.items():
        value = os.getenv(var)
        if value:
            masked = value[:8] + "..." if len(value) > 8 else value
            status = check_mark(True)
            print(f"  {status} {desc}")
            print(f"      {var}={masked}")
        else:
            if "optional" in desc:
                status = check_mark(True)
                print(f"  {status} {desc} (using default)")
            else:
                status = check_mark(False)
                print(f"  {status} {desc}")
                print(f"      {YELLOW}Set {var} or use --api-key flag{RESET}")
                if var == "GEMINI_API_KEY":
                    all_ok = False
    
    # Check Python packages
    print(f"\n{YELLOW}Python Packages:{RESET}")
    packages = [
        ("google.genai", "google-genai"),
        ("numpy", "numpy"),
        ("pandas", "pandas"),
        ("rdflib", "rdflib"),
        ("redis", "redis"),
        ("nltk", "nltk"),
    ]
    
    for module, package in packages:
        try:
            __import__(module)
            status = check_mark(True)
            print(f"  {status} {package}")
        except ImportError:
            status = check_mark(False)
            print(f"  {status} {package} (install with: pip install {package})")
            if module in ["google.genai", "numpy", "pandas", "rdflib"]:
                all_ok = False
    
    # Final summary
    print_header("📊 Summary")
    
    if all_ok:
        print(f"{GREEN}✅ All critical checks passed!{RESET}")
        print(f"\n{GREEN}You're ready to run benchmarking:{RESET}")
        print(f"   python benchmark_nef_text2kg.py")
    else:
        print(f"{YELLOW}⚠️  Some checks failed. Please fix the issues above.{RESET}")
        print(f"\n{YELLOW}Common fixes:{RESET}")
        print(f"   1. Install missing packages: pip install <package>")
        print(f"   2. Set GEMINI_API_KEY environment variable")
        print(f"   3. Ensure Text2KGBench is in the correct location")
        print(f"   4. Generate embeddings if missing (run Emeddings.py)")
    
    print()
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())

