#!/usr/bin/env python3
"""
Helper script to run Text2KGBench evaluation on NEF results.

This script runs the Text2KGBench evaluation script with the NEF config.
It should be run after benchmark_nef_text2kg.py has generated the results.

Supports easy testing with files in the Eval/ folder.
"""

import os
import sys
import subprocess
import json
import csv
import re
import tempfile
from pathlib import Path
from typing import List, Dict, Tuple, Optional

# Optional semantic matching imports
try:
    from google import genai
    from google.genai import types
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity
    SEMANTIC_AVAILABLE = True
except ImportError:
    SEMANTIC_AVAILABLE = False

# Paths
SCRIPT_DIR = Path(__file__).parent
TEXT2KG_ROOT = SCRIPT_DIR / "Text2KGBench" / "Text2KGBench-main"
EVAL_DIR = TEXT2KG_ROOT / "src" / "evaluation"
CONFIG_FILE = SCRIPT_DIR / "nef_text2kg_eval_config.json"
EVAL_FOLDER = SCRIPT_DIR / "Eval"


def create_eval_config():
    """Create a temporary config file for Eval folder testing."""
    # Find all ontology IDs from files in Eval folder
    eval_files = list(EVAL_FOLDER.glob("ont_*_nef_responses.jsonl"))
    if not eval_files:
        return None
    
    # Extract ontology IDs (e.g., "7_company" from "ont_7_company_nef_responses.jsonl")
    onto_list = []
    for file in eval_files:
        # Extract pattern: ont_{onto_id}_nef_responses.jsonl
        parts = file.stem.split("_")
        if len(parts) >= 3 and parts[0] == "ont":
            onto_id = "_".join(parts[1:-2])  # Everything between "ont" and "nef"
            onto_list.append(onto_id)
    
    if not onto_list:
        return None
    
    # Create config for Eval folder
    config = {
        "onto_list": sorted(set(onto_list)),
        "path_patterns": {
            "sys": os.path.relpath(EVAL_FOLDER / "ont_$$onto$$_nef_responses.jsonl", EVAL_DIR),
            "gt": os.path.relpath(EVAL_FOLDER / "ont_$$onto$$_ground_truth.jsonl", EVAL_DIR),
            "onto": "../../data/dbpedia_webnlg/ontologies/$$onto$$_ontology.json",
            "output": os.path.relpath(SCRIPT_DIR / "Eval" / "eval_metrics" / "ont_$$onto$$_eval_results.jsonl", EVAL_DIR)
        },
        "avg_out_file": os.path.relpath(SCRIPT_DIR / "Eval" / "eval_metrics" / "avg_eval_results.jsonl", EVAL_DIR)
    }
    
    # Create temporary config file
    temp_config = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
    json.dump(config, temp_config, indent=2)
    temp_config.close()
    
    return temp_config.name


def strip_quotes(s: str) -> str:
    """Strip quotes from string (handles escaped and regular quotes)."""
    s = s.strip()
    if s.startswith('\\"') and s.endswith('\\"'):
        s = s[2:-2]
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        s = s[1:-1]
    return s.strip()


def normalize_triple(sub: str, rel: str, obj: str) -> str:
    """Normalize triple for exact matching."""
    sub = strip_quotes(sub)
    rel = strip_quotes(rel)
    obj = strip_quotes(obj)
    # Remove spaces, underscores, lowercase
    sub = re.sub(r"(_|\s+)", '', sub).lower()
    rel = re.sub(r"(_|\s+)", '', rel).lower()
    obj = re.sub(r"(_|\s+)", '', obj).lower()
    return f"{sub}|{rel}|{obj}"


def format_triple(sub: str, rel: str, obj: str) -> str:
    """Format triple for display."""
    sub = strip_quotes(sub)
    rel = strip_quotes(rel)
    obj = strip_quotes(obj)
    return f"({sub}, {rel}, {obj})"


def load_triples_from_jsonl(file_path: Path) -> Dict[str, List[Tuple[str, str, str]]]:
    """Load triples from JSONL file."""
    triples_dict = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            if "triples" not in data:
                continue
            sent_id = data.get("id", "")
            triples = data["triples"]
            formatted = []
            for t in triples:
                if isinstance(t, list) and len(t) == 3:
                    formatted.append((t[0], t[1], t[2]))
                elif isinstance(t, dict):
                    formatted.append((t.get("sub", ""), t.get("rel", ""), t.get("obj", "")))
            if formatted:
                triples_dict[sent_id] = formatted
    return triples_dict


def compute_triple_embeddings(triple_strings: List[str]) -> Dict[str, np.ndarray]:
    """Embed all triple strings using Gemini (adapted from provided code)."""
    if not SEMANTIC_AVAILABLE or not triple_strings:
        return {}
    try:
        client = genai.Client()
        embeddings = {}
        print(f"   🔍 Computing embeddings for {len(triple_strings)} triples...")
        for i, triple in enumerate(triple_strings):
            if (i + 1) % 20 == 0:
                print(f"      Embedded {i+1}/{len(triple_strings)}...")
            resp = client.models.embed_content(
                model="models/embedding-001",
                contents=triple,
                config=types.EmbedContentConfig(task_type="SEMANTIC_SIMILARITY")
            )
            emb = resp.embedding.values if hasattr(resp, 'embedding') else resp.embeddings[0].values
            embeddings[triple] = np.array(emb)
        return embeddings
    except Exception as e:
        print(f"   ⚠️  Error computing embeddings: {e}")
        return {}


def find_semantic_match(triple_str: str, candidates: List[str], embeddings: Dict[str, np.ndarray], 
                       threshold: float = 0.94) -> Tuple[Optional[str], float]:
    """Find semantically similar match (adapted from provided code)."""
    if triple_str not in embeddings:
        return None, 0.0
    best_match, best_sim = None, 0.0
    for candidate in candidates:
        if candidate not in embeddings:
            continue
        sim = float(cosine_similarity([embeddings[triple_str]], [embeddings[candidate]])[0][0])
        if sim >= threshold and sim > best_sim:
            best_match, best_sim = candidate, sim
    return best_match, best_sim


def generate_comparison_table(
    gt_file: Path, pred_file: Path, output_csv: Path,
    max_rows: Optional[int] = None, use_semantic: bool = False, semantic_threshold: float = 0.94
) -> Dict[str, int]:
    """Generate side-by-side comparison table (CSV only) with optional semantic matching. Returns match counts."""
    print(f"\n📊 Generating comparison table...")
    gt_triples = load_triples_from_jsonl(gt_file)
    pred_triples = load_triples_from_jsonl(pred_file)
    all_sent_ids = sorted(set(list(gt_triples.keys()) + list(pred_triples.keys())))
    
    # Collect all unique triple strings for semantic matching
    all_triple_strings = set()
    if use_semantic:
        for sent_id in all_sent_ids:
            for t in gt_triples.get(sent_id, []):
                all_triple_strings.add(format_triple(*t))
            for t in pred_triples.get(sent_id, []):
                all_triple_strings.add(format_triple(*t))
        embeddings = compute_triple_embeddings(list(all_triple_strings))
    else:
        embeddings = {}
    
    comparison_rows = []
    total_exact, total_semantic, total_mismatch = 0, 0, 0
    row_count = 0
    
    for sent_id in all_sent_ids:
        if max_rows is not None and row_count >= max_rows:
            break
        gt_list = gt_triples.get(sent_id, [])
        pred_list = pred_triples.get(sent_id, [])
        gt_norm = {normalize_triple(*t): t for t in gt_list}
        pred_norm = {normalize_triple(*t): t for t in pred_list}
        exact = set(gt_norm.keys()) & set(pred_norm.keys())
        gt_unmatched = set(gt_norm.keys()) - exact
        pred_unmatched = set(pred_norm.keys()) - exact
        
        # Semantic matching for unmatched triples
        semantic_matches = {}
        if use_semantic and embeddings:
            for gt_n in gt_unmatched:
                gt_str = format_triple(*gt_norm[gt_n])
                candidates = [format_triple(*pred_norm[p]) for p in pred_unmatched]
                match, sim = find_semantic_match(gt_str, candidates, embeddings, semantic_threshold)
                if match:
                    for p_n in pred_unmatched:
                        if format_triple(*pred_norm[p_n]) == match:
                            semantic_matches[gt_n] = (p_n, sim)
                            break
        
        # Add exact matches
        for norm_key in exact:
            if max_rows is not None and row_count >= max_rows:
                break
            gt_t, pred_t = gt_norm[norm_key], pred_norm[norm_key]
            comparison_rows.append(_make_row(sent_id, gt_t, pred_t, "EXACT", "1.00"))
            total_exact += 1
            row_count += 1
        
        # Add semantic matches
        for gt_n, (pred_n, sim) in semantic_matches.items():
            if max_rows is not None and row_count >= max_rows:
                break
            comparison_rows.append(_make_row(sent_id, gt_norm[gt_n], pred_norm[pred_n], "SEMANTIC", f"{sim:.2f}"))
            total_semantic += 1
            row_count += 1
            gt_unmatched.discard(gt_n)
            pred_unmatched.discard(pred_n)
        
        # Add unmatched expected (false negatives)
        for gt_n in gt_unmatched:
            if max_rows is not None and row_count >= max_rows:
                break
            comparison_rows.append(_make_row(sent_id, gt_norm[gt_n], None, "MISSING", ""))
            total_mismatch += 1
            row_count += 1
        
        # Add unmatched predicted (false positives)
        for pred_n in pred_unmatched:
            if max_rows is not None and row_count >= max_rows:
                break
            comparison_rows.append(_make_row(sent_id, None, pred_norm[pred_n], "EXTRA", ""))
            total_mismatch += 1
            row_count += 1
    
    # Write CSV
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["sent_id", "match_type", "similarity", "expected", "expected_sub", "expected_rel", 
                  "expected_obj", "predicted", "predicted_sub", "predicted_rel", "predicted_obj"]
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        csv.DictWriter(f, fieldnames=fieldnames).writeheader()
        csv.DictWriter(f, fieldnames=fieldnames).writerows(comparison_rows)
    
    print(f"   ✅ Generated: {total_exact} exact, {total_semantic} semantic, {total_mismatch} mismatches")
    
    return {
        "exact": total_exact,
        "semantic": total_semantic,
        "mismatch": total_mismatch,
        "total": len(comparison_rows)
    }


def _make_row(sent_id: str, gt_t: Optional[Tuple], pred_t: Optional[Tuple], 
              match_type: str, similarity: str) -> Dict:
    """Helper to create a comparison row."""
    row = {"sent_id": sent_id, "match_type": match_type, "similarity": similarity}
    if gt_t:
        row.update({
            "expected": format_triple(*gt_t),
            "expected_sub": strip_quotes(gt_t[0]),
            "expected_rel": strip_quotes(gt_t[1]),
            "expected_obj": strip_quotes(gt_t[2])
        })
    else:
        row.update({"expected": "", "expected_sub": "", "expected_rel": "", "expected_obj": ""})
    if pred_t:
        row.update({
            "predicted": format_triple(*pred_t),
            "predicted_sub": strip_quotes(pred_t[0]),
            "predicted_rel": strip_quotes(pred_t[1]),
            "predicted_obj": strip_quotes(pred_t[2])
        })
    else:
        row.update({"predicted": "", "predicted_sub": "", "predicted_rel": "", "predicted_obj": ""})
    return row


def update_avg_eval_with_match_counts(avg_eval_file: Path, match_counts: Dict[str, Dict[str, int]]):
    """Update avg_eval_results.jsonl with match counts from comparison tables."""
    if not avg_eval_file.exists():
        print(f"   ⚠️  avg_eval_results.jsonl not found: {avg_eval_file}")
        return
    
    # Read existing entries
    entries = []
    with open(avg_eval_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line.strip()))
    
    # Update entries with match counts
    updated = False
    for entry in entries:
        onto_id = entry.get("onto", "")
        if onto_id and onto_id in match_counts:
            counts = match_counts[onto_id]
            entry["exact_matches"] = counts.get("exact", 0)
            entry["semantic_matches"] = counts.get("semantic", 0)
            entry["mismatches"] = counts.get("mismatch", 0)
            entry["total_comparison_rows"] = counts.get("total", 0)
            updated = True
    
    # Write back if updated
    if updated:
        with open(avg_eval_file, 'w', encoding='utf-8') as f:
            for entry in entries:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        print(f"   ✅ Updated {avg_eval_file} with match counts")
    else:
        print(f"   ⚠️  No matching ontology entries found in {avg_eval_file}")


def generate_comparison_tables_for_ontologies(config_data: dict, base_dir: Path, 
                                             use_eval_folder: bool, max_rows: Optional[int] = None,
                                             use_semantic: bool = False, semantic_threshold: float = 0.94) -> Dict[str, Dict[str, int]]:
    """Generate comparison tables for all ontologies in the config. Returns match counts."""
    onto_list = config_data.get("onto_list", [])
    path_patterns = config_data.get("path_patterns", {})
    match_counts = {}
    
    for onto in onto_list:
        onto_id = onto if isinstance(onto, str) else onto.get("id", "")
        if use_eval_folder:
            gt_file = EVAL_FOLDER / f"ont_{onto_id}_ground_truth.jsonl"
            pred_file = EVAL_FOLDER / f"ont_{onto_id}_nef_responses.jsonl"
            output_dir = EVAL_FOLDER
        else:
            gt_file = (EVAL_DIR / path_patterns.get("gt", "").replace("$$onto$$", onto_id)).resolve()
            pred_file = (EVAL_DIR / path_patterns.get("sys", "").replace("$$onto$$", onto_id)).resolve()
            output_dir = base_dir / "nef_text2kg_results"
        
        if not gt_file.exists() or not pred_file.exists():
            print(f"   ⚠️  Files not found for {onto_id}, skipping...")
            continue
        
        try:
            counts = generate_comparison_table(gt_file, pred_file, 
                                             output_dir / f"triple_comparison_{onto_id}.csv",
                                             max_rows, use_semantic, semantic_threshold)
            match_counts[onto_id] = counts
        except Exception as e:
            print(f"   ⚠️  Error for {onto_id}: {e}")
            import traceback
            traceback.print_exc()
    
    return match_counts


def main():
    """Run the Text2KGBench evaluation."""
    
    # Check if evaluation directory exists
    if not EVAL_DIR.exists():
        print(f"❌ ERROR: Text2KGBench evaluation directory not found: {EVAL_DIR}")
        print(f"   Please ensure Text2KGBench is located at: {TEXT2KG_ROOT}")
        return 1
    
    # Check if run_eval.py exists
    eval_script = EVAL_DIR / "run_eval.py"
    if not eval_script.exists():
        print(f"❌ ERROR: Evaluation script not found: {eval_script}")
        return 1
    
    # Check if Eval folder exists and has files (for easy testing)
    use_eval_folder = EVAL_FOLDER.exists() and any(EVAL_FOLDER.glob("ont_*_nef_responses.jsonl"))
    
    if use_eval_folder:
        print("="*80)
        print("📁 Using Eval folder for easy testing")
        print("="*80)
        temp_config_path = create_eval_config()
        if not temp_config_path:
            print(f"❌ ERROR: Could not create config from Eval folder")
            return 1
        config_rel = os.path.relpath(temp_config_path, EVAL_DIR)
        print(f"Created temporary config: {temp_config_path}")
    else:
        # Use regular config file
        if not CONFIG_FILE.exists():
            print(f"❌ ERROR: Config file not found: {CONFIG_FILE}")
            print(f"   Please ensure nef_text2kg_eval_config.json exists in: {SCRIPT_DIR}")
            return 1
        config_rel = os.path.relpath(CONFIG_FILE, EVAL_DIR)
    
    print("="*80)
    print("🔍 Running Text2KGBench Evaluation on NEF Results")
    print("="*80)
    print(f"Evaluation directory: {EVAL_DIR}")
    if use_eval_folder:
        print(f"Config: Temporary config (from Eval folder)")
    else:
        print(f"Config file: {CONFIG_FILE}")
    print(f"Config (relative): {config_rel}")
    print("="*80)
    print()
    
    # Create output directories if they don't exist
    if use_eval_folder:
        (EVAL_FOLDER / "eval_metrics").mkdir(parents=True, exist_ok=True)
    else:
        (SCRIPT_DIR / "nef_text2kg_results" / "eval_metrics").mkdir(parents=True, exist_ok=True)
    
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
        if use_eval_folder:
            print(f"Results saved to: {SCRIPT_DIR / 'Eval' / 'eval_metrics'}")
        else:
            print(f"Results saved to: {SCRIPT_DIR / 'nef_text2kg_results' / 'eval_metrics'}")
        print("="*80)
        
        # Generate comparison tables
        print()
        print("="*80)
        print("📊 Generating Comparison Tables")
        print("="*80)
        
        use_semantic = os.getenv("USE_SEMANTIC_MATCHING", "false").lower() == "true"
        if use_semantic and not SEMANTIC_AVAILABLE:
            print("   ⚠️  Semantic matching requested but dependencies not available")
            print("   Install: pip install google-genai numpy scikit-learn")
            use_semantic = False
        elif use_semantic:
            print("   ✅ Semantic matching enabled (threshold: 0.94)")
        
        with open(temp_config_path if use_eval_folder else CONFIG_FILE, 'r') as f:
            config_data = json.load(f)
        
        match_counts = generate_comparison_tables_for_ontologies(
            config_data, SCRIPT_DIR, use_eval_folder, max_rows=None,
            use_semantic=use_semantic, semantic_threshold=0.94
        )
        
        # Update avg_eval_results.jsonl with match counts
        if use_eval_folder:
            avg_eval_file = EVAL_FOLDER / "eval_metrics" / "avg_eval_results.jsonl"
        else:
            avg_eval_file = SCRIPT_DIR / "nef_text2kg_results" / "eval_metrics" / "avg_eval_results.jsonl"
        
        if avg_eval_file.exists() and match_counts:
            update_avg_eval_with_match_counts(avg_eval_file, match_counts)
        
        print("="*80)
        print("✅ Comparison tables generated successfully!")
        print("="*80)
        
        # Clean up temporary config if used
        if use_eval_folder and 'temp_config_path' in locals():
            try:
                os.unlink(temp_config_path)
            except Exception:
                pass
        
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

