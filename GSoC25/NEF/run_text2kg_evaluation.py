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
            sent_id = data.get("id", "")
            
            # Handle different formats
            if "triples" in data:
                triples = data["triples"]
                formatted_triples = []
                for t in triples:
                    if isinstance(t, list) and len(t) == 3:
                        # Format: ["subject", "predicate", "object"]
                        formatted_triples.append((t[0], t[1], t[2]))
                    elif isinstance(t, dict):
                        # Format: {"sub": "...", "rel": "...", "obj": "..."}
                        formatted_triples.append((t.get("sub", ""), t.get("rel", ""), t.get("obj", "")))
                triples_dict[sent_id] = formatted_triples
    return triples_dict


def generate_comparison_table(
    gt_file: Path,
    pred_file: Path,
    output_csv: Path,
    output_html: Path,
    max_rows: int = 100
):
    """Generate side-by-side comparison table (CSV and HTML)."""
    
    print(f"\n📊 Generating comparison table...")
    print(f"   Loading ground truth from: {gt_file}")
    gt_triples = load_triples_from_jsonl(gt_file)
    
    print(f"   Loading predictions from: {pred_file}")
    pred_triples = load_triples_from_jsonl(pred_file)
    
    # Collect all sentence IDs
    all_sent_ids = sorted(set(list(gt_triples.keys()) + list(pred_triples.keys())))
    
    # Prepare comparison data
    comparison_rows = []
    total_exact_matches = 0
    total_mismatches = 0
    
    row_count = 0
    for sent_id in all_sent_ids:
        if row_count >= max_rows:
            break
        
        gt_list = gt_triples.get(sent_id, [])
        pred_list = pred_triples.get(sent_id, [])
        
        # Normalize for matching
        gt_normalized = {normalize_triple(*t): t for t in gt_list}
        pred_normalized = {normalize_triple(*t): t for t in pred_list}
        
        # Find exact matches
        exact_matches = set(gt_normalized.keys()) & set(pred_normalized.keys())
        
        # Find unmatched
        gt_unmatched = set(gt_normalized.keys()) - exact_matches
        pred_unmatched = set(pred_normalized.keys()) - exact_matches
        
        # Add exact matches
        for norm_key in exact_matches:
            gt_t = gt_normalized[norm_key]
            pred_t = pred_normalized[norm_key]
            comparison_rows.append({
                "sent_id": sent_id,
                "expected": format_triple(*gt_t),
                "predicted": format_triple(*pred_t),
                "match_type": "EXACT",
                "similarity": "1.00",
                "expected_sub": strip_quotes(gt_t[0]),
                "expected_rel": strip_quotes(gt_t[1]),
                "expected_obj": strip_quotes(gt_t[2]),
                "predicted_sub": strip_quotes(pred_t[0]),
                "predicted_rel": strip_quotes(pred_t[1]),
                "predicted_obj": strip_quotes(pred_t[2]),
            })
            total_exact_matches += 1
            row_count += 1
            if row_count >= max_rows:
                break
        
        # Add unmatched expected (false negatives)
        for gt_norm in gt_unmatched:
            if row_count >= max_rows:
                break
            gt_t = gt_normalized[gt_norm]
            comparison_rows.append({
                "sent_id": sent_id,
                "expected": format_triple(*gt_t),
                "predicted": "",
                "match_type": "MISSING",
                "similarity": "",
                "expected_sub": strip_quotes(gt_t[0]),
                "expected_rel": strip_quotes(gt_t[1]),
                "expected_obj": strip_quotes(gt_t[2]),
                "predicted_sub": "",
                "predicted_rel": "",
                "predicted_obj": "",
            })
            total_mismatches += 1
            row_count += 1
        
        # Add unmatched predicted (false positives)
        for pred_norm in pred_unmatched:
            if row_count >= max_rows:
                break
            pred_t = pred_normalized[pred_norm]
            comparison_rows.append({
                "sent_id": sent_id,
                "expected": "",
                "predicted": format_triple(*pred_t),
                "match_type": "EXTRA",
                "similarity": "",
                "expected_sub": "",
                "expected_rel": "",
                "expected_obj": "",
                "predicted_sub": strip_quotes(pred_t[0]),
                "predicted_rel": strip_quotes(pred_t[1]),
                "predicted_obj": strip_quotes(pred_t[2]),
            })
            total_mismatches += 1
            row_count += 1
    
    # Write CSV
    print(f"   Writing CSV to: {output_csv}")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        fieldnames = [
            "sent_id", "match_type", "similarity",
            "expected", "expected_sub", "expected_rel", "expected_obj",
            "predicted", "predicted_sub", "predicted_rel", "predicted_obj"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(comparison_rows)
    
    # Write HTML
    print(f"   Writing HTML to: {output_html}")
    output_html.parent.mkdir(parents=True, exist_ok=True)
    html_content = generate_html_table(comparison_rows, total_exact_matches, total_mismatches)
    with open(output_html, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"   ✅ Comparison table generated: {total_exact_matches} exact matches, {total_mismatches} mismatches")


def generate_html_table(rows: List[Dict], exact: int, mismatch: int) -> str:
    """Generate HTML table with styling."""
    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Triple Comparison Table</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        table {{ border-collapse: collapse; width: 100%; margin-top: 20px; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #4CAF50; color: white; }}
        tr:nth-child(even) {{ background-color: #f2f2f2; }}
        .EXACT {{ background-color: #d4edda; }}
        .MISSING {{ background-color: #f8d7da; }}
        .EXTRA {{ background-color: #f8d7da; }}
        .summary {{ margin: 20px 0; padding: 15px; background-color: #e7f3ff; border-radius: 5px; }}
    </style>
</head>
<body>
    <h1>Triple Comparison: Expected vs Predicted</h1>
    <div class="summary">
        <h2>Summary</h2>
        <p><strong>Exact Matches:</strong> {exact}</p>
        <p><strong>Mismatches:</strong> {mismatch}</p>
        <p><strong>Total Rows:</strong> {len(rows)}</p>
    </div>
    <table>
        <thead>
            <tr>
                <th>Sentence ID</th>
                <th>Match Type</th>
                <th>Similarity</th>
                <th>Expected Triple</th>
                <th>Predicted Triple</th>
            </tr>
        </thead>
        <tbody>
"""
    
    for row in rows:
        match_type = row["match_type"]
        html += f"""
            <tr class="{match_type}">
                <td>{row['sent_id']}</td>
                <td>{match_type}</td>
                <td>{row['similarity']}</td>
                <td>{row['expected'] or '(missing)'}</td>
                <td>{row['predicted'] or '(missing)'}</td>
            </tr>
"""
    
    html += """
        </tbody>
    </table>
</body>
</html>
"""
    return html


def generate_comparison_tables_for_ontologies(config_data: dict, base_dir: Path, use_eval_folder: bool, max_rows: int = 100):
    """Generate comparison tables for all ontologies in the config."""
    onto_list = config_data.get("onto_list", [])
    path_patterns = config_data.get("path_patterns", {})
    
    for onto in onto_list:
        onto_id = onto if isinstance(onto, str) else onto.get("id", "")
        
        # Determine file paths
        if use_eval_folder:
            gt_file = EVAL_FOLDER / f"ont_{onto_id}_ground_truth.jsonl"
            pred_file = EVAL_FOLDER / f"ont_{onto_id}_nef_responses.jsonl"
            output_dir = EVAL_FOLDER
        else:
            # Paths in config are relative to EVAL_DIR
            gt_pattern = path_patterns.get("gt", "").replace("$$onto$$", onto_id)
            pred_pattern = path_patterns.get("sys", "").replace("$$onto$$", onto_id)
            
            # Resolve paths relative to EVAL_DIR
            gt_file = (EVAL_DIR / gt_pattern).resolve()
            pred_file = (EVAL_DIR / pred_pattern).resolve()
            output_dir = base_dir / "nef_text2kg_results"
        
        # Check if files exist
        if not gt_file.exists():
            print(f"   ⚠️  Ground truth file not found: {gt_file}, skipping {onto_id}...")
            continue
        if not pred_file.exists():
            print(f"   ⚠️  Prediction file not found: {pred_file}, skipping {onto_id}...")
            continue
        
        # Generate comparison table
        output_csv = output_dir / f"triple_comparison_{onto_id}.csv"
        output_html = output_dir / f"triple_comparison_{onto_id}.html"
        
        try:
            generate_comparison_table(gt_file, pred_file, output_csv, output_html, max_rows)
        except Exception as e:
            print(f"   ⚠️  Error generating comparison table for {onto_id}: {e}")
            import traceback
            traceback.print_exc()


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
        
        # Load config to get ontology list
        if use_eval_folder:
            with open(temp_config_path, 'r') as f:
                config_data = json.load(f)
        else:
            with open(CONFIG_FILE, 'r') as f:
                config_data = json.load(f)
        
        # Generate comparison tables for each ontology
        generate_comparison_tables_for_ontologies(
            config_data, 
            SCRIPT_DIR, 
            use_eval_folder,
            max_rows=100
        )
        
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

