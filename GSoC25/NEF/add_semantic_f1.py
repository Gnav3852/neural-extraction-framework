#!/usr/bin/env python3
"""
Add semantic F1 (and P/R) to avg_eval_results.jsonl using existing comparison table CSVs.

Reads Tables/triple_comparison_*.csv, counts EXACT/SEMANTIC/MISSING/EXTRA per ontology,
computes semantic_precision, semantic_recall, semantic_f1, then updates avg_eval_results.jsonl
in place for each ontology line and adds macro-averaged global semantic P/R/F1 to the global line.

Usage:
  python add_semantic_f1.py                          # use script dir for Tables/ and avg_eval_results.jsonl
  python add_semantic_f1.py --results-dir ExpRes/ExpOne  # use given dir (Tables/ and eval_metrics/ inside it)
"""

import argparse
import csv
import json
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()


def count_match_types(csv_path: Path) -> dict:
    """Count EXACT, SEMANTIC, MISSING, EXTRA from a comparison table CSV."""
    counts = {"EXACT": 0, "SEMANTIC": 0, "MISSING": 0, "EXTRA": 0}
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if "match_type" not in (reader.fieldnames or []):
            return counts
        for row in reader:
            mt = (row.get("match_type") or "").strip().upper()
            if mt in counts:
                counts[mt] += 1
    return counts


def semantic_f1_from_counts(counts: dict) -> tuple[float, float, float]:
    """Compute semantic precision, recall, F1 from match-type counts."""
    exact = counts["EXACT"]
    semantic = counts["SEMANTIC"]
    missing = counts["MISSING"]
    extra = counts["EXTRA"]
    correct = exact + semantic
    total_gold = correct + missing
    total_pred = correct + extra
    if total_pred <= 0:
        p = 0.0
    else:
        p = correct / total_pred
    if total_gold <= 0:
        r = 0.0
    else:
        r = correct / total_gold
    if p + r > 0:
        f1 = 2 * (p * r) / (p + r)
    else:
        f1 = 0.0
    return round(p, 4), round(r, 4), round(f1, 4)


def main():
    parser = argparse.ArgumentParser(description="Add semantic P/R/F1 to avg_eval_results.jsonl from Tables/*.csv")
    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        metavar="PATH",
        help="Results directory containing Tables/ and eval_metrics/avg_eval_results.jsonl (default: script dir)",
    )
    args = parser.parse_args()

    if args.results_dir:
        base = Path(args.results_dir).resolve()
        TABLES_DIR = base / "Tables"
        AVG_FILE = base / "eval_metrics" / "avg_eval_results.jsonl"
    else:
        TABLES_DIR = SCRIPT_DIR / "Tables"
        AVG_FILE = SCRIPT_DIR / "avg_eval_results.jsonl"

    if not TABLES_DIR.exists():
        print(f"ERROR: Tables directory not found: {TABLES_DIR}")
        return 1
    if not AVG_FILE.exists():
        print(f"ERROR: Avg file not found: {AVG_FILE}")
        return 1

    # Collect semantic metrics per ontology from CSVs
    semantic_by_onto = {}
    for csv_path in sorted(TABLES_DIR.glob("triple_comparison_*.csv")):
        stem = csv_path.stem
        if not stem.startswith("triple_comparison_"):
            continue
        onto_id = stem[len("triple_comparison_"):].strip()
        if not onto_id:
            continue
        counts = count_match_types(csv_path)
        p, r, f1 = semantic_f1_from_counts(counts)
        semantic_by_onto[onto_id] = {
            "semantic_precision": p,
            "semantic_recall": r,
            "semantic_f1": f1,
            "missing": counts["MISSING"],
            "extra": counts["EXTRA"],
        }

    if not semantic_by_onto:
        print("ERROR: No comparison tables found in Tables/")
        return 1

    # Read avg_eval_results.jsonl
    entries = []
    with open(AVG_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entries.append(json.loads(line))

    # Update ontology lines with semantic metrics
    updated = 0
    for entry in entries:
        onto = entry.get("onto")
        if onto is None:
            continue
        onto_id = onto if isinstance(onto, str) else onto.get("id", "")
        if onto_id not in semantic_by_onto:
            continue
        m = semantic_by_onto[onto_id]
        entry["semantic_precision"] = f"{m['semantic_precision']:.2f}"
        entry["semantic_recall"] = f"{m['semantic_recall']:.2f}"
        entry["semantic_f1"] = f"{m['semantic_f1']:.2f}"
        entry["missing"] = m["missing"]
        entry["extra"] = m["extra"]
        updated += 1

    # Macro-averaged global semantic P/R/F1 (match existing global avg_f1 style)
    n_onto = len(semantic_by_onto)
    global_p = sum(m["semantic_precision"] for m in semantic_by_onto.values()) / n_onto
    global_r = sum(m["semantic_recall"] for m in semantic_by_onto.values()) / n_onto
    global_f1 = sum(m["semantic_f1"] for m in semantic_by_onto.values()) / n_onto
    for entry in entries:
        if entry.get("id") == "global":
            entry["semantic_precision"] = f"{global_p:.2f}"
            entry["semantic_recall"] = f"{global_r:.2f}"
            entry["semantic_f1"] = f"{global_f1:.2f}"
            break

    # Write back
    with open(AVG_FILE, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"Updated {AVG_FILE} with semantic F1 for {updated} ontologies.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
