#!/usr/bin/env python3
"""
Rank ontologies by avg_f1 / recall and summarize Tables/*.csv match types.

Usage:
  python diagnose_text2kg_run.py /path/to/results_dir
  python diagnose_text2kg_run.py   # defaults to script directory (expects Tables/ + eval_metrics/)
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import List, Dict, Any


def load_avg_metrics(avg_path: Path) -> List[Dict[str, Any]]:
    rows = []
    if not avg_path.exists():
        return rows
    with open(avg_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def summarize_tables(tables_dir: Path) -> tuple[Counter, dict[str, Counter]]:
    totals: Counter = Counter()
    per_onto: dict[str, Counter] = defaultdict(Counter)
    for f in sorted(tables_dir.glob("triple_comparison_*.csv")):
        ont = f.stem.replace("triple_comparison_", "")
        c = Counter()
        with open(f, encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            if not reader.fieldnames or "match_type" not in reader.fieldnames:
                continue
            for row in reader:
                mt = (row.get("match_type") or "").strip().upper()
                if mt:
                    c[mt] += 1
                    totals[mt] += 1
        per_onto[ont] = c
    return totals, per_onto


def count_zero_triple_responses(results_dir: Path) -> int:
    n = 0
    for p in sorted(results_dir.glob("ont_*_nef_responses.jsonl")):
        with open(p, encoding="utf-8") as fh:
            for line in fh:
                rec = json.loads(line)
                if not rec.get("triples"):
                    n += 1
    return n


def main() -> int:
    ap = argparse.ArgumentParser(description="Diagnose a Text2KG NEF results directory")
    ap.add_argument(
        "results_dir",
        nargs="?",
        default=None,
        help="Directory containing Tables/ and eval_metrics/avg_eval_results.jsonl",
    )
    args = ap.parse_args()
    base = Path(args.results_dir).resolve() if args.results_dir else Path(__file__).parent.resolve()
    tables = base / "Tables"
    avg_file = base / "eval_metrics" / "avg_eval_results.jsonl"

    print(f"Results directory: {base}")
    if not tables.is_dir():
        print(f"ERROR: Tables/ not found: {tables}")
        return 1

    totals, per_onto = summarize_tables(tables)
    print("\n=== Global comparison row counts (Tables/) ===")
    for k in ["EXACT", "SEMANTIC", "MISSING", "EXTRA"]:
        print(f"  {k:10s}  {totals.get(k, 0)}")

    onto_rows = [r for r in load_avg_metrics(avg_file) if r.get("type") == "all_test_cases" and r.get("onto")]
    if onto_rows:
        def f1_key(r):
            try:
                return float(r.get("avg_f1", 0))
            except ValueError:
                return 0.0

        onto_rows.sort(key=f1_key)
        print("\n=== Ontologies ranked by avg_f1 (lowest first) ===")
        for r in onto_rows[:8]:
            print(
                f"  {r.get('onto',''):28s}  F1={r.get('avg_f1')}  "
                f"P={r.get('avg_precision')}  R={r.get('avg_recall')}"
            )
        print("  ...")
        for r in onto_rows[-3:]:
            print(
                f"  {r.get('onto',''):28s}  F1={r.get('avg_f1')}  "
                f"P={r.get('avg_precision')}  R={r.get('avg_recall')}"
            )

    for gid in ("global", "global_macro_legacy"):
        for r in load_avg_metrics(avg_file):
            if r.get("id") == gid:
                print(f"\n=== {gid} ===")
                for key in ("avg_precision", "avg_recall", "avg_f1", "semantic_precision", "semantic_recall", "semantic_f1"):
                    if key in r:
                        print(f"  {key}: {r[key]}")
                break

    z = count_zero_triple_responses(base)
    print(f"\n=== Zero-triple sentences (all ont_*_nef_responses.jsonl) ===\n  {z}")

    worst = onto_rows[:3] if onto_rows else []
    print("\n=== Sample SEMANTIC rows (first 2 per lowest-F1 ontology) ===")
    for r in worst:
        oid = r.get("onto", "")
        csv_path = tables / f"triple_comparison_{oid}.csv"
        if not csv_path.exists():
            continue
        print(f"\n-- {oid} (avg_f1={r.get('avg_f1')}) --")
        shown = 0
        with open(csv_path, encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                if (row.get("match_type") or "").strip().upper() != "SEMANTIC":
                    continue
                print(
                    f"  sent={row.get('sent_id')}  "
                    f"exp_rel={row.get('expected_rel')}  pred_rel={row.get('predicted_rel')}"
                )
                print(f"    sub: {row.get('expected_sub')[:60]} -> {row.get('predicted_sub')[:60]}")
                shown += 1
                if shown >= 2:
                    break

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
