#!/usr/bin/env python3
"""Compare two Text2KG result directories (Tables match counts + avg_eval global lines)."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def table_totals(tables_dir: Path) -> Counter:
    c: Counter = Counter()
    for f in sorted(tables_dir.glob("triple_comparison_*.csv")):
        with open(f, encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            if not reader.fieldnames or "match_type" not in reader.fieldnames:
                continue
            for row in reader:
                mt = (row.get("match_type") or "").strip().upper()
                if mt:
                    c[mt] += 1
    return c


def load_globals(avg_path: Path) -> dict:
    out = {}
    if not avg_path.exists():
        return out
    with open(avg_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            rid = rec.get("id")
            if rid in ("global", "global_macro_legacy"):
                out[rid] = rec
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Compare two NEF Text2KG result folders")
    ap.add_argument("baseline", type=str, help="Baseline results directory (e.g. OpenAI_4o_6shot_v2)")
    ap.add_argument("candidate", type=str, help="Candidate results directory (e.g. OpenAI_4o_improved_4)")
    args = ap.parse_args()
    a = Path(args.baseline).resolve()
    b = Path(args.candidate).resolve()
    for label, p in ("baseline", a), ("candidate", b):
        if not (p / "Tables").is_dir():
            print(f"ERROR: {label} missing Tables/: {p}")
            return 1

    ca, cb = table_totals(a / "Tables"), table_totals(b / "Tables")
    print(f"baseline:  {a}")
    print(f"candidate: {b}\n")
    print("=== Tables row deltas (candidate - baseline) ===")
    for k in ["EXACT", "SEMANTIC", "MISSING", "EXTRA"]:
        da = ca.get(k, 0)
        db = cb.get(k, 0)
        print(f"  {k:10s}  {db:5d} - {da:5d} = {db - da:+5d}")

    ga = load_globals(a / "eval_metrics" / "avg_eval_results.jsonl")
    gb = load_globals(b / "eval_metrics" / "avg_eval_results.jsonl")
    print("\n=== avg_eval_results.jsonl global rows ===")
    for gid in ("global", "global_macro_legacy"):
        ra, rb = ga.get(gid), gb.get(gid)
        print(f"\n-- {gid} --")
        if not ra and not rb:
            print("  (no rows in either run)")
            continue
        for key in ("avg_precision", "avg_recall", "avg_f1", "semantic_f1"):
            va = ra.get(key, "-") if ra else "-"
            vb = rb.get(key, "-") if rb else "-"
            print(f"  {key:18s}  baseline={va!s:>6}  candidate={vb!s:>6}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
