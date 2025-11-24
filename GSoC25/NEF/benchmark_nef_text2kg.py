#!/usr/bin/env python3
"""
Benchmark NEF using Text2KGBench DBpedia dataset.

This script:
1. Loads Text2KGBench test sentences
2. Runs NEF pipeline on each sentence with incremental progress updates
3. Converts NEF URIs to text format expected by Text2KGBench
4. Generates output files compatible with Text2KGBench evaluation
5. Provides detailed progress tracking for each test case
"""

import os
import json
import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

# Import NEF components
try:
    from NEF import EnhancedNEFPipeline, _bootstrap_gemini_client
except ImportError:
    # Fallback if running from different directory
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from NEF import EnhancedNEFPipeline, _bootstrap_gemini_client

# Text2KGBench paths (relative to this script)
SCRIPT_DIR = Path(__file__).parent
TEXT2KG_ROOT = SCRIPT_DIR / "Text2KGBench" / "Text2KGBench-main"
DBPEDIA_DATA = TEXT2KG_ROOT / "data" / "dbpedia_webnlg"

# DBpedia ontology list
DBPEDIA_ONTOLOGIES = [
    "1_university", "2_musicalwork", "3_airport", "4_building", 
    "5_athlete", "6_politician", "7_company", "8_celestialbody",
    "9_astronaut", "10_comicscharacter", "11_meanoftransportation",
    "12_monument", "13_food", "14_writtenwork", "15_sportsteam",
    "16_city", "17_artist", "18_scientist", "19_film"
]


def uri_to_text(uri: str) -> str:
    """Convert URI to readable text (similar to Bench.py logic)."""
    if not uri:
        return ""
    import re
    # Extract last segment after / or #
    parts = uri.replace("#", "/").split("/")
    last = parts[-1] if parts else ""
    # Split camelCase and underscores
    text = re.sub(r"[_\-\.]+", " ", last)
    text = re.sub(r"(?<=[a-z0-9])([A-Z])", r" \1", text)
    return re.sub(r"\s+", " ", text).strip()


def format_triple_for_text2kg(sub_text: str, pred_text: str, obj_text: str) -> List[str]:
    """
    Format triple as list [subject, predicate, object] for Text2KGBench.
    Handles special cases like quoted strings in objects.
    """
    # Clean up the texts
    sub_text = sub_text.strip()
    pred_text = pred_text.strip()
    obj_text = obj_text.strip()
    
    # Text2KGBench sometimes uses quoted strings for objects
    # We'll keep it simple and just return the cleaned text
    return [sub_text, pred_text, obj_text]


def print_progress_header(onto_id: str, total_sentences: int):
    """Print a nice header for each ontology."""
    print("\n" + "="*80)
    print(f"📊 Processing Ontology: {onto_id}")
    print(f"   Total sentences: {total_sentences}")
    print(f"   Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)


def print_sentence_progress(
    idx: int, 
    total: int, 
    sent_id: str, 
    sentence: str, 
    num_triples: int,
    elapsed_ms: float
):
    """Print progress for a single sentence."""
    progress_pct = (idx / total) * 100
    sentence_preview = sentence[:60] + "..." if len(sentence) > 60 else sentence
    
    print(f"\n[{idx:3d}/{total}] ({progress_pct:5.1f}%) | {sent_id}")
    print(f"  📝 Sentence: {sentence_preview}")
    print(f"  ✅ Triples extracted: {num_triples} | ⏱️  Time: {elapsed_ms:.1f}ms")
    
    if num_triples == 0:
        print(f"  ⚠️  Warning: No triples extracted for this sentence")


def print_triple_details(triples: List[List[str]], verbose: bool = False):
    """Print details about extracted triples."""
    if not triples:
        return
    
    if verbose:
        print(f"  📋 Extracted triples:")
        for i, (sub, pred, obj) in enumerate(triples, 1):
            print(f"     {i}. ({sub[:30]:30s} | {pred[:25]:25s} | {obj[:30]:30s})")
    else:
        # Just show count and first triple
        if triples:
            sub, pred, obj = triples[0]
            print(f"  📋 First triple: ({sub[:25]:25s} | {pred[:20]:20s} | {obj[:25]:25s})")
            if len(triples) > 1:
                print(f"  📋 ... and {len(triples) - 1} more")


def print_ontology_summary(onto_id: str, stats: Dict[str, Any]):
    """Print summary statistics for an ontology."""
    print("\n" + "-"*80)
    print(f"📈 Summary for {onto_id}:")
    print(f"   Total sentences: {stats['total']}")
    print(f"   Successful: {stats['successful']}")
    print(f"   Failed: {stats['failed']}")
    print(f"   Total triples: {stats['total_triples']}")
    print(f"   Avg triples/sentence: {stats['avg_triples']:.2f}")
    print(f"   Avg time/sentence: {stats['avg_time_ms']:.1f}ms")
    print(f"   Total time: {stats['total_time_s']:.1f}s")
    print("-"*80)


def run_nef_on_text2kg(
    pipeline: EnhancedNEFPipeline,
    test_file: Path,
    output_file: Path,
    verbose: bool = True,
    show_triples: bool = False
) -> Dict[str, Any]:
    """
    Run NEF on Text2KGBench test sentences and save results.
    
    Args:
        pipeline: Initialized NEF pipeline
        test_file: Path to test.jsonl file
        output_file: Path to save NEF output (JSONL format)
        verbose: Show detailed progress
        show_triples: Show all triple details (verbose mode)
    
    Returns:
        Dictionary with statistics
    """
    # Load all test sentences first
    test_items = []
    with open(test_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                test_items.append(json.loads(line))
    
    total = len(test_items)
    if total == 0:
        print(f"⚠️  No test sentences found in {test_file}")
        return {"total": 0, "successful": 0, "failed": 0, "total_triples": 0}
    
    # Print header
    onto_id = test_file.stem.replace("_test", "")
    print_progress_header(onto_id, total)
    
    results = []
    stats = {
        "total": total,
        "successful": 0,
        "failed": 0,
        "total_triples": 0,
        "total_time_ms": 0.0
    }
    
    # Process each sentence
    for idx, test_item in enumerate(test_items, 1):
        sent_id = test_item.get("id", f"unknown_{idx}")
        sentence = test_item.get("sent", "")
        
        if not sentence:
            if verbose:
                print(f"\n[{idx:3d}/{total}] ⚠️  Empty sentence for {sent_id}")
            results.append({
                "id": sent_id,
                "sent": sentence,
                "triples": []
            })
            stats["failed"] += 1
            continue
        
        # Run NEF pipeline with timing
        start_time = time.perf_counter()
        try:
            nef_triples = pipeline.run_pipeline(sentence, debug=False)
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            stats["total_time_ms"] += elapsed_ms
            
            # Convert NEF output (URIs) to Text2KGBench format (text labels)
            triples = []
            for s_uri, p_uri, o_uri, meta in nef_triples:
                # Convert URIs to text
                sub_text = uri_to_text(s_uri)
                pred_text = uri_to_text(p_uri) if p_uri else ""
                obj_text = uri_to_text(o_uri)
                
                if sub_text and pred_text and obj_text:
                    triples.append(format_triple_for_text2kg(sub_text, pred_text, obj_text))
            
            stats["total_triples"] += len(triples)
            stats["successful"] += 1
            
            # Print progress
            if verbose:
                print_sentence_progress(idx, total, sent_id, sentence, len(triples), elapsed_ms)
                if show_triples and triples:
                    print_triple_details(triples, verbose=True)
            
            results.append({
                "id": sent_id,
                "sent": sentence,
                "triples": triples
            })
            
        except Exception as e:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            stats["total_time_ms"] += elapsed_ms
            stats["failed"] += 1
            
            if verbose:
                print(f"\n[{idx:3d}/{total}] ❌ ERROR processing {sent_id}")
                print(f"   Error: {str(e)}")
                print(f"   Sentence: {sentence[:80]}...")
            
            results.append({
                "id": sent_id,
                "sent": sentence,
                "triples": []
            })
    
    # Calculate final stats
    stats["avg_triples"] = stats["total_triples"] / stats["successful"] if stats["successful"] > 0 else 0.0
    stats["avg_time_ms"] = stats["total_time_ms"] / total if total > 0 else 0.0
    stats["total_time_s"] = stats["total_time_ms"] / 1000.0
    
    # Save results
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    # Print summary
    print_ontology_summary(onto_id, stats)
    
    if verbose:
        print(f"💾 Saved {len(results)} results to {output_file}")
    
    return stats


def main():
    """Main benchmarking function."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Benchmark NEF on Text2KGBench DBpedia dataset with incremental progress updates",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run on all ontologies with default settings
  python benchmark_nef_text2kg.py

  # Run on specific ontologies only
  python benchmark_nef_text2kg.py --ontologies 1_university 7_company

  # Run with detailed triple output
  python benchmark_nef_text2kg.py --show-triples

  # Run quietly (minimal output)
  python benchmark_nef_text2kg.py --quiet
        """
    )
    parser.add_argument(
        "--ontologies",
        nargs="+",
        default=DBPEDIA_ONTOLOGIES,
        help=f"List of ontology IDs to benchmark (default: all {len(DBPEDIA_ONTOLOGIES)})"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="nef_text2kg_results",
        help="Directory to save NEF output files (default: nef_text2kg_results)"
    )
    parser.add_argument(
        "--embeddings",
        type=str,
        default=None,
        help="Path to embeddings.npy (default: auto-detect)"
    )
    parser.add_argument(
        "--predicates",
        type=str,
        default=None,
        help="Path to predicates.csv (default: auto-detect)"
    )
    parser.add_argument(
        "--redis-host",
        type=str,
        default=os.getenv("NEF_REDIS_HOST", "localhost"),
        help="Redis host (default: localhost or NEF_REDIS_HOST env var)"
    )
    parser.add_argument(
        "--redis-port",
        type=int,
        default=int(os.getenv("NEF_REDIS_PORT", "6379")),
        help="Redis port (default: 6379 or NEF_REDIS_PORT env var)"
    )
    parser.add_argument(
        "--redis-password",
        type=str,
        default=os.getenv("NEF_REDIS_PASSWORD", ""),
        help="Redis password (default: NEF_REDIS_PASSWORD env var or empty)"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="Gemini API key (default: GEMINI_API_KEY env var or prompt)"
    )
    parser.add_argument(
        "--show-triples",
        action="store_true",
        help="Show detailed triple information for each sentence"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Minimal output (only summaries)"
    )
    
    args = parser.parse_args()
    
    # Initialize NEF pipeline
    print("="*80)
    print("🚀 Initializing NEF Pipeline")
    print("="*80)
    try:
        client = _bootstrap_gemini_client(args.api_key)
        pipeline = EnhancedNEFPipeline(
            client=client,
            embeddings_path=args.embeddings,
            predicates_path=args.predicates,
            redis_host=args.redis_host,
            redis_port=args.redis_port,
            redis_password=args.redis_password if args.redis_password else None,
            verbose=not args.quiet,
        )
        print("✅ NEF Pipeline initialized successfully\n")
    except Exception as e:
        print(f"❌ ERROR: Failed to initialize NEF pipeline: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Check Text2KGBench data directory
    if not DBPEDIA_DATA.exists():
        print(f"❌ ERROR: Text2KGBench data directory not found: {DBPEDIA_DATA}")
        print(f"   Please ensure Text2KGBench is located at: {TEXT2KG_ROOT}")
        return 1
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"📁 Output directory: {output_dir.absolute()}\n")
    
    # Process each ontology
    all_stats = []
    start_time = time.time()
    
    for onto_id in args.ontologies:
        test_file = DBPEDIA_DATA / "test" / f"ont_{onto_id}_test.jsonl"
        output_file = output_dir / f"ont_{onto_id}_nef_responses.jsonl"
        
        if not test_file.exists():
            print(f"⚠️  Warning: Test file not found: {test_file}")
            print(f"   Skipping ontology: {onto_id}\n")
            continue
        
        stats = run_nef_on_text2kg(
            pipeline=pipeline,
            test_file=test_file,
            output_file=output_file,
            verbose=not args.quiet,
            show_triples=args.show_triples
        )
        
        stats["ontology"] = onto_id
        all_stats.append(stats)
    
    # Print final summary
    total_time = time.time() - start_time
    print("\n" + "="*80)
    print("📊 FINAL SUMMARY")
    print("="*80)
    print(f"Total ontologies processed: {len(all_stats)}")
    print(f"Total sentences: {sum(s['total'] for s in all_stats)}")
    print(f"Successful: {sum(s['successful'] for s in all_stats)}")
    print(f"Failed: {sum(s['failed'] for s in all_stats)}")
    print(f"Total triples extracted: {sum(s['total_triples'] for s in all_stats)}")
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    print("="*80)
    
    # Per-ontology breakdown
    if len(all_stats) > 1:
        print("\n📋 Per-ontology breakdown:")
        for stats in all_stats:
            print(f"  {stats['ontology']:25s} | "
                  f"Sentences: {stats['total']:3d} | "
                  f"Triples: {stats['total_triples']:3d} | "
                  f"Avg: {stats['avg_triples']:.2f} | "
                  f"Time: {stats['total_time_s']:.1f}s")
    
    print(f"\n💾 All results saved to: {output_dir.absolute()}")
    print(f"\n📝 Next step: Run Text2KGBench evaluation script:")
    print(f"   cd {TEXT2KG_ROOT / 'src' / 'evaluation'}")
    print(f"   python run_eval.py --eval_config_path ../../nef_text2kg_eval_config.json")
    print("="*80)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

