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
import statistics
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

# Import NEF components
try:
    from NEF import EnhancedNEFPipeline, _bootstrap_gemini_client, _bootstrap_openrouter_client
except ImportError:
    # Fallback if running from different directory
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from NEF import EnhancedNEFPipeline, _bootstrap_gemini_client, _bootstrap_openrouter_client

# Data paths: use NEF/Dataset (test + ontologies)
SCRIPT_DIR = Path(__file__).parent
TEXT2KG_ROOT = SCRIPT_DIR / "Text2KGBench" / "Text2KGBench-main"
DBPEDIA_DATA = SCRIPT_DIR / "Dataset"

# DBpedia ontology list
DBPEDIA_ONTOLOGIES = [
    "1_university", "2_musicalwork", "3_airport", "4_building", 
    "5_athlete", "6_politician", "7_company", "8_celestialbody",
    "9_astronaut", "10_comicscharacter", "11_meanoftransportation",
    "12_monument", "13_food", "14_writtenwork", "15_sportsteam",
    "16_city", "17_artist", "18_scientist", "19_film"
]


def load_ontology_relations(ontology_path: Path) -> List[str]:
    """
    Load allowed relations from Text2KGBench ontology file.
    Returns list of relation labels (e.g., ["academicStaffSize", "established", ...]).
    """
    with open(ontology_path, "r", encoding="utf-8") as f:
        ontology = json.load(f)
    relations = [rel["label"] for rel in ontology.get("relations", [])]
    return relations


def map_text2kg_relation_to_dbpedia_uri(relation_label: str) -> str:
    """
    Map Text2KGBench relation label to DBpedia predicate URI.
    
    Text2KGBench uses camelCase labels like "academicStaffSize"
    DBpedia uses: http://dbpedia.org/ontology/academicStaffSize
    
    Args:
        relation_label: Text2KGBench relation label (e.g., "academicStaffSize")
    
    Returns:
        DBpedia predicate URI (e.g., "http://dbpedia.org/ontology/academicStaffSize")
    """
    # Most Text2KGBench relations map directly to DBpedia ontology predicates
    return f"http://dbpedia.org/ontology/{relation_label}"


def get_allowed_predicate_uris(ontology_path: Path) -> List[str]:
    """
    Load ontology and return list of DBpedia predicate URIs that are allowed.
    
    Args:
        ontology_path: Path to Text2KGBench ontology JSON file
    
    Returns:
        List of DBpedia predicate URIs (e.g., ["http://dbpedia.org/ontology/academicStaffSize", ...])
    """
    relations = load_ontology_relations(ontology_path)
    uris = [map_text2kg_relation_to_dbpedia_uri(rel) for rel in relations]
    return uris


def load_ontology_metadata(ontology_path: Path) -> Dict[str, Dict[str, str]]:
    """
    Load ontology and return metadata for each predicate (label, domain, range).
    
    Args:
        ontology_path: Path to Text2KGBench ontology JSON file
    
    Returns:
        Dictionary mapping predicate URI to metadata:
        {
            "http://dbpedia.org/ontology/academicStaffSize": {
                "label": "academicStaffSize",
                "domain": "University",
                "range": "number"
            },
            ...
        }
    """
    with open(ontology_path, "r", encoding="utf-8") as f:
        ontology = json.load(f)
    
    metadata = {}
    for rel in ontology.get("relations", []):
        label = rel.get("label", "")
        if label:
            uri = map_text2kg_relation_to_dbpedia_uri(label)
            metadata[uri] = {
                "label": label,
                "domain": rel.get("domain", ""),
                "range": rel.get("range", "")
            }
    
    return metadata


def get_concept_label(ontology: Dict[str, Any], concept_qid: str) -> str:
    """
    Get the label for an ontology concept by its QID.
    
    Args:
        ontology: Ontology dictionary
        concept_qid: Concept QID (e.g., "Artist", "University")
    
    Returns:
        Concept label or empty string if not found
    """
    for concept in ontology.get("concepts", []):
        if concept.get("qid") == concept_qid:
            return concept.get("label", "")
    return ""


def format_ontology_for_extraction(ontology: Dict[str, Any]) -> str:
    """
    Format ontology for extraction prompt following Text2KGBench baseline pattern.
    
    Args:
        ontology: Ontology dictionary from JSON file
    
    Returns:
        Formatted string with concepts and relations for extraction prompt
    """
    # Get all concept labels (comma-separated)
    concepts = [c.get("label", "") for c in ontology.get("concepts", [])]
    concepts_str = ", ".join([c for c in concepts if c])
    
    # Get all relations in format: relation(domain, range)
    relations = []
    for rel in ontology.get("relations", []):
        rel_label = rel.get("label", "")
        if not rel_label:
            continue
        
        domain_qid = rel.get("domain", "")
        range_qid = rel.get("range", "")
        
        domain_label = get_concept_label(ontology, domain_qid) if domain_qid else ""
        range_label = get_concept_label(ontology, range_qid) if range_qid else ""
        
        # Format: relation(domain, range)
        if domain_label or range_label:
            relations.append(f"{rel_label}({domain_label}, {range_label})")
        else:
            relations.append(rel_label)
    
    relations_str = ", ".join(relations)
    
    return f"""CONTEXT:

Ontology Concepts: {concepts_str}
Ontology Relations: {relations_str}

Given the following ontology and sentence, please extract the triples from the sentence according to the relations in the ontology."""


def uri_to_entity_text(uri: str) -> str:
    """
    Convert entity URI to text, preserving underscores and TitleCase.
    Example: http://dbpedia.org/resource/AWH_Engineering_College → AWH_Engineering_College
    """
    if not uri:
        return ""
    import re
    # Extract last segment after / or #
    parts = uri.replace("#", "/").split("/")
    last = parts[-1] if parts else ""
    # Preserve underscores and case for entities
    return last.strip()


def uri_to_predicate_text(uri: str) -> str:
    """
    Convert predicate URI to text, preserving camelCase.
    Example: http://dbpedia.org/ontology/academicStaffSize → academicStaffSize
    """
    if not uri:
        return ""
    import re
    # Extract last segment after / or #
    parts = uri.replace("#", "/").split("/")
    last = parts[-1] if parts else ""
    # Preserve camelCase for predicates (no conversion)
    return last.strip()


def format_object_text(obj_text: str) -> str:
    """
    Format object text for Text2KGBench output.
    Returns the text as-is (no quotes added - evaluation script expects plain strings).
    The evaluation script normalizes triples by removing spaces/underscores and lowercasing,
    so the exact format doesn't matter as long as it's a plain string.
    """
    return obj_text.strip()


def format_triple_for_text2kg(sub_text: str, pred_text: str, obj_text: str) -> List[str]:
    """
    Format triple as list [subject, predicate, object] for Text2KGBench.
    Preserves format: underscores for entities, camelCase for predicates.
    Objects are plain strings (no quotes) as expected by the evaluation script.
    """
    # Clean up the texts
    sub_text = sub_text.strip()
    pred_text = pred_text.strip()
    obj_text = format_object_text(obj_text)
    
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


def calculate_latency_stats(latencies: List[float]) -> Dict[str, float]:
    """Calculate latency statistics (min, max, mean, median, p95, p99)."""
    if not latencies:
        return {}
    
    sorted_latencies = sorted(latencies)
    n = len(sorted_latencies)
    
    return {
        "min_ms": min(latencies),
        "max_ms": max(latencies),
        "mean_ms": statistics.mean(latencies),
        "median_ms": statistics.median(latencies),
        "p95_ms": sorted_latencies[int(n * 0.95)] if n > 0 else 0.0,
        "p99_ms": sorted_latencies[int(n * 0.99)] if n > 0 else 0.0,
        "stddev_ms": statistics.stdev(latencies) if n > 1 else 0.0,
    }


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
    
    # Print latency statistics
    if stats.get('latency_stats'):
        ls = stats['latency_stats']
        print(f"\n   ⏱️  Latency Statistics:")
        print(f"      Min: {ls.get('min_ms', 0):.1f}ms")
        print(f"      Max: {ls.get('max_ms', 0):.1f}ms")
        print(f"      Mean: {ls.get('mean_ms', 0):.1f}ms")
        print(f"      Median: {ls.get('median_ms', 0):.1f}ms")
        print(f"      P95: {ls.get('p95_ms', 0):.1f}ms")
        print(f"      P99: {ls.get('p99_ms', 0):.1f}ms")
        print(f"      Std Dev: {ls.get('stddev_ms', 0):.1f}ms")
    
    print("-"*80)


def run_nef_on_text2kg(
    pipeline: EnhancedNEFPipeline,
    test_file: Path,
    output_file: Path,
    ontology_path: Optional[Path] = None,
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
    
    # Reset pipeline's allowed predicates, metadata, and context for this ontology
    pipeline.allowed_predicates = None
    pipeline.predicate_metadata = None
    pipeline.ontology_context = None
    
    # Load ontology and get allowed predicates if ontology path is provided
    allowed_predicates = None
    predicate_metadata = None
    ontology_context = None
    if ontology_path and ontology_path.exists():
        try:
            with open(ontology_path, "r", encoding="utf-8") as f:
                ontology = json.load(f)
            
            allowed_predicates = get_allowed_predicate_uris(ontology_path)
            predicate_metadata = load_ontology_metadata(ontology_path)
            ontology_context = format_ontology_for_extraction(ontology)
            
            if verbose:
                print(f"   📚 Loaded ontology: {len(allowed_predicates)} allowed predicates")
                print(f"   📖 Loaded predicate metadata: {len(predicate_metadata)} predicates with semantic info")
                print(f"   📋 Loaded ontology context for extraction")
        except Exception as e:
            if verbose:
                print(f"   ⚠️  Warning: Could not load ontology: {e}")
                print(f"   Continuing without ontology filtering...")
    
    # Update pipeline's allowed predicates, metadata, and context if we have them
    if allowed_predicates:
        pipeline.allowed_predicates = set(allowed_predicates)
    if predicate_metadata:
        pipeline.predicate_metadata = predicate_metadata
    if ontology_context:
        pipeline.ontology_context = ontology_context
    
    results = []
    stats = {
        "total": total,
        "successful": 0,
        "failed": 0,
        "total_triples": 0,
        "total_time_ms": 0.0,
        "latencies_ms": []  # Store all latencies for statistics
    }
    
    # Open output file for incremental writing (using context manager for safety)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    def safe_write_json(output_f, result_dict, sent_id, verbose_flag):
        """Safely write JSON to file with validation and error handling."""
        try:
            # Ensure all values are strings and serializable
            safe_result = {
                "id": str(result_dict.get("id", sent_id)),
                "sent": str(result_dict.get("sent", "")),
                "triples": [[str(t[0]), str(t[1]), str(t[2])] for t in result_dict.get("triples", [])]
            }
            
            # Serialize to JSON
            json_str = json.dumps(safe_result, ensure_ascii=False)
            
            # Validate JSON string (check for null bytes or empty)
            if not json_str or len(json_str) < 10:
                if verbose_flag:
                    print(f"   ⚠️ WARNING: Invalid JSON for {sent_id}, skipping write")
                return False
            
            if '\x00' in json_str:  # Check for null bytes
                if verbose_flag:
                    print(f"   ⚠️ WARNING: Null bytes detected in JSON for {sent_id}, skipping write")
                return False
            
            # Write to file
            output_f.write(json_str + "\n")
            output_f.flush()
            return True
            
        except (UnicodeEncodeError, TypeError, ValueError) as e:
            if verbose_flag:
                print(f"   ⚠️ ERROR serializing JSON for {sent_id}: {e}")
            # Write safe fallback
            try:
                safe_fallback = {
                    "id": str(sent_id),
                    "sent": str(result_dict.get("sent", "")),
                    "triples": [],
                    "_error": str(e)
                }
                output_f.write(json.dumps(safe_fallback, ensure_ascii=False) + "\n")
                output_f.flush()
                return True
            except Exception as e2:
                if verbose_flag:
                    print(f"   ⚠️ CRITICAL: Could not write fallback JSON for {sent_id}: {e2}")
                return False
    
    with open(output_file, "w", encoding="utf-8", newline='\n') as output_f:
        # Process each sentence
        for idx, test_item in enumerate(test_items, 1):
            sent_id = test_item.get("id", f"unknown_{idx}")
            sentence = test_item.get("sent", "")
            
            if not sentence:
                if verbose:
                    print(f"\n[{idx:3d}/{total}] ⚠️  Empty sentence for {sent_id}")
                result = {
                    "id": sent_id,
                    "sent": sentence,
                    "triples": []
                }
                # Write immediately with safe function
                safe_write_json(output_f, result, sent_id, verbose)
                results.append(result)
                stats["failed"] += 1
                continue
            
            # Run NEF pipeline with timing
            start_time = time.perf_counter()
            try:
                nef_triples = pipeline.run_pipeline(sentence, debug=False)
                elapsed_ms = (time.perf_counter() - start_time) * 1000.0
                stats["total_time_ms"] += elapsed_ms
                stats["latencies_ms"].append(elapsed_ms)  # Collect latency
                
                # Convert NEF output (URIs) to Text2KGBench format (text labels)
                triples = []
                for s_uri, p_uri, o_uri, meta in nef_triples:
                    # Ensure all are strings (handles literals and URIs)
                    s_uri = str(s_uri) if s_uri is not None else ""
                    p_uri = str(p_uri) if p_uri is not None else ""
                    o_uri = str(o_uri) if o_uri is not None else ""
                    
                    # Convert URIs to text, preserving format
                    sub_text = uri_to_entity_text(s_uri)
                    pred_text = uri_to_predicate_text(p_uri) if p_uri else ""
                    obj_text = uri_to_entity_text(o_uri)  # Works for literals too
                    
                    if sub_text and pred_text and obj_text:
                        triples.append(format_triple_for_text2kg(sub_text, pred_text, obj_text))
                
                stats["total_triples"] += len(triples)
                stats["successful"] += 1
                
                # Print progress
                if verbose:
                    print_sentence_progress(idx, total, sent_id, sentence, len(triples), elapsed_ms)
                    if show_triples and triples:
                        print_triple_details(triples, verbose=True)
                
                result = {
                    "id": sent_id,
                    "sent": sentence,
                    "triples": triples
                }
                
                # Write incrementally (immediately after processing) with safe function
                safe_write_json(output_f, result, sent_id, verbose)
                results.append(result)
                
            except Exception as e:
                elapsed_ms = (time.perf_counter() - start_time) * 1000.0
                stats["total_time_ms"] += elapsed_ms
                stats["latencies_ms"].append(elapsed_ms)  # Collect latency even on error
                stats["failed"] += 1
                
                if verbose:
                    print(f"\n[{idx:3d}/{total}] ❌ ERROR processing {sent_id}")
                    print(f"   Error: {str(e)}")
                    print(f"   Sentence: {sentence[:80]}...")
                
                result = {
                    "id": sent_id,
                    "sent": sentence,
                    "triples": []
                }
                # Write immediately even on error with safe function
                safe_write_json(output_f, result, sent_id, verbose)
                results.append(result)
    
    # Calculate final stats
    stats["avg_triples"] = stats["total_triples"] / stats["successful"] if stats["successful"] > 0 else 0.0
    stats["avg_time_ms"] = stats["total_time_ms"] / total if total > 0 else 0.0
    stats["total_time_s"] = stats["total_time_ms"] / 1000.0
    
    # Calculate latency statistics
    stats["latency_stats"] = calculate_latency_stats(stats["latencies_ms"])
    
    # Print summary
    print_ontology_summary(onto_id, stats)
    
    if verbose:
        print(f"💾 Saved {len(results)} results incrementally to {output_file}")
    
    return stats


def save_latency_metrics(all_stats: List[Dict[str, Any]], output_dir: Path) -> Path:
    """Save detailed latency metrics to a JSON file."""
    metrics_file = output_dir / "latency_metrics.json"
    
    metrics = {
        "timestamp": datetime.now().isoformat(),
        "ontologies": []
    }
    
    for stats in all_stats:
        onto_metrics = {
            "ontology": stats.get("ontology", "unknown"),
            "total_sentences": stats.get("total", 0),
            "successful": stats.get("successful", 0),
            "failed": stats.get("failed", 0),
            "total_triples": stats.get("total_triples", 0),
            "avg_triples_per_sentence": stats.get("avg_triples", 0.0),
            "latency_stats": stats.get("latency_stats", {}),
            "all_latencies_ms": stats.get("latencies_ms", [])
        }
        metrics["ontologies"].append(onto_metrics)
    
    # Calculate overall stats
    all_latencies = []
    for stats in all_stats:
        all_latencies.extend(stats.get("latencies_ms", []))
    
    metrics["overall"] = {
        "total_sentences": sum(s.get("total", 0) for s in all_stats),
        "total_triples": sum(s.get("total_triples", 0) for s in all_stats),
        "latency_stats": calculate_latency_stats(all_latencies)
    }
    
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    
    return metrics_file


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

  # Use OpenRouter (e.g. GPT-4o mini) for extraction/disambiguation; Gemini still used for embeddings
  python benchmark_nef_text2kg.py --output-dir ExpRes/OpenAI --reasoner openrouter --reasoner-model openai/gpt-4o-mini
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
        "--reasoner",
        choices=["gemini", "openrouter"],
        default="gemini",
        help="Reasoning backend: gemini (default) or openrouter. Embeddings always use Gemini.",
    )
    parser.add_argument(
        "--reasoner-model",
        type=str,
        default="openai/gpt-4o-mini",
        help="Model for reasoning when --reasoner=openrouter (e.g. openai/gpt-4o-mini). Ignored when reasoner=gemini.",
    )
    parser.add_argument(
        "--openrouter-api-key",
        type=str,
        default=None,
        help="OpenRouter API key (or set OPENROUTER_API_KEY). Required when --reasoner=openrouter.",
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default="gemini-2.5-flash",
        help="Gemini model for reasoning when --reasoner=gemini (default: gemini-2.5-flash).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0,
        help="Sampling temperature for extraction and disambiguation (default 0).",
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

    # OpenRouter: bootstrap only when reasoner=openrouter
    openrouter_client = None
    reasoner_model = args.llm_model
    if args.reasoner == "openrouter":
        try:
            openrouter_client = _bootstrap_openrouter_client(args.openrouter_api_key)
            reasoner_model = args.reasoner_model
        except Exception as e:
            print(f"❌ ERROR: {e}")
            return 1

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
            llm_model=args.llm_model,
            redis_host=args.redis_host,
            redis_port=args.redis_port,
            redis_password=args.redis_password if args.redis_password else None,
            verbose=not args.quiet,
            reasoner=args.reasoner,
            reasoner_model=reasoner_model,
            openrouter_client=openrouter_client,
            temperature=args.temperature,
        )
        print("✅ NEF Pipeline initialized successfully\n")
    except Exception as e:
        print(f"❌ ERROR: Failed to initialize NEF pipeline: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Check Dataset directory (test + ontologies)
    if not DBPEDIA_DATA.exists():
        print(f"❌ ERROR: Dataset directory not found: {DBPEDIA_DATA}")
        print(f"   Please ensure Dataset/ (test/, ground_truth/, ontologies/) exists under NEF.")
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
        ontology_file = DBPEDIA_DATA / "ontologies" / f"{onto_id}_ontology.json"
        output_file = output_dir / f"ont_{onto_id}_nef_responses.jsonl"
        
        if not test_file.exists():
            print(f"⚠️  Warning: Test file not found: {test_file}")
            print(f"   Skipping ontology: {onto_id}\n")
            continue
        
        stats = run_nef_on_text2kg(
            pipeline=pipeline,
            test_file=test_file,
            output_file=output_file,
            ontology_path=ontology_file if ontology_file.exists() else None,
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
    
    # Save latency metrics
    metrics_file = save_latency_metrics(all_stats, output_dir)
    print(f"📊 Latency metrics saved to: {metrics_file}")
    
    print(f"\n📝 Next step: Run Text2KGBench evaluation script:")
    print(f"   cd {TEXT2KG_ROOT / 'src' / 'evaluation'}")
    print(f"   python run_eval.py --eval_config_path ../../nef_text2kg_eval_config.json")
    print("="*80)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

