# Benchmarking NEF with Text2KGBench

This guide explains how to benchmark the Neural Extraction Framework (NEF) using the Text2KGBench DBpedia dataset.

## Overview

The benchmarking process consists of two main steps:

1. **Run NEF on test sentences** - Extract triples using NEF pipeline
2. **Evaluate results** - Compare NEF output with ground truth using Text2KGBench evaluation script

## Prerequisites

1. **Text2KGBench dataset** - Should be located at:
   ```
   GSoC25/NEF/Text2KGBench/Text2KGBench-main/
   ```

2. **NEF dependencies**:
   - Gemini API key (set `GEMINI_API_KEY` environment variable or use `--api-key`)
   - Redis server running (for entity linking)
   - Precomputed embeddings (`embeddings.npy` and `predicates.csv`)

3. **Python packages**: All NEF dependencies plus `nltk` (for Text2KGBench evaluation)

## Step 1: Run NEF Benchmarking

Run the benchmarking script to process all test sentences:

```bash
cd GSoC25/NEF
python benchmark_nef_text2kg.py
```

### Options

- `--ontologies`: Specify which ontologies to process (default: all 19)
  ```bash
  python benchmark_nef_text2kg.py --ontologies 1_university 7_company
  ```

- `--output-dir`: Specify output directory (default: `nef_text2kg_results`)
  ```bash
  python benchmark_nef_text2kg.py --output-dir my_results
  ```

- `--show-triples`: Show detailed triple information for each sentence
  ```bash
  python benchmark_nef_text2kg.py --show-triples
  ```

- `--quiet`: Minimal output (only summaries)
  ```bash
  python benchmark_nef_text2kg.py --quiet
  ```

- `--embeddings` and `--predicates`: Specify paths to embeddings files
  ```bash
  python benchmark_nef_text2kg.py --embeddings ./embeddings.npy --predicates ./predicates.csv
  ```

- `--redis-host`, `--redis-port`, `--redis-password`: Redis connection settings
  ```bash
  python benchmark_nef_text2kg.py --redis-host localhost --redis-port 6379
  ```

### Output

The script generates:
- One JSONL file per ontology: `ont_{ontology_id}_nef_responses.jsonl`
- Each file contains: `{"id": "...", "sent": "...", "triples": [[sub, pred, obj], ...]}`

### Progress Tracking

The script provides incremental updates showing:
- Current ontology being processed
- Progress for each sentence (X/Total, percentage)
- Number of triples extracted per sentence
- Processing time per sentence
- Summary statistics per ontology
- Final overall summary

Example output:
```
================================================================================
📊 Processing Ontology: 1_university
   Total sentences: 72
   Started at: 2025-01-XX XX:XX:XX
================================================================================

[  1/72] (  1.4%) | ont_1_university_test_1
  📝 Sentence: The AWH Engineering College is located in Kuttikkattoor, Kerala...
  ✅ Triples extracted: 3 | ⏱️  Time: 1234.5ms
  📋 First triple: (awh engineering college | located in        | kuttikkattoor)

[  2/72] (  2.8%) | ont_1_university_test_2
  ...
```

## Step 2: Run Evaluation

After benchmarking, evaluate the results using Text2KGBench evaluation script:

### Option A: Using the helper script

```bash
cd GSoC25/NEF
python run_text2kg_evaluation.py
```

### Option B: Manual evaluation

```bash
cd Text2KGBench/Text2KGBench-main/src/evaluation
python run_eval.py --eval_config_path ../../nef_text2kg_eval_config.json
```

### Evaluation Metrics

The evaluation script calculates:

- **Precision**: `correct_triples / predicted_triples`
- **Recall**: `correct_triples / gold_triples`
- **F1**: Harmonic mean of precision and recall
- **Ontology Conformance (OC)**: Percentage of relations that match the ontology
- **Relation Hallucination (RH)**: `1 - OC`
- **Subject Hallucination (SH)**: Percentage of subjects not in sentence/ontology
- **Object Hallucination (OH)**: Percentage of objects not in sentence/ontology

### Evaluation Output

Results are saved to:
- Per-ontology metrics: `nef_text2kg_results/eval_metrics/ont_{ontology_id}_eval_results.jsonl`
- Average metrics: `nef_text2kg_results/eval_metrics/avg_eval_results.jsonl`

Each result file contains detailed metrics for each test sentence plus aggregated statistics.

## Understanding the Results

### Per-Sentence Metrics

Each sentence gets metrics like:
```json
{
  "id": "ont_1_university_test_1",
  "precision": "0.67",
  "recall": "1.00",
  "f1": "0.80",
  "onto_conf": "1.00",
  "rel_halluc": "0.00",
  "sub_halluc": "0.00",
  "obj_halluc": "0.00",
  "llm_triples": [...],
  "filtered_llm_triples": [...],
  "gt_triples": [...],
  "sent": "..."
}
```

### Aggregated Metrics

Average metrics per ontology:
```json
{
  "onto": "1_university",
  "type": "all_test_cases",
  "avg_precision": "0.45",
  "avg_recall": "0.38",
  "avg_f1": "0.41",
  "avg_onto_conf": "0.92",
  "avg_sub_halluc": "0.15",
  "avg_rel_halluc": "0.08",
  "avg_obj_halluc": "0.12"
}
```

## Troubleshooting

### No triples extracted

If NEF extracts no triples for many sentences:
- Check Redis connection (entity linking is required)
- Verify embeddings files are loaded correctly
- Check predicate threshold (try lowering with `--predicate-threshold` in NEF.py)
- Review sentence format (some may be too complex)

### Evaluation errors

- Ensure output files are in correct format (JSONL with `id`, `sent`, `triples`)
- Check that ground truth files exist for all ontologies
- Verify ontology files are accessible

### Performance issues

- Processing all 19 ontologies can take significant time
- Consider processing ontologies in batches
- Use `--quiet` mode to reduce output overhead

## Example Workflow

```bash
# 1. Set up environment
export GEMINI_API_KEY="your-key-here"
export NEF_REDIS_HOST="localhost"
export NEF_REDIS_PORT="6379"

# 2. Run benchmarking on a subset first (test)
python benchmark_nef_text2kg.py --ontologies 1_university --show-triples

# 3. If results look good, run on all ontologies
python benchmark_nef_text2kg.py

# 4. Run evaluation
python run_text2kg_evaluation.py

# 5. Check results
cat nef_text2kg_results/eval_metrics/avg_eval_results.jsonl
```

## Files Created

```
GSoC25/NEF/
├── benchmark_nef_text2kg.py          # Main benchmarking script
├── run_text2kg_evaluation.py         # Evaluation helper script
├── nef_text2kg_eval_config.json      # Evaluation config
├── nef_text2kg_results/              # Output directory
│   ├── ont_1_university_nef_responses.jsonl
│   ├── ont_2_musicalwork_nef_responses.jsonl
│   ├── ...
│   └── eval_metrics/
│       ├── ont_1_university_eval_results.jsonl
│       ├── ...
│       └── avg_eval_results.jsonl
└── README_BENCHMARKING.md            # This file
```

## Notes

- The script converts NEF URIs to text labels for comparison with Text2KGBench ground truth
- Entity names are normalized (lowercase, spaces removed) for matching
- Predicate matching uses the Text2KGBench evaluation normalization
- Some triples may not match exactly due to different entity naming conventions

