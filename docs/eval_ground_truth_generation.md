# Ground Truth Generation (`eval_ground_truth.jsonl`)

This project uses `scripts/eval/generate_eval_ground_truth.py` to create the evaluation ground truth file:

- **Script**: `scripts/eval/generate_eval_ground_truth.py`
- **Default output**: `data/qa_pairs/eval_ground_truth.jsonl`

## Purpose

The script standardizes QA data into a unified JSONL format used by evaluation scripts such as:

- `scripts/eval/run_eval_benchmark.py`
- `scripts/eval/evaluate_predictions.py`
- `scripts/eval/evaluate_generation.py`

## Supported Input Types

The script accepts one input file via `--in` with one of these formats:

- `.jsonl`
- `.json`
- `.csv`

If `--in` is not provided, it auto-detects from `data/qa_pairs/` (e.g., `eval_seed.json`, `qa_pairs.jsonl`, etc., depending on script version).

## Input Schema (Minimum)

Each record must include:

- `question` (or compatible aliases: `query`, `q`, `user_query`, `prompt`)
- `answer` (or compatible aliases: `gold_answer`, `reference_answer`, `ground_truth`, `gt_answer`)

Optional fields:

- `id` / `qid` / `question_id` / `uuid`
- `references` (or aliases: `reference_chunks`, `gold_chunks`, `evidence`, `citations`)
- retrieval metadata: `gold_doc_ids`, `gold_doc_id`, `doc_ids`, `source_ids`, `meta`

## Example Input (`.json`)

`data/qa_pairs/eval_seed.json` is a valid example:
- top-level JSON array
- each item is an object with `id`, `question`, `answer`, and optional `references`

## Output Format

The script writes one JSON object per line to `eval_ground_truth.jsonl`.  
Each output row contains:

- `id` (string)
- `qid` (string; same value as `id`)
- `question` (string)
- `gold_answer` (string)
- `answer` (string; backward compatibility)
- `references` (array)

Optional metadata fields are preserved when present.

## Usage

From repository root:

````bash
python [generate_eval_ground_truth.py](http://_vscodecontentref_/0) \
  --in [eval_seed.json](http://_vscodecontentref_/1) \
  --out [eval_ground_truth.jsonl](http://_vscodecontentref_/2)