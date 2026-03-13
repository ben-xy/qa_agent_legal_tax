# Chunking Strategies in This Project

## Scope

This document summarizes the chunking logic currently present in the codebase, how each strategy works, trade-offs, and when to use each one.

## Current Engineering Logic

There are currently **two chunking implementations** in the repository:

1. `src/eval/chunking.py` (used by chunking evaluation pipeline)
2. `src/utils/chunking_strategies.py` (utility module, currently not wired into the active chunk build pipeline)

### What is actively used now

The chunk evaluation flow (for A/B comparison) uses:

- `src/eval/chunking.py`
- `scripts/eval/chunk_eval/chunk_eval_pipeline.ipynb`
- `scripts/eval/chunk_eval/eval_chunking_compare.py`

The active flow builds two datasets:

- `data/tmp/acts_chunked_fixed`
- `data/tmp/acts_chunked_struct`

Then it swaps `data/acts_chunked` and runs retrieval/generation evaluation for each variant.

### What exists but is not currently wired

`src/utils/chunking_strategies.py` defines reusable chunking functions (`fixed_size_chunks`, `section_aware_chunks`) with richer metadata (`start_char`, `end_char`, heading context), but no current script imports this module directly.

## Strategies and Behavior

## 1) Fixed-size chunking

### Implementations

- `src/eval/chunking.py::chunk_fixed`
- `src/utils/chunking_strategies.py::fixed_size_chunks`

### Core idea

Split text by character windows with overlap:

- next start index = `current + (chunk_size - overlap)`
- if no content remains, stop

### Typical defaults in this repo

- Eval pipeline (`src/eval/chunking.py`): `chunk_size=800`, `overlap=120`
- Utility module (`src/utils/chunking_strategies.py`): `chunk_size=1200`, `overlap=200`

### Advantages

- Very simple and robust
- Deterministic output size and count
- Works even when source text has poor structure/formatting
- Usually better index density and recall for fragmented source text

### Disadvantages

- Can cut sentences/sections in unnatural places
- May lose legal context boundaries (e.g., split between section heading and body)
- Overlap introduces duplicated tokens and index/storage overhead

### Best-use scenarios

- Baseline retrieval benchmark
- Messy OCR/legal text where headings are unreliable
- Fast, stable fallback when structural parsing fails

## 2) Structure-aware chunking

### Implementations

- `src/eval/chunking.py::chunk_structure_aware` using `split_by_structure`
- `src/utils/chunking_strategies.py::section_aware_chunks`

### Core idea

Try to preserve legal section boundaries first, then fallback to fixed-size splitting for oversized blocks.

#### In `src/eval/chunking.py`

Boundary regex uses legal markers such as:

- `Section <number>(...)`
- `PART <roman numeral>`
- `CHAPTER <number>`

If a block exceeds `max_len`, it is split by `chunk_fixed`.

#### In `src/utils/chunking_strategies.py`

Boundary regex supports heading-like lines:

- Markdown headings (`# ...`)
- section lines starting with `Section ...`

If a block exceeds `max_chunk_size`, it falls back to internal fixed-size splitting and keeps section/heading metadata.

### Advantages

- Better semantic coherence for legal text
- More likely to keep heading + related provisions together
- Easier downstream attribution/citation when section context is preserved

### Disadvantages

- Depends on heading quality and formatting consistency
- If source structure is noisy, boundaries may be missed or fragmented
- Can produce uneven chunk lengths (affects embedding consistency and scoring)

### Best-use scenarios

- Legal/regulated corpora with clear section markers
- Citation-heavy answers where section context matters
- Explainability-focused retrieval analysis

## Practical Comparison in This Repo

Current A/B evaluation scripts compare:

- `fixed` (fixed-size)
- `struct` (structure-aware)

using:

- `scripts/eval/run_eval_benchmark.py`
- `scripts/eval/evaluate_predictions.py`
- `scripts/eval/chunk_eval/eval_chunking_compare.py`

This is the correct setup for measuring chunking impact under the same retriever/generation stack.

## Known Consistency Gap

There is a consistency gap between:

- active builder (`src/eval/chunking.py`)
- utility builder (`src/utils/chunking_strategies.py`)

They use different defaults and slightly different structure regex rules.

This is not a runtime bug, but it can cause confusion when comparing results across scripts.

## Recommendation

For now:

1. Use `src/eval/chunking.py` outputs for benchmark conclusions, because that is the currently wired path.
2. Treat `src/utils/chunking_strategies.py` as a reusable library candidate.
3. If you want one canonical implementation, refactor `src/eval/chunking.py` to call `src/utils/chunking_strategies.py` directly and unify defaults.

## Decision Guide

Choose **fixed-size** when:

- source formatting is inconsistent
- you need a stable baseline quickly
- recall under noisy text is the top priority

Choose **structure-aware** when:

- legal section markers are reliable
- answer grounding/citations are important
- you want semantically coherent chunks over uniform length
