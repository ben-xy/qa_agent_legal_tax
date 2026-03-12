# Metrics Evaluation Report

## 1. Scope
This document explains how metrics are computed in the project and summarizes the latest experiment outputs.

Code basis:
- scripts/eval/evaluate_predictions.py
- scripts/eval/evaluate_generation.py
- scripts/eval/run_eval_benchmark.py
- src/retrievers/hybrid_retriever.py
- src/retrievers/cohere_reranker.py

Latest output basis:
- outputs/eval_hybrid.json
- outputs/eval_hybrid_rerank.json
- outputs/ablation/eval_BM25_only.json
- outputs/ablation/eval_BM25_plus_Rerank.json
- outputs/ablation/eval_Hybrid.json
- outputs/ablation/eval_Hybrid_plus_Rerank.json

GT basis:
- data/qa_pairs/eval_ground_truth.jsonl (2 questions)

## 2. Metric Definitions
### Retrieval metrics at K
Computed in scripts/eval/evaluate_predictions.py using normalized text matching between:
- GT targets: gold_doc_ids, fallback to references, fallback to gold_citations
- Pred targets: retrieved_doc_ids, fallback to pred_citations

Reported metrics:
- recall@5
- precision@5
- ndcg@5
- mrr@5
- map@5

### Generation metrics
Computed in scripts/eval/evaluate_predictions.py and scripts/eval/evaluate_generation.py:
- exact_match: normalized string equality
- token_f1: token overlap F1
- rougeL_f1: ROUGE-L F1
- citation_hit_rate: now uses partial matching after normalization
  - exact match OR substring match between predicted citations and GT citations
  - GT citation source: gold_citations, fallback to references

Interpretation:
- Higher is better for all generation metrics.
- exact_match is strict and can be zero even when answers are semantically close.

## 3. Current Results
### 3.1 Main pipeline outputs
From outputs/eval_hybrid.json:
- retrieval metrics: all 0.0
- exact_match: 0.0
- token_f1: 0.0774
- rougeL_f1: 0.0637
- citation_hit_rate: 0.0

From outputs/eval_hybrid_rerank.json:
- retrieval metrics: all 0.0
- exact_match: 0.0
- token_f1: 0.0649
- rougeL_f1: 0.0647
- citation_hit_rate: 0.5

### 3.2 Ablation outputs
From outputs/ablation/*.json:

| Experiment | recall@5 | mrr@5 | exact_match | token_f1 | rougeL_f1 | citation_hit_rate |
|---|---:|---:|---:|---:|---:|---:|
| BM25_only | 0.0000 | 0.0000 | 0.0000 | 0.0816 | 0.0809 | 0.5000 |
| BM25_plus_Rerank | 0.0000 | 0.0000 | 0.0000 | 0.0763 | 0.0751 | 0.5000 |
| Hybrid | 0.0000 | 0.0000 | 0.0000 | 0.0636 | 0.0615 | 0.5000 |
| Hybrid_plus_Rerank | 0.0000 | 0.0000 | 0.0000 | 0.0621 | 0.0599 | 0.5000 |

## 4. Diagnosis
1. Retrieval is the bottleneck.
All retrieval metrics are zero across all runs. This indicates no overlap between predicted retrieval identifiers and GT targets under current matching rules.

2. Rerank cannot help when first-stage retrieval has no effective candidates.
In the current outputs, rerank variants do not improve retrieval and slightly reduce generation overlap metrics.

3. Citation metric behavior is now aligned to the GT schema.
Because GT uses references (not gold_citations), citation scoring now falls back correctly and supports partial matching.

## 5. Clear Conclusion
- Current best generation overlap in ablation is BM25_only.
- Hybrid and rerank are not improving outcomes under the present retrieval quality.
- This is not evidence that rerank is universally bad. It indicates first-stage retrieval quality and identifier alignment are not yet sufficient for rerank to provide gains.

## 6. Improvement Recommendations
1. Fix retrieval ID alignment first.
Ensure retrieved_doc_ids in predictions are directly comparable to GT targets (or make GT target format closer to predicted IDs).

2. Validate retrieved content quality.
Log top retrieved chunks per question before generation and verify they contain answer-bearing text.

3. Stabilize vector retrieval path.
If embedding quota or provider constraints occur, either:
- temporarily evaluate BM25-only, or
- switch to a stable embedding provider with enough quota.

4. Increase evaluation set size.
With only 2 questions, metrics are high variance. Expand GT to produce robust conclusions.

5. Keep rerank after recall is healthy.
Re-run ablation only after non-zero recall appears; then rerank impact will be meaningful.

## 7. Reproducibility Commands
- python scripts/eval/run_eval_benchmark.py --gt data/qa_pairs/eval_ground_truth.jsonl --out outputs/preds_hybrid.jsonl --enable-rerank false
- python scripts/eval/run_eval_benchmark.py --gt data/qa_pairs/eval_ground_truth.jsonl --out outputs/preds_hybrid_rerank.jsonl --enable-rerank true
- python scripts/eval/evaluate_predictions.py --gt data/qa_pairs/eval_ground_truth.jsonl --pred outputs/preds_hybrid.jsonl --k 5 --out outputs/eval_hybrid.json
- python scripts/eval/evaluate_predictions.py --gt data/qa_pairs/eval_ground_truth.jsonl --pred outputs/preds_hybrid_rerank.jsonl --k 5 --out outputs/eval_hybrid_rerank.json
