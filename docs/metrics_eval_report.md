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
- outputs/ablation/eval_BM25.json
- outputs/ablation/eval_BM25_plus_KG.json
- outputs/ablation/eval_BM25_plus_Rerank.json
- outputs/ablation/eval_BM25_plus_Rerank_plus_KG.json
- outputs/ablation/eval_Hybrid.json
- outputs/ablation/eval_Hybrid_plus_KG.json
- outputs/ablation/eval_Hybrid_plus_Rerank.json
- outputs/ablation/eval_Hybrid_plus_Rerank_plus_KG.json

GT basis:

- data/qa_pairs/eval_ground_truth.jsonl (2 questions)

## 2. Metric Definitions

### Retrieval metrics at K

Computed in scripts/eval/evaluate_predictions.py using normalized text matching between:

- GT targets: gold_doc_ids, fallback to references, fallback to gold_citations
- Pred targets: retrieved_doc_ids, fallback to pred_citations

Reported metrics:

- **recall@5**: fraction of relevant documents that appear in the top-5 retrieved results. Measures coverage — did we retrieve the right documents at all?
- **precision@5**: fraction of the top-5 retrieved results that are relevant. Measures purity — how many of the returned results are actually useful?
- **ndcg@5**: Normalized Discounted Cumulative Gain at 5. Rewards relevant documents appearing higher in the ranked list; a hit at rank 1 scores more than one at rank 5.
- **mrr@5**: Mean Reciprocal Rank at 5. The average of 1/rank for the first relevant hit across queries. High when the correct document tends to appear very early.
- **map@5**: Mean Average Precision at 5. The mean of per-query average precision scores, balancing both coverage and ranking quality across all queries.

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

![Ablation Comparison](eval_output.png)

| Rank | experiment                 | retrieval_avg |  gen_avg | recall@5 | mrr@5 | gen_token_f1 | gen_rougeL_f1 | gen_citation_hit_rate |
| ---: | -------------------------- | ------------: | -------: | -------: | ----: | -----------: | ------------: | --------------------: |
|    1 | Hybrid                     |      0.000000 | 0.160129 |      0.0 |   0.0 |     0.070973 |      0.069543 |                   0.5 |
|    2 | BM25_plus_Rerank_plus_KG   |      0.000000 | 0.158805 |      0.0 |   0.0 |     0.068739 |      0.066482 |                   0.5 |
|    3 | BM25_plus_Rerank           |      0.000000 | 0.158729 |      0.0 |   0.0 |     0.067883 |      0.067035 |                   0.5 |
|    4 | BM25                       |      0.000000 | 0.154949 |      0.0 |   0.0 |     0.060208 |      0.059588 |                   0.5 |
|    5 | Hybrid_plus_Rerank_plus_KG |      0.000000 | 0.143289 |      0.0 |   0.0 |     0.036249 |      0.036909 |                   0.5 |
|    6 | Hybrid_plus_Rerank         |      0.000000 | 0.053600 |      0.0 |   0.0 |     0.107944 |      0.106456 |                   0.0 |
|    7 | BM25_plus_KG               |      0.000000 | 0.038930 |      0.0 |   0.0 |     0.079351 |      0.076369 |                   0.0 |
|    8 | Hybrid_plus_KG             |      0.000000 | 0.032939 |      0.0 |   0.0 |     0.065925 |      0.065831 |                   0.0 |

Note: Because `retrieval_avg` is 0.0 for all runs, ranking is dominated by `gen_avg`.

### 3.1 Main pipeline outputs

From outputs/eval_hybrid.json:

- retrieval metrics: all 0.0
- exact_match: 0.0
- token_f1: 0.0515
- rougeL_f1: 0.0512
- citation_hit_rate: 0.5

From outputs/eval_hybrid_rerank.json:

- retrieval metrics: all 0.0
- exact_match: 0.0
- token_f1: 0.0647
- rougeL_f1: 0.0643
- citation_hit_rate: 0.5

### 3.2 Ablation outputs

| Experiment                 | recall@5 |  mrr@5 | exact_match | token_f1 | rougeL_f1 | citation_hit_rate |
| -------------------------- | -------: | -----: | ----------: | -------: | --------: | ----------------: |
| BM25                       |   0.0000 | 0.0000 |      0.0000 |   0.0602 |    0.0596 |            0.5000 |
| BM25_plus_KG               |   0.0000 | 0.0000 |      0.0000 |   0.0794 |    0.0764 |            0.0000 |
| BM25_plus_Rerank           |   0.0000 | 0.0000 |      0.0000 |   0.0763 |    0.0751 |            0.5000 |
| BM25_plus_Rerank_plus_KG   |   0.0000 | 0.0000 |      0.0000 |   0.0687 |    0.0665 |            0.5000 |
| Hybrid                     |   0.0000 | 0.0000 |      0.0000 |   0.0710 |    0.0695 |            0.5000 |
| Hybrid_plus_KG             |   0.0000 | 0.0000 |      0.0000 |   0.0659 |    0.0658 |            0.0000 |
| Hybrid_plus_Rerank         |   0.0000 | 0.0000 |      0.0000 |   0.1079 |    0.1065 |            0.0000 |
| Hybrid_plus_Rerank_plus_KG |   0.0000 | 0.0000 |      0.0000 |   0.0362 |    0.0369 |            0.5000 |

### 3.3 Why some high token-score runs rank low in gen_avg

`gen_avg` is the arithmetic mean of four generation metrics:

$$
gen\_avg = \frac{exact\_match + token\_f1 + rougeL\_f1 + citation\_hit\_rate}{4}
$$

This creates a balancing effect across metrics. For example:

- `Hybrid_plus_Rerank` has relatively high `token_f1` and `rougeL_f1`,
- but `citation_hit_rate = 0.0` and `exact_match = 0.0`,
- so its final `gen_avg` is still low.

By contrast, runs such as `Hybrid` and `BM25` have moderate overlap metrics plus `citation_hit_rate = 0.5`, which lifts their `gen_avg` ranking.

Therefore, a single strong metric (such as token overlap) does not guarantee a high overall rank when citation and exact-match signals are weak.

## 4. Diagnosis

1. Retrieval is the bottleneck.
   All retrieval metrics are zero across all runs. This indicates no overlap between predicted retrieval identifiers and GT targets under current matching rules.
2. Rerank cannot help when first-stage retrieval has no effective candidates.
   Rerank-only improvements are unstable across runs and configurations because stage-1 recall is still zero.
3. KG impact is mixed under current setup.
   Some KG-on runs reduce citation hit rate to 0.0 while improving token overlap in specific settings, indicating trade-offs rather than consistent gains.

## 5. Clear Conclusion

- Among the current 8-experiment ablation matrix, `Hybrid` ranks first by `gen_avg` (0.160), suggesting the combined BM25+vector approach produces the best balance of token overlap and citation recall.
- Retrieval metrics remain uniformly zero across all runs, so all conclusions are based on generation quality only.
- Therefore, current conclusions about rerank/KG should be treated as provisional until retrieval ID alignment is fixed.

## 6. Improvement Recommendations

1. Fix retrieval ID alignment first.
   Ensure retrieved_doc_ids in predictions are directly comparable to GT targets (or make GT target format closer to predicted IDs).
2. Validate retrieved content quality.
   Log top retrieved chunks per question before generation and verify they contain answer-bearing text.
3. Clean ablation outputs before each section 6.2 run.
   Remove stale `eval_*.json` files or write to a timestamped run folder, so summary rankings reflect only the current experiment matrix.
4. Stabilize vector retrieval path.
   If embedding quota or provider constraints occur, either:

- temporarily evaluate BM25-only, or
- switch to a stable embedding provider with enough quota.

5. Increase evaluation set size.
   With only 2 questions, metrics are high variance. Expand GT to produce robust conclusions.
6. Keep rerank/KG tuning after recall is healthy.
   Re-run ablation only after non-zero recall appears; then rerank impact will be meaningful.

## 7. Reproducibility Commands

- python scripts/eval/run_eval_benchmark.py --gt data/qa_pairs/eval_ground_truth.jsonl --out outputs/preds_hybrid.jsonl --enable-rerank false
- python scripts/eval/run_eval_benchmark.py --gt data/qa_pairs/eval_ground_truth.jsonl --out outputs/preds_hybrid_rerank.jsonl --enable-rerank true
- python scripts/eval/evaluate_predictions.py --gt data/qa_pairs/eval_ground_truth.jsonl --pred outputs/preds_hybrid.jsonl --k 5 --out outputs/eval_hybrid.json
- python scripts/eval/evaluate_predictions.py --gt data/qa_pairs/eval_ground_truth.jsonl --pred outputs/preds_hybrid_rerank.jsonl --k 5 --out outputs/eval_hybrid_rerank.json
- python scripts/eval/run_eval_benchmark.py --gt data/qa_pairs/eval_ground_truth.jsonl --out outputs/ablation/preds_Hybrid_plus_Rerank_plus_KG.jsonl --enable-rerank true
- python scripts/eval/evaluate_predictions.py --gt data/qa_pairs/eval_ground_truth.jsonl --pred outputs/ablation/preds_Hybrid_plus_Rerank_plus_KG.jsonl --k 5 --out outputs/ablation/eval_Hybrid_plus_Rerank_plus_KG.json
