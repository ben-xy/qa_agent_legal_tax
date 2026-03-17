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

- data/qa_pairs/eval_ground_truth.jsonl (10 questions)

## 2. Metric Definitions

### Retrieval metrics at K

Computed in scripts/eval/evaluate_predictions.py using normalized text matching between:

- GT targets: gold_doc_ids, fallback to references, fallback to gold_citations
- Pred targets: retrieved_doc_ids, fallback to pred_citations

Matching rule is overlap-aware after normalization:

- exact match OR substring match in either direction

Reported metrics:

- **recall@5**: fraction of relevant documents that appear in the top-5 retrieved results. Measures coverage.
- **precision@5**: fraction of the top-5 retrieved results that are relevant. Measures purity.
- **ndcg@5**: Normalized Discounted Cumulative Gain at 5. Rewards relevant documents appearing earlier.
- **mrr@5**: Mean Reciprocal Rank at 5. Higher when the first relevant hit appears earlier.
- **map@5**: Mean Average Precision at 5. Balances hit quality and ranking quality.

### Generation metrics

Computed in scripts/eval/evaluate_predictions.py and scripts/eval/evaluate_generation.py:

- exact_match: normalized string equality
- token_f1: token overlap F1
- rougeL_f1: ROUGE-L F1
- citation_hit_rate:
  - exact match OR substring match between predicted citations and GT citations
  - GT citation source: gold_citations, fallback to references

Interpretation:

- Higher is better for all generation metrics.
- exact_match is strict and can remain zero even when answers are semantically close.

## 3. Current Results

![Ablation Comparison](eval_output.png)

| Rank | experiment                 | retrieval_avg |  gen_avg | recall@5 | mrr@5 | gen_token_f1 | gen_rougeL_f1 | gen_citation_hit_rate |
| ---: | -------------------------- | ------------: | -------: | -------: | ----: | -----------: | ------------: | --------------------: |
|    1 | Hybrid_plus_Rerank         |      1.239796 | 0.248122 |   0.9333 | 1.000 |     0.114657 |      0.111163 |              0.766667 |
|    2 | Hybrid                     |      1.239796 | 0.243715 |   0.9333 | 1.000 |     0.106589 |      0.101605 |              0.766667 |
|    3 | BM25_plus_KG               |      1.239796 | 0.224886 |   0.9333 | 1.000 |     0.122516 |      0.110360 |              0.666667 |
|    4 | BM25_plus_Rerank_plus_KG   |      1.239796 | 0.222398 |   0.9333 | 1.000 |     0.096346 |      0.093247 |              0.700000 |
|    5 | Hybrid_plus_Rerank_plus_KG |      1.239796 | 0.219927 |   0.9333 | 1.000 |     0.109247 |      0.103795 |              0.666667 |
|    6 | BM25_plus_Rerank           |      1.239796 | 0.213107 |   0.9333 | 1.000 |     0.112713 |      0.106382 |              0.633333 |
|    7 | Hybrid_plus_KG             |      1.239796 | 0.212963 |   0.9333 | 1.000 |     0.093020 |      0.092167 |              0.666667 |
|    8 | BM25                       |      1.239796 | 0.151082 |   0.9333 | 1.000 |     0.072777 |      0.064886 |              0.466667 |

Note:

- Ranking excludes legacy file `outputs/ablation/eval_BM25_only.json`.
- Retrieval metrics are identical across these 8 runs, so rank differences come from generation metrics (`gen_avg`).

### 3.1 Main pipeline outputs

From outputs/eval_hybrid.json:

- recall@5: 0.9333
- precision@5: 0.8600
- ndcg@5: 2.4723
- mrr@5: 1.0000
- map@5: 0.9333
- exact_match: 0.0000
- token_f1: 0.1072
- rougeL_f1: 0.1012
- citation_hit_rate: 0.5333

From outputs/eval_hybrid_rerank.json:

- recall@5: 0.9333
- precision@5: 0.8600
- ndcg@5: 2.4723
- mrr@5: 1.0000
- map@5: 0.9333
- exact_match: 0.0000
- token_f1: 0.1036
- rougeL_f1: 0.1009
- citation_hit_rate: 0.6333

### 3.2 Ablation outputs

| Experiment                 | recall@5 |  mrr@5 | exact_match | token_f1 | rougeL_f1 | citation_hit_rate |
| -------------------------- | -------: | -----: | ----------: | -------: | --------: | ----------------: |
| BM25                       |   0.9333 | 1.0000 |      0.0000 |   0.0728 |    0.0649 |            0.4667 |
| BM25_plus_KG               |   0.9333 | 1.0000 |      0.0000 |   0.1225 |    0.1104 |            0.6667 |
| BM25_plus_Rerank           |   0.9333 | 1.0000 |      0.0000 |   0.1127 |    0.1064 |            0.6333 |
| BM25_plus_Rerank_plus_KG   |   0.9333 | 1.0000 |      0.0000 |   0.0963 |    0.0932 |            0.7000 |
| Hybrid                     |   0.9333 | 1.0000 |      0.0000 |   0.1066 |    0.1016 |            0.7667 |
| Hybrid_plus_KG             |   0.9333 | 1.0000 |      0.0000 |   0.0930 |    0.0922 |            0.6667 |
| Hybrid_plus_Rerank         |   0.9333 | 1.0000 |      0.0000 |   0.1147 |    0.1112 |            0.7667 |
| Hybrid_plus_Rerank_plus_KG |   0.9333 | 1.0000 |      0.0000 |   0.1092 |    0.1038 |            0.6667 |

### 3.3 Why some high token-score runs are not ranked first

`gen_avg` is the arithmetic mean of four generation metrics:

$$
gen\_avg = \frac{exact\_match + token\_f1 + rougeL\_f1 + citation\_hit\_rate}{4}
$$

This creates a balancing effect across metrics. For example:

- `BM25_plus_KG` has strong token overlap metrics,
- but `Hybrid_plus_Rerank` combines high overlap with top-tier citation hit rate,
- so `Hybrid_plus_Rerank` ranks higher overall.

Therefore, a single strong metric does not guarantee top overall rank.

## 4. Diagnosis

1. Retrieval is no longer the primary bottleneck.
   Retrieval metrics are consistently high across current runs (recall@5=0.9333, mrr@5=1.0).
2. Current rank differences come mainly from generation quality.
   Since retrieval metrics are identical in this batch, `gen_avg` is the key differentiator.
3. Rerank and KG effects are configuration-dependent.
   Rerank helps in Hybrid settings (`Hybrid_plus_Rerank` > `Hybrid`), while KG helps some BM25 settings but is not uniformly beneficial.
4. nDCG is currently >1.
   Under the current overlap-based relevance logic, repeated relevant matches can inflate DCG relative to idealized counting; interpret nDCG comparatively in this setup.

## 5. Clear Conclusion

- In the latest 8-experiment ablation matrix, `Hybrid_plus_Rerank` ranks first by `gen_avg` (0.2481), followed by `Hybrid` (0.2437).
- Retrieval alignment is functioning in current runs (non-zero retrieval metrics across all evaluated settings).
- The strongest practical strategy in this batch is Hybrid retrieval with rerank enabled.

## 6. Improvement Recommendations

1. Validate retrieved content quality, not just overlap-based retrieval scores.
2. Continue spot-checking top retrieved chunks to ensure evidence quality.
3. Investigate nDCG inflation (>1).
   Consider deduplicating relevance contributions in DCG or tightening relevance matching.
4. Improve exact match.
   Exact match is still 0.0 across runs; tune answer style/normalization to reduce format variance.
5. Keep notebook retry/skip logic for long ablation jobs.
   This improves robustness under intermittent provider/network failures.
6. Expand GT size further.
   10 questions are better than 2, but a larger set will yield more stable and generalizable rankings.

## 7. Reproducibility Commands

- python scripts/eval/run_eval_benchmark.py --gt data/qa_pairs/eval_ground_truth.jsonl --out outputs/preds_hybrid.jsonl --enable-rerank false
- python scripts/eval/run_eval_benchmark.py --gt data/qa_pairs/eval_ground_truth.jsonl --out outputs/preds_hybrid_rerank.jsonl --enable-rerank true
- python scripts/eval/evaluate_predictions.py --gt data/qa_pairs/eval_ground_truth.jsonl --pred outputs/preds_hybrid.jsonl --k 5 --out outputs/eval_hybrid.json
- python scripts/eval/evaluate_predictions.py --gt data/qa_pairs/eval_ground_truth.jsonl --pred outputs/preds_hybrid_rerank.jsonl --k 5 --out outputs/eval_hybrid_rerank.json
- python scripts/eval/run_eval_benchmark.py --gt data/qa_pairs/eval_ground_truth.jsonl --out outputs/ablation/preds_Hybrid_plus_Rerank_plus_KG.jsonl --enable-rerank true
- python scripts/eval/evaluate_predictions.py --gt data/qa_pairs/eval_ground_truth.jsonl --pred outputs/ablation/preds_Hybrid_plus_Rerank_plus_KG.jsonl --k 5 --out outputs/ablation/eval_Hybrid_plus_Rerank_plus_KG.json
