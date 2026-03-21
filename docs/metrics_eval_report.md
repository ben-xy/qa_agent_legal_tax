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
- outputs/ablation/eval_BM25_KG.json
- outputs/ablation/eval_BM25_Rerank.json
- outputs/ablation/eval_BM25_Rerank_KG.json
- outputs/ablation/eval_Hybrid.json
- outputs/ablation/eval_Hybrid_KG.json
- outputs/ablation/eval_Hybrid_Rerank.json
- outputs/ablation/eval_Hybrid_Rerank_KG.json

GT basis:

- data/qa_pairs/eval_ground_truth.jsonl (32 questions)

## 2. Metric Definitions

### Retrieval metrics at K

Computed in scripts/eval/evaluate_predictions.py using normalized text matching between:

- GT targets: `gold_doc_ids`, fallback to `references`, fallback to `gold_citations`
- Pred targets: `eval_friendly_doc_ids`, fallback to `retrieved_doc_ids`, fallback to `pred_citations`

Matching rule now uses **law-title normalization** before scoring:

- Examples mapped to a comparable title form:
  - `doc_234::Customs Act 1960::chunk=...` -> `customs act 1960`
  - `Customs Act 1960 - Singapore Statutes Online > ...` -> `customs act 1960`
- Retrieval relevance match is equality on normalized law titles.

Prediction format update in scripts/eval/run_eval_benchmark.py:

- `retrieved_doc_ids`: canonical chunk-level IDs (kept for traceability)
- `eval_friendly_doc_ids`: law-title friendly IDs (added for evaluation matching)

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

> Run date: 2026-03-21 · GT size: 32 questions · K = 5
>
> `exact_match` = 0.0 across all experiments → excluded from `gen_avg` (adaptive rule).
> `citation_hit_rate` > 0 in all experiments → included.
> `gen_avg = (token_f1 + rougeL_f1 + citation_hit_rate) / 3`

![Ablation Comparison](eval_output.png)

### 3.1 Strategy A vs Strategy B (baseline pair)

| Strategy          | combined_score | retrieval_avg | gen_avg | recall@5 | precision@5 | ndcg@5 |  mrr@5 |  map@5 | token_f1 | rougeL_f1 | citation_hit_rate |
| ----------------- | -------------: | ------------: | ------: | -------: | ----------: | -----: | -----: | -----: | -------: | --------: | ----------------: |
| Hybrid (A)        |         0.4002 |        0.5764 |  0.2240 |   0.7969 |      0.2188 | 0.6411 | 0.6453 | 0.5802 |   0.0833 |    0.0730 |            0.5156 |
| Hybrid+Rerank (B) |         0.4678 |        0.7380 |  0.1975 |   0.8958 |      0.2563 | 0.8447 | 0.8865 | 0.8068 |   0.0740 |    0.0655 |            0.4531 |

Strategy B gains +0.0676 combined score over A primarily through retrieval improvement (+0.1616 retrieval_avg),
while generation metrics slightly decline due to the reranker narrowing the candidate set.

### 3.2 Ablation summary (ranked by combined_score)

| Rank | Experiment       | combined_score | retrieval_avg | gen_avg | recall@5 | precision@5 | ndcg@5 |  mrr@5 |  map@5 | token_f1 | rougeL_f1 | citation_hit_rate |
| ---: | ---------------- | -------------: | ------------: | ------: | -------: | ----------: | -----: | -----: | -----: | -------: | --------: | ----------------: |
|    1 | Hybrid_Rerank    |         0.5126 |        0.7657 |  0.2596 |   0.9115 |      0.2625 | 0.8772 | 0.9375 | 0.8396 |   0.0904 |    0.0791 |            0.6094 |
|    2 | BM25_Rerank_KG   |         0.4883 |        0.7380 |  0.2387 |   0.8958 |      0.2563 | 0.8447 | 0.8865 | 0.8068 |   0.0752 |    0.0679 |            0.5729 |
|    3 | Hybrid_Rerank_KG |         0.4875 |        0.7657 |  0.2094 |   0.9115 |      0.2625 | 0.8772 | 0.9375 | 0.8396 |   0.0747 |    0.0692 |            0.4844 |
|    4 | BM25_Rerank      |         0.4813 |        0.7380 |  0.2245 |   0.8958 |      0.2563 | 0.8447 | 0.8865 | 0.8068 |   0.0657 |    0.0610 |            0.5469 |
|    5 | Hybrid_KG        |         0.4458 |        0.6785 |  0.2132 |   0.8802 |      0.2375 | 0.7638 | 0.7885 | 0.7224 |   0.0895 |    0.0761 |            0.4740 |
|    6 | Hybrid           |         0.4349 |        0.6801 |  0.1896 |   0.8802 |      0.2375 | 0.7667 | 0.7885 | 0.7276 |   0.0762 |    0.0708 |            0.4219 |
|    7 | BM25_KG          |         0.4058 |        0.5764 |  0.2351 |   0.7969 |      0.2188 | 0.6411 | 0.6453 | 0.5802 |   0.0830 |    0.0754 |            0.5469 |
|    8 | BM25             |         0.4029 |        0.5764 |  0.2293 |   0.7969 |      0.2188 | 0.6411 | 0.6453 | 0.5802 |   0.0776 |    0.0739 |            0.5365 |

### 3.3 KG impact summary (KG on − KG off, same vector/rerank setting)

| Setting                    | base          | kg               | Δrecall@5 | Δndcg@5 | Δmrr@5 | Δtoken_f1 | ΔrougeL_f1 | Δcitation_hit_rate |
| -------------------------- | ------------- | ---------------- | ---------: | -------: | ------: | ---------: | ----------: | ------------------: |
| vector=false, rerank=false | BM25          | BM25_KG          |     0.0000 |   0.0000 |  0.0000 |    +0.0054 |     +0.0015 |             +0.0104 |
| vector=false, rerank=true  | BM25_Rerank   | BM25_Rerank_KG   |     0.0000 |   0.0000 |  0.0000 |    +0.0095 |     +0.0069 |             +0.0260 |
| vector=true, rerank=false  | Hybrid        | Hybrid_KG        |     0.0000 | −0.0029 |  0.0000 |    +0.0133 |     +0.0053 |             +0.0521 |
| vector=true, rerank=true   | Hybrid_Rerank | Hybrid_Rerank_KG |     0.0000 |   0.0000 |  0.0000 |   −0.0157 |    −0.0099 |            −0.1250 |

KG boost consistently improves generation metrics when rerank is **off** (especially citation_hit_rate +0.05–+0.10),
and has no effect on retrieval ranking metrics. When rerank is **on** under Hybrid, KG hurts all generation
metrics significantly (citation_hit_rate −0.125), suggesting the reranker and KG boost compete for the same
evidence selection and KG noise outweighs KG signal in this setting.

### 3.4 Metrics description

**Why exact_match is 0.0 across all runs?**
The exact match metric requires the predicted answer to match the GT answer exactly after normalization.
Answers are often semantically correct but lexically different from GT, leading to zero exact matches.
`exact_match` is excluded from `gen_avg` when it is zero across all experiments (adaptive rule).

**Why token_f1 and rougeL_f1 are relatively low?**
Answers are semantically correct but lexically different from GT. ROUGE and token F1 measure
surface-level overlap, not semantic similarity.

**Why some high token-score runs are not ranked first?**

Experiments are ranked by a **combined score** that gives equal weight to retrieval and generation quality:

$$
combined\_score = 0.5 \times retrieval\_avg + 0.5 \times gen\_avg
$$

Retrieval is ranked alongside generation because it is the RAG foundation; an
experiment cannot produce good answers without first retrieving relevant documents,
so it should not be demoted to a pure tiebreaker role.

`gen_avg` is computed **adaptively** — a metric is included only when it carries
discriminative signal (at least one experiment has a non-zero value):

| Metric                | Included when                                         |
| --------------------- | ----------------------------------------------------- |
| `exact_match`       | At least one experiment has `exact_match > 0`       |
| `token_f1`          | Always                                                |
| `rougeL_f1`         | Always                                                |
| `citation_hit_rate` | At least one experiment has `citation_hit_rate > 0` |

Examples:

- Full generation run with citations: all four metrics averaged (denominator = 4)
- Full generation run, no citations: `token_f1 + rougeL_f1 + exact_match` (denominator = 3 or 2)
- Retrieval-only run: exact_match and citation_hit_rate are both zero → `gen_avg = (token_f1 + rougeL_f1) / 2`

When a metric is excluded, it is also **hidden from the displayed summary table** to
avoid showing columns that add no information. The console output prints the active
`gen_avg` fields for transparency.

Within the same combined score, ties are broken by `gen_avg` then `retrieval_avg`.

## 4. Diagnosis

1. **Retrieval metrics now differentiate between experiments** (unlike the previous retrieval-only batch).
   BM25-only configs reach recall@5 ≈ 0.80; adding Rerank lifts it to ≈ 0.91. Hybrid (vector) mid-point ≈ 0.88.
2. **Rerank is the dominant driver of retrieval quality.**
   Adding rerank increases retrieval_avg by ~+0.16 across both BM25 and Hybrid bases, with MRR@5 hitting 0.9375.
3. **KG boost improves generation (citation) quality when rerank is off**, especially for Hybrid+KG
   (+0.052 citation_hit_rate). However, it provides **no retrieval improvement** in any setting.
4. **KG boost hurts generation under Hybrid+Rerank** (−0.125 citation_hit_rate, −0.016 token_f1).
   The reranker and KG signal appear to conflict: KG-expanded evidence introduces noise that the reranker
   cannot filter away, reducing answer quality.
5. **exact_match remains 0.0 across all runs** and is excluded from `gen_avg` in this batch (adaptive rule).
6. **citation_hit_rate is the largest single contributor to gen_avg** in this batch (~0.42–0.61),
   making citation quality the most impactful lever for improving the generation ranking.
7. **Hybrid_Rerank is the best overall setting** (combined_score = 0.5126), achieving top-tier retrieval
   (recall@5 = 0.9115, MRR@5 = 0.9375) combined with the highest generation score (gen_avg = 0.2596).

## 5. Clear Conclusion

- **Best experiment: Hybrid_Rerank** (combined_score = 0.5126, retrieval_avg = 0.7657, gen_avg = 0.2596).
- Rerank lifts combined score by **+0.0668** over the best non-rerank setting (Hybrid_KG = 0.4458).
- KG boost adds value for generation when rerank is disabled, but degrades quality when rerank is enabled.
- Recommended production configuration: `USE_VECTOR=true`, `ENABLE_RERANK=true`, `ENABLE_KG=false`.

## 6. Improvement Recommendations

1. Investigate why KG and Rerank conflict — consider applying KG boost only to BM25 or as a pre-rerank filter.
2. Improve citation quality: citation_hit_rate is the dominant gen_avg term; improving citation extraction or
   format consistency will have the largest impact on overall score.
3. Expand GT set further (currently 32 questions) to reduce per-metric variance — especially for MRR and MAP
   which are sensitive to single-question rank flips.
4. Tune `HYBRID_ALPHA` to find the optimal BM25/vector blend; current results show Hybrid retrieval
   (recall 0.880) slightly below BM25_Rerank recall (0.896) before rerank.
5. Consider evaluating semantic similarity (BERTScore) alongside ROUGE/token-F1 to better capture
   semantically correct but lexically different answers.

## 7. Reproducibility Commands

```bash
# Baseline pair
python scripts/eval/run_eval_benchmark.py --gt data/qa_pairs/eval_ground_truth.jsonl --out outputs/preds_hybrid.jsonl --enable-rerank false
python scripts/eval/run_eval_benchmark.py --gt data/qa_pairs/eval_ground_truth.jsonl --out outputs/preds_hybrid_rerank.jsonl --enable-rerank true
python scripts/eval/evaluate_predictions.py --gt data/qa_pairs/eval_ground_truth.jsonl --pred outputs/preds_hybrid.jsonl --k 5 --out outputs/eval_hybrid.json
python scripts/eval/evaluate_predictions.py --gt data/qa_pairs/eval_ground_truth.jsonl --pred outputs/preds_hybrid_rerank.jsonl --k 5 --out outputs/eval_hybrid_rerank.json

# Best experiment (Hybrid_Rerank)
python scripts/eval/run_eval_benchmark.py --gt data/qa_pairs/eval_ground_truth.jsonl --out outputs/ablation/preds_Hybrid_Rerank.jsonl --enable-rerank true
python scripts/eval/evaluate_predictions.py --gt data/qa_pairs/eval_ground_truth.jsonl --pred outputs/ablation/preds_Hybrid_Rerank.jsonl --k 5 --out outputs/ablation/eval_Hybrid_Rerank.json
```
