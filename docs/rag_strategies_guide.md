# RAG Strategies Guide

## 1. Scope
This guide documents the currently implemented RAG strategies in this project, how each strategy works, its strengths and weaknesses, and when to use it.

For implementation details of the Knowledge Graph path specifically, see: `docs/knowledge_graph_guide.md`.

Code basis:
- `src/retrievers/hybrid_retriever.py`
- `src/retrievers/cohere_reranker.py`
- `src/graph/simple_kg.py`
- `src/agents/qa_agent.py`
- `config.py`

## 2. Retrieval Pipeline Architecture
At runtime, the QA flow is:

1. `QAAgent.process_query(...)` calls `HybridRetriever.retrieve(...)`.
2. Stage-1 retrieval computes candidates from BM25 and/or vector similarity.
3. Optional KG boost adjusts stage-1 scores.
4. Optional reranker reorders top candidates.
5. Top-k contexts are passed to the LLM for final answer generation.

Core scoring logic in stage-1 hybrid retrieval:

$$
s_{hybrid}(d, q) = \alpha \cdot \text{norm}(s_{bm25}(d, q)) + (1-\alpha) \cdot \text{norm}(s_{vec}(d, q))
$$

Optional KG-adjusted score:

$$
s_{kg}(d, q) = s_{hybrid}(d, q) + w_{kg} \cdot \mathbb{1}[d \in KG\_expand(q)]
$$

Where:
- $\alpha$ = `HYBRID_ALPHA`
- $w_{kg}$ = `KG_BOOST_WEIGHT`

## 3. Implemented Strategy Set

## Strategy A (Baseline): Hybrid Retrieval (BM25 + Vector Search)
Definition:
- Enable both lexical and semantic first-stage retrieval.
- No rerank, no KG boost.

Typical config:
- `USE_BM25=true`
- `USE_VECTOR=true`
- `ENABLE_RERANK=false`
- `ENABLE_KG=false`

Pros:
- Balanced recall across keyword-exact and semantic-similar queries.
- Lower latency and cost than adding rerank.
- Strong default when query styles are mixed.

Cons:
- Top results may still be noisy for long or ambiguous questions.
- Ranking quality is bounded by stage-1 scoring only.

Best use cases:
- General-purpose QA where response time matters.
- Early baseline experiments and ablations.

## Strategy B (Enhanced): Hybrid Retrieval + Rerank
Definition:
- Same stage-1 hybrid retrieval as Strategy A.
- Retrieve a wider candidate set (`RERANK_CANDIDATE_K`), then rerank to top-k.

Typical config:
- `USE_BM25=true`
- `USE_VECTOR=true`
- `ENABLE_RERANK=true`
- `RERANK_CANDIDATE_K=20` (or higher)
- `ENABLE_KG=false`

Pros:
- Usually better precision@k and answer grounding quality.
- Better disambiguation on multi-clause legal questions.

Cons:
- Extra latency and API cost (external rerank model).
- Quality depends on candidate recall from stage-1; rerank cannot recover missing evidence.

Best use cases:
- User-facing production answers where precision is more important than speed.
- Complex legal/tax questions with many near-duplicate chunks.

## Strategy C: Hybrid Retrieval + KG Boost
Definition:
- Stage-1 hybrid retrieval followed by lightweight KG score boost.
- No rerank.

Typical config:
- `USE_BM25=true`
- `USE_VECTOR=true`
- `ENABLE_RERANK=false`
- `ENABLE_KG=true`

How KG works here:
- Extract entities from query/chunks (Act names, section references).
- Expand one-hop neighbors in a co-occurrence graph.
- Add fixed score bonus to candidate chunks connected to expanded entities.

Pros:
- Improves retrieval for entity-centric legal queries.
- Cheap compared with full graph reasoning; simple to maintain.

Cons:
- Depends on regex-style entity extraction quality.
- Uses fixed bonus, which can over- or under-boost in some corpora.
- Not a full reasoning graph; mainly a retrieval-time bias.

Best use cases:
- Queries mentioning explicit Acts, Sections, or legal references.
- Cases where semantic embeddings alone miss sparse legal entities.

## Strategy D: Hybrid Retrieval + KG Boost + Rerank (Full Enhanced)
Definition:
- Stage-1 hybrid retrieval.
- KG boost applied before rerank.
- Final rerank to top-k.

Typical config:
- `USE_BM25=true`
- `USE_VECTOR=true`
- `ENABLE_KG=true`
- `ENABLE_RERANK=true`

Pros:
- Most expressive ranking pipeline among current options.
- KG helps candidate recall; rerank improves final ordering.

Cons:
- Highest latency and operational complexity.
- More tuning knobs (`HYBRID_ALPHA`, `KG_BOOST_WEIGHT`, candidate_k, rerank model).

Best use cases:
- High-stakes legal/tax QA where answer quality is prioritized over latency.
- Offline evaluation benchmarks and final model selection.

## 4. Additional Practical Modes (Ablation / Fallback)

## BM25-only mode
Config:
- `USE_BM25=true`
- `USE_VECTOR=false`

Use when:
- Embedding service is unavailable, rate-limited, or too expensive.
- Queries are mostly exact term/citation matching.

## Vector-only mode
Config:
- `USE_BM25=false`
- `USE_VECTOR=true`

Use when:
- Corpus language is semantically varied and keyword overlap is weak.
- You want robust semantic matching over synonyms/paraphrases.

## Keyword fallback mode
Trigger:
- Both BM25 and vector are disabled, or both fail.

Behavior:
- Falls back to a simple term-overlap keyword search.

Use when:
- Emergency fallback only; not recommended as primary strategy.

## 5. Decision Guide

| Goal | Recommended Strategy |
|---|---|
| Fast baseline with balanced quality | Strategy A |
| Better precision / less noisy top-k | Strategy B |
| Entity-heavy legal references | Strategy C |
| Best overall quality (higher latency) | Strategy D |
| Embedding outage or low budget | BM25-only mode |

## 6. Current Project Reality Check
Based on configuration and code wiring:

- Strategy A and Strategy B are fully implemented and directly supported.
- KG-enhanced strategies (C and D) are also implemented and can be toggled through config/env.
- Evaluation tooling already supports retrieval/rerank ablations and can be extended to include KG toggles.

If your local `.env` currently has `USE_VECTOR=false` with `ENABLE_RERANK=true` and `ENABLE_KG=true`, your effective runtime behavior is closer to:

- BM25-only stage-1 + KG boost + rerank

That is not pure Strategy A/B, and should be interpreted accordingly in experiment reports.

## 7. Recommended Evaluation Protocol
For fair comparison, run controlled ablations with identical data and prompts:

1. A: Hybrid (BM25+Vector), no KG, no rerank
2. B: A + rerank
3. C: A + KG
4. D: A + KG + rerank

Track at least:
- Retrieval: recall@k, precision@k, MRR, nDCG
- Generation: token F1, ROUGE-L, citation hit rate
- Cost/latency: average retrieval+rereank time per query

This makes strategy trade-offs explicit and reproducible.