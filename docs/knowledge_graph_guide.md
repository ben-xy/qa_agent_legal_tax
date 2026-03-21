# Knowledge Graph Guide

## 1. Scope

This document explains how the project implements Knowledge Graph (KG) support for retrieval and what practical impact it has on answer quality.

Code basis:
- src/graph/simple_kg.py
- src/retrievers/hybrid_retriever.py
- src/agents/qa_agent.py
- config.py

## 2. What the KG Is in This Project

This project uses a lightweight, retrieval-time legal KG:

- It is not an external graph database.
- It is built in memory from chunked legal documents at retriever initialization.
- It focuses on legal entities (Act names and Section references).
- It is used only to bias candidate ranking before optional rerank.

In short, KG here is a pragmatic ranking signal, not a full symbolic reasoning engine.

## 3. KG Data Model and Construction

Implementation lives in src/graph/simple_kg.py as SimpleLegalKG.

### 3.1 Node and edge semantics

- Nodes: normalized legal entities extracted from chunk text.
  - Example entity types:
    - Act names, e.g. Income Tax Act
    - Section references, e.g. Section 10(1), s. 10(1)
- Edges: undirected co-occurrence links between entities appearing in the same chunk.

### 3.2 Core internal structures

SimpleLegalKG stores two maps:

- entity_to_doc_ids: entity -> set of chunk IDs where the entity appears
- entity_graph: entity -> set of one-hop neighboring entities

Chunk identity is aligned with retriever-level _doc_id.

### 3.3 Entity extraction rules

Entity extraction is regex-based:

- Act pattern:\
  \b([A-Z][A-Za-z&\-\s]{2,80} Act)\b
- Section pattern (case-insensitive):\
  \b(?:section|s\.)\s*\d+[A-Za-z]?(?:\(\d+\))?\b

All extracted entities are lowercased before indexing.

## 4. Query-time KG Expansion

At query time, KG expansion is one-hop:

1. Extract seed entities from query.
2. Add one-hop neighbors from entity_graph.
3. Collect doc IDs from entity_to_doc_ids.
4. Truncate to KG_MAX_EXPANSION.

This returns a set of expanded_doc_ids used for ranking bias.

## 5. How KG Is Applied in Retrieval

Integration point is src/retrievers/hybrid_retriever.py.

### 5.1 Pipeline order

During retrieve(query, top_k):

1. Stage-1 hybrid retrieval computes candidate list and retrieval_score.
2. If ENABLE_KG=true, _apply_kg_boost runs before rerank.
3. If ENABLE_RERANK=true, reranker reorders the KG-adjusted candidates.

### 5.2 Boost formula

For each candidate chunk d:

s_boosted(d, q) = s_stage1(d, q) + KG_BOOST_WEIGHT * I[d in expanded_doc_ids]

Where:
- I[.] is 1 if the candidate chunk ID is in expanded_doc_ids, else 0.
- KG_BOOST_WEIGHT defaults to 0.2.

This means KG contributes a fixed additive bonus, not a learned score.

## 6. Configuration Controls

Defined in config.py:

- ENABLE_KG (default false): turn KG boost on/off
- KG_BOOST_WEIGHT (default 0.2): additive boost size
- KG_MAX_EXPANSION (default 50): max expanded doc IDs from KG

Related toggles often used together:

- ENABLE_RERANK
- RERANK_CANDIDATE_K
- USE_BM25
- USE_VECTOR

## 7. Role and Observed Effect

### 7.1 Functional role

KG helps retrieval in entity-heavy legal queries by increasing rank of chunks connected to query entities, especially where lexical/embedding signals are ambiguous.

### 7.2 Effect in latest evaluation

From the latest ablation results (see docs/metrics_eval_report.md):

- KG generally improves generation metrics when rerank is disabled.
- KG has near-zero impact on retrieval metrics in those paired settings.
- Under Hybrid + Rerank, KG can reduce generation quality (notably citation_hit_rate), indicating interaction/tension between KG bias and rerank selection.

Operational takeaway:

- Treat KG as an optional retrieval prior.
- Validate KG with rerank on your target query mix before enabling in production.

## 8. Strengths and Limitations

### Strengths

- Simple, fast, and dependency-light.
- Works directly on existing chunk corpus.
- Easy to tune with two knobs (weight + expansion cap).

### Limitations

- Regex entity extraction can miss variants and noisy formatting.
- One-hop graph is shallow; no multi-hop reasoning.
- Fixed boost is coarse and not query-adaptive.
- No edge weighting by evidence strength or frequency.

## 9. Suggested Improvements

1. Replace pure regex extraction with hybrid NER + legal dictionary normalization.
2. Add edge weights (e.g., PMI/co-occurrence counts) and weighted expansion.
3. Make KG boost query-adaptive (e.g., stronger boost only for entity-heavy queries).
4. Add conflict handling with rerank (e.g., cap KG-boosted items entering rerank top-N).
5. Log per-query KG hit diagnostics for easier error analysis.

## 10. Quick Enable/Disable Example

In .env:

ENABLE_KG=true
KG_BOOST_WEIGHT=0.2
KG_MAX_EXPANSION=50

For A/B testing with rerank:

- Variant A: ENABLE_KG=false, ENABLE_RERANK=true
- Variant B: ENABLE_KG=true, ENABLE_RERANK=true

Compare retrieval and generation metrics using scripts/eval/eval_metrics_pipeline.ipynb.
