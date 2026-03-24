# Response Guide

## 1. Scope

This document explains three user-facing fields shown in the CLI output:

- `Confidence Score`
- `Sources`
- `Legal Citations`

It describes what each field means, how it is currently computed, and what its limitations are in the current implementation.

Code basis:

- `main.py`
- `src/agents/qa_agent.py`
- `src/retrievers/hybrid_retriever.py`
- `src/services/llm_service.py`

## 2. Where These Fields Appear

The CLI prints these fields after the generated answer:

- `Confidence Score`: shown as a percentage
- `Confidence Breakdown`: shown as retrieval quality, answer consistency, citation coverage, and final weighted score
- `Sources`: shown as a bullet list
- `Legal Citations`: shown as a bullet list

These values are stored in the `AgentResponse` object and then rendered by `print_response()` in `main.py`.

## 3. Confidence Score

### 3.1 What it means

`Confidence Score` is a grounded confidence estimate that measures how well the answer is supported by the current pipeline output.

It is not a calibrated probability, and it should not be interpreted as:

- the probability that the answer is legally correct, or
- a formal uncertainty estimate from the language model.

Instead, it combines three observable signals:

- retrieval quality
- answer consistency with retrieved context
- citation coverage over the retrieved support set

### 3.2 Current algorithm

The score is computed in `QAAgent._calculate_confidence()`.

Formula:

$$
confidence = \frac{w_r \cdot retrieval\_quality + w_a \cdot answer\_consistency + w_c \cdot citation\_coverage}{w_r + w_a + w_c}
$$

Default weights:

- `w_r = 0.35`
- `w_a = 0.40`
- `w_c = 0.25`

These weights are now configurable through the application configuration:

- `CONFIDENCE_RETRIEVAL_WEIGHT`
- `CONFIDENCE_CONSISTENCY_WEIGHT`
- `CONFIDENCE_CITATION_WEIGHT`

The final score is clipped to the range `[0.0, 1.0]`.

Component definitions:

- `retrieval_quality`
   - measures whether enough documents were retrieved
   - checks whether documents have usable source labels and non-empty content
   - uses available retrieval scores such as `rerank_score`, `retrieval_score`, or `score`
- `answer_consistency`
   - checks whether answer sentences are supported by specific retrieved chunks
   - measures lexical overlap between each answer sentence and the best matching chunk
   - rewards a higher fraction of answer sentences that have strong chunk-level support
   - checks whether sentences containing citations are supported by chunks that also support those citations
   - checks whether the answer addresses the query terms
   - incorporates the validator confidence when available
   - applies a penalty if the answer contains uncertainty markers such as `might`, `may`, `possibly`, or `unclear`
- `citation_coverage`
   - measures how many extracted citations are supported by the retrieved documents
   - checks section identifiers against retrieved content
   - checks Act or Regulation names against retrieved content and source labels
   - assigns a low fallback score when no citations are present

### 3.3 Interpretation example

Example:

- `retrieval_quality = 0.80`
- `answer_consistency = 0.72`
- `citation_coverage = 0.60`

Then:

$$
confidence = 0.35 \cdot 0.80 + 0.40 \cdot 0.72 + 0.25 \cdot 0.60 = 0.718
$$

So the CLI would show approximately:

- `Confidence Score: 71.8%`

This makes the score more sensitive to grounding quality than the previous fixed heuristic.

### 3.4 Limitations

- It is still a heuristic aggregation, not a calibrated probability.
- Lexical overlap is only a proxy for semantic support.
- Chunk-level support still relies on lightweight token overlap rather than full semantic entailment.
- Citation coverage depends on regex-based citation extraction and lightweight support checks.
- Retrieval score scales may differ across BM25, hybrid retrieval, and reranking settings.

### 3.5 Practical takeaway

Treat `Confidence Score` as a grounded support indicator, not as proof of legal correctness.

## 4. Sources

### 4.1 What it means

`Sources` lists the document labels of the retrieved chunks used as context for answer generation.

This field answers the question:

- Which legal documents did the retriever provide to the model?

It does not necessarily mean:

- every listed source was used equally, or
- every source is explicitly quoted in the final answer.

### 4.2 Data flow

The source labels are produced in two stages:

1. `HybridRetriever` loads document chunks from `data/acts_chunked/`.
2. Each chunk is normalized so downstream code can rely on a standard `source` field.
3. `QAAgent.process_query()` converts the retrieved documents into the `sources` list.
4. `main.py` prints the set of unique source labels.

### 4.3 Current source extraction logic

Source labels are derived with the following priority:

1. `doc["source"]`
2. `doc["metadata"]["Law"]`
3. `doc["metadata"]["source"]`
4. `doc["metadata"]["title"]`
5. stem of `doc["_source_file"]`
6. fallback to `Unknown`

This means the system tries to display a human-readable legal document name whenever possible, such as:

- `Income Tax Act 1947`
- `Goods and Services Tax Act 1993`

### 4.4 Why duplicates may disappear

The CLI prints `set(response.sources)`, so repeated source names are collapsed into unique values.

This is convenient for display, but it also means:

- repeated evidence from the same Act is not shown multiple times, and
- the original retrieval order is not guaranteed in the printed list.

### 4.5 Limitations

- `Sources` reflects retrieved context, not verified citation grounding.
- It is chunk-level retrieval summarized into document labels, so section-level precision may be lost.
- If document metadata is incomplete, the system may still fall back to `Unknown`.

## 5. Legal Citations

### 5.1 What it means

`Legal Citations` is a post-processing field extracted from the generated answer text.

This field answers the question:

- Which legal references does the answer explicitly mention?

It is not taken directly from the retriever output. Instead, it is extracted after generation by scanning the answer text with pattern matching.

### 5.2 Current extraction algorithm

Citation extraction is implemented in `QAAgent._extract_citations()`.

The system searches for three categories of references:

1. Section references with an explicit Act or Regulation context
   - Example: `Section 43 of the Income Tax Act 1947`
2. Full Act or Regulation titles with year
   - Example: `Income Tax Act 1947`
3. Standalone section references
   - Example: `Section 39`

The extraction order is important. The implementation first looks for the most specific patterns, then shorter ones.

### 5.3 Filtering and deduplication

After raw matches are collected, the system applies additional filtering:

- normalize repeated whitespace
- strip trailing punctuation
- drop obvious sentence fragments such as `of the ...`
- sort matches by length in descending order
- keep the most specific citation when a shorter citation is fully contained in a longer one

For example:

- keep `Section 13K of the Income Tax Act 1947`
- drop `Income Tax Act 1947` if it is already contained inside the longer citation
- drop `Section 13K` if the longer section-plus-Act form already exists

### 5.4 Why this field can still be imperfect

Because citation extraction is regex-based and runs on generated text, it may still have failure modes such as:

- missing a valid citation that is phrased in an unexpected way
- extracting a standalone section number without the full Act context
- reflecting what the model wrote, even if the citation is incomplete or overly broad

In other words, `Legal Citations` is best understood as answer-text citation extraction, not formal legal parsing.

### 5.5 Practical takeaway

Use `Legal Citations` as a readable summary of legal references mentioned in the answer, but verify important references against the retrieved source text before relying on them in a legal or compliance setting.

## 6. Summary Table

| Field                | What it represents                            | Current method                                                             | Main limitation                                           |
| -------------------- | --------------------------------------------- | -------------------------------------------------------------------------- | --------------------------------------------------------- |
| `Confidence Score` | Grounded support confidence of the pipeline output | Weighted combination of retrieval quality, answer consistency, and citation coverage | Not calibrated; still heuristic rather than formal verification |
| `Sources`          | Retrieved document labels used as context     | Metadata-based source label extraction from retrieved chunks               | Shows retrieved context, not verified support             |
| `Legal Citations`  | Legal references mentioned in the answer text | Regex extraction plus filtering and deduplication                          | Depends on model wording; may miss or simplify references |

## 7. Suggested Future Improvements

1. Improve answer consistency from token-overlap chunk support to semantic entailment or answer-to-context verification.
2. Preserve source order and chunk rank when printing `Sources`.
3. Link `Legal Citations` back to specific supporting chunks instead of extracting only from the generated answer.
4. Add stricter citation validation to check whether each cited section or Act is actually supported by the retrieved context.
5. Introduce structured citation parsing for Singapore legal references to reduce regex ambiguity.
