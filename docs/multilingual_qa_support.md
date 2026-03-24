# Multilingual QA Support Guide / 多语言问答支持指南

[English](#english) | [中文](#中文)

---

## English

### Overview

The QA Agent supports multilingual queries including mixed-language questions (e.g., "在新加坡如果在地铁上吃东西，what kind of fine can I get?"). The system is designed to handle Chinese, English, and code-mixed inputs natively without explicit translation preprocessing.

### Architecture

The multilingual support is achieved through a **three-layer pipeline**:

```
User Query (Any Language)
    ↓
Layer 1: Input Passthrough (No translation)
    ↓
Layer 2: Hybrid Retrieval (BM25 + Vector Search)
    ↓
Layer 3: LLM Generation (Direct multilingual understanding)
    ↓
Answer (Same language context as query)
```

### Layer 1: Input Passthrough

**What happens:**
- User queries are accepted as-is, without language detection or pre-translation.
- Both UI and CLI pass the raw query to the agent pipeline.

**Key files:**
- [ui/qa_agent_ui.ipynb](../ui/qa_agent_ui.ipynb): Gradio web UI accepts and forwards queries directly
- [main.py](../main.py): CLI entry point passes queries unchanged

**Benefit:** Preserves the user's original intent and nuance in mixed-language queries.

### Layer 2: Hybrid Retrieval

**Components:**

1. **Keyword-based Search (BM25)**
   - Located in: [src/retrievers/hybrid_retriever.py](../src/retrievers/hybrid_retriever.py), method `_hybrid_retrieve()`
   - Strength: Works well for English queries and recognizable terms
   - Weakness: Chinese tokenization is less effective with default regex-based splitting
   - Uses: `re.findall(r"\w+", text.lower())` which favors Latin characters

2. **Vector Semantic Search (Primary multilingual handler)**
   - Embedding service: [src/services/embedding_service.py](../src/services/embedding_service.py)
   - Models supported:
     - Gemini (default): `gemini-embedding-2-preview` (multilingual-capable)
     - OpenAI: `text-embedding-3-small` (multilingual-capable)
   - Task types: `RETRIEVAL_QUERY` and `RETRIEVAL_DOCUMENT` optimize embedding context
   - Mechanism: Both queries and documents are mapped to a shared semantic space, enabling cross-lingual similarity matching

3. **Fusion**
   - Combines normalized BM25 and vector scores: `hybrid_scores = α·BM25_norm + (1-α)·vec_norm`
   - Default alpha (hybrid_alpha): 0.5
   - Configuration: [config.py](../config.py) → `HYBRID_ALPHA`

**Why this works for multilingual queries:**
- Even if BM25 fails to match Chinese terms, vector search uses semantic embeddings that are language-agnostic
- Both Chinese and English content are embedded into the same dimensional space
- Cosine similarity works across languages for semantically related content

### Layer 3: LLM Generation

**Mechanism:**
- LLM services ([src/services/llm_service.py](../src/services/llm_service.py)) receive:
  - Raw query (any language)
  - Retrieved context documents (mixed language)
  - Company context (optional)
- The LLM directly processes this multilingual input

**LLM Models:**
- Gemini (default): `gemini-2.5-flash` - strong multilingual understanding
- OpenAI: `gpt-4o-mini` - strong multilingual understanding

**System Prompt:**
The prompt is language-agnostic and instructs the LLM to answer based on provided context. See [src/services/llm_service.py](../src/services/llm_service.py), method `_get_system_prompt()`.

**Output:** Answer is typically in the language context of the retrieved documents, or as understood by the LLM from the query.

### Current Capabilities

✅ **What Works Well:**
- Pure English queries (optimized by default configuration)
- Pure Chinese queries (supported by vector search and LLM understanding)
- Code-mixed queries (e.g., Chinese + English in one question)
- Queries with proper nouns and legal act names in English mixed with Chinese narrative

⚠️ **Current Limitations:**

| Limitation | Reason | Impact |
|-----------|--------|--------|
| Query Classification | Currently keyword-based on English terms only (e.g., "tax", "deadline") | Chinese tax queries may be classified as "general" instead of "tax", but this doesn't block retrieval or answering |
| BM25 Tokenization | Default regex doesn't segment Chinese properly | Chinese keyword matching is weaker; vector search compensates |
| Reranking Model | Cohere default: `rerank-english-v3.0` (English-optimized) | Mild preference for English in reranking stage; vector pre-ranking usually mitigates this |
| Confidence Scoring | Tokenization uses `\w+` (English-centric) | Chinese answers may score slightly conservative on consistency metrics |

### Performance Expectations

**English Query Example:**
```
Q: "What are the tax filing deadlines for Singapore companies?"
Retrieval: ✅ BM25 + vector search both match
Classification: "tax" (keyword matched)
Answer: High confidence, well-sourced
```

**Chinese Query Example:**
```
Q: "新加坡公司的税务申报截止日期是什么?"
Retrieval: ⚠️ BM25 struggles, ✅ vector search succeeds
Classification: "general" (no English keywords)
Answer: Still accurate, but confidence may be slightly lower than English equivalent
```

**Mixed Query Example:**
```
Q: "在新加坡如果在地铁上吃东西，what kind of fine can I get?"
Retrieval: ✅ Vector search understands both parts semantically
Classification: "general"
Answer: Accurate; LLM bridges both languages naturally
```

---

## 中文

### 概览

问答代理支持多语言查询，包括混合语言问题（例如："在新加坡如果在地铁上吃东西，what kind of fine can I get?"）。系统被设计成能够直接处理中文、英文和代码混合输入，不需要显式的翻译预处理。

### 架构

多语言支持通过**三层流水线**实现：

```
用户问题（任意语言）
    ↓
第1层：输入直通（不翻译）
    ↓
第2层：混合检索（BM25 + 向量检索）
    ↓
第3层：LLM 生成（直接多语言理解）
    ↓
答案（与问题相同的语言上下文）
```

### 第1层：输入直通

**工作原理：**
- 用户问题被原样接收，不进行语言检测或预翻译。
- UI 和 CLI 都将原始问题直接传递给代理流水线。

**关键文件：**
- [ui/qa_agent_ui.ipynb](../ui/qa_agent_ui.ipynb)：Gradio 网页 UI 直接接收并转发问题
- [main.py](../main.py)：CLI 入口点原样传递问题

**优势：** 保留用户的原始意图和混合语言问题的细微差别。

### 第2层：混合检索

**组件：**

1. **关键词检索（BM25）**
   - 位置：[src/retrievers/hybrid_retriever.py](../src/retrievers/hybrid_retriever.py)（`_hybrid_retrieve()` 方法）
   - 优势：对英文查询和可识别词汇效果好
   - 不足：中文分词效果不理想（使用了基于正则的简单分词）
   - 实现：`re.findall(r"\w+", text.lower())` 倾向于拉丁字符

2. **向量语义检索（多语言的主要支撑）**
   - Embedding 服务：[src/services/embedding_service.py](../src/services/embedding_service.py)
   - 支持的模型：
     - Gemini（默认）：`gemini-embedding-2-preview`（多语言能力强）
     - OpenAI：`text-embedding-3-small`（多语言能力强）
   - 任务类型：`RETRIEVAL_QUERY` 和 `RETRIEVAL_DOCUMENT` 优化 embedding 上下文
   - 机制：问题和文档都被映射到共享的语义空间，实现跨语言相似性匹配

3. **融合**
   - 融合归一化的 BM25 和向量分数：`hybrid_scores = α·BM25_norm + (1-α)·vec_norm`
   - 默认 alpha（hybrid_alpha）：0.5
   - 配置位置：[config.py](../config.py) → `HYBRID_ALPHA`

**为什么对多语言问题有效：**
- 即使 BM25 无法匹配中文词汇，向量检索使用的语义 embedding 对语言无偏见
- 中文和英文内容都被 embedding 到同一维度空间
- 余弦相似度可跨语言工作，用于语义相关内容

### 第3层：LLM 生成

**工作原理：**
- LLM 服务（[src/services/llm_service.py](../src/services/llm_service.py)）接收：
  - 原始问题（任意语言）
  - 检索到的背景文档（可能是混合语言）
  - 公司上下文（可选）
- LLM 直接处理这个多语言输入

**LLM 模型：**
- Gemini（默认）：`gemini-2.5-flash` - 多语言理解能力强
- OpenAI：`gpt-4o-mini` - 多语言理解能力强

**系统提示词：**
提示词与语言无关，指示 LLM 基于提供的背景回答。见 [src/services/llm_service.py](../src/services/llm_service.py)（`_get_system_prompt()` 方法）。

**输出：** 答案通常采用检索文档的语言上下文，或由 LLM 从问题中理解的语言。

### 当前能力

✅ **效果好的情况：**
- 纯英文查询（默认配置优化）
- 纯中文查询（向量检索和 LLM 理解支持）
- 代码混合查询（例如单个问题中混合中文和英文）
- 包含英文专有名词和法律法案名称混合中文叙述的查询

⚠️ **当前限制：**

| 限制 | 原因 | 影响 |
|-----|------|------|
| 问题分类 | 基于英文关键词（如 "tax", "deadline"） | 中文税务问题可能被分类为 "general" 而非 "tax"，但不影响检索和回答 |
| BM25 分词 | 默认正则无法正确分割中文 | 中文关键词匹配较弱；向量检索可补偿 |
| 重排模型 | Cohere 默认：`rerank-english-v3.0`（英文优化） | 重排阶段对英文有轻微偏好；向量预排名通常能缓解 |
| 置信度评分 | 分词使用 `\w+`（英文为中心） | 中文答案在一致性指标上可能评分略低 |

### 性能预期

**英文查询示例：**
```
问：What are the tax filing deadlines for Singapore companies?
检索：✅ BM25 和向量检索都能匹配
分类：tax（关键词匹配）
答案：高置信度，来源充分
```

**中文查询示例：**
```
问：新加坡公司的税务申报截止日期是什么?
检索：⚠️ BM25 困难，✅ 向量检索成功
分类：general（没有英文关键词）
答案：仍然准确，但置信度可能略低于英文等效问题
```

**混合查询示例：**
```
问：在新加坡如果在地铁上吃东西，what kind of fine can I get?
检索：✅ 向量检索语义理解两部分
分类：general
答案：准确；LLM 自然桥接两种语言
```

---

## Improvement Roadmap / 改进路线

### Short-term Enhancements (可快速实施)

1. **Extend Query Classification to Bilingual**
   - Add Chinese tax/compliance keywords to classifier
   - File: [src/agents/qa_agent.py](../src/agents/qa_agent.py), method `_classify_query()`
   - Impact: Better routing for pure Chinese queries

2. **Improve Chinese Tokenization in Confidence Scoring**
   - Replace `\w+` with Unicode-aware tokenizer (e.g., `jieba` or spaCy with Chinese models)
   - File: [src/agents/qa_agent.py](../src/agents/qa_agent.py), method `_tokenize_for_scoring()`
   - Impact: Fairer confidence scoring for Chinese answers

3. **Switch Rerank Model**
   - Use `rerank-multilingual` if available from Cohere, or disable rerank for Chinese queries
   - File: [src/retrievers/cohere_reranker.py](../src/retrievers/cohere_reranker.py), [config.py](../config.py)
   - Impact: Remove English bias in reranking stage

### Medium-term Enhancements (可中期优化)

4. **Query Rewrite Module**
   - For Chinese queries: Generate English reformulation, retrieve with both, combine results
   - Pros: Leverages both BM25 and LLM understanding
   - Cons: Adds latency and API calls

5. **Language-specific Embeddings**
   - Use language detection + language-specific embedding models if needed
   - Example: Chinese → `bge-base-zh`, English → `bge-base-en`
   - Impact: Potentially better semantic retrieval per language

---

## Configuration / 配置

### Current Defaults (当前默认值)

```env
# 嵌入模型
EMBEDDING_PROVIDER=gemini
GEMINI_EMBEDDING_MODEL=gemini-embedding-2-preview

# LLM
LLM_PROVIDER=gemini
GEMINI_LLM_MODEL=gemini-2.5-flash

# 检索
USE_BM25=true
USE_VECTOR=true
HYBRID_ALPHA=0.5

# 重排（可能对中文有偏见）
ENABLE_RERANK=true
COHERE_RERANK_MODEL=rerank-english-v3.0
```

### Recommended Settings for Better Multilingual Support

```env
# Option A: Minimize English bias
ENABLE_RERANK=false  # 禁用重排以避免英文偏见
HYBRID_ALPHA=0.3     # 降低 BM25 权重，提升向量检索权重

# Option B: Use multilingual-specific models if available
EMBEDDING_PROVIDER=openai
EMBEDDING_MODEL=text-embedding-3-large  # 更强的多语言能力
```

---

## Debugging Multilingual Queries / 多语言查询调试

### Enable Debug Logs

```bash
LOG_LEVEL=DEBUG
RERANK_DEBUG_LOG=true
```

### Common Issues and Solutions

| Issue | 症状 | 解决方案 |
|-------|------|--------|
| Chinese query returns no result | 纯中文问题无回答 | 检查是否启用向量检索（USE_VECTOR=true）；检查 Gemini API 配额 |
| Low confidence on Chinese answers | 中文答案置信度低 | 这是正常的，因为分词偏英文；预计中文置信度比英文低 15-25% |
| Mixed-language query confusion | 混合查询回答不准确 | 尝试禁用重排（ENABLE_RERANK=false）或增加 RETRIEVAL_TOP_K |

---

## References / 参考

- [RAG Strategies Guide](./rag_strategies_guide.md)
- [Chunking Strategies Guide](./chunking_strategies_guide.md)
- [Metrics Evaluation Report](./metrics_eval_report.md)
- [Knowledge Graph Guide](./knowledge_graph_guide.md)
