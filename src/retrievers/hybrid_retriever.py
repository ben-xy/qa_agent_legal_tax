"""
Hybrid retriever combining BM25 keyword search and vector similarity search.
"""

import logging
from typing import List, Dict
from pathlib import Path
import re
import json
import numpy as np
import difflib

try:
    from rank_bm25 import BM25Okapi
    _HAS_BM25 = True
except Exception:
    _HAS_BM25 = False

from src.services.embedding_service import EmbeddingService
from src.retrievers.cohere_reranker import CohereReranker
from config import as_bool
from src.graph.simple_kg import SimpleLegalKG

logger = logging.getLogger(__name__)

class HybridRetriever:
    """
    Hybrid retriever combining BM25 and vector search.
    """
    
    def __init__(self, config: Dict):
        """Initialize hybrid retriever with configuration."""
        self.config = config
        self.enable_rerank = as_bool(self.config.get("ENABLE_RERANK", True), True)
        self.rerank_candidate_k = int(self.config.get("RERANK_CANDIDATE_K", 20))
        self.reranker = CohereReranker(
            api_key=self.config.get("COHERE_API_KEY"),
            model=self.config.get("COHERE_RERANK_MODEL", "rerank-english-v3.0"),
        )
        self.rerank_debug_log = as_bool(self.config.get("RERANK_DEBUG_LOG", False), False)
        self.documents = self._load_documents()

        # Ensure stable internal IDs for KG/retrieval alignment
        for i, d in enumerate(self.documents):
            d.setdefault("_doc_id", str(d.get("id") or f"doc_{i}"))

        # KG settings
        self.enable_kg = bool(self.config.get("ENABLE_KG", False))
        self.kg_boost_weight = float(self.config.get("KG_BOOST_WEIGHT", 0.2))
        self.kg_max_expansion = int(self.config.get("KG_MAX_EXPANSION", 50))
        self.kg = None
        if self.enable_kg:
            try:
                self.kg = SimpleLegalKG(self.documents)
                logger.info("Knowledge Graph enabled")
            except Exception as exc:
                logger.warning("Knowledge Graph initialization failed: %s", exc)
                self.kg = None

        self._bm25 = None
        self._bm25_corpus = None
        self._embedding_service = None
        self._doc_embeddings = None

        if _HAS_BM25:
            self._prepare_bm25()

        logger.info(f"HybridRetriever initialized with {len(self.documents)} documents")

    def _load_documents(self) -> List[Dict]:
        """Load documents from chunked acts."""
        documents = []
        acts_dir = Path(self.config.get('acts_chunked_dir', 'data/acts_chunked'))
        
        if not acts_dir.exists():
            logger.warning(f"acts_chunked_dir not found: {acts_dir}")
            return documents
        
        for file_path in sorted(acts_dir.glob('*.json')):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                if isinstance(data, list):
                    for d in data:
                        d["_source_file"] = str(file_path)
                    documents.extend(data)
                elif isinstance(data, dict):
                    data["_source_file"] = str(file_path)
                    documents.append(data)
            except Exception as e:
                logger.warning(f"Failed to load {file_path}: {e}")
        
        logger.info(f"Loaded {len(documents)} document chunks")
        return documents
    
    def _short_text(self, text: str, limit: int = 220) -> str:
        text = (text or "").replace("\n", " ").strip()
        return text if len(text) <= limit else text[:limit] + "..."

    def _doc_signature(self, doc: Dict) -> str:
        # Stable-enough signature for ranking comparison logs
        text = self._get_doc_text(doc) if hasattr(self, "_get_doc_text") else (
            doc.get("content") or doc.get("text") or doc.get("chunk") or str(doc)
        )
        return self._short_text(text, 120)

    def _highlight_diff(self, before: str, after: str) -> tuple[str, str]:
        """
        Highlight token-level differences:
        - removed tokens in before: [-token-]
        - added tokens in after: {+token+}
        """
        b_tokens = before.split()
        a_tokens = after.split()
        sm = difflib.SequenceMatcher(a=b_tokens, b=a_tokens)

        b_out, a_out = [], []
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            b_seg = b_tokens[i1:i2]
            a_seg = a_tokens[j1:j2]

            if tag == "equal":
                b_out.extend(b_seg)
                a_out.extend(a_seg)
            elif tag == "replace":
                b_out.extend([f"[-{t}-]" for t in b_seg])
                a_out.extend([f"{{+{t}+}}" for t in a_seg])
            elif tag == "delete":
                b_out.extend([f"[-{t}-]" for t in b_seg])
            elif tag == "insert":
                a_out.extend([f"{{+{t}+}}" for t in a_seg])

        return " ".join(b_out), " ".join(a_out)

    def _log_rerank_comparison(self, query: str, before_docs: List[Dict], after_docs: List[Dict], top_n: int = 5) -> None:
        n = min(top_n, len(before_docs), len(after_docs))
        if n == 0:
            return

        logger.info("=== RERANK DEBUG START ===")
        logger.info("Query: %s", query)

        # Rank movement summary
        before_rank = {self._doc_signature(d): i + 1 for i, d in enumerate(before_docs)}
        after_rank = {self._doc_signature(d): i + 1 for i, d in enumerate(after_docs)}
        moved = []
        for sig, b_rank in before_rank.items():
            if sig in after_rank:
                a_rank = after_rank[sig]
                if a_rank != b_rank:
                    moved.append((sig, b_rank, a_rank))
        for sig, b_rank, a_rank in moved[:10]:
            logger.info("Rank change: %d -> %d | %s", b_rank, a_rank, sig)

        logger.info("--- Top-%d BEFORE rerank ---", n)
        for i in range(n):
            d = before_docs[i]
            score = d.get("retrieval_score", d.get("score", None))
            logger.info("#%d score=%s | %s", i + 1, score, self._doc_signature(d))

        logger.info("--- Top-%d AFTER rerank ---", n)
        for i in range(n):
            d = after_docs[i]
            rscore = d.get("rerank_score", d.get("retrieval_score", d.get("score", None)))
            logger.info("#%d rerank_score=%s | %s", i + 1, rscore, self._doc_signature(d))

        logger.info("--- Highlighted differences by rank position ---")
        for i in range(n):
            b_text = self._doc_signature(before_docs[i])
            a_text = self._doc_signature(after_docs[i])
            hb, ha = self._highlight_diff(b_text, a_text)
            logger.info("Rank #%d BEFORE: %s", i + 1, hb)
            logger.info("Rank #%d AFTER : %s", i + 1, ha)

        logger.info("=== RERANK DEBUG END ===")

    def retrieve(self, query: str, 
                top_k: int = 5,
                query_type: str = 'general') -> List[Dict]:
        """
        Retrieve documents using stage-1 hybrid retrieval, then optional reranking.
        """
        logger.info(
            "retrieve() enter | docs=%d | enable_rerank=%s | has_bm25=%s",
            len(self.documents),
            self.enable_rerank,
            _HAS_BM25,
        )

        if not self.documents:
            return []

        # Always retrieve a wider candidate pool first.
        candidate_k = max(top_k, self.rerank_candidate_k)
        candidates = self._hybrid_retrieve(query=query, top_k=candidate_k, query_type=query_type)

        # Apply KG boost before rerank
        if self.enable_kg:
            candidates = self._apply_kg_boost(query=query, candidates=candidates)

        if self.enable_rerank:
            logger.info("Reranking top %d candidates with Cohere reranker", len(candidates))
            before_top = candidates[:top_k]
            reranked = self.reranker.rerank(query=query, candidates=candidates, top_n=top_k)

            if self.rerank_debug_log:
                self._log_rerank_comparison(
                    query=query,
                    before_docs=before_top,
                    after_docs=reranked,
                    top_n=min(5, top_k),
                )

            return reranked

        logger.info("Rerank disabled. Returning top %d stage-1 candidates", top_k)
        return candidates[:top_k]

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r"\w+", text.lower())
    
    def _prepare_bm25(self) -> None:
        corpus = [self._tokenize(self._get_doc_text(d)) for d in self.documents]
        self._bm25_corpus = corpus
        self._bm25 = BM25Okapi(corpus)
    
    def _get_doc_text(self, doc: Dict) -> str:
        for key in ("content", "text", "chunk", "body"):
            if key in doc and isinstance(doc[key], str):
                return doc[key]
        return str(doc)
    
    def _vector_scores(self, query: str) -> np.ndarray:
        if self._embedding_service is None:
            self._embedding_service = EmbeddingService(self.config)

        if self._doc_embeddings is None:
            embeddings = []
            texts = []
            for d in self.documents:
                if "embedding" in d and isinstance(d["embedding"], list):
                    embeddings.append(np.array(d["embedding"], dtype=np.float32))
                else:
                    texts.append(self._get_doc_text(d))

            if texts:
                new_embeds = self._embedding_service.embed_batch(texts)
                it = iter(new_embeds)
                merged = []
                for d in self.documents:
                    if "embedding" in d and isinstance(d["embedding"], list):
                        merged.append(np.array(d["embedding"], dtype=np.float32))
                    else:
                        merged.append(np.array(next(it), dtype=np.float32))
                embeddings = merged

            self._doc_embeddings = np.vstack(embeddings).astype(np.float32)

        q_embed = self._embedding_service.embed(query)
        q_embed = np.array(q_embed, dtype=np.float32)
        denom = (np.linalg.norm(self._doc_embeddings, axis=1) * np.linalg.norm(q_embed) + 1e-8)
        sims = np.dot(self._doc_embeddings, q_embed) / denom
        return sims
    
    def _minmax_norm(self, scores: np.ndarray) -> np.ndarray:
        scores = np.array(scores, dtype=np.float32)
        min_s, max_s = scores.min(), scores.max()
        if max_s - min_s < 1e-8:
            return np.zeros_like(scores)
        return (scores - min_s) / (max_s - min_s)
    
    def _topk_by_scores(self, scores: np.ndarray, top_k: int) -> List[Dict]:
        idx = np.argsort(scores)[::-1][:top_k]
        out = []
        for rank, i in enumerate(idx, start=1):
            doc = dict(self.documents[i])
            doc["retrieval_score"] = float(scores[i])
            doc["retrieval_rank"] = rank
            out.append(doc)
        return out

    def _keyword_search(self, query: str, top_k: int) -> List[Dict]:
        """Simple keyword-based search."""
        query_terms = query.lower().split()
        results = []
        
        for doc in self.documents:
            content = doc.get('content', '').lower()
            source = doc.get('source', '')
            
            # Count matching terms
            matches = sum(1 for term in query_terms if term in content)
            
            if matches > 0:
                doc_copy = doc.copy()
                doc_copy['score'] = matches / len(query_terms) if query_terms else 0
                results.append(doc_copy)
        
        # Sort by score and return top-k
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:top_k]

    def _hybrid_retrieve(self, query: str, top_k: int, **kwargs) -> List[Dict]:
        """Retrieve documents using hybrid approach."""
        if not self.documents:
            return []
        
        use_bm25 = self.config.get("use_bm25", True) and _HAS_BM25 and self._bm25 is not None
        use_vector = self.config.get("use_vector", True)

        if not use_bm25 and not use_vector:
            return self._keyword_search(query, top_k)

        bm25_scores = None
        if use_bm25:
            bm25_scores = self._bm25.get_scores(self._tokenize(query))

        vec_scores = None
        if use_vector:
            try:
                vec_scores = self._vector_scores(query)
            except Exception as e:
                logger.warning(f"Vector search failed, fallback to BM25/keyword: {e}")
                vec_scores = None

        if bm25_scores is None and vec_scores is None:
            return self._keyword_search(query, top_k)
        if bm25_scores is None:
            return self._topk_by_scores(vec_scores, top_k)
        if vec_scores is None:
            return self._topk_by_scores(bm25_scores, top_k)

        bm25_norm = self._minmax_norm(bm25_scores)
        vec_norm = self._minmax_norm(vec_scores)
        alpha = float(self.config.get("hybrid_alpha", 0.5))

        hybrid_scores = alpha * bm25_norm + (1.0 - alpha) * vec_norm
        candidates = self._topk_by_scores(hybrid_scores, top_k)

        logger.info(f"Retrieved {len(candidates)} documents for query (hybrid)")
        return candidates

    def _apply_kg_boost(self, query: str, candidates: List[Dict]) -> List[Dict]:
        """Apply lightweight KG boost to stage-1 candidates."""
        if not self.kg or not candidates:
            return candidates

        expanded_doc_ids = self.kg.expand(query=query, max_docs=self.kg_max_expansion)
        if not expanded_doc_ids:
            return candidates

        boosted = []
        for c in candidates:
            item = dict(c)
            base = float(item.get("retrieval_score", 0.0))
            doc_id = str(item.get("_doc_id") or item.get("id") or "")
            kg_hit = 1.0 if doc_id in expanded_doc_ids else 0.0
            item["kg_boost"] = kg_hit
            item["retrieval_score"] = base + (self.kg_boost_weight * kg_hit)
            boosted.append(item)

        boosted.sort(key=lambda x: x.get("retrieval_score", 0.0), reverse=True)
        return boosted
