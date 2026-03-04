"""
Hybrid retriever combining BM25 keyword search and vector similarity search.
"""

import logging
from typing import List, Dict, Optional
from abc import ABC, abstractmethod
from pathlib import Path
import re
import json
import numpy as np

try:
    from rank_bm25 import BM25Okapi
    _HAS_BM25 = True
except Exception:
    _HAS_BM25 = False

from src.services.embedding_service import EmbeddingService

logger = logging.getLogger(__name__)


class HybridRetriever:
    """
    Hybrid retriever combining BM25 and vector search.
    """
    
    def __init__(self, config: Dict):
        """Initialize hybrid retriever with configuration."""
        self.config = config
        self.documents = self._load_documents()

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
    
    def retrieve(self, query: str, 
                top_k: int = 5,
                query_type: str = 'general') -> List[Dict]:
        """
        Retrieve documents using hybrid approach.
        """
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
        results = self._topk_by_scores(hybrid_scores, top_k)

        logger.info(f"Retrieved {len(results)} documents for query (hybrid)")
        return results
    
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
        return [self.documents[i] for i in idx]

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
