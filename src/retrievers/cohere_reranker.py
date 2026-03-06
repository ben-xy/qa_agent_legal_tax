import os
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    import cohere
except Exception:
    cohere = None


class CohereReranker:
    def __init__(self, api_key: Optional[str] = None, model: str = "rerank-english-v3.0"):
        self.api_key = api_key or os.getenv("COHERE_API_KEY")
        self.model = model
        self.client = cohere.Client(self.api_key) if (cohere and self.api_key) else None

    @property
    def enabled(self) -> bool:
        return self.client is not None

    def rerank(self, query: str, candidates: List[Dict[str, Any]], top_n: int) -> List[Dict[str, Any]]:
        if not candidates:
            return []

        if not self.enabled:
            logger.info("Cohere rerank disabled. Returning original ranking.")
            return candidates[:top_n]

        docs = []
        for c in candidates:
            docs.append(c.get("content") or c.get("text") or c.get("chunk") or "")

        try:
            resp = self.client.rerank(
                model=self.model,
                query=query,
                documents=docs,
                top_n=min(top_n, len(docs)),
                return_documents=False,
            )
            reranked = []
            for r in resp.results:
                item = dict(candidates[r.index])
                item["rerank_score"] = float(r.relevance_score)
                reranked.append(item)
            return reranked
        except Exception as e:
            logger.warning("Cohere rerank failed. Fallback to original ranking: %s", e)
            return candidates[:top_n]