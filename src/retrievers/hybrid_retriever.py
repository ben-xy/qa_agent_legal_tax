"""
Hybrid retriever combining BM25 keyword search and vector similarity search.
"""

import logging
from typing import List, Dict, Optional
from abc import ABC, abstractmethod
import json
from pathlib import Path
import numpy as np

logger = logging.getLogger(__name__)


class HybridRetriever:
    """
    Hybrid retriever combining BM25 and vector search.
    """
    
    def __init__(self, config: Dict):
        """Initialize hybrid retriever with configuration."""
        self.config = config
        self.documents = self._load_documents()
        
        logger.info(f"HybridRetriever initialized with {len(self.documents)} documents")
    
    def _load_documents(self) -> List[Dict]:
        """Load documents from chunked acts."""
        documents = []
        acts_dir = Path(self.config.get('acts_chunked_dir', 'data/acts_chunked'))
        
        if not acts_dir.exists():
            logger.warning(f"Acts directory not found: {acts_dir}")
            return documents
        
        for file_path in sorted(acts_dir.glob('*.json')):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                if isinstance(data, list):
                    documents.extend(data)
                elif isinstance(data, dict):
                    documents.append(data)
                    
            except Exception as e:
                logger.error(f"Error loading document {file_path}: {e}")
                continue
        
        logger.info(f"Loaded {len(documents)} document chunks")
        return documents
    
    def retrieve(self, query: str, 
                top_k: int = 5,
                query_type: str = 'general') -> List[Dict]:
        """
        Retrieve documents using hybrid approach.
        
        Args:
            query: User query
            top_k: Number of top documents to return
            query_type: Type of query
        
        Returns:
            List of retrieved documents
        """
        if not self.documents:
            logger.warning("No documents available for retrieval")
            return []
        
        # Simple keyword-based retrieval for now
        results = self._keyword_search(query, top_k)
        
        logger.info(f"Retrieved {len(results)} documents for query")
        return results
    
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
