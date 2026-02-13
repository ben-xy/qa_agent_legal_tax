"""
Embedding service for generating document and query embeddings.
"""

import logging
from typing import List, Union
import numpy as np

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Service for generating embeddings using OpenAI API."""
    
    def __init__(self, config: dict):
        """
        Initialize embedding service.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.model = config.get('embedding_model', 'text-embedding-3-small')
        self.dimension = config.get('embedding_dimension', 1536)
        
        try:
            import openai
            self.client = openai.OpenAI(api_key=config.get('openai_api_key'))
        except ImportError:
            logger.error("OpenAI package not installed")
            raise
    
    def embed(self, text: str) -> np.ndarray:
        """
        Generate embedding for text.
        
        Args:
            text: Text to embed
        
        Returns:
            Embedding vector as numpy array
        """
        if not text:
            return np.zeros(self.dimension)
        
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=text
            )
            
            embedding = response.data[0].embedding
            return np.array(embedding, dtype=np.float32)
            
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise
    
    def embed_batch(self, texts: List[str], batch_size: int = 100) -> np.ndarray:
        """
        Generate embeddings for multiple texts in batches.
        
        Args:
            texts: List of texts to embed
            batch_size: Batch size for processing
        
        Returns:
            2D array of embeddings
        """
        embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            try:
                response = self.client.embeddings.create(
                    model=self.model,
                    input=batch
                )
                
                batch_embeddings = [item.embedding for item in response.data]
                embeddings.extend(batch_embeddings)
                
                logger.info(f"Embedded {i + len(batch)}/{len(texts)} texts")
                
            except Exception as e:
                logger.error(f"Error in batch embedding: {e}")
                raise
        
        return np.array(embeddings, dtype=np.float32)
    
    def similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
        
        Returns:
            Similarity score between 0 and 1
        """
        # Normalize vectors
        emb1 = embedding1 / (np.linalg.norm(embedding1) + 1e-10)
        emb2 = embedding2 / (np.linalg.norm(embedding2) + 1e-10)
        
        # Calculate cosine similarity
        similarity = np.dot(emb1, emb2)
        return float(similarity)
