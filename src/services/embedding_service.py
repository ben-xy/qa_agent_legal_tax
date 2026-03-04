"""
Embedding service for generating document and query embeddings.
"""

import logging
from typing import Any, Dict, List
import numpy as np

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Service for generating embeddings using OpenAI or Gemini API."""
    
    def __init__(self, config: dict):
        """
        Initialize embedding service.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.provider = self._config_get('embedding_provider', 'gemini').lower()
        self.model = self._config_get('embedding_model', 'text-embedding-3-small')
        self.dimension = int(self._config_get('embedding_dimension', 1536))
        self.client = None

        self._init_client()

    def _config_get(self, key: str, default: Any = None) -> Any:
        """Get config value with lowercase + uppercase compatibility."""
        return self.config.get(key, self.config.get(key.upper(), default))

    def _init_client(self) -> None:
        """Initialize provider-specific embedding client."""
        if self.provider == 'openai':
            try:
                import openai

                api_key = self._config_get('openai_api_key')
                if not api_key:
                    raise ValueError("OPENAI_API_KEY is required when EMBEDDING_PROVIDER=openai")

                self.client = openai.OpenAI(api_key=api_key)
            except ImportError as exc:
                logger.error("OpenAI package not installed")
                raise ImportError("Please install openai package") from exc

        elif self.provider == 'gemini':
            try:
                from google import genai

                api_key = self._config_get('google_api_key')
                if not api_key:
                    raise ValueError("GOOGLE_API_KEY is required when EMBEDDING_PROVIDER=gemini")

                self.client = genai.Client(api_key=api_key)
                self.model = self._config_get('gemini_embedding_model', 'models/text-embedding-004')
            except ImportError as exc:
                logger.error("google-genai package not installed")
                raise ImportError("Please install google-genai package") from exc
        else:
            raise ValueError(f"Unsupported embedding provider: {self.provider}")
    
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
            embedding = self._embed_openai(text) if self.provider == 'openai' else self._embed_gemini(text)
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
                if self.provider == 'openai':
                    response = self.client.embeddings.create(
                        model=self.model,
                        input=batch
                    )
                    batch_embeddings = [item.embedding for item in response.data]
                else:
                    batch_embeddings = [self._embed_gemini(text) for text in batch]
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

    def _embed_openai(self, text: str) -> List[float]:
        """Generate embedding using OpenAI API."""
        response = self.client.embeddings.create(
            model=self.model,
            input=text
        )
        return response.data[0].embedding

    def _embed_gemini(self, text: str) -> List[float]:
        """Generate embedding using Gemini API."""
        response = self.client.models.embed_content(
            model=self.model,
            contents=text
        )
        return response.embeddings[0].values
