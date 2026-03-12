"""
Embedding service for generating document and query embeddings.
"""

import logging
from typing import Any, List, Optional, Tuple
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
        self._gemini_types = None
        self._gemini_api_version = None

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
                from google.genai import types as genai_types

                api_key = self._config_get('google_api_key')
                if not api_key:
                    raise ValueError("GOOGLE_API_KEY is required when EMBEDDING_PROVIDER=gemini")

                self._gemini_types = genai_types
                configured_model = self._config_get('gemini_embedding_model', 'gemini-embedding-001')
                configured_api_version = self._config_get('gemini_api_version')

                self.client, self.model, self._gemini_api_version = self._resolve_gemini_model(
                    genai=genai,
                    genai_types=genai_types,
                    api_key=api_key,
                    configured_model=configured_model,
                    configured_api_version=configured_api_version,
                )

                logger.info(
                    "Using Gemini embedding model %s (api_version=%s, dimension=%s)",
                    self.model,
                    self._gemini_api_version or 'default',
                    self.dimension,
                )
            except ImportError as exc:
                logger.error("google-genai package not installed")
                raise ImportError("Please install google-genai package") from exc
        else:
            raise ValueError(f"Unsupported embedding provider: {self.provider}")

    def _normalize_gemini_model_name(self, model_name: Optional[str]) -> str:
        model_name = (model_name or '').strip()
        if model_name.startswith('models/'):
            model_name = model_name[len('models/'):]
        return model_name

    def _unique_values(self, values: List[Optional[str]]) -> List[Optional[str]]:
        seen = set()
        unique: List[Optional[str]] = []
        for value in values:
            key = value if value else '__default__'
            if key in seen:
                continue
            seen.add(key)
            unique.append(value)
        return unique

    def _build_gemini_client(self, genai, genai_types, api_key: str, api_version: Optional[str]):
        kwargs = {'api_key': api_key}
        if api_version:
            kwargs['http_options'] = genai_types.HttpOptions(api_version=api_version)
        return genai.Client(**kwargs)

    def _probe_gemini_model(self, client, genai_types, model_name: str) -> None:
        client.models.embed_content(
            model=model_name,
            contents='embedding probe',
            config=genai_types.EmbedContentConfig(output_dimensionality=self.dimension),
        )

    def _resolve_gemini_model(
        self,
        genai,
        genai_types,
        api_key: str,
        configured_model: Optional[str],
        configured_api_version: Optional[str],
    ) -> Tuple[Any, str, Optional[str]]:
        candidate_models = self._unique_values([
            self._normalize_gemini_model_name(configured_model),
            'gemini-embedding-001',
            'gemini-embedding-2-preview',
        ])
        candidate_versions = self._unique_values([
            configured_api_version,
            None,
            'v1beta',
            'v1',
        ])

        errors: List[str] = []
        for api_version in candidate_versions:
            client = self._build_gemini_client(genai, genai_types, api_key, api_version)
            for model_name in candidate_models:
                if not model_name:
                    continue
                try:
                    self._probe_gemini_model(client, genai_types, model_name)
                    return client, model_name, api_version
                except Exception as exc:
                    errors.append(f"model={model_name}, api_version={api_version or 'default'} -> {exc}")

        error_summary = '; '.join(errors[:6])
        raise RuntimeError(
            'No working Gemini embedding model found for the configured API key. '
            f'Tried models {candidate_models} across API versions {candidate_versions}. '
            f'Sample errors: {error_summary}'
        )
    
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
                    batch_embeddings = self._embed_gemini_batch(batch, task_type='RETRIEVAL_DOCUMENT')
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

    def _embed_gemini(self, text: str, task_type: Optional[str] = None) -> List[float]:
        """Generate embedding using Gemini API."""
        config = self._gemini_types.EmbedContentConfig(
            output_dimensionality=self.dimension,
            task_type=task_type,
        )
        response = self.client.models.embed_content(
            model=self.model,
            contents=text,
            config=config,
        )
        return response.embeddings[0].values

    def _embed_gemini_batch(self, texts: List[str], task_type: Optional[str] = None) -> List[List[float]]:
        """Generate embeddings for a batch of texts using Gemini API."""
        config = self._gemini_types.EmbedContentConfig(
            output_dimensionality=self.dimension,
            task_type=task_type,
        )
        response = self.client.models.embed_content(
            model=self.model,
            contents=texts,
            config=config,
        )
        return [item.values for item in response.embeddings]
