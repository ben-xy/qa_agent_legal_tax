"""
Application configuration management.
Supports multiple environments: development, production, testing.
"""

import os
import json
from pathlib import Path
from typing import Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Config:
    """Base configuration class."""
    
    # Project paths
    PROJECT_ROOT = Path(__file__).parent
    DATA_DIR = PROJECT_ROOT / "data"
    SRC_DIR = PROJECT_ROOT / "src"
    SCRIPTS_DIR = PROJECT_ROOT / "scripts"
    TESTS_DIR = PROJECT_ROOT / "tests"
    NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
    DOCS_DIR = PROJECT_ROOT / "docs"
    LOGS_DIR = PROJECT_ROOT / "logs"
    OUTPUTS_DIR = PROJECT_ROOT / "outputs"
    
    # Ensure directories exist
    for dir_path in [LOGS_DIR, OUTPUTS_DIR, DATA_DIR / "qa_pairs", OUTPUTS_DIR]:
        dir_path.mkdir(exist_ok=True, parents=True)
    
    # API Keys
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

    # Provider Configuration
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")  # openai | gemini
    EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "openai")  # openai | gemini
    
    # LLM Configuration
    LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4-turbo-preview")
    LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.3"))
    LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "2000"))

    # Gemini Configuration
    GEMINI_LLM_MODEL = os.getenv("GEMINI_LLM_MODEL", "gemini-1.5-pro")
    GEMINI_EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "models/text-embedding-004")
    
    # Embedding Configuration
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", "1536"))
    
    # Retrieval Configuration
    RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "5"))
    RETRIEVAL_SCORE_THRESHOLD = float(os.getenv("RETRIEVAL_SCORE_THRESHOLD", "0.5"))
    BM25_K1 = 1.5
    BM25_B = 0.75
    
    # Vector Store Configuration
    VECTOR_STORE_TYPE = os.getenv("VECTOR_STORE_TYPE", "faiss")
    VECTOR_STORE_PATH = DATA_DIR / "acts_embedding" / EMBEDDING_MODEL
    
    # Text Processing
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
    
    # Validation
    MIN_CONFIDENCE_SCORE = float(os.getenv("MIN_CONFIDENCE_SCORE", "0.6"))
    
    # Logging
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FORMAT = "[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s"
    LOG_FILE = LOGS_DIR / "qa_agent.log"
    
    # Data paths
    ACTS_CSV = DATA_DIR / "acts.csv"
    ACTS_CHUNKED_DIR = DATA_DIR / "acts_chunked"
    ACTS_HTML_DIR = DATA_DIR / "acts_html"
    ACTS_MD_DIR = DATA_DIR / "acts_md"
    QA_PAIRS_DIR = DATA_DIR / "qa_pairs"
    FINANCIAL_TEMPLATES_DIR = DATA_DIR / "financial_templates"
    REGULATIONS_DIR = DATA_DIR / "regulations"
    
    @classmethod
    def to_dict(cls) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            key: getattr(cls, key) for key in dir(cls)
            if not key.startswith('_') and key.isupper()
        }


class DevelopmentConfig(Config):
    """Development environment configuration."""
    DEBUG = True
    LLM_TEMPERATURE = 0.5
    LOG_LEVEL = "DEBUG"


class ProductionConfig(Config):
    """Production environment configuration."""
    DEBUG = False
    LLM_TEMPERATURE = 0.2
    LOG_LEVEL = "INFO"
    MIN_CONFIDENCE_SCORE = 0.7


class TestingConfig(Config):
    """Testing environment configuration."""
    TESTING = True
    LLM_MODEL = "gpt-3.5-turbo"
    RETRIEVAL_TOP_K = 3
    LOG_LEVEL = "WARNING"


def get_config(env: str = None) -> Config:
    """
    Get configuration based on environment variable.
    
    Args:
        env: Environment name. If None, reads from ENVIRONMENT variable.
    
    Returns:
        Config instance for the specified environment.
    """
    if env is None:
        env = os.getenv("ENVIRONMENT", "development")
    
    config_map = {
        "production": ProductionConfig,
        "prod": ProductionConfig,
        "testing": TestingConfig,
        "test": TestingConfig,
        "development": DevelopmentConfig,
        "dev": DevelopmentConfig,
    }
    
    config_class = config_map.get(env.lower(), DevelopmentConfig)
    return config_class()
