"""
Quick test script to verify retriever functionality.
"""

import logging
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import get_config
from src.retrievers.hybrid_retriever import HybridRetriever
from src.utils.logger import setup_logger

# Setup logging
config = get_config()
setup_logger(log_level=config.LOG_LEVEL)
logger = logging.getLogger(__name__)


def test_retrieval():
    """Test document retrieval."""
    logger.info("Starting retrieval test...")
    
    try:
        # Initialize retriever
        config_dict = config.to_dict()
        retriever = HybridRetriever(config=config_dict)
        
        if not retriever.documents:
            logger.warning("No documents loaded. Please check data/acts_chunked/ directory.")
            return
        
        logger.info(f"Loaded {len(retriever.documents)} documents")
        
        # Test queries
        test_queries = [
            "What is the GST rate in Singapore?",
            "How do I file income tax?",
            "What are the employment act requirements?",
            "Tell me about company registration",
        ]
        
        print("\n" + "=" * 80)
        print("RETRIEVAL TEST")
        print("=" * 80)
        
        for query in test_queries:
            print(f"\nQuery: {query}")
            print("-" * 80)
            
            # Retrieve documents
            results = retriever.retrieve(query=query, top_k=3)
            
            if results:
                for idx, doc in enumerate(results, 1):
                    source = doc.get('source', 'Unknown')
                    score = doc.get('score', 0)
                    content = doc.get('content', '')[:200]
                    
                    print(f"\n{idx}. Source: {source}")
                    print(f"   Score: {score:.4f}")
                    print(f"   Content: {content}...")
            else:
                print("No results found.")
        
        print("\n" + "=" * 80)
        logger.info("Retrieval test completed successfully")
        
    except Exception as e:
        logger.error(f"Error during retrieval test: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    test_retrieval()
