"""
Simple example demonstrating QA Agent usage.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import get_config
from src.agents.qa_agent import QAAgent
from src.retrievers.hybrid_retriever import HybridRetriever
from src.services.llm_service import LLMService
from src.validators.legal_validator import LegalValidator
from src.utils.logger import setup_logger


def demonstrate_agent():
    """Demonstrate QA Agent functionality."""
    
    # Setup
    config = get_config()
    setup_logger(log_level=config.LOG_LEVEL)
    config_dict = config.to_dict()
    
    print("\n" + "=" * 80)
    print("QA AGENT DEMONSTRATION")
    print("=" * 80 + "\n")
    
    try:
        # Initialize components
        print("Initializing agent components...")
        retriever = HybridRetriever(config=config_dict)
        llm_service = LLMService(config=config_dict)
        validator = LegalValidator(config=config_dict)
        
        # Create agent
        agent = QAAgent(
            retriever=retriever,
            llm_service=llm_service,
            validator=validator,
            config=config_dict
        )
        print("Agent initialized successfully.\n")
        
        # Example queries
        example_queries = [
            "What is the GST rate in Singapore?",
            "When should a company file its annual report?",
        ]
        
        for query in example_queries:
            print("-" * 80)
            print(f"Query: {query}\n")
            
            try:
                response = agent.process_query(query)
                
                print(f"Answer:\n{response.answer}\n")
                print(f"Confidence: {response.confidence_score:.1%}")
                print(f"Processing Time: {response.processing_time:.2f}s")
                
                if response.sources:
                    print(f"\nSources:")
                    for source in set(response.sources)[:3]:
                        print(f"  • {source}")
                
                if response.legal_citations:
                    print(f"\nCitations:")
                    for citation in response.legal_citations[:3]:
                        print(f"  • {citation}")
                
            except Exception as e:
                print(f"Error processing query: {e}")
        
        print("\n" + "=" * 80)
        print("DEMONSTRATION COMPLETE")
        print("=" * 80 + "\n")
        
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    demonstrate_agent()
