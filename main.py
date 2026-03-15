"""
Main entry point for QA Agent application.
Interactive CLI for querying the legal and tax QA agent.
"""

import logging
import sys
from pathlib import Path

from config import get_config
from src.agents.qa_agent import QAAgent
from src.retrievers.hybrid_retriever import HybridRetriever
from src.services.llm_service import LLMService
from src.validators.legal_validator import LegalValidator
from src.utils.logger import setup_logger

# Setup logging
config = get_config()
setup_logger(
    log_level=config.LOG_LEVEL,
    log_format=config.LOG_FORMAT,
    log_file=str(config.LOG_FILE)
)

logger = logging.getLogger(__name__)


def initialize_agent() -> QAAgent:
    """Initialize QA agent with all required components."""
    config_dict = config.to_dict()
    
    logger.info("Initializing QA Agent components...")
    
    try:
        # Initialize components
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
        
        logger.info("QA Agent initialized successfully")
        return agent
        
    except Exception as e:
        logger.error(f"Failed to initialize agent: {e}")
        raise


def print_welcome():
    """Print welcome message."""
    print("\n" + "=" * 80)
    print("   Singapore Legal & Tax QA Agent")
    print("=" * 80)
    print("\nThis agent can answer questions about:")
    print("  • Singapore legal regulations and acts")
    print("  • Tax rules and compliance requirements")
    print("  • Financial reporting standards")
    print("  • Employment and labor regulations")
    print("\nCommands:")
    print("  • Type your question and press Enter")
    print("  • 'history' - Show conversation history")
    print("  • 'quit'    - Exit the application")
    print("=" * 80 + "\n")


def print_response(response):
    """Pretty print agent response."""
    print("\n" + "-" * 80)
    print(f"Answer:\n{response.answer}")
    print(f"\nConfidence Score: {response.confidence_score:.1%}")
    if response.confidence_breakdown:
        breakdown = response.confidence_breakdown
        #print("Confidence Breakdown:")
        print(f"  • Retrieval Quality: {breakdown.get('retrieval_quality', 0.0):.1%}")
        print(f"  • Answer Consistency: {breakdown.get('answer_consistency', 0.0):.1%}")
        print(f"  • Citation Coverage: {breakdown.get('citation_coverage', 0.0):.1%}")
        print(f"  • Final Weighted Score: {breakdown.get('final', response.confidence_score):.1%}")
    print(f"Processing Time: {response.processing_time:.2f}s")
    
    if response.sources:
        print(f"\nSources:")
        for source in set(response.sources):
            print(f"  • {source}")
    
    if response.legal_citations:
        print(f"\nLegal Citations:")
        for citation in response.legal_citations:
            print(f"  • {citation}")
    
    print("-" * 80 + "\n")


def main():
    """Main interactive loop."""
    try:
        # Initialize agent
        print("Starting QA Agent...")
        agent = initialize_agent()
        print_welcome()
        
        # Interactive loop
        while True:
            try:
                user_input = input("You: ").strip()
                
                if not user_input:
                    continue
                
                # Handle commands
                if user_input.lower() == 'quit':
                    print("\nGoodbye!")
                    break
                elif user_input.lower() == 'history':
                    history = agent.get_conversation_history()
                    if history:
                        print("\nConversation History:")
                        for idx, item in enumerate(history, 1):
                            print(f"{idx}. {item['query'][:80]}...")
                    else:
                        print("\nNo conversation history.")
                    print()
                    continue
                
                # Process query
                logger.info(f"Processing user query: {user_input[:50]}...")
                response = agent.process_query(user_input)
                print_response(response)
                
            except KeyboardInterrupt:
                print("\n\nInterrupted by user.")
                break
            except Exception as e:
                logger.error(f"Error processing query: {e}", exc_info=True)
                print(f"\nError: {str(e)}\n")
                
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        print(f"\nFatal error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
