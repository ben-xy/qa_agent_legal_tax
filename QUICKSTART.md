"""
Quick Start Guide for QA Agent Legal Tax

This guide will help you get the QA Agent up and running quickly.
"""


# QUICK START GUIDE - Singapore Legal & Tax QA Agent

## Step 1: Install Dependencies
 Run this once to install all required Python packages:
 pip install -r requirements.txt
## Step 2: Configure Environment
 Create a .env file from the template:
 cp .env.example .env
 
 Edit .env and add your OpenAI API key:
 OPENAI_API_KEY=your_actual_api_key_here
## Step 3: Verify Installation
 Test the retriever:
 python scripts/test_system.py
## Step 4: Run Demo
 See the agent in action:
 python scripts/demo_agent.py
## Step 5: Start Interactive Chat
 Launch the interactive agent:
 python main.py

# USAGE EXAMPLES
## Example 1: Ask about taxes
You: What is the GST rate in Singapore?
Agent: Based on the relevant documents... [answer with citations]
## Example 2: Check filing deadlines
You: When must I file my annual report?
Agent: According to the Companies Act... [detailed answer]
## Example 3: View conversation history
You: history
Agent: Displays all previous queries in this session
## Example 4: Exit the application
You: quit
Agent: Goodbye!

# PROGRAMMATIC USAGE

## Example: Using the agent in Python code
from config import get_config
from src.agents.qa_agent import QAAgent
from src.retrievers.hybrid_retriever import HybridRetriever
from src.services.llm_service import LLMService
from src.validators.legal_validator import LegalValidator

## Initialize components
config = get_config()
config_dict = config.to_dict()

retriever = HybridRetriever(config=config_dict)
llm = LLMService(config=config_dict)
validator = LegalValidator(config=config_dict)

## Create agent
agent = QAAgent(retriever, llm, validator, config_dict)

## Process a query
response = agent.process_query("What is the GST rate in Singapore?")

print(f"Answer: {response.answer}")
print(f"Confidence: {response.confidence_score:.1%}")
print(f"Sources: {response.sources}")
print(f"Citations: {response.legal_citations}")


# CONFIGURATION OPTIONS`
 Edit config.py or set environment variables:
 LLM Configuration:
 ```
 LLM_MODEL=gpt-4-turbo-preview    # gpt-4, gpt-3.5-turbo, etc.
 LLM_TEMPERATURE=0.3              # 0-1, higher = more creative
 LLM_MAX_TOKENS=2000              # Max response length
 ```
 Retrieval Configuration:
 ```
 RETRIEVAL_TOP_K=5                # Number of documents to retrieve
 RETRIEVAL_SCORE_THRESHOLD=0.5    # Minimum relevance score
 ```
 Logging:
 ```
 LOG_LEVEL=INFO                   # DEBUG, INFO, WARNING, ERROR
 ENVIRONMENT=development          # development, production, testing
```
# TESTING

## Run unit tests:
```
pytest tests/ -v
```
## Run with coverage:
```
pytest tests/ --cov=src
```
## Run specific test:
```
pytest tests/test_agents.py::TestQAAgent::test_agent_initialization -v
```

# TROUBLESHOOTING

 Issue: "No documents available for retrieval"
 Solution: Ensure data/acts_chunked/ directory contains JSON files with legal documents
 Issue: "OpenAI API error"
 Solution: Check that OPENAI_API_KEY is correctly set in .env file
 Issue: "Module not found"
 Solution: Make sure you've installed dependencies: pip install -r requirements.txt
 Issue: "Connection timeout"
 Solution: Check internet connection and OpenAI API status

# NEXT STEPS
1. Customize prompts in src/prompts.py for your use case
2. Add more data by placing documents in data/acts_chunked/
3. Extend validators in src/validators/legal_validator.py
4. Add more test cases in tests/
5. Deploy using main.py as entry point

# RESOURCES
Documentation: See README.md
Implementation Details: See IMPLEMENTATION_SUMMARY.md
Code Examples: See scripts/demo_agent.py
Tests: See tests/test_agents.py

