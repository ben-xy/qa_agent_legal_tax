# Quick Start Guide - Singapore Legal & Tax QA Agent

This guide helps you run the project quickly with either OpenAI or Google Gemini.

## 1) Install Dependencies

```bash
pip install -r requirements.txt
```

## 2) Configure Environment

```bash
cp .env.example .env
```

Edit `.env` and choose one provider setup.

### Option A: OpenAI (default)

```dotenv
LLM_PROVIDER=openai
EMBEDDING_PROVIDER=openai

OPENAI_API_KEY=your_openai_api_key_here
LLM_MODEL=gpt-4o
EMBEDDING_MODEL=text-embedding-3-small

LLM_TEMPERATURE=0.3
LLM_MAX_TOKENS=2000
```

### Option B: Google Gemini

```dotenv
LLM_PROVIDER=gemini
EMBEDDING_PROVIDER=gemini

GOOGLE_API_KEY=your_google_api_key_here
GEMINI_LLM_MODEL=gemini-1.5-pro
GEMINI_EMBEDDING_MODEL=models/text-embedding-004

LLM_TEMPERATURE=0.3
LLM_MAX_TOKENS=2000
```

## 3) Verify Setup

```bash
python scripts/test_system.py
```

## 4) Run Demo

```bash
python scripts/demo_agent.py
```

## 5) Start Interactive Chat

```bash
python main.py
```

## Usage Examples

- Ask tax question: `What is the GST rate in Singapore?`
- Ask compliance question: `When must a company file its annual return?`
- Show history: `history`
- Exit: `quit`

## Programmatic Usage

```python
from config import get_config
from src.agents.qa_agent import QAAgent
from src.retrievers.hybrid_retriever import HybridRetriever
from src.services.llm_service import LLMService
from src.validators.legal_validator import LegalValidator

config = get_config().to_dict()

retriever = HybridRetriever(config=config)
llm = LLMService(config=config)
validator = LegalValidator(config=config)

agent = QAAgent(retriever, llm, validator, config)
response = agent.process_query("What is the GST rate in Singapore?")

print(response.answer)
print(response.confidence_score)
```

## Common Commands

```bash
pytest tests/ -v
pytest tests/ --cov=src
black src/
flake8 src/
```

## Troubleshooting

- `No documents available for retrieval`
	- Ensure `data/acts_chunked/` contains chunked JSON files.
- `OPENAI_API_KEY` or `GOOGLE_API_KEY` error
	- Confirm the selected provider and matching key are correctly set in `.env`.
- `Module not found`
	- Reinstall dependencies: `pip install -r requirements.txt`.

## Resources

- Main documentation: `README.md`
- Implementation notes: `IMPLEMENTATION_SUMMARY.md`
- Demo script: `scripts/demo_agent.py`
- Test suite: `tests/test_agents.py`

