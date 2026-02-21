# Singapore Legal & Tax QA Agent

## Overview

This QA Agent provides intelligent question-answering capabilities for Singapore legal, tax, and compliance questions. It uses Retrieval-Augmented Generation (RAG) with hybrid retrieval combining keyword search and vector similarity.

## Features

- **Legal & Tax QA**: Answer questions about Singapore acts, regulations, and tax rules
- **Hybrid Retrieval**: Combines BM25 keyword search with vector similarity
- **Dual Model Providers**: Supports both OpenAI and Google Gemini for LLM + embeddings
- **Citation Extraction**: Automatically extracts and references legal citations
- **Confidence Scoring**: Provides confidence scores for answers
- **Conversation History**: Maintains history of queries and responses

## Project Structure

```
qa_agent_legal_tax/
├── src/
│   ├── agents/                 # QA and report agents
│   ├── services/               # LLM, embedding, and document services
│   ├── retrievers/             # Document retrieval logic
│   ├── validators/             # Answer and report validators
│   ├── utils/                  # Logging, text processing utilities
│   └── models/                 # Data models
├── data/
│   ├── acts_chunked/           # Chunked legal documents
│   ├── acts_embedding/         # Vector embeddings
│   └── qa_pairs/               # Generated QA pairs
├── scripts/                    # Utility scripts
├── tests/                      # Unit tests
├── config.py                   # Configuration management
├── main.py                     # Entry point
└── requirements.txt            # Dependencies
```

## Installation & Quick Start
1. Clone the repository:
```bash
git clone <repository-url>
cd qa_agent_legal_tax
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your provider API key(s)
```

4. Choose provider in `.env`:
```bash
# Option A: OpenAI
LLM_PROVIDER=openai
EMBEDDING_PROVIDER=openai
OPENAI_API_KEY=your_openai_api_key_here

# Option B: Gemini
LLM_PROVIDER=gemini
EMBEDDING_PROVIDER=gemini
GOOGLE_API_KEY=your_google_api_key_here
```

5. **Verify and run**
   ```bash
   python scripts/test_system.py
   python scripts/demo_agent.py
   python main.py
   ```

## Usage

### Interactive CLI

Run the interactive QA agent:

```bash
python main.py
```

Example interactions:

```
You: What are the tax filing deadlines for Singapore companies?
Agent: <Answer with legal citations and sources>

You: How do I prepare a balance sheet according to SFRS?
Agent: <Financial reporting guidance>

You: history
Agent: <Show conversation history>

You: quit
```

### Programmatic Usage

```python
from config import get_config
from src.agents.qa_agent import QAAgent
from src.retrievers.hybrid_retriever import HybridRetriever
from src.services.llm_service import LLMService
from src.validators.legal_validator import LegalValidator

# Initialize
config = get_config().to_dict()
retriever = HybridRetriever(config)
llm = LLMService(config)
validator = LegalValidator(config)

agent = QAAgent(retriever, llm, validator, config)

# Process query
response = agent.process_query("What is the GST rate in Singapore?")
print(response.answer)
print(f"Confidence: {response.confidence_score:.1%}")
```

## Configuration

Set environment variables in .env (or edit `config.py`):

- `LLM_PROVIDER`: `openai` or `gemini` (default: `gemini`)
- `EMBEDDING_PROVIDER`: `openai` or `gemini` (default: `gemini`)
- `OPENAI_API_KEY`: Required when using OpenAI
- `GOOGLE_API_KEY`: Required when using Gemini
- `LLM_MODEL`: Gemini chat model (default: `emini-2.5-flash`)
- `GEMINI_LLM_MODEL`: Gemini chat model (default: `gemini-2.5-flash`)
- `EMBEDDING_MODEL`: OpenAI embedding model (default: `text-embedding-3-small`)
- `GEMINI_EMBEDDING_MODEL`: Gemini embedding model (default: `models/text-embedding-004`)
- `RETRIEVAL_TOP_K`: Number of documents to retrieve (default: 5)
- `LOG_LEVEL`: Logging level (DEBUG, INFO, WARNING, ERROR)

## Development

### Running Tests

```bash
pytest tests/ -v
```

### Code Quality

```bash
black src/
flake8 src/
```

### Logging

Logs are written to `logs/qa_agent.log` and console.

### Troubleshooting

- `No documents available for retrieval`
  - Ensure `data/acts_chunked/` contains chunked JSON files.
- `OPENAI_API_KEY` or `GOOGLE_API_KEY` error
  - Confirm the selected provider and matching key are correctly set in `.env`.
- `Module not found`
  - Reinstall dependencies: `pip install -r requirements.txt`.

## Data Pipeline

### 1. Document Chunking

Singapore legal acts are chunked and stored in `data/acts_chunked/`

### 2. Embedding Generation

Documents are embedded using the selected provider model (`EMBEDDING_PROVIDER`) and stored in `data/acts_embedding/`.

### 3. Retrieval

Hybrid retrieval combines:
- **BM25**: Keyword-based ranking
- **Vector Search**: Semantic similarity

### 4. Answer Generation

- Retrieved documents provide context
- The selected provider model (`LLM_PROVIDER`) generates comprehensive answers
- Citations are extracted and formatted

## Performance

- Average response time: < 5 seconds
- Confidence score: 60-95%
- Support for 50+ Singapore acts and regulations

## Limitations

- Requires valid API key for selected provider (`OPENAI_API_KEY` or `GOOGLE_API_KEY`)
- Response quality depends on document quality and query clarity
- Professional legal consultation recommended for critical decisions

## Future Enhancements

- [ ] Multi-language support
- [ ] Fine-tuned models for legal domain
- [ ] Financial report generation
- [ ] Interactive document upload
- [ ] Web UI interface
- [ ] API endpoints

## License

MIT License
