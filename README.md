# Singapore Legal & Tax QA Agent

## Overview

This QA Agent provides intelligent question-answering capabilities for Singapore legal, tax, and compliance questions. It uses Retrieval-Augmented Generation (RAG) with hybrid retrieval combining keyword search and vector similarity.

## Features

- **Legal & Tax QA**: Answer questions about Singapore acts, regulations, and tax rules
- **Hybrid Retrieval**: Combines BM25 keyword search with vector similarity
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

## Installation

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
# Edit .env with your LLM API key
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

Edit `config.py` or set environment variables:

- `OPENAI_API_KEY`: OpenAI API key
- `LLM_MODEL`: Model to use (default: gpt-4-turbo-preview)
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

## Data Pipeline

### 1. Document Chunking

Singapore legal acts are chunked and stored in `data/acts_chunked/`

### 2. Embedding Generation

Documents are embedded using `text-embedding-3-small` and stored in `data/acts_embedding/`

### 3. Retrieval

Hybrid retrieval combines:
- **BM25**: Keyword-based ranking
- **Vector Search**: Semantic similarity

### 4. Answer Generation

- Retrieved documents provide context
- GPT-4 generates comprehensive answers
- Citations are extracted and formatted

## Performance

- Average response time: < 5 seconds
- Confidence score: 60-95%
- Support for 50+ Singapore acts and regulations

## Limitations

- English language only
- Requires valid OpenAI API key
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
