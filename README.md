# Build Your First RAG with LangChain

A comprehensive tutorial on building a Retrieval-Augmented Generation (RAG) system using LangChain, LangGraph, and Ollama.

## Overview

This project teaches you how to build a RAG application from scratch, covering document loading, text chunking, embeddings, vector storage, retrieval, and generation using local LLMs.

## Features

- **Multi-format document loading**: HTML (web), PDF, CSV
- **Text chunking strategies**: Configurable chunk size and overlap
- **Local LLM inference**: Using Ollama (llama3, mistral)
- **Vector embeddings**: Using Ollama (nomic-embed-text)
- **LangGraph workflows**: State-based retrieval and generation
- **LangSmith tracing**: Optional observability and debugging

## Installation

### Prerequisites

- Python 3.11+
- [Ollama](https://ollama.ai/) installed and running

### Setup

1. Install the project dependencies with uv:

```bash
uv sync
```

Or install with pip:

```bash
pip install -e .
```

2. Copy the environment template:

```bash
cp .env.template .env
```

3. Pull required Ollama models:

```bash
ollama pull llama3      # or mistral
ollama pull nomic-embed-text
```

4. Ensure Ollama is running:

```bash
ollama serve
```

## Project Structure

```
.
├── pyproject.toml              # Project dependencies (uv)
├── uv.lock                     # Locked dependencies
├── .env.template               # Environment variables template
├── .env                        # Environment variables (user-specific)
├── README.md                   # This file
├── .gitignore                  # Git ignore rules
├── config.py                   # Shared env, model, and path config
│
├── example/
│   ├── config.py               # Local shim for running from example/
│   └── lang-chains-rag.py       # Complete RAG example (foundational)
│
├── exercise/
│   ├── config.py               # Local shim for running from exercise/
│   ├── load_documents.py       # Document loading utilities
│   ├── exercise1_multi_doc_rag.py    # Multi-document RAG
│   ├── exercise2_chunking_experiment.py  # Chunking strategy experiments
│   ├── exercise3_multi_format_loader.py  # Multi-format loading (HTML/PDF/CSV)
│   ├── exercise4_langchain_langsmith_upgrade.py  # LangChain + LangSmith
│
├── data/
│   └── sample.csv          # Sample CSV data
│
├── output/
│   └── chunking_experiment_results.png  # Experiment visualization
│
└── tests/
    ├── config.py               # Local shim for running from tests/
    ├── test_langsmith_auth.py  # Verify LangSmith API key/connection
    └── test_trace.py           # Test tracing with Ollama
```

## Usage

### Running the Example

Run the foundational RAG example:

```bash
python example/lang-chains-rag.py
```

This example:
- Loads content from a web article
- Chunks text into manageable pieces
- Creates embeddings and stores them in-memory
- Retrieves relevant context for a query
- Generates an answer using a local LLM

### Running Exercises

#### Exercise 1: Multi-Document RAG

```bash
python exercise/exercise1_multi_doc_rag.py
```

Learn how to load and query multiple documents simultaneously.

#### Exercise 2: Chunking Experiment

```bash
python exercise/exercise2_chunking_experiment.py
```

Experiment with different chunk sizes and overlaps to optimize retrieval.

#### Exercise 3: Multi-Format Loader

```bash
python exercise/exercise3_multi_format_loader.py
```

Load documents from HTML, PDF, and CSV sources in a single pipeline.

#### Exercise 4: LangChain + LangSmith Upgrade

```bash
python exercise/exercise4_langchain_langsmith_upgrade.py
```

Upgrade to the latest LangChain patterns with LangSmith tracing.

### Running Tests

This folder contains integration/verification tests for LangSmith tracing setup.

Verify LangSmith API key and connection:

```bash
python tests/test_langsmith_auth.py
```

Test full tracing functionality with Ollama:

```bash
python tests/test_trace.py
```

> **Note**: These are not unit tests. They are end-to-end verification scripts that test the LangSmith integration and Ollama connection. Run them after setting up your `.env` with LangSmith API keys.

## Dependencies

Core dependencies (from `pyproject.toml`):

- `langchain`
- `langchain-community`
- `langgraph`
- `beautifulsoup4`

Additional runtime dependencies (installed automatically):

- `langchain-ollama`
- `langchain-core`
- `langsmith`
- `python-dotenv`

## Environment Variables

Create a `.env` file based on `.env.template`:

```bash
# Ollama settings
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_TIMEOUT=30
CHAT_MODEL_LLAMA3=llama3
CHAT_MODEL_MISTRAL=mistral
EMBEDDING_MODEL=nomic-embed-text

# LangSmith (optional - for tracing)
LANGSMITH_TRACING=true
LANGCHAIN_TRACING_V2=true
LANGSMITH_ENDPOINT=https://eu.api.smith.langchain.com
LANGCHAIN_PROJECT=rag-experiments
LANGSMITH_API_KEY=your-api-key-here
```

## Key Concepts

### RAG Pipeline

1. **Load**: Fetch documents from various sources (web, PDF, CSV)
2. **Split**: Break documents into smaller chunks
3. **Embed**: Convert text chunks into vector embeddings
4. **Store**: Save embeddings in a vector database
5. **Retrieve**: Find relevant chunks based on user query
6. **Generate**: Use LLM to answer using retrieved context

### Chunking Strategies

The project demonstrates different chunk sizes:

| Chunk Size | Overlap | Use Case |
|------------|---------|----------|
| 300        | 0       | Precise, factual queries |
| 500        | 50      | Balanced general use |
| 1000       | 200     | Comprehensive context |

### Multi-Format Support

- **Web/HTML**: Using `WebBaseLoader` with BeautifulSoup
- **PDF**: Using `PyPDFLoader`
- **CSV**: Using `CSVLoader`

## Troubleshooting

### Ollama not running

Ensure Ollama is installed and running:

```bash
ollama serve
```

### Models not found

Pull required models:

```bash
ollama pull llama3
ollama pull nomic-embed-text
```

If you only want the local examples to run, set `LANGSMITH_TRACING=false` in `.env` to disable LangSmith uploads.

### No documents loaded

Check the source URLs or file paths are valid and accessible.

## License

MIT
