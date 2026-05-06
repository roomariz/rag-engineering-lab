import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(dotenv_path=BASE_DIR / ".env", override=True)

DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"

USER_AGENT = os.getenv("USER_AGENT", "my-rag-app/1.0")
LANGSMITH_TRACING = os.getenv("LANGSMITH_TRACING", os.getenv("LANGCHAIN_TRACING_V2", "false"))
LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2", LANGSMITH_TRACING)
LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT", "rag-experiments")
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY", "")
LANGSMITH_ENDPOINT = os.getenv("LANGSMITH_ENDPOINT", "")

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "30"))
CHAT_MODEL_LLAMA3 = os.getenv("CHAT_MODEL_LLAMA3", "llama3")
CHAT_MODEL_MISTRAL = os.getenv("CHAT_MODEL_MISTRAL", "mistral")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
