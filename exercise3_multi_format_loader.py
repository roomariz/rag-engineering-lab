import re
import bs4
import config
from typing_extensions import List, TypedDict

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import START, StateGraph

# Loaders
from langchain_community.document_loaders import (
    WebBaseLoader,
    PyPDFLoader,
    CSVLoader,
)

# Models
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings

# Vector store
from langchain_core.vectorstores import InMemoryVectorStore

import warnings
warnings.filterwarnings("ignore")

# --------------------------------------------------
# LLM + Embeddings
# --------------------------------------------------

llm = ChatOllama(
    model=config.CHAT_MODEL_MISTRAL,  # fast + good
    base_url=config.OLLAMA_BASE_URL,
    timeout=config.OLLAMA_TIMEOUT
)

embeddings = OllamaEmbeddings(
    model=config.EMBEDDING_MODEL,
    base_url=config.OLLAMA_BASE_URL
)

vector_store = InMemoryVectorStore(embeddings)

# --------------------------------------------------
# Helpers
# --------------------------------------------------

def clean_text(text):
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def sanitize(text):
    blocked = ["You should only respond", "Response Format:", "{", "}"]
    for b in blocked:
        text = text.replace(b, "")
    return text

# --------------------------------------------------
# 1. LOAD DOCUMENTS (MULTI-FORMAT)
# --------------------------------------------------

docs = []

# ---- HTML (Web) ----
web_loader = WebBaseLoader(
    web_paths=(
        "https://en.wikipedia.org/wiki/Chain-of-thought_prompting",
    ),
    bs_kwargs=dict(parse_only=bs4.SoupStrainer(id="mw-content-text"))
)
web_docs = web_loader.load()
docs.extend([
    Document(
        page_content=d.page_content,
        metadata={**d.metadata, "source_type": "web"}
    )
    for d in web_docs
])

# ---- PDF (from URL) ----
pdf_loader = PyPDFLoader("https://arxiv.org/pdf/2310.06825.pdf")
pdf_docs = pdf_loader.load()
docs.extend([
    Document(
        page_content=d.page_content,
        metadata={**d.metadata, "source_type": "pdf"}
    )
    for d in pdf_docs
])

# ---- CSV ----
csv_loader = CSVLoader(
    file_path=str(config.DATA_DIR / "sample.csv"),
    encoding="utf-8"
)
csv_docs = csv_loader.load()
csv_docs = [
    Document(
        page_content=f"Topic: {d.page_content.replace(',', ', ')}",
        metadata={**d.metadata, "source_type": "csv"}
    )
    for d in csv_docs
]
docs.extend(csv_docs)

print(f"\nLoaded total documents: {len(docs)}")

# --------------------------------------------------
# 2. CLEAN
# --------------------------------------------------

docs = [
    Document(
        page_content=clean_text(doc.page_content),
        metadata=doc.metadata
    )
    for doc in docs
]

# --------------------------------------------------
# 3. CHUNKING
# --------------------------------------------------

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

splits = text_splitter.split_documents(docs)

print(f"Total chunks: {len(splits)}")

# --------------------------------------------------
# 4. INDEX
# --------------------------------------------------

_ = vector_store.add_documents(splits)

# --------------------------------------------------
# 5. PROMPT
# --------------------------------------------------

prompt = ChatPromptTemplate.from_template(
    """You are a helpful assistant. Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

{context}

Question: {question}

Helpful Answer:"""
)

# --------------------------------------------------
# 6. STATE
# --------------------------------------------------

class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

# --------------------------------------------------
# 7. RETRIEVE
# --------------------------------------------------

def retrieve(state: State):
    query = state["question"].lower()

    docs = vector_store.similarity_search(query, k=6)

    # Source routing
    if "pdf" in query:
        docs = [d for d in docs if d.metadata.get("source_type") == "pdf"]
    elif "csv" in query:
        docs = [d for d in docs if d.metadata.get("source_type") == "csv"]

    # Fallback if no docs found
    if not docs:
        docs = vector_store.similarity_search(query, k=3)

    return {"context": docs[:3]}

# --------------------------------------------------
# 8. GENERATE
# --------------------------------------------------

def generate(state: State):
    docs_content = "\n\n".join(
        sanitize(doc.page_content[:500])  # truncate for speed
        for doc in state["context"]
    )

    messages = prompt.invoke({
        "question": state["question"],
        "context": docs_content + "\n\nAnswer using only factual information."
    })

    response = llm.invoke(messages)

    return {"answer": response.content.strip()}

# --------------------------------------------------
# 9. GRAPH
# --------------------------------------------------

graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()

# --------------------------------------------------
# 10. TEST
# --------------------------------------------------

questions = [
    "What is Chain-of-Thought prompting?",
    "What topics are discussed in the PDF?",
    "What topics are in the CSV data?",
]

for q in questions:
    result = graph.invoke({"question": q})

    print("\n" + "="*50)
    print("Question:", q)
    print("Answer:", result["answer"])

    # Deduplicate sources
    seen = set()
    unique_docs = []
    for doc in result["context"]:
        src = doc.metadata.get("source")
        if src not in seen:
            seen.add(src)
            unique_docs.append(src)

    print("\nSources:")
    for src in unique_docs:
        print("-", src)
