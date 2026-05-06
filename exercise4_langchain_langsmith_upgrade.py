import re

import config

print("ENV CHECK")
print("USER_AGENT =", config.USER_AGENT)
print("LANGSMITH_TRACING =", config.LANGSMITH_TRACING)
print("LANGCHAIN_TRACING_V2 =", config.LANGCHAIN_TRACING_V2)
print("LANGCHAIN_PROJECT =", config.LANGCHAIN_PROJECT)
api_key = config.LANGSMITH_API_KEY
print("LANGSMITH_API_KEY loaded =", "yes" if api_key else "no")

import bs4

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.document_loaders import (
    WebBaseLoader,
    PyPDFLoader,
    CSVLoader,
)

from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings

from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

import warnings
warnings.filterwarnings("ignore")

llm = ChatOllama(
    model=config.CHAT_MODEL_MISTRAL,
    base_url=config.OLLAMA_BASE_URL,
    timeout=config.OLLAMA_TIMEOUT
)

embeddings = OllamaEmbeddings(
    model=config.EMBEDDING_MODEL,
    base_url=config.OLLAMA_BASE_URL
)

vector_store = InMemoryVectorStore(embeddings)

def clean_text(text):
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def sanitize(text):
    blocked = ["You should only respond", "Response Format:", "{", "}"]
    for b in blocked:
        text = text.replace(b, "")
    return text

def format_docs(docs):
    if not docs:
        return "No relevant context found."
    regular_docs = [d for d in docs if d.metadata.get("source") != "system"]
    if not regular_docs:
        return "No relevant information found in the knowledge base."
    docs_content = "\n\n".join(
        sanitize(doc.page_content[:500])
        for doc in regular_docs
    )
    return docs_content

def extract_sources(docs):
    if not docs:
        return []
    regular_docs = [d for d in docs if d.metadata.get("source") != "system"]
    return list(set(d.metadata.get("source") for d in regular_docs if d.metadata.get("source")))

import hashlib

def deduplicate_docs(docs):
    seen = set()
    unique = []
    for d in docs:
        content_hash = hashlib.md5(d.page_content.strip().encode()).hexdigest()
        source = d.metadata.get("source_type", "unknown")
        key = (content_hash, source)
        if key not in seen:
            seen.add(key)
            unique.append(d)
    return unique

def retrieve_docs(query):
    if isinstance(query, dict):
        query = query.get("question", "")
    query_lower = query.lower()
    docs = vector_store.similarity_search(query, k=6)

    if "pdf" in query_lower:
        docs = [d for d in docs if d.metadata.get("source_type") == "pdf"]
    elif "csv" in query_lower:
        docs = [d for d in docs if d.metadata.get("source_type") == "csv"]

    if not docs:
        docs = vector_store.similarity_search(query, k=3)

    if not docs:
        return [Document(
            page_content="No relevant information found in the knowledge base.",
            metadata={"source": "system"}
        )]

    return deduplicate_docs(docs[:3])

retriever = RunnableLambda(retrieve_docs).with_config(
    run_name="Retriever",
    tags=["retrieval"]
)

from langchain_core.prompts import ChatPromptTemplate

template = """You are a helpful assistant. Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

{context}

Question: {question}

Helpful Answer:"""

prompt = ChatPromptTemplate.from_template(template)

output_parser = StrOutputParser()

def format_context(x):
    docs = x["context"]
    if all(d.metadata.get("source") == "system" for d in docs):
        return {
            "question": x["question"],
            "context": "No relevant information found in the available documents.",
            "sources": []
        }
    sources = extract_sources(docs)
    source_note = f"\n\nSources: {', '.join(sources)}" if sources else ""
    return {
        "question": x["question"],
        "context": format_docs(docs) + "\n\nAnswer using only factual information." + source_note
    }

rag_chain = (
    {
        "context": retriever,
        "question": RunnablePassthrough()
    }
    | RunnableLambda(format_context)
    | prompt
    | llm
    | output_parser
).with_config(
    run_name="RAG Pipeline"
)

docs = []

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

pdf_loader = PyPDFLoader("https://arxiv.org/pdf/2310.06825.pdf")
pdf_docs = pdf_loader.load()
docs.extend([
    Document(
        page_content=d.page_content,
        metadata={**d.metadata, "source_type": "pdf"}
    )
    for d in pdf_docs
])

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

docs = [
    Document(
        page_content=clean_text(doc.page_content),
        metadata=doc.metadata
    )
    for doc in docs
]

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

splits = text_splitter.split_documents(docs)

print(f"Total chunks: {len(splits)}")

_ = vector_store.add_documents(splits)

questions = [
    "What is Chain-of-Thought prompting?",
    "What topics are discussed in the PDF?",
    "What topics are in the CSV data?",
]

for q in questions:
    result = rag_chain.invoke(
        {"question": q},
        config={"tags": ["exercise4", "multi-format", "chunking-500"]}
    )

    print("\n" + "="*50)
    print("Question:", q)
    print("Answer:", result.strip())
