import bs4
import config
from langchain_community.document_loaders import (
    WebBaseLoader,
    PyPDFLoader,
    CSVLoader,
)

print("=" * 60)
print("LOADING DOCUMENTS FROM DIFFERENT SOURCES")
print("=" * 60)

docs = []

# ---- 1. HTML (Web) ----
print("\n[1] Loading HTML from Wikipedia...")
web_loader = WebBaseLoader(
    web_paths=(
        "https://en.wikipedia.org/wiki/Chain-of-thought_prompting",
    ),
    bs_kwargs=dict(parse_only=bs4.SoupStrainer(id="mw-content-text"))
)
web_docs = web_loader.load()
print(f"   Loaded {len(web_docs)} HTML document(s)")
if web_docs:
    print(f"   Preview: {web_docs[0].page_content[:200]}...")
    print(f"   Source: {web_docs[0].metadata.get('source')}")
docs.extend(web_docs)

# ---- 2. PDF ----
pdf_path = config.DATA_DIR / "sample.pdf"
if pdf_path.exists():
    print("\n[2] Loading PDF...")
    pdf_loader = PyPDFLoader(str(pdf_path))
    pdf_docs = pdf_loader.load()
    print(f"   Loaded {len(pdf_docs)} PDF document(s)")
    if pdf_docs:
        print(f"   Preview: {pdf_docs[0].page_content[:200]}...")
        print(f"   Source: {pdf_docs[0].metadata.get('source')}")
    docs.extend(pdf_docs)
else:
    print("\n[2] Skipping PDF (sample.pdf not found in data/)")

# ---- 3. CSV ----
print("\n[3] Loading CSV...")
csv_loader = CSVLoader(
    file_path=str(config.DATA_DIR / "sample.csv"),
    encoding="utf-8"
)
csv_docs = csv_loader.load()
print(f"   Loaded {len(csv_docs)} CSV document(s)")
if csv_docs:
    print(f"   Preview: {csv_docs[0].page_content[:200]}...")
    print(f"   Source: {csv_docs[0].metadata.get('source')}")
docs.extend(csv_docs)

print("\n" + "=" * 60)
print(f"TOTAL DOCUMENTS LOADED: {len(docs)}")
print("=" * 60)

# Print all docs summary
print("\n--- Document Summary ---")
for i, doc in enumerate(docs):
    source = doc.metadata.get("source", "unknown")
    content_preview = doc.page_content[:100].replace("\n", " ")
    print(f"Doc {i+1}: {source}")
    print(f"   {content_preview}...")
