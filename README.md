# RAG Example (Text + PDF)

A compact, course-built Retrieval-Augmented Generation (RAG) training project that demonstrates two pipelines:
- Text RAG over a plain text corpus (sliding-window chunking, OpenAI embeddings, Qdrant, LLM querying).
- PDF RAG with structure extraction (text, tables, images), segmenting, OpenAI embeddings, Qdrant, LLM querying.

This repository is intentionally simple and end-to-end runnable to help you understand the moving parts of a practical RAG system.

---

## Features at a glance

- Vector store: Qdrant (via Docker)
- Embeddings: OpenAI `text-embedding-3-small` (1536 dims)
- Chat model: configurable via `CHAT_MODEL` (default: `gpt-4o-mini`)
- Text corpus pipeline (sample German text in `data/35794-0.txt`)
- PDF pipeline (example: `data/BMI25028_pks-2024.pdf`)
- Structured PDF parsing with `pdfplumber` and `PyMuPDF`:
  - Text lines grouped and filtered
  - Tables extracted
  - Images extracted
  - Headline-based segmenting and JSON manifest
- Query helpers and minimal prompt templates that require citing chunk IDs

---

## Prerequisites

- Python 3.10+ (recommended 3.11+)
- Docker + Docker Compose (for Qdrant)
- An OpenAI API key

Optional:
- If you want to run `parse_pdf.py`, you’ll need `pypdf` (not strictly required otherwise).

---

## Setup

1) Clone and create a virtual environment
~~~
git clone <your-fork-or-origin> rag-example-202511-2
cd rag-example-202511-2

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
~~~

2) Environment variables (create a `.env` in repo root)
~~~
OPENAI_API_KEY=sk-xxx
# Optional: change chat model used by end_to_end/query
CHAT_MODEL=gpt-4o-mini
~~~

3) Start Qdrant (vector database)
~~~
docker compose up -d
# Qdrant will be available at http://localhost:6333 (HTTP) and http://localhost:6334 (gRPC)
# Data is persisted to ./qdrant_data
~~~

Notes:
- If you plan to use `parse_pdf.py`, also run `pip install pypdf`.
- The repository’s `requirements.txt` installs `qdrant-client`, `openai`, `python-dotenv`, `pymupdf`, `pymupdf-layout`, `pymupdf4llm`, `pdfplumber`, and others used throughout the scripts.

---

## Project structure

Key files and what they do:

- `chunking.py` — Sliding-window chunking of plain text by story (default: `data/35794-0.txt`).
- `embedding.py` — OpenAI embeddings helper (`text-embedding-3-small`) and cosine similarity utility.
- `indexing.py` — Builds the `my_collection` in Qdrant and indexes text chunks.
- `marc_version.py` — Retrieves and prints top chunk per story (simple retriever demo).
- `end_to_end.py` — Full text RAG: retrieve top chunks from Qdrant and ask the chat model with guardrails.
- `parse_pdf_into_structure.py` — Extracts PDF text (filtered), tables, and images; detects headlines; builds `parsed_pks/document_structure.json`.
- `embed_parsed_segment.py` — Chunks and embeds structured PDF segments; upserts to Qdrant as `pdf_segments`.
- `query_pdf.py` — Retrieves relevant PDF segments and queries the chat model.
- `process_pdf.py` — Converts a PDF to Markdown with `pymupdf4llm` (handy for inspection).
- `parse_pdf.py` — Minimal `pypdf`-based text extractor (optional, simplest path).
- `docker-compose.yaml` — Qdrant configuration.
- `data/` — Sample inputs (`35794-0.txt`, `BMI25028_pks-2024.pdf`).

---

## Pipeline 1: Text RAG

High-level flow:
- Input text file → sliding-window chunking → embeddings → upsert to Qdrant.
- Query → embed query → similarity search → build context → ask chat model.

Run end-to-end:

1) Index the sample text into Qdrant
~~~
python indexing.py
# Creates collection "my_collection" (1536 dims, COSINE) if missing
# Chunks are derived from chunking.get_chunks()
~~~

2) Try a simple retrieval printout
~~~
python marc_version.py
# Prints best match per story from "my_collection"
~~~

3) Ask the LLM with retrieved context
~~~
python end_to_end.py
# Uses CHAT_MODEL (default gpt-4o-mini)
# Trims chunk text to MAX_CHUNK_CHARS to keep prompts short
~~~

Important notes:
- Embedding model is `text-embedding-3-small` (1536 dims). If you change models, update vector size in `indexing.py` and any collection creation code accordingly.
- Chunking defaults: `window_size=3`, `stride=1`. Adjust in `chunking.get_chunks(...)`.

---

## Pipeline 2: PDF RAG

High-level flow:
- PDF → text/table/image extraction → headline-based segmenting → segment chunking → embeddings → upsert to Qdrant.
- Query → embed query → similarity search → reconstruct context → ask chat model.

1) Parse and structure the PDF
~~~
python parse_pdf_into_structure.py
# Outputs to ./parsed_pks:
# - content.txt (text + simple pipe tables for inspection)
# - manifest_text_tables.json
# - manifest_images.json (extracted images via PyMuPDF)
# - document_structure.json (headline-segmented text + attached tables)
~~~

2) Embed segments and upsert to Qdrant
~~~
python embed_parsed_segment.py
# Creates collection "pdf_segments" if missing (1536 dims)
# Upserts segment chunks with payload (segment_id, headline, page, etc.)
~~~

3) Query the PDF collection with LLM answering
~~~
python query_pdf.py
# Builds context blocks by matching retrieved points to structured segments
# Calls the chat model using a German, citation-focused system prompt
~~~

Extras:
- `process_pdf.py <file.pdf>`: converts a PDF to Markdown via `pymupdf4llm`, helpful for quick manual inspection or comparisons.

---

## Configuration and knobs

- OpenAI:
  - Embeddings: set in `embedding.py` and referenced in `indexing.py` / `embed_parsed_segment.py`.
  - Chat model: `CHAT_MODEL` environment variable (default: `gpt-4o-mini`).
  - API key: `.env` → `OPENAI_API_KEY`.

- Qdrant:
  - Collections: `"my_collection"` (text pipeline), `"pdf_segments"` (PDF pipeline).
  - Distance: cosine.
  - Vector size: 1536 (must match chosen embedding model).
  - Ports: 6333 (HTTP), 6334 (gRPC). Data persisted to `./qdrant_data`.

- Retrieval:
  - `limit` and `score_threshold`:
    - Text pipeline (`end_to_end.py`): `limit=5`, `score_threshold=0.5` (tune as needed).
    - `marc_version.py`: `limit=10`, `score_threshold=0.3` (prints top per story).
    - PDF pipeline (`query_pdf.py`): `limit=10`, `score_threshold=0.5`.
  - Context trimming: `MAX_CHUNK_CHARS` in `end_to_end.py` to keep prompts compact.

- Chunking:
  - Text corpus: sliding window (size/stride) in `chunking.get_chunks(...)`.
  - PDF segments: `embed_parsed_segment.chunk_text` uses a word-count approximation for `max_tokens` (rule-of-thumb 0.7 words per token).

- Prompts:
  - System prompts in `end_to_end.py` and `query_pdf.py` enforce:
    - Only use provided context
    - Answer in the user’s language (default German)
    - Provide short quotes with `[cid: ...]` markers

---

## Troubleshooting

- Connection errors to Qdrant
  - Make sure Docker is running and `docker compose up -d` succeeded.
  - Verify ports 6333/6334 are not blocked or used by another service.

- Vector dimension mismatch
  - If you switch from `text-embedding-3-small` to another model, update all Qdrant `VectorParams(size=...)` to match the new dimension.
  - Recreate collections or reindex if necessary.

- Rate limits or auth errors
  - Check `OPENAI_API_KEY` in `.env`.
  - Some environments require `export OPENAI_API_KEY=...` in the current shell.

- PDF extraction quality
  - Complex PDFs can be noisy. This project applies simple heuristics (e.g., skip numeric-heavy lines; avoid table regions when extracting lines).
  - Tune `is_headline(...)` and thresholds in `parse_pdf_into_structure.py` as needed.

- Optional dependency for `parse_pdf.py`
  - Install `pypdf` if you want to use the minimal extractor:
    ~~~
    pip install pypdf
    ~~~

---

## Example end-to-end commands

Quick start: text pipeline
~~~
# 1) Start Qdrant
docker compose up -d

# 2) Create venv + install deps
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 3) .env with your OpenAI key
echo "OPENAI_API_KEY=sk-xxx" > .env
echo "CHAT_MODEL=gpt-4o-mini" >> .env

# 4) Index
python indexing.py

# 5) Retrieve demo
python marc_version.py

# 6) End-to-end answer
python end_to_end.py
~~~

Quick start: PDF pipeline
~~~
# 1) Parse + structure
python parse_pdf_into_structure.py

# 2) Embed + upsert
python embed_parsed_segment.py

# 3) Query + answer
python query_pdf.py
~~~

---

## Notes on publishing

- Remove or replace sample data as needed, especially if it’s large or has licensing constraints.
- Add a LICENSE file before publishing. Common choices: MIT, Apache 2.0, or CC-BY for docs/data.
- Consider adding a simple `Makefile` or shell scripts for common tasks (start Qdrant, index, query).
- CI can be added to lint, type-check, and run smoke tests.

---

## Acknowledgements

- Qdrant (vector database)
- OpenAI API (embeddings + chat models)
- PyMuPDF (`pymupdf`) and `pymupdf4llm`
- `pdfplumber`

---