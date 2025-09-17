# Toyota/Lexus QA Agent (RAG + SQL) — Demo

## Overview
Single-agent demo that answers questions using:
- **SQL** over a local SQLite DB (`toyota.db`) built from CSVs.
- **RAG** over manuals/policies indexed into **Chroma** with OpenAI embeddings (cosine similarity).
- Minimal **Streamlit** UI.

## Repo layout
- `create_sql_db.py` — build `toyota.db` from CSVs.
- `index_chroma.py` — create/load Chroma collection from PDFs (page-chunked).
- `manuals_scraper.py` — fetch Toyota/Lexus owner’s manuals and save normalized filenames.
- `agent_core.py` — smolagents `ToolCallingAgent` + tools `sql_select` & `rag_search`.
- `system_prompt.txt` — agent system prompt.
- `streamlit_app.py` — Streamlit UI.
- `requirements.txt` — Python deps.

---

## Prerequisites
- Python **3.10+**
- Install deps:
    pip install -r requirements.txt
- **OPENAI_API_KEY** in env:
    export OPENAI_API_KEY=sk-...

## Prepare data

### 1) Build the SQL database
- Put CSVs in your data folder.
- Adjust paths in `create_sql_db.py` if needed.
- Run:
    python3 create_sql_db.py
- Output: `toyota.db` (tables created from CSV filenames).

### 2) (Optional) Download manuals
- Configure filters in `manuals_scraper.py` if needed.
- Run:
    python3 manuals_scraper.py
- Output: PDFs saved with normalized names like `brand_model_modelType_year_partNumber.pdf`.

### 3) Build the RAG index
- Place PDFs in your `docs/` or `manuals/` folders (paths are set in `index_chroma.py`).
- Run:
    python3 index_chroma.py
- Output: local Chroma collection with page-level chunks and metadata (`file_path`, `page`).

## Run

### Quick CLI check
    python3 agent_core.py

### Streamlit UI
    streamlit run streamlit_app.py
- Input box for the question.
- Answer panel.
- Panels for **citations** (if RAG used) and **SQL** (if SQL used).
- Debug log panel (captures agent/stdout).

