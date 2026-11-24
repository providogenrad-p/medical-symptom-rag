# Medical Symptom Analysis RAG

A Retrieval-Augmented Generation (RAG) assistant that analyses symptoms described in everyday language and answers **only from official MedlinePlus health topics**, with citations, red-flag warnings and a disclaimer. It includes a ChatGPT-style web chat and a terminal version.

> **Educational prototype. Not a medical device and not medical advice.**

## How it works

1. **Knowledge base:** 1,000+ public-domain health topic summaries from [MedlinePlus](https://medlineplus.gov/) (U.S. National Library of Medicine), downloaded from the official XML feed.
2. **Chunking:** section-aware splitting (by headings, then by size with overlap); every chunk keeps its topic and section.
3. **Embeddings + vector store:** `all-MiniLM-L6-v2` (local) and ChromaDB (cosine, HNSW).
4. **Retrieval:** a relevance gate (the system refuses when nothing is relevant) and at most two chunks per condition.
5. **Query expansion:** the LLM suggests candidate conditions **only to improve the search**; the answer is still written only from the retrieved passages.
6. **Generation:** `openai/gpt-oss-120b` on Groq, streamed, with a short or detailed answer format.

### Safety layers (outside the LLM)

- **A. Privacy:** names, phone numbers, IDs and dates are removed before anything is sent to the LLM.
- **B. Red flags:** fixed rules detect emergencies (e.g. possible meningitis or stroke) and show a warning instantly.
- **C. Checks:** every citation must point to a retrieved source, and a disclaimer is always added.

## Tech stack

Python · LangChain · Hugging Face sentence-transformers · ChromaDB · Groq · FastAPI + Uvicorn · HTML/CSS/JavaScript

## Setup

```bash
git clone https://github.com/<your-username>/medical-symptom-rag.git
cd medical-symptom-rag
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux
pip install -r requirements.txt
```

Create a `.env` file with your Groq API key (free at https://console.groq.com/keys):

```
GROQ_API_KEY=your_key_here
```

Build the knowledge base (once):

```bash
python fetch_medlineplus.py          # all topics (use --demo for ~35 topics)
python medical_rag_demo.py --ingest
```

## Run

Web chat (opens the browser at http://127.0.0.1:8000):

```bash
python medical_web.py
```

Terminal version:

```bash
python medical_rag_demo.py --symptoms "fever, severe headache, stiff neck, light hurts my eyes" --age 19
python medical_rag_demo.py --symptoms "..." --detailed      # full differential table
```

## Example questions

- *Fever, severe headache, stiff neck, light hurts my eyes* → red-flag warning, meningitis first
- *Always exhausted, cold hands and feet, gained weight, dry skin, thinning hair* → query expansion finds hypothyroidism
- *How do I change a flat car tire?* → nothing relevant, the system refuses

## Data source

Health topic summaries: MedlinePlus.gov, U.S. National Library of Medicine (public domain).
