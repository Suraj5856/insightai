# InsightAI – RAG Assistant for Your Documents

Ask questions about your PDFs, Word files and text files. InsightAI finds the most relevant parts of your documents and uses an LLM to answer **only from those parts**, with sources.

**Tech:** Python, FastAPI, LangChain, Google Gemini API (LLM + embeddings; OpenAI also supported), ChromaDB (vector database), Docker

## How it works

```
Document -> split into chunks -> embeddings -> ChromaDB
Question -> embedding -> top-K similar chunks -> prompt + LLM -> answer + sources
```

1. **Ingestion** (`app/ingestion.py`): loads PDF/DOCX/TXT/MD, splits into overlapping chunks, creates embeddings, stores in ChromaDB.
2. **Retrieval + Generation** (`app/rag.py`): finds the top-K chunks, ignores weak matches, asks the LLM to answer only from the context.
3. **Logging & monitoring** (`app/logger.py`): every question is logged with retrieval time, generation time, similarity score and token use. `/metrics` shows a summary.
4. **Evaluation** (`app/evaluation.py`): runs a test question set and reports retrieval hit rate, answer accuracy and latency.

## Project structure

```
insightai/
├── app/
│   ├── main.py          # FastAPI endpoints
│   ├── rag.py           # retrieval + LLM answer
│   ├── ingestion.py     # load, chunk, embed, store
│   ├── providers.py     # Gemini / OpenAI model setup
│   ├── vectorstore.py   # ChromaDB setup
│   ├── evaluation.py    # quality metrics
│   ├── logger.py        # logs + metrics summary
│   ├── schemas.py       # request/response models
│   └── config.py        # settings from .env
├── scripts/evaluate.py  # run evaluation
├── tests/test_api.py    # API tests (no API key needed)
├── data/docs/           # sample document
├── data/eval/           # evaluation questions
├── Dockerfile
├── requirements.txt
└── .env.example
```

## Setup

```bash
git clone https://github.com/<your-username>/insightai.git
cd insightai
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env            # Windows: copy .env.example .env
# open .env and add your GOOGLE_API_KEY
```

Get a free Gemini key from https://aistudio.google.com/apikey (free-tier limits and model names can change; check Google's pricing page).

## Run

```bash
uvicorn app.main:app --reload
```

Open http://127.0.0.1:8000/docs to try every endpoint in the browser.

## API

| Method | Endpoint   | What it does |
|--------|-----------|--------------|
| GET    | `/health`  | Check the server is running |
| POST   | `/ingest`  | Index all files in `data/docs` |
| POST   | `/upload`  | Upload and index one file (pdf, docx, txt, md) |
| POST   | `/ask`     | Ask a question, get answer + sources + timings |
| GET    | `/metrics` | Latency and retrieval-quality summary |

### Example

```bash
curl -X POST http://127.0.0.1:8000/ingest

curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "How many paid leaves do employees get per year?"}'
```

Response:

```json
{
  "answer": "Employees get 18 paid leaves per year.",
  "sources": [{"file": "company_handbook.txt", "page": null, "score": 0.71, "snippet": "..."}],
  "timings": {"retrieval_ms": 210.4, "generation_ms": 850.2, "total_ms": 1060.6}
}
```

## Evaluation

```bash
python -m scripts.evaluate
```

Prints retrieval hit rate, answer accuracy and average latency on `data/eval/eval_set.json`. Add your own questions to that file for your own documents.

## Tests

```bash
pytest
```

## Docker

```bash
docker build -t insightai .
docker run -p 8000:8000 --env-file .env insightai
```

## Settings (`.env`)

| Name | Default | Meaning |
|------|---------|---------|
| `LLM_PROVIDER` | gemini | `gemini` or `openai` |
| `GOOGLE_API_KEY` | – | Your Gemini key |
| `GEMINI_LLM_MODEL` | gemini-2.5-flash | Gemini chat model |
| `GEMINI_EMBEDDING_MODEL` | gemini-embedding-001 | Gemini embedding model |
| `OPENAI_API_KEY` | – | Only if provider is `openai` |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | 800 / 120 | Chunking |
| `TOP_K` | 4 | Chunks retrieved per question |

If you get a "model not found" error, model names have changed. Check the current names in Google AI Studio and update `.env`.
If you hit rate limits (429) on the free tier, wait a minute and retry, or ingest fewer files at once.

## Future improvements

- Chat history (follow-up questions)
- Re-ranking of retrieved chunks
- Simple web UI (React or Streamlit)

## License

MIT
