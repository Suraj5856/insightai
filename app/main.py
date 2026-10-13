from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile

from app.config import settings
from app.ingestion import SUPPORTED, ingest_files, ingest_folder
from app.logger import get_logger, read_summary
from app.rag import answer_question
from app.schemas import AskRequest, AskResponse, IngestResponse

log = get_logger()
app = FastAPI(title="InsightAI", description="RAG assistant for your documents", version="1.0.0")

MAX_UPLOAD_MB = 20


def _require_key():
    if not settings.active_api_key:
        raise HTTPException(status_code=500, detail=f"{settings.key_name} is not set. Add it to .env")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ingest", response_model=IngestResponse)
def ingest_default_folder():
    """Index every file inside data/docs."""
    _require_key()
    try:
        return ingest_folder()
    except Exception as e:
        log.exception("ingest failed")
        raise HTTPException(500, f"Ingest failed: {type(e).__name__}: {str(e)[:300]}") from e


@app.post("/upload", response_model=IngestResponse)
async def upload(file: UploadFile = File(...)):
    """Upload one document (pdf, txt, md, docx) and index it."""
    _require_key()
    name = Path(file.filename or "").name  # strips any folder path (security)
    if not name.lower().endswith(SUPPORTED):
        raise HTTPException(400, f"Unsupported file type. Use: {', '.join(SUPPORTED)}")
    data = await file.read()
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"File too large (max {MAX_UPLOAD_MB} MB)")
    path = settings.upload_dir / name
    path.write_bytes(data)
    try:
        return ingest_files([path])
    except Exception as e:
        log.exception("upload failed")
        raise HTTPException(500, f"Upload failed: {type(e).__name__}: {str(e)[:300]}") from e


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    _require_key()
    try:
        return answer_question(req.question, req.top_k)
    except Exception as e:  # keep the API alive, log the real error
        log.exception("ask failed")
        raise HTTPException(500, f"Something went wrong: {type(e).__name__}") from e


@app.get("/metrics")
def metrics():
    """Latency and retrieval-quality summary from the query log."""
    return read_summary()
