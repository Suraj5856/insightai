"""Load documents -> split into chunks -> embed -> store in Chroma."""
import time
from pathlib import Path

from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import settings
from app.logger import get_logger
from app.vectorstore import get_vectorstore

log = get_logger()

LOADERS = {
    ".pdf": PyPDFLoader,
    ".txt": lambda p: TextLoader(p, encoding="utf-8"),
    ".md": lambda p: TextLoader(p, encoding="utf-8"),
    ".docx": Docx2txtLoader,
}
SUPPORTED = tuple(LOADERS)


def load_file(path: Path) -> list[Document]:
    docs = LOADERS[path.suffix.lower()](str(path)).load()
    for d in docs:
        d.metadata["source"] = path.name  # keep only the file name
    return docs


def split_documents(docs: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap
    )
    return splitter.split_documents(docs)


def _add_in_batches(store, chunks: list[Document], ids: list[str]) -> None:
    """Add chunks in small batches and retry when the free tier says 'too many requests'."""
    size = settings.embed_batch_size
    for i in range(0, len(chunks), size):
        batch_docs, batch_ids = chunks[i : i + size], ids[i : i + size]
        for attempt in range(5):
            try:
                store.add_documents(batch_docs, ids=batch_ids)
                break
            except Exception as e:
                msg = str(e).lower()
                rate_limited = any(w in msg for w in ("429", "quota", "rate", "resource_exhausted"))
                if rate_limited and attempt < 4:
                    wait = 20 * (attempt + 1)
                    log.warning("Rate limit hit, waiting %ds (batch %d)", wait, i // size + 1)
                    time.sleep(wait)
                else:
                    raise
        time.sleep(settings.embed_pause_seconds)


def ingest_files(paths: list[Path]) -> dict:
    store = get_vectorstore()
    files_done, chunks_added = 0, 0
    for path in paths:
        if path.suffix.lower() not in SUPPORTED:
            log.warning("Skipping unsupported file: %s", path.name)
            continue
        chunks = split_documents(load_file(path))
        if not chunks:
            continue
        # stable ids -> re-uploading the same file replaces it instead of duplicating
        ids = [f"{path.name}-{i}" for i in range(len(chunks))]
        _add_in_batches(store, chunks, ids)
        files_done += 1
        chunks_added += len(chunks)
        log.info("Ingested %s (%d chunks)", path.name, len(chunks))
    return {"files_processed": files_done, "chunks_added": chunks_added}


def ingest_folder(folder: Path | None = None) -> dict:
    folder = folder or settings.docs_dir
    paths = sorted(p for p in folder.glob("*") if p.is_file())
    return ingest_files(paths)
