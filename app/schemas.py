from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000)
    top_k: int | None = Field(default=None, ge=1, le=10)


class Source(BaseModel):
    file: str
    page: int | None = None
    score: float
    snippet: str


class Timings(BaseModel):
    retrieval_ms: float
    generation_ms: float
    total_ms: float


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]
    timings: Timings
    tokens: dict | None = None


class IngestResponse(BaseModel):
    files_processed: int
    chunks_added: int
