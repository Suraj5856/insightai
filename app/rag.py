"""The RAG pipeline: retrieve relevant chunks, then ask the LLM using only them."""
import time

from langchain_core.prompts import ChatPromptTemplate

from app.config import settings
from app.logger import get_logger, log_query
from app.providers import get_llm
from app.vectorstore import get_vectorstore

log = get_logger()

NO_ANSWER = "I could not find this in the uploaded documents."
MIN_SCORE = 0.25  # chunks below this similarity are ignored

PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are InsightAI, a document assistant. Answer the question using ONLY the "
            "context below. If the answer is not in the context, reply exactly: "
            f"\"{NO_ANSWER}\" Keep the answer clear and short.\n\nContext:\n{{context}}",
        ),
        ("human", "{question}"),
    ]
)

def _text(msg) -> str:
    """Get plain text from an LLM message (some models return a list of parts)."""
    c = msg.content
    if isinstance(c, str):
        return c
    return "".join(x if isinstance(x, str) else x.get("text", "") for x in c)


def answer_question(question: str, top_k: int | None = None) -> dict:
    k = top_k or settings.top_k
    t0 = time.perf_counter()

    # 1) Retrieval
    results = get_vectorstore().similarity_search_with_relevance_scores(question, k=k)
    results = [(d, s) for d, s in results if s >= MIN_SCORE]
    t1 = time.perf_counter()

    sources = [
        {
            "file": d.metadata.get("source", "unknown"),
            "page": d.metadata.get("page"),
            "score": round(float(s), 3),
            "snippet": d.page_content[:200].replace("\n", " "),
        }
        for d, s in results
    ]

    # 2) Generation (skip the LLM call if nothing relevant was found)
    tokens = None
    if not results:
        answer = NO_ANSWER
    else:
        context = "\n\n---\n\n".join(d.page_content for d, _ in results)
        msg = (PROMPT | get_llm()).invoke({"context": context, "question": question})
        answer = _text(msg).strip()
        tokens = getattr(msg, "usage_metadata", None)
    t2 = time.perf_counter()

    timings = {
        "retrieval_ms": round((t1 - t0) * 1000, 1),
        "generation_ms": round((t2 - t1) * 1000, 1),
        "total_ms": round((t2 - t0) * 1000, 1),
    }
    log_query(
        {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "question": question,
            "num_sources": len(sources),
            "avg_score": round(sum(s["score"] for s in sources) / len(sources), 3) if sources else 0.0,
            "tokens": tokens,
            **timings,
        }
    )
    log.info("Q=%r sources=%d total=%.0fms", question, len(sources), timings["total_ms"])
    return {"answer": answer, "sources": sources, "timings": timings, "tokens": tokens}
