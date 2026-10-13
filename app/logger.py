"""Logging: app log file + one JSON line per question (used for metrics)."""
import json
import logging
import statistics
from logging.handlers import RotatingFileHandler

from app.config import settings

QUERY_LOG = settings.log_dir / "queries.jsonl"


def get_logger(name: str = "insightai") -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    file = RotatingFileHandler(settings.log_dir / "app.log", maxBytes=1_000_000, backupCount=3)
    file.setFormatter(fmt)
    logger.addHandler(console)
    logger.addHandler(file)
    return logger


def log_query(record: dict) -> None:
    with open(QUERY_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def read_summary() -> dict:
    """Summary of all logged questions: latency and retrieval quality."""
    if not QUERY_LOG.exists():
        return {"total_queries": 0}
    rows = [json.loads(line) for line in QUERY_LOG.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        return {"total_queries": 0}
    totals = sorted(r["total_ms"] for r in rows)
    p95 = totals[min(len(totals) - 1, int(len(totals) * 0.95))]
    return {
        "total_queries": len(rows),
        "avg_total_ms": round(statistics.mean(totals), 1),
        "p95_total_ms": round(p95, 1),
        "avg_retrieval_ms": round(statistics.mean(r["retrieval_ms"] for r in rows), 1),
        "avg_generation_ms": round(statistics.mean(r["generation_ms"] for r in rows), 1),
        "avg_retrieval_score": round(statistics.mean(r["avg_score"] for r in rows), 3),
        "no_answer_rate": round(sum(1 for r in rows if r["num_sources"] == 0) / len(rows), 3),
    }
