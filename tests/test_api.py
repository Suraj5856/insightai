from fastapi.testclient import TestClient

from app import main
from app.main import app

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_ask_rejects_short_question():
    r = client.post("/ask", json={"question": "a"})
    assert r.status_code == 422


def test_upload_rejects_bad_file_type(monkeypatch):
    monkeypatch.setattr(main.settings, "google_api_key", "test-key")
    r = client.post("/upload", files={"file": ("virus.exe", b"data")})
    assert r.status_code == 400


def test_ask_uses_pipeline(monkeypatch):
    monkeypatch.setattr(main.settings, "google_api_key", "test-key")
    fake = {
        "answer": "18 paid leaves.",
        "sources": [{"file": "a.txt", "page": None, "score": 0.9, "snippet": "..."}],
        "timings": {"retrieval_ms": 1, "generation_ms": 2, "total_ms": 3},
        "tokens": None,
    }
    monkeypatch.setattr(main, "answer_question", lambda q, k: fake)
    r = client.post("/ask", json={"question": "How many leaves?"})
    assert r.status_code == 200
    assert r.json()["answer"] == "18 paid leaves."


def test_metrics_endpoint():
    assert client.get("/metrics").status_code == 200
