from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=BASE_DIR / ".env", extra="ignore")

    # "gemini" (default, has a free tier) or "openai"
    llm_provider: str = "gemini"

    # Google Gemini
    google_api_key: str = ""
    gemini_llm_model: str = "gemini-flash-lite-latest"
    gemini_embedding_model: str = "gemini-embedding-001"

    # OpenAI (optional)
    openai_api_key: str = ""
    openai_llm_model: str = "gpt-4o-mini"
    openai_embedding_model: str = "text-embedding-3-small"

    chunk_size: int = 800
    chunk_overlap: int = 120
    top_k: int = 4
    embed_batch_size: int = 20      # chunks sent to the embedding API at once
    embed_pause_seconds: float = 2.0  # pause between batches (helps free-tier limits)

    docs_dir: Path = BASE_DIR / "data" / "docs"
    upload_dir: Path = BASE_DIR / "data" / "uploads"
    chroma_dir: str = str(BASE_DIR / "chroma_db")
    log_dir: Path = BASE_DIR / "logs"

    @property
    def provider(self) -> str:
        return self.llm_provider.strip().lower()

    @property
    def active_api_key(self) -> str:
        return self.google_api_key if self.provider == "gemini" else self.openai_api_key

    @property
    def key_name(self) -> str:
        return "GOOGLE_API_KEY" if self.provider == "gemini" else "OPENAI_API_KEY"


settings = Settings()
settings.upload_dir.mkdir(parents=True, exist_ok=True)
settings.log_dir.mkdir(parents=True, exist_ok=True)
