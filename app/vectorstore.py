from functools import lru_cache

from langchain_chroma import Chroma

from app.config import settings
from app.providers import get_embeddings


@lru_cache(maxsize=1)
def get_vectorstore() -> Chroma:
    return Chroma(
        # separate collection per provider, because embedding sizes are different
        collection_name=f"insightai_{settings.provider}",
        embedding_function=get_embeddings(),
        persist_directory=settings.chroma_dir,
        collection_metadata={"hnsw:space": "cosine"},  # score = 1 - cosine distance
    )


def count_chunks() -> int:
    return get_vectorstore()._collection.count()
