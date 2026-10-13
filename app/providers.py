"""One place that creates the LLM and the embedding model.
Switch provider with LLM_PROVIDER=gemini or LLM_PROVIDER=openai in .env"""
from functools import lru_cache

from app.config import settings


@lru_cache(maxsize=1)
def get_embeddings():
    if settings.provider == "gemini":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        return GoogleGenerativeAIEmbeddings(
            model=settings.gemini_embedding_model,
            google_api_key=settings.google_api_key or None,
        )
    if settings.provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(
            model=settings.openai_embedding_model, api_key=settings.openai_api_key or None
        )
    raise ValueError(f"Unknown LLM_PROVIDER: {settings.llm_provider}")


@lru_cache(maxsize=1)
def get_llm():
    if settings.provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=settings.gemini_llm_model,
            temperature=0,
            google_api_key=settings.google_api_key or None,
        )
    if settings.provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=settings.openai_llm_model, temperature=0, api_key=settings.openai_api_key or None
        )
    raise ValueError(f"Unknown LLM_PROVIDER: {settings.llm_provider}")
