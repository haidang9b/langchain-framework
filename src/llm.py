"""Model factories for Azure OpenAI chat and embeddings.

Modern LangChain v1 pattern: build model objects directly from
``langchain_openai`` and reuse them via ``with_structured_output`` /
``bind_tools`` at the call site. No deprecated ``LLMChain`` wrapper.
"""

from __future__ import annotations

from functools import lru_cache

from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings

from .config import Settings, load_settings


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached settings so we validate the environment only once per process."""
    return load_settings()


def get_chat_model(temperature: float = 0.0) -> AzureChatOpenAI:
    """Return the Azure OpenAI chat model used across the assistant."""
    s = get_settings()
    return AzureChatOpenAI(
        azure_endpoint=s.azure_endpoint,
        azure_deployment=s.azure_chat_deployment,
        api_version=s.azure_api_version,
        api_key=s.azure_api_key,
        temperature=temperature,
    )


def get_embeddings() -> AzureOpenAIEmbeddings:
    """Return the Azure OpenAI embeddings model used for the vector store."""
    s = get_settings()
    return AzureOpenAIEmbeddings(
        azure_endpoint=s.azure_endpoint,
        azure_deployment=s.azure_embedding_deployment,
        api_version=s.azure_api_version,
        api_key=s.azure_api_key,
    )
