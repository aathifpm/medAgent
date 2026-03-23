"""
MedCollab — Modular LLM Factory

Re-exported from src.utils for convenience.
"""

from __future__ import annotations
from langchain_core.language_models.chat_models import BaseChatModel
from src import config


def get_llm(temperature: float | None = None) -> BaseChatModel:
    """
    Return a ChatModel instance for the configured provider.

    Args:
        temperature: Override the default temperature. If None, uses config value.
    """
    temp = temperature if temperature is not None else config.TEMPERATURE
    provider = config.LLM_PROVIDER.lower().strip()

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        if not config.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is not set. Add it to your .env file.")
        return ChatOpenAI(
            model=config.OPENAI_MODEL,
            api_key=config.OPENAI_API_KEY,
            temperature=temp,
        )

    elif provider == "ollama":
        from langchain_community.chat_models import ChatOllama

        return ChatOllama(
            model=config.OLLAMA_MODEL,
            base_url=config.OLLAMA_BASE_URL,
            temperature=temp,
        )

    elif provider == "anthropic":
        from langchain_community.chat_models import ChatAnthropic

        if not config.ANTHROPIC_API_KEY:
            raise ValueError("ANTHROPIC_API_KEY is not set. Add it to your .env file.")
        return ChatAnthropic(
            model=config.ANTHROPIC_MODEL,
            api_key=config.ANTHROPIC_API_KEY,
            temperature=temp,
        )

    elif provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI

        if not config.GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY is not set. Add it to your .env file.")
        return ChatGoogleGenerativeAI(
            model=config.GOOGLE_MODEL,
            google_api_key=config.GOOGLE_API_KEY,
            temperature=temp,
        )

    elif provider == "azure":
        from langchain_openai import AzureChatOpenAI

        if not config.AZURE_OPENAI_API_KEY:
            raise ValueError("AZURE_OPENAI_API_KEY is not set. Add it to your .env file.")
        return AzureChatOpenAI(
            azure_deployment=config.AZURE_OPENAI_DEPLOYMENT,
            azure_endpoint=config.AZURE_OPENAI_ENDPOINT,
            api_key=config.AZURE_OPENAI_API_KEY,
            temperature=temp,
            api_version="2024-12-01-preview",
        )

    else:
        raise ValueError(
            f"Unsupported LLM_PROVIDER: '{provider}'. "
            f"Supported: openai, ollama, anthropic, google, azure"
        )
