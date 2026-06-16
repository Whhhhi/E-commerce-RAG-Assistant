from langchain_openai import OpenAIEmbeddings
from langchain_core.embeddings import Embeddings
from app.config import settings
import logging

logger = logging.getLogger(__name__)

_embedding_instance = None


def get_embedding() -> Embeddings:
    global _embedding_instance

    if _embedding_instance is not None:
        return _embedding_instance

    # ── 本地 Embedding API（LM Studio / llama.cpp server）──
    if settings.embedding_provider == "local":
        base_url = settings.embedding_api_base or settings.llm_api_base
        _embedding_instance = OpenAIEmbeddings(
            model=settings.embedding_model_name,
            api_key=settings.llm_api_key or "not-needed",
            base_url=base_url,
        )
        logger.info("本地 Embedding API → %s (model=%s)", base_url, settings.embedding_model_name)
        return _embedding_instance

    # ── OpenAI Embedding ──
    api_key = settings.embedding_api_key or settings.llm_api_key
    if not api_key:
        raise RuntimeError(
            "Embedding 服务未配置：请设置 EMBEDDING_API_KEY（远程 API）"
            "或 EMBEDDING_PROVIDER=local 及 LLM_API_BASE（本地模型）"
        )
    _embedding_instance = OpenAIEmbeddings(
        model=settings.embedding_model_name,
        api_key=api_key,
    )
    return _embedding_instance


def clear_embedding_cache():
    """清除 Embedding 单例（切换模型时调用）"""
    global _embedding_instance
    _embedding_instance = None
