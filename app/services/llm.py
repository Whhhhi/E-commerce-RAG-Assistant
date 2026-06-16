from langchain_openai import ChatOpenAI
from langchain_core.language_models import BaseChatModel
from app.config import settings
import logging

logger = logging.getLogger(__name__)

_llm_instance = None


def get_llm(temperature: float = None) -> BaseChatModel:
    global _llm_instance

    temp = temperature if temperature is not None else settings.llm_temperature

    if _llm_instance is not None:
        # OpenAI 兼容模式可以运行时调温；本地 API 不支持则静默忽略
        if settings.llm_provider != "local":
            _llm_instance.temperature = temp
        return _llm_instance

    # ── 本地模型 API（LM Studio / llama.cpp server / ollama）──
    if settings.llm_provider == "local":
        _llm_instance = ChatOpenAI(
            model_name=settings.llm_model_name,
            api_key=settings.llm_api_key or "not-needed",
            base_url=settings.llm_api_base,
            temperature=temp,
            max_tokens=settings.llm_max_tokens,
        )
        logger.info("本地 LLM API → %s (model=%s)", settings.llm_api_base, settings.llm_model_name)
        return _llm_instance

    # ── OpenAI / 兼容 API ──
    if not settings.llm_api_key:
        raise RuntimeError(
            "LLM 服务未配置：请设置 LLM_API_KEY（远程 API）"
            "或 LLM_PROVIDER=local 及 LLM_API_BASE（本地模型）"
        )
    _llm_instance = ChatOpenAI(
        model_name=settings.llm_model_name,
        api_key=settings.llm_api_key,
        base_url=settings.llm_api_base or None,
        temperature=temp,
        max_tokens=settings.llm_max_tokens,
    )
    return _llm_instance


def clear_llm_cache():
    """清除 LLM 单例（切换模型/配置时调用）"""
    global _llm_instance
    _llm_instance = None
