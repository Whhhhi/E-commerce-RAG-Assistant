from pydantic_settings import BaseSettings
from typing import Literal


class Settings(BaseSettings):
    app_name: str = "电商智能客服 RAG"
    debug: bool = False
    host: str = "0.0.0.0"
    port: int = 8000

    # ── LLM ──
    llm_provider: Literal["openai", "deepseek", "local"] = "openai"
    llm_model_name: str = "gpt-4o-mini"
    llm_api_key: str = ""
    llm_temperature: float = 0.1
    llm_max_tokens: int = 2048

    # ── 本地模型 API（provider=local 时生效）──
    # LM Studio / llama.cpp server / ollama 等提供的 OpenAI 兼容接口
    llm_api_base: str = "http://127.0.0.1:1234/v1"

    # ── Embedding ──
    embedding_provider: Literal["openai", "local"] = "openai"
    embedding_model_name: str = "text-embedding-3-small"
    embedding_api_key: str = ""
    embedding_dimension: int = 1536

    # ── 本地 Embedding API（embedding_provider=local 时生效）──
    # 留空则复用 llm_api_base；也可指定独立地址如 http://127.0.0.1:1234/v1
    embedding_api_base: str = ""

    # ── Retrieval ──
    retriever_top_k: int = 5
    retriever_rrf_k: int = 60
    retriever_faiss_weight: float = 1.0
    retriever_bm25_weight: float = 1.0

    # ── Document Processing ──
    chunk_size: int = 512
    chunk_overlap: int = 50
    max_file_size_mb: int = 20

    # ── Session ──
    session_max_rounds: int = 20

    # ── Data Paths ──
    data_dir: str = "data"
    vector_store_dir: str = "data/vector_store"
    bm25_index_dir: str = "data/bm25_index"
    documents_dir: str = "data/documents"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
