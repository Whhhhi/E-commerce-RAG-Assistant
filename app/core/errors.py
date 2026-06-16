from enum import Enum
from typing import Any, Optional


class ErrorCode(str, Enum):
    CHAT_MESSAGE_EMPTY = "CHAT_MESSAGE_EMPTY"
    CHAT_MESSAGE_TOO_LONG = "CHAT_MESSAGE_TOO_LONG"
    CHAT_SESSION_INVALID = "CHAT_SESSION_INVALID"

    UPLOAD_UNSUPPORTED_FORMAT = "UPLOAD_UNSUPPORTED_FORMAT"
    UPLOAD_FILE_TOO_LARGE = "UPLOAD_FILE_TOO_LARGE"
    UPLOAD_INDEX_ERROR = "UPLOAD_INDEX_ERROR"

    RAG_EMPTY_KNOWLEDGE_BASE = "RAG_EMPTY_KNOWLEDGE_BASE"
    RAG_NO_RETRIEVAL_RESULTS = "RAG_NO_RETRIEVAL_RESULTS"
    RAG_CHAIN_ERROR = "RAG_CHAIN_ERROR"

    LLM_SERVICE_UNAVAILABLE = "LLM_SERVICE_UNAVAILABLE"
    EMBEDDING_SERVICE_UNAVAILABLE = "EMBEDDING_SERVICE_UNAVAILABLE"
    INTERNAL_SERVER_ERROR = "INTERNAL_SERVER_ERROR"


class AppException(Exception):
    def __init__(
        self,
        code: ErrorCode,
        message: str,
        status_code: int = 400,
        details: Any = None,
    ):
        self.code = code
        self.message = message
        self.status_code = status_code
        self.details = details
        super().__init__(self.message)