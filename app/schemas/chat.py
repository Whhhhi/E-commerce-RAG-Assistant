from pydantic import BaseModel, Field, field_validator
from typing import Optional
from datetime import datetime


class ChatRequest(BaseModel):
    message: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        examples=["这款手机支持5G吗？"],
        description="用户输入的消息文本",
    )
    session_id: str = Field(
        ...,
        min_length=1,
        max_length=64,
        examples=["sess_abc123"],
        description="会话标识，用于关联对话历史",
    )
    user_id: Optional[str] = Field(
        default=None,
        max_length=64,
        description="用户标识（可选）",
    )

    @field_validator("message")
    @classmethod
    def message_not_blank(cls, v: str) -> str:
        stripped = v.strip()
        if not stripped:
            raise ValueError("消息内容不能为空白字符")
        return stripped


class Source(BaseModel):
    document_id: str
    document_name: str
    chunk_index: int = Field(ge=0)
    relevance_score: float = Field(ge=0.0, le=1.0)
    content_excerpt: str = Field(max_length=500)


class ChatResponse(BaseModel):
    answer: str = Field(description="RAG 生成的回答")
    sources: list[Source] = Field(default_factory=list)
    session_id: str
    intent: str = Field(
        description="识别到的用户意图",
        examples=["product_inquiry", "return_exchange", "order_tracking", "chitchat"],
    )
    conversation_id: str
    latency_ms: Optional[float] = None