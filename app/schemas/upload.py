from pydantic import BaseModel, Field
from typing import Optional


class UploadResponse(BaseModel):
    document_id: str
    file_name: str
    knowledge_base: str = Field(
        description="入库的知识库类别: product / policy"
    )
    chunks: int = Field(ge=0, description="总切块数")
    status: str = Field(
        description="索引状态: indexed / partial / failed",
        examples=["indexed", "partial", "failed"],
    )
    elapsed_seconds: float = Field(ge=0)
    indexed_chunks: Optional[int] = None
    failed_chunks: Optional[int] = None