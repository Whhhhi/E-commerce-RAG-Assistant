import time
import uuid
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from typing import Optional

from app.schemas.upload import UploadResponse
from app.knowledge.loader import load_document
from app.knowledge.splitter import split_documents
from app.services.vector_store import VectorStoreService
from app.services.bm25_store import BM25StoreService
from app.core.errors import ErrorCode

router = APIRouter()

ALLOWED_EXTENSIONS = {".pdf", ".md", ".txt", ".docx"}
MAX_FILE_SIZE = 20 * 1024 * 1024


@router.post("/upload", response_model=UploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    knowledge_base: Optional[str] = Form("product"),
):
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail={
                "code": "UPLOAD_UNSUPPORTED_FORMAT",
                "message": f"不支持的文件格式 '{ext}'，仅支持 {ALLOWED_EXTENSIONS}",
            },
        )

    content = await file.read()
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400,
            detail={
                "code": "UPLOAD_FILE_TOO_LARGE",
                "message": f"文件大小超过限制 ({MAX_FILE_SIZE//1024//1024}MB)",
            },
        )

    doc_id = f"doc_{uuid.uuid4().hex[:12]}"
    save_dir = Path("data") / "documents" / knowledge_base
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"{doc_id}{ext}"
    save_path.write_bytes(content)

    raw_docs = load_document(save_path, doc_id=doc_id)
    chunks = split_documents(raw_docs)

    t0 = time.time()

    try:
        vs = VectorStoreService(kb_name=knowledge_base)
        bm25 = BM25StoreService(kb_name=knowledge_base)
        vs.add_documents(chunks)
        bm25.add_documents(chunks)
    except RuntimeError as e:
        raise HTTPException(
            status_code=500,
            detail={
                "code": ErrorCode.EMBEDDING_SERVICE_UNAVAILABLE.value,
                "message": str(e),
            },
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "code": "UPLOAD_INDEX_ERROR",
                "message": f"索引构建失败: {str(e)}",
            },
        )

    elapsed = round(time.time() - t0, 2)

    return UploadResponse(
        document_id=doc_id,
        file_name=file.filename,
        knowledge_base=knowledge_base,
        chunks=len(chunks),
        status="indexed",
        elapsed_seconds=elapsed,
    )