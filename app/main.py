from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.core.errors import AppException, ErrorCode
from app.api.chat import router as chat_router, init_chains
from app.api.upload import router as upload_router
from app.config import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    # startup: 加载知识库索引
    init_chains()
    yield
    # shutdown: 清理资源（扩展预留）
    pass


app = FastAPI(
    title="电商智能客服 RAG API",
    version="1.0.0",
    debug=settings.debug,
    lifespan=lifespan,
)

# ── CORS：允许前端开发服务器跨域访问 ──
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── 生产模式：挂载前端构建产物 ──
import os
_frontend_dist = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend", "dist")
if os.path.isdir(_frontend_dist):
    app.mount("/", StaticFiles(directory=_frontend_dist, html=True), name="frontend")


@app.exception_handler(AppException)
async def app_exception_handler(request: Request, exc: AppException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "code": exc.code.value,
                "message": exc.message,
                "details": exc.details,
            }
        },
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "code": ErrorCode.INTERNAL_SERVER_ERROR.value,
                "message": "服务器内部错误",
                "details": str(exc) if app.debug else None,
            }
        },
    )


app.include_router(chat_router, tags=["chat"])
app.include_router(upload_router, tags=["upload"])


@app.get("/health")
async def health_check():
    return {"status": "healthy"}
