# 电商智能客服 RAG 系统 — 后端架构设计文档

> 版本：v1.0  
> 技术栈：FastAPI + LangChain + FAISS + BM25  
> 更新日期：2026-06-16

---

## 目录

1. [项目概述](#1-项目概述)
2. [目录结构与文件职责](#2-目录结构与文件职责)
3. [API 设计](#3-api-设计)
4. [数据模型（Pydantic v2 Schema）](#4-数据模型pydantic-v2-schema)
5. [RAG Chain 构建流程（LCEL）](#5-rag-chain-构建流程lcel)
6. [意图路由逻辑](#6-意图路由逻辑)
7. [Hybrid Retrieval（FAISS + BM25 + RRF）](#7-hybrid-retrievalfaiss--bm25--rrf)
8. [会话管理](#8-会话管理)
9. [文档上传与向量化建库流程](#9-文档上传与向量化建库流程)
10. [错误码设计](#10-错误码设计)
11. [配置与环境变量](#11-配置与环境变量)
12. [边界情况与异常处理策略](#12-边界情况与异常处理策略)

---

## 1. 项目概述

### 1.1 业务目标

构建一个面向电商场景的智能客服问答系统，基于 RAG（Retrieval-Augmented Generation）架构，能够：

- 回答商品咨询（规格、价格、库存）
- 处理退换货政策相关问题
- 查询订单状态（通过 Tool 调用）
- 进行日常闲聊

### 1.2 架构总览

```
用户请求
   │
   ▼
┌──────────────┐     ┌──────────────────┐
│  FastAPI App │────▶│   Intent Router   │
│  (main.py)   │     │  (意图分类器)      │
└──────────────┘     └───────┬──────────┘
                             │
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
       ┌──────────┐   ┌──────────┐   ┌──────────┐
       │商品 RAG   │   │政策 RAG   │   │ Tool 调用 │
       │(FAISS+   │   │(FAISS+   │   │(查单工具) │
       │ BM25)    │   │ BM25)    │   │          │
       └──────────┘   └──────────┘   └──────────┘
              │              │              │
              ▼              ▼              ▼
       ┌────────────────────────────────────────┐
       │         LLM (最终生成回答)               │
       └────────────────────────────────────────┘
```

---

## 2. 目录结构与文件职责

```
rag_backend/
├── app/
│   ├── __init__.py
│   ├── main.py                      # FastAPI 应用入口，注册路由、中间件、生命周期
│   ├── config.py                    # 配置管理（pydantic-settings），读取环境变量
│   │
│   ├── api/                         # HTTP 路由层
│   │   ├── __init__.py
│   │   ├── chat.py                  # POST /chat — 对话接口
│   │   └── upload.py                # POST /upload — 文档上传接口
│   │
│   ├── schemas/                     # Pydantic v2 请求/响应模型
│   │   ├── __init__.py
│   │   ├── chat.py                  # ChatRequest, ChatResponse, Source
│   │   └── upload.py                # UploadResponse, UploadError
│   │
│   ├── core/                        # 核心业务逻辑
│   │   ├── __init__.py
│   │   ├── intent_router.py         # 意图分类路由（LLM + 关键词回退）
│   │   ├── rag_chain.py             # LCEL RAG Chain 组装工厂
│   │   ├── hybrid_retriever.py      # FAISS + BM25 + RRF 混合检索器
│   │   ├── session.py               # 内存会话管理（session_id → 消息历史）
│   │   └── errors.py                # 自定义异常类 + 错误码枚举
│   │
│   ├── services/                    # 基础设施服务
│   │   ├── __init__.py
│   │   ├── embedding.py             # Embedding 模型加载与调用封装
│   │   ├── vector_store.py          # FAISS 向量库的构建/加载/检索
│   │   ├── bm25_store.py            # BM25 索引的构建/加载/检索
│   │   └── llm.py                   # LLM 客户端封装（支持 OpenAI / 本地模型切换）
│   │
│   ├── knowledge/                   # 知识库文档处理
│   │   ├── __init__.py
│   │   ├── loader.py                # 多格式文档加载器（PDF / Markdown / TXT）
│   │   └── splitter.py              # 文档分块策略（RecursiveCharacterTextSplitter）
│   │
│   └── tools/                       # 工具调用
│       ├── __init__.py
│       └── order_tool.py            # 订单查询工具（模拟或对接外部 API）
│
├── data/                            # 运行时数据目录（gitignore）
│   ├── vector_store/                # FAISS 索引文件 (.faiss + .pkl)
│   ├── bm25_index/                  # BM25 序列化索引 (.pkl)
│   └── documents/                   # 上传的原始文档备份
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py                  # pytest fixtures（测试用 LLM/Embedding mock）
│   ├── test_chat.py                 # /chat 接口集成测试
│   ├── test_upload.py               # /upload 接口集成测试
│   ├── test_intent_router.py        # 意图分类单元测试
│   ├── test_hybrid_retriever.py     # 混合检索器单元测试
│   └── test_rag_chain.py            # RAG Chain 单元测试
│
├── requirements.txt
├── .env.example
└── BACKEND_SPEC.md                  # 本文档
```

### 各层职责说明

| 层级 | 目录 | 职责 |
|------|------|------|
| **路由层** | `api/` | 只做 HTTP 协议适配：参数提取、状态码返回、调用 Service 层。不含业务逻辑 |
| **Schema 层** | `schemas/` | 纯数据定义。Pydantic v2 model，含 JSON Schema 示例、`field_validator` |
| **核心层** | `core/` | 与框架无关的业务逻辑：RAG 链组装、意图决策、检索融合算法 |
| **服务层** | `services/` | 基础设施封装：向量库、BM25、LLM、Embedding，可替换实现 |
| **知识层** | `knowledge/` | 文档处理管线：加载 → 分块 |

---

## 3. API 设计

### 3.1 POST /chat — 对话接口

**请求：**

```json
{
  "message": "这款手机支持5G吗？",
  "session_id": "sess_abc123",
  "user_id": "user_456"
}
```

**成功响应（200）：**

```json
{
  "answer": "是的，该手机支持 5G 全网通，兼容 SA/NSA 双模组网。",
  "sources": [
    {
      "document_id": "doc_001",
      "document_name": "产品规格-智能手机2026.pdf",
      "chunk_index": 3,
      "relevance_score": 0.92,
      "content_excerpt": "支持 5G 全网通，SA/NSA 双模..."
    }
  ],
  "session_id": "sess_abc123",
  "intent": "product_inquiry",
  "conversation_id": "conv_xyz"
}
```

**错误响应示例：**

```json
{
  "error": {
    "code": "RAG_EMPTY_KNOWLEDGE_BASE",
    "message": "知识库为空，请先上传文档",
    "details": null
  }
}
```

### 3.2 POST /upload — 上传文档

**请求：** `multipart/form-data`

| 字段 | 类型 | 说明 |
|------|------|------|
| `file` | File | 支持 .pdf / .md / .txt，最大 20MB |
| `knowledge_base` | str (可选) | 指定入库类别：`product` / `policy`，默认 `product` |

**成功响应（200）：**

```json
{
  "document_id": "doc_001",
  "file_name": "产品规格-智能手机2026.pdf",
  "knowledge_base": "product",
  "chunks": 47,
  "status": "indexed",
  "elapsed_seconds": 2.34
}
```

**部分成功响应（207，部分Chunk索引失败）：**

```json
{
  "document_id": "doc_002",
  "file_name": "大文件.pdf",
  "chunks": 100,
  "indexed_chunks": 98,
  "failed_chunks": 2,
  "status": "partial",
  "elapsed_seconds": 5.12
}
```

---

## 4. 数据模型（Pydantic v2 Schema）

### 4.1 `schemas/chat.py`

```python
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
    """引用来源"""
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
```

### 4.2 `schemas/upload.py`

```python
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
```

### 4.3 `schemas/error.py`

```python
from pydantic import BaseModel
from typing import Any, Optional


class ErrorDetail(BaseModel):
    code: str
    message: str
    details: Optional[Any] = None


class ErrorResponse(BaseModel):
    error: ErrorDetail
```

---

## 5. RAG Chain 构建流程（LCEL）

### 5.1 设计目标

- 使用 LangChain **LCEL（LangChain Expression Language）** 声明式构建 Chain
- 支持 **每个知识库独立 Chain**：商品知识库 → `product_rag_chain`，政策知识库 → `policy_rag_chain`
- 意图路由层根据分类选择对应的 Chain 执行

### 5.2 Chain 结构图

```
用户问题
   │
   ▼
┌──────────────────────┐
│  CondenseQuestion     │  ← 带历史上下文重写问题
│  (历史 + 当前问 →     │
│   独立问题)           │
└─────────┬────────────┘
          │ rewritten_question
          ▼
┌──────────────────────┐
│  HybridRetriever      │  ← FAISS + BM25 + RRF
│  (top_k=5)           │
└─────────┬────────────┘
          │ retrieved_docs
          ▼
┌──────────────────────┐
│  format_docs          │  ← 文档→文本格式化
└─────────┬────────────┘
          │ context
          ▼
┌──────────────────────┐
│  QA Prompt            │  ← System + Context + Question
│  (带引用的prompt模板) │
└─────────┬────────────┘
          │ prompt
          ▼
┌──────────────────────┐
│  LLM                  │  ← 调用大模型生成
│  (ChatOpenAI / 其他)  │
└─────────┬────────────┘
          │ raw_answer
          ▼
┌──────────────────────┐
│  OutputParser         │  ← 提取回答 + 来源标注
│  (StrOutputParser)    │
└─────────┬────────────┘
          │ answer
          ▼
      返回给用户
```

### 5.3 代码实现 (`core/rag_chain.py`)

```python
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from operator import itemgetter
from typing import List

from app.core.hybrid_retriever import HybridRetriever
from app.services.llm import get_llm

# ── Prompt 模板 ──────────────────────────────────────────────

# 问题重写 prompt（带历史）
CONDENSE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "你是一个电商客服助手。给定对话历史和最新的用户问题，"
        "将其重写为一个可以在没有历史上下文的独立问题。"
        "如果问题本身就是独立的，直接返回原问题。"
        "只返回问题本身，不要附加任何解释。",
    ),
    ("placeholder", "{chat_history}"),
    ("human", "{question}"),
])

# RAG 回答 prompt
QA_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "你是一个专业的电商客服助手。请基于以下提供的上下文信息回答用户问题。\n\n"
        "要求：\n"
        "1. 如果上下文包含足够信息，给出准确、简洁的回答\n"
        "2. 如果上下文不足以回答，如实告知用户你不知道，不要编造\n"
        "3. 在回答末尾用 [来源: 文档名称] 标注信息来源\n"
        "4. 保持回答语气友好、专业\n\n"
        "上下文信息：\n{context}",
    ),
    ("human", "{question}"),
])


def format_docs(docs: List[Document]) -> str:
    """将 Document 列表格式化为模型可读的文本块"""
    formatted = []
    for i, doc in enumerate(docs):
        source = doc.metadata.get("source", "未知来源")
        formatted.append(f"[文档 {i+1}] (来源: {source})\n{doc.page_content}")
    return "\n\n---\n\n".join(formatted)


def build_rag_chain(
    retriever: HybridRetriever,
    *,
    use_history: bool = True,
) -> RunnableParallel:
    """构建 LCEL RAG Chain

    Args:
        retriever: 混合检索器实例
        use_history: 是否启用对话历史重写（上线初期可关闭以简化排查）

    Returns:
        一个 Runnable 对象，输入 dict 含 question/chat_history，输出含 answer/docs
    """
    if use_history:
        # ── 带历史重写的完整 Chain ──
        chain = (
            RunnableParallel(
                # 先重写问题
                rewritten_question=RunnableParallel(
                    question=itemgetter("question"),
                    chat_history=itemgetter("chat_history"),
                ) | CONDENSE_PROMPT | get_llm() | StrOutputParser(),
                # 同时保留原始 question 用于 fallback
                original_question=itemgetter("question"),
                chat_history=itemgetter("chat_history"),
            )
            | RunnableParallel(
                context=(
                    # 用重写后的问题去检索
                    itemgetter("rewritten_question")
                    | retriever
                    | format_docs
                ),
                question=itemgetter("rewritten_question"),
                original_question=itemgetter("original_question"),
            )
        )
    else:
        # ── 简单版：无历史重写 ──
        chain = (
            RunnableParallel(
                context=itemgetter("question") | retriever | format_docs,
                question=itemgetter("question"),
            )
        )

    # 接上 QA prompt + LLM
    full_chain = chain | QA_PROMPT | get_llm() | StrOutputParser()

    return full_chain
```

### 5.4 Chain 调用示例

```python
chain = build_rag_chain(retriever=hybrid_retriever)

result = chain.invoke({
    "question": "这款手机支持5G吗？",
    "chat_history": [
        ("human", "你好"),
        ("ai", "你好！请问有什么可以帮助您的？"),
    ],
})
# result: "该手机支持 5G 全网通。[来源: 产品规格书.pdf]"
```

---

## 6. 意图路由逻辑

### 6.1 策略概述

采用 **两阶段分类** 策略：

1. **第一阶段（快速通道）**：关键词规则匹配，0 延迟，覆盖确定性场景
2. **第二阶段（LLM 分类）**：第一阶段无法确定时，由 LLM 做语义分类

```
用户消息
   │
   ▼
┌─────────────────────┐
│  关键词规则匹配       │──── 命中 ──▶ 返回意图
│  (正则 + 关键词字典)  │
└─────────┬───────────┘
          │ 未命中
          ▼
┌─────────────────────┐
│  LLM 语义分类         │──── 返回意图
│  (few-shot prompt)   │
└─────────────────────┘
```

### 6.2 意图分类定义

| 意图 ID | 含义 | 路由目标 | 示例 Query |
|---------|------|----------|-----------|
| `product_inquiry` | 商品咨询 | 商品 RAG Chain | "这款手机多少钱？" |
| `return_exchange` | 退换货咨询 | 政策 RAG Chain | "7天无理由怎么退？" |
| `order_tracking` | 订单查询 | Tool（查单工具） | "我的订单到哪了？" |
| `chitchat` | 闲聊 | 直接 LLM（无检索） | "你好，今天天气不错" |

### 6.3 代码实现 (`core/intent_router.py`)

```python
import re
from enum import Enum
from typing import Optional

from app.services.llm import get_llm
from langchain_core.prompts import ChatPromptTemplate


class Intent(str, Enum):
    PRODUCT_INQUIRY = "product_inquiry"
    RETURN_EXCHANGE = "return_exchange"
    ORDER_TRACKING = "order_tracking"
    CHITCHAT = "chitchat"


# ── 关键词规则字典 ────────────────────────────────────────────
# 每个意图对应一组正则模式，命中即快速返回

KEYWORD_RULES: dict[Intent, list[re.Pattern]] = {
    Intent.PRODUCT_INQUIRY: [
        re.compile(r"(价格|多少钱|售价|优惠|折扣|包邮|规格|参数|尺寸|颜色|版本|保修期)"),
        re.compile(r"(有货|库存|现货|预售|什么时候.*上)"),
        re.compile(r"(推荐|哪款|适合|对比|区别|好不?)"),
    ],
    Intent.RETURN_EXCHANGE: [
        re.compile(r"(退货|退款|换货|退换|售后|维修|返修)"),
        re.compile(r"(无理由|拒收|取消订单|质量问题|补偿)"),
        re.compile(r"(退钱|退差价|补发|少发|错发)"),
    ],
    Intent.ORDER_TRACKING: [
        re.compile(r"(订单|物流|快递|发货|配送|签收|运单号)"),
        re.compile(r"(到哪|多久.*到|什么时候.*到|配送中)"),
    ],
}

# ── LLM 分类 prompt ──────────────────────────────────────────

INTENT_CLASSIFY_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "你是一个电商客服意图分类器。请判断用户问题的意图类别，只返回一个类别名称。\n\n"
        "类别定义：\n"
        "- product_inquiry: 商品咨询（问价格、规格、推荐、库存等）\n"
        "- return_exchange: 退换货/售后（退货、退款、换货、维修等）\n"
        "- order_tracking: 订单查询（物流、配送、发货时间等）\n"
        "- chitchat: 闲聊/问候（打招呼、感谢、非业务相关）\n\n"
        "只输出类别名，不要附加任何其他文字。",
    ),
    ("human", "{message}"),
])

# ── 路由错误消息 ─────────────────────────────────────────────

UNKNOWN_INTENT_MSG = (
    "抱歉，我没能理解您的问题类型。您可以尝试以下方式：\n"
    "1. 咨询商品信息，例如「这款手机多少钱？」\n"
    "2. 查询退换货政策，例如「怎么退货？」\n"
    "3. 查询订单状态，例如「我的订单到哪了？」"
)


class IntentRouter:
    """意图路由器 — 关键词规则 + LLM 回退"""

    def __init__(self, use_llm_fallback: bool = True):
        self.use_llm_fallback = use_llm_fallback

    def classify(self, message: str) -> Intent:
        """对用户消息进行意图分类"""

        # ── Step 1: 快速通道 — 关键词匹配 ──
        for intent, patterns in KEYWORD_RULES.items():
            for pattern in patterns:
                if pattern.search(message):
                    return intent

        # ── Step 2: LLM 语义分类 ──
        if self.use_llm_fallback:
            try:
                chain = INTENT_CLASSIFY_PROMPT | get_llm(temperature=0)
                result = chain.invoke({"message": message}).content.strip().lower()

                # 校验返回值是否为合法 Intent
                for intent in Intent:
                    if intent.value in result:
                        return intent
            except Exception:
                # LLM 调用失败时优雅降级到 chitchat
                pass

        # ── Step 3: 默认降级 ──
        return Intent.CHITCHAT

    def route(self, message: str) -> dict:
        """路由：返回意图 + 是否需要 RAG / Tool / 直答"""
        intent = self.classify(message)

        routing_info = {
            "intent": intent,
            "use_rag": intent in (Intent.PRODUCT_INQUIRY, Intent.RETURN_EXCHANGE),
            "use_tool": intent == Intent.ORDER_TRACKING,
            "knowledge_base": (
                "product" if intent == Intent.PRODUCT_INQUIRY
                else "policy" if intent == Intent.RETURN_EXCHANGE
                else None
            ),
        }
        return routing_info
```

### 6.4 API 路由层调用示例 (`api/chat.py`)

```python
from fastapi import APIRouter, HTTPException
from app.schemas.chat import ChatRequest, ChatResponse
from app.schemas.error import ErrorResponse
from app.core.intent_router import IntentRouter
from app.core.rag_chain import build_rag_chain
from app.core.hybrid_retriever import HybridRetriever
from app.core.session import SessionManager

router = APIRouter()
intent_router = IntentRouter()
session_mgr = SessionManager()

# 两个知识库的检索器和 Chain（启动时初始化）
product_chain = None
policy_chain = None


@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    # 1. 获取/创建会话历史
    history = session_mgr.get_history(req.session_id)

    # 2. 意图路由
    route = intent_router.route(req.message)

    # 3. 按路由执行
    try:
        if route["use_rag"]:
            chain = (
                product_chain if route["knowledge_base"] == "product"
                else policy_chain
            )
            if chain is None:
                raise HTTPException(
                    status_code=400,
                    detail={
                        "code": "RAG_EMPTY_KNOWLEDGE_BASE",
                        "message": f"知识库 '{route['knowledge_base']}' 为空，请先上传文档",
                    },
                )
            answer = chain.invoke({
                "question": req.message,
                "chat_history": history,
            })

        elif route["use_tool"]:
            from app.tools.order_tool import query_order
            answer = query_order(req.message)

        else:  # chitchat
            from app.services.llm import get_llm
            llm = get_llm()
            answer = llm.invoke(req.message).content

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "code": "RAG_CHAIN_ERROR",
                "message": f"处理请求时出错: {str(e)}",
            },
        )

    # 4. 保存历史
    session_mgr.add_message(req.session_id, "human", req.message)
    session_mgr.add_message(req.session_id, "ai", answer)

    # 5. 返回
    return ChatResponse(
        answer=answer,
        session_id=req.session_id,
        intent=route["intent"].value,
        conversation_id=f"conv_{req.session_id}",
    )
```

---

## 7. Hybrid Retrieval（FAISS + BM25 + RRF）

### 7.1 设计思路

结合 **稠密检索（dense retrieval）** 与 **稀疏检索（sparse retrieval）** 的长处：

| 方法 | 优势 | 劣势 |
|------|------|------|
| **FAISS**（稠密） | 理解语义相似性；同义改写也能命中 | 对精确关键词匹配不敏感；需要 GPU/好的 embedding |
| **BM25**（稀疏） | 精确术语匹配强；速度快；零部署成本 | 语义鸿沟（"手机"和"移动终端"不匹配） |
| **RRF 融合** | 综合二者排序，鲁棒性高 | 需要调参 k（RRF 常数） |

### 7.2 RRF 算法

```
RRFscore(d) = Σ 1 / (k + rank_i(d))

其中：
  - rank_i(d) 是文档 d 在第 i 个检索器中的排名
  - k 是常数（通常取 60，防止单个检索器主导）
```

### 7.3 代码实现 (`core/hybrid_retriever.py`)

```python
import pickle
from pathlib import Path
from typing import List, Optional

import numpy as np
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever

from app.services.embedding import get_embedding


class HybridRetriever(BaseRetriever):
    """FAISS + BM25 混合检索器，使用 RRF 融合排序

    使用方式：
        retriever = HybridRetriever(
            faiss_index=faiss_vectorstore,
            bm25_index=bm25_retriever,
            top_k=5,
            rrf_k=60,
        )
        docs = retriever.invoke("查询文本")
    """

    faiss_index: Optional[FAISS] = None
    bm25_index: Optional[BM25Retriever] = None
    top_k: int = 5
    rrf_k: int = 60
    faiss_weight: float = 1.0
    bm25_weight: float = 1.0

    def __post_init__(self):
        """验证至少有一个检索器可用"""
        if self.faiss_index is None and self.bm25_index is None:
            raise ValueError("faiss_index 和 bm25_index 不能同时为 None")

    def _get_relevant_documents(self, query: str) -> List[Document]:
        """RRF 融合检索"""

        # ── 并行检索 ──
        faiss_results: List[Document] = []
        bm25_results: List[Document] = []

        if self.faiss_index is not None:
            faiss_results = self.faiss_index.similarity_search_with_relevance_scores(
                query, k=self.top_k * 2  # 多取一些供给 RRF 排序
            )
            # 将 score 存入 metadata
            faiss_results = [
                Document(
                    page_content=doc.page_content,
                    metadata={**doc.metadata, "__faiss_score": float(score)},
                )
                for doc, score in faiss_results
            ]

        if self.bm25_index is not None:
            bm25_results = self.bm25_index.get_relevant_documents(query)
            # BM25 取 top_k * 2 个
            bm25_results = bm25_results[: self.top_k * 2]

        # ── RRF 融合 ──
        doc_scores: dict[str, float] = {}

        # FAISS 贡献
        for rank, doc in enumerate(faiss_results):
            doc_id = doc.metadata.get("chunk_id", doc.page_content[:50])
            doc_scores[doc_id] = doc_scores.get(doc_id, 0.0) + self.faiss_weight / (
                self.rrf_k + rank + 1
            )

        # BM25 贡献
        for rank, doc in enumerate(bm25_results):
            doc_id = doc.metadata.get("chunk_id", doc.page_content[:50])
            doc_scores[doc_id] = doc_scores.get(doc_id, 0.0) + self.bm25_weight / (
                self.rrf_k + rank + 1
            )

        # ── 按 RRF 得分排序 ──
        ranked_ids = sorted(doc_scores, key=doc_scores.get, reverse=True)[:self.top_k]

        # ── 去重取最终结果 ──
        seen_ids = set()
        final_docs = []
        for doc in faiss_results + bm25_results:
            doc_id = doc.metadata.get("chunk_id", doc.page_content[:50])
            if doc_id in ranked_ids and doc_id not in seen_ids:
                doc.metadata["rrf_score"] = round(doc_scores[doc_id], 4)
                final_docs.append(doc)
                seen_ids.add(doc_id)

        return final_docs[: self.top_k]

    # ── 工厂方法 ──────────────────────────────────────────────

    @classmethod
    def from_knowledge_base(
        cls,
        kb_name: str,
        top_k: int = 5,
        rrf_k: int = 60,
    ) -> "HybridRetriever":
        """从持久化目录加载指定知识库的检索器"""
        base_path = Path("data")

        # FAISS
        faiss_path = base_path / "vector_store" / kb_name
        faiss_index = None
        if faiss_path.exists():
            embedding = get_embedding()
            faiss_index = FAISS.load_local(
                str(faiss_path),
                embeddings=embedding,
                allow_dangerous_deserialization=True,
            )

        # BM25
        bm25_path = base_path / "bm25_index" / f"{kb_name}.pkl"
        bm25_index = None
        if bm25_path.exists():
            with open(bm25_path, "rb") as f:
                bm25_index = pickle.load(f)

        return cls(
            faiss_index=faiss_index,
            bm25_index=bm25_index,
            top_k=top_k,
            rrf_k=rrf_k,
        )
```

### 7.4 FAISS 向量库服务 (`services/vector_store.py`)

```python
import pickle
from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy

from app.services.embedding import get_embedding


class VectorStoreService:
    """FAISS 向量库管理：构建、增量添加、持久化、加载"""

    def __init__(self, kb_name: str):
        self.kb_name = kb_name
        self.store_path = Path("data") / "vector_store" / kb_name
        self.store_path.mkdir(parents=True, exist_ok=True)
        self.embedding = get_embedding()
        self._index: FAISS | None = None

    def build(self, docs: List[Document]) -> int:
        """从文档列表构建向量库（全量重建）"""
        self._index = FAISS.from_documents(
            documents=docs,
            embedding=self.embedding,
            distance_strategy=DistanceStrategy.COSINE,
        )
        self._save()
        return len(docs)

    def add_documents(self, docs: List[Document]) -> int:
        """增量添加文档"""
        if self._index is None:
            self._index = self._load()
        if self._index is None:
            return self.build(docs)

        self._index.add_documents(docs)
        self._save()
        return len(docs)

    def _save(self):
        if self._index is not None:
            self._index.save_local(str(self.store_path))

    def _load(self) -> FAISS | None:
        if (self.store_path / "index.faiss").exists():
            return FAISS.load_local(
                str(self.store_path),
                embeddings=self.embedding,
                allow_dangerous_deserialization=True,
            )
        return None

    def get_index(self) -> FAISS | None:
        if self._index is None:
            self._index = self._load()
        return self._index
```

### 7.5 BM25 索引服务 (`services/bm25_store.py`)

```python
import pickle
from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever


class BM25StoreService:
    """BM25 索引管理"""

    def __init__(self, kb_name: str):
        self.kb_name = kb_name
        self.index_path = Path("data") / "bm25_index" / f"{kb_name}.pkl"
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self._retriever: BM25Retriever | None = None

    def build(self, docs: List[Document]) -> int:
        self._retriever = BM25Retriever.from_documents(docs)
        self._save()
        return len(docs)

    def add_documents(self, docs: List[Document]) -> int:
        if self._retriever is None:
            self._retriever = self._load()
        if self._retriever is None:
            return self.build(docs)

        self._retriever.add_documents(docs)
        self._save()
        return len(docs)

    def _save(self):
        if self._retriever is not None:
            with open(self.index_path, "wb") as f:
                pickle.dump(self._retriever, f)

    def _load(self) -> BM25Retriever | None:
        if self.index_path.exists():
            with open(self.index_path, "rb") as f:
                return pickle.load(f)
        return None

    def get_retriever(self) -> BM25Retriever | None:
        if self._retriever is None:
            self._retriever = self._load()
        return self._retriever
```

---

## 8. 会话管理

### 8.1 设计

- **存储方式**：内存字典（`dict[str, list]`），进程内有效
- **历史结构**：`[("human", msg), ("ai", reply), ...]` 符合 LangChain 的 `BaseChatMessageHistory`
- **容量限制**：每个 session 最多保留最近 20 轮对话（40 条消息），超出时裁剪

### 8.2 代码实现 (`core/session.py`)

```python
from typing import List, Tuple
import time
import threading


class SessionManager:
    """内存会话管理器，支持 session_id 粒度的对话历史"""

    MAX_ROUNDS = 20  # 最大保留轮数

    def __init__(self):
        self._store: dict[str, list] = {}
        self._lock = threading.Lock()

    def get_history(self, session_id: str) -> List[Tuple[str, str]]:
        """获取指定会话的历史记录"""
        with self._lock:
            return list(self._store.get(session_id, []))

    def add_message(self, session_id: str, role: str, content: str):
        """添加一条消息到会话历史，自动裁剪超长历史"""
        with self._lock:
            if session_id not in self._store:
                self._store[session_id] = []
            self._store[session_id].append((role, content))
            # 裁剪：保留最近 MAX_ROUNDS 轮（每轮 1 human + 1 ai = 2 条）
            max_messages = self.MAX_ROUNDS * 2
            if len(self._store[session_id]) > max_messages:
                self._store[session_id] = self._store[session_id][-max_messages:]

    def clear(self, session_id: str):
        """清除指定会话的历史"""
        with self._lock:
            self._store.pop(session_id, None)

    def cleanup_stale(self, max_age_seconds: int = 3600):
        """清理超过指定时间未活跃的会话（可由定时任务触发）"""
        # 简化实现：标记时间戳的扩展留作后续
        pass
```

### 8.3 限制说明

| 限制项 | 说明 | 建议的改进方向 |
|--------|------|---------------|
| **进程内存储** | 服务重启后历史丢失 | 切换为 Redis |
| **单实例** | 水平扩展时 session 不共享 | 使用 Redis + 一致性哈希 |
| **无过期机制** | 可能内存泄漏 | 引入 TTL 或 LRU 淘汰 |

---

## 9. 文档上传与向量化建库流程

### 9.1 处理流程

```
POST /upload (file)
   │
   ▼
┌──────────────┐
│ 1. 文件校验   │  ← 格式、大小、病毒扫描（预留）
└──────┬───────┘
       ▼
┌──────────────┐
│ 2. 保存原始   │  ← data/documents/{kb_name}/{doc_id}.bin
│    文件       │
└──────┬───────┘
       ▼
┌──────────────┐
│ 3. 文档加载   │  ← Loader：PDF→PyMuPDFLoader, MD→UnstructuredMarkdownLoader
└──────┬───────┘
       ▼
┌──────────────┐
│ 4. 文本分块   │  ← RecursiveCharacterTextSplitter
│              │     chunk_size=512, chunk_overlap=50
└──────┬───────┘
       ▼
┌──────────────┐
│ 5. 生成       │
│    Embedding  │  ← 调用 embedding 模型
└──────┬───────┘
       ▼
┌──────────────┐
│ 6. 写入       │
│    FAISS     │  ← 增量添加 / 全量重建
└──────┬───────┘
       ▼
┌──────────────┐
│ 7. 写入       │
│    BM25      │  ← 增量添加 / 全量重建
└──────┬───────┘
       ▼
┌──────────────┐
│ 8. 返回结果   │
└──────────────┘
```

### 9.2 代码实现 (`api/upload.py`)

```python
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

router = APIRouter()

ALLOWED_EXTENSIONS = {".pdf", ".md", ".txt"}
MAX_FILE_SIZE = 20 * 1024 * 1024  # 20MB


@router.post("/upload", response_model=UploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    knowledge_base: Optional[str] = Form("product"),
):
    # ── 1. 文件校验 ──
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

    # ── 2. 保存原始文件 ──
    doc_id = f"doc_{uuid.uuid4().hex[:12]}"
    save_dir = Path("data") / "documents" / knowledge_base
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"{doc_id}{ext}"
    save_path.write_bytes(content)

    # ── 3. 加载 + 分块 ──
    raw_docs = load_document(save_path, doc_id=doc_id)
    chunks = split_documents(raw_docs)

    # ── 4. 写入 FAISS + BM25 ──
    t0 = time.time()
    vs = VectorStoreService(kb_name=knowledge_base)
    bm25 = BM25StoreService(kb_name=knowledge_base)

    try:
        vs.add_documents(chunks)
        bm25.add_documents(chunks)
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
```

### 9.3 知识库文档处理 (`knowledge/`)

```python
# knowledge/loader.py

from pathlib import Path
from typing import List

from langchain_core.documents import Document


def load_document(file_path: Path, doc_id: str) -> List[Document]:
    """根据文件扩展名选择合适的 Loader 加载文档"""
    ext = file_path.suffix.lower()

    if ext == ".pdf":
        from langchain_community.document_loaders import PyMuPDFLoader
        loader = PyMuPDFLoader(str(file_path))
    elif ext == ".md":
        from langchain_community.document_loaders import UnstructuredMarkdownLoader
        loader = UnstructuredMarkdownLoader(str(file_path))
    elif ext == ".txt":
        from langchain_community.document_loaders import TextLoader
        loader = TextLoader(str(file_path), encoding="utf-8")
    else:
        raise ValueError(f"Unsupported file format: {ext}")

    docs = loader.load()
    # 注入元数据
    for doc in docs:
        doc.metadata["doc_id"] = doc_id
        doc.metadata["source"] = file_path.name
    return docs


# knowledge/splitter.py

from typing import List
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter


def split_documents(
    docs: List[Document],
    chunk_size: int = 512,
    chunk_overlap: int = 50,
) -> List[Document]:
    """将文档切块，注入 chunk_id 元数据"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", ".", " ", ""],
        length_function=len,
    )
    chunks = splitter.split_documents(docs)

    for idx, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = f"{chunk.metadata.get('doc_id', 'unknown')}_chunk_{idx}"
        chunk.metadata["chunk_index"] = idx

    return chunks
```

---

## 10. 错误码设计

### 10.1 错误码分类

采用 `CATEGORY_SPECIFIC_ERROR` 命名规范，方便前端按 code 做国际化。

| 状态码 | 错误码 | 含义 | 触发条件 |
|--------|--------|------|---------|
| **400** | `CHAT_MESSAGE_EMPTY` | 消息为空 | `message` 为空或纯空白 |
| 400 | `CHAT_MESSAGE_TOO_LONG` | 消息超长 | `message` > 2000 字符 |
| 400 | `CHAT_SESSION_INVALID` | 无效 session_id | `session_id` 为空或含非法字符 |
| **400** | `UPLOAD_UNSUPPORTED_FORMAT` | 不支持的文件格式 | 上传非 .pdf/.md/.txt 文件 |
| 400 | `UPLOAD_FILE_TOO_LARGE` | 文件超限 | 文件 > 20MB |
| **400** | `RAG_EMPTY_KNOWLEDGE_BASE` | 知识库为空 | 上传前即发起 chat |
| 400 | `RAG_NO_RETRIEVAL_RESULTS` | 检索无结果 | 知识库无匹配文档 |
| **500** | `RAG_CHAIN_ERROR` | RAG Chain 执行失败 | LLM 调用失败 / 超时 |
| 500 | `UPLOAD_INDEX_ERROR` | 索引构建失败 | FAISS/BM25 写入异常 |
| 500 | `LLM_SERVICE_UNAVAILABLE` | LLM 服务不可用 | API key 无效 / 配额超限 |
| 500 | `EMBEDDING_SERVICE_UNAVAILABLE` | Embedding 服务不可用 | Embedding API 调用失败 |
| 500 | `INTERNAL_SERVER_ERROR` | 未预期错误 | 兜底 catch |

### 10.2 异常与错误码映射 (`core/errors.py`)

```python
from enum import Enum
from typing import Any, Optional


class ErrorCode(str, Enum):
    # ── Chat 相关 ──
    CHAT_MESSAGE_EMPTY = "CHAT_MESSAGE_EMPTY"
    CHAT_MESSAGE_TOO_LONG = "CHAT_MESSAGE_TOO_LONG"
    CHAT_SESSION_INVALID = "CHAT_SESSION_INVALID"

    # ── Upload 相关 ──
    UPLOAD_UNSUPPORTED_FORMAT = "UPLOAD_UNSUPPORTED_FORMAT"
    UPLOAD_FILE_TOO_LARGE = "UPLOAD_FILE_TOO_LARGE"
    UPLOAD_INDEX_ERROR = "UPLOAD_INDEX_ERROR"

    # ── RAG 相关 ──
    RAG_EMPTY_KNOWLEDGE_BASE = "RAG_EMPTY_KNOWLEDGE_BASE"
    RAG_NO_RETRIEVAL_RESULTS = "RAG_NO_RETRIEVAL_RESULTS"
    RAG_CHAIN_ERROR = "RAG_CHAIN_ERROR"

    # ── 基础设施 ──
    LLM_SERVICE_UNAVAILABLE = "LLM_SERVICE_UNAVAILABLE"
    EMBEDDING_SERVICE_UNAVAILABLE = "EMBEDDING_SERVICE_UNAVAILABLE"
    INTERNAL_SERVER_ERROR = "INTERNAL_SERVER_ERROR"


class AppException(Exception):
    """业务异常基类"""

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
```

### 10.3 全局异常处理器 (`main.py`)

```python
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from app.core.errors import AppException, ErrorCode

app = FastAPI(title="电商智能客服 RAG API", version="1.0.0")


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
    """兜底——500 未预期异常"""
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
```

---

## 11. 配置与环境变量

### 11.1 `config.py`

```python
from pydantic_settings import BaseSettings
from typing import Literal


class Settings(BaseSettings):
    """应用配置，优先从环境变量读取，支持 .env 文件"""

    # ── 应用基础 ──
    app_name: str = "电商智能客服 RAG"
    debug: bool = False
    host: str = "0.0.0.0"
    port: int = 8000

    # ── LLM ──
    llm_provider: Literal["openai", "deepseek", "local"] = "openai"
    llm_model_name: str = "gpt-4o-mini"
    llm_api_key: str = ""
    llm_api_base: str = ""
    llm_temperature: float = 0.1
    llm_max_tokens: int = 2048

    # ── Embedding ──
    embedding_provider: Literal["openai", "local"] = "openai"
    embedding_model_name: str = "text-embedding-3-small"
    embedding_api_key: str = ""
    embedding_dimension: int = 1536

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
```

### 11.2 `.env.example`

```bash
# ── 应用 ──
DEBUG=false
HOST=0.0.0.0
PORT=8000

# ── LLM ──
LLM_PROVIDER=openai
LLM_MODEL_NAME=gpt-4o-mini
LLM_API_KEY=sk-xxxxxxxxxxxxxxxx
LLM_API_BASE=
LLM_TEMPERATURE=0.1
LLM_MAX_TOKENS=2048

# ── Embedding ──
EMBEDDING_PROVIDER=openai
EMBEDDING_MODEL_NAME=text-embedding-3-small
EMBEDDING_API_KEY=sk-xxxxxxxxxxxxxxxx

# ── Retrieval ──
RETRIEVER_TOP_K=5
RETRIEVER_RRF_K=60

# ── Document Processing ──
CHUNK_SIZE=512
CHUNK_OVERLAP=50
MAX_FILE_SIZE_MB=20
```

---

## 12. 边界情况与异常处理策略

### 12.1 Chat 接口特殊场景

| 场景 | 处理方式 |
|------|---------|
| **知识库为空** | 返回 `400 RAG_EMPTY_KNOWLEDGE_BASE`，引导用户先上传文档 |
| **检索无结果** | 降级：告诉用户未找到相关信息，建议重新表述问题 |
| **LLM 超时** | FastAPI 设置 `timeout` 中间件，超时返回 `504` |
| **消息过长** | Pydantic `max_length=2000` 校验，超标返回 `400` |
| **频繁请求** | 预留 Rate Limiter 接口（`slowapi` 集成） |
| **session_id 不存在** | 自动创建新会话（非错误，静默处理） |
| **特殊字符/XSS** | Pydantic validator 做 strip + 只允许合理字符集 |

### 12.2 Upload 接口特殊场景

| 场景 | 处理方式 |
|------|---------|
| **空文件** | 返回 `400`，提示文件内容为空 |
| **超大文件** | 在读入时立即校验 size 并拒绝 |
| **重复上传** | 以 doc_id 去重（同内容返回已存在的 doc_id） |
| **部分 chunk 索引失败** | 返回 `207 Partial`，注明失败数量 |
| **编码问题 (非 UTF-8)** | `TextLoader` 使用 `autodetect_encoding=True` 或捕获 `UnicodeDecodeError` |
| **PDF 含图片/扫描件** | 当前返回 OCR 受限提示，后续可集成 OCR 服务 |

### 12.3 优雅降级策略

```
                    ┌─ 双检索器正常 ──→ Hybrid Retrieval
                   │
[检索阶段] ──┤
                   │
                    └─ FAISS 异常 ──→ 仅用 BM25（记录告警日志）
                     └─ BM25 异常 ──→ 仅用 FAISS（记录告警日志）
                      └─ 双异常 ──→ 返回空上下文 → LLM 自行回答

                    ┌─ LLM 正常 ──→ 正常回答
                   │
[生成阶段] ──┤
                   │
                    └─ LLM 异常 ──→ 返回预设兜底消息（"系统正忙，请稍后再试"）
```

---

## 附录 A：`requirements.txt`

```txt
# Web framework
fastapi>=0.115.0
uvicorn[standard]>=0.30.0
python-multipart>=0.0.12

# Pydantic
pydantic>=2.0,<3.0
pydantic-settings>=2.0

# LangChain
langchain>=0.3.0
langchain-community>=0.3.0
langchain-openai>=0.2.0
langchain-core>=0.3.0

# Vector store
faiss-cpu>=1.8.0

# Document processing
pypdf>=4.0
pymupdf>=1.24.0
unstructured>=0.15.0

# Others
numpy>=1.26.0
python-dotenv>=1.0.0

# Testing
pytest>=8.0
pytest-asyncio>=0.24.0
httpx>=0.27.0  # for TestClient
```

## 附录 B：应用入口 (`main.py`)

```python
import uvicorn
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.api import chat, upload
from app.core.hybrid_retriever import HybridRetriever
from app.core.rag_chain import build_rag_chain


# ── 全局状态（按知识库持有检索器 & Chain） ──
product_retriever: HybridRetriever | None = None
policy_retriever: HybridRetriever | None = None
product_chain = None
policy_chain = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期：启动时加载索引，关闭时清理"""
    global product_retriever, policy_retriever, product_chain, policy_chain

    # 启动加载
    product_retriever = HybridRetriever.from_knowledge_base("product")
    policy_retriever = HybridRetriever.from_knowledge_base("policy")

    if product_retriever.faiss_index is not None:
        product_chain = build_rag_chain(retriever=product_retriever)
    if policy_retriever.faiss_index is not None:
        policy_chain = build_rag_chain(retriever=policy_retriever)

    yield

    # 关闭清理（当前无资源需要释放，留作扩展点）
    pass


app = FastAPI(
    title=settings.app_name,
    version="1.0.0",
    lifespan=lifespan,
)

# ── 中间件 ──
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── 注册路由 ──
app.include_router(chat.router, prefix="/api/v1", tags=["Chat"])
app.include_router(upload.router, prefix="/api/v1", tags=["Upload"])


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
    )
```

---

> **文档说明**  
> 本文档覆盖了《电商智能客服 RAG 系统》后端的完整架构设计。  
> 各模块之间的依赖关系清晰可追溯：路由层 → Schema 层 → 核心业务层 → 基础设施服务层。  
> 每个接口和关键组件都考虑了正常路径与异常路径，并给出了对应的处理策略。
