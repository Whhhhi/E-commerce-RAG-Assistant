# E-commerce RAG Assistant

基于 LangChain + ChromaDB 的轻量化私有知识库问答系统，适配电商客服场景。

## 项目特点

- **混合检索**：ChromaDB 向量检索 + BM25 关键词检索，使用 RRF（Reciprocal Rank Fusion）融合排序
- **多模型支持**：支持 OpenAI、DeepSeek、本地模型（LM Studio / llama.cpp / Ollama）
- **多文档格式**：支持 PDF、Markdown、TXT、DOCX 文件解析
- **意图路由**：自动识别用户意图，路由到对应知识库或工具
- **多轮对话**：基于 SessionManager 的对话历史管理，支持连续问答
- **幻觉抑制**：优化的 Prompt 模板，要求模型基于上下文回答，标注信息来源

## 项目结构

```
E-commerce RAG Assistant/
├── app/
│   ├── api/                 # API 接口层
│   │   ├── chat.py          # 聊天接口
│   │   └── upload.py        # 文档上传接口
│   ├── core/                # 核心业务逻辑
│   │   ├── rag_chain.py     # RAG 链构建
│   │   ├── hybrid_retriever.py  # 混合检索器
│   │   ├── intent_router.py     # 意图路由
│   │   ├── session.py       # 会话管理
│   │   └── errors.py        # 错误处理
│   ├── services/            # 服务层
│   │   ├── llm.py           # LLM 服务
│   │   ├── embedding.py     # Embedding 服务
│   │   ├── vector_store.py  # ChromaDB 向量存储
│   │   └── bm25_store.py    # BM25 索引存储
│   ├── knowledge/           # 知识库处理
│   │   ├── loader.py        # 文档加载
│   │   └── splitter.py      # 文档切片
│   ├── tools/               # 工具层
│   │   └── order_tool.py    # 订单查询工具
│   ├── schemas/             # 数据模型
│   ├── config.py            # 配置管理
│   └── main.py              # 应用入口
├── tests/                   # 测试文件
├── data/                    # 数据目录
│   ├── documents/           # 原始文档
│   ├── chroma_db/           # ChromaDB 持久化
│   └── bm25_index/          # BM25 索引
├── .env.example             # 配置示例
└── README.md
```

## 快速开始

### 1. 环境准备

```bash
# 克隆项目
git clone <repository_url>
cd E-commerce RAG Assistant

# 创建虚拟环境
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# 安装依赖
pip install -r requirements.txt
```

### 2. 配置环境变量

复制 `.env.example` 为 `.env`，根据实际需求修改配置：

```bash
cp .env.example .env
```

**关键配置项**：

| 配置项 | 说明 | 默认值 |
|--------|------|--------|
| `LLM_PROVIDER` | LLM 提供商：`openai` / `deepseek` / `local` | `openai` |
| `LLM_API_KEY` | API 密钥 | - |
| `LLM_MODEL_NAME` | 模型名称 | `gpt-4o-mini` |
| `LLM_API_BASE` | 本地模型 API 地址 | `http://127.0.0.1:1234/v1` |
| `EMBEDDING_PROVIDER` | Embedding 提供商：`openai` / `local` | `openai` |
| `CHUNK_SIZE` | 文档切片大小 | `512` |
| `RETRIEVER_TOP_K` | 检索返回数量 | `5` |

### 3. 启动服务

```bash
# 启动 API 服务
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

访问 http://localhost:8000/docs 查看 API 文档。

## API 接口

### 文档上传

```http
POST /upload
Content-Type: multipart/form-data

参数：
- file: 上传的文件（PDF/MD/TXT/DOCX）
- knowledge_base: 知识库名称（product/policy）

响应：
{
  "document_id": "doc_xxx",
  "file_name": "产品说明书.pdf",
  "knowledge_base": "product",
  "chunks": 15,
  "status": "indexed",
  "elapsed_seconds": 2.5
}
```

### 智能问答

```http
POST /chat
Content-Type: application/json

请求体：
{
  "session_id": "user_001",
  "message": "这个产品的保修期是多久？"
}

响应：
{
  "answer": "根据产品说明书，保修期为1年...",
  "sources": [],
  "session_id": "user_001",
  "intent": "product_query",
  "conversation_id": "conv_user_001",
  "latency_ms": 1500
}
```

### 健康检查

```http
GET /health

响应：
{
  "status": "healthy"
}
```

## 核心模块说明

### 混合检索器（HybridRetriever）

结合 ChromaDB 向量检索和 BM25 关键词检索，使用 RRF 算法融合排序：

```
用户查询 → ChromaDB检索(top_k*2) → 向量相似度排序
         → BM25检索(top_k*2)     → 关键词匹配排序
         → RRF融合               → 最终top_k结果
```

**RRF 公式**：
```
score(d) = Σ (1 / (k + rank(d)))
```

### 意图路由（IntentRouter）

自动识别用户意图，路由到对应处理模块：

| 意图类型 | 处理方式 | 示例 |
|----------|----------|------|
| `product_query` | 商品知识库 RAG | "这个产品有什么功能？" |
| `policy_query` | 政策知识库 RAG | "退货政策是怎样的？" |
| `order_query` | 订单查询工具 | "我的订单状态是什么？" |
| `general_chat` | 直接 LLM 回答 | "你好" |

### RAG 链（RAG Chain）

完整的问答处理流程：

```
用户问题 + 历史对话 → 问题重写（Condense） → 混合检索 → 上下文构建 → LLM生成 → 返回答案
```

**Prompt 模板**：
- 要求模型基于上下文回答
- 上下文不足时如实告知
- 标注信息来源

## 本地模型配置

### LM Studio

1. 下载并安装 LM Studio
2. 加载支持的模型（如 Chinese-Llama-2-7B）
3. 启动本地服务器（默认端口 1234）
4. 配置 `.env`：

```env
LLM_PROVIDER=local
LLM_API_BASE=http://127.0.0.1:1234/v1
LLM_MODEL_NAME=<模型名称>
```

### Ollama

```bash
# 安装 Ollama
# 拉取模型
ollama pull llama2-chinese

# 配置 .env
LLM_PROVIDER=local
LLM_API_BASE=http://127.0.0.1:11434/v1
LLM_MODEL_NAME=llama2-chinese
```

## 开发指南

### 运行测试

```bash
pytest tests/
```

### 添加新知识库

1. 上传文档时指定新的 `knowledge_base` 名称
2. 在 `intent_router.py` 中添加对应的意图识别规则
3. 在 `chat.py` 的 `init_chains()` 中初始化新的 RAG 链

### 自定义切片策略

修改 `app/knowledge/splitter.py`：

```python
def split_documents(docs, chunk_size=512, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "。", "！", "？", "；", "\n", ".", " ", ""]
    )
    # ...
```

## 技术栈

| 组件 | 技术 | 版本 |
|------|------|------|
| Web 框架 | FastAPI | 0.110+ |
| RAG 框架 | LangChain | 0.1+ |
| 向量数据库 | ChromaDB | - |
| 关键词检索 | BM25 | - |
| LLM | OpenAI / DeepSeek / 本地模型 | - |
| Embedding | OpenAI / 本地模型 | - |
| 文档解析 | PyMuPDF / Docx2txt | - |

## 常见问题

### Q: 上传文档后检索不到内容？

检查：
1. Embedding 服务是否正常启动
2. ChromaDB 持久化目录是否存在
3. 文档是否成功切片（查看返回的 chunks 数量）

### Q: 本地模型响应慢？

建议：
1. 使用量化模型（4-bit / 8-bit）
2. 减小 `LLM_MAX_TOKENS` 配置
3. 调整 `RETRIEVER_TOP_K` 减少上下文长度

### Q: 如何切换知识库？

在 `/chat` 请求中，系统会根据意图自动路由到对应知识库。如需手动指定，可修改 `intent_router.py` 的路由规则。

## 许可证

MIT License

## 贡献指南

欢迎提交 Issue 和 Pull Request。

---

**作者**：Whhhhi
**更新时间**：2026年