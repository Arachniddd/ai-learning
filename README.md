# RAG Agent Demo

一个基于 FastAPI、DeepSeek、ChromaDB 和 sentence-transformers 构建的课程资料 RAG Agent Demo。

项目支持上传 `.txt` / `.md` 文件，将文本切分成 chunk，生成 embedding 后写入本地 ChromaDB 向量库。用户提问时，系统可以进行查询改写、向量检索，并让 LLM 基于检索到的上下文回答问题。项目还包含一个简单前端页面、Agent 工具调用流程和 trace 日志。

## 当前能力

- 上传 `.txt` / `.md` 课程资料
- 支持普通文本切分、固定长度切分、Markdown 标题切分
- 使用 Pydantic 定义 `Chunk` / `RetrieveChunk` 数据结构
- 使用 sentence-transformers 生成文本 embedding
- 使用 ChromaDB 持久化存储和向量检索
- 使用 DeepSeek OpenAI-compatible API 完成总结、工具决策和最终回答
- 支持 query rewrite，让问题更适合向量检索
- 支持 Agent 工具调用：
  - `search_knowledge_base`
  - `list_chunks`
  - `summarize_text`
- 支持 Agent trace 日志，日志存储在 `logs/agent_logs.jsonl`
- 提供简单 HTML 前端页面
- 支持 Docker Compose 启动

## 技术栈

- Python 3.11+
- FastAPI
- Uvicorn
- Pydantic
- ChromaDB
- sentence-transformers
- DeepSeek API
- OpenAI Python SDK
- HTML / CSS / JavaScript
- Docker / Docker Compose

## 项目结构

```text
ai-learning/
├── app/
│   ├── main.py                     # FastAPI 应用入口，挂载前端和路由
│   ├── api/                        # API 路由层
│   │   ├── routes_basic.py          # 根路径健康检查
│   │   ├── routes_docs.py           # 文档上传、切分、chunk 管理
│   │   ├── routes_chat.py           # 直接 RAG 问答接口
│   │   ├── routes_agent.py          # Agent 入口
│   │   └── routes_logs.py           # Agent 日志读取
│   ├── agent/                      # Agent 执行层
│   │   ├── planner.py               # 让 LLM 判断是否需要工具
│   │   ├── tools.py                 # Agent 可调用工具
│   │   └── executor.py              # Agent 执行主流程
│   ├── llm/
│   │   └── client.py                # DeepSeek / OpenAI-compatible LLM 调用
│   ├── rag/
│   │   ├── splitter.py              # 文本切分逻辑
│   │   ├── vector_store.py          # ChromaDB 向量存储与检索
│   │   ├── retrieve.py              # 带 query rewrite 的检索封装
│   │   └── types.py                 # Chunk / RetrieveChunk 类型定义
│   ├── schemas/                    # FastAPI 请求模型
│   └── observability/
│       └── logger.py                # Agent trace 日志读写
├── static/
│   └── index.html                   # 简单前端页面
├── test/                            # 测试资料
├── data/
│   └── chroma/                      # ChromaDB 本地持久化数据
├── logs/
│   └── agent_logs.jsonl             # Agent trace 日志
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env.example
└── README.md
```

## 快速开始

### 1. 准备环境变量

复制 `.env.example` 为 `.env`，并填入 DeepSeek API Key：

```bash
cp .env.example .env
```

`.env` 示例：

```env
DEEPSEEK_API_KEY=your_api_key_here
```

注意：`.env` 不要提交到 Git。

### 2. 本地启动

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

启动后访问：

```text
http://127.0.0.1:8000/
```

FastAPI 调试文档：

```text
http://127.0.0.1:8000/docs
```

首次运行 sentence-transformers 时，模型可能需要下载。

### 3. Docker 启动

如果已经安装 Docker Desktop：

```bash
docker compose up --build
```

之后如果只是改业务代码，通常可以直接：

```bash
docker compose up
```

停止服务：

```bash
docker compose down
```

## 主要接口

### 前端页面

```text
GET /
GET /static/index.html
```

### 文档与知识库

```text
POST   /docs-api/upload_file     上传 .txt / .md 文件
POST   /docs-api/adddocument     直接添加文本到知识库
POST   /docs-api/split           文本切分
POST   /docs-api/summarize       总结文本
GET    /docs-api/chunks          查看当前知识库 chunks
DELETE /docs-api/chunks          清空知识库 chunks
```

### 问答与 Agent

```text
POST /chat/ask        直接 RAG 问答
POST /agent           Agent 自动决策是否使用工具
GET  /logs/agent      查看 Agent trace 日志
```

## 核心流程

### 文档入库流程

```text
用户上传文件
-> FastAPI 接收 UploadFile
-> 读取文件内容
-> splitter 根据文件类型切分文本
-> 生成 Chunk 对象
-> sentence-transformers 生成 embedding
-> ChromaDB 保存 id / document / embedding / metadata
```

### RAG 问答流程

```text
用户提问
-> query rewrite 改写检索问题
-> sentence-transformers 生成问题 embedding
-> ChromaDB 检索相似 chunks
-> 组装上下文 context
-> LLM 只根据上下文回答
-> 返回 answer / used_chunks / contexts
```

### Agent 流程

```text
用户输入 message
-> planner 调用 LLM 判断是否需要工具
-> 如果不需要工具，直接回答
-> 如果需要工具，选择 search_knowledge_base / list_chunks / summarize_text
-> executor 执行对应工具
-> 如有检索结果，再调用 LLM 生成最终答案
-> 记录 trace 到 logs/agent_logs.jsonl
```

## 我构建这个项目的流程总结

### 1. 从基础 FastAPI 服务开始

最初先搭建 FastAPI 应用，写出最小可运行接口：

```text
GET / -> {"message": "helloapi"}
```

这一步的目标是确认后端服务、虚拟环境、依赖安装和 Uvicorn 启动流程都能跑通。

### 2. 增加文本切分能力

随后抽出 `splitter.py`，实现文本切分：

- 先按段落切分
- 再支持固定长度 `chunk_size`
- 增加 `overlap`，避免上下文在 chunk 边界断裂
- 最后支持 Markdown 标题切分，让 `.md` 文档能按章节结构保留 `section`

这一阶段开始理解 RAG 里的 chunking：大模型不能一次处理无限文本，所以需要把文档拆成可检索的小片段。

### 3. 定义 Chunk 数据结构

为了避免项目里到处使用松散的 dict，我增加了 `Chunk` 和 `RetrieveChunk`：

```text
Chunk:
- id
- content
- source
- chunk_index
- section
- token_count

RetrieveChunk:
- distance
- score
- rerank_score
```

这样 splitter、vector store、LLM 上下文组装之间的数据格式更统一，也更方便类型提示和调试。

### 4. 接入 ChromaDB 向量数据库

文档切成 chunk 后，使用 sentence-transformers 生成 embedding，并通过 ChromaDB 保存：

```text
chunk content -> embedding -> ChromaDB
```

当前向量数据库保存在：

```text
data/chroma
```

ChromaDB 里主要保存：

- `id`: chunk 唯一标识
- `document`: chunk 正文
- `embedding`: 向量
- `metadata`: source、chunk_index、section、token_count 等信息

### 5. 接入 DeepSeek LLM

项目使用 OpenAI Python SDK 调 DeepSeek 的 OpenAI-compatible API：

```python
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
)
```

为了减少重复代码，我抽出了统一接口：

```python
chat_with_llm(...)
```

其他总结、工具决策、最终回答都通过它调用 LLM。

### 6. 实现 RAG 问答

RAG 问答分成两步：

```text
检索：search_vector_store
回答：answer_with_context
```

检索时先把问题转成 embedding，再从 ChromaDB 找相似 chunk。回答时把检索结果整理成 context，让 LLM 只根据 context 回答。

### 7. 增加 query rewrite

为了让向量检索更稳定，我增加了 `rewrite_query`：

```text
用户原始问题 -> 改写成更适合检索的问题
```

例如用户问得比较口语化时，LLM 会补充一些更像知识库关键词的表达，再用于检索。

### 8. 增加 Agent 工具调用

在直接 RAG 问答之外，我增加了 Agent 层：

- `planner`: 判断是否需要工具
- `tools`: 暴露可调用工具
- `executor`: 执行工具并组织最终输出

Agent 可以决定：

```text
是否直接回答
是否检索知识库
是否列出 chunks
是否总结用户给的文本
```

### 9. 增加日志和 trace

为了观察 Agent 的行为，我增加了日志系统：

```text
logs/agent_logs.jsonl
```

每次 Agent 执行都会记录：

- 用户问题
- 工具决策
- 工具名称
- 工具参数
- 工具结果
- 最终回答
- 耗时
- 错误信息

这让项目从“能跑”变成“能观察、能调试”。

### 10. 增加前端页面

一开始所有接口都只能在 Swagger 里调试。后来增加了 `static/index.html`，提供一个简单页面：

- 上传文件
- 输入问题
- 查看答案
- 查看工具结果
- 查看 Agent trace

这样项目开始从纯后端 API 变成一个可交互的小应用。

### 11. Docker 化项目

最后用 Dockerfile 和 docker-compose.yml 把项目容器化：

- Dockerfile 描述如何构建 Python 运行环境
- docker-compose.yml 描述如何启动服务、映射端口、加载 `.env`、挂载日志目录

这一步让项目更接近真实部署流程。

## 当前项目状态

这个项目目前是一个可运行的 RAG Agent 原型，已经具备：

- 后端 API
- 向量数据库
- LLM 调用
- Agent 工具调用
- 前端页面
- 日志系统
- Docker 启动方式

还可以继续改进的方向：

- 增加更完善的错误处理
- 增加文件删除和按 source 删除 chunks
- 增加更好的中文 embedding 模型
- 增加 rerank
- 增加用户会话和历史记录
- 增加正式前端框架
- 增加测试用例
- 增加部署配置
