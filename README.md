# RAG Agent Demo

    一个基于FastAPI, DeepSeek, ChormaDB 和 sentence-transformers 构建的RAG Agent Demo。

    用户可以上传 ".txt" 和 ".md" 形式的文件，Agent会自动切分文本、生成embedding、并且向量化储存在ChormaDB数据库中。
    当用户对Agent提问时，Agent会自行决定是否需要使用数据库来回答问题。如果需要的话，会在数据库中使用向量检索来搜寻相关chunking，并根据chunking中提到的数据或知识回答问题。

## Features

- 上传 `.txt` / `.md` 文件

- 文档 chunking，支持 chunk size 和 overlap

- 使用 sentence-transformers 生成文本 embedding

- 使用 ChromaDB 存储和检索向量

- 使用 DeepSeek API 生成回答

- 支持多工具 Agent：
  - `search_knowledge_base`
  - `list_chunks`
  - `summarize_text`

- 支持 Agent trace 日志

- 提供简单 HTML 前端页面

- 支持 Docker Compose 部署

## Tech Stack

- Python

- FastAPI

- Uvicorn

- ChormaDB

- Pydantic

- DeepSeek API

- OpenAI Python SDK

- Sentence Transformers

- Docker

- HTML/CSS


## Project Structure

```text
.
├── main.py              # FastAPI 入口
├── splitter.py          # 文本切分逻辑
├── llm_client.py        # DeepSeek API 调用
├── vector_store.py      # ChromaDB 向量数据库逻辑
├── logger.py            # Agent trace 日志
├── static/
│   └── index.html       # 前端页面
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .env.example
└── README.md
```


## 快速开始

### 1. 克隆项目

```bash
git clone https://github.com/Arachniddd/ai-learning.git
```

### 2. 使用 Docker 启动项目

如果你已经安装了 Docker Desktop，可以直接运行：

```bash
docker compose up --build
```

启动成功后，访问前端页面：

http://127.0.0.1:8000/static/index.html

访问 FastAPI 自动生成的接口文档：

http://127.0.0.1:8000/docs

如果之后只是重新启动，不需要重新构建镜像，可以使用：


```bash
docker compose up
```

如果想停止服务：

```bash
docker compose down
```