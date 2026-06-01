from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from app.api.routes_basic import router as basic_router
from app.api.routes_docs import router as docs_router
from app.api.routes_chat import router as chat_router
from app.api.routes_agent import router as agent_router
from app.api.routes_logs import router as logs_router


app = FastAPI(title="RAG Agent Demo")

BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "static"

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC_DIR / "index.html")

app.include_router(basic_router)
app.include_router(docs_router, prefix="/docs-api", tags=["documents"])
app.include_router(chat_router, prefix="/chat", tags=["chat"])
app.include_router(agent_router, prefix="/agent", tags=["agent"])
app.include_router(logs_router, prefix="/logs", tags=["logs"])
