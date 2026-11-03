"""FastAPI 接口层：把 rag.py 的能力暴露成 HTTP 服务。

启动：
    pip install fastapi uvicorn
    uvicorn api:app --reload --port 8000
"""

from fastapi import FastAPI
from pydantic import BaseModel

from rag import ask, find_hybrid

app = FastAPI(
    title="DeepSeek 文档问答 API",
    description="基于 RAG 的技术文档问答服务",
    version="0.1.0",
)


class AskRequest(BaseModel):
    question: str
    k: int = 3


class AskResponse(BaseModel):
    answer: str
    sources: list[str]


@app.get("/health")
def health():
    """健康检查：服务活着就返回 ok。"""
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask_api(req: AskRequest):
    """提问接口：返回答案和命中的原文片段。"""
    answer = ask(req.question, find_hybrid)
    sources = [c[:200] for c in find_hybrid(req.question, req.k)]
    return AskResponse(answer=answer, sources=sources)
