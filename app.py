"""
FastAPI 後端 - 把 Paper RAG with Guardrails 包成 web service

啟動：
    uvicorn app:app --reload --host 0.0.0.0 --port 8000

開瀏覽器看：
    http://localhost:8000          → 對話介面
    http://localhost:8000/docs     → Swagger UI（自動產生的 API 測試介面）

設計：
  - 啟動時建一個 RAGCore（singleton），所有 request 共用，避免每次重建
  - /chat 接 messages 陣列（前端維護 history），實現多輪對話 / in-context learning
  - 支援 mode 切換：baseline / guarded / compare（同時跑兩條 pipeline 並列回傳）
  - 後端 stateless，沒有 session、沒有持久記憶；前端 reload 即重置
"""
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from pipeline import BaselineRAG, GuardedRAG, PipelineResult
from rag_core import RAGCore


# === 全域 singleton：RAG 與 guardrails 都很重，只能載入一次 ===
state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """app 啟動時跑一次：載 RAG + Guardrails；關閉時什麼都不做"""
    print("[app] 初始化 RAGCore（首次會載 Qwen2.5:14b、連 Qdrant）...")
    rag = RAGCore()
    print("[app] 初始化 BaselineRAG...")
    state["baseline"] = BaselineRAG(rag=rag)
    print("[app] 初始化 GuardedRAG（會載入 NeMo + Presidio）...")
    state["guarded"] = GuardedRAG(rag=rag)
    # compare 模式平行跑兩條 pipeline，預先建好 thread pool 重用
    state["pool"] = ThreadPoolExecutor(max_workers=2, thread_name_prefix="rag-compare")
    print("[app] 就緒，開放 API 服務")
    yield
    state["pool"].shutdown(wait=False)
    print("[app] 關閉")


app = FastAPI(
    title="Paper RAG with Guardrails API",
    description="本地 RAG（LlamaIndex + Qdrant + Ollama Qwen2.5）+ NeMo / Presidio 雙閘 Guardrails",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS：允許任何來源（demo 用；production 要鎖網域）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# === Pydantic schemas ===
class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    messages: list[Message] = Field(..., description="完整對話歷史，最後一筆必須是 user")
    mode: Literal["baseline", "guarded", "compare"] = Field(
        "guarded",
        description="baseline = 無防護；guarded = NeMo + Presidio；compare = 兩條並排",
    )


class ChunkInfo(BaseModel):
    """單一 retrieved chunk，給前端的來源面板與 [chunk N] hover 預覽用"""
    chunk_id: str
    doc_id: str = ""
    page: Optional[int] = None
    text: str
    score: float


class SingleResult(BaseModel):
    """單一 pipeline 的執行結果"""
    reply: str
    blocked_at: Optional[Literal["INPUT", "OUTPUT"]] = None
    chunks: list[ChunkInfo] = []
    latency_ms: int


class ChatResponse(BaseModel):
    """統一回傳結構：results 是 dict[mode_name, SingleResult]，
    單模式時長度 1，compare 模式時同時帶 baseline + guarded 兩個 key。"""
    mode: str
    results: dict[str, SingleResult]


def _result_to_single(result: PipelineResult, latency_ms: int) -> SingleResult:
    return SingleResult(
        reply=result.final_response,
        blocked_at=result.blocked_at,
        chunks=[
            ChunkInfo(
                chunk_id=c.chunk_id,
                doc_id=c.doc_id,
                page=c.page,
                text=c.text,
                score=c.score,
            )
            for c in result.retrieved_chunks
        ],
        latency_ms=latency_ms,
    )


def _run_pipeline(name: str, msgs: list) -> tuple[str, SingleResult]:
    """跑單條 pipeline 並計時，回 (mode_name, SingleResult)，給單模式與 compare 共用。"""
    pipeline = state[name]
    t0 = time.time()
    result = pipeline.chat(msgs)
    latency_ms = int((time.time() - t0) * 1000)
    return name, _result_to_single(result, latency_ms)


@app.get("/api/health")
async def health():
    """健康檢查 — 順便回傳 pipeline 是否就緒"""
    return {
        "ok": True,
        "baseline_ready": "baseline" in state,
        "guarded_ready": "guarded" in state,
    }


# 注意：用 sync def（不是 async def）。
# 原因：NeMo Guardrails 與 LlamaIndex Ollama 都是 blocking IO，
# FastAPI 對 sync endpoint 會自動丟 thread pool 執行，
# 同時避免 NeMo 在 async context 裡呼叫 sync generate() 炸掉。
@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if not req.messages or req.messages[-1].role != "user":
        raise HTTPException(
            status_code=400, detail="messages 不能為空，且最後一筆必須是 user"
        )

    msgs = [m.model_dump() for m in req.messages]

    if req.mode == "compare":
        # 平行跑 baseline + guarded，回傳延遲取兩者最大值
        pool: ThreadPoolExecutor = state["pool"]
        futures = [pool.submit(_run_pipeline, n, msgs) for n in ("baseline", "guarded")]
        results = dict(f.result() for f in futures)
        return ChatResponse(mode="compare", results=results)

    name, single = _run_pipeline(req.mode, msgs)
    return ChatResponse(mode=req.mode, results={name: single})


# === 靜態前端（要放在最後 mount，否則會吃掉上面的 /api/* 路由）===
STATIC_DIR = Path(__file__).parent / "static"
if STATIC_DIR.is_dir():
    app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
