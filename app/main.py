import logging
import os
import sqlite3
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, RedirectResponse, Response
from langgraph.checkpoint.sqlite import SqliteSaver

from app.api.routes import router
from app.config import get_settings
from app.db.seed import seed
from app.db.session import init_db
from app.graph.builder import build_graph
from app.integrations.mock_service import router as mock_router
from app.rag.retriever import get_retriever

log = logging.getLogger("support")


@asynccontextmanager
async def lifespan(app: FastAPI):
    s = get_settings()
    logging.basicConfig(
        level=s.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    init_db()
    seed()
    app.state.rag_backend = get_retriever().backend
    # checkpointer persists paused (awaiting-approval) runs across restarts
    os.makedirs(os.path.dirname(s.checkpoint_db) or ".", exist_ok=True)
    conn = sqlite3.connect(s.checkpoint_db, check_same_thread=False)
    app.state.graph = build_graph(SqliteSaver(conn))
    log.info("ready (rag=%s)", app.state.rag_backend)
    yield
    conn.close()


app = FastAPI(title="Customer Support Resolution Agent", lifespan=lifespan)
app.include_router(router)
app.include_router(mock_router)


@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/docs")


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return Response(status_code=204)


@app.exception_handler(Exception)
async def unhandled(request: Request, exc: Exception):
    log.exception("unhandled error on %s", request.url.path)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal error. Our team has been notified.", "type": type(exc).__name__},
    )
