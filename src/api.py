"""HTTP layer for Crux.

Turns the in-RAM RAGSearch engine into three endpoints the browser can call:
  POST /upload  -> build a per-visitor index, return a session_id
  POST /chat    -> stream the answer back token-by-token (SSE)
  POST /clear   -> drop the session from memory

Raw uploaded files live only inside a temp dir that is deleted the moment the
index is built; only the vectors stay, in RAM, keyed by session_id.
"""

import os
import json
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from src.search import RAGSearch

MAX_FILES = 10
ALLOWED_ORIGINS = os.getenv("CRUX_ALLOWED_ORIGINS", "http://localhost:3000").split(",")

# session_id -> in-memory RAG engine. Lives only as long as the process.
SESSIONS: dict[str, RAGSearch] = {}

# Shared engine for chat before/without any upload (general knowledge only).
GENERAL = None


def get_general() -> RAGSearch:
    global GENERAL
    if GENERAL is None:
        GENERAL = RAGSearch.general()
    return GENERAL


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warm the slow parts BEFORE the first request so uploads feel instant:
    # loading the embedding model (torch + weights) is the real cold-start cost.
    from src.vectorstore import _get_model
    _get_model("all-MiniLM-L6-v2")
    get_general()
    yield


app = FastAPI(title="Crux API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/upload")
async def upload(
    files: list[UploadFile] = File(...),
    persona: str | None = Form(None),  # optional per-client house instruction
):
    if not files:
        raise HTTPException(400, "No files provided")
    if len(files) > MAX_FILES:
        raise HTTPException(400, f"Max {MAX_FILES} files per session")

    # basename guards against path traversal in the supplied filename
    uploads = [(os.path.basename(f.filename or "file"), await f.read()) for f in files]
    try:
        engine = RAGSearch.from_uploads(uploads, persona=persona)
    except Exception as exc:  # malformed/empty docs, unreadable files, etc.
        raise HTTPException(422, f"Could not read those documents: {exc}")
    # raw bytes are dropped here — only vectors (cached by hash) remain in RAM

    session_id = uuid.uuid4().hex
    SESSIONS[session_id] = engine
    return {"session_id": session_id, "file_count": len(files)}


class ChatBody(BaseModel):
    session_id: str | None = None
    message: str
    history: list = []
    effort: str | None = None  # "low" | "medium" | "high" — client-chosen answer depth
    name: str | None = None  # optional visitor name, used naturally in replies


@app.post("/chat")
def chat(body: ChatBody):
    # With a live session, answer from that document; otherwise chat from
    # general knowledge so Crux still works before any upload.
    engine = SESSIONS.get(body.session_id) if body.session_id else None
    answer = engine.stream_answer if engine else get_general().stream_answer

    def event_stream():
        grounded, citations = False, []
        for piece in answer(body.message, body.history, effort=body.effort, name=body.name):
            if isinstance(piece, dict):  # final frame from the generator
                grounded = piece.get("grounded", False)
                citations = piece.get("citations", [])
            elif piece:
                yield f"data: {json.dumps({'token': piece})}\n\n"
        yield "data: " + json.dumps({"done": True, "grounded": grounded, "citations": citations}) + "\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/clear")
def clear(body: ChatBody):
    SESSIONS.pop(body.session_id, None)
    return {"ok": True}
