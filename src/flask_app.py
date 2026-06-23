import os
import uuid
import json
import tempfile
import shutil

from flask import Flask, request, jsonify, Response, stream_with_context
from flask_cors import CORS
from dotenv import load_dotenv
from langchain_groq import ChatGroq

from src.data_loader import load_all_documents
from src.vectorstore import FaissVectorStore

load_dotenv()

app = Flask(__name__)
CORS(app)  # allows the Netlify frontend to call this API cross-domain

# One LLM instance shared across all requests
LLM = ChatGroq(groq_api_key=os.getenv("GROQ_API_KEY"), model_name="openai/gpt-oss-120b")

# Mode 1 — no documents uploaded, pure general knowledge
SYSTEM_GENERAL = (
    "You are Crux, a sharp and helpful AI assistant. "
    "Answer clearly and concisely from your knowledge. "
    "Do not mention documents or files unless the user brings them up."
)

# Mode 2 — documents uploaded, RAG answers with fallback to general knowledge
SYSTEM_DOCUMENT = (
    "You are Crux, a helpful assistant for the user's documents. "
    "Use the provided context when it answers the question. "
    "If the context does not contain the answer, say: "
    '\"Your documents don\'t cover this one, but generally:\" and answer from general knowledge. '
    "Keep answers clear and concise."
)

NOT_FOUND_THRESHOLD = 1.2
MAX_FILES = 10
MAX_PAGES = 200  # per document — protects the server from giant uploads hanging it


def count_pages(docs):
    """Pages per source. PDFs load one Document per page; txt/docx load one each."""
    counts = {}
    for d in docs:
        src = d.metadata.get("source", "")
        counts[src] = counts.get(src, 0) + 1
    return counts

# In-memory store: session_id -> {"store": FaissVectorStore, "store_dir": str, "mode": str}
# mode is "persistent" (default) or "session_only" (client opted in)
# Backend never auto-clears — only /clear or server restart wipes a session
sessions = {}


# ── /upload ────────────────────────────────────────────────────────────────────
# Receives uploaded files, chunks + embeds them, builds a FAISS index in a
# temp directory, and returns a session_id the frontend uses for all future calls.
@app.route("/upload", methods=["POST"])
def upload():
    files = request.files.getlist("files")
    if not files:
        return jsonify({"error": "No files provided"}), 400

    files = files[:MAX_FILES]  # enforce 5-document limit

    # Save uploaded files to a temp dir so load_all_documents can read them
    upload_tmp = tempfile.mkdtemp()
    store_dir = tempfile.mkdtemp()  # FAISS index lives here for this session
    try:
        for f in files:
            f.save(os.path.join(upload_tmp, f.filename))
        docs = load_all_documents(upload_tmp)

        # Reject any document over the page limit before we spend time embedding it
        pages = count_pages(docs)
        over = [os.path.basename(s) or "document" for s, n in pages.items() if n > MAX_PAGES]
        if over:
            shutil.rmtree(store_dir, ignore_errors=True)
            return jsonify({"error": f"These exceed {MAX_PAGES} pages: {', '.join(over)}"}), 400
        page_count = sum(pages.values())

        store = FaissVectorStore(store_dir)
        store.build_from_documents(docs)
    except Exception as e:
        shutil.rmtree(upload_tmp, ignore_errors=True)
        shutil.rmtree(store_dir, ignore_errors=True)
        return jsonify({"error": str(e)}), 500
    finally:
        shutil.rmtree(upload_tmp, ignore_errors=True)  # uploaded copies deleted immediately

    session_mode = request.form.get("session_mode", "persistent")
    session_id = str(uuid.uuid4())
    sessions[session_id] = {"store": store, "store_dir": store_dir, "mode": session_mode, "doc_count": len(files)}
    return jsonify({
        "session_id": session_id,
        "mode": session_mode,
        "doc_count": len(files),
        "file_count": len(files),
        "page_count": page_count,
    })


# ── /add ───────────────────────────────────────────────────────────────────────
# Adds more documents to an existing session's FAISS index.
# User uploads 2 docs first (/upload), then adds 3 more (/add) — all searchable together.
@app.route("/add", methods=["POST"])
def add_documents():
    session_id = request.form.get("session_id", "")
    if session_id not in sessions:
        return jsonify({"error": "Session not found. Upload documents first."}), 404

    files = request.files.getlist("files")
    if not files:
        return jsonify({"error": "No files provided"}), 400

    session = sessions[session_id]
    current_count = session["doc_count"]
    slots_left = MAX_FILES - current_count
    if slots_left <= 0:
        return jsonify({"error": f"Session already has {MAX_FILES} documents (limit reached)."}), 400

    files = files[:slots_left]  # only take what fits under the limit

    upload_tmp = tempfile.mkdtemp()
    try:
        for f in files:
            f.save(os.path.join(upload_tmp, f.filename))
        docs = load_all_documents(upload_tmp)
        session["store"].build_from_documents(docs)  # appends to existing FAISS index
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        shutil.rmtree(upload_tmp, ignore_errors=True)

    session["doc_count"] += len(files)
    return jsonify({"session_id": session_id, "doc_count": session["doc_count"]})


# ── /chat ──────────────────────────────────────────────────────────────────────
# Mode 1 (no session_id): answers from general knowledge, signals nudge to frontend
# Mode 2 (session_id present): searches FAISS first, falls back to general knowledge
# SSE format: each line is  data: {"text": "..."}
# Final line:  data: {"done": true, "mode": "general"/"document", "citations": [...]}
@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    session_id = data.get("session_id", "")
    message = (data.get("message") or "").strip()
    history = data.get("history") or []

    if not message:
        return jsonify({"error": "Empty message"}), 400

    has_docs = session_id in sessions
    # force_general=True means the user clicked "Yes, answer from general knowledge"
    # after we told them their documents don't cover the question.
    force_general = bool(data.get("force_general", False))

    def build_msgs(system, context=None):
        msgs = [{"role": "system", "content": system}]
        for m in history[-6:]:
            if m.get("role") in ("user", "assistant") and m.get("content"):
                msgs.append({"role": m["role"], "content": m["content"]})
        content = f"Context:\n{context}\n\nQuestion: {message}" if context else message
        msgs.append({"role": "user", "content": content})
        return msgs

    def stream(msgs, mode, citations=None):
        for chunk in LLM.stream(msgs):
            text = chunk.content or ""
            if text:
                yield f"data: {json.dumps({'text': text})}\n\n"
        yield f"data: {json.dumps({'done': True, 'mode': mode, 'citations': citations or []})}\n\n"

    # ── Mode 1 (no docs) OR user chose "answer from general knowledge" ──────────
    if not has_docs or force_general:
        def generate_general():
            msgs = build_msgs(SYSTEM_GENERAL)
            yield from stream(msgs, mode="general")

        return Response(stream_with_context(generate_general()), mimetype="text/event-stream")

    # ── Mode 2: documents uploaded ─────────────────────────────────────────────
    store = sessions[session_id]["store"]

    # Rewrite for typos before hitting FAISS
    cleaned = LLM.invoke([{
        "role": "user",
        "content": f"Rewrite this search query clearly, fix any typos. Return ONLY the rewritten query.\n\nQuery: {message}"
    }]).content.strip()

    # Enrich with last assistant reply so follow-up questions retrieve right chunks
    last_reply = next((m["content"] for m in reversed(history) if m.get("role") == "assistant"), "")
    search_query = f"{last_reply[:300]} {cleaned}" if last_reply else cleaned

    results = store.query(search_query, top_k=8)

    # Format citations: filename + page number
    seen, citations = set(), []
    for r in results:
        meta = r.get("metadata") or {}
        src = os.path.basename(meta.get("source", ""))
        pg = meta.get("page", "")
        if not src:
            continue
        try:
            page_str = f" · p. {int(pg) + 1}" if pg not in ("", None) else ""
        except (ValueError, TypeError):
            page_str = ""
        key = f"{src}{page_str}"
        if key not in seen:
            seen.add(key)
            citations.append(f"📄 {src}{page_str}")

    def generate_document():
        # Nothing relevant in the docs → DON'T answer. Ask the user first.
        # Frontend shows: "Your documents don't cover this one.
        #                  Answer from general knowledge? [Yes][No]"
        # "Yes" re-calls /chat with force_general=true (handled above).
        if not results or results[0]["distance"] > NOT_FOUND_THRESHOLD:
            yield f"data: {json.dumps({'done': True, 'mode': 'not_found', 'citations': []})}\n\n"
            return

        context = "\n\n".join(r["metadata"].get("text", "") for r in results if r.get("metadata"))
        msgs = build_msgs(SYSTEM_DOCUMENT, context=context)
        yield from stream(msgs, mode="document", citations=citations)

    return Response(stream_with_context(generate_document()), mimetype="text/event-stream")


# ── /clear ─────────────────────────────────────────────────────────────────────
# Wipes the session index and temp files. Called when user clicks "New session".
@app.route("/clear", methods=["POST"])
def clear():
    data = request.get_json()
    session_id = data.get("session_id", "")
    if session_id in sessions:
        shutil.rmtree(sessions[session_id]["store_dir"], ignore_errors=True)
        del sessions[session_id]
    return jsonify({"ok": True})


# ── /health ────────────────────────────────────────────────────────────────────
# Frontend pings this to confirm the server is up before showing the chat.
# Render also uses it for deploy health checks.
@app.route("/")
@app.route("/health")
def health():
    return jsonify({"status": "ok", "service": "crux"})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
