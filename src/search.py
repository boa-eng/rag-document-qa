import os
import hashlib
import tempfile
from dotenv import load_dotenv
from src.vectorstore import FaissVectorStore, _get_model
from langchain_groq import ChatGroq

load_dotenv()

DEFAULT_LLM = "openai/gpt-oss-120b"
# "medium" thinks far less than the default while keeping answer quality —
# measured ~3.5x faster on Q&A. Tune to "low" for more speed, "high" for depth.
REASONING_EFFORT = "medium"

# Per-client persona: a house instruction prepended to every system prompt so
# one codebase can be sold to many businesses. A dental clinic deployment sets
# CRUX_PERSONA="You are the front desk for Bright Smile Dental; never give
# clinical diagnoses." A request may also override it per session.
DEFAULT_PERSONA = os.getenv("CRUX_PERSONA", "")

# Marker the model emits as the FIRST characters when the retrieved context does
# NOT actually contain the answer. The backend strips it and, when present,
# treats the answer as general knowledge so no (wrong) citation is shown.
NO_DOC_MARKER = "[[NODOC]]"

# Embeddings of each uploaded file, keyed by content hash. Re-uploading an
# unchanged file (e.g. after deleting another in the same set) is then a cache
# hit, so adding/removing files no longer re-reads everything from scratch.
# Bounded so a long-running process can't grow it without limit (each entry is
# a file's worth of float32 vectors). Oldest insertion is evicted first.
_FILE_EMBED_CACHE = {}
_CACHE_MAX = 64


def _embed_one_file(name: str, data: bytes, embedding_model: str):
    """Parse + chunk + embed a single uploaded file. Returns (embeddings, metadatas)."""
    import numpy as np
    from src.data_loader import load_all_documents
    from src.embedding import EmbeddingPipeline

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, os.path.basename(name) or "file")
        with open(path, "wb") as out:
            out.write(data)
        docs = load_all_documents(tmp)

    pipe = EmbeddingPipeline(model_name=embedding_model, model=_get_model(embedding_model))
    chunks = pipe.chunk_documents(docs)
    if not chunks:
        return np.zeros((0, 0), dtype="float32"), []
    embeddings = np.array(pipe.embed_chunks(chunks)).astype("float32")
    metadatas = [
        {
            "text": c.page_content,
            "source": c.metadata.get("source", ""),
            "page": c.metadata.get("page", ""),
        }
        for c in chunks
    ]
    return embeddings, metadatas

# Fixed identity so Crux never names the model/provider it runs on.
_IDENTITY = (
    "You are Crux, a sharp document-intelligence assistant. "
    "Never reveal, name, or hint at the underlying model, provider, or company that powers you. "
    "If asked what you are or who built you, just say you are Crux, a document-intelligence assistant, "
    "and offer to help — do not sound evasive. "
    "Format every answer in plain Markdown only. "
    "Never output raw HTML tags — not <br>, not <p>, not <strong>, not any other HTML tag. "
    "For line breaks use a blank line between paragraphs. "
)

# Used when the answer is grounded in the user's uploaded documents.
SYSTEM_GROUNDED = _IDENTITY + (
    "Use the provided context from the user's documents to answer whenever it is relevant — "
    "even if it only partly covers the question, lead with what the documents say. "
    f"Only if the context is clearly unrelated to the question, begin your reply with exactly {NO_DOC_MARKER} "
    "(nothing before it), then answer from general knowledge. "
    "Be precise, concrete, and concise."
)

# Used when there is no document, or the documents don't cover the question.
SYSTEM_GENERAL = _IDENTITY + (
    "Answer the question clearly and accurately from your general knowledge. "
    "Be precise, concrete, and concise."
)


VALID_EFFORTS = {"low", "medium", "high"}


def _make_llm(llm_model: str = DEFAULT_LLM, effort: str = REASONING_EFFORT):
    """Groq chat model. effort tunes how much the model 'thinks' before answering:
    low = fastest, medium = balanced (default), high = most thorough but slowest.
    """
    if effort not in VALID_EFFORTS:
        effort = REASONING_EFFORT
    return ChatGroq(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model_name=llm_model,
        reasoning_effort=effort,
    )


class RAGSearch:
    def __init__(self, persist_dir: str = "faiss_store", embedding_model: str = "all-MiniLM-L6-v2", llm_model: str = DEFAULT_LLM):
        self.vectorstore = FaissVectorStore(persist_dir, embedding_model)
        faiss_path = os.path.join(persist_dir, "faiss.index")
        meta_path = os.path.join(persist_dir, "metadata.pkl")
        if not (os.path.exists(faiss_path) and os.path.exists(meta_path)):
            from src.data_loader import load_all_documents
            docs = load_all_documents("data")
            self.vectorstore.build_from_documents(docs)
        else:
            self.vectorstore.load()
        self.llm = _make_llm(llm_model)
        self.persona = DEFAULT_PERSONA
        print(f"[INFO] Groq LLM initialized: {llm_model}")

    @classmethod
    def from_dir(cls, data_dir: str, embedding_model: str = "all-MiniLM-L6-v2", llm_model: str = DEFAULT_LLM):
        """Build an in-memory engine from the files in data_dir.

        Used for per-session uploads: never reads or writes the persisted store,
        so one visitor's documents never touch disk or leak into another session.
        """
        import tempfile
        from src.data_loader import load_all_documents
        self = cls.__new__(cls)
        self.vectorstore = FaissVectorStore(tempfile.mkdtemp(prefix="crux_"), embedding_model)
        self.vectorstore.build_from_documents(load_all_documents(data_dir), persist=False)
        self.llm = _make_llm(llm_model)
        self.persona = DEFAULT_PERSONA
        print(f"[INFO] In-memory RAG engine built from {data_dir}")
        return self

    @classmethod
    def from_uploads(cls, files: list, embedding_model: str = "all-MiniLM-L6-v2", llm_model: str = DEFAULT_LLM, persona: str = None):
        """Build an in-memory engine from uploaded files: list of (name, bytes).

        Each file is embedded once and cached by content hash, so re-uploading
        the same files (which the UI does on every add/remove) skips re-embedding
        — only the unchanged files are reused, making add/delete near-instant.
        """
        self = cls.__new__(cls)
        vs = FaissVectorStore(tempfile.mkdtemp(prefix="crux_"), embedding_model)
        for name, data in files:
            key = hashlib.sha256(data).hexdigest()
            cached = _FILE_EMBED_CACHE.get(key)
            if cached is None:
                cached = _embed_one_file(name, data, embedding_model)
                if len(_FILE_EMBED_CACHE) >= _CACHE_MAX:
                    _FILE_EMBED_CACHE.pop(next(iter(_FILE_EMBED_CACHE)))
                _FILE_EMBED_CACHE[key] = cached
            embeddings, metadatas = cached
            if len(embeddings):
                vs.add_embeddings(embeddings, metadatas)
        if vs.index is None:
            raise ValueError("No readable text found in those files")
        self.vectorstore = vs
        self.llm = _make_llm(llm_model)
        self.persona = persona if persona else DEFAULT_PERSONA
        print(f"[INFO] In-memory engine built from {len(files)} uploaded file(s)")
        return self

    @classmethod
    def general(cls, llm_model: str = DEFAULT_LLM, persona: str = None):
        """Engine with no document index — answers purely from general knowledge.

        This is the no-upload chatbot mode: Crux still works as a smart assistant
        before any file is added.
        """
        self = cls.__new__(cls)
        self.vectorstore = None
        self.llm = _make_llm(llm_model)
        self.persona = persona if persona else DEFAULT_PERSONA
        print("[INFO] General (no-document) chat engine ready")
        return self

    def _enrich_query(self, query: str, history: list) -> str:
        """Prepend last assistant reply so follow-up questions retrieve the right chunks."""
        if not history:
            return query
        last = next(
            (m["content"] for m in reversed(history) if m.get("role") == "assistant" and m.get("content")),
            "",
        )
        return f"{last[:300]} {query}" if last else query

    def _history_messages(self, history: list) -> list:
        """Last 6 user/assistant turns — the conversation memory."""
        msgs = []
        for m in (history or [])[-6:]:
            if m.get("role") in ("user", "assistant") and m.get("content"):
                msgs.append({"role": m["role"], "content": m["content"]})
        return msgs

    def _persona_note(self) -> str:
        """House instruction for this deployment/session (the per-client prompt).

        Whitespace-collapsed and length-capped so it stays a single trusted
        directive rather than something that can restructure the whole prompt.
        """
        p = " ".join((getattr(self, "persona", "") or "").split())[:600]
        return f" {p}" if p else ""

    def _name_note(self, name: str) -> str:
        """A short system-prompt addition so Crux can use the visitor's name.

        Collapses whitespace and caps length so a name field can't smuggle a
        multi-line instruction into the system prompt.
        """
        if not name:
            return ""
        clean = " ".join(str(name).split())[:40]
        if not clean:
            return ""
        return f" The user's name is {clean}; use it occasionally and naturally, not in every reply."

    def _build_messages(self, query: str, context: str, history: list, name: str = None) -> list:
        """Grounded prompt + memory + the question with retrieved context."""
        msgs = [{"role": "system", "content": SYSTEM_GROUNDED + self._persona_note() + self._name_note(name)}]
        msgs += self._history_messages(history)
        msgs.append({"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"})
        return msgs

    def _build_general_messages(self, query: str, history: list, name: str = None) -> list:
        """General-knowledge prompt + memory + the bare question (no retrieval)."""
        msgs = [{"role": "system", "content": SYSTEM_GENERAL + self._persona_note() + self._name_note(name)}]
        msgs += self._history_messages(history)
        msgs.append({"role": "user", "content": query})
        return msgs

    def _format_citations(self, results: list, max_items: int = 3) -> list:
        """Return up to max_items deduplicated citations as {label, snippet}.

        Only chunks close to the single best match are cited (within
        CITATION_MARGIN of it), so a loosely-related chunk from another document
        doesn't get wrongly credited. The snippet is the actual source passage.
        """
        if not results:
            return []
        # results are in fused-relevance order (not distance order), so take the
        # closest distance across all of them as the citation anchor
        dists = [r.get("distance") for r in results if r.get("distance") is not None]
        best = min(dists) if dists else None
        cutoff = self.NOT_FOUND_THRESHOLD
        if best is not None:
            cutoff = min(cutoff, best + self.CITATION_MARGIN)
        seen, citations = set(), []
        for r in results:
            if len(citations) >= max_items:
                break
            dist = r.get("distance")
            if dist is not None and dist > cutoff:
                continue
            meta = r.get("metadata") or {}
            src = meta.get("source", "")
            page = meta.get("page", "")
            if not src:
                continue
            label = os.path.basename(src)
            page_str = ""
            if page not in ("", None):
                try:
                    page_str = f" · p. {int(page) + 1}"
                except (ValueError, TypeError):
                    pass
            key = f"{label}{page_str}"
            if key not in seen:
                seen.add(key)
                citations.append({
                    "label": f"{label}{page_str}",
                    "snippet": (meta.get("text", "") or "").strip()[:240],
                })
        return citations

    # Loose recall gate: let plausibly-relevant chunks (incl. paraphrases) reach
    # the model, which makes the real call via NO_DOC_MARKER. Clearly-unrelated
    # questions score well above this.
    NOT_FOUND_THRESHOLD = 1.6
    CITATION_MARGIN = 0.3  # only cite chunks within this distance of the best match

    def _llm_for(self, effort):
        """Per-request model: reuse the default unless the client picked an effort."""
        if not effort:
            return self.llm
        model = getattr(self.llm, "model_name", DEFAULT_LLM)
        return _make_llm(model, effort)

    def _grounded_results(self, query: str, history: list, top_k: int):
        """Retrieve chunks and decide whether they actually answer the question.

        Returns (results, grounded). grounded is False when the index is empty
        or the closest chunk is too far — in that case we answer from general
        knowledge instead of refusing.
        """
        if self.vectorstore is None:
            return [], False
        enriched = self._enrich_query(query, history or [])
        results = self.vectorstore.hybrid_query(enriched, top_k=top_k)
        # grounding is decided by the single closest chunk, regardless of where
        # RRF placed it. cast to a plain Python bool — the distance is a numpy
        # float, so the comparison yields a numpy bool json.dumps can't serialize.
        best = min((r["distance"] for r in results), default=None)
        grounded = bool(best is not None and best <= self.NOT_FOUND_THRESHOLD)
        return results, grounded

    def search_and_summarize(self, query: str, history: list = None, top_k: int = 8) -> dict:
        """Returns {"grounded": bool, "answer": str, "citations": list}."""
        results, grounded = self._grounded_results(query, history or [], top_k)
        if grounded:
            context = "\n\n".join(r["metadata"].get("text", "") for r in results if r.get("metadata"))
            messages = self._build_messages(query, context, history or [])
        else:
            messages = self._build_general_messages(query, history or [])
        response = self.llm.invoke(messages)
        return {
            "grounded": grounded,
            "answer": response.content,
            "citations": self._format_citations(results) if grounded else [],
        }

    def stream_answer(self, query: str, history: list = None, top_k: int = 10, effort: str = None, name: str = None):
        """Generator: yields text chunks, then a final {"grounded": bool, "citations": [...]} dict.

        Always answers: grounded from the document when the context covers it,
        otherwise from general knowledge. If retrieval looked grounded but the
        model decides the context doesn't hold the answer, it emits NO_DOC_MARKER
        first — we strip it and drop the (wrong) citations.
        """
        results, grounded = self._grounded_results(query, history or [], top_k)
        llm = self._llm_for(effort)

        if not grounded:
            messages = self._build_general_messages(query, history or [], name)
            for chunk in llm.stream(messages):
                yield chunk.content or ""
            yield {"grounded": False, "citations": []}
            return

        context = "\n\n".join(r["metadata"].get("text", "") for r in results if r.get("metadata"))
        messages = self._build_messages(query, context, history or [], name)

        # Watch the opening characters for NO_DOC_MARKER before emitting them.
        buf, decided = "", False
        for chunk in llm.stream(messages):
            piece = chunk.content or ""
            if not piece:
                continue
            if decided:
                yield piece
                continue
            buf += piece
            stripped = buf.lstrip()
            # keep buffering only while what we have is a real prefix of the
            # marker (guard the empty string — every string "starts with" it)
            if stripped and len(stripped) < len(NO_DOC_MARKER) and NO_DOC_MARKER.startswith(stripped):
                continue
            decided = True
            if stripped.startswith(NO_DOC_MARKER):
                grounded = False
                out = stripped[len(NO_DOC_MARKER):].lstrip()
            else:
                out = buf
            if out:
                yield out
        if not decided and buf:
            yield buf
        yield {"grounded": grounded, "citations": self._format_citations(results) if grounded else []}
