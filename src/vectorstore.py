import os
import re
import faiss
import numpy as np
import pickle
from functools import lru_cache
from typing import List, Any
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from src.embedding import EmbeddingPipeline


def _tokenize(text: str) -> List[str]:
    """Lowercase word tokens for BM25 (keyword) matching."""
    return re.findall(r"\w+", (text or "").lower())


@lru_cache(maxsize=4)
def _get_model(name: str) -> SentenceTransformer:
    """Load each embedding model once and reuse it across sessions."""
    return SentenceTransformer(name)


class FaissVectorStore:
    def __init__(self, persist_dir: str = "faiss_store", embedding_model: str = "all-MiniLM-L6-v2", chunk_size: int = 1000, chunk_overlap: int = 200):
        self.persist_dir = persist_dir
        os.makedirs(self.persist_dir, exist_ok=True)
        self.index = None
        self.metadata = []
        # BM25 keyword index, built lazily and rebuilt when chunk count changes
        self._bm25 = None
        self._bm25_n = -1
        self.embedding_model = embedding_model
        self.model = _get_model(embedding_model)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        print(f"[INFO] Loaded embedding model: {embedding_model}")

    def build_from_documents(self, documents: List[Any], persist: bool = True):
        print(f"[INFO] Building vector store from {len(documents)} raw documents...")
        emb_pipe = EmbeddingPipeline(model_name=self.embedding_model, chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap, model=self.model)
        chunks = emb_pipe.chunk_documents(documents)
        embeddings = emb_pipe.embed_chunks(chunks)
        metadatas = [
            {
                "text": chunk.page_content,
                "source": chunk.metadata.get("source", ""),
                "page": chunk.metadata.get("page", ""),
            }
            for chunk in chunks
        ]
        self.add_embeddings(np.array(embeddings).astype('float32'), metadatas)
        if persist:
            self.save()
            print(f"[INFO] Vector store built and saved to {self.persist_dir}")
        else:
            print("[INFO] Vector store built in memory (not persisted)")

    def add_embeddings(self, embeddings: np.ndarray, metadatas: List[Any] = None):
        metadatas = metadatas or []
        # Index and metadata are looked up by the same row position, so a count
        # mismatch silently returns wrong/None citations. Fail loudly instead.
        if len(metadatas) != embeddings.shape[0]:
            raise ValueError(
                f"embeddings ({embeddings.shape[0]}) and metadatas ({len(metadatas)}) must match"
            )
        dim = embeddings.shape[1]
        if self.index is None:
            self.index = faiss.IndexFlatL2(dim)
        self.index.add(embeddings)
        self.metadata.extend(metadatas)
        print(f"[INFO] Added {embeddings.shape[0]} vectors to Faiss index.")

    def save(self):
        faiss_path = os.path.join(self.persist_dir, "faiss.index")
        meta_path = os.path.join(self.persist_dir, "metadata.pkl")
        faiss.write_index(self.index, faiss_path)
        with open(meta_path, "wb") as f:
            pickle.dump(self.metadata, f)
        print(f"[INFO] Saved Faiss index and metadata to {self.persist_dir}")

    def load(self):
        faiss_path = os.path.join(self.persist_dir, "faiss.index")
        meta_path = os.path.join(self.persist_dir, "metadata.pkl")
        self.index = faiss.read_index(faiss_path)
        with open(meta_path, "rb") as f:
            self.metadata = pickle.load(f)
        print(f"[INFO] Loaded Faiss index and metadata from {self.persist_dir}")

    def search(self, query_embedding: np.ndarray, top_k: int = 8):
        D, I = self.index.search(query_embedding, top_k)
        results = []
        for idx, dist in zip(I[0], D[0]):
            meta = self.metadata[idx] if idx < len(self.metadata) else None
            results.append({"index": idx, "distance": dist, "metadata": meta})
        return results

    def query(self, query_text: str, top_k: int = 8):
        print(f"[INFO] Querying vector store for: '{query_text}'")
        query_emb = self.model.encode([query_text]).astype('float32')
        return self.search(query_emb, top_k=top_k)

    def _ensure_bm25(self):
        """(Re)build the keyword index when the set of chunks has changed."""
        n = len(self.metadata)
        if n == 0:
            self._bm25, self._bm25_n = None, 0
            return
        if self._bm25 is None or self._bm25_n != n:
            corpus = [_tokenize(m.get("text", "")) for m in self.metadata]
            self._bm25 = BM25Okapi(corpus)
            self._bm25_n = n

    def hybrid_query(self, query_text: str, top_k: int = 8, dense_k: int = None):
        """Blend semantic (FAISS) and keyword (BM25) retrieval.

        Dense vectors catch paraphrases; BM25 catches exact terms the embedding
        glosses over (names, codes, jargon). The two rankings are merged with
        Reciprocal Rank Fusion (RRF) — a tuning-free way to combine rankers:
        each list contributes 1/(K+rank) to a chunk's score, so being near the
        top of *either* ranker lifts a chunk. This is the fix for "answered from
        one document but ignored the other": a keyword-strong chunk in the second
        doc now surfaces even if its vector wasn't the closest.

        Each returned chunk keeps its true (squared L2) distance so the grounding
        and citation thresholds in search.py stay comparable.
        """
        if self.index is None:
            return []
        dense_k = dense_k or max(top_k, 10)
        query_emb = self.model.encode([query_text]).astype('float32')

        dense = self.search(query_emb, top_k=min(dense_k, self.index.ntotal))
        dense_dist = {int(r["index"]): float(r["distance"]) for r in dense}
        ranked_lists = [[int(r["index"]) for r in dense]]

        self._ensure_bm25()
        if self._bm25 is not None:
            scores = self._bm25.get_scores(_tokenize(query_text))
            order = np.argsort(scores)[::-1][:dense_k]
            # a 0 score means no shared terms — don't let it pollute the ranking
            ranked_lists.append([int(i) for i in order if scores[i] > 0])

        K = 60  # standard RRF damping constant
        fused = {}
        for lst in ranked_lists:
            for rank, idx in enumerate(lst):
                fused[idx] = fused.get(idx, 0.0) + 1.0 / (K + rank + 1)

        results = []
        for idx in sorted(fused, key=lambda i: fused[i], reverse=True)[:top_k]:
            dist = dense_dist.get(idx)
            if dist is None:
                # BM25-only hit: recover its vector to compute a comparable L2
                vec = self.index.reconstruct(idx).astype('float32')
                dist = float(np.sum((query_emb[0] - vec) ** 2))
            meta = self.metadata[idx] if idx < len(self.metadata) else None
            results.append({"index": idx, "distance": dist, "metadata": meta})
        return results
