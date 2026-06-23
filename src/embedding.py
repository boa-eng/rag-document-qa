from typing import List, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import numpy as np
from src.data_loader import load_all_documents


class EmbeddingPipeline:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", chunk_size: int = 1000, chunk_overlap: int = 200, model: "SentenceTransformer | None" = None):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        # Reuse a preloaded (cached) model when given — loading SentenceTransformer
        # fresh on every upload was the real indexing slowdown.
        self.model = model if model is not None else SentenceTransformer(model_name)

    def chunk_documents(self, documents: List[Any]) -> List[Any]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = splitter.split_documents(documents)
        print(f"[INFO] Split {len(documents)} documents into {len(chunks)} chunks.")
        return chunks

    def embed_chunks(self, chunks: List[Any]) -> np.ndarray:
        # Bigger batch = faster CPU throughput; no progress bar in the server logs.
        embeddings = self.model.encode(
            [chunk.page_content for chunk in chunks],
            batch_size=64,
            show_progress_bar=False,
        )
        print(f"[INFO] Created embeddings for {len(embeddings)} chunks.")
        return embeddings

