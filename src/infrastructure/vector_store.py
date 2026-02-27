"""
Vector Store — ChromaDB-backed vector storage for RAG retrieval.

Provides semantic search over industrial documentation embeddings.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

from src.config.settings import Settings, get_settings

logger = logging.getLogger(__name__)


class VectorStore:
    """
    Abstraction over ChromaDB vector store.

    Responsibilities:
    - Create and manage embeddings
    - Persist and load vector store
    - Perform similarity search with scores
    - Support incremental document addition
    """

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self._settings = settings or get_settings()
        self._embeddings = self._create_embeddings()
        self._store: Optional[Chroma] = None

    def _create_embeddings(self) -> OpenAIEmbeddings:
        """Create configured embedding model."""
        return OpenAIEmbeddings(
            model=self._settings.embedding_model,
            openai_api_key=self._settings.openai_api_key,
        )

    # ── Public API ───────────────────────────────────────────────────────

    def build_from_documents(self, documents: list[Document]) -> None:
        """
        Build the vector store from a list of documents.

        Args:
            documents: Pre-split document chunks.
        """
        if not documents:
            logger.warning("No documents provided for vector store build.")
            return

        persist_dir = self._settings.vector_store_path

        self._store = Chroma.from_documents(
            documents=documents,
            embedding=self._embeddings,
            persist_directory=persist_dir,
        )

        logger.info(
            f"Vector store built with {len(documents)} chunks"
            + (f", persisted to {persist_dir}" if persist_dir else "")
        )

    def add_documents(self, documents: list[Document]) -> None:
        """Add documents to an existing vector store."""
        if self._store is None:
            self.build_from_documents(documents)
            return

        self._store.add_documents(documents)
        logger.info(f"Added {len(documents)} chunks to vector store.")

    def similarity_search(
        self, query: str, k: Optional[int] = None
    ) -> list[tuple[Document, float]]:
        """
        Search for similar documents with relevance scores.

        Args:
            query: Search query text.
            k: Number of results to return.

        Returns:
            List of (Document, score) tuples sorted by relevance.
        """
        if self._store is None:
            logger.warning("Vector store not initialized. Returning empty results.")
            return []

        top_k = k or self._settings.retrieval_top_k

        try:
            results = self._store.similarity_search_with_relevance_scores(
                query, k=top_k
            )
            logger.debug(
                f"Similarity search for '{query[:50]}...' returned {len(results)} results"
            )
            return results

        except Exception as e:
            logger.error(f"Similarity search failed: {e}")
            return []

    @property
    def is_initialized(self) -> bool:
        """Check if the vector store has been built."""
        return self._store is not None

    @property
    def document_count(self) -> int:
        """Return approximate number of stored chunks."""
        if self._store is None:
            return 0
        try:
            return self._store._collection.count()
        except Exception:
            return 0
