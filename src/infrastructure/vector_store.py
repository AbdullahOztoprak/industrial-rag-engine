"""
Vector Store — ChromaDB-backed vector storage for RAG retrieval.

Provides semantic search over industrial documentation embeddings.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Optional, cast

from pydantic import SecretStr

from src.config.settings import Settings, get_settings

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from langchain_core.documents import Document
    from langchain_openai import OpenAIEmbeddings
else:
    Document = Any
    OpenAIEmbeddings = Any


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
        self._store: Optional[Any] = None

    def _create_embeddings(self) -> Any:
        """Create configured embedding model or return None when unavailable."""
        api_key = (
            SecretStr(self._settings.openai_api_key) if self._settings.openai_api_key else None
        )
        try:
            from langchain_openai import OpenAIEmbeddings as _OpenAIEmbeddings

            return _OpenAIEmbeddings(model=self._settings.embedding_model, api_key=api_key)
        except Exception:
            logger.debug(
                "OpenAIEmbeddings unavailable in this environment; continuing without embeddings."
            )
            return None

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

        try:
            from langchain_community.vectorstores import Chroma as _Chroma

            if self._embeddings is None:
                logger.warning("Embeddings not available; cannot build vector store.")
                return

            self._store = _Chroma.from_documents(
                documents=documents, embedding=self._embeddings, persist_directory=persist_dir
            )
        except Exception as e:
            logger.error(f"Failed to initialize Chroma vector store: {e}")
            self._store = None

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
            results = self._store.similarity_search_with_relevance_scores(query, k=top_k)
            logger.debug(f"Similarity search for '{query[:50]}...' returned {len(results)} results")
            return cast(list[tuple[Document, float]], results)

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
            count_fn = cast(Any, self._store._collection.count)
            return int(count_fn())
        except Exception:
            return 0
