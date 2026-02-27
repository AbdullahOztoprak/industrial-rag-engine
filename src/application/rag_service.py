"""
RAG Service — Retrieval-Augmented Generation pipeline.

Orchestrates document loading, vector storage, and context-enhanced
LLM responses with source attribution.
"""

from __future__ import annotations

import logging
from typing import Optional

from src.config.settings import Settings, get_settings
from src.domain import DocumentChunk, RetrievalResult, SourceAttribution
from src.infrastructure.document_loader import DocumentLoader
from src.infrastructure.vector_store import VectorStore

logger = logging.getLogger(__name__)


class RAGService:
    """
    RAG pipeline service.

    Responsibilities:
    - Initialize and manage the document ingestion pipeline
    - Perform context retrieval for queries
    - Build augmented prompts with source attribution
    - Track retrieval quality metrics
    """

    def __init__(
        self,
        document_loader: Optional[DocumentLoader] = None,
        vector_store: Optional[VectorStore] = None,
        settings: Optional[Settings] = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._document_loader = document_loader or DocumentLoader(self._settings)
        self._vector_store = vector_store or VectorStore(self._settings)
        self._is_initialized = False
        self._total_queries = 0

    # ── Initialization ───────────────────────────────────────────────────

    def initialize(self, docs_directory: Optional[str] = None) -> bool:
        """
        Load documents and build the vector store.

        Args:
            docs_directory: Override directory for documents.

        Returns:
            True if initialization was successful.
        """
        try:
            logger.info("Initializing RAG pipeline...")
            chunks = self._document_loader.load_directory(docs_directory)

            if not chunks:
                logger.warning("No documents loaded. RAG will return empty context.")
                return False

            self._vector_store.build_from_documents(chunks)
            self._is_initialized = True
            logger.info(f"RAG pipeline initialized: {len(chunks)} chunks indexed")
            return True

        except Exception as e:
            logger.error(f"RAG initialization failed: {e}")
            self._is_initialized = False
            return False

    def add_document(self, file_path: str) -> bool:
        """Add a single document to the existing index."""
        try:
            chunks = self._document_loader.load_single_file(file_path)
            self._vector_store.add_documents(chunks)
            logger.info(f"Document added to RAG index: {file_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to add document: {e}")
            return False

    # ── Retrieval ────────────────────────────────────────────────────────

    def retrieve(self, query: str, top_k: Optional[int] = None) -> RetrievalResult:
        """
        Retrieve relevant document chunks for a query.

        Args:
            query: User query string.
            top_k: Number of results to return.

        Returns:
            RetrievalResult with ranked chunks and scores.
        """
        self._total_queries += 1

        if not self._is_initialized:
            logger.warning("RAG not initialized. Returning empty result.")
            return RetrievalResult(query=query, chunks=[], relevance_scores=[])

        results = self._vector_store.similarity_search(query, k=top_k)

        chunks: list[DocumentChunk] = []
        scores: list[float] = []

        for doc, score in results:
            chunk = DocumentChunk(
                content=doc.page_content,
                source=doc.metadata.get("source", "unknown"),
                page=doc.metadata.get("page"),
                chunk_index=doc.metadata.get("chunk_index", 0),
                metadata=doc.metadata,
            )
            chunks.append(chunk)
            scores.append(round(score, 4))

        logger.debug(f"Retrieved {len(chunks)} chunks for query: '{query[:50]}...'")
        return RetrievalResult(query=query, chunks=chunks, relevance_scores=scores)

    def build_augmented_prompt(self, query: str, retrieval: RetrievalResult) -> str:
        """
        Build an augmented prompt with retrieved context.

        Args:
            query: Original user query.
            retrieval: RetrievalResult from retrieve().

        Returns:
            Augmented prompt string with context.
        """
        if not retrieval.chunks:
            return query

        context_parts: list[str] = []
        for i, chunk in enumerate(retrieval.chunks):
            source_name = chunk.metadata.get("document_name", chunk.source)
            context_parts.append(f"[Source {i + 1}: {source_name}]\n{chunk.content}")

        context = "\n\n---\n\n".join(context_parts)

        return (
            f"Use the following context from industrial documentation to answer the question.\n"
            f"If the context doesn't contain relevant information, use your general knowledge "
            f"but clearly indicate that it's not from the provided documentation.\n\n"
            f"CONTEXT:\n{context}\n\n"
            f"QUESTION: {query}\n\n"
            f"Answer with technical precision. Cite [Source N] when using context."
        )

    def get_source_attributions(self, retrieval: RetrievalResult) -> list[SourceAttribution]:
        """Convert retrieval results to source attributions."""
        attributions: list[SourceAttribution] = []

        for chunk, score in zip(retrieval.chunks, retrieval.relevance_scores):
            attributions.append(
                SourceAttribution(
                    document=chunk.metadata.get("document_name", chunk.source),
                    section=(
                        chunk.content[:100] + "..." if len(chunk.content) > 100 else chunk.content
                    ),
                    relevance_score=score,
                    excerpt=chunk.content[:200],
                )
            )

        return attributions

    # ── Metrics ──────────────────────────────────────────────────────────

    @property
    def is_initialized(self) -> bool:
        return self._is_initialized

    @property
    def total_queries(self) -> int:
        return self._total_queries

    @property
    def document_count(self) -> int:
        return self._vector_store.document_count
