"""
Document Loader — Ingestion pipeline for industrial documentation.

Supports TXT and PDF files with metadata extraction.
Designed for modularity: easy to extend with new file formats.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from src.config.settings import Settings, get_settings

logger = logging.getLogger(__name__)

# File type → Loader mapping
LOADER_MAPPING: dict[str, type] = {
    "*.txt": TextLoader,
    "*.pdf": PyPDFLoader,
}


class DocumentLoader:
    """
    Industrial document ingestion pipeline.
    
    Responsibilities:
    - Load documents from configured directory
    - Split documents into semantic chunks
    - Enrich chunks with metadata
    - Support incremental document addition
    """

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self._settings = settings or get_settings()
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=self._settings.chunk_size,
            chunk_overlap=self._settings.chunk_overlap,
            separators=["\n## ", "\n### ", "\n\n", "\n", ". ", " ", ""],
        )

    def load_directory(self, directory: Optional[str] = None) -> list[Document]:
        """
        Load and split all documents from the configured directory.

        Args:
            directory: Override directory path. Defaults to settings.

        Returns:
            List of Document chunks ready for embedding.
        """
        docs_dir = Path(directory or self._settings.docs_directory)

        if not docs_dir.exists():
            logger.warning(f"Documents directory not found: {docs_dir}")
            return []

        all_documents: list[Document] = []

        for glob_pattern, loader_cls in LOADER_MAPPING.items():
            try:
                loader = DirectoryLoader(
                    str(docs_dir),
                    glob=f"**/{glob_pattern}",
                    loader_cls=loader_cls,
                    show_progress=False,
                )
                docs = loader.load()
                all_documents.extend(docs)
                logger.info(
                    f"Loaded {len(docs)} documents matching {glob_pattern} from {docs_dir}"
                )
            except Exception as e:
                logger.error(f"Error loading {glob_pattern} files: {e}")

        if not all_documents:
            logger.warning("No documents loaded from any source.")
            return []

        # Split into chunks
        chunks = self._splitter.split_documents(all_documents)

        # Enrich metadata
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = i
            chunk.metadata["total_chunks"] = len(chunks)
            source = chunk.metadata.get("source", "unknown")
            chunk.metadata["document_name"] = Path(source).stem

        logger.info(
            f"Document ingestion complete: {len(all_documents)} docs → {len(chunks)} chunks"
        )
        return chunks

    def load_single_file(self, file_path: str) -> list[Document]:
        """
        Load and split a single document file.

        Args:
            file_path: Path to the document.

        Returns:
            List of Document chunks.

        Raises:
            FileNotFoundError: If the file doesn't exist.
            ValueError: If the file type is unsupported.
        """
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"Document not found: {file_path}")

        suffix = path.suffix.lower()
        loader_cls = None
        for pattern, cls in LOADER_MAPPING.items():
            if pattern.endswith(suffix):
                loader_cls = cls
                break

        if loader_cls is None:
            raise ValueError(
                f"Unsupported file type: {suffix}. "
                f"Supported: {list(LOADER_MAPPING.keys())}"
            )

        try:
            loader = loader_cls(str(path))
            docs = loader.load()
            chunks = self._splitter.split_documents(docs)

            for i, chunk in enumerate(chunks):
                chunk.metadata["chunk_index"] = i
                chunk.metadata["document_name"] = path.stem

            logger.info(f"Loaded {path.name}: {len(docs)} pages → {len(chunks)} chunks")
            return chunks

        except Exception as e:
            logger.error(f"Failed to load {file_path}: {e}")
            raise
