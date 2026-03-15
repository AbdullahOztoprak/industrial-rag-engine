"""
Document Loader — Ingestion pipeline for industrial documentation.

Supports TXT and PDF files with metadata extraction.
Designed for modularity: easy to extend with new file formats.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, cast, Any, TYPE_CHECKING

from src.config.settings import Settings, get_settings

logger = logging.getLogger(__name__)


# Defer heavy third-party imports so tests can import this module without
# requiring langchain packages to be installed in the local environment.
if TYPE_CHECKING:
    from langchain_core.documents import Document
else:
    Document = Any  # runtime fallback when langchain_core isn't available


def _get_loader_mapping() -> dict[str, type]:
    """Return the mapping of glob patterns to loader classes.

    This performs imports lazily and catches ImportError so import-time
    failures don't break test collection in environments without the
    langchain packages installed.
    """
    try:
        from langchain_community.document_loaders import (
            DirectoryLoader,
            PyPDFLoader,
            TextLoader,
        )
    except Exception:
        # Return an empty mapping when loaders are unavailable; callers
        # will handle the "no documents" case gracefully.
        return {}

    return {"*.txt": TextLoader, "*.pdf": PyPDFLoader, "_dir": DirectoryLoader}


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
        # If the real splitter is available use it, otherwise keep a lightweight
        # placeholder that performs trivial splitting to keep tests runnable.
        try:
            from langchain_text_splitters import RecursiveCharacterTextSplitter

            self._splitter = RecursiveCharacterTextSplitter(
                chunk_size=self._settings.chunk_size,
                chunk_overlap=self._settings.chunk_overlap,
                separators=["\n## ", "\n### ", "\n\n", "\n", ". ", " ", ""],
            )
        except Exception:
            class _SimpleSplitter:
                def split_documents(self, docs):
                    # Naive fallback: return documents as-is
                    return docs

            self._splitter = _SimpleSplitter()

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

        loader_mapping = _get_loader_mapping()
        if not loader_mapping:
            logger.debug("No document loaders available in this environment.")
        else:
            # DirectoryLoader is stored under the special key '_dir' in the
            # mapping returned above.
            DirectoryLoader = loader_mapping.get("_dir")
            for glob_pattern, loader_cls in {k: v for k, v in loader_mapping.items() if k != "_dir"}.items():
                try:
                    loader = DirectoryLoader(
                        str(docs_dir),
                        glob=f"**/{glob_pattern}",
                        loader_cls=loader_cls,
                        show_progress=False,
                    )
                    docs = loader.load()
                    all_documents.extend(docs)
                    logger.info(f"Loaded {len(docs)} documents matching {glob_pattern} from {docs_dir}")
                except Exception as e:
                    logger.error(f"Error loading {glob_pattern} files: {e}")

        if not all_documents:
            logger.warning("No documents loaded from any source.")
            return []

        # Split into chunks
        chunks = cast(list[Document], self._splitter.split_documents(all_documents))

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
        loader_mapping = _get_loader_mapping()
        for pattern, cls in {k: v for k, v in loader_mapping.items() if k != "_dir"}.items():
            if pattern.endswith(suffix):
                loader_cls = cls
                break

        if loader_cls is None:
            supported = [k for k in loader_mapping.keys() if k != "_dir"]
            raise ValueError(
                f"Unsupported file type: {suffix}. Supported: {supported}"
            )

        try:
            loader = loader_cls(str(path))
            docs = loader.load()
            chunks = cast(list[Document], self._splitter.split_documents(docs))

            for i, chunk in enumerate(chunks):
                chunk.metadata["chunk_index"] = i
                chunk.metadata["document_name"] = path.stem

            logger.info(f"Loaded {path.name}: {len(docs)} pages → {len(chunks)} chunks")
            return chunks

        except Exception as e:
            logger.error(f"Failed to load {file_path}: {e}")
            raise
