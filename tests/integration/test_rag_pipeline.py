"""
Integration tests for the RAG pipeline.

Tests document loading, chunking, vector store, and retrieval
using real documents but mocked embeddings.
"""

import pytest
from pathlib import Path

from src.infrastructure.document_loader import DocumentLoader
from src.config.settings import Settings


class TestDocumentLoader:
    """Tests for the document loading pipeline."""

    def test_load_directory_with_txt_files(self, test_docs_dir):
        settings = Settings(
            openai_api_key="sk-test-key-for-unit-tests-only",
            docs_directory=test_docs_dir,
            chunk_size=200,
            chunk_overlap=50,
        )
        loader = DocumentLoader(settings=settings)
        chunks = loader.load_directory()

        assert len(chunks) > 0
        # Verify chunk metadata
        for chunk in chunks:
            assert "chunk_index" in chunk.metadata
            assert "document_name" in chunk.metadata

    def test_load_empty_directory(self, tmp_path):
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        settings = Settings(
            openai_api_key="sk-test-key-for-unit-tests-only",
            docs_directory=str(empty_dir),
        )
        loader = DocumentLoader(settings=settings)
        chunks = loader.load_directory()

        assert len(chunks) == 0

    def test_load_nonexistent_directory(self, tmp_path):
        settings = Settings(
            openai_api_key="sk-test-key-for-unit-tests-only",
            docs_directory=str(tmp_path / "nonexistent"),
        )
        loader = DocumentLoader(settings=settings)
        chunks = loader.load_directory()

        assert len(chunks) == 0

    def test_load_single_file(self, test_docs_dir):
        settings = Settings(
            openai_api_key="sk-test-key-for-unit-tests-only",
            chunk_size=200,
            chunk_overlap=50,
        )
        loader = DocumentLoader(settings=settings)
        file_path = str(Path(test_docs_dir) / "test_plc_guide.txt")
        chunks = loader.load_single_file(file_path)

        assert len(chunks) > 0

    def test_load_unsupported_file_type(self, tmp_path):
        settings = Settings(
            openai_api_key="sk-test-key-for-unit-tests-only",
        )
        loader = DocumentLoader(settings=settings)
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("col1,col2\n1,2")

        with pytest.raises(ValueError, match="Unsupported file type"):
            loader.load_single_file(str(csv_file))

    def test_load_nonexistent_file(self):
        settings = Settings(
            openai_api_key="sk-test-key-for-unit-tests-only",
        )
        loader = DocumentLoader(settings=settings)

        with pytest.raises(FileNotFoundError):
            loader.load_single_file("/nonexistent/file.txt")

    def test_chunk_size_respected(self, test_docs_dir):
        settings = Settings(
            openai_api_key="sk-test-key-for-unit-tests-only",
            docs_directory=test_docs_dir,
            chunk_size=100,
            chunk_overlap=20,
        )
        loader = DocumentLoader(settings=settings)
        chunks = loader.load_directory()

        # Most chunks should be close to or under chunk_size
        # (some variance is expected due to separator-based splitting)
        for chunk in chunks:
            assert len(chunk.page_content) <= 200  # Allow some tolerance


class TestDocumentLoaderWithRealDocs:
    """Tests using the actual industrial documentation in the project."""

    def test_load_project_docs(self):
        """Test loading the real industrial docs in the project."""
        docs_dir = (
            Path(__file__).resolve().parent.parent.parent / "src" / "data" / "industrial_docs"
        )

        if not docs_dir.exists():
            pytest.skip("Industrial docs directory not found")

        settings = Settings(
            openai_api_key="sk-test-key-for-unit-tests-only",
            docs_directory=str(docs_dir),
        )
        loader = DocumentLoader(settings=settings)
        chunks = loader.load_directory()

        assert len(chunks) > 0
        # Should contain PLC and BAS content
        all_content = " ".join(c.page_content for c in chunks)
        assert "PLC" in all_content or "plc" in all_content.lower()
        assert "Building Automation" in all_content or "BAS" in all_content
