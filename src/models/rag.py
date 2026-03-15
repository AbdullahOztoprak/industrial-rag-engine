"""Retrieval Augmented Generation (RAG) implementation for industrial documentation."""

import os
from pathlib import Path
from typing import Any, Optional, cast

from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import SecretStr


class IndustrialRAG:
    """RAG system for industrial automation documentation."""

    def __init__(
        self,
        docs_dir: Optional[str] = None,
        embedding_model: str = "text-embedding-ada-002",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        api_key: Optional[str] = None,
    ):
        """Initialize the RAG system.

        Args:
            docs_dir: Directory containing industrial documentation
            embedding_model: Model to use for embeddings
            chunk_size: Size of text chunks for processing
            chunk_overlap: Overlap between chunks
            api_key: OpenAI API key
        """
        self.api_key: str = api_key or os.getenv("OPENAI_API_KEY") or ""
        if not self.api_key:
            raise ValueError("OpenAI API key is required")

        self.docs_dir = docs_dir or os.path.join(os.getcwd(), "src", "data", "industrial_docs")
        self.embedding_model = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embeddings = OpenAIEmbeddings(
            model=embedding_model,
            api_key=SecretStr(self.api_key),
        )
        self.vectorstore: Optional[Chroma] = None

    def load_documents(self) -> None:
        """Load documents from the specified directory."""
        if not os.path.exists(self.docs_dir):
            print(f"Directory not found: {self.docs_dir}")
            return

        loaders: list[DirectoryLoader] = []

        try:
            text_loader = DirectoryLoader(self.docs_dir, glob="**/*.txt", loader_cls=TextLoader)
            loaders.append(text_loader)
        except Exception as e:
            print(f"Error loading text files: {e}")

        documents = []
        for loader in loaders:
            try:
                documents.extend(loader.load())
            except Exception as e:
                print(f"Error in document loading: {e}")

        # Load PDFs explicitly (DirectoryLoader typing does not accept PyPDFLoader)
        for pdf_path in Path(self.docs_dir).rglob("*.pdf"):
            try:
                documents.extend(PyPDFLoader(str(pdf_path)).load())
            except Exception as e:
                print(f"Error loading PDF file {pdf_path}: {e}")

        print(f"Loaded {len(documents)} documents")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
        )

        splits = text_splitter.split_documents(documents)
        print(f"Split into {len(splits)} chunks")

        self.vectorstore = Chroma.from_documents(documents=splits, embedding=self.embeddings)

        print("Vector store created successfully")

    def query(self, question: str, model: str = "gpt-3.5-turbo") -> dict[str, Any]:
        """Query the RAG system.

        Args:
            question: Question to ask the system
            model: LLM model to use for generation

        Returns:
            Dictionary containing response and sources
        """
        if not self.vectorstore:
            return {
                "answer": "Error: Documents not loaded. Please load documents first.",
                "sources": [],
            }

        llm = ChatOpenAI(model=model, temperature=0.7, api_key=SecretStr(self.api_key))

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(),
            return_source_documents=True,
        )

        result = cast(dict[str, Any], qa_chain({"query": question}))

        sources = []
        source_docs = result.get("source_documents", [])
        for doc in source_docs:
            if hasattr(doc, "metadata") and "source" in doc.metadata:
                sources.append(doc.metadata["source"])

        return {
            "answer": result["result"],
            "sources": list(set(sources)) if sources else [],
        }

    def add_document(self, file_path: str) -> bool:
        """Add a new document to the vector store.

        Args:
            file_path: Path to the document

        Returns:
            Success status
        """
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            return False

        try:
            loader: TextLoader | PyPDFLoader
            if file_path.endswith(".pdf"):
                loader = PyPDFLoader(file_path)
            elif file_path.endswith(".txt"):
                loader = TextLoader(file_path)
            else:
                print(f"Unsupported file type: {file_path}")
                return False

            document = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
            )
            splits = text_splitter.split_documents(document)

            if self.vectorstore:
                self.vectorstore.add_documents(splits)
            else:
                self.vectorstore = Chroma.from_documents(
                    documents=splits, embedding=self.embeddings
                )

            print(f"Added document: {file_path}")
            return True

        except Exception as e:
            print(f"Error adding document: {e}")
            return False
