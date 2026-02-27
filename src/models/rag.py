"""Retrieval Augmented Generation (RAG) implementation for industrial documentation."""

import os
from typing import Dict, Optional, Any

from langchain.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI


class IndustrialRAG:
    """RAG system for industrial automation documentation."""

    def __init__(
        self,
        docs_dir: str = None,
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
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required")

        self.docs_dir = docs_dir or os.path.join(os.getcwd(), "src", "data", "industrial_docs")
        self.embedding_model = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embeddings = OpenAIEmbeddings(model=embedding_model, openai_api_key=self.api_key)
        self.vectorstore = None

    def load_documents(self) -> None:
        """Load documents from the specified directory."""
        if not os.path.exists(self.docs_dir):
            print(f"Directory not found: {self.docs_dir}")
            return

        loaders = []

        try:
            text_loader = DirectoryLoader(self.docs_dir, glob="**/*.txt", loader_cls=TextLoader)
            loaders.append(text_loader)
        except Exception as e:
            print(f"Error loading text files: {e}")

        try:
            pdf_loader = DirectoryLoader(self.docs_dir, glob="**/*.pdf", loader_cls=PyPDFLoader)
            loaders.append(pdf_loader)
        except Exception as e:
            print(f"Error loading PDF files: {e}")

        documents = []
        for loader in loaders:
            try:
                documents.extend(loader.load())
            except Exception as e:
                print(f"Error in document loading: {e}")

        print(f"Loaded {len(documents)} documents")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
        )

        splits = text_splitter.split_documents(documents)
        print(f"Split into {len(splits)} chunks")

        self.vectorstore = Chroma.from_documents(documents=splits, embedding=self.embeddings)

        print("Vector store created successfully")

    def query(self, question: str, model: str = "gpt-3.5-turbo") -> Dict[str, Any]:
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

        llm = ChatOpenAI(model_name=model, temperature=0.7, openai_api_key=self.api_key)

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(),
            return_source_documents=True,
        )

        result = qa_chain({"query": question})

        sources = []
        if hasattr(result, "source_documents"):
            for doc in result.source_documents:
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
