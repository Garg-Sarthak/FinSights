# from langchain_community.document_loaders import PyMuPDFLoader as pdfloader
from langchain_pymupdf4llm import PyMuPDF4LLMLoader as pdfloader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import OllamaEmbeddings
from typing import List

def load_file(file_path="aapl.pdf"):
    loader = pdfloader(file_path=file_path,mode='page',
                    pages_delimiter="/n-----PAGE ENDS HERE THIS IS A DELIMTER-----/n")

    data = loader.load_and_split()
    return data

def chunk_file(data : List):
    chunker = SemanticChunker(OllamaEmbeddings())
    docs = chunker.create_documents(data)

    return docs