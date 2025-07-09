from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_text_splitters import SentenceTransformersTokenTextSplitter, SpacyTextSplitter

def chunk_text(text,chunk_size = 1000, chunk_overlap = 200):
    print(f"splitting text : begin")
    splitter = SentenceTransformersTokenTextSplitter(chunk_size = chunk_size,
                                              chunk_overlap = chunk_overlap,
                                              model_name="all-MiniLM-L6-v2",)
    docs = splitter.split_text(text)
    print(f"text split into : {len(docs)} chunks")
    print(f"splitting text : end")
    return docs
    