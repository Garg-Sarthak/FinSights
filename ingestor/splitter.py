from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunk_text(text,chunk_size = 1000, chunk_overlap = 200):
    print(f"splitting text : begin")
    splitter = RecursiveCharacterTextSplitter(chunk_size = chunk_size,
                                              chunk_overlap = chunk_overlap)
    docs = splitter.split_text(text)
    print(f"splitting text : end")
    return docs
    