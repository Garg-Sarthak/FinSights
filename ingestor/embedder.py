from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

def embed_chunks(chunks):
    print(f"embedding : begin")
    embeddings = model.encode(chunks, show_progress_bar=True)
    print(f"embedding : end")
    return embeddings
