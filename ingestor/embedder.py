from sentence_transformers import SentenceTransformer


def embed_chunks(chunks):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    print(f"embedding : begin")
    embeddings = model.encode(chunks, show_progress_bar=True)
    print(f"embedding : end")
    return embeddings
