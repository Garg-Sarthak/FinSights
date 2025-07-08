import chromadb
from chromadb.config import Settings

# client = chromadb.Client(Settings(persistent_directory))
persistent_client = chromadb.PersistentClient(path="/Users/sarthak/Desktop/FinSights/chroma")
collection = persistent_client.get_or_create_collection(name="filings")

def store_chunks(chunks, embeddings, metadata):
    for i, (chunk, vector) in enumerate(zip(chunks, embeddings)):
        collection.add(
            documents=[chunk],
            embeddings=[vector],
            ids=[f"{metadata['company'] or ''}_{metadata['year'] or ''}_{i}"],
            metadatas=[metadata]
        )


