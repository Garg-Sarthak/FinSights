import chromadb
from chromadb.config import Settings
import dotenv
import os
import uuid
dotenv.load_dotenv()


# path = dotenv.get_key("../.env",key_to_get="path")
path = os.getenv("path")

def initialise_client():
    # client = chromadb.Client(Settings(persistent_directory))
    persistent_client = chromadb.PersistentClient(path=path or "../db")
    return persistent_client

def store_labeled_chunks_from_embeddings(
    labeled_chunks,
    collection_name,
    company=None,
    year=None,
    source=None
):
    client = initialise_client()
    collection = client.get_or_create_collection(name=collection_name)
    # collection = initialise_client(collection_name)
    # collection = client.get_or_create_collection(name=collection_name)


    BATCH_SIZE = 4000
    total_chunks = len(labeled_chunks)

    for i in range(0, total_chunks, BATCH_SIZE):
        batch_end = min(i + BATCH_SIZE, total_chunks)
        current_batch = labeled_chunks[i:batch_end]
        documents = []
        embeddings = []
        metadatas = []
        ids = []

        # for idx, chunk in enumerate(labeled_chunks):
        for idx, chunk in enumerate(current_batch):
            documents.append(chunk["chunk"])
            embeddings.append(chunk["embedding"]) 
            ids.append(str(uuid.uuid4()))

            metadatas.append({
                "section": chunk["assigned_sections"][0],  # assuming top-1 section
                "company": company,
                "year": year,
                "source": source,
                "position": idx+i
            })

        collection.add(
            documents=documents,
            embeddings=embeddings,
            ids=ids,
            metadatas=metadatas
        )

        print(f" Stored {len(documents)} labeled chunks with precomputed embeddings in '{collection_name}'")
    print(f"\nStored a total of {total_chunks} labeled chunks in '{collection_name}'")

def store_chunks(chunks, embeddings, metadata):
    collection = setup_init()

    if 'year' not in metadata.keys() : 
        metadata['year'] = None

    # for i, (chunk, vector) in enumerate(zip(chunks, embeddings)):
    #     # if (i==1) : print(f"{i},{vector}")
    #     collection.add(
    #         documents=[chunk],
    #         embeddings=[vector],
    #         ids=[f"{metadata['company'] or ''}_{metadata['year'] or ''}_{i}"],
    #         metadatas=[{"company" : metadata['company'].lower(),"year":metadata["year"] if metadata["year"] else "None"}]

    #     )
    collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids=[f"{metadata['company'].lower()}_{metadata['year']}_{i}" for i in range(len(chunks))],
        metadatas=[
            {
                "company": metadata['company'].lower(),
                "year": str(metadata["year"]) if metadata["year"] else "None"
            }
            for _ in chunks
        ]
    )




if __name__ == "__main__":
    # print(dotenv.get_key("../.env",key_to_get="path"))
    print(f"storing to : {path}")


