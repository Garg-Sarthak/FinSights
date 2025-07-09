from sentence_transformers import SentenceTransformer
import chromadb
import dotenv

model = SentenceTransformer("all-MiniLM-L6-v2")

def initialise_client():
    try:
        path = dotenv.get_key("../.env","path")
        if not path :
            raise Exception("No path found")
        persistent_client = chromadb.PersistentClient(path=path)
        return persistent_client
    except Exception as e:
        print("Error while initialising client : ",e)

def search_vector_database(query_text:str,company:str,top_n=10,year=None):
    try :
        print("initialising client")
        client = initialise_client()
        print("client initialised")

        print("embedding queries...")
        embeddings = model.encode(query_text,show_progress_bar=True,convert_to_numpy=True)
        print("embedding queries ended...")

        # top = client.get_collection("filings").query(query_texts=query_text,n_results=top_n)
        top = client.get_collection("labeled_chunks").get(
                                                     where={"company":"apple"}
                                                    )
        return top 
    except Exception as e:
        print(f"Error while searching vector database : {e}")

if __name__ == "__main__":
    print(search_vector_database("the rise in profits is sharp",company='apple',top_n=1))

