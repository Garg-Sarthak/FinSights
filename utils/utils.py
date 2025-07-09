import chromadb
import dotenv

def init_client():
    try:
        path = dotenv.get_key("../.env","path")
        if not path :
            raise Exception("No path found")
        persistent_client = chromadb.PersistentClient(path=path)
        return persistent_client
    except Exception as e:
        print("Error while initialising client : ",e)