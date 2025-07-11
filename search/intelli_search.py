import os
import chromadb
from typing import List, Dict
import dotenv

def init_client():
    try:
        # path = dotenv.get_key("../.env","path")
        dotenv.load_dotenv()
        path = os.getenv("path")
        if not path :
            raise Exception("No path found")
        persistent_client = chromadb.PersistentClient(path=path)
        return persistent_client
    except Exception as e:
        print("Error while initialising client : ",e)



def get_section_chunks(section : str, company : str, years : List[str|int]):
    try:
        result = {}
        client = init_client()
        collection = client.get_collection("labeled_chunks")
        for year in years:
            year = str(year)
            print(f"finding for {company}_{year} under the section : {section}")
            get_res = client.get_collection("labeled_chunks").get(
                                                     where={"$and":[{"company":company},{"year":year},{"section":section}]},
                                                     include=['documents']
                                                    )
            result[(int)(year)] = get_res["documents"]
            # print(get_res)
            print(f"found {len(result[(int)(year)])} chunks for {company}_{year} under the section : {section}")
        return result
    except Exception as e:
        print("no client or collection")
        print(e)
        return {}