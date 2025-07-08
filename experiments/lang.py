from langchain_community.embeddings import HuggingFaceInstructEmbeddings


embeddings = HuggingFaceInstructEmbeddings(
    query_instruction="Represent financial query for retrieval: "
)

if __name__ == "__main__":
    e = embeddings.embed_query("this")
    print(e)