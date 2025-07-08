from loader import extract_text_from_pdf
from splitter import chunk_text
from embedder import embed_chunks
from store import store_chunks
from assign_sections import assign_sections_to_chunks
from store import store_labeled_chunks_from_embeddings

pdf_path = "../data/raw/aapl_2024.pdf"
print("here")
text = extract_text_from_pdf(pdf_path)
chunks = chunk_text(text)
embeddings = embed_chunks(chunks)
# store_chunks(chunks, embeddings, {"company": "Apple"})
labeled_chunks = assign_sections_to_chunks(chunks,embeddings)
for i in range(len(labeled_chunks)):
    labeled_chunks[i]["embedding"] = embeddings[i]

store_labeled_chunks_from_embeddings(
    labeled_chunks=labeled_chunks,
    company="Tesla",
    year="2024",
    source="earnings_call"
)
