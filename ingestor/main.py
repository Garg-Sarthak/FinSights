from loader import extract_text_from_pdf
from splitter import chunk_text
from embedder import embed_chunks
from store import store_chunks

pdf_path = "../aapl_2024.pdf"
print("here")
text = extract_text_from_pdf(pdf_path)
chunks = chunk_text(text)
embeddings = embed_chunks(chunks)
store_chunks(chunks, embeddings, {"company": "Apple", "year": 2024})