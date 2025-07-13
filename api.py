from fastapi import FastAPI, UploadFile, File, Form, HTTPException
import os
import uvicorn

from ingestor.loader import extract_text_from_pdf
from ingestor.splitter import chunk_text
from ingestor.embedder import embed_chunks
from ingestor.assign_sections import assign_sections_to_chunks
from ingestor.store import store_labeled_chunks_from_embeddings

from search.intelli_search import get_section_chunks
from llm_functions.summarise import summarise_section, compare_sections

from typing import List
from pydantic import BaseModel

app = FastAPI()

@app.get("/")
def health():
    print("server is running fine")
    return "server is running fine"

@app.post("/upload")
async def upload(
    company: str = Form(...),
    year: str = Form(...),
    file: UploadFile = File(...)
):
    print(file,company,year)
    try:
        temp_file_path = f"temp_{file.filename}"
        with open(temp_file_path, "wb") as buffer:
            buffer.write(await file.read())

        print("Step 1/5: Loading text from PDF")
        text = extract_text_from_pdf(temp_file_path)

        print("Step 2/5: Splitting text into chunks")
        chunks = chunk_text(text)

        print("Step 3/5: Embedding text chunks")
        embeddings = embed_chunks(chunks)

        print("Step 4/5: Assigning section labels")
        labeled_chunks = assign_sections_to_chunks(chunks, embeddings)

        print("Step 5/5: Storing labeled chunks...")
        store_labeled_chunks_from_embeddings(
            collection_name="labeled_chunks",
            labeled_chunks=labeled_chunks,
            company=company.lower(),
            year=str(year),
            source=file.filename
        )
        print("\n--- Ingestion Complete ---")

        os.remove(temp_file_path)
        return {"msg": f"File '{file.filename}' for {company}_{year} ingested successfully"}


    except Exception as e:
        if 'temp_file_path' in locals() and os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


class CompareBody(BaseModel):
    section : str
    company : str
    years : List[int]

@app.post("/compare")
async def compare(
   request :CompareBody
):  
    try:
        section = request.section
        company = request.company
        years = request.years
        # years = years_str.split(sep=',')
        print(f"comparing {section} of {company} for years : {years}")
        section_chunks = get_section_chunks(section=section, company=company, years=years)
        if not all(section_chunks.values()):
            print("\nError: Could not retrieve document chunks. Please ensure data for all specified years has been ingested.")
            return

        print("\nStep 1 of 2: Generating summaries...")
        summaries = summarise_section(chunks=section_chunks, company=company, years=years, section=section)
        if not summaries:
            print("\nError: Failed to generate summaries.")
            return

        print("\nStep 2 of 2: Generating comparison...")
        comparison = compare_sections(summaries_by_year=summaries, company=company, section=section, years=years)
        
        print("\n" + "="*25 + " ANALYSIS COMPLETE " + "="*25)
        print(comparison)
        print("="*70)
        return {"comparision_result":comparison}

    except Exception as e:
        print(f"\nAn unexpected error occurred during analysis: {e}")
        # return f"\nAn unexpected error occurred during analysis: {e}"
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")



if __name__ == "__main__":
    uvicorn.run(app,host="127.0.0.1",port=8080,reload=True)


