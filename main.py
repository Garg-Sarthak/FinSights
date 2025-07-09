from typing import List
import argparse

from ingestor.loader import extract_text_from_pdf
from ingestor.splitter import chunk_text
from ingestor.embedder import embed_chunks
from ingestor.assign_sections import assign_sections_to_chunks
from ingestor.store import store_labeled_chunks_from_embeddings
from search.intelli_search import get_section_chunks
from llm_functions.summarise import summarise_section, compare_sections

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def handle_ingestion(file_path: str, company: str, year: str):
    """
    Runs the full data ingestion pipeline for a single PDF document.
    """
    print("="*50)
    print(f"Starting ingestion for: {file_path}")
    print(f"Company: {company.capitalize()}, Year: {year}")
    print("="*50)

    try:
        print("Step 1/5: Loading text from PDF...")
        text = extract_text_from_pdf(file_path)

        print("Step 2/5: Splitting text into chunks...")
        chunks = chunk_text(text)

        print("Step 3/5: Embedding text chunks...")
        embeddings = embed_chunks(chunks)

        print("Step 4/5: Assigning section labels...")
        labeled_chunks = assign_sections_to_chunks(chunks, embeddings)

        print("Step 5/5: Storing labeled chunks...")
        store_labeled_chunks_from_embeddings(
            collection_name="labeled_chunks",
            labeled_chunks=labeled_chunks,
            company=company.lower(),
            year=str(year),
            source=file_path
        )
        print("\n--- Ingestion Complete ---")

    except Exception as e:
        print(f"\nAn error occurred during ingestion: {e}")


def handle_comparison(company: str, years: List[str], section: str):
    """
    Runs the full analysis and comparison pipeline.
    """
    print("="*50)
    print(f"Starting analysis for {company.capitalize()} | Section: {section} | Years: {years}")
    print("="*50)

    try:
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

    except Exception as e:
        print(f"\nAn unexpected error occurred during analysis: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Earnings Transcript Delta Insight Engine.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    parser_ingest = subparsers.add_parser("ingest", help="Ingest and process a new financial document.")
    parser_ingest.add_argument("--file", type=str, required=True, help="Path to the PDF file.")
    parser_ingest.add_argument("--company", type=str, required=True, help="Company name (e.g., 'apple').")
    parser_ingest.add_argument("--year", type=str, required=True, help="Filing year (e.g., '2023').")

    parser_compare = subparsers.add_parser("compare", help="Compare a section across two years for a company.")
    parser_compare.add_argument("--company", type=str, required=True, help="Company name (e.g., 'apple').")
    parser_compare.add_argument("--section", type=str, required=True, help="Section to analyze (e.g., 'risk_factors').")
    parser_compare.add_argument("--years", nargs='+', required=True, help="List of years to compare (e.g., '2022 2023').")

    args = parser.parse_args()

    if args.command == "ingest":
        handle_ingestion(file_path=args.file, company=args.company, year=args.year)
    elif args.command == "compare":
        handle_comparison(company=args.company, years=args.years, section=args.section)