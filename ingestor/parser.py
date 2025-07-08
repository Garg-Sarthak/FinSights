
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
import pymupdf
from typing import List, Dict, Any

model = SentenceTransformer("all-MiniLM-L6-v2")

CANONICAL_SECTIONS = {
    "business_overview": "Overview of the company's business, operations, core products/services, markets served, and strategic vision.",
    
    "management_discussion": "Management's discussion and analysis (MD&A) of financial condition, results of operations, liquidity, and capital resources.",
    
    "risk_factors": "Discussion of key risks, competitive threats, market uncertainties, and factors that could adversely affect performance.",
    
    "financial_highlights": "High-level summary of financial metrics such as revenue, EBITDA, margins, EPS, etc., typically discussed at the start of calls.",
    
    "financial_statements": "Tabular and textual presentation of balance sheet, income statement, cash flow statement, and supporting financial notes.",
    
    "legal_proceedings": "Details of current or potential lawsuits, regulatory investigations, or legal disputes involving the company.",
    
    "competitive_landscape": "Discussion of market competition, competitive advantages, and positioning relative to industry peers.",
    
    "macroeconomic_outlook": "Management commentary on broader economic trends (interest rates, inflation, geopolitics) impacting business operations.",
    
    "guidance_and_forecast": "Forward-looking statements, earnings guidance, and projections provided by management for upcoming quarters or years.",
    
    "strategic_initiatives": "New initiatives, restructuring plans, M&A activity, R&D investments, or geographic expansions underway or planned.",
    
    "capital_allocation": "Commentary on share buybacks, dividends, debt issuance/repayment, or capital expenditure strategies.",
    
    "esg_and_sustainability": "Environmental, social, and governance (ESG) initiatives, goals, disclosures, and impact reporting.",
    
    "executive_commentary": "Prepared remarks by CEO, CFO, or other executives typically given at the beginning of earnings calls.",
    
    "qna_session": "Unscripted analyst Q&A session, including responses from executives to investor/analyst questions.",
    
    "segment_reporting": "Breakdown of performance by business unit, geography, or product category with segment-specific commentary.",
    
    "revenue_drivers": "Detailed discussion of revenue growth drivers, including pricing, volume, customer trends, and product mix shifts.",
    
    "cost_and_margin_analysis": "Insights into COGS, SG&A, operating margins, and any material changes in expense structure.",
    
    "liquidity_and_cashflow": "Commentary on working capital, cash reserves, cash generation, and funding plans.",
    
    "shareholder_updates": "Messages directed at shareholders including investor relations, annual meeting notes, or equity structure changes.",
    
    "accounting_changes": "Disclosure of changes in accounting policy, restatements, or unusual adjustments affecting financials.",
    
    "regulatory_updates": "New government policies, compliance mandates, or industry-specific regulation affecting operations.",
    
    "human_capital": "Talent strategy, layoffs/hiring, DEI updates, employee productivity, or labor-related discussions.",
    
    "technology_and_innovation": "Commentary on new tech initiatives, digital transformation, software systems, or intellectual property."
}

section_names = list(CANONICAL_SECTIONS.keys())
section_descriptions = [CANONICAL_SECTIONS[name] for name in section_names]
golden_embeddings = model.encode(section_descriptions, convert_to_tensor=True)

def loader(file_path:str):
    doc = pymupdf.open(filename=file_path)

    toc = doc.get_layer

def get_baseline_font_size(doc:pymupdf.Document):
    mpp = {}
    for page in doc:    
        for block in page.get_text("dict").get('blocks'):
            if block:
                # print(block)
                if 'lines' in block.keys():
                    for line in block['lines']:
                        for span in line.get('spans'):
                            # print(span.get('size'))
                            size = span.get('size')
                            if size in mpp.keys():
                                mpp[size] += 1
                            else : mpp[size] = 1
    baseLineFontSize = 9.0
    baseFreq = 0
    for key,val in mpp.items():
        if val > baseFreq:
            baseLineFontSize = key
            val = baseFreq
    return baseLineFontSize



def find_headings(doc: pymupdf.Document, baseline_font_size: float) -> List[Dict[str, Any]]:
    headings = []
    font_size_threshold = 1.1

    for page_num, page in enumerate(doc, 1):
        blocks = page.get_text("dict").get("blocks", [])
        for block in blocks:
            if block.get("lines"):
                for line in block.get("lines"):
                    if line.get("spans"):
                        first_span = line["spans"][0]
                        span_size = first_span.get("size", 0)
                        

                        is_large_font = span_size > (baseline_font_size * font_size_threshold)

                        is_bold = first_span.get("flags", 0) & 16
                        if is_bold and is_large_font:
                            full_title = "".join(span.get("text", "") for span in line["spans"]).strip()
                            
                            if len(full_title) > 3:
                                headings.append({
                                    "title": full_title,
                                    "page": page_num,
                                    "bbox": line.get("bbox")  
                                })
    sorted_headings = sorted(headings, key=lambda h: (h["page"], h["bbox"][1]))
    
    return sorted_headings


# def extract_text_between_headings(doc: pymupdf.Document, sorted_headings: list) -> list:
    
#     final_sections = []
    
#     for i in range(len(sorted_headings)):
#         current_heading = sorted_headings[i]
        
#         if i + 1 < len(sorted_headings):
#             next_heading = sorted_headings[i + 1]
#         else:
#             next_heading = None 

#         section_text = ""
#         start_page = current_heading['page']
#         end_page = next_heading['page'] if next_heading else doc.page_count

#         page = doc[start_page - 1]
#         for block in page.get_text("blocks"):
#             if block['bbox'][1] > current_heading['bbox'][1]: 
#                 section_text += block['lines'][0]['spans'][0]['text'] 
        
#         for page_num in range(start_page, end_page - 1): 
#             section_text += doc[page_num].get_text()

#         if next_heading and end_page > start_page:
#             page = doc[end_page - 1]
#             for block in page.get_text("blocks"):
#                 if block['bbox'][1] < next_heading['bbox'][1]:
#                     section_text += block['lines'][0]['spans'][0]['text']

#         final_sections.append((current_heading['name'], section_text))

#     return final_sections

def extract_text_between_headings(doc: pymupdf.Document, sorted_headings: List[Dict[str, Any]]) -> List[tuple]:

    """
    Extracts the text content for each section defined by a sorted list of headings.

    Args:
        doc: The PyMuPDF Document object.
        sorted_headings: A list of heading dictionaries, sorted by page and vertical position.

    Returns:
        A list of tuples, where each tuple is (section_name, section_text).
    """
    final_sections = []
    
    # Iterate through the headings to define the boundaries of each section
    for i in range(len(sorted_headings)):
        current_heading = sorted_headings[i]
        
        # The next heading in the list defines the end of the current section.
        # If it's the last heading, the section runs to the end of the document.
        if i + 1 < len(sorted_headings):
            next_heading = sorted_headings[i + 1]
        else:
            next_heading = None

        # --- Extraction Logic ---
        section_text = ""
        start_page = current_heading['page']
        # If it's the last section, it ends on the last page of the document.
        end_page = next_heading['page'] if next_heading else doc.page_count

        # --- Iterate through all pages that contain this section ---
        for page_num in range(start_page, end_page + 1):
            page = doc[page_num - 1]
            
            # Use page.get_text("blocks") which returns a list of tuples
            blocks = page.get_text("blocks")
            
            for block in blocks:
                # A block is a tuple: (x0, y0, x1, y1, text, block_type, block_no)
                block_y0 = block[1]  # The vertical position of the block's top
                
                # --- Conditionals to decide if a block belongs to the current section ---
                
                # Case 1: All blocks on pages between the start and end heading pages
                if page_num > start_page and page_num < end_page:
                    section_text += block[4] # block[4] is the text content
                
                # Case 2: Blocks on the same page as the start heading
                elif page_num == start_page:
                    # The block must be BELOW the current heading
                    if block_y0 > current_heading['bbox'][1]:
                        # And if the section ends on this same page, it must also be ABOVE the next heading
                        if end_page == start_page and next_heading and block_y0 < next_heading['bbox'][1]:
                            section_text += block[4]
                        # If the section spans multiple pages, take everything below the heading
                        elif end_page > start_page:
                            section_text += block[4]

                # Case 3: Blocks on the same page as the end heading
                elif page_num == end_page:
                    # The block must be ABOVE the next heading
                    if next_heading and block_y0 < next_heading['bbox'][1]:
                        section_text += block[4]

        # Add the classified name and the extracted text to our final list
        final_sections.append((current_heading.get('name', 'unknown'), section_text.strip()))

    return final_sections



import chromadb
import uuid

def store_chunks_to_chromadb(chunks, embeds, collection_name="chunk_embeddings"):
    client = chromadb.Client()
    collection = client.get_or_create_collection(name=collection_name)

    assert len(chunks) == len(embeds), "chunks and embeds must be same length"

    ids = [str(uuid.uuid4()) for _ in range(len(chunks))]

    embed_vectors = [vec.tolist() if not isinstance(vec, list) else vec for vec in embeds]

    collection.add(
        documents=chunks,
        embeddings=embed_vectors,
        ids=ids,
        metadatas=[{"id":1} for _ in range(len(chunks))]
    )

    print(f"Stored {len(chunks)} chunks in collection '{collection_name}'")