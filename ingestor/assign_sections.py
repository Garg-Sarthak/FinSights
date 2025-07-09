from loader import extract_text_from_pdf
from embedder import embed_chunks
from splitter import chunk_text
from sentence_transformers import SentenceTransformer
import chromadb
from langchain_text_splitters import SentenceTransformersTokenTextSplitter, SpacyTextSplitter


def add_sections_to_embeddings():
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

    model = SentenceTransformer("all-MiniLM-L6-v2")
    section_names = list(CANONICAL_SECTIONS.keys())
    section_descriptions = list(CANONICAL_SECTIONS.values())
    section_embeddings = model.encode(section_descriptions,show_progress_bar=True)
    client = chromadb.Client()
    section_collection = client.get_or_create_collection("section_embeddings")

    section_collection.add(
        documents=section_descriptions,
        embeddings=[e.tolist() for e in section_embeddings],
        ids=section_names,
        metadatas=[{"section": name} for name in section_names]
    )
    return section_collection

def assign_sections_to_chunks(chunks, chunk_embeddings, top_k=2, similarity_threshold=0.65):
    section_collection = add_sections_to_embeddings()
    labeled_chunks = []

    for chunk, chunk_embed in zip(chunks, chunk_embeddings):
        results = section_collection.query(
            query_embeddings=[chunk_embed.tolist()],
            n_results=top_k
        )

        sections = results["ids"][0]
        scores = results["distances"][0]

        final_labels = []
        for section, distance in zip(sections, scores):
            if distance < (2 - similarity_threshold): 
                final_labels.append(section)

        if not final_labels:
            final_labels = ["general_body"]

        labeled_chunks.append({
            "chunk": chunk,
            "assigned_sections": final_labels,
            "raw_distances": scores[:len(final_labels)],
            "embedding" : chunk_embed
        })

    return labeled_chunks