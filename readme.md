# 📊 Earnings-Transcript Delta Insight Engine

An end-to-end, LLM-powered CLI tool for analyzing how key financial sections—like **Risk Factors** and **MD&A**—evolve across years in earnings call transcripts and 10-K filings.

---

## ✅ What It Does (Current MVP)

This project is a complete **semantic document intelligence engine** for company filings. It allows structured, multi-year comparison of narrative financial content.

### 🔹 1. Ingestion Pipeline
- Parses raw PDF documents (10-Ks, earnings transcripts)
- Recursively chunks and embeds text using `sentence-transformers`
- Assigns section labels (e.g., *Risk Factors*, *Business Overview*) via semantic similarity
- Stores enriched chunks with metadata (`company`, `year`, `section`) in **ChromaDB**

### 🔹 2. Insight & Comparison Engine
- Supports section-level retrieval for a given company and year
- Runs token-efficient LLM summarization using Gemini (Map-Reduce batching)
- Highlights **added**, **removed**, and **modified** content between two years

### 🔹 3. CLI Application
- `ingest`: Fully processes and stores a new financial document
- `compare`: Compares the same section across any two years and returns a bullet-point delta summary

---

## 🧭 Plannede Future Enhancements

These features are not part of the current MVP, but are planned for future iterations:

### 🟡 Report Generator
- Export year-over-year delta analysis into clean, professional PDF reports using Markdown → HTML → PDF pipelines

### 🟡 FastAPI Layer
- Expose the CLI functionality via a Frontend, use Redis + Celery to enable parallel asynchronous processing, increasing the rate at which analysis is possible.

### 🟡 LLM Tool Wrappers & Agents
- Wrap core functions into LangChain tools for agentic planning and autonomous multi-hop analysis

### 🟡 Advanced NLP Analytics
- Add sentiment scoring on management commentary 

---

## 🔧 Current Tech Stack

- **Python**  
- **ChromaDB** – for vector storage and filtering  
- **sentence-transformers** – for semantic chunk embedding  
- **Gemini API** – for scalable summarization and delta analysis  

---

## 🧠 Why It Matters

Traditional financial document comparison is manual, time-consuming, and error-prone. This project automates the process of understanding **what changed**, **where**, and **why**—giving investors, analysts, and researchers fast, explainable summaries across time.

---

## 📁 Structure

```bash
.
├── ingest.py           # Ingestion pipeline (PDF → ChromaDB)
├── compare.py          # Section-wise delta analyzer
├── main.py             # CLI wrapper (Typer)
├── utils/              # Chunkers, vector utils, labeling logic
├── store.py            # Storage abstraction over ChromaDB
└── README.md
