# 🤖 Agent-Driven Qualitative Analysis Platform (FinSights)

An end-to-end, conversational AI platform for analyzing how key financial sections—like **Risk Factors** and **Management's Discussion & Analysis**—evolve across years in earnings call transcripts and 10-K filings. This project has evolved from a simple CLI tool into a full-fledged, API-driven agentic system.

---

## ✅ What It Does (Current Version)

This project is a complete **semantic document intelligence platform**. It allows users to ask complex, multi-step, natural language questions about financial documents and receive synthesized insights.

### 🔹 1. Ingestion Pipeline (`POST /ingest`)
- Ingests raw PDF documents (10-Ks, earnings transcripts) via a FastAPI endpoint.
- Recursively chunks and embeds text using `sentence-transformers`.
- Innovated a **dynamic labeling system** that assigns section labels (e.g., *Risk Factors*) via vector similarity search against canonical definitions.
- Stores enriched chunks with metadata (`company`, `year`, `section`) in **ChromaDB**.

### 🔹 2. Agentic Analysis Engine (`POST /agent_query`)
- **Multi-Step Conversational Agent:** Built with **LangChain**, the agent can understand complex queries, create a plan, and autonomously use its tools to find an answer.
- **Specialized Toolkit:** The agent is equipped with a suite of analytical tools for summarization, sentiment analysis (using **FinBERT**), two-year comparisons, and multi-year trend analysis.
- **Optimized & Performant:** Implements a token-efficient, batched **Map-Reduce** strategy for processing large documents and parallelizes independent tool calls to ensure low-latency responses.
- **Built-in Guardrails:** A custom system prompt gives the agent a professional persona and strict rules, such as asking for clarification on vague queries and refusing to answer off-topic questions.

### 🔹 3. API-Driven Architecture
- The entire system is built as a **FastAPI** service, decoupling the backend logic from any specific user interface.
- Provides endpoints for both document ingestion and conversational analysis.

---

## 🧭 Future Roadmap

With the core agent and API implemented, the next steps focus on usability and production-readiness:

### 🟡 Streamlit User Interface (Next Up)
- Build a simple, interactive web interface to provide a user-friendly chat experience with the agent and an easy-to-use document uploader.

### 🟡 Asynchronous Ingestion (Celery + Redis)
- Convert the slow, blocking ingestion process into a non-blocking background job using a Celery task queue with a Redis broker. This will make the application highly responsive.

### 🟡 Report Generator
- Add a tool to the agent's toolkit that can take a final analysis and format it into a clean, professional PDF report for download.

---

## 🔧 Current Tech Stack

- **Python**
- **LangChain** – for the core agentic reasoning loop and tool management.
- **FastAPI** – for the robust, API-driven backend architecture.
- **ChromaDB** – for vector storage and semantic retrieval.
- **Google Gemini API** – for scalable summarization and synthesis.
- **Sentence-Transformers** & **FinBERT** – for embedding and sentiment analysis.

---

## 🧠 Why It Matters

Traditional financial document comparison is manual, time-consuming, and limited to keyword searches. This platform automates the process of understanding **what changed**, **where**, and **why**—giving investors and analysts a powerful conversational tool to get fast, synthesized insights across time.

---

## 📁 Project Structure

```bash
.
├── api.py              # FastAPI server (ingest, analyze, agent endpoints)
├── agents/
│   ├── agent.py        # The core agentic loop and custom prompt
│   ├── tools.py        # The agent's toolkit (summarize, compare, sentiment)
├── ingestor/           # The complete data ingestion pipeline
├── search/             # The search functionality based on company, section, etc.
├── llm_functions/
└── README.md
