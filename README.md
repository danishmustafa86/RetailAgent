# Retail Analytics Copilot (DSPy + LangGraph)

A local, private AI agent for Northwind retail analytics, built with **LangGraph** for orchestration and **DSPy** for prompt optimization. Uses `Llama 3.1 8B` locally via Ollama.

## 1. Graph Design
My agent uses a hybrid architecture implemented in `agent/graph_hybrid.py` with distinct planning and execution steps:

* **Router:** Classifies queries as `sql`, `rag`, or `hybrid` using DSPy.
* **Retrieval:** Uses BM25 with improved chunking (header-aware + bullet-list splitting) and query expansion for better recall on policy/KPI questions.
* **Planner:** A dedicated DSPy node that analyzes the retrieved context to extract strict constraints (Date Ranges, Filters, KPI Formulas) before any SQL is written.
* **SQL Generator:** Takes the structured **Plan** from the previous node and the database schema to generate executable SQLite queries.
* **Repair Loop:** If SQL execution fails, the error is fed back into the generator for up to 2 retry attempts (Resilience).
* **Synthesizer:** Combines SQL results and text context to produce the final typed answer.

## 2. DSPy Optimization
I optimized the **SQL Generator (`GenerateSQL`)** module using `BootstrapFewShot`.
* **Module:** The SQL generator is compiled with few-shot examples demonstrating correct SQLite syntax, table aliases, and JOIN patterns.
* **Metric:** Valid SQL execution rate (queries that run without syntax errors).
* **Approach:** Added explicit examples showing proper use of `BETWEEN`, `COALESCE`, table prefixes (`oi.UnitPrice`), and mandatory JOINs when referencing related tables.
* **Impact:** Significantly reduced common errors like missing JOINs, ambiguous column names, and invented SQL functions.

## 3. Assumptions & Trade-offs
* **CostOfGoods:** As per the KPI documentation, I assume `CostOfGoods ≈ 0.7 * UnitPrice` when calculating Gross Margin, as the specific column is missing in the simplified DB.
* **Dates:** The agent relies on the Planner to resolve named date ranges (e.g., "Summer 1997") into concrete `YYYY-MM-DD` strings.
* **SQLite Views:** The agent expects lowercase compatibility views named `orders`, `order_items`, `products`, and `customers` to exist in `data/northwind.sqlite`, as described in the assignment (these mirror the canonical `Orders`, `"Order Details"`, `Products`, and `Customers` tables).
* **Local Model:** All processing is done on CPU using quantized Llama 3.1 8B (via Ollama), prioritizing privacy and accuracy over speed.
* **Improvements:** 
  - Switched from Phi-3.5 to Llama 3.1 for significantly better SQL generation quality and reduced syntax errors.
  - Enhanced RAG chunking to split bullet-list documents (like `product_policy.md`) line-by-line, preserving headers for context.
  - Added query expansion in BM25 search to improve retrieval recall for domain-specific terms (e.g., "return policy" → "returns policy days window").


