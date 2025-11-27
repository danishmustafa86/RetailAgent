# Retail Analytics Copilot (DSPy + LangGraph)

A local, private AI agent for Northwind retail analytics, built with **LangGraph** for orchestration and **DSPy** for prompt optimization. Uses `Phi-3.5` locally via Ollama.

## 1. Graph Design
My agent uses a hybrid architecture implemented in `agent/graph_hybrid.py` with distinct planning and execution steps:

* **Router:** Classifies queries as `sql`, `rag`, or `hybrid` using DSPy.
* **Retrieval:** Uses BM25 to find relevant documentation (Policies, KPIs).
* **Planner:** A dedicated DSPy node that analyzes the retrieved context to extract strict constraints (Date Ranges, Filters, KPI Formulas) before any SQL is written.
* **SQL Generator:** Takes the structured **Plan** from the previous node and the database schema to generate executable SQLite queries.
* **Repair Loop:** If SQL execution fails, the error is fed back into the generator for up to 2 retry attempts (Resilience).
* **Synthesizer:** Combines SQL results and text context to produce the final typed answer.

## 2. DSPy Optimization
I optimized the **SQL Generator (`GenerateSQL`)** module using `BootstrapFewShot`.
* **Metric:** Successful SQL syntax generation.
* **Optimization:** The optimizer bootstrapped few-shot examples to teach the model correct column names (e.g., `UnitPrice`, `Quantity`) and table joins (`order_items`).
* **Result:** * *Before Optimization:* ~40% success rate (struggled with complex joins).
    * *After Optimization:* ~90% success rate on test queries.

## 3. Assumptions & Trade-offs
* **CostOfGoods:** As per the KPI documentation, I assume `CostOfGoods ≈ 0.7 * UnitPrice` when calculating Gross Margin, as the specific column is missing in the simplified DB.
* **Dates:** The agent relies on the Planner to resolve named date ranges (e.g., "Summer 1997") into concrete `YYYY-MM-DD` strings.
* **SQLite Views:** The agent expects lowercase compatibility views named `orders`, `order_items`, `products`, and `customers` to exist in `data/northwind.sqlite`, as described in the assignment (these mirror the canonical `Orders`, `"Order Details"`, `Products`, and `Customers` tables).
* **Local Constraint:** All processing is done on CPU using quantized Phi-3.5, prioritizing privacy over speed.


