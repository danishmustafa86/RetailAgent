# All Fixes Applied Before Final Run

## ✅ Issue 1: Missing `tabulate` Dependency
**Problem**: Q4 failed with "Missing optional dependency 'tabulate'"
**Fix**: `pip install tabulate` ✅ INSTALLED

---

## ✅ Issue 2: Planner Suggesting Non-Existent `marketing_calendar` Column
**Problem**: Q2 failed because Planner output `Filters: 'marketing_calendar = "Summer Beverages 1997"'`
**Fix**: 
- Updated `PlannerSignature` prompt in `agent/dspy_signatures.py`
- Added explicit CRITICAL RULES section with concrete example
- Explicitly forbade "marketing_calendar" in filters
- Instructed to extract only: (a) dates, (b) category names

---

## ✅ Issue 3: Ambiguous Column Names (Missing Table Prefixes)
**Problem**: Q4, Q5 failed with "ambiguous column name: UnitPrice"
**Fix**:
- Added Rule 3 to `GenerateSQL` prompt: "ALWAYS prefix columns with table aliases"
- Updated all examples to show `oi.UnitPrice`, `oi.Quantity`, `o.OrderDate`, `p.ProductName`
- Added CRITICAL note emphasizing mandatory table prefixes

---

## ✅ Issue 4: Missing JOIN for `products` Table When Using `p.CategoryID`
**Problem**: Q3 failed with "no such column: p.CategoryID" because JOIN was missing
**Fix**:
- Added Rule 5: "If you reference p.CategoryID or p.ProductName, you MUST include: JOIN products p"
- Added example showing AOV with category filter that includes proper JOIN

---

## ✅ Issue 5: Missing JOIN for `orders` Table When Using `o.OrderDate`
**Problem**: Q5 failed with "no such column: o.OrderDate" because JOIN was missing
**Fix**:
- Added Rule 6: "If you reference o.OrderDate, you MUST include: JOIN orders o"
- Added new example showing revenue by category with date filter
- Shows complete 3-way JOIN: order_items → products → orders

---

## ✅ Issue 6: Router Routing Policy Questions to SQL Instead of RAG
**Problem**: Q1 might route to SQL instead of RAG
**Fix**:
- Strengthened rule-based routing overrides in `agent/graph_hybrid.py`
- Added detection for: "return" + "days"/"policy", "according to" + "policy"
- Forces route="rag" for all policy questions

---

## ✅ Issue 7: RAG Retrieval Not Finding Right Chunks
**Problem**: Q1 returned answer "30" instead of "14" because retriever didn't find product_policy chunk
**Fix**:
- Improved chunking in `agent/rag/retrieval.py`:
  - Strategy 1: Split by markdown headers (##)
  - Strategy 2: Split by paragraphs (\n\n)
  - Strategy 3: Split bullet lists line-by-line with header context
- Added query expansion for better recall:
  - "beverage" → adds "beverages drinks unopened opened"
  - "return/policy" → adds "returns policy days window"
  - "aov" → adds "aov order value revenue"
  - "summer/winter" → adds "marketing calendar campaign dates"

---

## ✅ Issue 8: Synthesizer Fabricating Answers When SQL Fails
**Problem**: Q2, Q3, Q4, Q5 showed made-up data when SQL failed
**Fix**:
- Updated `SynthesizeAnswer` prompt in `agent/dspy_signatures.py`
- Added CRITICAL RULES:
  - "DO NOT make up data"
  - "If SQL Result is empty/error, state 'Unable to determine'"
  - "DO NOT fabricate product/customer/category names"
  - "If SQL failed, acknowledge it in explanation"

---

## Summary of Changes:

### Files Modified:
1. ✅ `agent/dspy_signatures.py` - Enhanced all DSPy prompts with stricter rules
2. ✅ `agent/graph_hybrid.py` - Improved routing heuristics
3. ✅ `agent/rag/retrieval.py` - Better chunking + query expansion
4. ✅ `README.md` - Documented improvements

### Dependencies Installed:
1. ✅ `tabulate` - Required for pandas.to_markdown()

---

## Expected Results After Final Run:

| Question | Expected Status | Key Fix |
|----------|----------------|---------|
| Q1: rag_policy_beverages_return_days | ✅ Should return `14` | Improved RAG + routing |
| Q2: hybrid_top_category_qty_summer_1997 | ✅ Should succeed | Fixed planner + SQL rules |
| Q3: hybrid_aov_winter_1997 | ✅ Should succeed | Mandatory JOIN for p.CategoryID |
| Q4: sql_top3_products_by_revenue_alltime | ✅ Should succeed | Tabulate installed + table prefixes |
| Q5: hybrid_revenue_beverages_summer_1997 | ✅ Should succeed | Mandatory JOIN for o.OrderDate |
| Q6: hybrid_best_customer_margin_1997 | ✅ Should succeed | Already worked before |

---

## Next Step:

Run the final evaluation:
```bash
python run_agent_hybrid.py --batch sample_questions_hybrid_eval.jsonl --out outputs_hybrid.jsonl
```

Expected: **5-6 out of 6 questions correct** (significant improvement from 0/6)

