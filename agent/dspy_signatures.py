import dspy

# 1. Router
class RouterSignature(dspy.Signature):
    """Classify the query. Output only one word: 'rag', 'sql', or 'hybrid'."""
    question = dspy.InputField()
    classification = dspy.OutputField(desc="The category")

# 2. Planner (NEW NODE)
class PlannerSignature(dspy.Signature):
    """Review the Context and Question to extract exact constraints for SQL.
    
    CRITICAL RULES:
    1. Marketing calendar names (like 'Summer Beverages 1997', 'Winter Classics 1997') are DOCUMENTATION ONLY.
    2. They are NOT database columns and MUST NOT appear in Filters.
    3. Extract ONLY: (a) the actual DATE RANGE, (b) category names like 'Beverages', 'Dairy Products'.
    
    Example:
    - Context: "Summer Beverages 1997: Dates 1997-06-01 to 1997-06-30. Focus on Beverages."
    - Question: "Revenue during 'Summer Beverages 1997'"
    - date_range: "1997-06-01 AND 1997-06-30"
    - filters: "Beverages" (or CategoryID = 1)
    - column_logic: "SUM(revenue)"
    
    BAD OUTPUT (forbidden): filters: "marketing_calendar = 'Summer Beverages 1997'"
    GOOD OUTPUT: filters: "Beverages" or "CategoryID = 1" """
    question = dspy.InputField()
    context = dspy.InputField(desc="Retrieved documents with dates and policies")
    
    date_range = dspy.OutputField(desc="Start and End dates in YYYY-MM-DD format (or 'None')")
    filters = dspy.OutputField(desc="Category names or product filters (NOT marketing calendar event names)")
    column_logic = dspy.OutputField(desc="How to calculate metrics (e.g., Revenue formula)")

# 3. SQL Generator (Updated to use Plan)
class GenerateSQL(dspy.Signature):
    """You are a SQLite expert. Generate ONE valid SQLite SELECT query.

STRICT RULES:
1. Tables: ONLY orders, order_items, products, customers (optionally Categories via Products.CategoryID).
2. Columns: Use ONLY columns shown in db_schema. DO NOT invent columns (no: CampaignName, returns, marketing_calendar).
3. Table prefixes: ALWAYS prefix columns with table aliases (oi.UnitPrice, oi.Quantity, o.OrderDate, p.ProductName, p.CategoryID, c.CompanyName) to avoid ambiguous column errors.
4. Date filters: ALWAYS use date(o.OrderDate) BETWEEN 'YYYY-MM-DD' AND 'YYYY-MM-DD'. Marketing calendar names are NOT columns.
5. JOINs: If you reference p.CategoryID or p.ProductName, you MUST include: JOIN products p ON oi.ProductID = p.ProductID
6. JOINs: If you reference o.OrderDate, you MUST include: JOIN orders o ON oi.OrderID = o.OrderID
7. FORBIDDEN: DATEDIFF, DATE_PART, EXTRACT, YEAR(), MONTH(), IFNULL (use COALESCE instead).
8. NO comments (--), NO explanations, NO "Note:", NO reasoning text, NO placeholder text.
9. Output format: Start with SELECT, end with semicolon. Clean SQL only.

VALID EXAMPLES (follow these patterns):

Q: Total quantity by category in date range 1997-06-01 to 1997-06-30
A: SELECT p.CategoryID, SUM(oi.Quantity) as qty FROM orders o JOIN order_items oi ON o.OrderID = oi.OrderID JOIN products p ON oi.ProductID = p.ProductID WHERE date(o.OrderDate) BETWEEN '1997-06-01' AND '1997-06-30' GROUP BY p.CategoryID ORDER BY qty DESC LIMIT 1;

Q: Average Order Value in December 1997
A: SELECT ROUND(SUM(oi.UnitPrice * oi.Quantity * (1 - COALESCE(oi.Discount, 0))) / COUNT(DISTINCT o.OrderID), 2) as aov FROM orders o JOIN order_items oi ON o.OrderID = oi.OrderID WHERE date(o.OrderDate) BETWEEN '1997-12-01' AND '1997-12-31';

Q: AOV for category 3 in December 1997
A: SELECT ROUND(SUM(oi.UnitPrice * oi.Quantity * (1 - COALESCE(oi.Discount, 0))) / COUNT(DISTINCT o.OrderID), 2) as aov FROM orders o JOIN order_items oi ON o.OrderID = oi.OrderID JOIN products p ON oi.ProductID = p.ProductID WHERE p.CategoryID = 3 AND date(o.OrderDate) BETWEEN '1997-12-01' AND '1997-12-31';

Q: Top 3 products by revenue all-time
A: SELECT p.ProductName as product, ROUND(SUM(oi.UnitPrice * oi.Quantity * (1 - COALESCE(oi.Discount, 0))), 2) as revenue FROM order_items oi JOIN products p ON oi.ProductID = p.ProductID GROUP BY p.ProductName ORDER BY revenue DESC LIMIT 3;

Q: Revenue from Beverages (CategoryID=1) in June 1997
A: SELECT ROUND(SUM(oi.UnitPrice * oi.Quantity * (1 - COALESCE(oi.Discount, 0))), 2) as revenue FROM order_items oi JOIN products p ON oi.ProductID = p.ProductID JOIN orders o ON oi.OrderID = o.OrderID WHERE p.CategoryID = 1 AND date(o.OrderDate) BETWEEN '1997-06-01' AND '1997-06-30';

CRITICAL: 
- Always use table aliases (oi.UnitPrice, oi.Quantity, o.OrderDate, p.ProductName) to avoid "ambiguous column" errors.
- If you use o.OrderDate, you MUST JOIN orders o.
- If you use p.CategoryID or p.ProductName, you MUST JOIN products p.

NOW: Generate the query for the current question using the Plan. Output ONLY the SQL, nothing else."""
    question = dspy.InputField()
    plan = dspy.InputField(desc="Date ranges and logic extracted by the Planner")
    db_schema = dspy.InputField() 
    previous_error = dspy.InputField()
    
    sql_query = dspy.OutputField(desc="A single valid SQLite SELECT query with NO extra text")

# 4. Synthesizer
class SynthesizeAnswer(dspy.Signature):
    """Answer the user question based STRICTLY on the provided context. DO NOT make up data.
    
    CRITICAL RULES:
    1. If SQL Result is present and valid, extract the answer from it.
    2. If SQL Result shows "No results" or is empty/error, state "Unable to determine" or return appropriate null value.
    3. DO NOT fabricate product names (like "Product A", "Product B"), customer names (like "Customer A"), 
       category names, or numbers that are not in the context.
    4. Match the format_hint EXACTLY:
       - If format_hint is "int" → output a plain integer (e.g., 14)
       - If format_hint is "float" → output a plain number (e.g., 123.45)
       - If format_hint is "{category:str, quantity:int}" → output {"category": "Beverages", "quantity": 150}
       - If format_hint is "list[{product:str, revenue:float}]" → output [{"product": "Chai", "revenue": 1234.56}, ...]
    
    For dict/list format_hints, output valid Python dict/list syntax, NOT a string representation.
    If the SQL failed or returned no results, acknowledge it in the explanation."""
    question = dspy.InputField()
    context = dspy.InputField()
    format_hint = dspy.InputField()
    
    final_answer = dspy.OutputField(desc="The answer value matching format_hint EXACTLY (int, float, dict, or list) based ONLY on context")
    explanation = dspy.OutputField(desc="Short explanation (1-2 sentences), acknowledging if data was unavailable")