import dspy

# 1. Router
class RouterSignature(dspy.Signature):
    """Classify the query. Output only one word: 'rag', 'sql', or 'hybrid'."""
    question = dspy.InputField()
    classification = dspy.OutputField(desc="The category")

# 2. Planner (NEW NODE)
class PlannerSignature(dspy.Signature):
    """Review the Context and Question to extract exact constraints for SQL."""
    question = dspy.InputField()
    context = dspy.InputField(desc="Retrieved documents with dates and policies")
    
    date_range = dspy.OutputField(desc="Start and End dates in YYYY-MM-DD format (or 'None')")
    filters = dspy.OutputField(desc="Specific Categories, Products, or Regions to filter")
    column_logic = dspy.OutputField(desc="How to calculate metrics (e.g., Revenue formula)")

# 3. SQL Generator (Updated to use Plan)
class GenerateSQL(dspy.Signature):
    """You are a SQLite expert. Generate ONE valid SQLite SELECT query.

STRICT RULES:
1. Tables: ONLY orders, order_items, products, customers (optionally Categories via Products.CategoryID).
2. Columns: Use ONLY columns shown in db_schema. DO NOT invent columns like CampaignName, returns, etc.
3. Date filters: ALWAYS use date(OrderDate) BETWEEN 'YYYY-MM-DD' AND 'YYYY-MM-DD'.
4. FORBIDDEN: DATEDIFF, DATE_PART, EXTRACT, YEAR(), MONTH(), IFNULL (use COALESCE instead).
5. NO comments (--), NO explanations, NO "Note:", NO reasoning text, NO placeholder text.
6. Output format: Start with SELECT, end with semicolon. Clean SQL only.

VALID EXAMPLES (follow these patterns):

Q: Total quantity by category in date range 1997-06-01 to 1997-06-30
A: SELECT p.CategoryID, SUM(oi.Quantity) as qty FROM orders o JOIN order_items oi ON o.OrderID = oi.OrderID JOIN products p ON oi.ProductID = p.ProductID WHERE date(o.OrderDate) BETWEEN '1997-06-01' AND '1997-06-30' GROUP BY p.CategoryID ORDER BY qty DESC LIMIT 1;

Q: Average Order Value in December 1997
A: SELECT ROUND(SUM(oi.UnitPrice * oi.Quantity * (1 - COALESCE(oi.Discount, 0))) / COUNT(DISTINCT o.OrderID), 2) as aov FROM orders o JOIN order_items oi ON o.OrderID = oi.OrderID WHERE date(o.OrderDate) BETWEEN '1997-12-01' AND '1997-12-31';

Q: Top 3 products by revenue all-time
A: SELECT p.ProductName as product, SUM(oi.UnitPrice * oi.Quantity * (1 - COALESCE(oi.Discount, 0))) as revenue FROM order_items oi JOIN products p ON oi.ProductID = p.ProductID GROUP BY p.ProductName ORDER BY revenue DESC LIMIT 3;

NOW: Generate the query for the current question using the Plan. Output ONLY the SQL, nothing else."""
    question = dspy.InputField()
    plan = dspy.InputField(desc="Date ranges and logic extracted by the Planner")
    db_schema = dspy.InputField() 
    previous_error = dspy.InputField()
    
    sql_query = dspy.OutputField(desc="A single valid SQLite SELECT query with NO extra text")

# 4. Synthesizer
class SynthesizeAnswer(dspy.Signature):
    """Answer the user question.
    Format your answer as a JSON object with keys: 'final_answer' and 'explanation'.
    """
    question = dspy.InputField()
    context = dspy.InputField()
    format_hint = dspy.InputField()
    
    final_answer = dspy.OutputField(desc="The answer value matching format_hint")
    explanation = dspy.OutputField(desc="Short explanation string")