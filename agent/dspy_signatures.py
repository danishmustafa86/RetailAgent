import dspy

class RouterSignature(dspy.Signature):
    """Classify user question into: 'rag' (policy/text), 'sql' (pure data stats), or 'hybrid' (requires lookup then data)."""
    question = dspy.InputField()
    classification = dspy.OutputField(desc="One of: 'rag', 'sql', 'hybrid'")

class GenerateSQL(dspy.Signature):
    """Generate executable SQLite query. 
    Rules:
    - Use 'order_items' (OrderID, ProductID, UnitPrice, Quantity, Discount).
    - Use 'orders', 'products', 'customers'.
    - Revenue = SUM(UnitPrice * Quantity * (1 - Discount)).
    - For dates, use YYYY-MM-DD strings.
    """
    question = dspy.InputField()
    context = dspy.InputField(desc="RAG context providing dates or definitions")
    # CHANGED: 'schema' -> 'db_schema' to avoid conflict
    db_schema = dspy.InputField() 
    previous_error = dspy.InputField(desc="Error from previous attempt, if any")
    
    sql_query = dspy.OutputField(desc="SQL query starting with SELECT")

class SynthesizeAnswer(dspy.Signature):
    """Answer the question based on tool outputs. Follow the format hint exactly."""
    question = dspy.InputField()
    context = dspy.InputField(desc="Combined SQL results and RAG docs")
    format_hint = dspy.InputField(desc="Strict output format (int, float, json)")
    
    final_answer = dspy.OutputField(desc="The answer matching the format hint")
    explanation = dspy.OutputField(desc="Brief logic explanation (max 2 sentences)")