import dspy
from typing import TypedDict, List, Literal
from langgraph.graph import StateGraph, END

# Import our components
from agent.dspy_signatures import RouterSignature, GenerateSQL, SynthesizeAnswer
from agent.rag.retrieval import LocalRetriever
from agent.tools.sqlite_tool import get_schema_string, execute_query

# -- SETUP DSPy --
# Connect to Ollama
lm = dspy.LM("ollama_chat/phi3.5:3.8b-mini-instruct-q4_K_M", api_base="http://localhost:11434", api_key="")
dspy.configure(lm=lm)

# Initialize Modules
router = dspy.Predict(RouterSignature)
sql_gen = dspy.ChainOfThought(GenerateSQL) # CoT helps SQL generation
synthesizer = dspy.Predict(SynthesizeAnswer)
retriever = LocalRetriever()

# -- STATE --
class AgentState(TypedDict):
    question: str
    format_hint: str
    route: str
    rag_context: str
    citations: List[str]
    sql_query: str
    sql_result: str
    error: str
    retry_count: int
    final_output: dict

# -- NODES --

def route_question(state: AgentState):
    pred = router(question=state["question"])
    return {"route": pred.classification.lower().strip()}

def retrieve_docs(state: AgentState):
    # Search docs
    results = retriever.search(state["question"])
    # If hybrid, we might want to search specifically for entities mentioned
    context_str = "\n".join([r[0] for r in results])
    citations = [r[1] for r in results]
    return {"rag_context": context_str, "citations": citations}

def generate_sql_node(state: AgentState):
    schema_str = get_schema_string()
    
    prev_error = state.get("error", "")
    
    # CHANGED: schema=schema -> db_schema=schema_str
    pred = sql_gen(
        question=state["question"],
        context=state.get("rag_context", ""),
        db_schema=schema_str,
        previous_error=prev_error
    )
    
    clean_sql = pred.sql_query.replace("```sql", "").replace("```", "").strip()
    return {"sql_query": clean_sql}

def execute_sql_node(state: AgentState):
    query = state["sql_query"]
    result, error = execute_query(query)
    
    if error:
        return {"error": error, "retry_count": state.get("retry_count", 0) + 1}
    else:
        return {"sql_result": result, "error": None}

def synthesize_node(state: AgentState):
    context = f"RAG Context: {state.get('rag_context','')}\nSQL Result: {state.get('sql_result','')}"
    
    pred = synthesizer(
        question=state["question"],
        context=context,
        format_hint=state["format_hint"]
    )
    
    # Determine DB citations based on SQL content
    db_citations = []
    if state.get("sql_query"):
        q_lower = state["sql_query"].lower()
        if "orders" in q_lower: db_citations.append("Orders")
        if "order_items" in q_lower: db_citations.append("Order Details")
        if "products" in q_lower: db_citations.append("Products")
        if "customers" in q_lower: db_citations.append("Customers")
    
    all_citations = state.get("citations", []) + db_citations

    # FIX: Down-weight confidence based on repairs 
    # Start at 1.0. Deduct 0.2 for every retry.
    retry_penalty = state.get("retry_count", 0) * 0.2
    base_score = 1.0 if not state.get("error") else 0.0
    final_confidence = max(0.0, base_score - retry_penalty)

    output = {
        "id": "unknown", 
        "final_answer": pred.final_answer,
        "sql": state.get("sql_query", ""),
        "confidence": round(final_confidence, 2),
        "explanation": pred.explanation,
        "citations": list(set(all_citations))
    }
    return {"final_output": output}
# -- EDGES --

def router_edge(state: AgentState):
    r = state["route"]
    if "sql" in r: return "generate_sql" # Direct SQL
    if "hybrid" in r: return "retrieve_docs" # RAG -> SQL
    return "retrieve_docs" # Default to RAG only

def retrieval_edge(state: AgentState):
    if "hybrid" in state["route"] or "sql" in state["route"]:
        return "generate_sql"
    return "synthesize"

def execution_check(state: AgentState):
    if state["error"]:
        if state["retry_count"] < 2: # Max 2 repairs [cite: 97]
            return "generate_sql"
        else:
            return "synthesize" # Give up and try to answer with what we have
    return "synthesize"

# -- GRAPH --

workflow = StateGraph(AgentState)

workflow.add_node("router", route_question)
workflow.add_node("retrieve_docs", retrieve_docs)
workflow.add_node("generate_sql", generate_sql_node)
workflow.add_node("execute_sql", execute_sql_node)
workflow.add_node("synthesize", synthesize_node)

workflow.set_entry_point("router")

workflow.add_conditional_edges("router", router_edge, {
    "generate_sql": "generate_sql",
    "retrieve_docs": "retrieve_docs"
})

workflow.add_conditional_edges("retrieve_docs", retrieval_edge, {
    "generate_sql": "generate_sql",
    "synthesize": "synthesize"
})

workflow.add_edge("generate_sql", "execute_sql")

workflow.add_conditional_edges("execute_sql", execution_check, {
    "generate_sql": "generate_sql", # Repair loop
    "synthesize": "synthesize"
})

workflow.add_edge("synthesize", END)

app = workflow.compile()