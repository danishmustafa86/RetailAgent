import dspy 
from typing import TypedDict, List
from langgraph.graph import StateGraph, END  #type:ignore

# Import components
from agent.dspy_signatures import RouterSignature, PlannerSignature, GenerateSQL, SynthesizeAnswer
from agent.rag.retrieval import LocalRetriever
from agent.tools.sqlite_tool import get_schema_string, execute_query

# -- SETUP DSPy --
lm = dspy.LM(
    "ollama_chat/llama3.1:8b",
    api_base="http://localhost:11434",
    api_key=""
)
dspy.configure(lm=lm)

# Initialize base Modules (router / planner / synthesizer / retriever are fixed;
# the SQL generator can be overridden, e.g. with a DSPy-optimized module).
router = dspy.Predict(RouterSignature)
planner = dspy.Predict(PlannerSignature)
sql_gen = dspy.ChainOfThought(GenerateSQL)
synthesizer = dspy.Predict(SynthesizeAnswer)
retriever = LocalRetriever()


# -- STATE --
class AgentState(TypedDict):
    question: str
    format_hint: str
    route: str
    rag_context: str
    plan: str
    citations: List[str]
    sql_query: str
    sql_result: str
    error: str
    retry_count: int
    # Heuristic signal: max BM25 score from retriever for this question
    retrieval_score: float
    final_output: dict


# -- NODES --

def route_question(state: AgentState):
    """Route question to rag / sql / hybrid, with a small rule-based override for obvious cases."""
    question_text = state["question"]
    q_lower = question_text.lower()

    pred = router(question=question_text)
    route = pred.classification.lower().strip()

    # Heuristic overrides to improve behavior on known patterns
    if ("product policy" in q_lower or "returns & policy" in q_lower or "return window" in q_lower 
        or "return" in q_lower and ("days" in q_lower or "policy" in q_lower)
        or "according to" in q_lower and "policy" in q_lower):
        # Pure policy questions should rely on docs only.
        route = "rag"
    elif any(kw in q_lower for kw in ["kpi", "average order value", "aov", "gross margin"]):
        # KPI questions that reference campaigns should usually be hybrid.
        if "summer beverages" in q_lower or "winter classics" in q_lower:
            route = "hybrid"

    print(f"[router] route={route}")
    return {"route": route}


def retrieve_docs(state: AgentState):
    results = retriever.search(state["question"])
    # results are (content, doc_id, score)
    context_str = "\n".join([r[0] for r in results])
    citations = [r[1] for r in results]
    scores = [r[2] for r in results] if results else []
    retrieval_score = max(scores) if scores else 0.0
    print(f"[retrieve_docs] k={len(results)} citations={citations} max_score={retrieval_score:.3f}")
    return {"rag_context": context_str, "citations": citations, "retrieval_score": retrieval_score}


def planner_node(state: AgentState):
    pred = planner(
        question=state["question"],
        context=state.get("rag_context", "")
    )
    plan_str = f"Date Range: {pred.date_range}\nFilters: {pred.filters}\nLogic: {pred.column_logic}"
    print(f"[planner] plan={plan_str.replace(chr(10), ' | ')}")
    return {"plan": plan_str}


def generate_sql_node(state: AgentState):
    schema_str = get_schema_string()
    prev_error = state.get("error", "")

    pred = sql_gen(
        question=state["question"],
        plan=state.get("plan", ""),
        db_schema=schema_str,
        previous_error=prev_error
    )

    # Aggressive SQL cleaning to handle model errors
    raw = pred.sql_query.strip()
    
    # Remove markdown code fences
    raw = raw.replace("```sql", "").replace("```", "")
    
    # Remove common commentary keywords and everything after them
    for marker in ["\nNote:", "\nThis query", "\n--", "InstanceClass", "{reasoning}"]:
        if marker in raw:
            raw = raw.split(marker)[0]
    
    # Take only the first valid SQL statement (up to first semicolon)
    if ";" in raw:
        raw = raw.split(";")[0] + ";"
    
    # Remove inline comments (-- ...) line by line, preserving the SQL
    lines = []
    for line in raw.split("\n"):
        # Strip inline comments but keep the SQL part before them
        if "--" in line:
            line = line.split("--")[0].rstrip()
        if line.strip():
            lines.append(line)
    
    clean_sql = "\n".join(lines).strip()
    print(f"[generate_sql] sql={clean_sql}")
    return {"sql_query": clean_sql}


def execute_sql_node(state: AgentState):
    query = state["sql_query"]
    print(f"[execute_sql] running SQL (len={len(query)})")
    result, error = execute_query(query)

    if error:
        print(f"[execute_sql] error={error}")
        return {"error": error, "retry_count": state.get("retry_count", 0) + 1}
    else:
        print("[execute_sql] success")
        return {"sql_result": result, "error": None}


def synthesize_node(state: AgentState):
    context = f"RAG Context: {state.get('rag_context','')}\nSQL Result: {state.get('sql_result','')}"

    pred = synthesizer(
        question=state["question"],
        context=context,
        format_hint=state["format_hint"]
    )

    # Citations Logic
    db_citations = []
    if state.get("sql_query"):
        q_lower = state["sql_query"].lower()
        if "orders" in q_lower:
            db_citations.append("Orders")
        if "order_items" in q_lower:
            db_citations.append("Order Details")
        if "products" in q_lower:
            db_citations.append("Products")
        if "customers" in q_lower:
            db_citations.append("Customers")

    all_citations = state.get("citations", []) + db_citations

    # Confidence Logic (heuristic):
    # - reward successful SQL with non-empty results
    # - reward strong retrieval scores
    # - down-weight when repaired multiple times
    retry_count = state.get("retry_count", 0)
    retrieval_score = float(state.get("retrieval_score", 0.0) or 0.0)
    # normalize BM25-ish score into [0, 1]
    retr_norm = max(0.0, min(retrieval_score / 5.0, 1.0))

    sql_result = state.get("sql_result")
    sql_ok = bool(sql_result) and sql_result not in ("No results found.", "")
    has_error = bool(state.get("error"))

    base_score = 0.2
    if sql_ok and not has_error:
        base_score += 0.5  # strong signal when SQL executed and returned rows
    if retr_norm > 0.0:
        base_score += 0.3 * retr_norm  # retrieval coverage

    final_confidence = max(0.0, min(1.0, base_score - retry_count * 0.15))

    output = {
        "id": "unknown",
        "final_answer": pred.final_answer,
        "sql": state.get("sql_query", ""),
        "confidence": round(final_confidence, 2),
        "explanation": pred.explanation,
        "citations": list(set(all_citations)),
    }
    print(f"[synthesize] confidence={output['confidence']} citations={output['citations']}")
    return {"final_output": output}


# -- EDGES --

def router_edge(state: AgentState):
    r = state["route"]
    if "sql" in r:
        return "planner"
    if "hybrid" in r:
        return "retrieve_docs"
    return "retrieve_docs"


def retrieval_edge(state: AgentState):
    if "hybrid" in state["route"] or "sql" in state["route"]:
        return "planner"
    return "synthesize"


def execution_check(state: AgentState):
    if state["error"]:
        if state["retry_count"] < 2:
            return "generate_sql"
        else:
            return "synthesize"
    return "synthesize"


# -- GRAPH FACTORY --

def build_app(override_sql_gen=None):
    """
    Build and compile a LangGraph workflow.

    If override_sql_gen is provided (e.g., a DSPy-optimized module),
    it will be used instead of the default sql_gen.
    """
    global sql_gen
    if override_sql_gen is not None:
        sql_gen = override_sql_gen

    workflow = StateGraph(AgentState)

    workflow.add_node("router", route_question)
    workflow.add_node("retrieve_docs", retrieve_docs)
    workflow.add_node("planner", planner_node)
    workflow.add_node("generate_sql", generate_sql_node)
    workflow.add_node("execute_sql", execute_sql_node)
    workflow.add_node("synthesize", synthesize_node)

    workflow.set_entry_point("router")

    workflow.add_conditional_edges("router", router_edge, {
        "planner": "planner",
        "retrieve_docs": "retrieve_docs",
    })

    workflow.add_conditional_edges("retrieve_docs", retrieval_edge, {
        "planner": "planner",
        "synthesize": "synthesize",
    })

    workflow.add_edge("planner", "generate_sql")
    workflow.add_edge("generate_sql", "execute_sql")

    workflow.add_conditional_edges("execute_sql", execution_check, {
        "generate_sql": "generate_sql",
        "synthesize": "synthesize",
    })

    workflow.add_edge("synthesize", END)

    return workflow.compile()


# Default compiled app using the base (non-optimized) SQL generator.
app = build_app()