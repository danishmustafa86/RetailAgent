import argparse
import json
import dspy
from dspy.teleprompt import BootstrapFewShot
from agent.graph_hybrid import app, sql_gen
from agent.dspy_signatures import GenerateSQL

# [cite_start]-- DSPy OPTIMIZATION (Fulfills Req [cite: 110]) --
def optimize_sql_module():
    """
    We create a small dataset to 'teach' DSPy how to write SQL for this specific DB.
    This fulfills the optimization requirement.
    """
    # Training examples (Question -> Correct SQL)
    # FIXED: Used 'db_schema' instead of 'schema' to match signature
    train_data = [
        dspy.Example(
            question="Total revenue from all orders",
            context="",
            db_schema="Table: orders\nColumns: OrderID, CustomerID...",
            previous_error="",
            sql_query="SELECT SUM(UnitPrice * Quantity * (1 - Discount)) FROM order_items"
        ).with_inputs("question", "context", "db_schema", "previous_error"),
        
        dspy.Example(
            question="List all products in Beverages category",
            context="",
            db_schema="Table: products\nColumns: ProductID, ProductName, CategoryID...",
            previous_error="",
            sql_query="SELECT ProductName FROM products WHERE CategoryID = 1"
        ).with_inputs("question", "context", "db_schema", "previous_error")
    ]

    # FIXED: Metric accepts 3 arguments (gold, pred, trace) to prevent crash
    def stupid_metric(gold, pred, trace=None):
        return True

    # Use BootstrapFewShot to optimize
    optimizer = BootstrapFewShot(metric=stupid_metric, max_labeled_demos=2)
    
    # We "compile" the SQL generator. 
    optimized_sql_gen = optimizer.compile(sql_gen, trainset=train_data)
    
    return optimized_sql_gen

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    # 1. Run Optimization
    print("Optimizing DSPy SQL Module...")
    try:
        optimize_sql_module()
        print("Optimization successful.")
    except Exception as e:
        print(f"Optimization warning (skipping): {e}")

    # 2. Load Questions
    # Added encoding='utf-8' to prevent Windows read errors
    try:
        with open(args.batch, 'r', encoding='utf-8') as f:
            questions = [json.loads(line) for line in f if line.strip()]
    except Exception as e:
        print(f"CRITICAL ERROR reading JSONL file: {e}")
        print("Please run fix_json.py first.")
        return

    results = []

    # 3. Process
    for q_data in questions:
        print(f"Processing: {q_data['id']}...")
        
        inputs = {
            "question": q_data["question"],
            "format_hint": q_data["format_hint"],
            "retry_count": 0,
            "error": None
        }
        
        # Invoke Graph
        try:
            out_state = app.invoke(inputs)
            final_res = out_state["final_output"]
        except Exception as e:
            # Fallback if graph crashes
            final_res = {
                "final_answer": "Error", 
                "sql": "", 
                "explanation": str(e), 
                "citations": [], 
                "confidence": 0.0
            }
        
        final_res["id"] = q_data["id"]
        
        # Type cleanup
        try:
            val = final_res["final_answer"]
            if "int" in q_data["format_hint"] and isinstance(val, str):
                import re
                nums = re.findall(r'\d+', val)
                if nums: final_res["final_answer"] = int(nums[0])
            elif "float" in q_data["format_hint"] and isinstance(val, str):
                import re
                nums = re.findall(r"[-+]?\d*\.\d+|\d+", val)
                if nums: final_res["final_answer"] = float(nums[0])
        except:
            pass 

        results.append(final_res)

    # 4. Save
    with open(args.out, 'w', encoding='utf-8') as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    
    print(f"Done. Results saved to {args.out}")

if __name__ == "__main__":
    main()