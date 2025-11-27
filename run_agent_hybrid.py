import argparse
import json
import multiprocessing
import re

import dspy
from dspy.teleprompt import BootstrapFewShot

from agent.graph_hybrid import build_app, sql_gen as base_sql_gen, app as default_app


# -- DSPy OPTIMIZATION --
def optimize_sql_module():
    print("   [Optimizer] Starting optimization...")

    # Training examples (Question -> Plan -> Correct SQL)
    train_data = [
        dspy.Example(
            question="Total revenue from all orders",
            plan="Calculate sum of (UnitPrice * Quantity * (1-Discount)) from order_items table.",
            db_schema="Table: orders\nColumns: OrderID, CustomerID...",
            previous_error="",
            sql_query="SELECT SUM(UnitPrice * Quantity * (1 - Discount)) FROM order_items",
        ).with_inputs("question", "plan", "db_schema", "previous_error"),
        dspy.Example(
            question="List all products in Beverages category",
            plan="Filter products where CategoryID corresponds to Beverages (ID 1).",
            db_schema="Table: products\nColumns: ProductID, ProductName, CategoryID...",
            previous_error="",
            sql_query="SELECT ProductName FROM products WHERE CategoryID = 1",
        ).with_inputs("question", "plan", "db_schema", "previous_error"),
    ]

    def sql_metric(gold, pred, trace=None):
        """Very lightweight metric: reward non-empty SQL that compiles pattern-wise."""
        try:
            return isinstance(pred.sql_query, str) and "select" in pred.sql_query.lower()
        except Exception:
            return False

    # Note: On Windows, this step must be under freeze_support in __main__
    optimizer = BootstrapFewShot(metric=sql_metric, max_labeled_demos=2)
    optimized_sql_gen = optimizer.compile(base_sql_gen, trainset=train_data)

    print("   [Optimizer] Optimization finished.")
    return optimized_sql_gen


def _validate_and_fix_answer(raw_value, format_hint):
    """
    Validate and coerce final_answer to match the required format_hint.
    Returns (ok: bool, fixed_value).
    """
    fh = format_hint.strip()
    fh_canonical = fh.replace(" ", "")

    # Simple scalar types
    if fh == "int":
        if isinstance(raw_value, int):
            return True, raw_value
        if isinstance(raw_value, (float,)):
            return True, int(raw_value)
        if isinstance(raw_value, str):
            nums = re.findall(r"\d+", raw_value)
            if nums:
                return True, int(nums[0])
        return False, raw_value

    if fh == "float":
        if isinstance(raw_value, (int, float)):
            return True, float(raw_value)
        if isinstance(raw_value, str):
            nums = re.findall(r"[-+]?\d*\.\d+|\d+", raw_value)
            if nums:
                return True, float(nums[0])
        return False, raw_value

    # Known object formats for this assignment
    if fh_canonical == "{category:str,quantity:int}":
        # Expect dict with keys category (str) and quantity (int)
        if isinstance(raw_value, dict):
            cat = raw_value.get("category")
            qty = raw_value.get("quantity")
            if isinstance(cat, str) and isinstance(qty, (int, float)):
                return True, {"category": cat, "quantity": int(qty)}
        if isinstance(raw_value, str):
            try:
                obj = json.loads(raw_value)
                cat = obj.get("category")
                qty = obj.get("quantity")
                if isinstance(cat, str) and isinstance(qty, (int, float)):
                    return True, {"category": cat, "quantity": int(qty)}
            except Exception:
                pass
        return False, raw_value

    if fh_canonical == "{customer:str,margin:float}":
        if isinstance(raw_value, dict):
            cust = raw_value.get("customer")
            marg = raw_value.get("margin")
            if isinstance(cust, str) and isinstance(marg, (int, float)):
                return True, {"customer": cust, "margin": float(marg)}
        if isinstance(raw_value, str):
            try:
                obj = json.loads(raw_value)
                cust = obj.get("customer")
                marg = obj.get("margin")
                if isinstance(cust, str) and isinstance(marg, (int, float)):
                    return True, {"customer": cust, "margin": float(marg)}
            except Exception:
                pass
        return False, raw_value

    if fh_canonical == "list[{product:str,revenue:float}]":
        # Expect list of dicts with product (str) and revenue (float)
        if isinstance(raw_value, list):
            fixed_list = []
            for item in raw_value:
                if not isinstance(item, dict):
                    return False, raw_value
                prod = item.get("product")
                rev = item.get("revenue")
                if not isinstance(prod, str) or not isinstance(rev, (int, float)):
                    return False, raw_value
                fixed_list.append({"product": prod, "revenue": float(rev)})
            return True, fixed_list
        if isinstance(raw_value, str):
            try:
                arr = json.loads(raw_value)
                if isinstance(arr, list):
                    fixed_list = []
                    for item in arr:
                        prod = item.get("product")
                        rev = item.get("revenue")
                        if not isinstance(prod, str) or not isinstance(rev, (int, float)):
                            return False, raw_value
                        fixed_list.append({"product": prod, "revenue": float(rev)})
                    return True, fixed_list
            except Exception:
                pass
        return False, raw_value

    # Fallback: assume it's acceptable as-is
    return True, raw_value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    # 1. Run Optimization and build graph
    print("1. Optimizing DSPy SQL Module...")
    try:
        optimized_sql_gen = optimize_sql_module()
        app = build_app(optimized_sql_gen)
        print("   Success. Using optimized SQL generator.")
    except Exception as e:
        print(f"   Warning: Optimization failed ({e}). Continuing with base module.")
        app = default_app

    # 2. Load Questions
    try:
        with open(args.batch, "r", encoding="utf-8") as f:
            questions = [json.loads(line) for line in f if line.strip()]
    except Exception as e:
        print(f"CRITICAL ERROR reading JSONL file: {e}")
        return

    results = []

    # 3. Process
    print(f"2. Processing {len(questions)} questions...")
    for q_data in questions:
        print(f"   > Processing ID: {q_data['id']}...")

        inputs = {
            "question": q_data["question"],
            "format_hint": q_data["format_hint"],
            "retry_count": 0,
            "error": None,
        }

        final_res = None
        last_exception = None

        # Repair loop: up to 2 attempts on output shape in addition to SQL repair in-graph
        for attempt in range(3):
            try:
                out_state = app.invoke(inputs)
                final_res = out_state["final_output"]
            except Exception as e:
                last_exception = e
                inputs["retry_count"] = inputs.get("retry_count", 0) + 1
                inputs["error"] = str(e)
                continue

            # Validate and coerce final_answer based on format_hint
            ok, fixed_val = _validate_and_fix_answer(
                final_res.get("final_answer"), q_data["format_hint"]
            )
            final_res["final_answer"] = fixed_val
            if ok:
                break

            # Mark validation error and give the graph another chance
            inputs["retry_count"] = inputs.get("retry_count", 0) + 1
            inputs["error"] = "output_format_mismatch"

        if final_res is None:
            # Catastrophic failure fallback
            final_res = {
                "final_answer": "Error",
                "sql": "",
                "explanation": str(last_exception) if last_exception else "Unknown error",
                "citations": [],
                "confidence": 0.0,
            }

        final_res["id"] = q_data["id"]
        results.append(final_res)

    # 4. Save
    with open(args.out, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    print(f"Done. Results saved to {args.out}")


if __name__ == "__main__":
    # --- WINDOWS PROTECTION BLOCK ---
    multiprocessing.freeze_support()
    main()