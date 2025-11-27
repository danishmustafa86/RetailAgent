import sqlite3
import pandas as pd

DB_PATH = "data/northwind.sqlite"

def get_schema_string():
    """Returns a compact schema string for the LLM."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # We only care about our views/tables
    tables = ['orders', 'order_items', 'products', 'customers']
    schema_str = ""
    
    for t in tables:
        cursor.execute(f"PRAGMA table_info({t})")
        columns = cursor.fetchall()
        col_names = [c[1] for c in columns]
        schema_str += f"Table: {t}\nColumns: {', '.join(col_names)}\n\n"
        
    conn.close()
    return schema_str

def execute_query(sql: str):
    """Executes SQL and returns results + column names or error."""
    try:
        conn = sqlite3.connect(DB_PATH)
        # Safety: Read only allowed logic (basic check)
        if "drop" in sql.lower() or "delete" in sql.lower():
            return None, "Error: Unsafe query detected."
            
        df = pd.read_sql_query(sql, conn)
        conn.close()
        
        if df.empty:
            return "No results found.", None
            
        return df.to_markdown(index=False), None
        
    except Exception as e:
        return None, str(e)