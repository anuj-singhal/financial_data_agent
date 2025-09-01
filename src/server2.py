# oracle_mcp_server.py
"""
FastMCP-based MCP server that exposes a single tool: execute_query(query: str).
Run: python oracle_mcp_server.py
This server uses the stdio transport by default (mcp.run()).
"""

from typing import Any, Dict, List
import oracledb
from mcp.server.fastmcp import FastMCP

# <-- fill these with your Oracle connection info -->
DB_CONFIG = {
    "user": "C##agenticai",
    "password": "agenticai",
    "dsn": "localhost:1521/orcl"
}

mcp = FastMCP("Oracle MCP Server")

@mcp.tool(description="Run a SQL query on the Oracle DB and return structured output")
async def execute_query(query: str) -> Dict[str, Any]:
    """
    Returns a structured dict:
      - {"columns": [...], "rows": [[...],[...], ...]}  (for SELECT)
      - {"message": "..."}  (for DML)
      - {"error": "..."}    (on exception)
    """
    try:
        conn = oracledb.connect(**DB_CONFIG)
        cur = conn.cursor()
        cur.execute(query)

        if cur.description:  # query returned columns -> SELECT
            cols: List[str] = [d[0] for d in cur.description]
            rows: List[List[Any]] = [list(r) for r in cur.fetchall()]
            cur.close()
            conn.close()
            return {"columns": cols, "rows": rows}

        # No result set -> DML (INSERT/UPDATE/DELETE)
        affected = cur.rowcount
        conn.commit()
        cur.close()
        conn.close()
        return {"message": f"Query executed. Rows affected: {affected}"}

    except Exception as e:
        # Return structured error so clients can show it nicely
        return {"error": str(e)}


if __name__ == "__main__":
    # mcp.run() defaults to stdio transport (good for local/subprocess usage).
    mcp.run()
