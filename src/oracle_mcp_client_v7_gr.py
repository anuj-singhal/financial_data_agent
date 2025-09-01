# oracle_mcp_client_gr_dynamic_charts.py
import asyncio
import os
import json
import hashlib
from contextlib import AsyncExitStack

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import gradio as gr

from mcp.client.stdio import stdio_client, StdioServerParameters
from mcp.client.session import ClientSession
from openai import AsyncOpenAI

# ---------------- Configuration (keep as-is) ----------------
SERVER_SCRIPT = "C:\\Anuj\\AI\\MCP\\mcp-servers\\oracle_server\\server2.py"
llm = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ---------------- Memory & Cache ----------------
conversation_history = []
query_cache = {}  # cache_key -> {"columns": [...], "rows": [...], "task": str}

# ---------------- Data Dictionary (use your own) ----------------
data_dictionary = {
    "uae_banks_financial_data": {
        "description": "Contains quarterly financial performance metrics for UAE banks",
        "columns": {
            "year": "Financial reporting year",
            "quarter": "Quarter 1-4",
            "bank": "Bank name",
            "ytd_income": "Year-to-date total operating income",
            "ytd_profit": "Year-to-date net profit",
            "loans_advances": "Total loans and advances",
            "nim": "Net Interest Margin",
            "deposits": "Total customer deposits",
            "casa": "Current Account and Savings Account deposits",
            "cost_income": "Cost-to-Income ratio",
            "npl_ratio": "Non-Performing Loans ratio",
            "rote": "Return on Tangible Equity",
            "cet1": "Common Equity Tier 1 capital ratio",
            "share_price": "Market share price per share",
        }
    }
}

# ---------------- Utilities ----------------
def get_cache_key(task: str) -> str:
    return hashlib.sha256(task.strip().encode("utf-8")).hexdigest()[:16]

async def summarize_result(columns, rows, max_rows=6) -> str:
    """Ask the LLM for a short summary (1-2 sentences) using small sample rows."""
    sample = rows[:max_rows]
    prompt = f"""
You are an assistant that summarizes database query results.
Columns: {columns}
Sample Rows: {sample}
Generate a very concise 1-2 sentence summary suitable for conversation history.
"""
    resp = await llm.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You summarize DB results concisely."},
            {"role": "user", "content": prompt},
        ],
    )
    return resp.choices[0].message.content.strip()

async def check_cache_for_task(task: str) -> dict | None:
    """Ask LLM if the user's task can be answered from cached results; return the data JSON if yes."""
    if not query_cache:
        return None
    # small summaries for prompt
    summaries = []
    for k, v in query_cache.items():
        cols = v.get("columns", [])
        sample = v.get("rows", [])[:3]
        summaries.append(f"{k}: columns({cols}), sample({sample})")
    prompt = f"""
We have cached query results (key : summary):
{summaries}

Conversation history: {conversation_history}
User task: {task}

Rules:
- If the task can be answered from cache, return JSON:
  {{ "columns": [...], "rows": [...] }}
- If it cannot be answered from cache, return exactly: NO_CACHE
- Do not fabricate any data; only reuse cache.
"""
    resp = await llm.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a cache assistant."},
            {"role": "user", "content": prompt},
        ],
    )
    reply = resp.choices[0].message.content.strip()
    if reply == "NO_CACHE":
        return None
    try:
        return json.loads(reply)
    except Exception as e:
        print("⚠️ Could not parse cache assistant output:", e)
        return None

async def natural_language_to_sql(task: str, tools, data_dict: dict) -> str:
    table_descriptions = "\n".join([
        f"{table}: {info['description']}. Columns: {', '.join(f'{col} ({desc})' for col, desc in info['columns'].items())}"
        for table, info in data_dict.items()
    ])
    tool_descriptions = "\n".join([f"- {t.name}: {t.description}" for t in tools])
    prompt = f"""
You are an Oracle SQL expert. Database schema:
{table_descriptions}

Available tools:
{tool_descriptions}

User task: {task}

Rules:
- Use only existing tables/columns above.
- Return SQL only. If invalid, return NO_QUERY.
- No explanations. Remove special chars like ``` or ;.
"""
    resp = await llm.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Oracle SQL expert assistant."},
            {"role": "user", "content": prompt},
        ],
    )
    sql = resp.choices[0].message.content.strip()
    return sql.replace(";", "")

# ---------------- Chart Generator ----------------
def generate_chart(df: pd.DataFrame, chart_type: str, x_col: str, y_col: str | None, color_col: str | None):
    """Generate a Plotly chart. Supports categorical-driven charts and 3-column visuals."""
    if df is None or df.empty:
        return None

    # Normalize None strings
    color_col = None if (color_col in (None, "None", "")) else color_col

    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    all_cols = df.columns.tolist()
    cat_cols = [c for c in all_cols if c not in numeric_cols]

    # Defensive: if x not available, pick first column
    if x_col not in all_cols:
        x_col = all_cols[0]

    # Helper: aggregated frame for plotting when needed
    def agg_frame(group_cols, value_col=None, agg="sum"):
        if value_col and value_col in numeric_cols:
            return df.groupby(group_cols)[value_col].agg(agg).reset_index()
        else:
            # counts
            return df.groupby(group_cols).size().reset_index(name="count")

    try:
        # BAR / GROUPED / STACKED
        if chart_type in ("bar", "stacked_bar", "grouped_bar"):
            if y_col and y_col in numeric_cols:
                if color_col:
                    agg = agg_frame([x_col, color_col], y_col, "sum")
                    fig = px.bar(agg, x=x_col, y=y_col, color=color_col,
                                 barmode="group" if chart_type=="grouped_bar" else "stack")
                else:
                    agg = agg_frame([x_col], y_col, "sum")
                    fig = px.bar(agg, x=x_col, y=y_col)
            else:
                # y is categorical or None -> counts
                if color_col:
                    agg = agg_frame([x_col, color_col], None)
                    fig = px.bar(agg, x=x_col, y="count", color=color_col, barmode="group")
                else:
                    agg = agg_frame([x_col], None)
                    fig = px.bar(agg, x=x_col, y="count")
            fig.update_layout(barmode="group" if chart_type=="grouped_bar" else ("stack" if chart_type=="stacked_bar" else "group"))

        # LINE
        elif chart_type == "line":
            if y_col and y_col in numeric_cols:
                if color_col:
                    agg = agg_frame([x_col, color_col], y_col, "mean")
                    fig = px.line(agg, x=x_col, y=y_col, color=color_col)
                else:
                    agg = agg_frame([x_col], y_col, "mean")
                    fig = px.line(agg, x=x_col, y=y_col)
            else:
                return None

        # SCATTER
        elif chart_type == "scatter":
            # need numeric y; x can be numeric or categorical (categorical as jitter)
            if y_col not in numeric_cols:
                return None
            if x_col in numeric_cols:
                fig = px.scatter(df, x=x_col, y=y_col, color=color_col)
            else:
                # categorical x -> plot jitter by converting category index to numeric for visualization, show color by category
                fig = px.strip(df, x=x_col, y=y_col, color=color_col, jitter=0.3)

        # SCATTER 3D
        elif chart_type == "scatter_3d":
            # require at least 3 numeric cols
            nums = [c for c in df.columns if c in numeric_cols]
            if len(nums) >= 3:
                fig = px.scatter_3d(df, x=nums[0], y=nums[1], z=nums[2], color=color_col)
            else:
                return None

        # BOX / VIOLIN (distribution of y across categories)
        elif chart_type in ("box", "violin"):
            if y_col in numeric_cols:
                if x_col in cat_cols:
                    if chart_type == "box":
                        fig = px.box(df, x=x_col, y=y_col, color=color_col)
                    else:
                        fig = px.violin(df, x=x_col, y=y_col, color=color_col, box=True)
                else:
                    # if x is numeric, treat x as category by binning
                    df["_x_binned"] = pd.cut(df[x_col], bins=10) if pd.api.types.is_numeric_dtype(df[x_col]) else df[x_col]
                    if chart_type == "box":
                        fig = px.box(df, x="_x_binned", y=y_col, color=color_col)
                    else:
                        fig = px.violin(df, x="_x_binned", y=y_col, color=color_col, box=True)
            else:
                return None

        # PIE (single dimension)
        elif chart_type == "pie":
            if y_col and y_col in numeric_cols:
                agg = agg_frame([x_col], y_col, "sum")
                fig = px.pie(agg, names=x_col, values=y_col)
            else:
                agg = agg_frame([x_col], None)
                fig = px.pie(agg, names=x_col, values="count")

        # HEATMAP for two categories
        elif chart_type == "heatmap":
            # require x_col and color_col to be categorical-like
            if not color_col:
                return None
            # choose agg column numeric if available, else counts
            numeric = df.select_dtypes(include=["number"]).columns
            agg_col = numeric[0] if len(numeric) > 0 else None
            if agg_col:
                pivot = df.pivot_table(index=color_col, columns=x_col, values=agg_col, aggfunc="sum", fill_value=0)
                fig = px.imshow(pivot, labels=dict(x=x_col, y=color_col, color=f"sum({agg_col})"))
            else:
                # counts pivot
                pivot = df.groupby([color_col, x_col]).size().unstack(fill_value=0)
                fig = px.imshow(pivot)

        # TREEMAP / SUNBURST for hierarchical categorical breakdown
        elif chart_type in ("treemap", "sunburst"):
            # allow x_col + optional color_col as hierarchy; use y_col as value if numeric else counts
            path = [x_col] + ([color_col] if color_col else [])
            if y_col and y_col in numeric_cols:
                agg = df.groupby(path)[y_col].sum().reset_index()
                if chart_type == "treemap":
                    fig = px.treemap(agg, path=path, values=y_col)
                else:
                    fig = px.sunburst(agg, path=path, values=y_col)
            else:
                agg = df.groupby(path).size().reset_index(name="count")
                if chart_type == "treemap":
                    fig = px.treemap(agg, path=path, values="count")
                else:
                    fig = px.sunburst(agg, path=path, values="count")

        else:
            # fallback bar
            agg = agg_frame([x_col], y_col if (y_col in numeric_cols) else None)
            fig = px.bar(agg, x=x_col, y=(y_col if y_col in numeric_cols else "count"))

        fig.update_layout(margin=dict(l=20, r=20, t=40, b=20))
        return fig

    except Exception as e:
        print("Chart generation error:", e)
        return None

# ---------------- MCP query handler ----------------
async def handle_task_async(user_task: str):
    """Return pandas DataFrame for the user_task (uses cache when possible)."""
    exit_stack = AsyncExitStack()
    server_params = StdioServerParameters(command="python", args=[SERVER_SCRIPT])

    # start stdio transport and session (keeps pattern you asked for)
    stdio_transport = await exit_stack.enter_async_context(stdio_client(server_params))
    stdio, write = stdio_transport
    session = await exit_stack.enter_async_context(ClientSession(stdio, write))
    await session.initialize()
    tools = (await session.list_tools()).tools

    try:
        # Check cache first
        cached = await check_cache_for_task(user_task)
        if cached:
            cols, rows = cached.get("columns", []), cached.get("rows", [])
            df = pd.DataFrame(rows, columns=cols)
            summary = await summarize_result(cols, rows)
            conversation_history.append({"role": "assistant", "content": f"Result summary: {summary}"})
            return df

        # Generate SQL
        sql = await natural_language_to_sql(user_task, tools, data_dictionary)
        if sql == "NO_QUERY":
            return pd.DataFrame()

        # Execute via MCP tool
        result = await session.call_tool("execute_query", {"query": sql})

        structured = None
        if result.content and hasattr(result.content[0], "text"):
            try:
                structured = json.loads(result.content[0].text)
            except Exception as e:
                print("⚠️ parse error:", e)
                return pd.DataFrame()

        if structured and "rows" in structured:
            cols, rows = structured.get("columns", []), structured["rows"]
            cache_key = get_cache_key(user_task)
            query_cache[cache_key] = {"columns": cols, "rows": rows, "task": user_task}
            summary = await summarize_result(cols, rows)
            conversation_history.append({"role": "assistant", "content": f"Result summary: {summary}"})
            df = pd.DataFrame(rows, columns=cols)
            return df

        return pd.DataFrame()

    finally:
        await exit_stack.aclose()

# ---------------- Gradio handlers (synchronous wrappers) ----------------
def run_query_and_populate(task: str):
    """
    1) runs the query (async)
    2) returns: dataframe to display and updated dropdown options for x/y/color
    """
    df = asyncio.run(handle_task_async(task))

    if df is None or df.empty:
        # Return empty df and reset dropdowns
        return pd.DataFrame(), gr.update(choices=[], value=None), gr.update(choices=[], value=None), gr.update(choices=[None], value=None)

    cols = df.columns.tolist()
    # X-axis: all columns (categorical or numeric)
    x_choices = cols.copy()
    # Y-axis: offer both numeric and all columns (user may want counts); default to numeric if available
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    y_choices = numeric_cols.copy() if numeric_cols else cols.copy()
    # Color choices: all columns + None
    color_choices = [None] + cols

    # sensible defaults
    x_default = cols[0]
    y_default = numeric_cols[0] if numeric_cols else cols[0]
    color_default = None

    # return dataframe and update dropdowns
    return (
        df,
        gr.update(choices=x_choices, value=x_default),
        gr.update(choices=y_choices, value=y_default),
        gr.update(choices=color_choices, value=color_default)
    )

def draw_chart_from_df(df, chart_type, x_col, y_col, color_col):
    """
    Draw chart from the displayed dataframe (gradio passes df as pandas DataFrame).
    """
    if df is None or (isinstance(df, list) and len(df) == 0):
        return None

    # Gradio may pass DataFrame as a list-of-lists or pandas DataFrame; normalize:
    if not isinstance(df, pd.DataFrame):
        try:
            df = pd.DataFrame(df[1:], columns=df[0])  # if passed as list with header
        except Exception:
            df = pd.DataFrame(df)

    fig = generate_chart(df, chart_type, x_col, y_col, color_col)
    return fig

with gr.Blocks() as demo:
    gr.Markdown("## AI Data Agent — Query + Interactive Visualization")

    with gr.Row():
        # Left panel
        with gr.Column(scale=1):
            task_in = gr.Textbox(
                lines=2, 
                placeholder="Describe what you want (e.g., top banks by quarterly profit in 2024)...", 
                label="Task / Natural Language Query"
            )
            run_btn = gr.Button("Run Query")

            chart_type = gr.Dropdown(
                choices=["bar", "grouped_bar", "stacked_bar", "line", "scatter", "scatter_3d", 
                         "box", "violin", "pie", "heatmap", "treemap", "sunburst"], 
                value="bar", 
                label="Chart Type"
            )

            with gr.Row():
                x_dropdown = gr.Dropdown(choices=[], value=None, label="X-axis")
                y_dropdown = gr.Dropdown(choices=[], value=None, label="Y-axis (numeric preferred)")
                color_dropdown = gr.Dropdown(choices=[None], value=None, label="Color / Group (Optional)")

            draw_btn = gr.Button("Draw Chart")

        # Right panel
        with gr.Column(scale=2):
            plot_out = gr.Plot(label="Chart")

    # Bottom full-width DataFrame
    with gr.Row():
        df_display = gr.Dataframe(value=pd.DataFrame(), label="Query Result (run query first)")


    # run_btn populates df and dropdowns
    run_btn.click(
        fn=run_query_and_populate,
        inputs=[task_in],
        outputs=[df_display, x_dropdown, y_dropdown, color_dropdown]
    )

    # draw_btn builds chart from selected columns and df
    draw_btn.click(
        fn=draw_chart_from_df,
        inputs=[df_display, chart_type, x_dropdown, y_dropdown, color_dropdown],
        outputs=[plot_out]
    )

demo.launch()
