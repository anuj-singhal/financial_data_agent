"""
Chart generation module using Plotly.
"""

import logging
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Optional, List

logger = logging.getLogger(__name__)

class ChartGenerator:
    """
    Generates various types of charts from pandas DataFrames.
    """
    
    def __init__(self):
        """Initialize chart generator."""
        self.supported_types = [
            "bar", "grouped_bar", "stacked_bar", "line", "scatter", "scatter_3d",
            "box", "violin", "pie", "heatmap", "treemap", "sunburst"
        ]
        logger.info("ChartGenerator initialized")
    
    def generate(self, df: pd.DataFrame, chart_type: str, x_col: str, 
                y_col: Optional[str] = None, color_col: Optional[str] = None) -> Optional[go.Figure]:
        """
        Generate a Plotly chart from DataFrame.
        
        Args:
            df: Input DataFrame
            chart_type: Type of chart to generate
            x_col: Column for x-axis
            y_col: Column for y-axis (optional)
            color_col: Column for color grouping (optional)
            
        Returns:
            Plotly figure or None if generation fails
        """
        if df is None or df.empty:
            logger.warning("Cannot generate chart from empty DataFrame")
            return None
        
        # Validate chart type
        if chart_type not in self.supported_types:
            logger.warning(f"Unsupported chart type: {chart_type}")
            chart_type = "bar"  # Fallback to bar chart
        
        # Normalize None strings
        color_col = None if color_col in (None, "None", "") else color_col
        
        try:
            # Identify column types
            numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
            all_cols = df.columns.tolist()
            cat_cols = [c for c in all_cols if c not in numeric_cols]
            
            # Validate columns
            if x_col not in all_cols:
                logger.warning(f"X column '{x_col}' not found, using first column")
                x_col = all_cols[0] if all_cols else None
            
            # Generate chart based on type
            fig = self._generate_chart_by_type(
                df, chart_type, x_col, y_col, color_col, 
                numeric_cols, cat_cols
            )
            
            if fig:
                fig.update_layout(
                    margin=dict(l=20, r=20, t=40, b=20),
                    template="plotly_white"
                )
                logger.debug(f"Generated {chart_type} chart successfully")
            
            return fig
            
        except Exception as e:
            logger.error(f"Chart generation error: {e}")
            return None
    
    def _generate_chart_by_type(self, df: pd.DataFrame, chart_type: str, 
                                x_col: str, y_col: Optional[str], color_col: Optional[str],
                                numeric_cols: List[str], cat_cols: List[str]) -> Optional[go.Figure]:
        """
        Internal method to generate specific chart types.
        
        Args:
            df: Input DataFrame
            chart_type: Type of chart
            x_col: X-axis column
            y_col: Y-axis column
            color_col: Color grouping column
            numeric_cols: List of numeric columns
            cat_cols: List of categorical columns
            
        Returns:
            Plotly figure or None
        """
        # Helper function for aggregation
        def agg_frame(group_cols, value_col=None, agg_func="sum"):
            if value_col and value_col in numeric_cols:
                return df.groupby(group_cols)[value_col].agg(agg_func).reset_index()
            else:
                return df.groupby(group_cols).size().reset_index(name="count")
        
        # Bar charts
        if chart_type in ("bar", "stacked_bar", "grouped_bar"):
            if y_col and y_col in numeric_cols:
                if color_col:
                    agg = agg_frame([x_col, color_col], y_col, "sum")
                    fig = px.bar(agg, x=x_col, y=y_col, color=color_col)
                else:
                    agg = agg_frame([x_col], y_col, "sum")
                    fig = px.bar(agg, x=x_col, y=y_col)
            else:
                if color_col:
                    agg = agg_frame([x_col, color_col])
                    fig = px.bar(agg, x=x_col, y="count", color=color_col)
                else:
                    agg = agg_frame([x_col])
                    fig = px.bar(agg, x=x_col, y="count")
            
            barmode = {"grouped_bar": "group", "stacked_bar": "stack"}.get(chart_type, "group")
            fig.update_layout(barmode=barmode)
            return fig
        
        # Line chart
        elif chart_type == "line":
            if y_col and y_col in numeric_cols:
                if color_col:
                    agg = agg_frame([x_col, color_col], y_col, "mean")
                    return px.line(agg, x=x_col, y=y_col, color=color_col)
                else:
                    agg = agg_frame([x_col], y_col, "mean")
                    return px.line(agg, x=x_col, y=y_col)
            return None
        
        # Scatter plot
        elif chart_type == "scatter":
            if y_col not in numeric_cols:
                return None
            if x_col in numeric_cols:
                return px.scatter(df, x=x_col, y=y_col, color=color_col)
            else:
                return px.strip(df, x=x_col, y=y_col, color=color_col)
        
        # 3D Scatter
        elif chart_type == "scatter_3d":
            nums = [c for c in df.columns if c in numeric_cols]
            if len(nums) >= 3:
                return px.scatter_3d(df, x=nums[0], y=nums[1], z=nums[2], color=color_col)
            return None
        
        # Box/Violin plots
        elif chart_type in ("box", "violin"):
            if y_col in numeric_cols:
                if chart_type == "box":
                    return px.box(df, x=x_col, y=y_col, color=color_col)
                else:
                    return px.violin(df, x=x_col, y=y_col, color=color_col, box=True)
            return None
        
        # Pie chart
        elif chart_type == "pie":
            if y_col and y_col in numeric_cols:
                agg = agg_frame([x_col], y_col, "sum")
                return px.pie(agg, names=x_col, values=y_col)
            else:
                agg = agg_frame([x_col])
                return px.pie(agg, names=x_col, values="count")
        
        # Heatmap
        elif chart_type == "heatmap":
            if not color_col:
                return None
            numeric = df.select_dtypes(include=["number"]).columns
            agg_col = numeric[0] if len(numeric) > 0 else None
            if agg_col:
                pivot = df.pivot_table(index=color_col, columns=x_col, values=agg_col, 
                                      aggfunc="sum", fill_value=0)
                return px.imshow(pivot, labels=dict(x=x_col, y=color_col, 
                                                   color=f"sum({agg_col})"))
            else:
                pivot = df.groupby([color_col, x_col]).size().unstack(fill_value=0)
                return px.imshow(pivot)
        
        # Treemap/Sunburst
        elif chart_type in ("treemap", "sunburst"):
            path = [x_col] + ([color_col] if color_col else [])
            if y_col and y_col in numeric_cols:
                agg = df.groupby(path)[y_col].sum().reset_index()
                if chart_type == "treemap":
                    return px.treemap(agg, path=path, values=y_col)
                else:
                    return px.sunburst(agg, path=path, values=y_col)
            else:
                agg = df.groupby(path).size().reset_index(name="count")
                if chart_type == "treemap":
                    return px.treemap(agg, path=path, values="count")
                else:
                    return px.sunburst(agg, path=path, values="count")
        
        # Fallback to bar chart
        else:
            agg = agg_frame([x_col], y_col if (y_col in numeric_cols) else None)
            return px.bar(agg, x=x_col, y=(y_col if y_col in numeric_cols else "count"))
