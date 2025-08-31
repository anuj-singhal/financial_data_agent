"""
Main data agent orchestrating queries, caching, and chart generation.
"""

import asyncio
import logging
from typing import Optional, Tuple, List, Dict, Any
import pandas as pd

from config import ServerConfig, LLMConfig, CacheConfig, UIConfig, DATA_DICTIONARY
from utils import get_cache_key, timer_decorator
from cache_manager import CacheManager
from llm_service import LLMService
from chart_generator import ChartGenerator
from mcp_client import MCPClient

logger = logging.getLogger(__name__)

class DataAgent:
    """
    Main orchestrator for data queries, caching, and visualization.
    """
    
    def __init__(self, server_config: ServerConfig = None, 
                 llm_config: LLMConfig = None,
                 cache_config: CacheConfig = None,
                 ui_config: UIConfig = None):
        """
        Initialize data agent with configurations.
        
        Args:
            server_config: Server configuration
            llm_config: LLM configuration
            cache_config: Cache configuration
            ui_config: UI configuration
        """
        self.server_config = server_config or ServerConfig()
        self.llm_config = llm_config or LLMConfig()
        self.cache_config = cache_config or CacheConfig()
        self.ui_config = ui_config or UIConfig()
        
        # Initialize components
        self.cache_manager = CacheManager(
            max_entries=self.cache_config.max_entries,
            ttl_seconds=self.cache_config.ttl_seconds
        )
        self.llm_service = LLMService(
            api_key=self.llm_config.api_key,
            model=self.llm_config.model
        )
        self.chart_generator = ChartGenerator()
        self.mcp_client = MCPClient(
            server_script=self.server_config.script_path,
            command=self.server_config.command
        )
        
        logger.info("DataAgent initialized successfully")
    
    @timer_decorator
    async def process_query(self, task: str) -> pd.DataFrame:
        """
        Process a natural language query and return results.
        
        Args:
            task: Natural language query task
            
        Returns:
            DataFrame with query results
        """
        try:
            # Check cache first
            cache_key = get_cache_key(task)
            cached_entry = self.cache_manager.get(cache_key)
            
            if cached_entry:
                logger.info(f"Using cached result for task: {task[:50]}...")
                df = pd.DataFrame(cached_entry.rows, columns=cached_entry.columns)
                return df
            
            # Check if any cached data can answer the query
            cache_summaries = self.cache_manager.get_summary()
            cached_data = await self.llm_service.check_cache_relevance(task, cache_summaries)
            
            if cached_data:
                logger.info("Found relevant cached data for task")
                cols = cached_data.get("columns", [])
                rows = cached_data.get("rows", [])
                df = pd.DataFrame(rows, columns=cols)
                
                # Update conversation history
                summary = await self.llm_service.summarize_result(cols, rows)
                return df
            
            # Generate and execute new SQL query
            tools = await self.mcp_client.get_tools()
            sql = await self.llm_service.generate_sql(task, tools, DATA_DICTIONARY)

            logger.info(f"Query created by LLM for task: {task}")
            logger.info(f"LLM SQL Query: {sql}")
            
            if sql == "NO_QUERY":
                logger.warning("Could not generate valid SQL for task")
                return pd.DataFrame()
            
            # Execute query
            df = await self.mcp_client.execute_query(sql)
            
            if df is not None and not df.empty:
                # Cache the result
                cols = df.columns.tolist()
                rows = df.values.tolist()
                self.cache_manager.set(cache_key, cols, rows, task)
                
                # Update conversation history
                summary = await self.llm_service.summarize_result(cols, rows)
                logger.info(f"Query executed successfully for task: {task[:50]}...")
            else:
                logger.warning("Query returned empty results")
                df = pd.DataFrame()
            
            return df
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return pd.DataFrame()
    
    def generate_chart(self, df: pd.DataFrame, chart_type: str, 
                      x_col: str, y_col: Optional[str] = None, 
                      color_col: Optional[str] = None):
        """
        Generate a chart from DataFrame.
        
        Args:
            df: Input DataFrame
            chart_type: Type of chart
            x_col: X-axis column
            y_col: Y-axis column (optional)
            color_col: Color grouping column (optional)
            
        Returns:
            Plotly figure or None
        """
        return self.chart_generator.generate(df, chart_type, x_col, y_col, color_col)
    
    def get_column_options(self, df: pd.DataFrame) -> Tuple[List[str], List[str], List[Optional[str]]]:
        """
        Get column options for chart axes based on DataFrame.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Tuple of (x_choices, y_choices, color_choices)
        """
        if df is None or df.empty:
            return [], [], [None]
        
        cols = df.columns.tolist()
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        
        # X-axis: all columns
        x_choices = cols.copy()
        
        # Y-axis: prefer numeric columns
        y_choices = numeric_cols.copy() if numeric_cols else cols.copy()
        
        # Color: all columns plus None option
        color_choices = [None] + cols
        
        return x_choices, y_choices, color_choices
    
    def get_default_selections(self, df: pd.DataFrame) -> Tuple[str, str, Optional[str]]:
        """
        Get default column selections for chart.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Tuple of (x_default, y_default, color_default)
        """
        if df is None or df.empty:
            return "", "", None
        
        cols = df.columns.tolist()
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        
        x_default = cols[0] if cols else ""
        y_default = numeric_cols[0] if numeric_cols else (cols[0] if cols else "")
        color_default = None
        
        return x_default, y_default, color_default