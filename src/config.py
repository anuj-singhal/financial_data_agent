"""
Configuration module for Oracle MCP Client.
Contains all configuration settings, constants, and data dictionary.
"""

import os
from typing import Dict, Any
from dataclasses import dataclass, field

@dataclass
class ServerConfig:
    """Server configuration settings."""
    script_path: str = "C:\\Anuj\\AI\\MCP\\mcp-servers\\oracle_server\\server2.py"
    command: str = "python"
    
@dataclass
class LLMConfig:
    """LLM configuration settings."""
    api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY"))
    model: str = "gpt-4o-mini"
    max_tokens: int = 500
    temperature: float = 0.7

@dataclass
class CacheConfig:
    """Cache configuration settings."""
    max_entries: int = 100
    ttl_seconds: int = 3600  # 1 hour
    max_sample_rows: int = 6

@dataclass
class UIConfig:
    """UI configuration settings."""
    chart_types: list = field(default_factory=lambda: [
        "bar", "grouped_bar", "stacked_bar", "line", "scatter", "scatter_3d",
        "box", "violin", "pie", "heatmap", "treemap", "sunburst"
    ])
    default_chart_type: str = "bar"
    theme: str = "default"

# Data Dictionary
DATA_DICTIONARY: Dict[str, Dict[str, Any]] = {
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