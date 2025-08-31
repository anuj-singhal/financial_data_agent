"""
LLM service module for natural language processing.
"""

import json
import logging
from typing import List, Dict, Any, Optional
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

class LLMService:
    """
    Service for interacting with LLM for SQL generation and result summarization.
    """
    
    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        """
        Initialize LLM service.
        
        Args:
            api_key: OpenAI API key
            model: Model name to use
        """
        self.client = AsyncOpenAI(api_key=api_key)
        self.model = model
        self.conversation_history: List[Dict[str, str]] = []
        logger.info(f"LLMService initialized with model: {model}")
    
    async def summarize_result(self, columns: List[str], rows: List[List[Any]], 
                              max_rows: int = 6) -> str:
        """
        Generate a concise summary of query results.
        
        Args:
            columns: Column names
            rows: Data rows
            max_rows: Maximum rows to include in summary
            
        Returns:
            Summary string
        """
        try:
            sample = rows[:max_rows]
            prompt = f"""
                You are an assistant that summarizes database query results.
                Columns: {columns}
                Sample Rows: {sample}
                Generate a very concise 1-2 sentence summary suitable for conversation history.
                """
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You summarize DB results concisely."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=100
            )
            summary = response.choices[0].message.content.strip()
            self.conversation_history.append({"role": "assistant", "content": f"Result summary: {summary}"})
            logger.debug(f"Generated summary: {summary[:50]}...")
            return summary
        except Exception as e:
            logger.error(f"Error summarizing result: {e}")
            return "Summary generation failed"
    
    async def check_cache_relevance(self, task: str, cache_summaries: Dict[str, Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        Check if cached results can answer the user's task.
        
        Args:
            task: User task
            cache_summaries: Summary of cached queries
            
        Returns:
            Cached data if relevant, None otherwise
        """
        if not cache_summaries:
            return None
        
        try:
            summaries_str = "\n".join([
                f"{k}: columns({v['columns']}), sample({v['sample_rows']})"
                for k, v in cache_summaries.items()
            ])
            
            prompt = f"""
                We have cached query results (key : summary):
                {summaries_str}

                Conversation history: {self.conversation_history}
                User task: {task}

                Rules:
                - If the task can be answered from cache, return JSON:
                {{ "columns": [...], "rows": [...] }}
                - If it cannot be answered from cache, return exactly: NO_CACHE
                - Do not fabricate any data; only reuse cache.
                """
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a cache assistant."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=1000
            )
            
            reply = response.choices[0].message.content.strip()
            if reply == "NO_CACHE":
                logger.debug("Cache cannot answer task")
                return None
            
            data = json.loads(reply)
            logger.debug("Cache can answer task")
            return data
        except Exception as e:
            logger.error(f"Error checking cache relevance: {e}")
            return None
    
    async def generate_sql(self, task: str, tools: List[Any], data_dict: Dict[str, Any]) -> str:
        """
        Generate SQL query from natural language task.
        
        Args:
            task: User task in natural language
            tools: Available MCP tools
            data_dict: Data dictionary describing tables and columns
            
        Returns:
            SQL query string or "NO_QUERY" if invalid
        """
        try:
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
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "Oracle SQL expert assistant."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=500
            )
            
            sql = response.choices[0].message.content.strip()
            sql = sql.replace(";", "").replace("```", "")
            logger.debug(f"Generated SQL: {sql[:100]}...")
            return sql
        except Exception as e:
            logger.error(f"Error generating SQL: {e}")
            return "NO_QUERY"