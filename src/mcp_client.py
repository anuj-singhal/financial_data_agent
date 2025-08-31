"""
MCP (Model Context Protocol) client for Oracle database interactions.
"""

import json
import logging
import asyncio
from typing import Dict, Any, Optional, List
from contextlib import AsyncExitStack
import pandas as pd

from mcp.client.stdio import stdio_client, StdioServerParameters
from mcp.client.session import ClientSession

logger = logging.getLogger(__name__)

class MCPClient:
    """
    Client for interacting with MCP server for database operations.
    """
    
    def __init__(self, server_script: str, command: str = "python"):
        """
        Initialize MCP client.
        
        Args:
            server_script: Path to server script
            command: Command to run server (default: python)
        """
        self.server_script = server_script
        self.command = command
        self.tools = None
        logger.info(f"MCPClient initialized with server: {server_script}")
    
    async def execute_query(self, sql: str) -> Optional[pd.DataFrame]:
        """
        Execute SQL query via MCP server.
        
        Args:
            sql: SQL query string
            
        Returns:
            DataFrame with results or None if failed
        """
        exit_stack = AsyncExitStack()
        server_params = StdioServerParameters(
            command=self.command, 
            args=[self.server_script]
        )
        
        try:
            # Start stdio transport and session
            stdio_transport = await exit_stack.enter_async_context(
                stdio_client(server_params)
            )
            stdio, write = stdio_transport
            session = await exit_stack.enter_async_context(
                ClientSession(stdio, write)
            )
            
            await session.initialize()
            self.tools = (await session.list_tools()).tools
            
            logger.debug(f"Executing SQL: {sql[:100]}...")
            
            # Execute query via MCP tool
            result = await session.call_tool("execute_query", {"query": sql})
            
            # Parse result
            if result.content and hasattr(result.content[0], "text"):
                try:
                    structured = json.loads(result.content[0].text)
                    if structured and "rows" in structured:
                        cols = structured.get("columns", [])
                        rows = structured["rows"]
                        df = pd.DataFrame(rows, columns=cols)
                        logger.info(f"Query executed successfully, returned {len(rows)} rows")
                        return df
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse query result: {e}")
            
            return None
            
        except Exception as e:
            logger.error(f"MCP query execution failed: {e}")
            return None
            
        finally:
            await exit_stack.aclose()
    
    async def get_tools(self) -> List[Any]:
        """
        Get available tools from MCP server.
        
        Returns:
            List of available tools
        """
        if self.tools:
            return self.tools
        
        exit_stack = AsyncExitStack()
        server_params = StdioServerParameters(
            command=self.command, 
            args=[self.server_script]
        )
        
        try:
            stdio_transport = await exit_stack.enter_async_context(
                stdio_client(server_params)
            )
            stdio, write = stdio_transport
            session = await exit_stack.enter_async_context(
                ClientSession(stdio, write)
            )
            
            await session.initialize()
            self.tools = (await session.list_tools()).tools
            logger.info(f"Retrieved {len(self.tools)} tools from MCP server")
            return self.tools
            
        except Exception as e:
            logger.error(f"Failed to get MCP tools: {e}")
            return []
            
        finally:
            await exit_stack.aclose()