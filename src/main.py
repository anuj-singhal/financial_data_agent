"""
Main entry point for the Oracle MCP Client application.
"""

import asyncio
import logging
import sys
from pathlib import Path

from config import ServerConfig, LLMConfig, CacheConfig, UIConfig
from utils import setup_logging
from data_agent import DataAgent
from ui_manager import UIManager

def main():
    """
    Main function to run the application.
    """
    # Setup logging
    setup_logging("INFO")
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("Starting Oracle MCP Client Application")
        
        # Initialize configurations
        server_config = ServerConfig()
        llm_config = LLMConfig()
        cache_config = CacheConfig()
        ui_config = UIConfig()
        
        # Validate API key
        if not llm_config.api_key:
            logger.error("OpenAI API key not found. Please set OPENAI_API_KEY environment variable.")
            sys.exit(1)
        
        # Validate server script path
        if not Path(server_config.script_path).exists():
            logger.error(f"Server script not found at: {server_config.script_path}")
            sys.exit(1)
        
        # Initialize data agent
        data_agent = DataAgent(
            server_config=server_config,
            llm_config=llm_config,
            cache_config=cache_config,
            ui_config=ui_config
        )
        
        # Initialize UI manager
        ui_manager = UIManager(data_agent, ui_config)
        
        # Create and launch interface
        interface = ui_manager.create_interface()
        
        logger.info("Launching Gradio interface...")
        interface.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False,
            show_error=True,
            quiet=False
        )
        
    except KeyboardInterrupt:
        logger.info("Application stopped by user")
    except Exception as e:
        logger.error(f"Application error: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()