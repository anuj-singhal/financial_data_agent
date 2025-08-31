"""
Gradio UI management module with ChatInterface.
"""

import asyncio
import logging
from typing import Tuple, Any, Optional, List
import pandas as pd
import gradio as gr
from gradio import themes
import inspect

from data_agent import DataAgent
from config import UIConfig

logger = logging.getLogger(__name__)

class UIManager:
    """
    Manages the Gradio user interface.
    """
    
    def __init__(self, data_agent: DataAgent, ui_config: UIConfig = None):
        """
        Initialize UI manager.
        
        Args:
            data_agent: Data agent instance
            ui_config: UI configuration
        """
        self.data_agent = data_agent
        self.ui_config = ui_config or UIConfig()
        self.current_df = None
        self.chat_history = []
        logger.info("UIManager initialized")
    
    def process_chat_query(self, message: str, history: List) -> str:
        """
        Process chat messages and execute queries.
        
        Args:
            message: User message
            history: Chat history
            
        Returns:
            Response message
        """
        try:
            if not message or message.strip() == "":
                return "Please enter a valid query."
            
            # Run async query
            df = asyncio.run(self.data_agent.process_query(message))
            
            if df is None or df.empty:
                return "Query returned no results. Please try a different query."
            
            # Store current dataframe
            self.current_df = df
            
            # Format response
            response = f"✅ Query executed successfully!\n\n"
            response += f"📊 **Results**: {len(df)} rows returned\n"
            response += f"📋 **Columns**: {', '.join(df.columns.tolist())}\n\n"
            
            # Add sample of data
            if len(df) > 5:
                response += f"**Preview (first 5 rows):**\n```\n{df.head().to_string()}\n```\n\n"
                response += "💡 *Click 'Load Data for Visualization' to see full data and create charts.*"
            else:
                response += f"**Data:**\n```\n{df.to_string()}\n```"
            
            return response
            
        except Exception as e:
            logger.error(f"Error in process_chat_query: {e}")
            return f"❌ Query failed: {str(e)}\n\nPlease check your query and try again."
    
    def get_current_data(self) -> Tuple[pd.DataFrame, gr.Dropdown, gr.Dropdown, gr.Dropdown]:
        """
        Get current DataFrame and update dropdowns.
        
        Returns:
            Tuple of (dataframe, x_dropdown, y_dropdown, color_dropdown)
        """
        if self.current_df is None or self.current_df.empty:
            gr.Warning("No data available. Please run a query first.")
            return self._empty_results()
        
        # Get column options
        x_choices, y_choices, color_choices = self.data_agent.get_column_options(self.current_df)
        x_default, y_default, color_default = self.data_agent.get_default_selections(self.current_df)
        
        gr.Info(f"Data loaded: {len(self.current_df)} rows")
        
        return (
            self.current_df,
            gr.update(choices=x_choices, value=x_default),
            gr.update(choices=y_choices, value=y_default),
            gr.update(choices=color_choices, value=color_default)
        )
    
    def draw_chart_from_df(self, df: Any, chart_type: str, x_col: str, 
                           y_col: Optional[str], color_col: Optional[str]):
        """
        Generate chart from DataFrame.
        
        Args:
            df: Input DataFrame (from Gradio)
            chart_type: Type of chart
            x_col: X-axis column
            y_col: Y-axis column
            color_col: Color grouping column
            
        Returns:
            Plotly figure or None
        """
        try:
            # Handle Gradio DataFrame format
            if df is None or (isinstance(df, list) and len(df) == 0):
                gr.Warning("No data available for charting")
                return None
            
            # Convert to pandas DataFrame if needed
            if not isinstance(df, pd.DataFrame):
                try:
                    df = pd.DataFrame(df[1:], columns=df[0]) if isinstance(df, list) else pd.DataFrame(df)
                except Exception:
                    df = pd.DataFrame(df)
            
            if df.empty:
                gr.Warning("DataFrame is empty")
                return None
            
            # Generate chart
            fig = self.data_agent.generate_chart(df, chart_type, x_col, y_col, color_col)
            
            if fig:
                gr.Info(f"{chart_type.replace('_', ' ').title()} chart generated successfully")
            else:
                gr.Warning("Could not generate chart with selected options")
            
            return fig
            
        except Exception as e:
            logger.error(f"Error in draw_chart_from_df: {e}")
            gr.Error(f"Chart generation failed: {str(e)}")
            return None
    
    def _empty_results(self) -> Tuple[pd.DataFrame, gr.Dropdown, gr.Dropdown, gr.Dropdown]:
        """
        Return empty results for UI update.
        
        Returns:
            Tuple of empty components
        """
        return (
            pd.DataFrame(),
            gr.update(choices=[], value=None),
            gr.update(choices=[], value=None),
            gr.update(choices=[None], value=None)
        )
    
    def create_interface(self) -> gr.Blocks:
        """
        Create the Gradio interface.
        
        Returns:
            Gradio Blocks interface
        """
        # Check Gradio Dataframe parameters for version compatibility
        dataframe_params = {
            "value": pd.DataFrame(),
            "label": "Query Results",
            "interactive": False,
            "wrap": True
        }
        
        # Check available parameters for Dataframe component
        try:
            df_signature = inspect.signature(gr.Dataframe.__init__)
            available_params = list(df_signature.parameters.keys())
            
            # Try to add height/max_rows if available
            if "height" in available_params:
                dataframe_params["height"] = 400
            elif "max_rows" in available_params:
                dataframe_params["max_rows"] = 20
            elif "row_count" in available_params:
                dataframe_params["row_count"] = (10, "dynamic")
        except Exception as e:
            logger.debug(f"Could not inspect Dataframe parameters: {e}")
        
        with gr.Blocks(
            title="Oracle Data Agent",
            theme=gr.themes.Soft(),
            css="""
            .gradio-container {
                font-family: 'Inter', sans-serif;
            }
            .gr-button {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                border: none;
                color: white;
            }
            .gr-button:hover {
                transform: translateY(-2px);
                box-shadow: 0 5px 15px rgba(0,0,0,0.2);
            }
            """
        ) as demo:
            # Header
            gr.Markdown(
                """
                # 🚀 Oracle Data Agent
                ### AI-Powered Data Query & Interactive Visualization Platform
                
                Transform natural language queries into SQL, execute them, and create stunning visualizations instantly.
                """
            )
            
            with gr.Tabs():
                # Query Tab
                with gr.TabItem("📊 Query & Visualize"):
                    with gr.Row():
                        # Left Panel - Chat Query
                        with gr.Column(scale=1):
                            gr.Markdown("### 🔍 Query Your Data")
                            
                            # Chat Interface replacing the textbox
                            with gr.Group():
                                chat_interface = gr.ChatInterface(
                                    fn=self.process_chat_query,
                                    examples=[
                                        "Show top 5 banks by profit in Q4 2024",
                                        "Compare quarterly revenue across all banks",
                                        "What's the average NIM by quarter?",
                                        "Show loan growth trends for the largest banks"
                                    ]
                                )
                            
                            with gr.Row():
                                load_data_button = gr.Button(
                                    "📊 Load Data for Visualization",
                                    variant="primary",
                                    size="lg"
                                )
                            
                        # Right Panel - Visualization
                        with gr.Column(scale=2):
                            gr.Markdown("### 📈 Chart Configuration")

                            with gr.Row():
                                chart_type = gr.Dropdown(
                                    choices=self.ui_config.chart_types,
                                    value=self.ui_config.default_chart_type,
                                    label="Chart Type",
                                    info="Select visualization type"
                                )
                                x_dropdown = gr.Dropdown(
                                    choices=[],
                                    value=None,
                                    label="X-Axis",
                                    info="Select column for X-axis"
                                )
                                
                                y_dropdown = gr.Dropdown(
                                    choices=[],
                                    value=None,
                                    label="Y-Axis",
                                    info="Select column for Y-axis (numeric preferred)"
                                )
                                
                                color_dropdown = gr.Dropdown(
                                    choices=[None],
                                    value=None,
                                    label="Color/Group By",
                                    info="Optional: Select column for color grouping"
                                )
                            
                            draw_button = gr.Button(
                                "🎨 Generate Chart",
                                variant="primary",
                                size="lg"
                            )

                            plot_output = gr.Plot(
                                label="Visualization",
                                show_label=False
                            )
                    
                    # Bottom - Data Table
                    with gr.Row():
                        df_display = gr.Dataframe(**dataframe_params)
                
                # Settings Tab
                with gr.TabItem("⚙️ Settings"):
                    gr.Markdown("### Cache Management")
                    with gr.Row():
                        cache_info = gr.Markdown("Cache entries: 0")
                        clear_cache_btn = gr.Button("Clear Cache")
                    
                    gr.Markdown("### Export Options")
                    with gr.Row():
                        export_format = gr.Dropdown(
                            choices=["CSV", "Excel", "JSON"],
                            value="CSV",
                            label="Export Format"
                        )
                        export_btn = gr.Button("Export Data")
                
                # Help Tab
                with gr.TabItem("❓ Help"):
                    gr.Markdown(
                        """
                        ### How to Use
                        
                        1. **Chat Query**: Type your data request in natural language in the chat
                        2. **Execute**: Send your message to run the query
                        3. **Load Data**: Click "Load Data for Visualization" to populate the data table
                        4. **Visualize**: Select chart type and axes, then click "Generate Chart"
                        
                        ### Example Queries
                        
                        - "Show top 10 banks by profit in 2024"
                        - "Compare quarterly revenue across all banks"
                        - "What's the average NIM by quarter?"
                        - "Show loan growth trends for the largest banks"
                        - "Which bank has the highest ROE?"
                        - "List all banks with assets over $1 trillion"
                        
                        ### Chart Types
                        
                        - **Bar/Grouped/Stacked**: Compare categories
                        - **Line**: Show trends over time
                        - **Scatter**: Explore relationships
                        - **Pie**: Show proportions
                        - **Heatmap**: Display matrix data
                        - **Box/Violin**: Show distributions
                        
                        ### Tips
                        - Use the example queries as templates
                        - Be specific in your queries for better results
                        - After running a query, load the data before creating visualizations
                        """
                    )
            
            # Status Bar
            with gr.Row():
                status_text = gr.Markdown("Ready", elem_id="status-bar")
            
            # Event Handlers
            load_data_button.click(
                fn=self.get_current_data,
                inputs=[],
                outputs=[df_display, x_dropdown, y_dropdown, color_dropdown]
            )
            
            draw_button.click(
                fn=self.draw_chart_from_df,
                inputs=[df_display, chart_type, x_dropdown, y_dropdown, color_dropdown],
                outputs=[plot_output]
            )
            
            def update_cache_info():
                count = len(self.data_agent.cache_manager._cache)
                return f"Cache entries: {count}"
            
            clear_cache_btn.click(
                fn=lambda: (self.data_agent.cache_manager.clear(), "Cache cleared!"),
                inputs=[],
                outputs=[cache_info]
            ).then(
                fn=update_cache_info,
                inputs=[],
                outputs=[cache_info]
            )
        
        return demo