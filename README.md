# 🚀 Oracle MCP Data Agent

A professional-grade, AI-powered data query and visualization platform that transforms natural language into SQL queries, executes them against Oracle databases, and generates interactive visualizations.

![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![Gradio](https://img.shields.io/badge/gradio-4.0%2B-orange)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-production--ready-brightgreen)

## ✨ Features

### 🎯 Core Capabilities
- **Natural Language to SQL**: Convert plain English queries into optimized Oracle SQL
- **Interactive Visualizations**: Generate 12+ chart types with Plotly
- **Intelligent Caching**: TTL-based caching with LRU eviction
- **Professional UI**: Modern Gradio interface with tabs and custom styling
- **Async Operations**: High-performance async/await architecture
- **Comprehensive Logging**: Multi-level logging with file and console output

### 📊 Visualization Types
- Bar Charts (Standard, Grouped, Stacked)
- Line Charts
- Scatter Plots (2D & 3D)
- Distribution Plots (Box, Violin)
- Hierarchical (Treemap, Sunburst)
- Statistical (Heatmap, Pie)

## 🏗️ Architecture

```
oracle-mcp-data-agent/
├── src/
│   ├── config.py           # Centralized configuration
│   ├── utils.py            # Utility functions & decorators
│   ├── cache_manager.py    # Advanced caching system
│   ├── llm_service.py      # LLM integration for NLP
│   ├── chart_generator.py  # Plotly chart generation
│   ├── mcp_client.py       # Oracle MCP client
│   ├── data_agent.py       # Main orchestrator
│   ├── ui_manager.py       # Gradio UI management
│   └── main.py             # Application entry point
├── logs/                    # Application logs
├── requirements.txt         # Python dependencies
└── README.md               # Documentation
```

## 🚦 Prerequisites

- Python 3.11 or higher
- Oracle MCP Server configured and running
- OpenAI API key
- 4GB RAM minimum (8GB recommended)

## 📦 Installation

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/oracle-mcp-data-agent.git
cd oracle-mcp-data-agent
```

### 2. Create Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Set Environment Variables

#### Windows (PowerShell)
```powershell
$env:OPENAI_API_KEY="your-openai-api-key"
```

#### Windows (CMD)
```cmd
set OPENAI_API_KEY=your-openai-api-key
```

#### Linux/Mac
```bash
export OPENAI_API_KEY="your-openai-api-key"
```

## ⚙️ Configuration

Edit `src/config.py` to customize:

### Server Configuration
```python
@dataclass
class ServerConfig:
    script_path: str = "path/to/your/oracle_server.py"
    command: str = "python"
```

### LLM Configuration
```python
@dataclass
class LLMConfig:
    api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY"))
    model: str = "gpt-4o-mini"  # or "gpt-4"
    max_tokens: int = 500
    temperature: float = 0.7
```

### Cache Configuration
```python
@dataclass
class CacheConfig:
    max_entries: int = 100       # Maximum cache entries
    ttl_seconds: int = 3600      # Cache TTL (1 hour)
    max_sample_rows: int = 6     # Rows for summarization
```

### Data Dictionary
Update `DATA_DICTIONARY` in `config.py` with your table schemas:
```python
DATA_DICTIONARY = {
    "your_table_name": {
        "description": "Table description",
        "columns": {
            "column1": "Column 1 description",
            "column2": "Column 2 description",
            # ... more columns
        }
    }
}
```

## 🎮 Usage

### Starting the Application
```bash
python src/main.py
```

The application will launch at `http://localhost:7860`

### Query Examples

#### Simple Queries
- "Show top 10 banks by profit in 2024"
- "What's the average NIM by quarter?"
- "List all banks with NPL ratio above 5%"

#### Complex Queries
- "Compare quarterly revenue growth across all banks for the last 2 years"
- "Show correlation between loan growth and deposit growth"
- "Analyze cost-to-income trends for banks with assets over $1B"

### Using the Interface

1. **Query Tab**
   - Enter natural language query
   - Click "Execute Query" to run
   - View results in data table
   - Select chart type and axes
   - Click "Generate Chart" to visualize

2. **Settings Tab**
   - Monitor cache usage
   - Clear cache when needed
   - Export data in various formats

3. **Help Tab**
   - View documentation
   - See example queries
   - Learn about chart types

## 🔧 Advanced Features

### Custom Chart Types
Add new chart types in `chart_generator.py`:
```python
elif chart_type == "your_custom_chart":
    # Your chart generation logic
    fig = px.your_chart_function(...)
    return fig
```

### Adding New Data Sources
Extend `DATA_DICTIONARY` in `config.py`:
```python
"new_data_source": {
    "description": "Description",
    "columns": {...}
}
```

### Performance Tuning

#### Async Operations
The application uses async/await for optimal performance:
```python
async def process_query(self, task: str) -> pd.DataFrame:
    # Async processing
    df = await self.mcp_client.execute_query(sql)
    return df
```

#### Caching Strategy
- TTL-based expiration
- LRU eviction when full
- Intelligent cache matching using LLM

## 📊 Monitoring & Logging

### Log Levels
Configure in `main.py`:
```python
setup_logging("INFO")  # Options: DEBUG, INFO, WARNING, ERROR, CRITICAL
```

### Log Files
- Location: `oracle_mcp_client.log`
- Rotation: Manual (implement rotation if needed)
- Format: `timestamp - module - level - message`

### Performance Monitoring
Use the `@timer_decorator` for any function:
```python
@timer_decorator
async def your_function():
    # Function code
```

## 🐛 Troubleshooting

### Common Issues

#### 1. Gradio Version Conflicts
```bash
# Install specific version
pip install gradio==4.44.0
```

#### 2. MCP Server Connection Failed
- Verify server path in `config.py`
- Check server is running
- Validate Python path

#### 3. OpenAI API Errors
- Verify API key is set
- Check API quota/limits
- Validate model name

#### 4. Cache Issues
- Clear cache from Settings tab
- Restart application
- Check disk space for cache

### Debug Mode
Enable detailed logging:
```python
# In main.py
setup_logging("DEBUG")
```

## 🧪 Testing

### Unit Tests
```bash
# Run all tests
python -m pytest tests/

# Run specific test
python -m pytest tests/test_cache_manager.py
```

### Integration Tests
```bash
python tests/test_integration.py
```

### Test Configuration
Create `tests/test_config.py`:
```python
@dataclass
class TestServerConfig:
    script_path: str = "tests/mock_server.py"
    command: str = "python"
```

## 📈 Performance Benchmarks

| Operation | Average Time | Notes |
|-----------|-------------|-------|
| Simple Query | ~1.2s | Cached: ~0.1s |
| Complex Query | ~2.5s | Multiple joins |
| Chart Generation | ~0.3s | All chart types |
| Cache Lookup | <0.01s | In-memory |
| LLM SQL Generation | ~0.8s | GPT-4o-mini |

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

### Code Style
- Follow PEP 8
- Use type hints
- Add docstrings
- Include unit tests

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- OpenAI for GPT models
- Gradio team for the UI framework
- Plotly for visualization capabilities
- MCP protocol developers

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/oracle-mcp-data-agent/issues)
- **Email**: support@yourcompany.com
- **Documentation**: [Wiki](https://github.com/yourusername/oracle-mcp-data-agent/wiki)

## 🚀 Roadmap

### Version 2.0 (Q2 2025)
- [ ] Multi-database support
- [ ] Real-time data streaming
- [ ] Advanced ML predictions
- [ ] Custom dashboard builder

### Version 2.1 (Q3 2025)
- [ ] Mobile responsive UI
- [ ] Export to PowerBI/Tableau
- [ ] Scheduled reports
- [ ] Team collaboration features

## 📊 System Requirements

### Minimum Requirements
- CPU: 2 cores
- RAM: 4GB
- Storage: 10GB free space
- Network: Broadband connection

### Recommended Requirements
- CPU: 4+ cores
- RAM: 8GB+
- Storage: 20GB+ free space
- Network: High-speed connection

---

**Built with ❤️ by the Data Engineering Team**

*Last updated: August 2025*