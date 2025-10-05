# Semantic Kernel UI

A web-based interface for Large Language Models built with Microsoft Semantic Kernel and Streamlit.

## Features

### Core Capabilities
- **Multi-Provider Support**: OpenAI, Azure OpenAI, and Anthropic Claude
- **Single Agent Chat**: Direct conversations with configurable personality and model settings
- **Multi-Agent Workflows**: Coordinate specialized AI agents with pre-built templates
- **Conversation Memory**: Persistent storage with semantic search via ChromaDB
- **Streaming Responses**: Real-time token-by-token output

### Built-in Plugins (8)

**Computation & Utilities**
- **Calculator**: Mathematical operations, unit conversions (length, weight, temperature)
- **DateTime**: Time operations, timezone conversion, date calculations
- **Text Processing**: Word counting, JSON formatting, Base64 encoding, hash generation

**Data & Integration**
- **HTTP/API**: Secure read-only HTTP requests, JSON parsing, public API access
- **File Index**: Safe read-only file browsing with path restrictions
- **Web Search**: Internet search via SerpAPI or Google Custom Search
- **Export**: Conversation export to Markdown, JSON, CSV formats

**Customization**
- **Personality**: Adjust response tone (friendly, professional, technical, creative, casual)

### Agent Presets (5)

Pre-configured multi-agent workflows for common tasks:
- **Code Review**: Developer + Quality Reviewer
- **Content Creation**: Researcher + Writer + Editor
- **Research Report**: Researcher + Analyst + Report Writer
- **Technical Design**: Architect + Technical Analyst + Design Reviewer
- **Debug Session**: Debug Expert + Systems Analyst

## Installation

### Requirements
- Python 3.11+
- API key for OpenAI, Azure OpenAI, or Anthropic Claude

### Setup

1. Clone and install dependencies:
```bash
git clone <repository-url>
cd semantick
pip install -r requirements.txt
```

2. Configure environment variables:
```bash
cp .env.example .env
# Edit .env with your API credentials
```

3. Run the application:
```bash
streamlit run src/semantic_kernel_ui/app.py
```

## Configuration

### API Providers

**OpenAI**
```bash
OPENAI_API_KEY=sk-your-key-here
```

**Azure OpenAI**
```bash
AZURE_OPENAI_API_KEY=your-azure-key
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT=your-deployment-name
```

**Anthropic Claude**
```bash
ANTHROPIC_API_KEY=your-anthropic-key
```

### Optional Settings

**Memory & Storage**
```bash
MEMORY_PERSIST_DIRECTORY=./memory          # Conversation storage directory
VECTOR_DB_DIRECTORY=./memory/vector_db     # ChromaDB vector storage
USE_VECTOR_DB=true                         # Enable semantic search
DEFAULT_SEARCH_RESULTS=5                   # Default search result count (1-50)
MAX_PAGINATION_LIMIT=50                    # Max pagination limit (1-200)
```

**Web Search Integration**
```bash
SERPAPI_KEY=your-serpapi-key
# OR
GOOGLE_CSE_API_KEY=your-google-key
GOOGLE_CSE_ENGINE_ID=your-engine-id
```

**File Index Plugin**
```bash
ALLOWED_DIRECTORIES=.                      # Comma-separated allowed directories
MAX_FILE_SIZE_MB=10                        # Maximum file size for reading
```

**Application Settings**
```bash
APP_TITLE=Semantic Kernel UI               # Browser tab title
DEBUG_MODE=false                           # Enable debug logging
TEMPERATURE=0.7                            # Default model temperature (0.0-2.0)
MAX_TOKENS=2000                            # Default max response tokens
```

## Usage

### Single Agent Mode
1. Select "Single Agent" from the mode dropdown
2. Choose your model and personality in Model Settings
3. Configure temperature and max tokens
4. Click "Initialize System"
5. Start chatting

### Multi-Agent Mode
1. Select "Multi-Agent" from the mode dropdown
2. Choose a preset workflow or create custom agents
3. Define the topic and objectives
4. Set maximum rounds
5. Initialize and watch agents collaborate

### Plugin Playground
1. Select "Plugin Playground" mode
2. Choose from available plugins
3. Test plugin functions with sample inputs
4. View results in real-time

### Memory Explorer
1. Select "Memory Explorer" mode
2. Search conversations semantically
3. Browse conversation history
4. Export or delete conversations

## Development

### Project Structure
```
semantick/
├── src/semantic_kernel_ui/
│   ├── app.py                 # Main Streamlit application
│   ├── config.py              # Configuration management
│   ├── agent_presets.py       # Multi-agent workflow templates
│   ├── core/
│   │   ├── kernel_manager.py  # LLM interaction manager
│   │   └── agent_manager.py   # Multi-agent orchestration
│   ├── connectors/            # LLM provider connectors
│   │   ├── openai_chat.py     # OpenAI integration
│   │   ├── azure_openai_chat.py  # Azure OpenAI integration
│   │   └── anthropic_chat.py  # Anthropic Claude integration
│   ├── memory/
│   │   ├── memory_manager.py  # Conversation storage
│   │   └── vector_store.py    # ChromaDB integration
│   └── plugins/               # 8 built-in plugins
│       ├── calculator.py
│       ├── datetime_utils.py
│       ├── text_processing.py
│       ├── http_api.py
│       ├── filesystem.py
│       ├── websearch.py
│       ├── export.py
│       └── personality.py
└── tests/                     # Comprehensive test suite (115 tests)
```

### Running Tests
```bash
pytest tests/ -v
```

### Code Quality
- Type hints throughout
- Comprehensive test coverage (115 tests, 100% pass rate)
- Security-focused design (read-only plugins, path validation)
- Pydantic validation for all configurations

## Security

### Plugin Security
- **FileIndex**: Path whitelist, no write operations, size limits, symlink protection
- **HTTP/API**: URL validation, localhost blocking, read-only, timeout protection
- **Calculator**: Safe AST evaluation, no eval/exec usage

### Best Practices
- API keys stored in environment variables
- No credentials in code or version control
- Sandboxed file access with configurable restrictions
- Request size and timeout limits

## Plugin Development

Create custom plugins by implementing the `@kernel_function` decorator:

```python
from semantic_kernel.functions import kernel_function
from typing import Annotated

class MyPlugin:
    @kernel_function(
        name="my_function",
        description="Description of what this does"
    )
    def my_function(
        self,
        input_param: Annotated[str, "Parameter description"]
    ) -> Annotated[str, "Return value description"]:
        return f"Processed: {input_param}"
```

Register in `src/semantic_kernel_ui/plugins/__init__.py` and add to app.py plugin initialization.

## Troubleshooting

### Common Issues

**No API provider configured**
- Verify `.env` file exists with valid API keys
- Check environment variables are loaded correctly

**ChromaDB errors**
- Clear vector database: `rm -rf memory/vector_db`
- Ensure write permissions for memory directory

**Import errors**
- Reinstall dependencies: `pip install -r requirements.txt`
- Verify Python version: `python --version` (requires 3.11+)

**Plugin not appearing**
- Check plugin is imported in `__init__.py`
- Verify plugin added to app.py `_ensure_plugins()`

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass: `pytest tests/`
5. Submit a pull request

## License

MIT License

## Acknowledgments

- Built with [Microsoft Semantic Kernel](https://github.com/microsoft/semantic-kernel)
- UI powered by [Streamlit](https://streamlit.io)
- Vector storage by [ChromaDB](https://www.trychroma.com)
