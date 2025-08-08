# Semantic UI

A minimal web interface for Large Language Models using Microsoft Semantic Kernel and Streamlit.

## Features

- **Single Agent Chat**: Direct conversation with OpenAI/Azure OpenAI models
- **Multi-Agent Mode**: Coordinate multiple specialized AI agents  
- **Plugin System**: Calculator, web search, file operations, and personality plugins
- **Memory Explorer**: Conversation history management

## Requirements

- Python 3.11+
- OpenAI API key or Azure OpenAI credentials

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure environment:**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

3. **Run the application:**
   ```bash
   python -m streamlit run src/semantic_kernel_ui/app.py
   ```

## Configuration

Set your API credentials in `.env`:

```bash
# OpenAI
OPENAI_API_KEY=sk-your-key-here

# Or Azure OpenAI
AZURE_OPENAI_API_KEY=your-azure-key
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT=your-deployment-name
```

## Usage

1. Select your interaction mode (Single Agent, Multi-Agent, Plugin Playground, Memory Explorer)
2. Configure your API provider in the sidebar
3. Click "Initialize System"
4. Start chatting

## Development

```bash
# Install with Poetry
poetry install

# Run tests
pytest

# Start development server
poetry run streamlit run src/semantic_kernel_ui/app.py
```

## License

MIT License
