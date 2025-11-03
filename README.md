# Semantic Kernel UI

A web-based interface for Large Language Models built with Microsoft Semantic Kernel and Streamlit.

## Features

### Core Capabilities
- **Multi-Provider Support**: OpenAI, Azure OpenAI, and Anthropic Claude
- **Single Agent Chat**: Direct conversations with configurable personality and model settings
- **Multi-Agent Workflows**: Coordinate specialized AI agents with pre-built templates
- **Conversation Memory**: Persistent storage with semantic search via ChromaDB
- **Streaming Responses**: Real-time token-by-token output

### Built-in Plugins

**Computation & Utilities**
- **Calculator**: Mathematical operations, unit conversions (length, weight, temperature)
- **DateTime**: Time operations, timezone conversion, date calculations
- **Text Processing**: Word counting, JSON formatting, Base64 encoding, hash generation

**Data & Integration**
- **HTTP/API**: Secure read-only HTTP requests, JSON parsing, public API access
- **File Index**: Safe read-only file browsing with path restrictions
- **Web Search**: Internet search via SerpAPI or Google Custom Search
- **Export**: Conversation export to Markdown, JSON, CSV formats
- **Document Intelligence**: OCR and document analysis with Tesseract (local/free) or Azure AI (cloud/production)

**Customization**
- **Personality**: Adjust response tone (friendly, professional, technical, creative, casual)

### Agent Presets

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

## Docker Deployment

### Quick Start with Docker

1. Build and run with Docker Compose:
```bash
docker-compose up -d
```

2. Access the application:
- **HTTPS**: https://localhost (production mode with nginx)
- **HTTP**: http://localhost (development mode, direct Streamlit on default port)

### Docker Configuration

The application includes production-ready Docker setup:
- **Streamlit**: Runs as non-root user
- **Nginx**: Reverse proxy with TLS, security headers, and rate limiting
- **Persistent Storage**: Conversation memory and vector DB preserved in `./memory`

### Security Notes

Before deploying to production:
1. Change the default nginx password in `nginx/nginx.conf`
2. Replace self-signed certificates with valid TLS certificates
3. Update CORS and CSP headers for your domain
4. Review and adjust rate limiting settings as needed

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
DEFAULT_SEARCH_RESULTS=5                   # Default search result count
MAX_PAGINATION_LIMIT=50                    # Max pagination limit
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

**Document Intelligence Plugin**
```bash
ENABLE_DOCUMENT_INTELLIGENCE=true          # Enable document OCR/analysis
ENABLE_LLM_CLASSIFICATION=true             # Use LLM for document classification
ENABLE_TESSERACT_OCR=true                  # Enable Tesseract OCR (local, free)
ENABLE_AZURE_DOC_INTELLIGENCE=false        # Enable Azure AI (cloud, production)

# Tesseract OCR (local, free)
TESSERACT_CMD=                             # Auto-detect or set path
TESSERACT_LANGUAGES=eng+fin+swe            # Supported languages

# Azure AI Document Intelligence (cloud, production)
AZURE_DOC_INTELLIGENCE_ENDPOINT=https://your-resource.cognitiveservices.azure.com/
AZURE_DOC_INTELLIGENCE_KEY=your-key

# Performance Settings
DOC_INTELLIGENCE_MAX_CONTENT_LENGTH=10000  # Max content length
DOC_INTELLIGENCE_CACHE_RESULTS=true        # Cache OCR results
```

**Application Settings**
```bash
APP_TITLE=Semantic Kernel UI               # Browser tab title
DEBUG_MODE=false                           # Enable debug logging
TEMPERATURE=0.7                            # Default model temperature
MAX_TOKENS=2000                            # Default max response tokens
```

**Authentication** (Optional - disabled by default)
```bash
# Enable authentication
ENABLE_AUTH=false

# JWT Token Authentication (secret-based)
JWT_SECRET=your-secret-key-change-this-in-production
JWT_ALGORITHM=HS256
JWT_EXPIRATION_HOURS=24

# Google OAuth
ENABLE_GOOGLE_OAUTH=false
GOOGLE_CLIENT_ID=your-google-client-id
GOOGLE_CLIENT_SECRET=your-google-client-secret
GOOGLE_REDIRECT_URI=http://localhost:8501

# Facebook OAuth
ENABLE_FACEBOOK_OAUTH=false
FACEBOOK_APP_ID=your-facebook-app-id
FACEBOOK_APP_SECRET=your-facebook-app-secret
FACEBOOK_REDIRECT_URI=http://localhost:8501

# Restrict access to specific users (optional - comma-separated emails)
ALLOWED_USERS=user1@example.com,user2@example.com
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

### Authentication

Authentication is optional and disabled by default. When enabled, users must authenticate before accessing the application.

**JWT Token Authentication:**
1. Generate a token using the provided script:
   ```bash
   python scripts/generate_token.py --email user@example.com --name "User Name"
   ```
2. Copy the generated token
3. Enter the token on the login page
4. Click "Login with Token"

**OAuth Authentication:**
1. Set up OAuth credentials with Google or Facebook
2. Configure the appropriate environment variables
3. Click the OAuth provider button on the login page
4. Authorize the application
5. You'll be redirected back and automatically logged in

**Restricting Access:**
- Leave `ALLOWED_USERS` empty to allow any authenticated user
- Set `ALLOWED_USERS` to a comma-separated list of emails to restrict access

**Security Notes:**
- Always use a strong, random `JWT_SECRET` in production
- Never commit secrets to version control
- Use HTTPS in production (OAuth providers require it)
- Rotate JWT secrets periodically
- Set appropriate token expiration times

## Development

### Project Structure
```
semantick/
├── src/semantic_kernel_ui/
│   ├── app.py                 # Main Streamlit application
│   ├── config.py              # Configuration management
│   ├── agent_presets.py       # Multi-agent workflow templates
│   ├── auth/                  # Authentication module
│   │   └── auth_manager.py    # OAuth and JWT authentication
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
│   └── plugins/               # Built-in plugins
│       ├── calculator.py
│       ├── datetime_utils.py
│       ├── text_processing.py
│       ├── http_api.py
│       ├── filesystem.py
│       ├── websearch.py
│       ├── export.py
│       ├── document_intelligence.py
│       ├── ocr_backends.py
│       └── personality.py
└── tests/                     # Comprehensive test suite
```

### Running Tests
```bash
# Run all tests
pytest tests/ -v

# Run with coverage
make test-cov

# Run linting
make lint
```

### Code Quality
- Type hints throughout with mypy compliance
- Comprehensive test coverage with high pass rate
- Code quality enforced with flake8
- Security-focused design (read-only plugins, path validation)
- Pydantic validation for all configurations

## Security

A comprehensive security audit has been completed with strong scores across all areas.

### Plugin Security
- **FileIndex**: Path whitelist, no write operations, size limits, symlink protection
- **HTTP/API**: URL validation, SSRF protection (localhost blocking), read-only, timeout protection
- **Calculator**: Safe AST evaluation, no eval/exec usage
- **Document Intelligence**: File type validation, size limits, safe OCR processing

### Infrastructure Security
- **Docker**: Non-root user, resource limits, minimal base image
- **Nginx**: TLS 1.2+, HSTS, CSP headers, rate limiting
- **Network**: HTTP to HTTPS redirect, security headers, server tokens hidden

### Best Practices
- API keys stored in environment variables only
- No credentials in code or version control
- Sandboxed file access with configurable restrictions
- Request size and timeout limits
- Input validation with Pydantic models
- Path traversal protection

### Recommendations
- Add authentication layer for production deployments
- Change default nginx password before deployment
- Regular security updates for dependencies
- Monitor rate limiting and access logs

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
