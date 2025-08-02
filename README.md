# 🧠 ClerkAI: AI-Powered Office Assistant with Model Context Protocol (MCP)

ClerkAI is an AI-native internal assistant that automates document processing, data entry, validation, and reporting through natural language interactions. It uses OCR, NLP, and LLMs combined with Model Context Protocol to make business workflows intelligent, extensible, and low-maintenance.

## 🌟 Features

- 📄 **Document Ingestion**: Accepts scanned documents, PDFs, and emails
- 🔍 **Smart Extraction**: Uses OCR + NLP to extract structured data
- 🧠 **LLM Integration**: Understands natural language queries and commands
- 📊 **Interactive Reports**: Generates dynamic HTML-based dashboards
- 🔄 **Extensible via MCP**: Schemas, workflows, and tools are defined and updated via `model_context.json`
- 🛠️ **CrewAI Integration**: Orchestrates workflows and UI components
- 🚀 **FastAPI Backend**: Modern, fast, and well-documented REST API
- 🐳 **Docker Ready**: Full containerization with Docker Compose
- 📈 **Monitoring & Logging**: Comprehensive logging and health checks

## 🏗️ Architecture Overview

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   FastAPI API   │    │  CrewAI Agents  │    │ Model Context   │
│   Layer         │◄──►│  & Workflows    │◄──►│ Protocol (MCP)  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Document      │    │   NLP & LLM     │    │  PostgreSQL     │
│   Processing    │    │   Services      │    │  Database       │
│   (OCR)         │    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Core Components

1. **FastAPI Application** (`src/main.py`)
   - RESTful API endpoints
   - Authentication and authorization
   - Request/response handling

2. **Document Processing Pipeline**
   - OCR Service (`src/services/ocr_service.py`)
   - NLP Service (`src/services/nlp_service.py`)
   - LLM Service (`src/services/llm_service.py`)

3. **Model Context Protocol** (`src/mcp/`)
   - Dynamic tool and workflow definitions
   - Schema validation and management
   - Runtime configuration updates

4. **CrewAI Orchestration** (`src/crew/`)
   - Multi-agent workflow execution
   - Task coordination and management
   - Error handling and recovery

5. **Database Layer** (`src/models/`)
   - SQLAlchemy ORM models
   - Alembic migrations
   - Relationship management

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- PostgreSQL 15+
- Redis 7+
- Tesseract OCR
- Docker & Docker Compose (optional)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd travel-planner-be
   ```

2. **Set up virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Configure environment**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

4. **Set up database**
   ```bash
   # Start PostgreSQL and Redis
   # Create database
   createdb clerkai
   
   # Run migrations
   alembic upgrade head
   ```

5. **Start the application**
   ```bash
   python -m uvicorn src.main:app --reload
   ```

### Docker Setup

1. **Using Docker Compose**
   ```bash
   docker-compose up -d
   ```

   This will start:
   - ClerkAI application (port 8000)
   - PostgreSQL database (port 5432)
   - Redis cache (port 6379)
   - Celery worker for background tasks

2. **Run migrations**
   ```bash
   docker-compose exec app alembic upgrade head
   ```

## 📝 Configuration

### Environment Variables

Key environment variables (see `.env.example` for complete list):

```bash
# Application
ENVIRONMENT=development
DATABASE_URL=postgresql://clerkuser:clerkpass@localhost:5432/clerkai

# OpenAI (for LLM features)
OPENAI_API_KEY=your-openai-api-key-here

# OCR
TESSERACT_CMD=/usr/bin/tesseract  # Optional, uses system default
OCR_LANGUAGE=eng

# File Processing
UPLOAD_DIR=uploads
MAX_FILE_SIZE=10485760  # 10MB
```

### Model Context Protocol Configuration

The `model_context.json` file defines:

- **Tools**: Available processing tools (OCR, NLP, LLM)
- **Workflows**: Multi-step processing pipelines
- **Schemas**: Data extraction templates
- **Global Config**: System-wide settings

Example tool definition:
```json
{
  "tools": {
    "document_ocr": {
      "name": "document_ocr",
      "description": "Extract text from documents using OCR",
      "input_schema": {
        "type": "object",
        "properties": {
          "file_path": {"type": "string"},
          "language": {"type": "string", "default": "eng"}
        }
      }
    }
  }
}
```

## 🔧 API Usage

### Document Upload

```bash
curl -X POST "http://localhost:8000/api/v1/documents/upload" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -F "file=@document.pdf" \
  -F "document_type=invoice" \
  -F "title=Monthly Invoice"
```

### List Documents

```bash
curl -X GET "http://localhost:8000/api/v1/documents/" \
  -H "Authorization: Bearer YOUR_TOKEN"
```

### Execute Workflow

```bash
curl -X POST "http://localhost:8000/api/v1/workflows/document_processing/execute" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "input_data": {
      "file_path": "/path/to/document.pdf",
      "schema": "invoice"
    }
  }'
```

### Dashboard

Visit `http://localhost:8000/api/v1/reports/dashboard/html` for an interactive dashboard.

## 🛠️ Development

### Project Structure

```
├── src/
│   ├── api/                 # FastAPI routes and schemas
│   │   ├── routers/         # API endpoint routers
│   │   └── schemas/         # Pydantic models
│   ├── models/              # SQLAlchemy database models
│   ├── services/            # Core processing services
│   │   ├── ocr_service.py   # OCR text extraction
│   │   ├── nlp_service.py   # Natural language processing
│   │   └── llm_service.py   # Large language model integration
│   ├── mcp/                 # Model Context Protocol
│   ├── crew/                # CrewAI workflow orchestration
│   ├── reports/             # Reporting and dashboard generation
│   └── utils/               # Utility functions
├── config/                  # Configuration modules
├── alembic/                 # Database migrations
├── tests/                   # Test suites
├── docker-compose.yml       # Docker services
├── model_context.json       # MCP configuration
└── requirements.txt         # Python dependencies
```

### Adding New Document Types

1. **Define Schema** in `model_context.json`:
   ```json
   {
     "schemas": {
       "purchase_order": {
         "name": "purchase_order",
         "description": "Purchase order document schema",
         "fields": {
           "po_number": {"type": "string", "required": true},
           "vendor": {"type": "string", "required": true},
           "total": {"type": "number", "required": true}
         }
       }
     }
   }
   ```

2. **Create Workflow**:
   ```json
   {
     "workflows": {
       "po_processing": {
         "name": "po_processing",
         "description": "Process purchase orders",
         "steps": [
           {
             "name": "extract_text",
             "tool": "document_ocr",
             "input_mapping": {"file_path": "${input.file_path}"}
           },
           {
             "name": "extract_data",
             "tool": "llm_data_extraction",
             "input_mapping": {
               "text": "${steps.extract_text.output.text}",
               "schema": "${schemas.purchase_order}"
             }
           }
         ]
       }
     }
   }
   ```

### Testing

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_ocr_service.py

# Run with coverage
pytest --cov=src tests/
```

### Code Quality

```bash
# Format code
black src/ tests/

# Sort imports
isort src/ tests/

# Lint code
flake8 src/ tests/

# Type checking
mypy src/
```

## 📊 Monitoring

### Health Checks

- Application: `GET /health`
- Detailed: `GET /health/detailed`
- Database: `GET /health/ready`

### Logging

Logs are written to:
- Console (development)
- File: `logs/clerkai.log` (with rotation)
- Structured JSON format in production

### Metrics

Prometheus metrics available at `:9090/metrics` when enabled.

## 🔐 Security

### Authentication

ClerkAI supports multiple authentication methods:

1. **JWT Tokens**: For web applications
2. **API Keys**: For service-to-service communication

### File Security

- File type validation
- Size limits
- Virus scanning (when configured)
- Secure file storage

## 🚢 Deployment

### Docker Production

```bash
# Build production image
docker build -t clerkai:latest .

# Run with production settings
docker-compose -f docker-compose.prod.yml up -d
```

### Environment-Specific Configs

- Development: `docker-compose.yml`
- Production: `docker-compose.prod.yml`
- Testing: `docker-compose.test.yml`

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Run quality checks
6. Submit a pull request

### Development Setup

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Set up pre-commit hooks
pre-commit install

# Run tests before committing
pytest
```

## 📚 Documentation

- API Documentation: `http://localhost:8000/docs` (Swagger UI)
- ReDoc: `http://localhost:8000/redoc`
- Architecture: See `docs/architecture.md`
- Deployment: See `docs/deployment.md`

## 🔧 Troubleshooting

### Common Issues

1. **OCR not working**
   - Install Tesseract: `sudo apt-get install tesseract-ocr`
   - Set `TESSERACT_CMD` in environment

2. **Database connection errors**
   - Check PostgreSQL is running
   - Verify connection string in `.env`

3. **File upload failures**
   - Check `UPLOAD_DIR` permissions
   - Verify `MAX_FILE_SIZE` settings

4. **LLM features not working**
   - Set `OPENAI_API_KEY` environment variable
   - Check API key permissions

### Performance Tuning

- Adjust `CREW_MAX_WORKERS` for concurrent processing
- Configure Redis for caching
- Use appropriate database connection pooling

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [FastAPI](https://fastapi.tiangolo.com/) - Modern web framework
- [spaCy](https://spacy.io/) - NLP library
- [Tesseract](https://github.com/tesseract-ocr/tesseract) - OCR engine
- [CrewAI](https://crewai.io/) - Multi-agent workflows
- [SQLAlchemy](https://sqlalchemy.org/) - Database ORM

## 📞 Support

For support and questions:

- 📧 Email: support@clerkai.example
- 💬 Discussions: GitHub Discussions
- 🐛 Issues: GitHub Issues
- 📖 Wiki: GitHub Wiki

---

**ClerkAI** - Making office automation intelligent and accessible! 🚀