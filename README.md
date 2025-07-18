# 🧠 ClerkAI: AI-Powered Office Assistant with Model Context Protocol (MCP)

ClerkAI is an AI-native internal assistant that automates document processing, data entry, validation, and reporting through natural language interactions. It uses OCR, NLP, and LLMs (GPT-4+) combined with Model Context Protocol to make business workflows intelligent, extensible, and low-maintenance.

---

## 🚀 Features

- 📄 **Document Ingestion**: Accepts scanned documents, PDFs, and emails.
- 🔍 **Smart Extraction**: Uses OCR + NLP to extract structured data.
- 🧠 **LLM Interaction**: Understands natural language queries and commands.
- 📊 **Interactive Reports**: Generates dynamic HTML-based dashboards.
- 🔄 **Extensible via MCP**: Schemas, workflows, and tools are defined and updated via `model_context.json`.

---

## 🏗️ Architecture Overview

```plaintext
[User Input]
    ↓
[Frontend (React + MUI)]
    ↓
[Backend (Node.js or Spring Boot API)]
    ↓
[LLM Engine (OpenAI GPT-4o + MCP)]
    ↓
[Tool Invocation → DB, OCR, Report Engine]
    ↓
[Structured Data + Visual Output]
```

### 📁 Directory Structure

```bash
clerkai/
├── backend/
│   ├── src/
│   ├── routes/
│   ├── services/
│   └── model_context.json         # MCP schema and tool config
├── frontend/
│   ├── src/
│   ├── components/
│   └── pages/
├── ingest/
│   └── ocr/
│       └── extractors.py          # OCR + NLP pipelines
├── reports/
│   └── render.js                  # HTML/CSV/PDF export engine
├── scripts/
│   └── setup-context.ts           # Sync model context with LLM
├── .env
└── README.md
```

---

## 🔧 Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/samuelj90/clerkAi.git
cd clerkai
```

### 2. Configure Environment

Create a `.env` file:

```env
OPENAI_API_KEY=your_key_here
DB_URL=postgres://user:pass@localhost:5432/clerkai
AZURE_STORAGE_KEY=...
OCR_ENGINE=tesseract
```

### 3. Install Dependencies

**Backend:**
```bash
cd backend
npm install
```

**Frontend:**
```bash
cd ../frontend
npm install
```

### 4. Run the App

**Terminal 1 - Backend**
```bash
cd backend
npm run dev
```

**Terminal 2 - Frontend**
```bash
cd frontend
npm run dev
```

---

## 📦 MCP Context (`model_context.json`)

This file defines:

- **tools** – e.g., DocumentIngestionTool, DatabaseQueryTool
- **schemas** – e.g., Invoice, Vendor, Employee
- **dashboards** – Report layouts and inputs
- **workflows** – Multi-step automations (e.g., PDF → SQL entry)
- **user_roles** – Role-based permissions for different users

**Example:**

```json
{
  "schemas": {
     "Invoice": {
        "invoice_number": "string",
        "amount": "float",
        "due_date": "date"
     }
  },
  "tools": [
     {
        "name": "DatabaseQueryTool",
        "description": "Run SQL on the PostgreSQL DB"
     }
  ],
  "workflows": [
     {
        "name": "ExtractAndStoreInvoice",
        "steps": ["OCR", "Parse", "SchemaMap", "DatabaseWrite"]
     }
  ]
}
```

---

## 🧠 Natural Language Examples

- **"Add this vendor from the scanned form"**  
  ✔️ Triggers document ingestion, maps to Vendor schema, inserts to DB

- **"Show me sales totals by category for Q2"**  
  ✔️ GPT generates SQL, pulls data, renders HTML bar chart

- **"Email all unpaid invoices to vendors"**  
  ✔️ Uses schema + tool + workflow to send templated emails

---

## 📊 Report Exports

- **HTML** (interactive dashboard)
- **PDF** (via Puppeteer)
- **CSV/Excel** (via ExcelJS)

---

## 🔐 Security

- Azure AD SSO (via Microsoft Entra ID)
- RBAC (Admin, Finance User, Viewer)
- Audit logs for AI-generated queries
- PII-aware extraction and storage policies

---

## 🧩 Extending the Assistant

Add a new schema by editing `model_context.json`:

```json
"Schemas": {
  "PurchaseOrder": {
     "po_id": "string",
     "amount": "float",
     "vendor": "string"
  }
}
```

Restart backend or hot-reload the context:

```bash
node scripts/setup-context.ts
```

LLM now understands:

> "Show me all purchase orders above $10,000"

---

## 🤖 Tech Stack

| Layer      | Tech Used                    |
|------------|-----------------------------|
| Frontend   | React + Vite + MUI          |
| Backend    | Node.js (or Spring Boot)    |
| LLM        | OpenAI GPT-4 / GPT-4o       |
| Context    | Model Context Protocol (JSON)|
| OCR        | Tesseract / Google Vision   |
| DB         | PostgreSQL / MongoDB        |
| Reports    | Recharts, jsPDF, ExcelJS    |
| Auth       | Azure AD, OAuth2            |

---

## 📌 Roadmap

- Voice-based interaction
- Context visual editor
- Self-hosted LLM fallback
- Multi-language support
- Fine-tuned RAG over internal docs

---

## 🤝 Contributing

- Fork the repo
- Add your context/schemas or tools
- Submit a PR with your enhancement

