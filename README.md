# Wonderland RAG Assistant API

A FastAPI backend for a citation-grounded Retrieval-Augmented Generation (RAG)
chatbot based on *Alice's Adventures in Wonderland* by Lewis Carroll.

The application retrieves relevant passages from a processed Project Gutenberg
corpus, then asks a language model to provide a concise answer grounded only
in those retrieved passages.

## Features

- Retrieval-Augmented Generation workflow
- Sentence Transformers embeddings using `all-MiniLM-L6-v2`
- ChromaDB vector retrieval over 197 processed book chunks
- Groq-powered text generation
- Chapter-level citations and source excerpts
- Grounding rule: answer only from retrieved source passages
- Unsupported-question fallback response
- `GET /health` health endpoint
- `GET /ready` readiness endpoint
- `POST /api/v1/chat` chatbot endpoint
- `POST /api/v1/feedback` feedback endpoint
- Pydantic request validation
- Request IDs and response latency headers
- Structured API logging
- Automated tests with `pytest` and FastAPI `TestClient`
- Dockerfile for reproducible container deployment

## Architecture

```text
User Question
    ↓
FastAPI API
    ↓
Sentence Transformers Query Embedding
    ↓
ChromaDB Similarity Search
    ↓
Relevant Alice in Wonderland Chunks
    ↓
Groq LLM with Grounded Prompt
    ↓
Cited Answer + Source Metadata
```

## Corpus

- Book: *Alice's Adventures in Wonderland*
- Author: Lewis Carroll
- Source: Project Gutenberg eBook #11
- Source URL: https://www.gutenberg.org/files/11/11-h/11-h.htm
- Processed corpus: `data/processed/alice_chunks.json`

The chatbot preserves chapter metadata on each chunk so API responses can cite
the source chapter used for retrieval.

## API endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Returns service health and version |
| `GET` | `/ready` | Returns RAG service readiness and loaded chunk count |
| `POST` | `/api/v1/chat` | Returns a grounded chatbot answer and source citations |
| `POST` | `/api/v1/feedback` | Records an answer rating and optional comment |

## Example chat request

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Who does Alice follow down the rabbit-hole?",
    "top_k": 2
  }'
```

## Example response

```json
{
  "request_id": "req_example123",
  "answer": "Alice follows a White Rabbit down the rabbit-hole. [Chapter I]\n\nEducational information only—not personalized advice.",
  "sources": [
    {
      "chunk_id": "alice-chapter-I-chunk-000",
      "chapter_number": "I",
      "chapter_title": "Down the Rabbit-Hole",
      "source_url": "[https://www.gutenberg.org/files/11/11-h/11-h.htm](https://www.gutenberg.org/files/11/11-h/11-h.htm)",
      "excerpt": "CHAPTER I. Down the Rabbit-Hole...",
      "retrieval_distance": 0.4109
    }
  ],
  "retrieved_chunk_count": 2,
  "latency_ms": 999,
  "status": "success"
}
```

## Local setup

### 1. Clone the repository

```bash
git clone [https://github.com/d1lemon/wonderland-rag-assistant.git](https://github.com/d1lemon/wonderland-rag-assistant.git)
cd wonderland-rag-assistant
```

### 2. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set environment variables

Copy the template:

```bash
cp .env.example .env
```

Set these values using your operating system, terminal, or a local `.env` loader:

```text
GROQ_API_KEY=your_groq_api_key
GITHUB_USERNAME=your_github_username
GROQ_MODEL=groq/compound-mini
ALLOWED_ORIGINS=http://localhost,http://localhost:3000,http://127.0.0.1:3000
```

Never commit a real `.env` file or API key.

### 5. Run the API

```bash
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Open interactive API documentation:

```text
http://127.0.0.1:8000/docs
```

## Run tests

```bash
pytest -q
```

Current test coverage includes health, readiness, request-ID headers, invalid
chat requests, required request fields, and feedback endpoint validation.

## Docker

Build the image:

```bash
docker build -t wonderland-rag-api .
```

Run the container:

```bash
docker run --rm -p 8080:8080 \
  -e GROQ_API_KEY="your_groq_api_key" \
  -e GITHUB_USERNAME="your_github_username" \
  -e GROQ_MODEL="groq/compound-mini" \
  -e ALLOWED_ORIGINS="http://localhost:3000" \
  wonderland-rag-api
```

Then test:

```text
http://127.0.0.1:8080/health
```

## Security and limitations

- Do not store API keys in source code, notebooks, logs, or GitHub.
- The current feedback endpoint logs feedback but does not persist it. A later
  version will add Supabase authentication and a database.
- The current ChromaDB collection is built in memory at service startup.
- Before public deployment, add user authentication, database-backed feedback,
  rate limiting, a strict CORS allowlist, monitoring, and error tracking.
- This bot is educational and should answer only from retrieved source context.
