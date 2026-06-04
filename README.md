# Research Learning Agent

A small RAG project for learning from papers, course PDFs, and open-source project notes.

## Current Goal

The current version focuses on a local PDF question-answering workflow:

1. Upload a PDF.
2. Extract text from the PDF.
3. Split the text into chunks.
4. Build a local semantic retrieval index.
5. Ask questions and return answers with source chunks.

The project is still intentionally retrieval-only. It does not call an LLM yet. The answer is a grounded draft made from retrieved chunks, which helps us verify the data pipeline before adding Agent and generation logic.

## Version Notes

### V1

V1 used `TF-IDF + cosine similarity`.

This made the first version fast and easy to understand, but it mostly matched keywords.

### V2

V2 uses `sentence-transformers` semantic embeddings by default.

The default embedding model is:

```text
BAAI/bge-small-zh-v1.5
```

This model is better for Chinese learning materials and can also handle English technical text. If semantic embedding is unavailable, the app falls back to TF-IDF retrieval.

### V3

V3 adds optional LLM answer generation.

If `OPENAI_API_KEY` is configured, `/query` sends the retrieved chunks to an LLM and asks it to answer with source citations like `[1]`.

If `OPENAI_API_KEY` is not configured, the app still works locally and returns a retrieval-only answer with the most relevant source chunks.

## Project Structure

```text
backend/
  app/
    main.py
    rag.py
    schemas.py
data/
  uploads/
tests/
```

## Environment

This machine already has a useful conda environment named `graph-rag`.

Install or refresh the V1 dependencies:

```powershell
conda run -n graph-rag python -m pip install -r requirements.txt
```

Optional LLM configuration:

```powershell
$env:OPENAI_API_KEY="your-api-key"
$env:RLA_OPENAI_BASE_URL="https://your-openai-compatible-endpoint/v1"
$env:RLA_LLM_MODEL="gpt-4o-mini"
```

`RLA_OPENAI_BASE_URL` is optional. Use it when you call an OpenAI-compatible relay or gateway.

`RLA_LLM_MODEL` is optional. If it is not set, the app uses `gpt-4o-mini`.

For local development, you can also copy `.env.example` to `.env` and fill in your own values. `.env` is ignored by Git.

## Run The API

Use the `graph-rag` conda environment if it is available on this machine.

```powershell
conda run -n graph-rag uvicorn backend.app.main:app --reload
```

Then open:

```text
http://127.0.0.1:8000/docs
```

## API Endpoints

### Health Check

```text
GET /health
```

### Upload A PDF

```text
POST /documents/upload
```

Use the Swagger UI at `http://127.0.0.1:8000/docs` and upload a PDF through the form.

### List Uploaded Documents

```text
GET /documents
```

### Ask A Question

```text
POST /query
```

Example JSON body:

```json
{
  "question": "What is retrieval augmented generation?",
  "top_k": 4
}
```

The response includes:

```text
retrieval_mode: semantic or tfidf
answer_mode: llm, retrieval_only, or no_sources
model: the LLM model name, or null when no LLM is used
```

## How V1 Works

```text
PDF upload
  -> pypdf extracts text
  -> text is split into overlapping chunks
  -> sentence-transformers encodes chunks into vectors
  -> query is encoded into a vector
  -> cosine-like vector similarity retrieves relevant chunks
  -> optional LLM generates a cited answer
  -> API returns the answer and source chunks
```

## Next Milestones

1. Add persistent vector storage.
2. Add Agent tools for arXiv and GitHub.
3. Add a frontend.
4. Add evaluation data for retrieval quality.
5. Add Docker deployment.
