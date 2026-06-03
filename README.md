# Research Learning Agent

A small RAG project for learning from papers, course PDFs, and open-source project notes.

## V1 Goal

V1 focuses on a local PDF question-answering workflow:

1. Upload a PDF.
2. Extract text from the PDF.
3. Split the text into chunks.
4. Build a simple local retrieval index.
5. Ask questions and return answers with source chunks.

V1 is intentionally retrieval-only. It does not call an LLM yet. The answer is a grounded draft made from retrieved chunks, which helps us verify the data pipeline before adding Agent and generation logic.

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

## How V1 Works

```text
PDF upload
  -> pypdf extracts text
  -> text is split into overlapping chunks
  -> TF-IDF builds a local vector index
  -> query retrieves the most relevant chunks
  -> API returns a grounded draft answer with sources
```

## Next Milestones

1. Replace TF-IDF retrieval with semantic embeddings.
2. Add persistent vector storage.
3. Add an LLM answer generator with citations.
4. Add Agent tools for arXiv and GitHub.
5. Add a frontend.
