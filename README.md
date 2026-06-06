# Research Learning Agent

[中文文档](README.zh-CN.md)

A small RAG project for learning from papers, course PDFs, and open-source project notes.

## Current Goal

The current version focuses on a local paper-learning workflow:

1. Upload a PDF.
2. Extract text from the PDF.
3. Split the text into chunks.
4. Build a local semantic retrieval index.
5. Ask questions and return answers with source chunks.
6. Generate study summaries, key points, and reading plans with an optional LLM.
7. Use a scientific-style web frontend for daily testing.
8. Manage the backend paper library with extracted metadata and duplicate markers.
9. Compare papers within a research direction using retrieved evidence.

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

V3 uses the OpenAI-compatible Chat Completions API, so it can work with many relay or gateway services that expose a `/v1` endpoint.

### V4

V4 adds local index persistence.

Uploaded PDFs are still saved under `data/uploads/`. Extracted document metadata and chunks are saved to:

```text
data/index/rag_store.json
```

When the API starts, it loads this JSON file and rebuilds the in-memory retrieval index. This means the app no longer forgets uploaded documents after a restart.

### V5

V5 adds study workflow endpoints on top of retrieval and LLM answering.

Instead of asking a free-form question only, you can ask the system to run common learning tasks:

```text
POST /study/summary
POST /study/key-points
POST /study/reading-plan
```

These endpoints retrieve relevant chunks and use task-specific prompts to generate study-friendly answers.

### V6

V6 adds a React + Vite frontend.

The frontend provides a scientific learning workspace with:

```text
PDF upload
document list
index rebuild
free-form question answering
study summary
key points
reading plan
source chunk display
retrieval/answer mode badges
```

The frontend runs on `http://127.0.0.1:5173` and proxies API calls to the backend at `http://127.0.0.1:8000`.

### V7

V7 changes the product direction from a single-document learning tool to a direction-level literature research workspace.

Instead of asking the system to study every uploaded paper, users can enter a research direction such as:

```text
water quality prediction neural networks
graph neural networks for recommendation
remote sensing change detection
```

The system then:

1. Searches the backend paper library for relevant evidence chunks.
2. Aggregates matching chunks into paper-level candidates.
3. Ranks the most relevant papers.
4. Generates a direction review, method map, or detailed briefing from the selected evidence.

New V7 endpoints:

```text
POST /literature/search
POST /literature/review
POST /literature/methods
POST /literature/details
```

### V8

V8 upgrades the backend PDF collection into a more usable paper library.

When a PDF is uploaded or the library is reindexed, the system now tries to extract local paper metadata from the first pages:

```text
title
authors
year
venue
doi
abstract
keywords
duplicate_of
duplicate_reason
```

Duplicate detection currently uses simple local rules:

```text
same DOI
same normalized title
same filename
```

The frontend now shows paper titles, DOI, year, abstract previews, and duplicate markers in both the left paper library and the direction-level literature results.

V8 does not call external metadata services yet. Metadata extraction is intentionally local and heuristic, so some PDFs with unusual layouts may still need future enrichment through Crossref, Semantic Scholar, or arXiv.

### V9

V9 adds Crossref DOI metadata enrichment.

If a paper already has a DOI from local PDF parsing, the backend can call the public Crossref Works API and use the returned record to improve:

```text
title
authors
year
venue
publisher
DOI
external URL
reference count
subjects/keywords
abstract when Crossref provides one
```

New V9 endpoint:

```text
POST /documents/enrich-metadata
```

The frontend includes a Crossref refresh button in the paper library. Metadata records now include:

```text
metadata_source: local or crossref
is_enriched: true or false
```

### V10

V10 adds Semantic Scholar title-search fallback for metadata enrichment.

The enrichment endpoint still tries Crossref first when a DOI is available. If the paper has no DOI, or Crossref cannot find a usable record, the backend searches Semantic Scholar by paper title and accepts the best result only when the title similarity is high enough.

V10 can enrich papers without DOI with:

```text
title
authors
year
venue
DOI when Semantic Scholar exposes one
abstract
official URL
reference count
citation count
fields of study
metadata confidence
title match score
```

Metadata records now support:

```text
metadata_source: local, crossref, or semantic_scholar
metadata_confidence: local, low, medium, or high
metadata_match_score: similarity score for Semantic Scholar title matching
```

### V11

V11 adds section-aware retrieval.

Each chunk now stores a coarse paper section label such as:

```text
abstract
introduction
related_work
methods
experiments
results
discussion
conclusion
references
unknown
```

`/query`, `/study/*`, and `/literature/*` accept an optional `section_filter` field. This lets the system retrieve evidence only from a specific part of the paper, for example only `methods` or only `experiments`.

Source chunks now include:

```text
section
```

Literature paper candidates also include:

```text
evidence_sections
```

The frontend adds a section filter control and displays the section label on each evidence card.

### V12

V12 adds local paper-library filtering and sorting.

`GET /documents` now accepts query parameters for managing a larger local library:

```text
query
keyword
year_from
year_to
source
has_doi
duplicate
sort_by
```

The frontend paper library now includes filters for:

```text
title, author, DOI, or abstract search
keyword search
year range
metadata source: local, crossref, or semantic_scholar
DOI presence
duplicate status
sorting by title, year, citations, references, source, or filename
```

V12 is intentionally local-only. It does not connect to Elsevier or other journal APIs yet. The goal is to make the existing backend paper library easier to manage before adding more external data sources.

### V13

V13 adds a lightweight evaluation workflow for direction-level retrieval.

The backend now has a small built-in evaluation set. Each case runs the existing literature search pipeline, checks whether the retrieved papers and evidence chunks cover expected terms, and returns a score.

New V13 endpoint:

```text
POST /evaluation/literature
```

The frontend adds a `检索评估` mode that shows:

```text
passed cases
average score
matched terms
missing terms
paper count
evidence count
```

This is not a full academic benchmark yet. It is a local regression check that helps detect whether later retrieval changes make direction-level search better or worse.

### V14

V14 adds a paper comparison workflow for direction-level research.

New V14 endpoint:

```text
POST /literature/compare
```

Given a research direction and focus, the system retrieves representative papers, ranks them, keeps evidence chunks, and asks the optional LLM to compare:

```text
research problem
method differences
experiment and data clues
strengths
limitations
reproducible follow-up ideas
```

If no LLM is configured, the endpoint still returns the relevant paper ranking and evidence chunks in retrieval-only mode.

### V15

V15 adds external research discovery.

Instead of only working from PDFs that are already in the local library, the app can search public literature metadata sources for candidate papers:

```text
Semantic Scholar
Crossref
arXiv
OpenAlex
```

New V15 endpoints:

```text
POST /discovery/search
POST /discovery/import-metadata
```

`/discovery/search` returns external candidate papers with title, authors, year, venue, DOI, abstract, citation/reference counts, source URL, PDF URL when available, open-access marker, and an import marker when the same DOI or title already exists locally.

`/discovery/import-metadata` imports a selected candidate into the local paper library as a metadata-only record. It does not download the PDF automatically; the imported record has `pages=0` and `chunks=0` until a PDF is uploaded or linked in a later version.

## Project Structure

```text
backend/
  app/
    discovery.py
    main.py
    rag.py
    schemas.py
data/
  uploads/
frontend/
  src/
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
$env:RLA_OPENAI_WIRE_API="responses"
```

`RLA_OPENAI_BASE_URL` is optional. Use it when you call an OpenAI-compatible relay or gateway.

`RLA_LLM_MODEL` is optional. If it is not set, the app uses `gpt-4o-mini`.

`RLA_OPENAI_WIRE_API` can be `responses` or `chat`. It defaults to `responses`.

For local development, you can also copy `.env.example` to `.env` and fill in your own values. `.env` is ignored by Git.

Never commit your real API key. Keep it in your shell environment or local `.env` file only.

## Run The API

Use the `graph-rag` conda environment if it is available on this machine.

```powershell
conda run -n graph-rag uvicorn backend.app.main:app --reload
```

Then open:

```text
http://127.0.0.1:8000/docs
```

## Run The Frontend

Install frontend dependencies:

```powershell
cd frontend
npm install
```

Start the Vite dev server:

```powershell
npm run dev -- --host 127.0.0.1 --port 5173
```

Then open:

```text
http://127.0.0.1:5173
```

The frontend calls the backend through `/api`, so keep the FastAPI server running on port `8000`.

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

V12 supports optional filters:

```text
GET /documents?query=water&keyword=remote&year_from=2024&source=crossref&has_doi=true&duplicate=false&sort_by=year_desc
```

### Reindex Existing Uploads

```text
POST /documents/reindex
```

This scans existing PDF files in `data/uploads/`, extracts their text again, rebuilds chunks, saves `data/index/rag_store.json`, and refreshes the retrieval index.

In V8, reindexing also refreshes local paper metadata and duplicate markers.

### Enrich Metadata With Crossref

```text
POST /documents/enrich-metadata
```

This tries to enrich existing paper metadata through Crossref for documents that have a DOI.

If Crossref cannot enrich a paper, V10 falls back to Semantic Scholar title search for papers that have a usable title.

### Ask A Question

```text
POST /query
```

Example JSON body:

```json
{
  "question": "What is retrieval augmented generation?",
  "top_k": 4,
  "section_filter": "methods"
}
```

The response includes:

```text
retrieval_mode: semantic or tfidf
answer_mode: llm, retrieval_only, or no_sources
model: the LLM model name, or null when no LLM is used
```

`llm_error_fallback` means retrieval worked, but the LLM request failed. Common causes are an invalid API key, unsupported model name, wrong relay base URL, or relay account limits.

### Study Workflows

```text
POST /study/summary
POST /study/key-points
POST /study/reading-plan
```

Example JSON body:

```json
{
  "topic": "这些文档",
  "focus": "研究方法和实验结论",
  "top_k": 6,
  "section_filter": "experiments"
}
```

Use `/query` for free-form questions. Use `/study/*` when you want structured learning output such as a document overview, key points, or a reading plan.

### Direction-Level Literature Workflows

```text
POST /literature/search
POST /literature/review
POST /literature/methods
POST /literature/details
POST /literature/compare
```

Example JSON body:

```json
{
  "query": "water quality prediction neural networks",
  "focus": "methods and experiments",
  "top_k_documents": 3,
  "evidence_k": 8,
  "section_filter": "methods"
}
```

Use `/literature/search` when you only want relevant paper ranking and evidence chunks.

Use `/literature/review` for a direction-level literature overview.

Use `/literature/methods` when you want method categories, technical details, advantages, limitations, and corresponding papers.

Use `/literature/details` when you want a focused briefing for proposal writing, reproduction, or deeper reading.

Use `/literature/compare` when you want to compare representative papers on research problems, methods, experiments, strengths, limitations, and follow-up ideas.

### External Literature Discovery

```text
POST /discovery/search
POST /discovery/import-metadata
```

Example search body:

```json
{
  "query": "graph neural networks for recommendation",
  "focus": "survey and benchmark papers",
  "sources": ["semantic_scholar", "crossref", "arxiv", "openalex"],
  "limit_per_source": 5
}
```

Use `/discovery/search` to find candidate papers outside the local library. Use `/discovery/import-metadata` to save a selected candidate into the local library as metadata before you upload the PDF.

### Literature Retrieval Evaluation

```text
POST /evaluation/literature
```

Example JSON body:

```json
{
  "top_k_documents": 5,
  "evidence_k": 18,
  "section_filter": null
}
```

This runs built-in evaluation cases and returns per-case matched terms, missing terms, score, papers, and evidence chunks.

## How V1 Works

```text
PDF upload
  -> pypdf extracts text
  -> text is split into overlapping chunks
  -> metadata and chunks are saved to data/index/rag_store.json
  -> sentence-transformers encodes chunks into vectors
  -> query is encoded into a vector
  -> cosine-like vector similarity retrieves relevant chunks
  -> optional LLM generates a cited answer or study workflow output
  -> API returns the answer and source chunks
```

## How V7 Works

```text
research direction
  -> retrieve more candidate chunks from the whole paper library
  -> group chunks by document_id
  -> score and rank paper-level candidates
  -> keep evidence from top papers
  -> optional LLM generates a review, method map, or detailed briefing
  -> frontend displays paper ranking, answer, and evidence chunks
```

## How V8 Works

```text
PDF upload or reindex
  -> extract first-page text
  -> infer title, authors, year, venue, DOI, abstract, and keywords
  -> mark likely duplicate papers by DOI, title, or filename
  -> persist metadata in data/index/rag_store.json
  -> API returns metadata with documents and literature candidates
  -> frontend renders paper cards instead of raw PDF filenames
```

## How V9 Works

```text
existing paper metadata with DOI
  -> call Crossref Works API by DOI
  -> map official title, authors, year, venue, publisher, URL, and reference count
  -> mark metadata_source as crossref
  -> persist enriched metadata in data/index/rag_store.json
  -> frontend displays the enriched source and bibliographic fields
```

## How V10 Works

```text
existing paper metadata
  -> try Crossref by DOI first when DOI is available
  -> if Crossref has no usable match, search Semantic Scholar by title
  -> compare the returned title with the local title
  -> accept only matches above the similarity threshold
  -> save source, confidence, citation count, fields of study, and match score
  -> frontend displays the enriched source and confidence information
```

## How V11 Works

```text
PDF upload or reindex
  -> extract page text
  -> split text into chunks
  -> infer a coarse section label from nearby headings
  -> persist section on each chunk in data/index/rag_store.json
  -> apply optional section_filter during retrieval ranking
  -> return section labels with source chunks and paper candidates
  -> frontend lets the user focus on Abstract, Methods, Experiments, Results, or Conclusion
```

## How V12 Works

```text
paper library filters
  -> frontend collects local filter values
  -> GET /documents receives query parameters
  -> backend filters the existing local paper metadata
  -> backend sorts the filtered documents
  -> frontend displays a smaller, more manageable paper list
```

## How V13 Works

```text
built-in evaluation cases
  -> run literature search for each case
  -> collect returned papers and evidence chunks
  -> compare retrieved evidence against expected terms
  -> compute per-case score and pass/fail
  -> compute average score across cases
  -> frontend displays the evaluation report
```

## How V14 Works

```text
research direction
  -> retrieve candidate chunks from the paper library
  -> group chunks into paper-level candidates
  -> rank representative papers
  -> build a comparison prompt from the selected evidence
  -> optional LLM writes a paper comparison
  -> frontend displays the comparison, paper ranking, and evidence chunks
```

## Next Milestones

1. Add safe open-access PDF download for discovery results.
2. Add BibTeX / Markdown export for selected papers.
3. Add paper status labels such as unread, reading, read, ignored, and priority.
4. Add external provider adapters for Elsevier, Scopus, and ScienceDirect when API keys are available.
