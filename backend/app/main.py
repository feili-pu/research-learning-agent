from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile

from .rag import RagStore
from .schemas import (
    DocumentIngestResponse,
    DocumentSummary,
    QueryRequest,
    QueryResponse,
    SourceChunk,
)

app = FastAPI(
    title="Research Learning Agent",
    description="V1 local RAG API for learning from uploaded PDFs.",
    version="0.1.0",
)

store = RagStore(upload_dir=Path("data/uploads"))


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/documents/upload", response_model=DocumentIngestResponse)
async def upload_document(file: UploadFile = File(...)) -> DocumentIngestResponse:
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Please upload a PDF file.")

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")

    document = store.ingest_pdf(file.filename, content)
    return DocumentIngestResponse(
        document_id=document.document_id,
        filename=document.filename,
        pages=document.pages,
        chunks=document.chunks,
    )


@app.get("/documents", response_model=list[DocumentSummary])
def list_documents() -> list[DocumentSummary]:
    return [
        DocumentSummary(
            document_id=document.document_id,
            filename=document.filename,
            pages=document.pages,
            chunks=document.chunks,
        )
        for document in store.list_documents()
    ]


@app.post("/query", response_model=QueryResponse)
def query_documents(request: QueryRequest) -> QueryResponse:
    results = store.search(request.question, request.top_k)
    answer = store.draft_answer(request.question, results)

    return QueryResponse(
        question=request.question,
        answer=answer,
        sources=[
            SourceChunk(
                document_id=result.chunk.document_id,
                filename=result.chunk.filename,
                page=result.chunk.page,
                chunk_id=result.chunk.chunk_id,
                score=result.score,
                text=result.chunk.text,
            )
            for result in results
        ],
    )
