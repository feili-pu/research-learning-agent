from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile

from .answerer import Answerer
from .literature import LiteratureService
from .rag import RagStore
from .schemas import (
    DocumentIngestResponse,
    DocumentSummary,
    LiteratureRequest,
    LiteratureReviewResponse,
    LiteratureSearchResponse,
    QueryRequest,
    QueryResponse,
    SourceChunk,
    StudyRequest,
    StudyResponse,
)
from .study import StudyService

app = FastAPI(
    title="Research Learning Agent",
    description="Local RAG API for learning from uploaded PDFs.",
    version="0.7.0",
)

store = RagStore(upload_dir=Path("data/uploads"))
answerer = Answerer()
study_service = StudyService(store=store, answerer=answerer)
literature_service = LiteratureService(store=store, answerer=answerer)


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


@app.post("/documents/reindex", response_model=list[DocumentSummary])
def reindex_documents() -> list[DocumentSummary]:
    return [
        DocumentSummary(
            document_id=document.document_id,
            filename=document.filename,
            pages=document.pages,
            chunks=document.chunks,
        )
        for document in store.reindex_uploads()
    ]


@app.post("/query", response_model=QueryResponse)
def query_documents(request: QueryRequest) -> QueryResponse:
    results = store.search(request.question, request.top_k)
    answer = answerer.answer(request.question, results)

    return QueryResponse(
        question=request.question,
        retrieval_mode=store.active_retrieval_mode,
        answer_mode=answer.answer_mode,
        model=answer.model,
        answer=answer.answer,
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


@app.post("/study/summary", response_model=StudyResponse)
def study_summary(request: StudyRequest) -> StudyResponse:
    return study_service.summary(request)


@app.post("/study/key-points", response_model=StudyResponse)
def study_key_points(request: StudyRequest) -> StudyResponse:
    return study_service.key_points(request)


@app.post("/study/reading-plan", response_model=StudyResponse)
def study_reading_plan(request: StudyRequest) -> StudyResponse:
    return study_service.reading_plan(request)


@app.post("/literature/search", response_model=LiteratureSearchResponse)
def literature_search(request: LiteratureRequest) -> LiteratureSearchResponse:
    return literature_service.search(request)


@app.post("/literature/review", response_model=LiteratureReviewResponse)
def literature_review(request: LiteratureRequest) -> LiteratureReviewResponse:
    return literature_service.review(request)


@app.post("/literature/methods", response_model=LiteratureReviewResponse)
def literature_methods(request: LiteratureRequest) -> LiteratureReviewResponse:
    return literature_service.methods(request)


@app.post("/literature/details", response_model=LiteratureReviewResponse)
def literature_details(request: LiteratureRequest) -> LiteratureReviewResponse:
    return literature_service.details(request)
