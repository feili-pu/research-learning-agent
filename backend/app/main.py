from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile

from .answerer import Answerer
from .crossref import CrossrefClient
from .discovery import DiscoveryService
from .evaluation import EvaluationService
from .literature import LiteratureService
from .presenters import document_summary, paper_metadata, source_chunks
from .rag import RagStore
from .semantic_scholar import SemanticScholarClient
from .schemas import (
    DiscoveryImportRequest,
    DiscoveryImportResponse,
    DiscoveryRequest,
    DiscoveryResponse,
    DocumentBulkResponse,
    DocumentDeleteRequest,
    DocumentExportResponse,
    DocumentIngestResponse,
    DocumentSummary,
    LiteratureEvaluationRequest,
    LiteratureEvaluationResponse,
    LiteratureRequest,
    LiteratureReviewResponse,
    LiteratureSearchResponse,
    QueryRequest,
    QueryResponse,
    StudyRequest,
    StudyResponse,
)
from .study import StudyService

app = FastAPI(
    title="ScholarScope",
    description="Local RAG API for learning from uploaded PDFs.",
    version="0.16.0",
)

store: RagStore | None = None
answerer: Answerer | None = None
study_service: StudyService | None = None
literature_service: LiteratureService | None = None
evaluation_service: EvaluationService | None = None
discovery_service: DiscoveryService | None = None


def get_store() -> RagStore:
    global store
    if store is None:
        store = RagStore(upload_dir=Path("data/uploads"))
    return store


def get_answerer() -> Answerer:
    global answerer
    if answerer is None:
        answerer = Answerer()
    return answerer


def get_study_service() -> StudyService:
    global study_service
    current_store = get_store()
    current_answerer = get_answerer()
    if (
        study_service is None
        or study_service.store is not current_store
        or study_service.answerer is not current_answerer
    ):
        study_service = StudyService(store=current_store, answerer=current_answerer)
    return study_service


def get_literature_service() -> LiteratureService:
    global literature_service
    current_store = get_store()
    current_answerer = get_answerer()
    if (
        literature_service is None
        or literature_service.store is not current_store
        or literature_service.answerer is not current_answerer
    ):
        literature_service = LiteratureService(store=current_store, answerer=current_answerer)
    return literature_service


def get_evaluation_service() -> EvaluationService:
    global evaluation_service
    current_literature_service = get_literature_service()
    if (
        evaluation_service is None
        or evaluation_service.literature_service is not current_literature_service
    ):
        evaluation_service = EvaluationService(literature_service=current_literature_service)
    return evaluation_service


def get_discovery_service() -> DiscoveryService:
    global discovery_service
    current_answerer = get_answerer()
    if (
        discovery_service is None
        or discovery_service.answerer is not current_answerer
    ):
        discovery_service = DiscoveryService(store_provider=get_store, answerer=current_answerer)
    return discovery_service


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

    document = get_store().ingest_pdf(file.filename, content)
    return DocumentIngestResponse(
        document_id=document.document_id,
        filename=document.filename,
        pages=document.pages,
        chunks=document.chunks,
        metadata=paper_metadata(document.metadata),
    )


@app.get("/documents", response_model=list[DocumentSummary])
def list_documents(
    query: str | None = None,
    keyword: str | None = None,
    year_from: int | None = None,
    year_to: int | None = None,
    source: str | None = None,
    has_doi: bool | None = None,
    duplicate: bool | None = None,
    sort_by: str = "title",
) -> list[DocumentSummary]:
    return [
        document_summary(document)
        for document in get_store().filter_documents(
            query=query,
            keyword=keyword,
            year_from=year_from,
            year_to=year_to,
            source=source,
            has_doi=has_doi,
            duplicate=duplicate,
            sort_by=sort_by,
        )
    ]


@app.post("/documents/reindex", response_model=list[DocumentSummary])
def reindex_documents() -> list[DocumentSummary]:
    return [document_summary(document) for document in get_store().reindex_uploads()]


@app.post("/documents/enrich-metadata", response_model=list[DocumentSummary])
def enrich_document_metadata() -> list[DocumentSummary]:
    return [
        document_summary(document)
        for document in get_store().enrich_metadata(CrossrefClient(), SemanticScholarClient())
    ]


@app.post("/documents/delete", response_model=DocumentBulkResponse)
def delete_documents(request: DocumentDeleteRequest) -> DocumentBulkResponse:
    current_store = get_store()
    deleted_ids = current_store.delete_documents(request.document_ids)
    return DocumentBulkResponse(
        deleted_document_ids=deleted_ids,
        documents=[document_summary(document) for document in current_store.list_documents()],
    )


@app.post("/documents/merge-duplicates", response_model=DocumentBulkResponse)
def merge_duplicate_documents() -> DocumentBulkResponse:
    current_store = get_store()
    deleted_ids = current_store.merge_duplicates()
    return DocumentBulkResponse(
        deleted_document_ids=deleted_ids,
        documents=[document_summary(document) for document in current_store.list_documents()],
    )


@app.get("/documents/export", response_model=DocumentExportResponse)
def export_documents(format: str = "bibtex") -> DocumentExportResponse:
    normalized = (format or "bibtex").strip().lower()
    if normalized not in {"bibtex", "ris", "csv"}:
        raise HTTPException(status_code=400, detail="format must be one of bibtex, ris, or csv")
    content = get_store().export_documents(normalized)
    extension = "bib" if normalized == "bibtex" else normalized
    return DocumentExportResponse(
        format=normalized,
        filename=f"scholarscope-library.{extension}",
        content=content,
        document_count=len(get_store().list_documents()),
    )


@app.post("/query", response_model=QueryResponse)
def query_documents(request: QueryRequest) -> QueryResponse:
    current_store = get_store()
    results = current_store.search(request.question, request.top_k, request.section_filter)
    answer = get_answerer().answer(request.question, results)

    return QueryResponse(
        question=request.question,
        retrieval_mode=current_store.active_retrieval_mode,
        answer_mode=answer.answer_mode,
        model=answer.model,
        answer=answer.answer,
        sources=source_chunks(results),
    )


@app.post("/study/summary", response_model=StudyResponse)
def study_summary(request: StudyRequest) -> StudyResponse:
    return get_study_service().summary(request)


@app.post("/study/key-points", response_model=StudyResponse)
def study_key_points(request: StudyRequest) -> StudyResponse:
    return get_study_service().key_points(request)


@app.post("/study/reading-plan", response_model=StudyResponse)
def study_reading_plan(request: StudyRequest) -> StudyResponse:
    return get_study_service().reading_plan(request)


@app.post("/literature/search", response_model=LiteratureSearchResponse)
def literature_search(request: LiteratureRequest) -> LiteratureSearchResponse:
    return get_literature_service().search(request)


@app.post("/literature/review", response_model=LiteratureReviewResponse)
def literature_review(request: LiteratureRequest) -> LiteratureReviewResponse:
    return get_literature_service().review(request)


@app.post("/literature/methods", response_model=LiteratureReviewResponse)
def literature_methods(request: LiteratureRequest) -> LiteratureReviewResponse:
    return get_literature_service().methods(request)


@app.post("/literature/details", response_model=LiteratureReviewResponse)
def literature_details(request: LiteratureRequest) -> LiteratureReviewResponse:
    return get_literature_service().details(request)


@app.post("/literature/compare", response_model=LiteratureReviewResponse)
def literature_compare(request: LiteratureRequest) -> LiteratureReviewResponse:
    return get_literature_service().compare(request)


@app.post("/evaluation/literature", response_model=LiteratureEvaluationResponse)
def evaluate_literature(request: LiteratureEvaluationRequest) -> LiteratureEvaluationResponse:
    return get_evaluation_service().evaluate_literature(request)


@app.post("/discovery/search", response_model=DiscoveryResponse)
def discovery_search(request: DiscoveryRequest) -> DiscoveryResponse:
    return get_discovery_service().search(request)


@app.post("/discovery/import-metadata", response_model=DiscoveryImportResponse)
def discovery_import_metadata(request: DiscoveryImportRequest) -> DiscoveryImportResponse:
    document = get_discovery_service().import_metadata(request.paper)
    return DiscoveryImportResponse(
        document=document_summary(document),
        duplicate=bool(document.metadata.duplicate_of),
    )
