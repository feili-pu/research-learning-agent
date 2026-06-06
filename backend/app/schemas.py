from pydantic import BaseModel, Field


class PaperMetadata(BaseModel):
    title: str | None = None
    authors: str | None = None
    year: int | None = None
    venue: str | None = None
    doi: str | None = None
    abstract: str | None = None
    publisher: str | None = None
    external_url: str | None = None
    reference_count: int | None = None
    citation_count: int | None = None
    fields_of_study: list[str] = Field(default_factory=list)
    metadata_confidence: str = "local"
    metadata_match_score: float | None = None
    metadata_source: str = "local"
    is_enriched: bool = False
    keywords: list[str] = Field(default_factory=list)
    duplicate_of: str | None = None
    duplicate_reason: str | None = None


class DocumentIngestResponse(BaseModel):
    document_id: str
    filename: str
    pages: int
    chunks: int
    metadata: PaperMetadata


class QueryRequest(BaseModel):
    question: str = Field(min_length=1)
    top_k: int = Field(default=4, ge=1, le=10)
    section_filter: str | None = None


class SourceChunk(BaseModel):
    document_id: str
    filename: str
    page: int
    chunk_id: str
    score: float
    text: str
    section: str = "unknown"


class QueryResponse(BaseModel):
    question: str
    retrieval_mode: str
    answer_mode: str
    model: str | None
    answer: str
    sources: list[SourceChunk]


class DocumentSummary(BaseModel):
    document_id: str
    filename: str
    pages: int
    chunks: int
    metadata: "PaperMetadata"


class StudyRequest(BaseModel):
    topic: str = Field(default="这些文档", min_length=1)
    focus: str | None = None
    top_k: int = Field(default=6, ge=1, le=12)
    section_filter: str | None = None


class StudyResponse(BaseModel):
    task: str
    topic: str
    retrieval_mode: str
    answer_mode: str
    model: str | None
    answer: str
    sources: list[SourceChunk]


class LiteratureRequest(BaseModel):
    query: str = Field(min_length=1)
    focus: str | None = None
    top_k_documents: int = Field(default=5, ge=1, le=10)
    evidence_k: int = Field(default=18, ge=3, le=40)
    section_filter: str | None = None


class PaperCandidate(BaseModel):
    document_id: str
    filename: str
    pages: int
    chunks: int
    metadata: PaperMetadata
    score: float
    evidence_count: int
    evidence_pages: list[int]
    evidence_sections: list[str] = Field(default_factory=list)
    preview: str


class LiteratureSearchResponse(BaseModel):
    query: str
    retrieval_mode: str
    papers: list[PaperCandidate]
    sources: list[SourceChunk]


class LiteratureReviewResponse(BaseModel):
    task: str
    query: str
    retrieval_mode: str
    answer_mode: str
    model: str | None
    answer: str
    papers: list[PaperCandidate]
    sources: list[SourceChunk]


class LiteratureEvaluationRequest(BaseModel):
    top_k_documents: int = Field(default=5, ge=1, le=10)
    evidence_k: int = Field(default=18, ge=3, le=40)
    section_filter: str | None = None


class EvaluationCaseResult(BaseModel):
    name: str
    query: str
    focus: str | None
    section_filter: str | None
    expected_terms: list[str]
    matched_terms: list[str]
    missing_terms: list[str]
    score: float
    passed: bool
    papers: list[PaperCandidate]
    sources: list[SourceChunk]


class LiteratureEvaluationResponse(BaseModel):
    retrieval_mode: str
    total_cases: int
    passed_cases: int
    average_score: float
    cases: list[EvaluationCaseResult]


class DiscoveryRequest(BaseModel):
    query: str = Field(min_length=1)
    focus: str | None = None
    sources: list[str] = Field(default_factory=lambda: ["semantic_scholar", "openalex"])
    limit_per_source: int = Field(default=3, ge=1, le=20)


class DiscoveryPaper(BaseModel):
    source: str
    source_id: str | None = None
    title: str
    authors: str | None = None
    year: int | None = None
    venue: str | None = None
    doi: str | None = None
    abstract: str | None = None
    external_url: str | None = None
    pdf_url: str | None = None
    reference_count: int | None = None
    citation_count: int | None = None
    fields_of_study: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    is_open_access: bool = False
    relevance_score: float = 0.0
    imported_document_id: str | None = None


class DiscoveryResponse(BaseModel):
    query: str
    focus: str | None
    sources: list[str]
    queries_used: list[str] = Field(default_factory=list)
    papers: list[DiscoveryPaper]
    errors: list[str] = Field(default_factory=list)


class DiscoveryImportRequest(BaseModel):
    paper: DiscoveryPaper


class DiscoveryImportResponse(BaseModel):
    document: DocumentSummary
    duplicate: bool = False
