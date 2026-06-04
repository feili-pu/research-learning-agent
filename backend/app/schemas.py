from pydantic import BaseModel, Field


class DocumentIngestResponse(BaseModel):
    document_id: str
    filename: str
    pages: int
    chunks: int


class QueryRequest(BaseModel):
    question: str = Field(min_length=1)
    top_k: int = Field(default=4, ge=1, le=10)


class SourceChunk(BaseModel):
    document_id: str
    filename: str
    page: int
    chunk_id: str
    score: float
    text: str


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


class StudyRequest(BaseModel):
    topic: str = Field(default="这些文档", min_length=1)
    focus: str | None = None
    top_k: int = Field(default=6, ge=1, le=12)


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


class PaperCandidate(BaseModel):
    document_id: str
    filename: str
    pages: int
    chunks: int
    score: float
    evidence_count: int
    evidence_pages: list[int]
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
