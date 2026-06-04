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
