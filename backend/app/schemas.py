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
    answer: str
    sources: list[SourceChunk]


class DocumentSummary(BaseModel):
    document_id: str
    filename: str
    pages: int
    chunks: int

