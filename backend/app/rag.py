from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4
import re

from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class Chunk:
    document_id: str
    filename: str
    page: int
    chunk_id: str
    text: str


@dataclass
class Document:
    document_id: str
    filename: str
    pages: int
    chunks: int


@dataclass
class SearchResult:
    chunk: Chunk
    score: float


class RagStore:
    def __init__(self, upload_dir: Path) -> None:
        self.upload_dir = upload_dir
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.documents: dict[str, Document] = {}
        self.chunks: list[Chunk] = []
        self.vectorizer = TfidfVectorizer()
        self.matrix = None

    def ingest_pdf(self, filename: str, content: bytes) -> Document:
        safe_name = self._safe_filename(filename)
        document_id = uuid4().hex
        stored_name = f"{document_id}-{safe_name}"
        pdf_path = self.upload_dir / stored_name
        pdf_path.write_bytes(content)

        pages = self._extract_pdf_pages(pdf_path)
        new_chunks = self._chunk_pages(document_id, safe_name, pages)
        document = Document(
            document_id=document_id,
            filename=safe_name,
            pages=len(pages),
            chunks=len(new_chunks),
        )

        self.documents[document_id] = document
        self.chunks.extend(new_chunks)
        self._rebuild_index()
        return document

    def list_documents(self) -> list[Document]:
        return list(self.documents.values())

    def search(self, question: str, top_k: int) -> list[SearchResult]:
        if not self.chunks or self.matrix is None:
            return []

        query_vector = self.vectorizer.transform([question])
        scores = cosine_similarity(query_vector, self.matrix)[0]
        ranked_indexes = scores.argsort()[::-1][:top_k]

        return [
            SearchResult(chunk=self.chunks[index], score=float(scores[index]))
            for index in ranked_indexes
            if scores[index] > 0
        ]

    def draft_answer(self, question: str, results: list[SearchResult]) -> str:
        if not results:
            return (
                "I could not find relevant content in the uploaded documents yet. "
                "Try uploading a PDF that contains this topic, or ask with more specific keywords."
            )

        excerpts = []
        for result in results:
            text = self._shorten(result.chunk.text, max_chars=500)
            excerpts.append(f"- Page {result.chunk.page}: {text}")

        joined = "\n".join(excerpts)
        return (
            "V1 uses retrieval-only answering, so this is a grounded draft from the most relevant chunks.\n\n"
            f"Question: {question}\n\n"
            f"Relevant evidence:\n{joined}"
        )

    def _rebuild_index(self) -> None:
        texts = [chunk.text for chunk in self.chunks]
        self.matrix = self.vectorizer.fit_transform(texts) if texts else None

    def _extract_pdf_pages(self, pdf_path: Path) -> list[str]:
        reader = PdfReader(str(pdf_path))
        pages = []
        for page in reader.pages:
            pages.append(page.extract_text() or "")
        return pages

    def _chunk_pages(
        self,
        document_id: str,
        filename: str,
        pages: list[str],
        chunk_size: int = 900,
        overlap: int = 150,
    ) -> list[Chunk]:
        chunks: list[Chunk] = []

        for page_number, page_text in enumerate(pages, start=1):
            normalized = self._normalize_text(page_text)
            if not normalized:
                continue

            start = 0
            part = 1
            while start < len(normalized):
                end = min(start + chunk_size, len(normalized))
                text = normalized[start:end].strip()
                if text:
                    chunks.append(
                        Chunk(
                            document_id=document_id,
                            filename=filename,
                            page=page_number,
                            chunk_id=f"{document_id}-p{page_number}-c{part}",
                            text=text,
                        )
                    )
                if end == len(normalized):
                    break
                start = max(end - overlap, start + 1)
                part += 1

        return chunks

    def _safe_filename(self, filename: str) -> str:
        name = Path(filename).name.strip() or "document.pdf"
        name = re.sub(r"[^A-Za-z0-9._-]+", "-", name)
        return name if name.lower().endswith(".pdf") else f"{name}.pdf"

    def _normalize_text(self, text: str) -> str:
        return re.sub(r"\s+", " ", text).strip()

    def _shorten(self, text: str, max_chars: int) -> str:
        if len(text) <= max_chars:
            return text
        return text[: max_chars - 3].rstrip() + "..."

