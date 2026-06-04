from dataclasses import dataclass
import json
from pathlib import Path
from uuid import uuid4
import re

import numpy as np
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None


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
    def __init__(
        self,
        upload_dir: Path,
        index_dir: Path = Path("data/index"),
        retrieval_mode: str = "semantic",
        embedding_model_name: str = "BAAI/bge-small-zh-v1.5",
    ) -> None:
        self.upload_dir = upload_dir
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.index_dir = index_dir
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.store_path = self.index_dir / "rag_store.json"
        self.documents: dict[str, Document] = {}
        self.chunks: list[Chunk] = []
        self.retrieval_mode = retrieval_mode
        self.active_retrieval_mode = "tfidf"
        self.embedding_model_name = embedding_model_name
        self.embedding_model = None
        self.embedding_matrix = None
        self.vectorizer = TfidfVectorizer()
        self.tfidf_matrix = None
        self._load_store()
        self._rebuild_index()

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
        self._save_store()
        self._rebuild_index()
        return document

    def list_documents(self) -> list[Document]:
        return list(self.documents.values())

    def reindex_uploads(self) -> list[Document]:
        self.documents = {}
        self.chunks = []

        for pdf_path in sorted(self.upload_dir.glob("*.pdf")):
            document_id, filename = self._document_info_from_path(pdf_path)
            pages = self._extract_pdf_pages(pdf_path)
            new_chunks = self._chunk_pages(document_id, filename, pages)
            self.documents[document_id] = Document(
                document_id=document_id,
                filename=filename,
                pages=len(pages),
                chunks=len(new_chunks),
            )
            self.chunks.extend(new_chunks)

        self._save_store()
        self._rebuild_index()
        return self.list_documents()

    def search(self, question: str, top_k: int) -> list[SearchResult]:
        if not self.chunks:
            return []

        if self.active_retrieval_mode == "semantic" and self.embedding_matrix is not None:
            scores = self._semantic_scores(question)
        elif self.tfidf_matrix is not None:
            scores = self._tfidf_scores(question)
        else:
            return []

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
            "V2 uses retrieval-only answering, so this is a grounded draft from the most relevant chunks.\n\n"
            f"Question: {question}\n\n"
            f"Relevant evidence:\n{joined}"
        )

    def _rebuild_index(self) -> None:
        texts = [chunk.text for chunk in self.chunks]
        self.tfidf_matrix = self.vectorizer.fit_transform(texts) if texts else None
        self.embedding_matrix = None
        self.active_retrieval_mode = "tfidf"

        if texts and self.retrieval_mode == "semantic":
            self.embedding_matrix = self._build_embedding_matrix(texts)
            if self.embedding_matrix is not None:
                self.active_retrieval_mode = "semantic"

    def _build_embedding_matrix(self, texts: list[str]):
        if SentenceTransformer is None:
            return None

        if self.embedding_model is None:
            self.embedding_model = SentenceTransformer(self.embedding_model_name)

        return self.embedding_model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

    def _semantic_scores(self, question: str):
        query_embedding = self.embedding_model.encode(
            [question],
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]
        return np.matmul(self.embedding_matrix, query_embedding)

    def _tfidf_scores(self, question: str):
        query_vector = self.vectorizer.transform([question])
        return cosine_similarity(query_vector, self.tfidf_matrix)[0]

    def _load_store(self) -> None:
        if not self.store_path.exists():
            return

        data = json.loads(self.store_path.read_text(encoding="utf-8"))
        self.documents = {
            item["document_id"]: Document(**item)
            for item in data.get("documents", [])
        }
        self.chunks = [Chunk(**item) for item in data.get("chunks", [])]

    def _save_store(self) -> None:
        data = {
            "documents": [
                {
                    "document_id": document.document_id,
                    "filename": document.filename,
                    "pages": document.pages,
                    "chunks": document.chunks,
                }
                for document in self.documents.values()
            ],
            "chunks": [
                {
                    "document_id": chunk.document_id,
                    "filename": chunk.filename,
                    "page": chunk.page,
                    "chunk_id": chunk.chunk_id,
                    "text": chunk.text,
                }
                for chunk in self.chunks
            ],
        }
        self.store_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

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

    def _document_info_from_path(self, pdf_path: Path) -> tuple[str, str]:
        name = pdf_path.name
        match = re.match(r"^([a-f0-9]{32})-(.+)$", name)
        if match:
            return match.group(1), match.group(2)
        return uuid4().hex, self._safe_filename(name)

    def _normalize_text(self, text: str) -> str:
        return re.sub(r"\s+", " ", text).strip()

    def _shorten(self, text: str, max_chars: int) -> str:
        if len(text) <= max_chars:
            return text
        return text[: max_chars - 3].rstrip() + "..."
