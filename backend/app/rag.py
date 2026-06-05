from dataclasses import dataclass, field
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
class PaperMetadata:
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
    fields_of_study: list[str] = field(default_factory=list)
    metadata_confidence: str = "local"
    metadata_match_score: float | None = None
    metadata_source: str = "local"
    is_enriched: bool = False
    keywords: list[str] = field(default_factory=list)
    duplicate_of: str | None = None
    duplicate_reason: str | None = None


@dataclass
class Document:
    document_id: str
    filename: str
    pages: int
    chunks: int
    metadata: PaperMetadata = field(default_factory=PaperMetadata)


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
            metadata=self._extract_metadata(safe_name, pages),
        )

        self.documents[document_id] = document
        self.chunks.extend(new_chunks)
        self._mark_duplicates()
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
                metadata=self._extract_metadata(filename, pages),
            )
            self.chunks.extend(new_chunks)

        self._mark_duplicates()
        self._save_store()
        self._rebuild_index()
        return self.list_documents()

    def enrich_metadata(self, crossref_client, semantic_scholar_client=None) -> list[Document]:
        for document in self.documents.values():
            work = crossref_client.fetch_by_doi(document.metadata.doi) if document.metadata.doi else None
            if work is None:
                work = self._semantic_scholar_fallback(document.metadata, semantic_scholar_client)
            if work is not None:
                self._apply_external_metadata(document.metadata, work)

        self._mark_duplicates()
        self._save_store()
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
        self.documents = {}
        for item in data.get("documents", []):
            metadata = self._metadata_from_dict(item.get("metadata", {}))
            self.documents[item["document_id"]] = Document(
                document_id=item["document_id"],
                filename=item["filename"],
                pages=item["pages"],
                chunks=item["chunks"],
                metadata=metadata,
            )
        self.chunks = [Chunk(**item) for item in data.get("chunks", [])]
        self._mark_duplicates()

    def _save_store(self) -> None:
        data = {
            "documents": [
                {
                    "document_id": document.document_id,
                    "filename": document.filename,
                    "pages": document.pages,
                    "chunks": document.chunks,
                    "metadata": {
                        "title": document.metadata.title,
                        "authors": document.metadata.authors,
                        "year": document.metadata.year,
                        "venue": document.metadata.venue,
                        "doi": document.metadata.doi,
                        "abstract": document.metadata.abstract,
                        "publisher": document.metadata.publisher,
                        "external_url": document.metadata.external_url,
                        "reference_count": document.metadata.reference_count,
                        "citation_count": document.metadata.citation_count,
                        "fields_of_study": document.metadata.fields_of_study,
                        "metadata_confidence": document.metadata.metadata_confidence,
                        "metadata_match_score": document.metadata.metadata_match_score,
                        "metadata_source": document.metadata.metadata_source,
                        "is_enriched": document.metadata.is_enriched,
                        "keywords": document.metadata.keywords,
                        "duplicate_of": document.metadata.duplicate_of,
                        "duplicate_reason": document.metadata.duplicate_reason,
                    },
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

    def _extract_metadata(self, filename: str, pages: list[str]) -> PaperMetadata:
        first_pages = "\n".join(pages[:3])
        normalized = self._normalize_text(first_pages)
        title = self._extract_title(filename, first_pages, normalized)
        abstract = self._extract_abstract(normalized)
        return PaperMetadata(
            title=title,
            authors=self._extract_authors(first_pages),
            year=self._extract_year(normalized),
            venue=self._extract_venue(normalized),
            doi=self._extract_doi(normalized),
            abstract=abstract,
            keywords=self._extract_keywords(normalized),
        )

    def _apply_external_metadata(self, metadata: PaperMetadata, work) -> None:
        metadata.title = work.title or metadata.title
        metadata.authors = work.authors or metadata.authors
        metadata.year = work.year or metadata.year
        metadata.venue = work.venue or metadata.venue
        metadata.doi = work.doi or metadata.doi
        metadata.abstract = work.abstract or metadata.abstract
        metadata.publisher = work.publisher or metadata.publisher
        metadata.external_url = work.external_url or metadata.external_url
        metadata.reference_count = work.reference_count if work.reference_count is not None else metadata.reference_count
        metadata.citation_count = getattr(work, "citation_count", None) if getattr(work, "citation_count", None) is not None else metadata.citation_count
        metadata.fields_of_study = getattr(work, "fields_of_study", []) or metadata.fields_of_study
        metadata.keywords = work.keywords or metadata.keywords
        metadata.metadata_source = "semantic_scholar" if hasattr(work, "match_score") else "crossref"
        metadata.metadata_confidence = getattr(work, "confidence", "high" if metadata.metadata_source == "crossref" else metadata.metadata_confidence)
        metadata.metadata_match_score = getattr(work, "match_score", metadata.metadata_match_score)
        metadata.is_enriched = True

    def _semantic_scholar_fallback(self, metadata: PaperMetadata, client):
        if client is None or not metadata.title:
            return None
        return client.search_by_title(metadata.title)

    def _extract_title(self, filename: str, raw_text: str, normalized: str) -> str:
        lines = [self._normalize_text(line) for line in raw_text.splitlines()]
        for index, line in enumerate(lines):
            if self._looks_like_title(line, lines, index):
                title_lines = [line]
                for next_line in lines[index + 1 : index + 4]:
                    if self._looks_like_title_continuation(next_line):
                        title_lines.append(next_line)
                    else:
                        break
                return self._clean_title(" ".join(title_lines))

        abstract_match = re.search(r"(.{12,220}?)\bAbstract\b", normalized, flags=re.IGNORECASE)
        if abstract_match:
            candidate = abstract_match.group(1)
            candidate = re.sub(r"^(article|research article|contents lists available at sciencedirect)\b", "", candidate, flags=re.IGNORECASE)
            title = self._clean_title(candidate)
            if title:
                return title

        return Path(filename).stem

    def _extract_authors(self, raw_text: str) -> str | None:
        lines = [self._normalize_text(line) for line in raw_text.splitlines() if self._normalize_text(line)]
        for index, line in enumerate(lines[:12]):
            if self._looks_like_title(line, lines, index):
                author_lines = []
                for candidate in lines[index + 1 : index + 10]:
                    if self._looks_like_title_continuation(candidate):
                        continue
                    if re.search(r"college|university|institute|laboratory|department|school|academy", candidate, re.IGNORECASE):
                        break
                    if re.search(r"\b(Abstract|Introduction|Keywords|ARTICLE INFO|Contents lists|journal homepage)\b", candidate, re.IGNORECASE):
                        break
                    if re.match(r"^[a-z]\s*$|^[a-z]\s*,", candidate):
                        continue
                    if 2 <= len(candidate) <= 180:
                        author_lines.append(candidate)
                    if len(author_lines) >= 4:
                        break
                return self._clean_authors(" ".join(author_lines)) if author_lines else None
        return None

    def _extract_year(self, text: str) -> int | None:
        matches = re.findall(r"\b(19[8-9]\d|20[0-4]\d)\b", text)
        if not matches:
            return None
        return int(matches[0])

    def _extract_venue(self, text: str) -> str | None:
        patterns = [
            r"journal homep?\s*age:\s*www\.\s*elsevier\.com/\s*locate/([A-Za-z0-9_-]+)",
            r"\b([A-Z][A-Za-z& ]{3,60})\s+\d{1,4}\s*\(\d{4}\)",
        ]
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return self._normalize_text(match.group(1))
        return None

    def _extract_doi(self, text: str) -> str | None:
        match = re.search(r"\b10\.\d{4,9}/[-._;()/:A-Za-z0-9]+", text)
        if not match:
            return None
        return match.group(0).rstrip(" .;,)")

    def _extract_abstract(self, text: str) -> str | None:
        match = re.search(
            r"\bAbstract\b[:.\s-]*(.*?)(?:\bKeywords\b|\b1\.\s*Introduction\b|\bIntroduction\b)",
            text,
            flags=re.IGNORECASE,
        )
        if not match:
            return None
        abstract = self._normalize_text(match.group(1))
        return self._shorten(abstract, max_chars=900) if abstract else None

    def _extract_keywords(self, text: str) -> list[str]:
        match = re.search(r"\bKeywords\b[:.\s-]*(.*?)(?:\b1\.\s*Introduction\b|\bIntroduction\b)", text, flags=re.IGNORECASE)
        if not match:
            return []
        raw_keywords = re.split(r"[;,|]", match.group(1))
        return [keyword.strip() for keyword in raw_keywords if 2 <= len(keyword.strip()) <= 60][:12]

    def _looks_like_title(self, line: str, lines: list[str] | None = None, index: int = 0) -> bool:
        if not 12 <= len(line) <= 220:
            return False
        blocked = [
            "abstract",
            "introduction",
            "keywords",
            "contents lists available",
            "journal homepage",
            "science direct",
            "research papers",
            "received",
            "accepted",
            "available online",
        ]
        lower = line.lower()
        if any(item in lower for item in blocked):
            return False
        if re.search(r"\b\d{1,4}\s*\(\d{4}\)\s*\d+", line):
            return False
        if re.search(r"\b(doi|https?://|www\.)\b", lower):
            return False
        if re.search(r"\b\d{4}-\d{4}\b|©|all rights reserved|elsevier", lower):
            return False
        if lines:
            nearby = " ".join(lines[max(0, index - 2) : min(len(lines), index + 3)]).lower()
            if "journal homepage" in nearby and len(line.split()) <= 4:
                return False
        return sum(char.isalpha() for char in line) >= 8

    def _looks_like_title_continuation(self, line: str) -> bool:
        if not 8 <= len(line) <= 160:
            return False
        lower = line.lower()
        if any(item in lower for item in ["abstract", "keywords", "article info", "a r t i c l e", "a b s t r a c t"]):
            return False
        if re.match(r"^[a-z]\s+|^[a-z]\s*,|^[a-z]\s*$", lower):
            return False
        if re.match(r"^([A-Z][A-Za-z.-]+(\s+[A-Z][A-Za-z.-]+){0,3}\s*,\s*)+[A-Z][A-Za-z.-]+", line):
            return False
        if re.match(r"^[A-Z][A-Za-z.-]+(?:\s+[A-Z][A-Za-z.-]+){1,3}$", line):
            return False
        if re.search(r"college|university|institute|laboratory|department|school|academy", lower):
            return False
        if line.count(",") >= 2 or "*" in line or "∗" in line:
            return False
        if re.search(r"\b\d{4}-\d{4}\b|©|all rights reserved|elsevier|journal homepage|available online", lower):
            return False
        return sum(char.isalpha() for char in line) >= 8

    def _clean_title(self, title: str) -> str:
        title = self._normalize_text(title)
        title = re.sub(r"^[^A-Za-z0-9]+", "", title)
        title = re.sub(r"\s+", " ", title)
        title = title.strip(" -:;,.")
        return self._shorten(title, max_chars=220)

    def _clean_authors(self, authors: str) -> str:
        authors = self._normalize_text(authors)
        authors = re.sub(r"\s+", " ", authors)
        authors = authors.strip(" ,;")
        return self._shorten(authors, max_chars=260)

    def _metadata_from_dict(self, data: dict) -> PaperMetadata:
        return PaperMetadata(
            title=data.get("title"),
            authors=data.get("authors"),
            year=data.get("year"),
            venue=data.get("venue"),
            doi=data.get("doi"),
            abstract=data.get("abstract"),
            publisher=data.get("publisher"),
            external_url=data.get("external_url"),
            reference_count=data.get("reference_count"),
            citation_count=data.get("citation_count"),
            fields_of_study=list(data.get("fields_of_study", [])),
            metadata_confidence=data.get("metadata_confidence", "local"),
            metadata_match_score=data.get("metadata_match_score"),
            metadata_source=data.get("metadata_source", "local"),
            is_enriched=bool(data.get("is_enriched", False)),
            keywords=list(data.get("keywords", [])),
            duplicate_of=data.get("duplicate_of"),
            duplicate_reason=data.get("duplicate_reason"),
        )

    def _mark_duplicates(self) -> None:
        seen_doi: dict[str, str] = {}
        seen_title: dict[str, str] = {}
        seen_filename: dict[str, str] = {}

        for document in self.documents.values():
            document.metadata.duplicate_of = None
            document.metadata.duplicate_reason = None

            doi_key = document.metadata.doi.lower() if document.metadata.doi else None
            title_key = self._normalize_identity(document.metadata.title)
            filename_key = document.filename.lower()

            duplicate_of = None
            reason = None
            if doi_key and doi_key in seen_doi:
                duplicate_of = seen_doi[doi_key]
                reason = "same_doi"
            elif title_key and title_key in seen_title:
                duplicate_of = seen_title[title_key]
                reason = "same_title"
            elif filename_key in seen_filename:
                duplicate_of = seen_filename[filename_key]
                reason = "same_filename"

            if duplicate_of:
                document.metadata.duplicate_of = duplicate_of
                document.metadata.duplicate_reason = reason

            if doi_key and doi_key not in seen_doi:
                seen_doi[doi_key] = document.document_id
            if title_key and title_key not in seen_title:
                seen_title[title_key] = document.document_id
            if filename_key not in seen_filename:
                seen_filename[filename_key] = document.document_id

    def _normalize_identity(self, value: str | None) -> str | None:
        if not value:
            return None
        normalized = re.sub(r"[^a-z0-9]+", "", value.lower())
        return normalized if len(normalized) >= 12 else None
