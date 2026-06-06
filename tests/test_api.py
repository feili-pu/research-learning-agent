from pathlib import Path

from fastapi.testclient import TestClient
import numpy as np

from backend.app.answerer import Answerer
from backend.app import main
from backend.app.crossref import CrossrefWork
from backend.app.discovery import DiscoveryService, ProviderResult
from backend.app.evaluation import EvaluationService
from backend.app.literature import LiteratureService
from backend.app.rag import Chunk, RagStore
from backend.app.semantic_scholar import SemanticScholarClient, SemanticScholarWork
from backend.app.schemas import LiteratureRequest, StudyRequest
from backend.app.study import StudyService


def make_pdf_with_text(text: str) -> bytes:
    stream = f"BT /F1 18 Tf 72 720 Td ({text}) Tj ET".encode("latin-1")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Length " + str(len(stream)).encode("ascii") + b" >>\nstream\n" + stream + b"\nendstream",
    ]

    pdf = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for index, obj in enumerate(objects, start=1):
        offsets.append(len(pdf))
        pdf.extend(f"{index} 0 obj\n".encode("ascii"))
        pdf.extend(obj)
        pdf.extend(b"\nendobj\n")

    xref_at = len(pdf)
    pdf.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    pdf.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        pdf.extend(f"{offset:010d} 00000 n \n".encode("ascii"))

    pdf.extend(
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
        f"startxref\n{xref_at}\n%%EOF\n".encode("ascii")
    )
    return bytes(pdf)


def test_health_check() -> None:
    client = TestClient(main.app)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_upload_and_query_pdf(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    main.store = RagStore(upload_dir=tmp_path / "uploads", index_dir=tmp_path / "index", retrieval_mode="tfidf")
    main.answerer = Answerer()
    client = TestClient(main.app)
    pdf_bytes = make_pdf_with_text(
        "Retrieval augmented generation uses search to ground answers in documents."
    )

    upload_response = client.post(
        "/documents/upload",
        files={"file": ("rag.pdf", pdf_bytes, "application/pdf")},
    )
    query_response = client.post(
        "/query",
        json={"question": "What does retrieval augmented generation use?", "top_k": 2},
    )

    assert upload_response.status_code == 200
    assert upload_response.json()["chunks"] >= 1
    assert query_response.status_code == 200
    assert query_response.json()["retrieval_mode"] == "tfidf"
    assert query_response.json()["answer_mode"] == "retrieval_only"
    assert query_response.json()["model"] is None
    assert query_response.json()["sources"]
    assert "section" in query_response.json()["sources"][0]


def test_semantic_store_can_build_index(monkeypatch, tmp_path: Path) -> None:
    class FakeSentenceTransformer:
        def __init__(self, model_name: str) -> None:
            self.model_name = model_name

        def encode(self, texts, normalize_embeddings=True, show_progress_bar=False):
            return np.array([[1.0, 0.0] for _ in texts])

    monkeypatch.setattr("backend.app.rag._load_sentence_transformer", lambda: FakeSentenceTransformer)
    store = RagStore(upload_dir=tmp_path / "uploads", index_dir=tmp_path / "index", retrieval_mode="semantic")
    pdf_bytes = make_pdf_with_text(
        "Retrieval augmented generation grounds model answers with search results."
    )

    store.ingest_pdf("semantic.pdf", pdf_bytes)
    results = store.search("How does RAG use search?", top_k=1)

    assert store.active_retrieval_mode == "semantic"
    assert results


def test_search_can_filter_by_section(tmp_path: Path) -> None:
    store = RagStore(upload_dir=tmp_path / "uploads", index_dir=tmp_path / "index", retrieval_mode="tfidf")
    store.chunks = [
        Chunk(
            document_id="paper-1",
            filename="paper.pdf",
            page=1,
            chunk_id="paper-1-abstract",
            text="Abstract graph retrieval overview and motivation.",
            section="abstract",
        ),
        Chunk(
            document_id="paper-1",
            filename="paper.pdf",
            page=2,
            chunk_id="paper-1-methods",
            text="Methods graph neural network training and reranking pipeline.",
            section="methods",
        ),
    ]
    store._rebuild_index()

    results = store.search("graph neural network training", top_k=2, section_filter="methods")

    assert len(results) == 1
    assert results[0].chunk.chunk_id == "paper-1-methods"
    assert results[0].chunk.section == "methods"


def test_answerer_returns_retrieval_only_without_api_key(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    store = RagStore(upload_dir=tmp_path / "uploads", index_dir=tmp_path / "index", retrieval_mode="tfidf")
    pdf_bytes = make_pdf_with_text("RAG answers should cite retrieved source chunks.")
    store.ingest_pdf("answer.pdf", pdf_bytes)
    results = store.search("What should RAG cite?", top_k=1)

    answer = Answerer().answer("What should RAG cite?", results)

    assert answer.answer_mode == "retrieval_only"
    assert answer.model is None
    assert "Sources:" in answer.answer


def test_answerer_falls_back_when_llm_fails(monkeypatch, tmp_path: Path) -> None:
    class BrokenResponses:
        def create(self, **kwargs):
            raise RuntimeError("fake failure")

    class BrokenClient:
        responses = BrokenResponses()

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    store = RagStore(upload_dir=tmp_path / "uploads", index_dir=tmp_path / "index", retrieval_mode="tfidf")
    pdf_bytes = make_pdf_with_text("The answer must stay grounded in retrieved chunks.")
    store.ingest_pdf("fallback.pdf", pdf_bytes)
    results = store.search("Where should the answer stay grounded?", top_k=1)
    answerer = Answerer()
    answerer.client = BrokenClient()

    answer = answerer.answer("Where should the answer stay grounded?", results)

    assert answer.answer_mode == "llm_error_fallback"
    assert answer.model is None
    assert "Error type: RuntimeError" in answer.answer
    assert "LLM generation failed" in answer.answer
    assert "Sources:" in answer.answer


def test_answerer_uses_chat_completions(monkeypatch, tmp_path: Path) -> None:
    class Message:
        content = "Grounded answer [1]"

    class Choice:
        message = Message()

    class Response:
        choices = [Choice()]

    class FakeCompletions:
        def __init__(self):
            self.kwargs = None

        def create(self, **kwargs):
            self.kwargs = kwargs
            return Response()

    class FakeChat:
        def __init__(self):
            self.completions = FakeCompletions()

    class FakeClient:
        def __init__(self):
            self.chat = FakeChat()

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("RLA_OPENAI_WIRE_API", "chat")
    store = RagStore(upload_dir=tmp_path / "uploads", index_dir=tmp_path / "index", retrieval_mode="tfidf")
    pdf_bytes = make_pdf_with_text("Chat completions should answer with source citations.")
    store.ingest_pdf("chat.pdf", pdf_bytes)
    results = store.search("What should the answer include?", top_k=1)
    answerer = Answerer()
    fake_client = FakeClient()
    answerer.client = fake_client

    answer = answerer.answer("What should the answer include?", results)

    assert answer.answer_mode == "llm"
    assert answer.answer == "Grounded answer [1]"
    assert fake_client.chat.completions.kwargs["model"] == answerer.model
    assert fake_client.chat.completions.kwargs["messages"][0]["role"] == "system"


def test_answerer_uses_responses_api_by_default(monkeypatch, tmp_path: Path) -> None:
    class Response:
        output_text = "Grounded response answer [1]"

    class FakeResponses:
        def __init__(self):
            self.kwargs = None

        def create(self, **kwargs):
            self.kwargs = kwargs
            return Response()

    class FakeClient:
        def __init__(self):
            self.responses = FakeResponses()

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.delenv("RLA_OPENAI_WIRE_API", raising=False)
    store = RagStore(upload_dir=tmp_path / "uploads", index_dir=tmp_path / "index", retrieval_mode="tfidf")
    pdf_bytes = make_pdf_with_text("Responses API should answer with source citations.")
    store.ingest_pdf("responses.pdf", pdf_bytes)
    results = store.search("What should the answer include?", top_k=1)
    answerer = Answerer()
    fake_client = FakeClient()
    answerer.client = fake_client

    answer = answerer.answer("What should the answer include?", results)

    assert answer.answer_mode == "llm"
    assert answer.answer == "Grounded response answer [1]"
    assert fake_client.responses.kwargs["model"] == answerer.model
    assert fake_client.responses.kwargs["input"][0]["role"] == "system"


def test_store_persists_and_loads_chunks(tmp_path: Path) -> None:
    upload_dir = tmp_path / "uploads"
    index_dir = tmp_path / "index"
    first_store = RagStore(upload_dir=upload_dir, index_dir=index_dir, retrieval_mode="tfidf")
    pdf_bytes = make_pdf_with_text("Persistent indexes keep uploaded document chunks after restart.")

    first_store.ingest_pdf("persist.pdf", pdf_bytes)
    second_store = RagStore(upload_dir=upload_dir, index_dir=index_dir, retrieval_mode="tfidf")
    results = second_store.search("What keeps chunks after restart?", top_k=1)

    assert (index_dir / "rag_store.json").exists()
    assert len(second_store.list_documents()) == 1
    assert results


def test_reindex_uploads_rebuilds_store(tmp_path: Path) -> None:
    upload_dir = tmp_path / "uploads"
    index_dir = tmp_path / "index"
    upload_dir.mkdir(parents=True)
    (upload_dir / "manual.pdf").write_bytes(
        make_pdf_with_text("Manual uploads can be reindexed into the local store.")
    )
    store = RagStore(upload_dir=upload_dir, index_dir=index_dir, retrieval_mode="tfidf")

    documents = store.reindex_uploads()
    results = store.search("What can be reindexed?", top_k=1)

    assert len(documents) == 1
    assert documents[0].filename == "manual.pdf"
    assert results


def test_metadata_extraction_finds_core_fields(tmp_path: Path) -> None:
    store = RagStore(upload_dir=tmp_path / "uploads", index_dir=tmp_path / "index", retrieval_mode="tfidf")
    pages = [
        """
        A Reliable Method for Water Quality Prediction
        Alice Zhang, Bob Li
        Journal of Environmental Intelligence 42 (2025) 100-120
        https://doi.org/10.1016/j.example.2025.100120
        Abstract This paper proposes a reliable method for water quality prediction using neural networks.
        Keywords water quality; neural networks; prediction
        1. Introduction Water quality prediction matters.
        """
    ]

    metadata = store._extract_metadata("paper.pdf", pages)

    assert metadata.title == "A Reliable Method for Water Quality Prediction"
    assert metadata.authors == "Alice Zhang, Bob Li"
    assert metadata.year == 2025
    assert metadata.doi == "10.1016/j.example.2025.100120"
    assert metadata.abstract.startswith("This paper proposes")
    assert "water quality" in metadata.keywords


def test_duplicate_detection_marks_same_doi(tmp_path: Path) -> None:
    store = RagStore(upload_dir=tmp_path / "uploads", index_dir=tmp_path / "index", retrieval_mode="tfidf")
    first = store.ingest_pdf(
        "first.pdf",
        make_pdf_with_text("First Study Abstract duplicate doi 10.1234/example.2025.001 Keywords test Introduction text."),
    )
    second = store.ingest_pdf(
        "second.pdf",
        make_pdf_with_text("Second Study Abstract duplicate doi 10.1234/example.2025.001 Keywords test Introduction text."),
    )

    assert store.documents[first.document_id].metadata.duplicate_of is None
    assert store.documents[second.document_id].metadata.duplicate_of == first.document_id
    assert store.documents[second.document_id].metadata.duplicate_reason == "same_doi"


def test_documents_endpoint_returns_metadata(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    main.store = RagStore(upload_dir=tmp_path / "uploads", index_dir=tmp_path / "index", retrieval_mode="tfidf")
    main.answerer = Answerer()
    client = TestClient(main.app)
    pdf_bytes = make_pdf_with_text(
        "Metadata Test Paper Abstract This paper has metadata. Keywords rag Introduction content."
    )
    client.post(
        "/documents/upload",
        files={"file": ("metadata.pdf", pdf_bytes, "application/pdf")},
    )

    response = client.get("/documents")

    assert response.status_code == 200
    data = response.json()
    assert data[0]["metadata"]["title"]
    assert "duplicate_of" in data[0]["metadata"]


def test_documents_endpoint_filters_library(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    main.store = RagStore(upload_dir=tmp_path / "uploads", index_dir=tmp_path / "index", retrieval_mode="tfidf")
    main.answerer = Answerer()
    first = main.store.ingest_pdf(
        "water-quality.pdf",
        make_pdf_with_text("Water Quality Retrieval Abstract remote sensing neural model Keywords water Introduction text."),
    )
    first.metadata.title = "Water Quality Retrieval With Remote Sensing"
    first.metadata.authors = "Alice Zhang"
    first.metadata.year = 2025
    first.metadata.doi = "10.1234/water.2025"
    first.metadata.metadata_source = "crossref"
    first.metadata.keywords = ["water quality", "remote sensing"]

    second = main.store.ingest_pdf(
        "biometric.pdf",
        make_pdf_with_text("Biometric Template Protection Abstract face security Keywords biometric Introduction text."),
    )
    second.metadata.title = "Biometric Template Protection"
    second.metadata.authors = "Bob Li"
    second.metadata.year = 2023
    second.metadata.metadata_source = "local"
    second.metadata.keywords = ["biometric", "security"]

    duplicate = main.store.ingest_pdf(
        "water-copy.pdf",
        make_pdf_with_text("Water Quality Duplicate Abstract same study Keywords water Introduction text."),
    )
    duplicate.metadata.title = "Water Quality Retrieval Copy"
    duplicate.metadata.year = 2025
    duplicate.metadata.doi = "10.1234/water.2025"
    duplicate.metadata.metadata_source = "crossref"
    duplicate.metadata.keywords = ["water quality"]
    duplicate.metadata.duplicate_of = first.document_id
    duplicate.metadata.duplicate_reason = "same_doi"

    client = TestClient(main.app)

    filtered = client.get(
        "/documents",
        params={
            "query": "water",
            "keyword": "remote",
            "year_from": 2024,
            "source": "crossref",
            "has_doi": "true",
            "duplicate": "false",
            "sort_by": "year_desc",
        },
    )
    duplicates = client.get("/documents", params={"duplicate": "true"})

    assert filtered.status_code == 200
    assert [item["document_id"] for item in filtered.json()] == [first.document_id]
    assert duplicates.status_code == 200
    assert [item["document_id"] for item in duplicates.json()] == [duplicate.document_id]


def test_crossref_enrichment_updates_metadata(tmp_path: Path) -> None:
    class FakeCrossrefClient:
        def fetch_by_doi(self, doi: str):
            return CrossrefWork(
                title="Official Crossref Title",
                authors="Alice Zhang, Bob Li",
                year=2026,
                venue="Journal of Reliable Metadata",
                doi=doi.upper(),
                abstract="Official abstract from Crossref.",
                publisher="Test Publisher",
                external_url="https://doi.org/10.1234/example.2026.001",
                reference_count=42,
                keywords=["metadata", "crossref"],
            )

    store = RagStore(upload_dir=tmp_path / "uploads", index_dir=tmp_path / "index", retrieval_mode="tfidf")
    document = store.ingest_pdf(
        "crossref.pdf",
        make_pdf_with_text("Local Title Abstract doi 10.1234/example.2026.001 Keywords local Introduction text."),
    )

    documents = store.enrich_metadata(FakeCrossrefClient())
    metadata = documents[0].metadata

    assert documents[0].document_id == document.document_id
    assert metadata.title == "Official Crossref Title"
    assert metadata.authors == "Alice Zhang, Bob Li"
    assert metadata.year == 2026
    assert metadata.venue == "Journal of Reliable Metadata"
    assert metadata.publisher == "Test Publisher"
    assert metadata.external_url == "https://doi.org/10.1234/example.2026.001"
    assert metadata.reference_count == 42
    assert metadata.metadata_source == "crossref"
    assert metadata.is_enriched is True
    assert metadata.keywords == ["metadata", "crossref"]


def test_semantic_scholar_fallback_updates_metadata(tmp_path: Path) -> None:
    class EmptyCrossrefClient:
        def fetch_by_doi(self, doi: str):
            return None

    class FakeSemanticScholarClient:
        def search_by_title(self, title: str):
            return SemanticScholarWork(
                title="Semantic Scholar Title",
                authors="Semantic Author",
                year=2024,
                venue="Semantic Venue",
                doi="10.9999/semantic",
                abstract="Semantic abstract.",
                external_url="https://www.semanticscholar.org/paper/test",
                reference_count=12,
                citation_count=34,
                fields_of_study=["Computer Science", "Medicine"],
                keywords=["Computer Science", "Medicine"],
                confidence="high",
                match_score=0.98,
            )

    store = RagStore(upload_dir=tmp_path / "uploads", index_dir=tmp_path / "index", retrieval_mode="tfidf")
    store.ingest_pdf(
        "semantic.pdf",
        make_pdf_with_text("Semantic Local Title Abstract no doi Keywords local Introduction text."),
    )

    documents = store.enrich_metadata(EmptyCrossrefClient(), FakeSemanticScholarClient())
    metadata = documents[0].metadata

    assert metadata.title == "Semantic Scholar Title"
    assert metadata.metadata_source == "semantic_scholar"
    assert metadata.metadata_confidence == "high"
    assert metadata.metadata_match_score == 0.98
    assert metadata.citation_count == 34
    assert metadata.fields_of_study == ["Computer Science", "Medicine"]


def test_semantic_scholar_client_rejects_low_similarity() -> None:
    client = SemanticScholarClient()
    score = client._title_similarity("Graph retrieval augmented generation", "Unrelated medical imaging paper")

    assert score < 0.72
    assert client._confidence(0.96) == "high"
    assert client._confidence(0.88) == "medium"
    assert client._confidence(0.75) == "low"


def test_enrich_metadata_endpoint_uses_crossref_client(monkeypatch, tmp_path: Path) -> None:
    class FakeCrossrefClient:
        def fetch_by_doi(self, doi: str):
            return CrossrefWork(
                title="Endpoint Enriched Title",
                authors="Endpoint Author",
                year=2025,
                venue="Endpoint Venue",
                doi=doi,
                publisher="Endpoint Publisher",
                external_url="https://doi.org/10.5555/endpoint",
                reference_count=7,
            )

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(main, "CrossrefClient", lambda: FakeCrossrefClient())
    main.store = RagStore(upload_dir=tmp_path / "uploads", index_dir=tmp_path / "index", retrieval_mode="tfidf")
    main.answerer = Answerer()
    client = TestClient(main.app)
    client.post(
        "/documents/upload",
        files={
            "file": (
                "endpoint.pdf",
                make_pdf_with_text("Endpoint Paper Abstract doi 10.5555/endpoint Keywords test Introduction text."),
                "application/pdf",
            )
        },
    )

    response = client.post("/documents/enrich-metadata")

    assert response.status_code == 200
    metadata = response.json()[0]["metadata"]
    assert metadata["title"] == "Endpoint Enriched Title"
    assert metadata["metadata_source"] == "crossref"
    assert metadata["is_enriched"] is True
    assert metadata["reference_count"] == 7


def test_study_summary_endpoint(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    main.store = RagStore(upload_dir=tmp_path / "uploads", index_dir=tmp_path / "index", retrieval_mode="tfidf")
    main.answerer = Answerer()
    main.study_service = StudyService(store=main.store, answerer=main.answerer)
    client = TestClient(main.app)
    pdf_bytes = make_pdf_with_text("A study assistant summarizes methods, experiments, and conclusions.")
    client.post(
        "/documents/upload",
        files={"file": ("study.pdf", pdf_bytes, "application/pdf")},
    )

    response = client.post(
        "/study/summary",
        json={"topic": "study assistant", "focus": "methods", "top_k": 2},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["task"] == "summary"
    assert data["topic"] == "study assistant"
    assert data["answer_mode"] == "retrieval_only"
    assert data["sources"]


def test_study_service_task_types(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    store = RagStore(upload_dir=tmp_path / "uploads", index_dir=tmp_path / "index", retrieval_mode="tfidf")
    answerer = Answerer()
    service = StudyService(store=store, answerer=answerer)
    pdf_bytes = make_pdf_with_text("Reading plans and key points help students learn papers.")
    store.ingest_pdf("tasks.pdf", pdf_bytes)

    summary = service.summary(request=StudyRequest(topic="papers"))
    key_points = service.key_points(request=StudyRequest(topic="papers"))
    reading_plan = service.reading_plan(request=StudyRequest(topic="papers"))

    assert summary.task == "summary"
    assert key_points.task == "key_points"
    assert reading_plan.task == "reading_plan"


def test_answerer_reads_custom_base_url_and_model(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("RLA_OPENAI_BASE_URL", "https://example.test/v1/")
    monkeypatch.setenv("RLA_LLM_MODEL", "test-model")
    monkeypatch.setenv("RLA_OPENAI_WIRE_API", "responses")

    answerer = Answerer()

    assert answerer.base_url == "https://example.test/v1"
    assert answerer.model == "test-model"
    assert answerer.wire_api == "responses"
    assert answerer.client is not None


def test_literature_search_ranks_papers(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    store = RagStore(upload_dir=tmp_path / "uploads", index_dir=tmp_path / "index", retrieval_mode="tfidf")
    answerer = Answerer()
    service = LiteratureService(store=store, answerer=answerer)
    store.ingest_pdf(
        "graph-rag.pdf",
        make_pdf_with_text("Graph retrieval augmented generation uses graph search for literature review."),
    )
    store.ingest_pdf(
        "vision.pdf",
        make_pdf_with_text("Computer vision segmentation uses image masks and convolutional networks."),
    )

    response = service.search(
        LiteratureRequest(query="graph retrieval augmented generation", top_k_documents=2, evidence_k=4)
    )

    assert response.retrieval_mode == "tfidf"
    assert response.papers
    assert response.papers[0].filename == "graph-rag.pdf"
    assert response.papers[0].evidence_count >= 1
    assert response.sources


def test_literature_methods_endpoint(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    main.store = RagStore(upload_dir=tmp_path / "uploads", index_dir=tmp_path / "index", retrieval_mode="tfidf")
    main.answerer = Answerer()
    main.study_service = StudyService(store=main.store, answerer=main.answerer)
    main.literature_service = LiteratureService(store=main.store, answerer=main.answerer)
    client = TestClient(main.app)
    pdf_bytes = make_pdf_with_text(
        "Literature review methods compare retrieval, reranking, topic clustering, and evidence synthesis."
    )
    client.post(
        "/documents/upload",
        files={"file": ("methods.pdf", pdf_bytes, "application/pdf")},
    )

    response = client.post(
        "/literature/methods",
        json={
            "query": "literature review methods",
            "focus": "retrieval and reranking",
            "top_k_documents": 3,
            "evidence_k": 5,
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["task"] == "method_map"
    assert data["query"] == "literature review methods"
    assert data["answer_mode"] == "retrieval_only"
    assert data["papers"]
    assert data["sources"]


def test_literature_compare_endpoint(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    main.store = RagStore(upload_dir=tmp_path / "uploads", index_dir=tmp_path / "index", retrieval_mode="tfidf")
    main.answerer = Answerer()
    main.study_service = StudyService(store=main.store, answerer=main.answerer)
    main.literature_service = LiteratureService(store=main.store, answerer=main.answerer)
    client = TestClient(main.app)
    main.store.ingest_pdf(
        "graph-rag-a.pdf",
        make_pdf_with_text(
            "Graph RAG paper studies retrieval augmented generation with graph search and neighborhood evidence."
        ),
    )
    main.store.ingest_pdf(
        "graph-rag-b.pdf",
        make_pdf_with_text(
            "Another Graph RAG paper compares graph retrieval, reranking, experiments, and limitations."
        ),
    )

    response = client.post(
        "/literature/compare",
        json={
            "query": "graph retrieval augmented generation",
            "focus": "method differences and limitations",
            "top_k_documents": 2,
            "evidence_k": 6,
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["task"] == "paper_compare"
    assert data["answer_mode"] == "retrieval_only"
    assert len(data["papers"]) >= 1
    assert data["sources"]


def test_literature_evaluation_endpoint(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    main.store = RagStore(upload_dir=tmp_path / "uploads", index_dir=tmp_path / "index", retrieval_mode="tfidf")
    main.answerer = Answerer()
    main.literature_service = LiteratureService(store=main.store, answerer=main.answerer)
    main.evaluation_service = EvaluationService(literature_service=main.literature_service)
    client = TestClient(main.app)
    main.store.ingest_pdf(
        "water-methods.pdf",
        make_pdf_with_text(
            "Abstract water quality prediction. Methods model training for water quality remote sensing experiments."
        ),
    )
    main.store.ingest_pdf(
        "biometric-security.pdf",
        make_pdf_with_text(
            "Abstract biometric security. Template protection methods discuss biometric template security limitations."
        ),
    )

    response = client.post(
        "/evaluation/literature",
        json={"top_k_documents": 3, "evidence_k": 8},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["total_cases"] == 3
    assert "average_score" in data
    assert data["cases"]
    assert {"name", "matched_terms", "missing_terms", "score", "passed"} <= set(data["cases"][0])


def test_discovery_service_dedupes_and_marks_imported(tmp_path: Path) -> None:
    class FakeProvider:
        def search(self, query: str, limit: int):
            return [
                ProviderResult(
                    source="semantic_scholar",
                    source_id="s2-1",
                    title="Graph Retrieval Augmented Generation",
                    authors="Alice Zhang",
                    year=2025,
                    doi="10.1234/graph-rag",
                    abstract="Graph retrieval for grounded generation.",
                    citation_count=12,
                    relevance_score=0.91,
                ),
                ProviderResult(
                    source="crossref",
                    source_id="10.1234/graph-rag",
                    title="Graph Retrieval Augmented Generation",
                    authors="Alice Zhang",
                    year=2025,
                    doi="10.1234/graph-rag",
                    abstract="Duplicate DOI record.",
                    citation_count=10,
                    relevance_score=0.75,
                ),
            ]

    store = RagStore(upload_dir=tmp_path / "uploads", index_dir=tmp_path / "index", retrieval_mode="tfidf")
    service = DiscoveryService(store=store, providers={"semantic_scholar": FakeProvider()})

    response = service.search(
        main.DiscoveryRequest(query="graph retrieval augmented generation", sources=["semantic_scholar"])
    )
    document = service.import_metadata(response.papers[0])
    response_after_import = service.search(
        main.DiscoveryRequest(query="graph retrieval augmented generation", sources=["semantic_scholar"])
    )

    assert len(response.papers) == 1
    assert response.papers[0].doi == "10.1234/graph-rag"
    assert document.pages == 0
    assert document.chunks == 0
    assert document.metadata.metadata_source == "semantic_scholar"
    assert response_after_import.papers[0].imported_document_id == document.document_id


def test_discovery_endpoint_search_and_import(monkeypatch, tmp_path: Path) -> None:
    class FakeProvider:
        def search(self, query: str, limit: int):
            return [
                ProviderResult(
                    source="openalex",
                    source_id="https://openalex.org/W1",
                    title="Water Quality Prediction With Neural Networks",
                    authors="Bob Li",
                    year=2024,
                    venue="Journal of Water AI",
                    abstract="Neural models for water quality prediction.",
                    external_url="https://openalex.org/W1",
                    fields_of_study=["Environmental Science"],
                    keywords=["water quality", "neural networks"],
                    relevance_score=0.88,
                )
            ]

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    main.store = RagStore(upload_dir=tmp_path / "uploads", index_dir=tmp_path / "index", retrieval_mode="tfidf")
    main.answerer = Answerer()
    main.discovery_service = None
    monkeypatch.setattr(
        main,
        "DiscoveryService",
        lambda store: DiscoveryService(store=store, providers={"openalex": FakeProvider()}),
    )
    client = TestClient(main.app)

    search_response = client.post(
        "/discovery/search",
        json={"query": "water quality prediction", "sources": ["openalex"], "limit_per_source": 3},
    )
    paper = search_response.json()["papers"][0]
    import_response = client.post("/discovery/import-metadata", json={"paper": paper})
    documents_response = client.get("/documents", params={"source": "openalex"})

    assert search_response.status_code == 200
    assert search_response.json()["papers"][0]["source"] == "openalex"
    assert import_response.status_code == 200
    assert import_response.json()["document"]["metadata"]["title"] == "Water Quality Prediction With Neural Networks"
    assert import_response.json()["document"]["chunks"] == 0
    assert documents_response.status_code == 200
    assert documents_response.json()[0]["metadata"]["metadata_source"] == "openalex"
