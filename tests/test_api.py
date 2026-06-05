from pathlib import Path

from fastapi.testclient import TestClient

from backend.app.answerer import Answerer
from backend.app import main
from backend.app.literature import LiteratureService
from backend.app.rag import RagStore
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


def test_semantic_store_can_build_index(tmp_path: Path) -> None:
    store = RagStore(upload_dir=tmp_path / "uploads", index_dir=tmp_path / "index", retrieval_mode="semantic")
    pdf_bytes = make_pdf_with_text(
        "Retrieval augmented generation grounds model answers with search results."
    )

    store.ingest_pdf("semantic.pdf", pdf_bytes)
    results = store.search("How does RAG use search?", top_k=1)

    assert store.active_retrieval_mode == "semantic"
    assert results


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
