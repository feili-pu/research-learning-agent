from pathlib import Path

from fastapi.testclient import TestClient

from backend.app.answerer import Answerer
from backend.app import main
from backend.app.rag import RagStore


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
    main.store = RagStore(upload_dir=tmp_path, retrieval_mode="tfidf")
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
    store = RagStore(upload_dir=tmp_path, retrieval_mode="semantic")
    pdf_bytes = make_pdf_with_text(
        "Retrieval augmented generation grounds model answers with search results."
    )

    store.ingest_pdf("semantic.pdf", pdf_bytes)
    results = store.search("How does RAG use search?", top_k=1)

    assert store.active_retrieval_mode == "semantic"
    assert results


def test_answerer_returns_retrieval_only_without_api_key(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    store = RagStore(upload_dir=tmp_path, retrieval_mode="tfidf")
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
    store = RagStore(upload_dir=tmp_path, retrieval_mode="tfidf")
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
    store = RagStore(upload_dir=tmp_path, retrieval_mode="tfidf")
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
    store = RagStore(upload_dir=tmp_path, retrieval_mode="tfidf")
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
