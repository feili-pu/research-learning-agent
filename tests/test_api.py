from pathlib import Path

from fastapi.testclient import TestClient

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


def test_upload_and_query_pdf(tmp_path: Path) -> None:
    main.store = RagStore(upload_dir=tmp_path)
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
    assert query_response.json()["sources"]
