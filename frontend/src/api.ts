export type PaperMetadata = {
  title: string | null;
  authors: string | null;
  year: number | null;
  venue: string | null;
  doi: string | null;
  abstract: string | null;
  keywords: string[];
  duplicate_of: string | null;
  duplicate_reason: string | null;
};

export type DocumentSummary = {
  document_id: string;
  filename: string;
  pages: number;
  chunks: number;
  metadata: PaperMetadata;
};

export type SourceChunk = {
  document_id: string;
  filename: string;
  page: number;
  chunk_id: string;
  score: number;
  text: string;
};

export type QueryResponse = {
  question: string;
  retrieval_mode: string;
  answer_mode: string;
  model: string | null;
  answer: string;
  sources: SourceChunk[];
};

export type StudyResponse = QueryResponse & {
  task: string;
  topic: string;
};

export type PaperCandidate = {
  document_id: string;
  filename: string;
  pages: number;
  chunks: number;
  metadata: PaperMetadata;
  score: number;
  evidence_count: number;
  evidence_pages: number[];
  preview: string;
};

export type LiteratureSearchResponse = {
  query: string;
  retrieval_mode: string;
  papers: PaperCandidate[];
  sources: SourceChunk[];
};

export type LiteratureReviewResponse = {
  task: string;
  query: string;
  retrieval_mode: string;
  answer_mode: string;
  model: string | null;
  answer: string;
  papers: PaperCandidate[];
  sources: SourceChunk[];
};

const API_BASE = "/api";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, init);
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `Request failed with ${response.status}`);
  }
  return response.json() as Promise<T>;
}

export function getDocuments() {
  return request<DocumentSummary[]>("/documents");
}

export function reindexDocuments() {
  return request<DocumentSummary[]>("/documents/reindex", { method: "POST" });
}

export function uploadDocument(file: File) {
  const body = new FormData();
  body.append("file", file);
  return request<DocumentSummary>("/documents/upload", {
    method: "POST",
    body
  });
}

export function askQuestion(question: string, topK: number) {
  return request<QueryResponse>("/query", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, top_k: topK })
  });
}

export function runStudyTask(
  task: "summary" | "key-points" | "reading-plan",
  topic: string,
  focus: string,
  topK: number
) {
  return request<StudyResponse>(`/study/${task}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      topic,
      focus: focus || null,
      top_k: topK
    })
  });
}

export function searchLiterature(
  query: string,
  focus: string,
  topKDocuments: number,
  evidenceK: number
) {
  return request<LiteratureSearchResponse>("/literature/search", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      query,
      focus: focus || null,
      top_k_documents: topKDocuments,
      evidence_k: evidenceK
    })
  });
}

export function runLiteratureTask(
  task: "review" | "methods" | "details",
  query: string,
  focus: string,
  topKDocuments: number,
  evidenceK: number
) {
  return request<LiteratureReviewResponse>(`/literature/${task}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      query,
      focus: focus || null,
      top_k_documents: topKDocuments,
      evidence_k: evidenceK
    })
  });
}
