export type PaperMetadata = {
  title: string | null;
  authors: string | null;
  year: number | null;
  venue: string | null;
  doi: string | null;
  abstract: string | null;
  publisher: string | null;
  external_url: string | null;
  reference_count: number | null;
  citation_count: number | null;
  fields_of_study: string[];
  metadata_confidence: string;
  metadata_match_score: number | null;
  metadata_source: string;
  is_enriched: boolean;
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

export type DocumentFilters = {
  query?: string;
  keyword?: string;
  year_from?: string;
  year_to?: string;
  source?: string;
  has_doi?: string;
  duplicate?: string;
  sort_by?: string;
};

export type SourceChunk = {
  document_id: string;
  filename: string;
  page: number;
  chunk_id: string;
  score: number;
  text: string;
  section: string;
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
  evidence_sections: string[];
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

export type EvaluationCaseResult = {
  name: string;
  query: string;
  focus: string | null;
  section_filter: string | null;
  expected_terms: string[];
  matched_terms: string[];
  missing_terms: string[];
  score: number;
  passed: boolean;
  papers: PaperCandidate[];
  sources: SourceChunk[];
};

export type LiteratureEvaluationResponse = {
  retrieval_mode: string;
  total_cases: number;
  passed_cases: number;
  average_score: number;
  cases: EvaluationCaseResult[];
};

export type DiscoveryPaper = {
  source: string;
  source_id: string | null;
  title: string;
  authors: string | null;
  year: number | null;
  venue: string | null;
  doi: string | null;
  abstract: string | null;
  external_url: string | null;
  pdf_url: string | null;
  reference_count: number | null;
  citation_count: number | null;
  fields_of_study: string[];
  keywords: string[];
  is_open_access: boolean;
  relevance_score: number;
  imported_document_id: string | null;
};

export type DiscoveryResponse = {
  query: string;
  focus: string | null;
  sources: string[];
  queries_used: string[];
  papers: DiscoveryPaper[];
  errors: string[];
};

export type DiscoveryImportResponse = {
  document: DocumentSummary;
  duplicate: boolean;
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

export function getDocuments(filters: DocumentFilters = {}) {
  const params = new URLSearchParams();
  Object.entries(filters).forEach(([key, value]) => {
    if (value !== undefined && value !== "") {
      params.set(key, value);
    }
  });
  const query = params.toString();
  return request<DocumentSummary[]>(query ? `/documents?${query}` : "/documents");
}

export function reindexDocuments() {
  return request<DocumentSummary[]>("/documents/reindex", { method: "POST" });
}

export function enrichMetadata() {
  return request<DocumentSummary[]>("/documents/enrich-metadata", { method: "POST" });
}

export function uploadDocument(file: File) {
  const body = new FormData();
  body.append("file", file);
  return request<DocumentSummary>("/documents/upload", {
    method: "POST",
    body
  });
}

export function askQuestion(question: string, topK: number, sectionFilter: string) {
  return request<QueryResponse>("/query", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, top_k: topK, section_filter: sectionFilter || null })
  });
}

export function runStudyTask(
  task: "summary" | "key-points" | "reading-plan",
  topic: string,
  focus: string,
  topK: number,
  sectionFilter: string
) {
  return request<StudyResponse>(`/study/${task}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      topic,
      focus: focus || null,
      top_k: topK,
      section_filter: sectionFilter || null
    })
  });
}

export function searchLiterature(
  query: string,
  focus: string,
  topKDocuments: number,
  evidenceK: number,
  sectionFilter: string
) {
  return request<LiteratureSearchResponse>("/literature/search", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      query,
      focus: focus || null,
      top_k_documents: topKDocuments,
      evidence_k: evidenceK,
      section_filter: sectionFilter || null
    })
  });
}

export function runLiteratureTask(
  task: "review" | "methods" | "details" | "compare",
  query: string,
  focus: string,
  topKDocuments: number,
  evidenceK: number,
  sectionFilter: string
) {
  return request<LiteratureReviewResponse>(`/literature/${task}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      query,
      focus: focus || null,
      top_k_documents: topKDocuments,
      evidence_k: evidenceK,
      section_filter: sectionFilter || null
    })
  });
}

export function runLiteratureEvaluation(
  topKDocuments: number,
  evidenceK: number,
  sectionFilter: string
) {
  return request<LiteratureEvaluationResponse>("/evaluation/literature", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      top_k_documents: topKDocuments,
      evidence_k: evidenceK,
      section_filter: sectionFilter || null
    })
  });
}

export function searchDiscovery(
  query: string,
  focus: string,
  sources: string[],
  limitPerSource: number
) {
  return request<DiscoveryResponse>("/discovery/search", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      query,
      focus: focus || null,
      sources,
      limit_per_source: limitPerSource
    })
  });
}

export function importDiscoveredPaper(paper: DiscoveryPaper) {
  return request<DiscoveryImportResponse>("/discovery/import-metadata", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ paper })
  });
}
